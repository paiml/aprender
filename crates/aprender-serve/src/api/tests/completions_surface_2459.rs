//! Falsifiers for the `/v1/completions` and batch surface, found by auditing the
//! serve crate after the #2375/#2376 dogfood batches.
//!
//! Every assertion here is something a client observes over HTTP, or the exact
//! value a request field is turned into before it reaches the engine. All of them
//! were RED against `main` @ 9b19970db; the verbatim red output is in the commit
//! message.
//!
//! The four defects:
//!
//! 1. `POST /v1/completions` passed an EMPTY stop set to the decode loop, so the
//!    model's EOS token could not end a completion. Measured against a live
//!    `apr serve run qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, same server, same
//!    prompt, `max_tokens: 200`: `/generate` returned 5 tokens (`"2+2=4"`),
//!    `/v1/completions` returned 200 tokens, `finish_reason: "length"`, and the
//!    answer buried under `"I'm sorry, but I can't assist with that."` repeated
//!    to the budget.
//! 2. `top_p` is a documented field of `CompletionRequest` and was dropped on the
//!    quantized path: `{"temperature":2.0,"top_p":0.000001}` returned output
//!    byte-identical to `{"temperature":2.0}`.
//! 3. `POST /v1/chat/completions` answered 500 for a context-window overflow while
//!    `/generate`, `/stream/generate`, `/v1/completions`, `/realize/embed` and
//!    `/v1/embeddings` all answered 400 for the same condition on the same server.
//! 4. `POST /v1/batch/completions` "tokenized" prompts as raw BYTES and "decoded"
//!    results by truncating each token id to a byte — then returned HTTP 200 with
//!    throughput statistics.

use axum::http::StatusCode;

use super::native_routes_2376::post;
use crate::api::AppState;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Context window of the fixture model. Small so an over-long prompt is cheap.
#[cfg(feature = "gpu")]
const FIXTURE_CONTEXT: usize = 128;

/// A quantized-only server (`apr serve run model.gguf`) whose model declares
/// `eos_token_id`. The EOS is a parameter because these falsifiers turn on
/// whether that specific token ends a completion.
#[cfg(feature = "gpu")]
fn quantized_state_with_eos(eos: Option<u32>) -> AppState {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: FIXTURE_CONTEXT,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: eos,
    };
    AppState::with_quantized_model(create_test_quantized_model(&config))
        .expect("build quantized AppState")
}

/// The greedy continuation this fixture model actually produces for `prompt`.
///
/// MEASURED, never assumed: the falsifiers below need a token the model really
/// emits (to use as EOS) and one it really does not (as the control), and a
/// guessed pair would make either direction pass for the wrong reason.
#[cfg(feature = "gpu")]
fn greedy_continuation(prompt_text: &str, budget: usize) -> (Vec<u32>, usize) {
    use crate::gguf::QuantizedGenerateConfig;

    let state = quantized_state_with_eos(None);
    let tokenizer = state.tokenizer.clone().expect("fixture tokenizer");
    let prompt_ids = tokenizer.encode(prompt_text);
    assert!(
        !prompt_ids.is_empty(),
        "the fixture prompt must tokenize to something, or the handler rejects it \
         before generating and these falsifiers measure nothing"
    );
    let model = state.quantized_model().expect("fixture quantized model");
    // `deterministic` == temperature 0 / top_k 1 == the argmax path the handler
    // takes for a request with `"temperature": 0.0`, so the token ids observed
    // here are the token ids the route will produce.
    let generated = model
        .generate_with_cache(&prompt_ids, &QuantizedGenerateConfig::deterministic(budget))
        .expect("fixture generation");
    let prompt_len = prompt_ids.len();
    (generated[prompt_len..].to_vec(), prompt_len)
}

/// A token id in vocabulary range that the fixture model does NOT emit within
/// `emitted`. Used as the control EOS: naming one the model never samples proves
/// the "stops early" case is caused by the EOS and not by generation being broken.
#[cfg(feature = "gpu")]
fn token_never_emitted(emitted: &[u32]) -> u32 {
    (0..256_u32)
        .find(|t| !emitted.contains(t))
        .expect("the fixture vocabulary is larger than the generated budget")
}

#[cfg(feature = "gpu")]
fn json_field(body: &str, path: &[&str]) -> serde_json::Value {
    let mut value: serde_json::Value = serde_json::from_str(body).expect("JSON body");
    for key in path {
        value = value
            .get(*key)
            .unwrap_or_else(|| panic!("missing `{key}` in body: {body}"))
            .clone();
    }
    value
}

// ---------------------------------------------------------------------------
// 1. /v1/completions must end a completion at the model's EOS
// ---------------------------------------------------------------------------

/// The falsifier. With the model's EOS as the first token it samples, the route
/// must return ZERO completion tokens; with an EOS it never samples, the same
/// route on the same fixture must run the whole budget.
///
/// Pre-fix (`stop_tokens: Vec::new()`) BOTH cases return the full budget, so the
/// first assertion fails: 8 completion tokens where 0 are required.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn completions_end_at_the_models_eos_token() {
    const PROMPT: &str = "token7 token9";
    const BUDGET: usize = 8;

    let (continuation, _prompt_len) = greedy_continuation(PROMPT, BUDGET);
    let eos = *continuation
        .first()
        .expect("the fixture model must generate at least one token");
    let control_eos = token_never_emitted(&continuation);

    let body =
        format!(r#"{{"model":"m","prompt":"{PROMPT}","max_tokens":{BUDGET},"temperature":0.0}}"#);

    // Control FIRST: an EOS the model never emits must NOT shorten anything.
    // Without this, "0 tokens" below could just mean generation is broken.
    let (status, control_body) =
        post(quantized_state_with_eos(Some(control_eos)), "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::OK, "control request: {control_body}");
    assert_eq!(
        json_field(&control_body, &["usage", "completion_tokens"]),
        serde_json::json!(BUDGET),
        "control: with an EOS ({control_eos}) this model never samples, the completion \
         must run the full {BUDGET}-token budget: {control_body}"
    );

    // The defect: the token the model DOES sample first is the model's EOS, so the
    // completion must end immediately.
    let (status, stopped_body) =
        post(quantized_state_with_eos(Some(eos)), "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::OK, "eos request: {stopped_body}");
    assert_eq!(
        json_field(&stopped_body, &["usage", "completion_tokens"]),
        serde_json::json!(0),
        "the model's EOS ({eos}) is the first token it samples here, so /v1/completions \
         must stop at it and return 0 completion tokens; it ran to max_tokens instead, \
         which is what made every real completion run past end-of-text: {stopped_body}"
    );
    assert_eq!(
        json_field(&stopped_body, &["choices"])[0]["finish_reason"],
        serde_json::json!("stop"),
        "a completion ended by EOS reports finish_reason \"stop\", not \"length\": \
         {stopped_body}"
    );
}

/// `/v1/completions` and `/generate` must agree about where this model ends a
/// sequence. They are the same engine on the same server; only the config builder
/// differed, and that is exactly how the two routes came to disagree.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn completions_and_generate_agree_about_the_eos_token() {
    const PROMPT: &str = "token7 token9";
    const BUDGET: usize = 8;

    let (continuation, _) = greedy_continuation(PROMPT, BUDGET);
    let eos = *continuation.first().expect("at least one generated token");

    let (_, generate_body) = post(
        quantized_state_with_eos(Some(eos)),
        "/generate",
        &format!(r#"{{"prompt":"{PROMPT}","max_tokens":{BUDGET},"temperature":0.0,"strategy":"greedy"}}"#),
    )
    .await;
    let (_, completions_body) = post(
        quantized_state_with_eos(Some(eos)),
        "/v1/completions",
        &format!(r#"{{"model":"m","prompt":"{PROMPT}","max_tokens":{BUDGET},"temperature":0.0}}"#),
    )
    .await;

    assert_eq!(
        json_field(&completions_body, &["usage", "completion_tokens"]),
        json_field(&generate_body, &["num_generated"]),
        "/v1/completions and /generate must produce the same number of tokens for the \
         same prompt on the same model; /generate honoured the EOS and /v1/completions \
         did not.\n  /generate: {generate_body}\n  /v1/completions: {completions_body}"
    );
}

// ---------------------------------------------------------------------------
// 2. top_p is threaded, and an unsatisfiable top_p is refused
// ---------------------------------------------------------------------------

/// `top_p` must reach the engine config. Pre-fix the quantized builder took
/// `..Default::default()`, so the config carried `top_p = 1.0` (no nucleus)
/// whatever the request said.
#[test]
fn completion_config_threads_top_p_and_the_models_eos() {
    use crate::api::realize_handlers::completion_quantized_config;
    use crate::api::CompletionRequest;
    use crate::tokenizer::BPETokenizer;

    let tokenizer = BPETokenizer::new(
        vec!["<unk>".to_string(), "hi".to_string()],
        vec![],
        "<unk>",
    )
    .expect("test tokenizer");
    let request = CompletionRequest {
        model: "m".to_string(),
        prompt: "hi".to_string(),
        max_tokens: Some(16),
        temperature: Some(0.7),
        top_p: Some(0.25),
        stop: None,
        stream: false,
        n: crate::api::ChoiceCount::ONE,
    };

    let config = completion_quantized_config(
        &request,
        &tokenizer,
        Some(7),
        16,
        0.7,
        false,
        &crate::generate::CancelToken::never(),
    );

    assert!(
        (config.top_p - 0.25).abs() < f32::EPSILON,
        "the handler dropped top_p: expected 0.25, got {}",
        config.top_p
    );
    assert_eq!(
        config.stop_tokens,
        vec![7],
        "the model's EOS must be in the stop set, or the decode loop cannot end a \
         completion; got {:?}",
        config.stop_tokens
    );
}

/// A request that omits `top_p` must be unchanged: the fix threads the field, it
/// does not invent a nucleus for requests that never asked for one.
#[test]
fn completion_config_without_top_p_keeps_the_engine_default() {
    use crate::api::realize_handlers::completion_quantized_config;
    use crate::api::CompletionRequest;
    use crate::gguf::QuantizedGenerateConfig;
    use crate::tokenizer::BPETokenizer;

    let tokenizer = BPETokenizer::new(
        vec!["<unk>".to_string(), "hi".to_string()],
        vec![],
        "<unk>",
    )
    .expect("test tokenizer");
    let request = CompletionRequest {
        model: "m".to_string(),
        prompt: "hi".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
        stream: false,
        n: crate::api::ChoiceCount::ONE,
    };

    let config = completion_quantized_config(
        &request,
        &tokenizer,
        None,
        256,
        0.7,
        false,
        &crate::generate::CancelToken::never(),
    );

    let defaults = QuantizedGenerateConfig::default();
    assert!(
        (config.top_p - defaults.top_p).abs() < f32::EPSILON,
        "a request with no top_p must keep the engine default {}, got {}",
        defaults.top_p,
        config.top_p
    );
}

/// `top_p: 5.0` is not satisfiable. `/generate` already refuses it with 400; this
/// route returned 200 — harmless only while the value was being discarded.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn out_of_range_top_p_is_refused_not_ignored() {
    let (status, body) = post(
        quantized_state_with_eos(None),
        "/v1/completions",
        r#"{"model":"m","prompt":"token7","max_tokens":2,"top_p":5.0}"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a top_p outside (0, 1] must be refused, not accepted and ignored: {body}"
    );
    assert!(
        body.contains("top_p"),
        "the refusal must name the field the client got wrong: {body}"
    );

    // And the converse: a top_p this server CAN honour is accepted, so the
    // assertion above cannot be satisfied by refusing every request.
    let (status, body) = post(
        quantized_state_with_eos(None),
        "/v1/completions",
        r#"{"model":"m","prompt":"token7","max_tokens":2,"top_p":0.9}"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a top_p inside (0, 1] must be served: {body}"
    );
}

// ---------------------------------------------------------------------------
// 3. A context overflow is a client error on EVERY generating route
// ---------------------------------------------------------------------------

/// The same over-long prompt must get the same class of answer from the chat route
/// as from its neighbours. Pre-fix `/v1/chat/completions` answered 500 — which
/// tells an OpenAI SDK to retry a request that can never succeed.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn context_overflow_is_a_client_error_on_the_chat_route_too() {
    // Longer than FIXTURE_CONTEXT once tokenized; asserted below rather than assumed.
    let long_prompt = "token5 ".repeat(FIXTURE_CONTEXT * 4);
    let state = quantized_state_with_eos(None);
    let tokenizer = state.tokenizer.clone().expect("fixture tokenizer");
    assert!(
        tokenizer.encode(&long_prompt).len() > FIXTURE_CONTEXT,
        "the fixture prompt must actually exceed the {FIXTURE_CONTEXT}-token context \
         window, or this test proves nothing about overflow handling"
    );
    let trimmed = long_prompt.trim_end();

    let (chat_status, chat_body) = post(
        quantized_state_with_eos(None),
        "/v1/chat/completions",
        &format!(r#"{{"model":"m","messages":[{{"role":"user","content":"{trimmed}"}}]}}"#),
    )
    .await;
    let (generate_status, generate_body) = post(
        quantized_state_with_eos(None),
        "/generate",
        &format!(r#"{{"prompt":"{trimmed}","max_tokens":4}}"#),
    )
    .await;

    assert_eq!(
        generate_status,
        StatusCode::BAD_REQUEST,
        "baseline: /generate already classifies a context overflow as a client error: \
         {generate_body}"
    );
    assert_eq!(
        chat_status,
        StatusCode::BAD_REQUEST,
        "a prompt that does not fit the context window is fully determined by the \
         request, so /v1/chat/completions must answer 4xx like every sibling route; \
         a 5xx makes every OpenAI SDK retry it: {chat_body}"
    );
}

// ---------------------------------------------------------------------------
// 4. /v1/batch/completions must use the tokenizer, not a byte cast
// ---------------------------------------------------------------------------

/// Prompts must be tokenized by the server's tokenizer. The shipped code used
/// `p.bytes().map(|b| b as u32)`, so the model was asked a different question
/// than the client sent — and answered with HTTP 200.
#[test]
#[cfg(feature = "gpu")]
fn batch_prompts_are_tokenized_not_byte_cast() {
    use crate::api::gpu_handlers::batch_prompt_tokens;
    use crate::tokenizer::BPETokenizer;

    let tokenizer = BPETokenizer::new(
        vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ],
        vec![],
        "<unk>",
    )
    .expect("test tokenizer");
    let prompts = vec!["hello world".to_string()];

    let tokens = batch_prompt_tokens(&tokenizer, &prompts);

    assert_eq!(
        tokens,
        vec![tokenizer.encode("hello world")],
        "batch prompts must go through the SAME tokenizer as every other route"
    );
    let byte_cast: Vec<u32> = "hello world".bytes().map(u32::from).collect();
    assert_ne!(
        tokens[0], byte_cast,
        "the byte cast is what shipped: it fed the model the ids {byte_cast:?} for a \
         prompt whose tokens are {:?}",
        tokens[0]
    );
}

/// Generated ids must be decoded through the vocabulary. The shipped code did
/// `t as u8 as char`, which silently truncates every id above 255.
#[test]
#[cfg(feature = "gpu")]
fn batch_results_are_decoded_through_the_vocabulary() {
    use crate::api::gpu_handlers::batch_decode;
    use crate::tokenizer::BPETokenizer;

    let mut vocab: Vec<String> = (0..300).map(|i| format!("tok{i}")).collect();
    vocab[0] = "<unk>".to_string();
    vocab[257] = "ANSWER".to_string();
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test tokenizer");

    let decoded = batch_decode(&tokenizer, &[257]);

    assert_eq!(
        decoded,
        tokenizer.decode(&[257]).expect("decode 257"),
        "batch results must be decoded with the tokenizer"
    );
    let byte_cast: String = [257_u32].iter().map(|&t| t as u8 as char).collect();
    assert_ne!(
        decoded, byte_cast,
        "the shipped byte cast rendered token 257 as {byte_cast:?} instead of its \
         vocabulary entry"
    );
}
