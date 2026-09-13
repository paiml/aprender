//! OpenAI-compatible API handlers
//!
//! Extracted from api/mod.rs (PMAT-802) to reduce module size.
//! Contains chat completion, streaming, and model list handlers.
#![allow(unreachable_pub)] // Items re-exported as pub from api/mod.rs

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Extension, Json,
};
use futures::stream::Stream;

use super::{
    build_trace_data, clean_chat_output, format_chat_messages, AppState, ChatChoice,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ErrorResponse,
    FinishReason, OpenAIModel, OpenAIModelsResponse, StreamMode, Usage,
};
use crate::generate::{CancelToken, GenerationConfig, SamplingStrategy};
use crate::tokenizer::BPETokenizer;

// ============================================================================
// Shared helpers — eliminate duplication across backend paths
// ============================================================================

/// Record failure and return an error response.
fn fail_response(state: &AppState, status: StatusCode, msg: impl std::fmt::Display) -> Response {
    state.metrics.record_failure();
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

/// Current Unix timestamp.
fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Get tokenizer from state or return 500.
#[allow(clippy::result_large_err)]
fn require_tokenizer(state: &AppState) -> Result<Arc<BPETokenizer>, Response> {
    state.tokenizer.clone().ok_or_else(|| {
        fail_response(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "No tokenizer available",
        )
    })
}

/// Format chat messages, tokenize, validate non-empty.
#[allow(clippy::result_large_err)]
fn tokenize_chat_prompt(
    tokenizer: &BPETokenizer,
    messages: &[ChatMessage],
    model_hint: Option<&str>,
    state: &AppState,
) -> Result<Vec<u32>, Response> {
    let prompt_text = format_chat_messages(messages, model_hint);
    let ids = tokenizer.encode(&prompt_text);
    if ids.is_empty() {
        return Err(fail_response(
            state,
            StatusCode::BAD_REQUEST,
            "Messages cannot be empty",
        ));
    }
    Ok(ids)
}

/// Extract common generation parameters from the request.
///
/// GH-330: EOS resolution follows Design by Contract priority:
/// 1. Model config (class invariant from GGUF/APR metadata)
/// 2. Tokenizer vocabulary lookup (runtime fallback)
///
/// No hardcoded magic numbers.
fn chat_gen_params(
    request: &ChatCompletionRequest,
    tokenizer: &BPETokenizer,
    model_eos: Option<u32>,
) -> (usize, f32, u32) {
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);
    // GH-330: Model config EOS first, then tokenizer lookup, then 0 (disabled)
    let eos_token_id = model_eos
        .or_else(|| tokenizer.get_token_id("<|im_end|>"))
        .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
        .unwrap_or(0);
    (max_tokens, temperature, eos_token_id)
}

/// EOS stop tokens for a chat request, honouring `ignore_eos` (PERF-039).
///
/// Every decode loop in the tree stops on `config.stop_tokens.contains(&next)`,
/// so an EMPTY `stop_tokens` already means "never stop on a token" — the engine
/// mechanism for ignore-EOS existed before this function did. What was missing
/// was any way for a client to ask for it: `ignore_eos` had no wire
/// representation at all, on either side.
///
/// APR-PERF-GATE-001 v2.2 §4.3.1 pins W1 at `max_tokens = 128` with ignore-EOS
/// precisely so the work per band is fixed. `max_tokens` still bounds the loop,
/// so `ignore_eos` cannot produce an unbounded generation.
fn chat_stop_tokens(request: &ChatCompletionRequest, eos_token_id: u32) -> Vec<u32> {
    if request.ignore_eos.unwrap_or(false) {
        Vec::new()
    } else {
        vec![eos_token_id]
    }
}

/// Refuse `ignore_eos: true` on a backend that cannot honour it (PERF-039).
///
/// `Some(response)` means the caller must return it immediately.
///
/// Three chat backends stop on EOS in a way that no request field reaches:
/// `try_safetensors_cuda_backend` passes a positional `eos_id: u32` into
/// `model.generate`, and `try_apr_transformer_backend` and `registry_fallback`
/// build configs whose stop behaviour is the model's own. Serving those
/// requests anyway would hand the benchmark a token budget it did not get
/// while the receipt recorded `ignore_eos: true` — a measurement of unpinned
/// work labelled as pinned. Fail closed instead: 501 names the backend, and
/// the operator learns the gate cannot be run on this model rather than
/// learning nothing.
fn reject_unsupported_ignore_eos(
    state: &AppState,
    request: &ChatCompletionRequest,
    backend: &str,
) -> Option<Response> {
    if request.ignore_eos.unwrap_or(false) {
        return Some(fail_response(
            state,
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "`ignore_eos` is not supported by the {backend} chat backend; it would be \
                 silently ignored and the response would stop on EOS anyway. Refused rather \
                 than served, so a benchmark cannot record a pinned token budget it did not \
                 receive (APR-PERF-GATE-001 v2.2 §4.3.1)."
            ),
        ));
    }
    None
}

/// Resolve the effective top-k for a chat sampling config (PMAT-760).
///
/// Honors the request's `top_k` (the documented sampling control) when set, else defaults to
/// 40; `temperature == 0.0` forces greedy (top_k = 1), and an explicit `top_k = 1` likewise
/// forces greedy regardless of temperature (qwen3-moe-sampling-v1 V1_001). The chat backends
/// previously hardcoded `if temperature == 0.0 { 1 } else { 40 }`, silently DROPPING
/// request.top_k — drift from batch.rs, which honors it.
fn resolve_chat_top_k(temperature: f32, requested: Option<usize>) -> usize {
    if temperature == 0.0 {
        1
    } else {
        requested.unwrap_or(40)
    }
}

#[cfg(test)]
mod pmat760_top_k_tests {
    use super::resolve_chat_top_k;

    #[test]
    fn honors_requested_top_k() {
        // The request's top_k must be used, not the hardcoded 40 (the pre-PMAT-760 bug).
        assert_eq!(resolve_chat_top_k(0.7, Some(10)), 10);
        assert_eq!(resolve_chat_top_k(1.0, Some(100)), 100);
    }

    #[test]
    fn defaults_to_40_when_unset() {
        assert_eq!(resolve_chat_top_k(0.7, None), 40);
    }

    #[test]
    fn temperature_zero_forces_greedy() {
        // temp==0 => greedy (top_k=1) regardless of the requested top_k.
        assert_eq!(resolve_chat_top_k(0.0, None), 1);
        assert_eq!(resolve_chat_top_k(0.0, Some(50)), 1);
    }

    #[test]
    fn explicit_top_k_one_is_greedy_at_any_temperature() {
        // qwen3-moe-sampling-v1 V1_001: top_k=1 forces greedy regardless of temperature.
        assert_eq!(resolve_chat_top_k(0.9, Some(1)), 1);
    }
}

/// PMAT-821: Build a [`QuantizedGenerateConfig`] for a dense chat backend, threading
/// EVERY request sampling parameter into the config.
///
/// The dense `/v1/chat/completions` backends (`try_cuda_backend`,
/// `try_quantized_backend`) previously built the config reading ONLY
/// `max_tokens`/`temperature`/`top_k`, then `..Default::default()`. That DROPPED
/// `top_p`, `repeat_penalty`, `repeat_last_n`, and `seed` from the HTTP request
/// before generation — so even with the sampler fixed (#2081 top_p, #2099
/// repeat_penalty), the dense chat endpoint silently used the neutral DEFAULTS
/// (`top_p = 1.0`, `repeat_penalty = 1.0`). The MoE path
/// (`try_qwen3_moe_backend`) already threaded all of these; this helper closes
/// the dense-path gap and keeps both backends DRY.
///
/// Defaults fall back to [`QuantizedGenerateConfig::default`] field-by-field, so a
/// request that omits a param produces a byte-identical config to the pre-fix
/// behavior for that field (the no-regression invariant). `max_tokens`,
/// `temperature`, and `top_k` keep their existing chat semantics
/// (`chat_gen_params` / `resolve_chat_top_k`).
///
/// Discharges F-CHAT-HANDLER-THREADS-PARAMS-001 in `contracts/openai-compat-v1.yaml`.
///
/// `cancel` is the request's cancellation signal (aprender#2376(3)). It is a
/// required parameter rather than an `Option` so a new chat backend cannot
/// silently produce a config whose decode loop outlives its client.
fn chat_quantized_config(
    request: &ChatCompletionRequest,
    tokenizer: &BPETokenizer,
    model_eos: Option<u32>,
    trace: bool,
    cancel: &crate::generate::CancelToken,
) -> crate::gguf::QuantizedGenerateConfig {
    let defaults = crate::gguf::QuantizedGenerateConfig::default();
    let (max_tokens, temperature, eos_token_id) = chat_gen_params(request, tokenizer, model_eos);
    crate::gguf::QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: resolve_chat_top_k(temperature, request.top_k),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        repeat_penalty: request.repeat_penalty.unwrap_or(defaults.repeat_penalty),
        repeat_last_n: request.repeat_last_n.unwrap_or(defaults.repeat_last_n),
        seed: request.seed.unwrap_or(defaults.seed),
        stop_tokens: chat_stop_tokens(request, eos_token_id),
        trace,
        cancel: cancel.clone(),
        ..defaults
    }
}

#[cfg(test)]
mod perf039_ignore_eos_tests {
    use super::{chat_quantized_config, chat_stop_tokens, ChatCompletionRequest, ChatMessage};
    use crate::tokenizer::BPETokenizer;

    fn tokenizer() -> BPETokenizer {
        BPETokenizer::new(vec!["<unk>".to_string(), "hi".to_string()], vec![], "<unk>")
            .expect("test tokenizer")
    }

    fn request(ignore_eos: Option<bool>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "default".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            ignore_eos,
            ..Default::default()
        }
    }

    // MUTATION TABLE for `chat_stop_tokens`. The outcome each row EXCLUDES is
    // "the field was accepted on the wire and then dropped", which is the only
    // failure mode that matters here: a dropped `ignore_eos` leaves EOS live
    // while the receipt records a pinned token budget.
    //
    //  ignore_eos   | stop_tokens | meaning
    //  -------------|-------------|----------------------------------------
    //  absent       | [eos]       | unchanged from pre-PERF-039
    //  Some(false)  | [eos]       | explicit opt-out is not an opt-in
    //  Some(true)   | []          | every decode loop then never stops on a token

    #[test]
    fn absent_ignore_eos_keeps_eos_stopping() {
        assert_eq!(chat_stop_tokens(&request(None), 7), vec![7]);
    }

    #[test]
    fn explicit_false_keeps_eos_stopping() {
        assert_eq!(chat_stop_tokens(&request(Some(false)), 7), vec![7]);
    }

    #[test]
    fn ignore_eos_empties_the_stop_set() {
        assert!(
            chat_stop_tokens(&request(Some(true)), 7).is_empty(),
            "an empty stop set is what every decode loop reads as ignore-EOS"
        );
    }

    /// The W1 path: `chat_quantized_config` feeds `try_quantized_backend`
    /// (CPU GGUF) and `try_cuda_backend`, which is the backend
    /// APR-PERF-GATE-001 §4.3.1's Q4_K_M model actually runs on.
    #[test]
    fn quantized_chat_config_honors_ignore_eos() {
        let tok = tokenizer();
        let cancel = crate::generate::CancelToken::never();
        let on = chat_quantized_config(&request(Some(true)), &tok, Some(7), false, &cancel);
        assert!(
            on.stop_tokens.is_empty(),
            "W1's backend must generate exactly max_tokens"
        );
        let off = chat_quantized_config(&request(None), &tok, Some(7), false, &cancel);
        assert_eq!(
            off.stop_tokens,
            vec![7],
            "default behaviour must not change"
        );
    }

    /// `ignore_eos` must survive DESERIALIZATION, not just exist as a field.
    ///
    /// serde ignores unknown fields by default, so before this field existed a
    /// client POSTing `"ignore_eos": true` got a 200 and EOS-terminated output
    /// with nothing anywhere saying the parameter had been discarded. That is
    /// the state this test excludes.
    #[test]
    fn ignore_eos_deserializes_from_the_wire() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"ignore_eos":true,"seed":9}"#,
        )
        .expect("wire request must deserialize");
        assert_eq!(req.ignore_eos, Some(true));
        assert_eq!(req.seed, Some(9), "seed must survive the wire too");
        let bare: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .expect("request without the field must still deserialize");
        assert_eq!(bare.ignore_eos, None);
    }
}

#[cfg(test)]
mod perf039_ignore_eos_fail_closed_tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt;

    /// A backend that cannot honour `ignore_eos` must REFUSE, not serve.
    ///
    /// The test app resolves to a chat backend that stops on EOS in a way no
    /// request field reaches. Serving the request would return 200 and
    /// EOS-terminated text while the client believed the token count was
    /// pinned — the exact "recorded but never compared" shape this epic
    /// exists to remove. 501 is the outcome; the row that matters is that it
    /// is NOT 200.
    #[tokio::test]
    async fn ignore_eos_on_an_unsupporting_backend_is_501_not_200() {
        let app = crate::api::test_helpers::create_test_app_shared();
        let body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "ignore_eos": true
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("request builds"),
            )
            .await
            .expect("handler responds");
        assert_ne!(
            response.status(),
            StatusCode::OK,
            "serving `ignore_eos` on a backend that drops it reports pinned work that never happened"
        );
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }

    /// The same request WITHOUT `ignore_eos` must be unaffected. Without this
    /// row the test above would also pass on a handler that 501s everything.
    #[tokio::test]
    async fn the_same_request_without_ignore_eos_is_not_501() {
        let app = crate::api::test_helpers::create_test_app_shared();
        let body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("request builds"),
            )
            .await
            .expect("handler responds");
        assert_ne!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }
}

#[cfg(test)]
mod pmat821_chat_handler_threading_tests {
    use super::{chat_quantized_config, ChatCompletionRequest, ChatMessage};
    use crate::gguf::QuantizedGenerateConfig;
    use crate::tokenizer::BPETokenizer;

    /// Minimal tokenizer for handler-level config tests (no model weights needed).
    fn test_tokenizer() -> BPETokenizer {
        let vocab: Vec<String> = vec!["<unk>".to_string(), "hi".to_string()];
        BPETokenizer::new(vocab, vec![], "<unk>").expect("test tokenizer")
    }

    /// Minimal request with all sampling params unset (the no-param baseline).
    fn base_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "default".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            ignore_eos: None,
            n: crate::api::ChoiceCount::ONE,
            stream: false,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
            stream_options: None,
        }
    }

    #[test]
    fn handler_threads_top_p_into_config() {
        // RED on the PMAT-821 bug: the dense handler dropped request.top_p, so the
        // config used the neutral default 1.0 instead of the requested 0.5.
        // GREEN after fix: config.top_p == 0.5. This is testable WITHOUT the
        // in-flight sampler fixes (#2081) — it asserts the HANDLER→CONFIG threading.
        let mut request = base_request();
        request.top_p = Some(0.5);
        // temperature must be non-zero or resolve_chat_top_k forces greedy (top_k=1),
        // which is orthogonal to top_p but keeps the request realistic.
        request.temperature = Some(0.7);
        let tokenizer = test_tokenizer();
        let config = chat_quantized_config(
            &request,
            &tokenizer,
            None,
            false,
            &crate::generate::CancelToken::never(),
        );
        assert!(
            (config.top_p - 0.5).abs() < f32::EPSILON,
            "handler dropped top_p: expected 0.5, got {}",
            config.top_p
        );
    }

    #[test]
    fn handler_threads_repeat_penalty_into_config() {
        // RED on bug: request.repeat_penalty dropped → config uses default 1.0.
        let mut request = base_request();
        request.repeat_penalty = Some(1.3);
        request.temperature = Some(0.7);
        let tokenizer = test_tokenizer();
        let config = chat_quantized_config(
            &request,
            &tokenizer,
            None,
            false,
            &crate::generate::CancelToken::never(),
        );
        assert!(
            (config.repeat_penalty - 1.3).abs() < f32::EPSILON,
            "handler dropped repeat_penalty: expected 1.3, got {}",
            config.repeat_penalty
        );
    }

    #[test]
    fn handler_threads_repeat_last_n_and_seed_into_config() {
        let mut request = base_request();
        request.repeat_last_n = Some(128);
        request.seed = Some(7);
        request.temperature = Some(0.7);
        let tokenizer = test_tokenizer();
        let config = chat_quantized_config(
            &request,
            &tokenizer,
            None,
            false,
            &crate::generate::CancelToken::never(),
        );
        assert_eq!(config.repeat_last_n, 128, "handler dropped repeat_last_n");
        assert_eq!(config.seed, 7, "handler dropped seed");
    }

    #[test]
    fn no_param_request_uses_defaults_byte_identical() {
        // No-regression invariant: a request that sets NONE of the sampling params
        // produces a config whose sampling fields equal QuantizedGenerateConfig's
        // defaults — byte-identical to the pre-PMAT-821 behavior for the no-param case.
        let request = base_request();
        let tokenizer = test_tokenizer();
        let defaults = QuantizedGenerateConfig::default();
        let config = chat_quantized_config(
            &request,
            &tokenizer,
            None,
            false,
            &crate::generate::CancelToken::never(),
        );
        assert!(
            (config.top_p - defaults.top_p).abs() < f32::EPSILON,
            "no-param top_p must equal default"
        );
        assert!(
            (config.repeat_penalty - defaults.repeat_penalty).abs() < f32::EPSILON,
            "no-param repeat_penalty must equal default"
        );
        assert_eq!(
            config.repeat_last_n, defaults.repeat_last_n,
            "no-param repeat_last_n must equal default"
        );
        assert_eq!(
            config.seed, defaults.seed,
            "no-param seed must equal default"
        );
        // temperature default for chat is 0.7 (chat_gen_params), which forces top_k=1.
        // That is the existing chat behavior, not a regression introduced here.
    }
}

/// PMAT-756: apply OpenAI stop sequences to a chat completion's text and compute the
/// matching `finish_reason`. The `/v1/chat/completions` path previously ignored
/// `request.stop` entirely — the returned message kept the stop string and ran to
/// `max_tokens`. Reuses the shared earliest-position [`truncate_at_stop`] helper (PMAT-754)
/// so chat behaves identically to the `/v1/completions` backends.
///
/// `finish_reason` is `"stop"` whenever a stop string truncated the text (even if
/// `max_tokens` was also reached — a matched stop sequence takes precedence per OpenAI
/// semantics), `"length"` only when the generation hit `max_tokens` with no stop match,
/// else `"stop"` (the model emitted EOS naturally). Pure + unit-tested.
fn finalize_chat_text(
    text: String,
    stops: Option<&[String]>,
    completion_tokens: usize,
    max_tokens: usize,
) -> (String, String) {
    // #2465(2): the body moved to `apply_stop_sequences`, which `/v1/completions`
    // now calls as well. Chat behaviour is unchanged — this is the same computation,
    // in one place, so a fix to either surface reaches both.
    let (text, finish_reason) = crate::api::realize_handlers::apply_stop_sequences(
        text,
        stops,
        completion_tokens,
        max_tokens,
    );
    (text, finish_reason.as_str().to_string())
}

/// PMAT-801: parse tool calls out of a chat completion's generated text.
///
/// Returns `(message, finish_reason)` ready for the response. The ENTIRE
/// tool-calling path is gated by the caller on `request.tools.is_some()`; this
/// helper additionally honours `tool_choice: "none"` by skipping parsing. When
/// the parser finds at least one tool call, `tool_calls` is populated (id,
/// type:"function", function:{name, arguments-as-JSON-STRING}) and
/// `finish_reason` becomes `"tool_calls"`. Otherwise the message is a normal
/// assistant text turn and the supplied `finish_reason` is preserved.
fn build_tool_calling_message(
    text: String,
    finish_reason: String,
    tools: &[super::OpenAiTool],
    tool_choice: Option<&crate::grammar::ToolChoice>,
) -> (ChatMessage, String) {
    use crate::grammar::{ToolCallParser, ToolChoice};

    // tool_choice:"none" → never parse; behave like a plain text turn.
    if matches!(tool_choice, Some(ToolChoice::None)) {
        return (
            ChatMessage {
                role: "assistant".to_string(),
                content: text,
                ..Default::default()
            },
            finish_reason,
        );
    }

    let defs: Vec<crate::grammar::ToolDefinition> =
        tools.iter().map(super::OpenAiTool::to_grammar).collect();
    let mut parser = ToolCallParser::new(defs);
    let calls = parser.parse(&text);

    if calls.is_empty() {
        return (
            ChatMessage {
                role: "assistant".to_string(),
                content: text,
                ..Default::default()
            },
            finish_reason,
        );
    }

    let response_calls: Vec<super::ResponseToolCall> = calls
        .into_iter()
        .map(super::ResponseToolCall::from)
        .collect();
    (
        ChatMessage {
            role: "assistant".to_string(),
            // OpenAI emits empty/null content alongside tool_calls.
            content: String::new(),
            tool_calls: Some(response_calls),
            ..Default::default()
        },
        "tool_calls".to_string(),
    )
}

/// PMAT-801: map a request's `tool_choice` to the `grammar::ToolChoice` library
/// type (or `None` when the request omitted it). Centralised so every backend
/// call site stays a one-liner.
fn request_tool_choice(request: &ChatCompletionRequest) -> Option<crate::grammar::ToolChoice> {
    request
        .tool_choice
        .as_ref()
        .map(super::OpenAiToolChoice::to_grammar)
}

/// Build a non-streaming ChatCompletionResponse.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_chat_response(
    request_id: String,
    model: String,
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    max_tokens: usize,
    stops: Option<&[String]>,
    trace_level: Option<&str>,
    latency: Duration,
    tools: Option<&[super::OpenAiTool]>,
    tool_choice: Option<crate::grammar::ToolChoice>,
    timings: Option<super::Timings>,
) -> Response {
    let (brick_trace, step_trace, layer_trace) = build_trace_data(
        trace_level,
        latency.as_micros() as u64,
        prompt_tokens,
        completion_tokens,
        28,
    );
    let (text, finish_reason) = finalize_chat_text(text, stops, completion_tokens, max_tokens);

    // PMAT-801 no-regression: the tool-calling path is reached ONLY when the
    // request carried `tools`. Without tools the message is byte-identical to
    // pre-PMAT-801 (a plain assistant text turn + the original finish_reason).
    let (message, finish_reason) = match tools {
        Some(tools) => build_tool_calling_message(text, finish_reason, tools, tool_choice.as_ref()),
        None => (
            ChatMessage {
                role: "assistant".to_string(),
                content: text,
                ..Default::default()
            },
            finish_reason,
        ),
    };

    Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        brick_trace,
        step_trace,
        layer_trace,
        timings,
    })
    .into_response()
}

/// Serialize a value to an SSE event, returning `None` if serialization fails.
fn sse_event(value: &impl serde::Serialize) -> Option<Result<Event, Infallible>> {
    serde_json::to_string(value)
        .ok()
        .map(|data| Ok(Event::default().data(data)))
}

/// Decode a single streamed token, returning the text if non-empty.
///
/// The decode is deliberately RAW. `clean_chat_output()` must never be applied
/// per token: it opens with `text.trim_start()` and closes with `.trim()`, so
/// running it on one token at a time deletes the leading space or newline that
/// BPE carries on the token itself (`"Ġquick"` -> `" quick"` -> `"quick"`).
/// Concatenating the deltas then yields `"Thequickbrownfox"` while the
/// non-streaming response for the same prompt reads `"The quick brown fox"`.
/// Its other jobs cannot work per-token either: stop sequences and the
/// `Human:`/`Assistant:` turn markers span several tokens, so a single-token
/// `find()` never matches them. Whole-response cleaning belongs to the
/// non-streaming path, which already does it.
///
/// This mirrors the same removal PMAT-759 made on `pregenerated_sse_response`.
fn decode_token(tokenizer: &BPETokenizer, token_id: u32) -> Option<String> {
    let text = tokenizer.decode(&[token_id]).ok()?;
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Build a pre-generated SSE streaming response (all tokens already generated).
///
/// PMAT-759: precompute char-safe, stop-truncated deltas (the same fix as PMAT-758's
/// chat_completions_stream handler) — the previous per-token `decode_token()` split
/// multi-byte UTF-8 (emoji/CJK -> U+FFFD) and ignored request.stop on the cuda/gpu/cached
/// chat STREAMING backends (the production GPU streaming path). All three callers passed
/// `clean = false`, so the old `clean` param is dropped in favour of `stops`.
///
/// PP-27: this builder declares `stream_mode: "replayed"` on its first chunk.
/// It is not a slower live stream — the whole generation finished before the
/// first delta was written, so a client's inter-token gaps here measure the
/// SSE writer, not the model. A receipt that recorded `ttft`/`itl_p95` off this
/// path would be recording the wrong thing, which is why the mode is declared
/// rather than inferred.
fn pregenerated_sse_response(
    token_ids: Vec<u32>,
    tokenizer: Arc<BPETokenizer>,
    request_id: String,
    model_name: String,
    stops: Option<&[String]>,
    max_tokens: usize,
    prompt_tokens: usize,
) -> Response {
    let completion_tokens = token_ids.len();
    let StreamedText { deltas, stopped } = streaming_text_deltas(&tokenizer, &token_ids, stops);
    // #2375(6): `max_tokens` is a parameter so this path CANNOT emit a finish
    // reason without knowing the budget it was generated under. The terminal
    // chunk now agrees with the non-streaming body for the same request.
    let finish = FinishReason::from_generation(stopped, completion_tokens, max_tokens);
    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };
    let stream = async_stream::stream! {
        if let Some(evt) = sse_event(&ChatCompletionChunk::initial_with_mode(
            &request_id,
            &model_name,
            StreamMode::Replayed,
        )) {
            yield evt;
        }

        for delta in &deltas {
            let chunk = ChatCompletionChunk::content(&request_id, &model_name, delta);
            if let Some(evt) = sse_event(&chunk) {
                yield evt;
            }
        }

        // PP-27: no `timings` here. This path has no prefill/decode split to
        // report — generation was already over when the response was built.
        if let Some(evt) = sse_event(&ChatCompletionChunk::done_with_usage(
            &request_id,
            &model_name,
            finish,
            usage,
            None,
        )) {
            yield evt;
        }
        yield Ok::<_, Infallible>(Event::default().data("[DONE]".to_string()));
    };
    Sse::new(stream).into_response()
}

/// The `on_token` callback every streaming backend hands to its decode loop.
///
/// This is the mechanism that stops an ABANDONED stream, and the reason the
/// cancellation layer is allowed to disarm its guard on completion
/// (aprender#2375(1)): a streaming handler returns its SSE response while the
/// decode loop is still running, so cancelling on completion emptied every
/// reply. What covers the abandoned case instead is this chain —
///
/// > hyper drops the response body → the SSE receiver drops → this send fails →
/// > the callback returns `false` → `generate_with_cache_streaming` breaks.
///
/// It is one function because all three streaming backends (quantized, CUDA,
/// Qwen3-MoE) carried their own copy of `|t| tx.blocking_send(Ok(t)).is_ok()`,
/// and a copy is exactly how one of them would end up ignoring the send result
/// and burning a core for a client that left.
///
/// The abandonment is RECORDED, which is what makes the mechanism observable:
/// `record_stream_abandoned` fires once per abandoned stream, and a second
/// record for the same stream means the loop did not break.
pub(crate) fn streaming_token_sink(
    tx: tokio::sync::mpsc::Sender<Result<u32, String>>,
    metrics: Arc<crate::metrics::MetricsCollector>,
) -> impl FnMut(u32) -> bool {
    move |token_id| {
        if tx.blocking_send(Ok(token_id)).is_ok() {
            return true;
        }
        metrics.record_stream_abandoned();
        false
    }
}

/// Build a true-streaming SSE response with keep-alive (tokens arrive via channel).
///
/// Deltas are raw per-token decodes — see `decode_token`. The `clean` parameter
/// this function used to take is gone on purpose: two of its three call sites
/// passed `true`, and per-token cleaning silently deleted every space and
/// newline from the stream.
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
///
/// PP-27: this builder declares `stream_mode: "live"` on its first chunk and
/// carries `usage` — and, when the engine reported them, the server-measured
/// `timings` — on the terminal one. `timings_rx` is the engine's return path:
/// a backend that cannot separate prefill from decode simply passes `None` and
/// the terminal chunk carries no `timings` key, which is the honest reading.
pub(crate) fn true_streaming_sse_response(
    rx: tokio::sync::mpsc::Receiver<Result<u32, String>>,
    tokenizer: Arc<BPETokenizer>,
    request_id: String,
    model_name: String,
    metrics: Arc<crate::metrics::MetricsCollector>,
    start: Instant,
    max_tokens: usize,
    prompt_tokens: usize,
    timings_rx: Option<tokio::sync::oneshot::Receiver<super::PhaseTimings>>,
) -> Response {
    use tokio_stream::wrappers::ReceiverStream;
    use tokio_stream::StreamExt;

    let token_stream = ReceiverStream::new(rx);
    let mut completion_tokens = 0usize;

    let stream = async_stream::stream! {
        if let Some(evt) = sse_event(&ChatCompletionChunk::initial_with_mode(
            &request_id,
            &model_name,
            StreamMode::Live,
        )) {
            yield evt;
        }

        tokio::pin!(token_stream);
        while let Some(result) = token_stream.next().await {
            match result {
                Ok(token_id) => {
                    completion_tokens += 1;
                    if let Some(text) = decode_token(&tokenizer, token_id) {
                        let chunk = ChatCompletionChunk::content(&request_id, &model_name, &text);
                        if let Some(evt) = sse_event(&chunk) {
                            yield evt;
                        }
                    }
                }
                Err(e) => {
                    if let Some(evt) = sse_event(&serde_json::json!({ "error": e })) {
                        yield evt;
                    }
                    break;
                }
            }
        }

        // #2375(6): a token stream that delivered the whole budget was cut off at
        // `max_tokens`; anything shorter ended on a stop/EOS token.
        let finish = FinishReason::from_generation(false, completion_tokens, max_tokens);
        // The engine has finished by the time the token channel closed, so the
        // oneshot either already carries the measurement or never will.
        let timings = match timings_rx {
            Some(rx) => rx
                .await
                .ok()
                .and_then(|phases| phases.to_timings(prompt_tokens, completion_tokens)),
            None => None,
        };
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        };
        if let Some(evt) = sse_event(&ChatCompletionChunk::done_with_usage(
            &request_id,
            &model_name,
            finish,
            usage,
            timings,
        )) {
            yield evt;
        }

        metrics.record_success(completion_tokens, start.elapsed());
        yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

// ============================================================================
// Backend dispatch — each returns Some(response) if handled, None to fallthrough
// ============================================================================

/// GPU (non-batched) backend.
#[cfg(feature = "gpu")]
fn try_gpu_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
    cancel: &crate::generate::CancelToken,
) -> Option<Response> {
    use crate::gpu::GpuGenerateConfig;

    let gpu_model_lock = state.gpu_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    // GH-319: Use actual model architecture for chat template detection
    let arch_hint = state.model_architecture();
    let prompt_ids =
        match tokenize_chat_prompt(&tokenizer, &request.messages, arch_hint.as_deref(), state) {
            Ok(ids) => ids,
            Err(r) => return Some(r),
        };
    let prompt_tokens = prompt_ids.len();
    let prompt_usize: Vec<usize> = prompt_ids.iter().map(|&x| x as usize).collect();
    let (max_tokens, temperature, eos_token_id) =
        chat_gen_params(request, &tokenizer, state.model_eos_token_id());

    let gpu_config = GpuGenerateConfig {
        max_tokens,
        temperature,
        top_k: resolve_chat_top_k(temperature, request.top_k),
        stop_tokens: chat_stop_tokens(request, eos_token_id)
            .into_iter()
            .map(|t| t as usize)
            .collect(),
        trace: state.should_trace(trace_level),
        cancel: cancel.clone(),
    };

    let mut model = match gpu_model_lock.write() {
        Ok(m) => m,
        Err(e) => {
            return Some(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("GPU model lock error: {e}"),
            ));
        },
    };
    let generated = match model.generate(&prompt_usize, &gpu_config) {
        Ok(g) => g,
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let token_ids: Vec<u32> = generated
        .iter()
        .skip(prompt_tokens)
        .map(|&x| x as u32)
        .collect();
    let completion_tokens = token_ids.len();

    if request.stream {
        state
            .metrics
            .record_success(completion_tokens, start.elapsed());
        return Some(pregenerated_sse_response(
            token_ids,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            request.stop.as_deref(),
            max_tokens,
            prompt_tokens,
        ));
    }

    let text = match tokenizer.decode(&token_ids) {
        Ok(t) => clean_chat_output(&t),
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let latency = start.elapsed();
    state.metrics.record_success(completion_tokens, latency);
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        request.stop.as_deref(),
        trace_level,
        latency,
        request.tools.as_deref(),
        request_tool_choice(request),
        // This backend does not separate prefill from decode; §3 timings are
        // absent rather than zero.
        None,
    ))
}

/// Cached model (GPU batched) backend.
#[cfg(feature = "gpu")]
fn try_cached_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
    cancel: &crate::generate::CancelToken,
) -> Option<Response> {
    use crate::gguf::QuantizedGenerateConfig;

    let cached_model = state.cached_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    // GH-319: Use actual model architecture for chat template detection
    let arch_hint = state.model_architecture();
    let prompt_ids =
        match tokenize_chat_prompt(&tokenizer, &request.messages, arch_hint.as_deref(), state) {
            Ok(ids) => ids,
            Err(r) => return Some(r),
        };
    let prompt_tokens = prompt_ids.len();
    let (max_tokens, temperature, eos_token_id) =
        chat_gen_params(request, &tokenizer, state.model_eos_token_id());

    let q_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: resolve_chat_top_k(temperature, request.top_k),
        stop_tokens: chat_stop_tokens(request, eos_token_id),
        trace: state.should_trace(trace_level),
        cancel: cancel.clone(),
        ..Default::default()
    };

    let generated = match cached_model
        .model()
        .generate_with_cache(&prompt_ids, &q_config)
    {
        Ok(g) => g,
        // aprender#2376(9): context-budget rejections are 400, not 500.
        Err(e) => return Some(fail_response(state, super::generation_error_status(&e), e)),
    };

    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();

    if request.stream {
        state
            .metrics
            .record_success(completion_tokens, start.elapsed());
        return Some(pregenerated_sse_response(
            token_ids,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            request.stop.as_deref(),
            max_tokens,
            prompt_tokens,
        ));
    }

    let text = match tokenizer.decode(&token_ids) {
        Ok(t) => clean_chat_output(&t),
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let latency = start.elapsed();
    state.metrics.record_success(completion_tokens, latency);
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        request.stop.as_deref(),
        trace_level,
        latency,
        request.tools.as_deref(),
        request_tool_choice(request),
        // This backend does not separate prefill from decode; §3 timings are
        // absent rather than zero.
        None,
    ))
}

#[cfg(test)]
mod pmat756_chat_stop_tests {
    use super::finalize_chat_text;

    fn stops(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn truncates_at_stop_and_reports_stop_reason() {
        // A stop string in the text is removed; finish_reason is "stop".
        let s = stops(&["<|im_end|>"]);
        let (text, reason) = finalize_chat_text(
            "Hello world<|im_end|>trailing".to_string(),
            Some(&s),
            5,
            256,
        );
        assert_eq!(text, "Hello world");
        assert_eq!(reason, "stop");
    }

    #[test]
    fn stop_match_beats_length_when_max_tokens_also_hit() {
        // OpenAI semantics: a matched stop sequence takes precedence over "length"
        // even when completion_tokens >= max_tokens.
        let s = stops(&["STOP"]);
        let (text, reason) = finalize_chat_text("answerSTOPmore".to_string(), Some(&s), 256, 256);
        assert_eq!(text, "answer");
        assert_eq!(reason, "stop");
    }

    #[test]
    fn no_stop_match_and_max_tokens_hit_is_length() {
        let s = stops(&["<|im_end|>"]);
        let (text, reason) = finalize_chat_text("full answer".to_string(), Some(&s), 256, 256);
        assert_eq!(text, "full answer");
        assert_eq!(reason, "length");
    }

    #[test]
    fn no_stop_match_under_max_tokens_is_stop() {
        let s = stops(&["<|im_end|>"]);
        let (text, reason) = finalize_chat_text("short".to_string(), Some(&s), 3, 256);
        assert_eq!(text, "short");
        assert_eq!(reason, "stop");
    }

    #[test]
    fn truncates_at_earliest_position_not_first_listed() {
        // Mirrors FALSIFY-STOP-TRUNCATE-754: cut at the earliest POSITION,
        // regardless of which stop is listed first.
        let s = stops(&["world", "hello"]);
        let (text, reason) = finalize_chat_text("hello world".to_string(), Some(&s), 5, 256);
        assert_eq!(text, "");
        assert_eq!(reason, "stop");
    }

    #[test]
    fn none_stops_leaves_text_and_uses_length_when_maxed() {
        let (text, reason) = finalize_chat_text("untouched".to_string(), None, 256, 256);
        assert_eq!(text, "untouched");
        assert_eq!(reason, "length");
    }

    #[test]
    fn empty_stop_strings_are_ignored() {
        let s = stops(&["", ""]);
        let (text, reason) = finalize_chat_text("kept".to_string(), Some(&s), 1, 256);
        assert_eq!(text, "kept");
        assert_eq!(reason, "stop");
    }
}

#[cfg(test)]
mod pmat801_tool_calling_tests {
    //! PMAT-801: wire aprender's tool-calling library into /v1/chat/completions.
    //!
    //! Falsifier contract (apr-serve-openai-compat-v1 §F-TOOL-CALL-801):
    //!   - A request WITH `tools` + generated text containing a tool call →
    //!     `tool_calls` populated + `finish_reason == "tool_calls"` (RED without wiring).
    //!   - A request WITHOUT `tools` → message is a plain assistant text turn,
    //!     `tool_calls` is None, `finish_reason` unchanged (regression guard).
    //!   - `arguments` serializes as a JSON STRING, not a nested object.
    //!   - `tool_choice: "none"` → parsing is skipped even when tools are present.
    use super::{build_tool_calling_message, request_tool_choice};
    use crate::api::{
        ChatCompletionRequest, ChatMessage, OpenAiFunctionDef, OpenAiTool, OpenAiToolChoice,
        OpenAiToolChoiceFunction,
    };

    fn weather_tool() -> OpenAiTool {
        OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunctionDef {
                name: "get_weather".to_string(),
                description: "Get the weather for a city".to_string(),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                })),
            },
        }
    }

    /// FALSIFIER (RED without wiring): tools present + a tool call in the text →
    /// `tool_calls` populated, `finish_reason == "tool_calls"`, args is a STRING.
    #[test]
    fn tool_call_in_text_populates_tool_calls_and_finish_reason() {
        let tools = vec![weather_tool()];
        let generated = r#"{"name": "get_weather", "arguments": {"city": "NYC"}}"#.to_string();
        let (msg, reason) = build_tool_calling_message(generated, "stop".to_string(), &tools, None);

        assert_eq!(
            reason, "tool_calls",
            "finish_reason must flip to tool_calls"
        );
        let calls = msg
            .tool_calls
            .as_ref()
            .expect("tool_calls must be populated when a tool call is parsed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_type, "function");
        assert_eq!(calls[0].function.name, "get_weather");
        // arguments is a JSON STRING (not a nested object), per OpenAI wire format.
        let parsed: serde_json::Value = serde_json::from_str(&calls[0].function.arguments)
            .expect("arguments must itself be valid JSON when parsed as a string");
        assert_eq!(parsed["city"], "NYC");
    }

    /// arguments must serialize as a STRING in the response JSON, not an object.
    #[test]
    fn arguments_serialize_as_json_string_not_object() {
        let tools = vec![weather_tool()];
        let generated = r#"{"name": "get_weather", "arguments": {"city": "NYC"}}"#.to_string();
        let (msg, _) = build_tool_calling_message(generated, "stop".to_string(), &tools, None);
        let json = serde_json::to_string(&msg).expect("serialize message");
        // The wire form must contain an escaped string for arguments.
        assert!(
            json.contains(r#""arguments":"{"#) || json.contains(r#""arguments":"{\""#),
            "arguments must be a JSON string, got: {json}"
        );
        // And must NOT contain a bare object: "arguments":{"city"
        assert!(
            !json.contains(r#""arguments":{"city""#),
            "arguments must NOT be a nested object, got: {json}"
        );
    }

    /// REGRESSION GUARD: text without a tool call → no tool_calls, reason unchanged.
    #[test]
    fn no_tool_call_in_text_leaves_message_plain() {
        let tools = vec![weather_tool()];
        let (msg, reason) = build_tool_calling_message(
            "The weather is sunny.".to_string(),
            "stop".to_string(),
            &tools,
            None,
        );
        assert_eq!(reason, "stop", "finish_reason preserved when no tool call");
        assert!(msg.tool_calls.is_none());
        assert_eq!(msg.content, "The weather is sunny.");
        assert_eq!(msg.role, "assistant");
    }

    /// tool_choice "none" skips parsing even when a tool call is present in text.
    #[test]
    fn tool_choice_none_skips_parsing() {
        let tools = vec![weather_tool()];
        let choice = OpenAiToolChoice::Mode("none".to_string()).to_grammar();
        let generated = r#"{"name": "get_weather", "arguments": {"city": "NYC"}}"#.to_string();
        let (msg, reason) = build_tool_calling_message(
            generated.clone(),
            "stop".to_string(),
            &tools,
            Some(&choice),
        );
        assert_eq!(reason, "stop", "tool_choice none must NOT emit tool_calls");
        assert!(msg.tool_calls.is_none());
        assert_eq!(msg.content, generated);
    }

    /// Specific tool_choice maps to grammar ToolChoice::Specific(name).
    #[test]
    fn specific_tool_choice_maps_to_grammar() {
        let choice = OpenAiToolChoice::Specific {
            choice_type: "function".to_string(),
            function: OpenAiToolChoiceFunction {
                name: "get_weather".to_string(),
            },
        };
        match choice.to_grammar() {
            crate::grammar::ToolChoice::Specific(name) => assert_eq!(name, "get_weather"),
            other => panic!("expected Specific, got {other:?}"),
        }
    }

    /// Request without tools yields no grammar tool_choice (gate stays closed).
    #[test]
    fn request_without_tool_choice_is_none() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .expect("deserialize bare request");
        assert!(req.tools.is_none(), "no tools field → None");
        assert!(request_tool_choice(&req).is_none());
    }

    /// OpenAI tools + tool_choice deserialize from the standard request JSON.
    #[test]
    fn openai_request_with_tools_deserializes() {
        let body = r#"{
            "model": "m",
            "messages": [{"role":"user","content":"weather in NYC?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
                }
            }],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("deserialize");
        let tools = req.tools.as_ref().expect("tools present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        // tool_choice "auto" maps to grammar Auto.
        assert!(matches!(
            request_tool_choice(&req),
            Some(crate::grammar::ToolChoice::Auto)
        ));
        // The grammar ToolDefinition extracts the "city" param as required.
        let def = tools[0].to_grammar();
        assert_eq!(def.name, "get_weather");
        assert!(def
            .parameters
            .iter()
            .any(|p| p.name == "city" && p.required));
    }

    /// A tool RESULT message (role "tool" + tool_call_id) round-trips through serde.
    #[test]
    fn tool_result_message_round_trips() {
        let body = r#"{"role":"tool","content":"{\"temp\":72}","tool_call_id":"call_0"}"#;
        let msg: ChatMessage = serde_json::from_str(body).expect("deserialize tool result");
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_0"));
        assert_eq!(msg.content, r#"{"temp":72}"#);
    }

    /// Regression: a plain assistant message serializes WITHOUT the tool fields
    /// (skip_serializing_if), so non-tool responses are byte-identical to before.
    #[test]
    fn plain_message_omits_tool_fields_in_json() {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: "hello".to_string(),
            ..Default::default()
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        assert_eq!(json, r#"{"role":"assistant","content":"hello"}"#);
    }
}

include!("cuda_chat_backend.rs");
include!("chat_completions_stream.rs");
