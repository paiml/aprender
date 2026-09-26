//! #3571: `apr serve` answers a Qwen3.5 chat request with what `apr run`'s
//! one-shot path produces for the same prompt — streaming and not — and no
//! dense endpoint decodes through the zero-layer base any more.

use super::*;
use crate::api::{create_router, Qwen35Served};
use crate::gguf::qwen35_session::Qwen35Session;
use crate::gguf::{MappedGGUFModel, QuantizedGenerateConfig};
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

/// The real hybrid file the rest of the Qwen3.5 tests are specified against.
const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

fn state_or_skip(no_gpu: bool) -> Option<(AppState, Arc<MappedGGUFModel>)> {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {MODEL_PATH} is absent");
        return None;
    }
    let mapped = Arc::new(MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF"));
    let vocab = mapped
        .model
        .vocabulary()
        .expect("the GGUF has a vocabulary");
    let session = Qwen35Session::load(&mapped, no_gpu).expect("load the hybrid");
    let state = AppState::with_qwen35_session(session, mapped.clone(), vocab).expect("app state");
    Some((state, mapped))
}

const QUESTION: &str = "Name the capital of Peru in one word.";

/// What `apr run` answers (#3990: the GGUF's OWN template, which serve and run both render), the GGUF's tokenizer, the
/// one-shot generate on the same route, greedy, stopping at the model's EOS.
fn one_shot_answer(mapped: &MappedGGUFModel, max_tokens: usize, no_gpu: bool) -> String {
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: QUESTION.to_string(),
        ..Default::default()
    }];
    let prompt = mapped
        .model
        .encode(&crate::api::format_chat_messages_official(
            Some(&mapped.model),
            &messages,
            mapped.model.architecture(),
        ))
        .expect("encode");
    let eos: Vec<u32> = mapped.model.eos_token_id().into_iter().collect();
    let config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: eos.clone(),
        ..Default::default()
    };
    // #4263: `apr run`'s load — the host once per file, a device state sized
    // to the one call.
    let qwen = crate::gguf::qwen35_session::Qwen35Forward::cached_host(
        std::path::Path::new(MODEL_PATH),
        mapped,
    )
    .expect("host");
    let mut one = crate::gguf::qwen35_session::Qwen35Session::load_for_run(
        qwen,
        mapped,
        no_gpu,
        prompt.len() + max_tokens,
    )
    .expect("load");
    let turn = one
        .generate(&prompt, &config, &mut |_| true)
        .expect("one-shot generate");
    let (tokens, used_gpu) = (turn.tokens, turn.used_gpu);
    assert_eq!(
        used_gpu, !no_gpu,
        "the reference ran on the route asked for"
    );
    let mut generated = tokens[prompt.len()..].to_vec();
    if generated.last().is_some_and(|t| eos.contains(t)) {
        generated.pop();
    }
    clean_chat_output(&mapped.model.decode(&generated))
}

async fn post(app: axum::Router, uri: &str, body: serde_json::Value) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn chat_body(stream: bool, max_tokens: usize) -> serde_json::Value {
    serde_json::json!({
        "model": "anything-the-client-likes",
        "messages": [{"role": "user", "content": QUESTION}],
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "stream": stream,
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn a_chat_request_answers_what_apr_run_answers() {
    let Some((state, mapped)) = state_or_skip(true) else {
        return;
    };
    let want = one_shot_answer(&mapped, 16, true);
    assert!(
        !want.trim().is_empty(),
        "the reference answer is empty: {want:?}"
    );

    let (status, body) = post(
        create_router(state),
        "/v1/chat/completions",
        chat_body(false, 16),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    let got = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert_eq!(
        got, want,
        "serve must hand the model apr run's tokens: {body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_chat_request_answers_what_apr_run_answers() {
    let Some((state, mapped)) = state_or_skip(true) else {
        return;
    };
    let want = one_shot_answer(&mapped, 16, true);

    let (status, body) = post(
        create_router(state),
        "/v1/chat/completions",
        chat_body(true, 16),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut got = String::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let chunk: serde_json::Value = serde_json::from_str(data).expect("SSE chunk");
        if let Some(piece) = chunk["choices"][0]["delta"]["content"].as_str() {
            got.push_str(piece);
        }
    }
    assert_eq!(got.trim(), want.trim(), "streamed: {body}");
}

/// Is this completion the shape #3571 measured — output that is present but carries
/// no information?
///
/// The original defect answered `/v1/completions` with HTTP 200 and **1024 tokens of
/// `"\n"`**: a dense endpoint decoding through a zero-layer base. Two ways that looks:
/// nothing at all, or one character repeated.
///
/// This is a FUNCTION rather than an inline assertion so it can be proven RED against
/// the historical output directly, without needing a model that reproduces the bug.
fn is_degenerate_completion(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    let mut chars = trimmed.chars();
    let first = chars.next().unwrap_or(' ');
    chars.all(|c| c == first)
}

#[test]
fn the_degeneracy_check_catches_the_defect_it_is_named_for() {
    // #3571's exact output. If this line ever goes green, the test below has been
    // replaced by one that cannot see the defect it is named for -- which is worse
    // than the 503 it used to assert, because it would look like coverage.
    assert!(is_degenerate_completion(&"\n".repeat(1024)), "1024 newlines");
    assert!(is_degenerate_completion(""), "empty");
    assert!(is_degenerate_completion("   \t  "), "whitespace only");
    assert!(is_degenerate_completion("aaaaaaaa"), "one character repeated");
    // And it must NOT fire on a real answer, or it would fail every green run.
    assert!(!is_degenerate_completion(" Lima, and the capital of Haiti is"));
    assert!(!is_degenerate_completion(" Paris"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dense_endpoint_no_longer_decodes_through_the_zero_layer_base() {
    let Some((state, _)) = state_or_skip(true) else {
        return;
    };
    // #3571 measured this at batch-1: HTTP 200 with 1024 tokens of "\n".
    //
    // THIS TEST ASSERTED `status != 200` UNTIL #3874. That was a proxy, not the
    // intent: when it was written there was no correct path for this endpoint on a
    // Qwen3.5 hybrid, so "do not answer at all" was the only way to say "do not
    // answer with garbage". It was green because the endpoint 503'd -- the same
    // reason the feature was broken.
    //
    // #3874 added the Qwen3.5 arm, so the endpoint now answers correctly and the
    // proxy is obsolete. The intent is unchanged and is now asserted directly: a
    // dense endpoint must not answer with output decoded through a base that holds
    // no model. `is_degenerate_completion` is proven against #3571's exact output in
    // `the_degeneracy_check_catches_the_defect_it_is_named_for`.
    //
    // The assertion reads the FIELD, never a line or a position: aprender-55's
    // must-RED fixture showed a "non-empty" check survive an empty-text mutant
    // because it read the last line, and the last line became the id line when the
    // text vanished.
    let (status, body) = post(
        create_router(state),
        "/v1/completions",
        serde_json::json!({"model": "x", "prompt": "The capital of Peru is", "max_tokens": 8}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "the endpoint must answer: {body}");
    let doc: serde_json::Value = serde_json::from_str(&body).expect("a JSON body");
    let text = doc["choices"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("no choices[0].text in {body}"));
    assert!(
        !is_degenerate_completion(text),
        "a dense endpoint answered through the base (#3571): text={text:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_past_the_declared_context_is_a_400_naming_both_numbers() {
    let Some((mut state, mapped)) = state_or_skip(true) else {
        return;
    };
    // The real declared context is 262 144 tokens — a prompt past it takes
    // minutes to tokenize in a debug build. The check under test is the
    // handler's, so the served state declares a context of 16 instead.
    const CONTEXT: usize = 16;
    state.qwen35_session = Some(Arc::new(Qwen35Served {
        context_length: CONTEXT,
        on_gpu: std::sync::atomic::AtomicBool::new(false),
        session: std::sync::Mutex::new(Qwen35Session::load(&mapped, true).expect("load")),
    }));
    let (status, body) = post(
        create_router(state),
        "/v1/chat/completions",
        serde_json::json!({"model": "x", "messages": [{"role": "user", "content": QUESTION}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body.contains("refused whole rather than truncated"),
        "{body}"
    );
    assert!(
        body.contains(&format!("declares a context of {CONTEXT}")),
        "{body}"
    );
}

/// A reply the context cuts short decodes only what the context left and says
/// `finish_reason: "length"` — the request asked for 50, the context leaves 2.
#[tokio::test(flavor = "multi_thread")]
async fn a_reply_the_context_cuts_short_decodes_the_budget_and_reports_length() {
    let Some((mut state, mapped)) = state_or_skip(true) else {
        return;
    };
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: QUESTION.to_string(),
        ..Default::default()
    }];
    let prompt_tokens = mapped
        .model
        .encode(&crate::api::format_chat_messages_official(
            Some(&mapped.model),
            &messages,
            mapped.model.architecture(),
        ))
        .expect("encode")
        .len();
    state.qwen35_session = Some(Arc::new(Qwen35Served {
        context_length: prompt_tokens + 2,
        on_gpu: std::sync::atomic::AtomicBool::new(false),
        session: std::sync::Mutex::new(Qwen35Session::load(&mapped, true).expect("load")),
    }));
    let mut body = chat_body(false, 50);
    body["ignore_eos"] = serde_json::json!(true);
    let (status, body) = post(create_router(state), "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    assert_eq!(json["choices"][0]["finish_reason"], "length", "{body}");
    assert_eq!(
        json["usage"]["completion_tokens"], 2,
        "the budget, not max_tokens: {body}"
    );
}

/// A Qwen3.5 server is healthy and ready: its session is the model. Measured
/// before this row counted it: `/health` answered 503 "loading" forever, so
/// every probe — and every harness that waits on it — treated a working server
/// as down.
#[tokio::test(flavor = "multi_thread")]
async fn a_qwen35_server_reports_healthy_and_ready() {
    let Some((state, _)) = state_or_skip(true) else {
        return;
    };
    for path in ["/health", "/health/ready"] {
        let response = create_router(state.clone())
            .oneshot(
                Request::builder()
                    .uri(path)
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("the router answers");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
        assert_eq!(status, StatusCode::OK, "{path}: {json}");
        assert_eq!(json["model_loaded"], true, "{path}: {json}");
        assert_eq!(json["compute_mode"], "cpu", "--no-gpu session: {json}");
    }
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
async fn gpu_a_chat_request_answers_from_the_gpu_session() {
    if !crate::cuda::CudaExecutor::is_available() {
        eprintln!("SKIP: no CUDA device");
        return;
    }
    let Some((state, mapped)) = state_or_skip(false) else {
        return;
    };
    let served = state.qwen35_session().expect("session");
    assert!(
        served.session.lock().expect("lock").on_gpu(),
        "a CUDA host serves from the GPU"
    );
    let want = one_shot_answer(&mapped, 16, false);

    let (status, body) = post(
        create_router(state),
        "/v1/chat/completions",
        chat_body(false, 16),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    let got = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert_eq!(
        got, want,
        "the GPU session answers what apr run --gpu answers: {body}"
    );
    assert!(
        served.session.lock().expect("lock").on_gpu(),
        "no fallback during the request"
    );
    assert!(
        served.on_gpu.load(std::sync::atomic::Ordering::Relaxed),
        "/health's flag agrees with the session"
    );
}

/// #4250 (the #3596 countermeasure, 0.69.3 G0): every serve route that takes a
/// prompt prefills it through the BATCHED prefill, not one token at a time.
///
/// 0.69.3 shipped the batched prefill on `apr run` only; serve kept looping
/// `forward_single` over the prompt (~77 tok/s at 0.69.1, the decode rate), and
/// every gate stayed green because none of them sent a prompt through the
/// router. The counter is `Qwen35Session::batched_prefills`, which moves only
/// when `Qwen35CudaModel::prefill` returned logits — the F2 probe, the
/// per-token fallback and decode steps never move it. Each request below
/// starts a prompt the session does not hold, so each must add exactly one.
///
/// RED under `APR_QWEN35_SESSION_PREFILL=per-token` (the session's own
/// switch back to the one-token loop), and with the batched branch deleted
/// from `try_advance_to`.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
async fn gpu_every_serve_route_prefills_through_the_batched_prefill() {
    if !crate::cuda::CudaExecutor::is_available() {
        eprintln!("SKIP: no CUDA device");
        return;
    }
    let Some((state, mapped)) = state_or_skip(false) else {
        return;
    };
    let served = state.qwen35_session().expect("session");
    let prefills = || served.session.lock().expect("lock").batched_prefills();
    assert!(
        served.session.lock().expect("lock").on_gpu(),
        "a CUDA host serves from the GPU"
    );
    assert_eq!(prefills(), 0, "no prompt has been served yet");
    let want = one_shot_answer(&mapped, 16, false);

    // 1. /v1/chat/completions, not streamed — and the batched answer is right.
    let (status, body) = post(
        create_router(state.clone()),
        "/v1/chat/completions",
        chat_body(false, 16),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    assert_eq!(
        json["choices"][0]["message"]["content"].as_str(),
        Some(want.as_str()),
        "the batched serve prefill answers what apr run --gpu answers: {body}"
    );
    assert_eq!(
        prefills(),
        1,
        "/v1/chat/completions prefilled its prompt one token at a time (#3596)"
    );

    // 2. /v1/chat/completions, streamed — the SSE path spawns its own generate.
    // The session holds prompt + reply, so the same prompt again does not
    // extend it: a fresh prefill from position 0.
    let (status, body) = post(
        create_router(state.clone()),
        "/v1/chat/completions",
        chat_body(true, 8),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        prefills(),
        2,
        "streamed /v1/chat/completions prefilled one token at a time (#3596)"
    );

    // 3. /v1/completions — a raw prompt of several tokens.
    let (status, body) = post(
        create_router(state.clone()),
        "/v1/completions",
        serde_json::json!({"model": "x", "prompt": "The capital of Peru is", "max_tokens": 8}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        prefills(),
        3,
        "/v1/completions prefilled its prompt one token at a time (#3596)"
    );

    assert!(
        served.session.lock().expect("lock").on_gpu(),
        "no fallback to the CPU during the requests — a CPU prefill is never batched"
    );
}

// ── #3715: the OLLAMA wire reaches the hybrid too ──────────────────────────
//
// Alfredo reported `/api/chat` 500ing on a Qwen3.5 hybrid, on the grounds that
// `ollama_handlers.rs` holds zero references to `qwen35` or the hybrid session
// while `cuda_chat_backend.rs` holds one. The reference count is correct and
// the conclusion does not follow: `ollama_chat_handler` DELEGATES to
// `openai_chat_completions_handler`, which calls `try_qwen35_backend` first and
// unconditionally, so the Ollama wire reaches the hybrid through the OpenAI
// handler rather than by naming it.
//
// Measured on `71421b6e5` before writing these tests — `apr serve run
// Qwen3.5-0.8B-Q4_K_M.gguf`, both a default build and a `--features cuda`
// build: `/api/chat`, `/api/generate`, `/api/chat` streaming and `/api/tags`
// all answered 200, and `/api/chat` returned "4" to "What is 2+2?".
//
// What IS true is the risk Alfredo names: `/api/chat` has its own translation
// layer which has diverged from the OpenAI one twice independently (this
// report, and the `tool_calls` gap fixed in #3825). Nothing asserted that the
// two wires stay on one path, so the next refactor that gives `/api/chat` its
// own generation is a silent regression. These tests are that assertion: they
// fail the moment the Ollama wire stops answering from the hybrid, whatever
// the reason.

/// The `/api/chat` wire answers from the hybrid session, not a 500 and not a
/// dense decode. RED if the Ollama handler ever grows its own generation path.
#[tokio::test(flavor = "multi_thread")]
async fn the_ollama_chat_wire_answers_from_the_hybrid() {
    let Some((state, mapped)) = state_or_skip(true) else {
        return;
    };
    let want = one_shot_answer(&mapped, 16, true);
    assert!(!want.trim().is_empty(), "reference answer empty: {want:?}");

    let (status, body) = post(
        create_router(state),
        "/api/chat",
        serde_json::json!({
            "model": "anything-the-client-likes",
            "messages": [{"role": "user", "content": QUESTION}],
            "options": {"temperature": 0.0, "top_k": 1, "num_predict": 16},
            "stream": false,
        }),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::OK,
        "#3715: /api/chat must reach the Qwen3.5 hybrid, not fail: {body}"
    );
    let json: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    let got = json["message"]["content"].as_str().expect("content");
    assert_eq!(
        got, want,
        "#3715: /api/chat must hand the model the same tokens the OpenAI wire \
         does — one predicate, two callers. Divergence here is the defect: {body}"
    );
}

/// `/api/generate`, the other ollama-compat generation route, on the same path.
#[tokio::test(flavor = "multi_thread")]
async fn the_ollama_generate_wire_answers_from_the_hybrid() {
    let Some((state, _mapped)) = state_or_skip(true) else {
        return;
    };
    let (status, body) = post(
        create_router(state),
        "/api/generate",
        serde_json::json!({
            "model": "anything-the-client-likes",
            "prompt": QUESTION,
            "options": {"temperature": 0.0, "top_k": 1, "num_predict": 16},
            "stream": false,
        }),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::OK,
        "#3715: /api/generate must reach the hybrid: {body}"
    );
    let json: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    let got = json["response"].as_str().expect("response");
    assert!(
        !got.trim().is_empty(),
        "#3715: /api/generate answered 200 with an empty body: {body}"
    );
}

/// The two wires must agree. This is the one that catches a divergence whose
/// symptom is a WRONG answer rather than an error — the shape a status-code
/// assertion cannot see, and the shape #3825's `tool_calls` gap had.
#[tokio::test(flavor = "multi_thread")]
async fn both_chat_wires_agree_on_the_same_request() {
    let Some((state, mapped)) = state_or_skip(true) else {
        return;
    };
    let openai = post(
        create_router(state),
        "/v1/chat/completions",
        chat_body(false, 16),
    )
    .await;

    let Some((state2, _)) = state_or_skip(true) else {
        return;
    };
    let ollama = post(
        create_router(state2),
        "/api/chat",
        serde_json::json!({
            "model": "anything-the-client-likes",
            "messages": [{"role": "user", "content": QUESTION}],
            "options": {"temperature": 0.0, "top_k": 1, "num_predict": 16},
            "stream": false,
        }),
    )
    .await;

    assert_eq!(openai.0, StatusCode::OK, "openai wire: {}", openai.1);
    assert_eq!(ollama.0, StatusCode::OK, "ollama wire: {}", ollama.1);

    let a: serde_json::Value = serde_json::from_str(&openai.1).expect("JSON");
    let b: serde_json::Value = serde_json::from_str(&ollama.1).expect("JSON");
    let a_content = a["choices"][0]["message"]["content"].as_str().expect("a");
    let b_content = b["message"]["content"].as_str().expect("b");
    assert_eq!(
        a_content, b_content,
        "#3715: the two chat wires disagree on one request against one model. \
         /api/chat has its own translation layer and has diverged from the \
         OpenAI one twice before (#3715 report, #3825 tool_calls)."
    );
    let _ = mapped;
}

/// #3962: the five realizar RAW routes answered 503 "No model available" on every
/// Qwen3.5 model (aprender-83's CRUX serve sweep). Each must now answer what `apr
/// run` answers for the same rendered prompt — not 200 with an echo, not a 503.
#[tokio::test(flavor = "multi_thread")]
async fn every_raw_route_answers_from_the_hybrid() {
    let Some((state, mapped)) = state_or_skip(true) else {
        return;
    };
    let expected = one_shot_answer(&mapped, 16, true);
    assert!(!expected.is_empty(), "the reference answered nothing");
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: QUESTION.to_string(),
        ..Default::default()
    }];
    let rendered = crate::api::format_chat_messages_official(
        Some(&mapped.model),
        &messages,
        mapped.model.architecture(),
    );
    let one = serde_json::json!({"prompt": rendered, "max_tokens": 16, "temperature": 0.0});
    let many = serde_json::json!({"prompts": [rendered], "max_tokens": 16, "temperature": 0.0});
    for (route, body) in [
        ("/generate", &one),
        ("/batch/generate", &many),
        ("/realize/batch", &many),
        ("/stream/generate", &one),
        ("/realize/generate", &one),
    ] {
        let (status, text) = post(create_router(state.clone()), route, body.clone()).await;
        assert_eq!(status, StatusCode::OK, "{route}: {text}");
        let answer = if route.contains("batch") {
            let v: serde_json::Value = serde_json::from_str(&text).expect("json");
            v["results"][0]["text"].as_str().expect("results[0].text").to_string()
        } else if text.starts_with("event:") || text.contains("\nevent:") || text.contains("data:") {
            text.lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
                .filter_map(|v| v["text"].as_str().map(str::to_string))
                .collect::<String>()
        } else {
            let v: serde_json::Value = serde_json::from_str(&text).expect("json");
            v["text"].as_str().expect("text").to_string()
        };
        assert!(!answer.contains(QUESTION), "{route} echoed the prompt: {answer:?}");
        assert_eq!(
            clean_chat_output(&answer),
            expected,
            "{route} answered differently from `apr run` on the same rendered prompt"
        );
    }
}

/// #3723 MUST-RED: a request that asks for thinking ON is served the model's OWN template
/// rendered with `enable_thinking=true` -- both the vLLM/SGLang spelling
/// (`chat_template_kwargs.enable_thinking`) and apr's/Ollama's (`think`), on the OpenAI and
/// Ollama wires. The official ON and OFF prompts differ in length, so `prompt_tokens` says
/// which one the model was handed.
#[tokio::test(flavor = "multi_thread")]
async fn a_thinking_on_request_is_served_the_official_on_prompt_3723() {
    let Some((state, mapped)) = state_or_skip(true) else {
        return;
    };
    let msgs = [crate::chat_template::ChatMessage::new("user", QUESTION)];
    let render = |t: Option<bool>| {
        let p = crate::chat_template::render_official_for_model(&mapped.model, &msgs, t).expect("renders");
        mapped.model.encode(&p).expect("encodes").len()
    };
    let (on_len, off_len) = (render(Some(true)), render(Some(false)));
    assert_ne!(on_len, off_len, "the probe must distinguish the ON and OFF prompts");
    let app = create_router(state);

    let openai = |extra: serde_json::Value| {
        let mut b = chat_body(false, 1);
        for (k, v) in extra.as_object().expect("object") {
            b[k] = v.clone();
        }
        b
    };
    for (label, extra, want) in [
        ("chat_template_kwargs ON", serde_json::json!({"chat_template_kwargs": {"enable_thinking": true}}), on_len),
        ("think ON", serde_json::json!({"think": true}), on_len),
        ("chat_template_kwargs OFF", serde_json::json!({"chat_template_kwargs": {"enable_thinking": false}}), off_len),
        ("absent = OFF", serde_json::json!({}), off_len),
    ] {
        let (status, body) = post(app.clone(), "/v1/chat/completions", openai(extra)).await;
        assert_eq!(status, StatusCode::OK, "{label}: {body}");
        let json: serde_json::Value = serde_json::from_str(&body).expect("JSON");
        assert_eq!(json["usage"]["prompt_tokens"].as_u64(), Some(want as u64), "{label}: {body}");
    }

    let (status, body) = post(
        app.clone(),
        "/api/chat",
        serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": QUESTION}],
            "options": {"temperature": 0.0, "top_k": 1, "num_predict": 1},
            "stream": false,
            "think": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "/api/chat think: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    assert_eq!(json["prompt_eval_count"].as_u64(), Some(on_len as u64), "/api/chat think: {body}");

    // Two spellings that disagree are refused, not picked between.
    let (status, body) = post(
        app,
        "/v1/chat/completions",
        openai(serde_json::json!({"think": false, "chat_template_kwargs": {"enable_thinking": true}})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("contradicts"), "{body}");
}

/// #3723: `chat_template_kwargs` carries only what apr honours; any other key is refused at
/// deserialization, never silently dropped.
#[test]
fn chat_template_kwargs_parse_and_refuse_unknown_keys_3723() {
    let parse = |v: serde_json::Value| serde_json::from_value::<ChatCompletionRequest>(v);
    let base = serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "x"}]});
    let with = |k: &str, v: serde_json::Value| {
        let mut b = base.clone();
        b[k] = v;
        b
    };
    assert_eq!(parse(base.clone()).expect("parses").thinking(), None);
    assert_eq!(parse(with("chat_template_kwargs", serde_json::json!({"enable_thinking": true}))).expect("parses").thinking(), Some(true));
    assert_eq!(parse(with("think", serde_json::json!(false))).expect("parses").thinking(), Some(false));
    assert!(parse(with("chat_template_kwargs", serde_json::json!({"reasoning_effort": "high"}))).is_err(), "an unknown kwarg is refused");
}

/// #4272: the stream's release rule. Nothing is sent while the decode ends in
/// half a character, and a tail that could still become a stop sequence is held.
#[test]
fn a_stream_delta_holds_back_half_characters_and_possible_stops() {
    use crate::api::realize_handlers::qwen35_stream_delta;
    assert_eq!(qwen35_stream_delta("Lima", 0, &[]).as_deref(), Some("Lima"));
    assert_eq!(qwen35_stream_delta("Lima", 4, &[]), None, "nothing new");
    assert_eq!(qwen35_stream_delta("Lim\u{FFFD}", 0, &[]), None, "half a char");
    let stops = ["END".to_string()];
    // "EN" could still become "END": two bytes are held.
    assert_eq!(qwen35_stream_delta("LimaEN", 0, &stops).as_deref(), Some("Lima"));
    assert_eq!(qwen35_stream_delta("LimaEN", 4, &stops), None);
    // The hold never splits a character.
    assert_eq!(qwen35_stream_delta("aé", 0, &["xy".to_string()]).as_deref(), Some("a"));
}

/// #4272: `stream: true` on `/v1/completions` was BUFFERED — the whole completion
/// was generated, then sliced into chunks, and no chunk carried `usage`. Now the
/// session's `on_token` drives the stream. Measured here, on the real hybrid:
/// the first text chunk arrives well before the stream ends (a buffered stream
/// delivers every chunk at the same instant), the chunks concatenate to exactly
/// the non-streamed text, and the terminal chunk carries the same `usage`.
#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_completion_arrives_token_by_token_and_ends_with_usage() {
    use http_body_util::BodyExt;
    let Some((state, _)) = state_or_skip(true) else {
        return;
    };
    let body = |stream: bool| {
        serde_json::json!({"model": "x", "prompt": "The capital of Peru is",
            "max_tokens": 24, "temperature": 0.0, "stream": stream})
    };
    let (status, plain) = post(create_router(state.clone()), "/v1/completions", body(false)).await;
    assert_eq!(status, StatusCode::OK, "{plain}");
    let plain: serde_json::Value = serde_json::from_str(&plain).expect("JSON");
    let plain_text = plain["choices"][0]["text"].as_str().expect("text").to_string();

    let t0 = std::time::Instant::now();
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body(true).to_string()))
                .expect("request"),
        )
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let (mut buf, mut first_text_at, mut text, mut deltas, mut usage) =
        (String::new(), None, String::new(), 0usize, None);
    while let Some(frame) = body.frame().await {
        let frame = frame.expect("frame");
        let Some(data) = frame.data_ref() else {
            continue;
        };
        buf.push_str(&String::from_utf8_lossy(data));
        while let Some(end) = buf.find("\n\n") {
            let event: String = buf.drain(..end + 2).collect();
            let Some(payload) = event.trim().strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if payload == "[DONE]" {
                continue;
            }
            let chunk: serde_json::Value = serde_json::from_str(payload).expect("chunk JSON");
            let piece = chunk["choices"][0]["text"].as_str().unwrap_or_default();
            if !piece.is_empty() {
                first_text_at.get_or_insert_with(|| t0.elapsed());
                deltas += 1;
                text.push_str(piece);
            }
            if !chunk["usage"].is_null() {
                usage = Some(chunk["usage"].clone());
            }
        }
    }
    let total = t0.elapsed();
    let first = first_text_at.expect("at least one text chunk");
    eprintln!("#4272: first text chunk at {first:?} of {total:?}, {deltas} deltas");
    assert_eq!(text, plain_text, "the stream must say what the body says");
    assert!(deltas >= 2, "one delta is a buffered reply: {deltas}");
    assert!(
        first.as_secs_f64() < 0.8 * total.as_secs_f64(),
        "the first chunk came at {first:?} of {total:?}: buffered, not streamed"
    );
    let usage = usage.expect("the terminal chunk carries usage (#4272)");
    assert_eq!(usage, plain["usage"], "the stream's usage is the body's");
}

/// PRM-S1 v2 (#4354): `POST /v1/chat/prompt-ids` reports the ids the chat path
/// prefills — the same count the chat reply bills as `prompt_tokens`, and the same
/// ids as rendering and encoding directly — in both thinking modes.
#[tokio::test(flavor = "multi_thread")]
async fn prompt_ids_are_the_ids_the_chat_path_prefills_4354() {
    let Some((state, mapped)) = state_or_skip(true) else {
        return;
    };
    let msgs = [crate::chat_template::ChatMessage::new("user", QUESTION)];
    let app = create_router(state);
    let mut seen = Vec::new();
    for thinking in [false, true] {
        let mut body = chat_body(false, 1);
        body["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": thinking });
        let (status, reply) = post(app.clone(), "/v1/chat/prompt-ids", body.clone()).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let json: serde_json::Value = serde_json::from_str(&reply).expect("JSON");
        let ids: Vec<u32> = serde_json::from_value(json["prompt_ids"].clone()).expect("ids");
        let direct = crate::chat_template::render_official_for_model(&mapped.model, &msgs, Some(thinking))
            .expect("renders");
        assert_eq!(json["prompt"].as_str(), Some(direct.as_str()), "thinking={thinking}");
        assert_eq!(Some(ids.clone()), mapped.model.encode(&direct), "thinking={thinking}");

        let (status, chat) = post(app.clone(), "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "{chat}");
        let chat: serde_json::Value = serde_json::from_str(&chat).expect("JSON");
        assert_eq!(chat["usage"]["prompt_tokens"].as_u64(), Some(ids.len() as u64));
        seen.push(ids);
    }
    assert_ne!(seen[0], seen[1], "the probe must distinguish the ON and OFF prompts");
}

/// #3991 applied to the new route: it is mounted and listed only where a Qwen3.5
/// session can answer it.
#[test]
fn prompt_ids_route_is_listed_only_with_a_qwen35_session_4354() {
    let config = crate::api::RouterConfig::default();
    let mut caps = crate::api::RouteCapabilities::all();
    let listed = |c| crate::api::advertised_routes_for(&config, c);
    assert!(listed(caps).iter().any(|r| r == "POST /v1/chat/prompt-ids"));
    caps.qwen35_prompt_ids = false;
    assert!(!listed(caps).iter().any(|r| r.contains("/v1/chat/prompt-ids")));
}
