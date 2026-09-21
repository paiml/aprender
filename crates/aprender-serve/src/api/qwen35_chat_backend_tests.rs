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

/// What `apr run` answers: the same template, the GGUF's tokenizer, the
/// one-shot generate on the same route, greedy, stopping at the model's EOS.
fn one_shot_answer(mapped: &MappedGGUFModel, max_tokens: usize, no_gpu: bool) -> String {
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: QUESTION.to_string(),
        ..Default::default()
    }];
    let prompt = mapped
        .model
        .encode(&format_chat_messages(
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
    let base =
        crate::gguf::forward_qwen35::Qwen35Model::create_base_model(&mapped.model, mapped.data())
            .expect("base");
    let (tokens, used_gpu) = crate::gguf::forward_qwen35::run_qwen35_generate_dispatch(
        mapped, &base, &prompt, &config, no_gpu,
    )
    .expect("one-shot generate");
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

#[tokio::test(flavor = "multi_thread")]
async fn a_dense_endpoint_no_longer_decodes_through_the_zero_layer_base() {
    let Some((state, _)) = state_or_skip(true) else {
        return;
    };
    // #3571 measured this at batch-1: HTTP 200 with 1024 tokens of "\n".
    let (status, body) = post(
        create_router(state),
        "/v1/completions",
        serde_json::json!({"model": "x", "prompt": "The capital of Peru is", "max_tokens": 8}),
    )
    .await;
    assert_ne!(
        status,
        StatusCode::OK,
        "a dense endpoint answered through the base: {body}"
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
        .encode(&format_chat_messages(
            &messages,
            mapped.model.architecture(),
        ))
        .expect("encode")
        .len();
    state.qwen35_session = Some(Arc::new(Qwen35Served {
        context_length: prompt_tokens + 2,
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
}
