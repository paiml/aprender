//! FALSIFY-LOGPROBS-CUDA-4971: chat `logprobs` on the dense CUDA backend
//! (ASOC-INV-021), through the real router, on the direct path and through the
//! batch scheduler (one request at a time, so its m=1 path).
//!
//! Before this the CUDA backend answered a request for logprobs with 501. It
//! needs a CUDA device and tinyllama; without either the test prints SKIP,
//! which is not_measured, never a pass.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::Value;
use tower::util::ServiceExt;

use super::native_routes_2376::{body_string, post};
use crate::api::{create_router, AppState};

const CHAT: &str = "/v1/chat/completions";

const TINYLLAMA: [&str; 2] = [
    "/home/noah/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "/mnt/nvme-raid0/cache/apr-home/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
];

/// tinyllama on CUDA device 0 behind the direct path, or `None` (and a SKIP
/// line) when the model or the device is missing.
fn cuda_state() -> Option<AppState> {
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};
    let Some(path) = TINYLLAMA
        .iter()
        .map(std::path::Path::new)
        .find(|p| p.exists())
    else {
        eprintln!("SKIP: tinyllama not present");
        return None;
    };
    let mapped = MappedGGUFModel::from_path(path).expect("map tinyllama");
    let vocab = mapped
        .model
        .vocabulary()
        .expect("tinyllama has a vocabulary");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("load tinyllama");
    let cuda = match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(cuda) => cuda,
        Err(e) => {
            eprintln!("SKIP: no CUDA device: {e}");
            return None;
        },
    };
    Some(AppState::with_cuda_model_and_vocab(cuda, vocab).expect("CUDA state"))
}

/// The same model behind a running batch scheduler, as `apr serve` builds it.
fn scheduled(state: &AppState) -> AppState {
    use crate::api::cuda_batch_scheduler::{spawn_cuda_batch_scheduler, CudaBatchConfig};
    let model = state.cuda_model().expect("a CUDA state").clone();
    let tx = spawn_cuda_batch_scheduler(
        model,
        CudaBatchConfig::default(),
        crate::api::InFlightCounter::new(),
    );
    state.clone().with_cuda_batch_tx(tx)
}

fn chat_body(extra: &str) -> String {
    format!(
        r#"{{"model":"default","messages":[{{"role":"user","content":"The capital of France is"}}],"max_tokens":8,"temperature":0{extra}}}"#
    )
}

fn logprob(entry: &Value) -> f64 {
    entry["logprob"].as_f64().expect("logprob is a number")
}

async fn chat_json(state: &AppState, extra: &str) -> Value {
    let (status, body) = post(state.clone(), CHAT, &chat_body(extra)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_str(&body).expect("chat reply is JSON")
}

/// The `logprobs.content` entries of a streamed reply, in order, and its
/// `usage.completion_tokens`.
async fn streamed_entries(state: &AppState, extra: &str) -> (Vec<Value>, u64) {
    let body = chat_body(&format!(r#","stream":true{extra}"#));
    let (status, body) = post(state.clone(), CHAT, &body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("an SSE chunk is JSON"))
        .collect();
    assert!(
        chunks.iter().all(|c| c.get("error").is_none()),
        "the stream carried an error: {body}"
    );
    let completion_tokens = chunks
        .iter()
        .find_map(|c| c["usage"]["completion_tokens"].as_u64())
        .expect("the terminal chunk carries usage");
    let entries = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["logprobs"]["content"].as_array())
        .flatten()
        .cloned()
        .collect();
    (entries, completion_tokens)
}

/// One entry per completion token, each with its best alternatives, the
/// chosen one a best one (greedy); the same reply as without logprobs; and the
/// same entries streamed.
async fn logprobs_are_served(state: &AppState, path: &str) {
    let reply = chat_json(state, r#","logprobs":true,"top_logprobs":3"#).await;
    let completion_tokens = reply["usage"]["completion_tokens"]
        .as_u64()
        .expect("usage.completion_tokens");
    let content = reply["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap_or_else(|| panic!("{path}: no choices[0].logprobs.content: {reply}"));
    assert!(completion_tokens > 0, "{path}: generated nothing: {reply}");
    assert_eq!(content.len() as u64, completion_tokens, "{path}: {reply}");
    for entry in content {
        let tops = entry["top_logprobs"].as_array().expect("top_logprobs");
        assert_eq!(tops.len(), 3, "{path}: {entry}");
        assert!(logprob(entry) <= 0.0, "{path}: {entry}");
        assert!(
            tops.windows(2).all(|w| logprob(&w[0]) >= logprob(&w[1])),
            "{path}: {entry}"
        );
        assert!(
            (logprob(entry) - logprob(&tops[0])).abs() < 1e-4,
            "{path}: {entry}"
        );
    }

    let without = chat_json(state, "").await;
    assert_eq!(
        reply["choices"][0]["message"]["content"], without["choices"][0]["message"]["content"],
        "{path}: asking for logprobs changed the reply"
    );
    assert_eq!(
        reply["usage"]["completion_tokens"], without["usage"]["completion_tokens"],
        "{path}"
    );
    let choice = without["choices"][0].as_object().expect("choice object");
    assert!(!choice.contains_key("logprobs"), "{path}: {without}");

    let (streamed, streamed_tokens) =
        streamed_entries(state, r#","logprobs":true,"top_logprobs":3"#).await;
    assert_eq!(
        streamed.len() as u64,
        streamed_tokens,
        "{path}: {streamed:?}"
    );
    assert_eq!(streamed.len(), content.len(), "{path}: {streamed:?}");
    for (s, u) in streamed.iter().zip(content) {
        assert_eq!(s["token"], u["token"], "{path}: {s} vs {u}");
        assert!((logprob(s) - logprob(u)).abs() < 1e-3, "{path}: {s} vs {u}");
    }
}

async fn traced_logprobs_are_refused(state: &AppState) {
    let request = Request::builder()
        .method("POST")
        .uri(CHAT)
        .header("content-type", "application/json")
        .header("X-Trace-Level", "brick")
        .body(Body::from(chat_body(r#","logprobs":true"#)))
        .expect("build request");
    let response = create_router(state.clone())
        .oneshot(request)
        .await
        .expect("dispatch");
    let status = response.status();
    let body = body_string(response).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert!(body.contains("CUDA (traced)"), "{body}");
}

/// One test, so one model on the device and one request at a time on it.
#[tokio::test]
async fn the_dense_cuda_backend_serves_chat_logprobs_direct_and_scheduled() {
    let Some(direct) = cuda_state() else {
        return;
    };
    logprobs_are_served(&direct, "direct").await;
    traced_logprobs_are_refused(&direct).await;
    logprobs_are_served(&scheduled(&direct), "scheduled").await;
}
