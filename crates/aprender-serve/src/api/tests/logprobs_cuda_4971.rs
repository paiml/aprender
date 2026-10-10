//! FALSIFY-LOGPROBS-CUDA-4971: chat `logprobs` on the dense CUDA backend
//! (ASOC-INV-021), through the real router, on the direct path and through the
//! batch scheduler: one request at a time (its m=1 path), and two at once in
//! one m=2 batch; and on the iteration scheduler's batched path, driven
//! directly, where requests join and recycle into slots mid-batch.
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

/// The same model behind a batch scheduler that holds its first request until
/// a second fills a batch of two, and the counter whose peak shows the pair
/// decoded as one m=2 batch.
fn batch_of_two(state: &AppState) -> (AppState, std::sync::Arc<crate::api::InFlightCounter>) {
    use crate::api::cuda_batch_scheduler::{spawn_cuda_batch_scheduler, CudaBatchConfig};
    let model = state.cuda_model().expect("a CUDA state").clone();
    let counter = crate::api::InFlightCounter::new();
    let config = CudaBatchConfig {
        max_batch: 2,
        window_ms: 10_000,
    };
    let tx = spawn_cuda_batch_scheduler(model, config, counter.clone());
    (state.clone().with_cuda_batch_tx(tx), counter)
}

/// The request fields that ask for logprobs with 3 alternatives a step.
const WITH: &str = r#","logprobs":true,"top_logprobs":3"#;

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

/// `reply`'s `logprobs.content`, checked: one entry per completion token, each
/// with its 3 best alternatives, best first, the chosen one the best (greedy).
fn checked_entries<'a>(reply: &'a Value, path: &str) -> &'a [Value] {
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
    content
}

/// The tokens and logprobs of `a` and `b`, which must be the same.
fn same_entries(a: &[Value], b: &[Value], path: &str) {
    assert_eq!(a.len(), b.len(), "{path}: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x["token"], y["token"], "{path}: {x} vs {y}");
        assert!((logprob(x) - logprob(y)).abs() < 1e-3, "{path}: {x} vs {y}");
    }
}

/// One entry per completion token, each with its best alternatives, the
/// chosen one a best one (greedy); the same reply as without logprobs; and the
/// same entries streamed.
async fn logprobs_are_served(state: &AppState, path: &str) {
    let reply = chat_json(state, WITH).await;
    let content = checked_entries(&reply, path);

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

    let (streamed, streamed_tokens) = streamed_entries(state, WITH).await;
    assert_eq!(
        streamed.len() as u64,
        streamed_tokens,
        "{path}: {streamed:?}"
    );
    same_entries(&streamed, content, path);
}

/// A batched turn (m = 2) records its logprobs too. One slot asking for them
/// and one not: the same reply, entries only on the one that asked. Both
/// asking, one streamed: the same entries, and the first pair's.
async fn batched_logprobs_are_served(state: &AppState) {
    let (pair, counter) = batch_of_two(state);
    let (reply, without) = tokio::join!(chat_json(&pair, WITH), chat_json(&pair, ""));
    assert_eq!(
        counter.peak_in_flight(),
        2,
        "the pair did not decode as one batch"
    );
    let content = checked_entries(&reply, "batched");
    assert_eq!(
        reply["choices"][0]["message"]["content"], without["choices"][0]["message"]["content"],
        "batched: asking for logprobs changed the reply"
    );
    assert_eq!(
        reply["usage"]["completion_tokens"], without["usage"]["completion_tokens"],
        "batched"
    );
    let choice = without["choices"][0].as_object().expect("choice object");
    assert!(!choice.contains_key("logprobs"), "batched: {without}");

    let (pair, counter) = batch_of_two(state);
    let ((streamed, streamed_tokens), again) =
        tokio::join!(streamed_entries(&pair, WITH), chat_json(&pair, WITH));
    assert_eq!(
        counter.peak_in_flight(),
        2,
        "the streamed pair did not decode as one batch"
    );
    assert_eq!(
        streamed.len() as u64,
        streamed_tokens,
        "batched: {streamed:?}"
    );
    let again = checked_entries(&again, "batched again");
    same_entries(&streamed, again, "batched streamed");
    same_entries(again, content, "batched again");
}

/// One request to the iteration scheduler, and the ends its tokens and (when
/// it asks) its records arrive on.
struct Turn {
    max_tokens: usize,
    tokens: tokio::sync::mpsc::Receiver<Result<u32, String>>,
    records: Option<tokio::sync::mpsc::UnboundedReceiver<crate::gguf::logprobs::StepLogprobs>>,
}

fn turn(
    prompt_ids: &[u32],
    max_tokens: usize,
    asks: bool,
) -> (crate::api::cuda_batch_scheduler::CudaBatchRequest, Turn) {
    use crate::api::cuda_batch_scheduler::{CudaBatchRequest, RecordSink};
    let (token_tx, tokens) = tokio::sync::mpsc::channel(64);
    let (logprobs, records) = if asks {
        let (records, rx) = tokio::sync::mpsc::unbounded_channel();
        (Some(RecordSink { top_n: 3, records }), Some(rx))
    } else {
        (None, None)
    };
    let request = CudaBatchRequest {
        prompt_ids: prompt_ids.to_vec(),
        // temperature 0 and no stop tokens: each turn runs to its max_tokens
        config: crate::gguf::QuantizedGenerateConfig::deterministic(max_tokens),
        token_tx,
        non_streaming: false,
        enqueue_time: std::time::Instant::now(),
        timing_tx: None,
        logprobs,
    };
    let turn = Turn {
        max_tokens,
        tokens,
        records,
    };
    (request, turn)
}

/// The turn ran to the end, and if it asked, one record per token: steps from
/// 0, each naming its token, the best 3 first, every logprob <= 0, and at
/// temperature 0 the chosen one as likely as the best.
fn turn_is_served(mut turn: Turn, path: &str) {
    let mut tokens = Vec::new();
    while let Ok(token) = turn.tokens.try_recv() {
        tokens.push(token.unwrap_or_else(|e| panic!("{path}: {e}")));
    }
    assert_eq!(tokens.len(), turn.max_tokens, "{path}: tokens {tokens:?}");
    let Some(mut rx) = turn.records else {
        return;
    };
    let mut records = Vec::new();
    while let Ok(record) = rx.try_recv() {
        records.push(record);
    }
    assert_eq!(records.len(), tokens.len(), "{path}: one record per token");
    for (i, (record, &token)) in records.iter().zip(&tokens).enumerate() {
        assert_eq!(record.step, i, "{path}");
        assert_eq!(record.chosen, token, "{path}: step {i}");
        assert_eq!(record.top.len(), 3, "{path}: step {i}");
        assert!(
            record.top.windows(2).all(|w| w[0].logprob >= w[1].logprob),
            "{path}: step {i} not best first"
        );
        assert!(
            record.chosen_logprob <= 0.0 && record.top.iter().all(|t| t.logprob <= 0.0),
            "{path}: step {i}"
        );
        assert!(
            (record.chosen_logprob - record.top[0].logprob).abs() < 1e-4,
            "{path}: step {i} chose below the best at temperature 0"
        );
    }
}

/// The iteration scheduler's batched path, driven directly so the path is not
/// a race: two prompts set up together (one asking, one not), a third that
/// joins a free slot (or recycles one, if only two fit) and a fourth that
/// recycles the slot of whichever finishes first. Each that asked gets its own
/// records, and the one that did not gets its tokens.
fn iteration_scheduler_records(state: &AppState) {
    use std::collections::VecDeque;
    let model = state.cuda_model().expect("a CUDA state");
    // "The capital of France is", "Hello,", "What is", "Once upon a time"
    let (first, one) = turn(&[1, 450, 7483, 310, 3444, 338], 6, true);
    let (second, two) = turn(&[1, 15043, 29892], 10, false);
    let (third, three) = turn(&[1, 1724, 338], 8, true);
    let (fourth, four) = turn(&[1, 9038, 2501, 263, 931], 8, true);
    let (_open, mut rx) = tokio::sync::mpsc::channel(4);
    let mut waiting = VecDeque::from([third, fourth]);
    let mut iterations = 0;
    crate::api::iteration_scheduler::process_iteration_batch(
        model,
        vec![first, second],
        &mut rx,
        &mut waiting,
        3,
        &mut iterations,
    );
    assert!(waiting.is_empty(), "iteration: {} never ran", waiting.len());
    assert!(iterations > 1, "iteration: the batched path did not run");
    turn_is_served(one, "iteration set up");
    turn_is_served(two, "iteration set up, not asking");
    turn_is_served(three, "iteration joined");
    turn_is_served(four, "iteration recycled");
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
    batched_logprobs_are_served(&direct).await;
    iteration_scheduler_records(&direct);
}
