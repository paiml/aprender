//! #5293: the Qwen3.5 batch engine behind the real scheduler, on the real
//! Qwen3.5-0.8B file.

use super::*;
use crate::api::cuda_batch_scheduler::spawn_cuda_batch_scheduler;

/// The real hybrid file, as in the session and per-layer parity tests.
const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Greedy tokens each request asks for.
const STEPS: usize = 8;

/// Four ragged prompts (1, 4, 7 and 2 tokens), all distinct, so a slot that
/// read another slot's state would decode a different stream.
const PROMPTS: [&[u32]; 4] = [
    &[9707],
    &[785, 6722, 315, 9625],
    &[22043, 374, 264, 3974, 11, 1879, 13],
    &[16, 10],
];

/// Send `prompts` to the scheduler at once and wait for every reply.
fn band(tx: &tokio::sync::mpsc::Sender<CudaBatchRequest>, prompts: &[&[u32]]) -> Vec<Vec<u32>> {
    let receivers: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (token_tx, rx) = tokio::sync::mpsc::channel(64);
            let config = QuantizedGenerateConfig {
                max_tokens: STEPS,
                temperature: 0.0,
                top_k: 1,
                stop_tokens: Vec::new(),
                ..QuantizedGenerateConfig::default()
            };
            tx.try_send(CudaBatchRequest {
                prompt_ids: p.to_vec(),
                config,
                token_tx,
                non_streaming: false,
                enqueue_time: std::time::Instant::now(),
                timing_tx: None,
                logprobs: None,
            })
            .unwrap_or_else(|e| panic!("queue {p:?}: {e}"));
            rx
        })
        .collect();
    receivers
        .into_iter()
        .map(|mut rx| {
            let mut got = Vec::new();
            while let Some(t) = rx.blocking_recv() {
                got.push(t.expect("no error"));
            }
            got
        })
        .collect()
}

/// FALSIFY (#5293): four concurrent greedy requests form one batch of m = 4
/// on the GPU, and each gets exactly the tokens it got when it was served
/// alone (one turn of the session).
#[test]
#[serial_test::serial]
fn four_concurrent_requests_batch_and_match_their_solo_turns() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {MODEL_PATH} is absent");
        return;
    }
    if crate::cuda::CudaExecutor::new(0).is_err() {
        eprintln!("SKIP: no CUDA device");
        return;
    }
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let session = Qwen35Session::load_cached(std::path::Path::new(MODEL_PATH), &mapped, false)
        .expect("load the session");
    assert!(
        session.on_gpu(),
        "the session must hold the GPU for this test"
    );
    let served = Arc::new(Qwen35Served {
        context_length: session.context_length(),
        session: std::sync::Mutex::new(session),
        on_gpu: std::sync::atomic::AtomicBool::new(true),
    });
    let engine = Arc::new(std::sync::RwLock::new(Qwen35BatchEngine::new(served)));
    let counter = crate::api::InFlightCounter::new();
    let config = CudaBatchConfig {
        max_batch: 4,
        window_ms: 200,
    };
    let tx = spawn_cuda_batch_scheduler(engine, config, counter.clone());

    let alone: Vec<Vec<u32>> = PROMPTS.iter().map(|p| band(&tx, &[p])[0].clone()).collect();
    assert!(
        counter.peak_in_flight() <= 1,
        "a lone request is never batched"
    );
    assert!(
        alone.iter().all(|s| s.len() == STEPS),
        "each solo turn decodes its whole budget: {alone:?}"
    );
    assert!(
        alone.windows(2).any(|w| w[0] != w[1]),
        "the solo streams are all equal, so they cannot tell slots apart: {alone:?}"
    );

    let together = band(&tx, &PROMPTS);
    assert_eq!(
        counter.peak_in_flight(),
        4,
        "the band of 4 formed one batch of m = 4"
    );
    for (i, got) in together.iter().enumerate() {
        assert_eq!(got, &alone[i], "request {i} diverged from its solo turn");
    }
    eprintln!("[#5293] m=4: 4 requests x {STEPS} tokens equal their solo turns");
}

/// A greedy slot is one that temperature 0 or `top_k` 1 makes argmax.
#[test]
fn greedy_is_temperature_zero_or_top_k_one() {
    let at = |temperature, top_k| QuantizedGenerateConfig {
        temperature,
        top_k,
        ..QuantizedGenerateConfig::default()
    };
    assert!(is_greedy(&at(0.0, 40)));
    assert!(is_greedy(&at(0.8, 1)));
    assert!(!is_greedy(&at(0.8, 40)));
}
