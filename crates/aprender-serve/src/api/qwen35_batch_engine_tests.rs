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

/// What one request got: its tokens, and with logprobs one record per token.
type Reply = (Vec<u32>, Vec<StepLogprobs>);

/// The greedy config every band but the sampled one asks with.
fn greedy() -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens: STEPS,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: Vec::new(),
        ..QuantizedGenerateConfig::default()
    }
}

/// Send `prompts` to the scheduler at once, each asking for `top_n` logprobs
/// when `Some`, and wait for every reply.
fn band(
    tx: &tokio::sync::mpsc::Sender<CudaBatchRequest>,
    prompts: &[&[u32]],
    top_n: Option<usize>,
) -> Vec<Reply> {
    band_with(tx, prompts, top_n, &greedy())
}

/// [`band`], every request asking with `config`.
fn band_with(
    tx: &tokio::sync::mpsc::Sender<CudaBatchRequest>,
    prompts: &[&[u32]],
    top_n: Option<usize>,
    config: &QuantizedGenerateConfig,
) -> Vec<Reply> {
    let receivers: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (token_tx, rx) = tokio::sync::mpsc::channel(64);
            let (records, records_rx) = tokio::sync::mpsc::unbounded_channel();
            let config = config.clone();
            tx.try_send(CudaBatchRequest {
                prompt_ids: p.to_vec(),
                config,
                token_tx,
                non_streaming: false,
                enqueue_time: std::time::Instant::now(),
                timing_tx: None,
                logprobs: top_n.map(|top_n| RecordSink { top_n, records }),
            })
            .unwrap_or_else(|e| panic!("queue {p:?}: {e}"));
            (rx, records_rx)
        })
        .collect();
    receivers
        .into_iter()
        .map(|(mut rx, mut records_rx)| {
            let mut got = Vec::new();
            while let Some(t) = rx.blocking_recv() {
                got.push(t.expect("no error"));
            }
            let mut steps = Vec::new();
            while let Ok(step) = records_rx.try_recv() {
                steps.push(step);
            }
            (got, steps)
        })
        .collect()
}

/// The scheduler over the real 0.8B file on the GPU, with its in-flight
/// counter.
struct Scheduler {
    tx: Option<tokio::sync::mpsc::Sender<CudaBatchRequest>>,
    counter: Arc<crate::api::InFlightCounter>,
    /// The session, so the test can wait for the scheduler to let it go.
    served: std::sync::Weak<Qwen35Served>,
    /// The engine, for its counters; weak, so it never outlives the scheduler.
    engine: std::sync::Weak<std::sync::RwLock<Qwen35BatchEngine>>,
}

impl Scheduler {
    fn tx(&self) -> &tokio::sync::mpsc::Sender<CudaBatchRequest> {
        self.tx.as_ref().expect("held until drop")
    }

    /// Prompt rows the engine resumed from kept copies so far.
    fn reused(&self) -> u64 {
        let engine = self
            .engine
            .upgrade()
            .expect("the scheduler holds its engine");
        let rows = engine.read().expect("the engine lock").reused_rows();
        rows
    }
}

impl Drop for Scheduler {
    /// Close the queue and wait for the scheduler thread to drop the session.
    /// A test that returned first would race that drop (`cublasDestroy`)
    /// against the process's exit handlers unloading cuBLAS: SIGSEGV after
    /// every test passed (gx10).
    fn drop(&mut self) {
        drop(self.tx.take());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while self.served.strong_count() > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

/// The scheduler, or `None` (the test skips) without the file or a device.
fn scheduler() -> Option<Scheduler> {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {MODEL_PATH} is absent");
        return None;
    }
    if crate::cuda::CudaExecutor::new(0).is_err() {
        eprintln!("SKIP: no CUDA device");
        return None;
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
    let engine = Arc::new(std::sync::RwLock::new(Qwen35BatchEngine::new(Arc::clone(
        &served,
    ))));
    let counter = crate::api::InFlightCounter::new();
    let config = CudaBatchConfig {
        max_batch: 4,
        window_ms: 200,
    };
    Some(Scheduler {
        tx: Some(spawn_cuda_batch_scheduler(
            Arc::clone(&engine),
            config,
            counter.clone(),
        )),
        counter,
        served: Arc::downgrade(&served),
        engine: Arc::downgrade(&engine),
    })
}

/// Each prompt served alone, one band per prompt.
fn solo(tx: &tokio::sync::mpsc::Sender<CudaBatchRequest>, top_n: Option<usize>) -> Vec<Reply> {
    PROMPTS
        .iter()
        .map(|p| band(tx, &[p], top_n).remove(0))
        .collect()
}

/// FALSIFY (#5293): four concurrent greedy requests form one batch of m = 4
/// on the GPU, and each gets exactly the tokens it got when it was served
/// alone (one turn of the session).
#[test]
#[serial_test::serial]
fn four_concurrent_requests_batch_and_match_their_solo_turns() {
    let Some(s) = scheduler() else {
        return;
    };
    let alone: Vec<Vec<u32>> = solo(s.tx(), None).into_iter().map(|r| r.0).collect();
    assert!(
        s.counter.peak_in_flight() <= 1,
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

    let together = band(s.tx(), &PROMPTS, None);
    assert_eq!(
        s.counter.peak_in_flight(),
        4,
        "the band of 4 formed one batch of m = 4"
    );
    for (i, (got, _)) in together.iter().enumerate() {
        assert_eq!(got, &alone[i], "request {i} diverged from its solo turn");
    }
    eprintln!("[#5293] m=4: 4 requests x {STEPS} tokens equal their solo turns");
}

/// FALSIFY (#5279 P7): a band whose prompts extend an earlier band's resumes
/// each slot from the copy the earlier slot kept, so the shared rows (the
/// earlier prompts but their last tokens: 0 + 3 + 6 + 1) are not prefilled
/// again, and every request still gets exactly the tokens its solo turn got.
#[test]
#[serial_test::serial]
fn a_band_extending_an_earlier_one_resumes_its_prefixes_and_matches_solo() {
    let Some(s) = scheduler() else {
        return;
    };
    let longer: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| [p, &[11, 13][..]].concat())
        .collect();
    let longer: Vec<&[u32]> = longer.iter().map(Vec::as_slice).collect();
    let alone: Vec<Vec<u32>> = longer
        .iter()
        .map(|p| band(s.tx(), &[p], None).remove(0).0)
        .collect();
    assert_eq!(s.reused(), 0, "a lone request is the session's turn");

    let _ = band(s.tx(), &PROMPTS, None);
    assert_eq!(s.reused(), 0, "the first band has nothing to resume");
    let together = band(s.tx(), &longer, None);
    assert_eq!(s.counter.peak_in_flight(), 4, "both bands formed m = 4");
    assert_eq!(
        s.reused(),
        3 + 6 + 1,
        "each slot resumed its earlier prompt's head"
    );
    for (i, (got, _)) in together.iter().enumerate() {
        assert_eq!(got, &alone[i], "request {i} diverged from its solo turn");
    }
    eprintln!("[#5279] P7: 10 shared rows resumed, 4 requests equal their solo turns");
}

/// How far a batched slot's logprob may sit from its solo turn's. The solo
/// turn reads step 0 from the session's batched prefill and the batch from the
/// single-token forward of the prompt's last token, so their logits differ in
/// the last bits (2e-4 measured on gx10); the tokens may not differ at all.
const LOGPROB_TOLERANCE: f32 = 1e-2;

/// FALSIFY (#5293/#4971): four concurrent requests that ask for logprobs still
/// form one batch of m = 4, and each gets the tokens it got alone, and records
/// with the same steps, chosen tokens and top 3, logprobs within
/// [`LOGPROB_TOLERANCE`].
#[test]
#[serial_test::serial]
fn logprobs_requests_batch_and_record_what_their_solo_turns_record() {
    let Some(s) = scheduler() else {
        return;
    };
    let alone = solo(s.tx(), Some(3));
    assert!(
        alone
            .iter()
            .all(|(t, r)| t.len() == STEPS && r.len() == STEPS),
        "each solo turn decodes and records its whole budget"
    );

    let together = band(s.tx(), &PROMPTS, Some(3));
    assert_eq!(
        s.counter.peak_in_flight(),
        4,
        "the band of 4 asking for logprobs formed one batch of m = 4"
    );
    let mut drift = 0.0_f32;
    for (i, (tokens, records)) in together.iter().enumerate() {
        assert_eq!(tokens, &alone[i].0, "request {i}'s tokens diverged");
        assert_eq!(
            records.len(),
            alone[i].1.len(),
            "request {i}'s record count"
        );
        for (got, want) in records.iter().zip(&alone[i].1) {
            let ids = |r: &StepLogprobs| r.top.iter().map(|t| t.token_id).collect::<Vec<_>>();
            assert_eq!(
                (got.step, got.chosen, ids(got)),
                (want.step, want.chosen, ids(want)),
                "request {i}'s record diverged"
            );
            let pairs = std::iter::once((got.chosen_logprob, want.chosen_logprob)).chain(
                got.top
                    .iter()
                    .zip(&want.top)
                    .map(|(g, w)| (g.logprob, w.logprob)),
            );
            for (g, w) in pairs {
                drift = drift.max((g - w).abs());
            }
        }
    }
    assert!(
        drift <= LOGPROB_TOLERANCE,
        "a batched logprob sits {drift} from its solo turn's"
    );
    eprintln!("[#5293] m=4 with logprobs: tokens equal their solo turns, logprob drift {drift:e}");
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

/// FALSIFY (#5312): four concurrent requests that sample (temperature 0.8,
/// top 40, one fixed seed) form one batch of m = 4, and each draws exactly
/// the tokens its solo turn draws with that seed; a second seed draws other
/// tokens, so the seed is what decides them.
#[test]
#[serial_test::serial]
fn sampled_requests_batch_and_draw_what_their_seeded_solo_turns_draw() {
    let Some(s) = scheduler() else {
        return;
    };
    let sampled = |seed| QuantizedGenerateConfig {
        temperature: 0.8,
        top_k: 40,
        seed,
        ..greedy()
    };
    let solo_at = |seed| -> Vec<Vec<u32>> {
        PROMPTS
            .iter()
            .map(|p| band_with(s.tx(), &[p], None, &sampled(seed)).remove(0).0)
            .collect()
    };
    let alone = solo_at(7);
    assert!(
        alone.iter().all(|t| t.len() == STEPS),
        "each solo turn draws its whole budget: {alone:?}"
    );
    assert_ne!(alone, solo_at(8), "a second seed draws other tokens");

    let together = band_with(s.tx(), &PROMPTS, None, &sampled(7));
    assert_eq!(
        s.counter.peak_in_flight(),
        4,
        "the sampled band of 4 formed one batch of m = 4"
    );
    for (i, (got, _)) in together.iter().enumerate() {
        assert_eq!(got, &alone[i], "request {i} drew other tokens than alone");
    }
    eprintln!("[#5312] m=4 sampled: 4 requests x {STEPS} tokens equal their seeded solo turns");
}
