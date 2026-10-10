//! PMAT-044: CUDA batch scheduler for continuous batching on `/v1/chat/completions`
//!
//! Accumulates streaming requests and processes them in batches using
//! `generate_batched_streaming` for weight sharing across concurrent requests.

#[cfg(feature = "cuda")]
use crate::gguf::{OwnedQuantizedModelCuda, QuantizedGenerateConfig};
use std::sync::Arc;

/// Request submitted to the batch scheduler
#[cfg(feature = "cuda")]
pub struct CudaBatchRequest {
    /// Tokenized prompt IDs
    pub prompt_ids: Vec<u32>,
    /// Generation configuration (max tokens, temperature, stop tokens)
    pub config: QuantizedGenerateConfig,
    /// Channel to stream generated token IDs back to the HTTP handler
    pub token_tx: tokio::sync::mpsc::Sender<Result<u32, String>>,
    /// realizr#212: When true, scheduler accumulates tokens internally (Vec::push)
    /// and bulk-sends after generation — eliminates per-token channel overhead.
    pub non_streaming: bool,
    /// PMAT-086: Timestamp when request was enqueued (for queue latency measurement)
    pub enqueue_time: std::time::Instant,
    /// PP-LLAMA-001 §3: return path for the SERVER-measured phase split.
    ///
    /// Filled on the single-request fast path, which is the c=1 regime §7.2
    /// gates `prefill_ratio` in. On the batched path it is DROPPED, and the
    /// handler therefore reports no `timings`: a batch's prefill is shared
    /// across m prompts and its decode is interleaved, so there is no per-request
    /// phase split to report and inventing one would put a fabricated numerator
    /// into a gated ratio.
    pub timing_tx: Option<tokio::sync::oneshot::Sender<crate::api::PhaseTimings>>,
    /// #4971: `Some` when the request asked for logprobs. The single-request
    /// path sends each token's record here ahead of the token. The batched
    /// path records none, so the handler meets a token without its record and
    /// fails the request: never a reply without what it asked for.
    pub logprobs: Option<RecordSink>,
}

/// #4971: where a request that asked for logprobs gets each token's record,
/// sent ahead of the token, with the `top_n` most likely tokens of its step.
#[cfg(feature = "cuda")]
pub struct RecordSink {
    /// How many of each step's best tokens a record carries.
    pub top_n: usize,
    /// The handler's end, a [`StreamLogprobs`](crate::api::chat_logprobs::StreamLogprobs).
    pub records: tokio::sync::mpsc::UnboundedSender<crate::gguf::logprobs::StepLogprobs>,
}

/// One dense turn of `session`, as [`dense_stream`](crate::gguf::dense_session::dense_stream),
/// and with `logprobs` (#4971) each token's record sent ahead of the token. A
/// record that cannot be sent means the handler is gone, which `on_token`
/// sees too.
#[cfg(feature = "cuda")]
pub(crate) fn dense_stream_recording<F: crate::session::ArchForward>(
    session: &mut crate::session::Session<F>,
    prompt: &[u32],
    config: &QuantizedGenerateConfig,
    logprobs: Option<&RecordSink>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> crate::error::Result<(Vec<u32>, bool)> {
    use crate::gguf::dense_session::{dense_stream, dense_stream_with_logprobs};
    let Some(sink) = logprobs else {
        return dense_stream(session, prompt, config, on_token);
    };
    dense_stream_with_logprobs(session, prompt, config, sink.top_n, &mut |token, record| {
        let _ = sink.records.send(record);
        on_token(token)
    })
}

/// PMAT-044: Batch scheduler configuration
#[cfg(feature = "cuda")]
pub struct CudaBatchConfig {
    /// Maximum batch size (default 4)
    pub max_batch: usize,
    /// Window timeout in ms — how long to wait for batch to fill (default 10ms)
    pub window_ms: u64,
}

#[cfg(feature = "cuda")]
impl CudaBatchConfig {
    /// PP-13/PP-24: this scheduler's identity and admission ceiling, for
    /// `/v1/effective-config`.
    ///
    /// The identity and these numbers used to be printed to stdout and then
    /// MOVED into the spawned task, so after startup nothing could say which of
    /// the two schedulers was running or how many requests it would admit —
    /// which is exactly what PP-24 derives the concurrency ladder from.
    #[must_use]
    pub fn report(&self, admission_ceiling_reason: &'static str) -> crate::api::SchedulerReport {
        crate::api::SchedulerReport {
            kind: "cuda_batch",
            max_in_flight: self.max_batch,
            window_ms: self.window_ms,
            // This scheduler prefills whole prompts; it has no chunk size and
            // no per-step token budget, and says so rather than reporting 0.
            prefill_chunk_size: None,
            token_budget: None,
            slots_admitted: self.max_batch,
            admission_ceiling_reason,
            in_flight_now: None,
            peak_in_flight: None,
        }
    }
}

#[cfg(feature = "cuda")]
impl Default for CudaBatchConfig {
    fn default() -> Self {
        let max_batch = std::env::var("CUDA_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        // PMAT-068: Default 0ms window — zero-latency c=1, requests batch naturally
        // at c>1 from queue contention. Saves ~1ms TTFT at c=1.
        // Override with CUDA_BATCH_WINDOW_MS=10 for throughput-optimized batching.
        let window_ms = std::env::var("CUDA_BATCH_WINDOW_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        Self {
            max_batch,
            window_ms,
        }
    }
}

/// realizr#212: Generate tokens for a single request, using bulk-send for non-streaming.
/// Shared by both batch scheduler (PMAT-044) and iteration scheduler (PMAT-088).
#[cfg(feature = "cuda")]
pub fn generate_single_request(cuda_model: &mut OwnedQuantizedModelCuda, req: CudaBatchRequest) {
    let timing_tx = req.timing_tx;
    let req = CudaBatchRequest {
        timing_tx: None,
        ..req
    };
    let generate_start = std::time::Instant::now();
    generate_single_request_inner(cuda_model, req);
    // §3: the engine timed prefill; decode is the remainder of the call this
    // function made. Both phases are measured — `PhaseTimings::to_timings`
    // refuses to build a wire block unless they are.
    if let Some(tx) = timing_tx {
        let mut phases = cuda_model.take_phase_timings();
        if let Some(prefill_ms) = phases.prefill_ms {
            let total_ms = generate_start.elapsed().as_secs_f64() * 1000.0;
            phases.decode_ms = Some((total_ms - prefill_ms).max(0.0));
        }
        let _ = tx.send(phases);
    }
}

/// #4280: the single-request path is one turn of the one engine,
/// [`Session`](crate::session::Session) over the dense CUDA forward, borrowing
/// the scheduler's model for the turn. The session owns the loop — prefill,
/// token choice, repetition penalty, stop tokens — and records the turn in the
/// engine witness, which is how `tests_engine_identity` sees this entry.
#[cfg(feature = "cuda")]
fn generate_single_request_inner(cuda_model: &mut OwnedQuantizedModelCuda, req: CudaBatchRequest) {
    // §3 / PP-2: phase timings belong to THIS request, never the last one's.
    let _ = cuda_model.take_phase_timings();
    if req.prompt_ids.is_empty() {
        return;
    }
    let mut session = crate::session::Session::new(
        crate::gguf::dense_session_borrowed::BorrowedCudaForward::new(cuda_model),
    );
    // `dense_stream` never hands on the stop token that ends the turn, as the
    // pre-port loop checked it before emitting; with logprobs (#4971) each
    // token's record goes to the handler ahead of the token.
    let (prompt, config, logprobs) = (&req.prompt_ids, &req.config, req.logprobs.as_ref());
    if req.non_streaming {
        let mut tokens = Vec::new();
        let result = dense_stream_recording(&mut session, prompt, config, logprobs, &mut |tid| {
            tokens.push(tid);
            true
        });
        match result {
            Ok(_) => {
                for t in tokens {
                    if req.token_tx.try_send(Ok(t)).is_err() {
                        break;
                    }
                }
            },
            Err(e) => {
                let _ = req.token_tx.try_send(Err(e.to_string()));
            },
        }
    } else {
        let result = dense_stream_recording(&mut session, prompt, config, logprobs, &mut |tid| {
            req.token_tx.try_send(Ok(tid)).is_ok()
        });
        if let Err(e) = result {
            let _ = req.token_tx.try_send(Err(e.to_string()));
        }
    }
}

/// Spawn the batch scheduler background task.
///
/// Returns a sender for submitting requests.
#[cfg(feature = "cuda")]
pub fn spawn_cuda_batch_scheduler(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: CudaBatchConfig,
    in_flight: Arc<crate::api::InFlightCounter>,
) -> tokio::sync::mpsc::Sender<CudaBatchRequest> {
    let (tx, rx) = tokio::sync::mpsc::channel::<CudaBatchRequest>(256);

    // Run the scheduler in a blocking thread (CUDA ops are synchronous)
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("PMAT-044: failed to create scheduler runtime");

        rt.block_on(async move {
            cuda_batch_scheduler_loop(model, config, rx, in_flight).await;
        });
    });

    tx
}

/// PP-24: holds the in-flight count for one batch and releases it on EVERY
/// exit, including the four early returns `process_cuda_batch` takes on error.
///
/// A hand-placed decrement would have to be repeated at each of them, and the
/// one that gets forgotten leaves `in_flight_now` permanently above zero — a
/// counter that only ever rises is worse than no counter, because a reader
/// cannot tell it from a busy server.
///
/// The count MIRRORS `BatchState::m`, the scheduler's own number of live
/// slots, synced once per decode step. Counting requests as they were pulled
/// in was measured wrong twice over (cross-vendor review of PP-24): a staggered
/// prompt was counted in the batch it arrived with AND again when it joined,
/// and a request taking a recycled slot was never counted at all.
#[cfg(feature = "cuda")]
struct BatchInFlight<'a> {
    counter: &'a crate::api::InFlightCounter,
}

#[cfg(feature = "cuda")]
impl BatchInFlight<'_> {
    /// Publish the scheduler's live slot count as the in-flight figure.
    fn sync(&self, live_slots: usize) {
        self.counter.set(live_slots);
    }
}

#[cfg(feature = "cuda")]
impl Drop for BatchInFlight<'_> {
    fn drop(&mut self) {
        // The batch ended on this path, whichever path it was.
        self.counter.set(0);
    }
}

#[cfg(feature = "cuda")]
async fn cuda_batch_scheduler_loop(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: CudaBatchConfig,
    mut rx: tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    in_flight: Arc<crate::api::InFlightCounter>,
) {
    eprintln!(
        "[PMAT-044] Batch scheduler started: max_batch={}, window={}ms",
        config.max_batch, config.window_ms
    );

    // PMAT-097: Track recent batch sizes to detect concurrent traffic.
    // When we've recently seen batches > 1, even singleton batches should
    // wait briefly for peers to avoid starting m=1 batches that block the
    // channel for ~1.7s (256 tokens × 6.7ms). This fixes TTFT P99.9
    // tail latency at c=4 (1889ms → ~50ms target).
    let mut recent_batch_gt1 = false;

    loop {
        // Wait for at least one request
        let first = match rx.recv().await {
            Some(req) => req,
            None => {
                eprintln!("[PMAT-044] Batch scheduler shutting down (channel closed)");
                return;
            },
        };

        // Accumulate more requests within the window.
        // PMAT-095: Adaptive batch window — zero overhead at c=1, auto-batching at c>1.
        // Phase 1: Non-blocking drain (captures requests queued during GPU processing).
        // Phase 2: If drain found peers AND batch not full, short timed wait for stragglers.
        // This eliminates the c=1 TTFT penalty of fixed batch windows while giving
        // consistent M=max batches at c>1.
        let batch = if config.window_ms == 0 {
            drain_adaptive(first, &mut rx, config.max_batch, recent_batch_gt1).await
        } else {
            match accumulate_window(first, &mut rx, &config).await {
                Accumulated::Ready(batch) => batch,
                Accumulated::Closed(batch) => {
                    process_cuda_batch(&model, batch, &mut rx, config.max_batch, &in_flight);
                    return;
                },
            }
        };

        let batch_size = batch.len();
        let batch_start = std::time::Instant::now();

        // PMAT-097: Update concurrency hint for next batch's adaptive wait.
        recent_batch_gt1 = batch_size > 1;

        // Process the batch (PMAT-073: pass rx for mid-batch joins)
        process_cuda_batch(&model, batch, &mut rx, config.max_batch, &in_flight);

        log_batch_done(batch_size, batch_start.elapsed());
    }
}

/// PMAT-044: how a timed accumulation window ended.
#[cfg(feature = "cuda")]
enum Accumulated {
    /// The window expired or the batch filled; the scheduler keeps going.
    Ready(Vec<CudaBatchRequest>),
    /// The channel closed mid-window: run this batch, then stop.
    Closed(Vec<CudaBatchRequest>),
}

/// PMAT-086/095/097: the zero-window batch — a cooperative yield, a
/// non-blocking drain, then a short adaptive wait for stragglers.
#[cfg(feature = "cuda")]
async fn drain_adaptive(
    first: CudaBatchRequest,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    max_batch: usize,
    recent_batch_gt1: bool,
) -> Vec<CudaBatchRequest> {
    let mut batch = vec![first];
    // PMAT-086/095: Zero-latency drain with cooperative yield + adaptive wait.
    tokio::task::yield_now().await;
    while batch.len() < max_batch {
        match rx.try_recv() {
            Ok(req) => batch.push(req),
            Err(_) => break,
        }
    }
    // PMAT-095/097: Adaptive wait for batch formation.
    // Phase 2a: If we found peers, wait 3ms for stragglers.
    // Phase 2b (PMAT-097): If we're singleton but recently saw concurrent traffic,
    // wait 2ms for peers. Fixes c=4 TTFT P99.9 tail (m=1 batches block 1.7s).
    // At true c=1, recent_batch_gt1 stays false → no wait → zero overhead.
    let should_wait = if batch.len() > 1 && batch.len() < max_batch {
        true // Phase 2a: found peers, wait for more
    } else if batch.len() == 1 && recent_batch_gt1 && max_batch > 1 {
        true // Phase 2b: singleton but concurrent traffic detected
    } else {
        false
    };
    if should_wait {
        let wait_ms = if batch.len() > 1 { 3 } else { 2 };
        wait_for_stragglers(&mut batch, rx, max_batch, wait_ms).await;
    }
    batch
}

/// PMAT-095: up to `max_batch`, whatever reaches `rx` within `wait_ms`; a
/// closed channel ends the wait early.
#[cfg(feature = "cuda")]
async fn wait_for_stragglers(
    batch: &mut Vec<CudaBatchRequest>,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    max_batch: usize,
    wait_ms: u64,
) {
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_millis(wait_ms);
    while batch.len() < max_batch {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(req)) => batch.push(req),
            Ok(None) => break,
            Err(_timeout) => break,
        }
    }
}

/// PMAT-044: a fixed window of `config.window_ms` for peers to join `first`.
#[cfg(feature = "cuda")]
async fn accumulate_window(
    first: CudaBatchRequest,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    config: &CudaBatchConfig,
) -> Accumulated {
    let mut batch = vec![first];
    let deadline =
        tokio::time::Instant::now() + tokio::time::Duration::from_millis(config.window_ms);

    while batch.len() < config.max_batch {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(req)) => batch.push(req),
            Ok(None) => {
                eprintln!("[PMAT-044] Channel closed during accumulation");
                return Accumulated::Closed(batch);
            },
            Err(_timeout) => break, // Window expired
        }
    }
    Accumulated::Ready(batch)
}

/// PMAT-044: one line per batch, its wall time and per-slot rate.
#[cfg(feature = "cuda")]
fn log_batch_done(batch_size: usize, elapsed: std::time::Duration) {
    eprintln!(
        "[PMAT-044] Batch m={} done in {:.1}ms ({:.1} tok/s/slot)",
        batch_size,
        elapsed.as_secs_f64() * 1000.0,
        if elapsed.as_secs_f64() > 0.0 {
            1000.0 / elapsed.as_secs_f64() / batch_size as f64
        } else {
            0.0
        }
    );
}

/// The channel one request's tokens (or its error) go back on.
#[cfg(feature = "cuda")]
type TokenSender = tokio::sync::mpsc::Sender<Result<u32, String>>;

/// A slot's per-token callback; `false` means its caller has gone.
#[cfg(feature = "cuda")]
type TokenCallback = Box<dyn FnMut(u32) -> bool + Send>;

/// The callback that streams a slot's tokens to `token_tx`.
#[cfg(feature = "cuda")]
fn token_callback(token_tx: TokenSender) -> TokenCallback {
    Box::new(move |token_id: u32| -> bool { token_tx.try_send(Ok(token_id)).is_ok() })
}

#[cfg(feature = "cuda")]
fn process_cuda_batch(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    batch: Vec<CudaBatchRequest>,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    max_batch: usize,
    counter: &crate::api::InFlightCounter,
) {
    let m = batch.len();
    // PP-24: what this server ACTUALLY ran concurrently, as opposed to the
    // ceiling it advertises. The ladder is derived from the ceiling (known
    // before the band) and checked against this peak (known after it).
    let in_flight = BatchInFlight { counter };

    // PERF-041: `channel_empty` is passed as `true` because THIS scheduler does
    // not consult `rx` before taking the fast path — that omission is F-BATCH-003
    // ("a request arriving mid-generation queues", measured RED 2026-08-26 at a
    // 1.59 latency ratio against a 1.60 QUEUED prediction). Wiring `rx.is_empty()`
    // here is NOT the fix and is deliberately not done: at m == 1 the batched path
    // costs ~3x per token, so a joining peer would slow the in-flight request
    // instead of speeding itself. batch-admission-v1 known_limitations says the
    // remedy is closing that per-token gap, which is engine work with its own
    // measurement. What routing through the predicate buys today is the
    // F-BATCH-004 knob below, which makes the gap measurable.
    if crate::api::batch_admission::fast_path_eligible(
        m,
        true,
        crate::api::batch_admission::force_batched_path(),
    ) {
        // Single request — use the optimized single-request path (138 tok/s vs 46 batched).
        // PMAT-073: Mid-batch joins only work for initial batches >1. For c=1→c=2
        // staggered arrivals, the second request queues until the first completes.
        // This is acceptable because the fast path's 3x better ITL outweighs the
        // latency benefit of mid-batch join for the second request.
        let req = batch.into_iter().next().unwrap();
        run_fast_path(model, req);
        return;
    }

    // PMAT-099: Staggered prefill — prefill first prompt only, join rest during decode.
    // FALSIFIED for short prompts: per-slot join overhead (14ms×3) exceeds batched prefill (20ms).
    // c=4 short prompts: staggered 241 vs batched 260 aggregate (-7.3%), TTFT 87 vs 40ms (+118%).
    // May win for long prompts (>500 tokens) — not yet tested.
    // Default OFF. Enable: STAGGERED_PREFILL=1.
    let staggered = std::env::var("STAGGERED_PREFILL").as_deref() == Ok("1") && m > 1;

    // Build Phase 1 inputs: first request only (staggered) or all requests (batched)
    let Phase1 {
        prompts,
        configs,
        mut error_senders,
        callbacks,
        mut pending_joins,
    } = phase1_inputs(batch, staggered);

    // Phase 1: Setup + Prefill (under lock)
    // Staggered: prefills 1 prompt, pre-allocates max_batch KV slots.
    // Non-staggered: prefills all M prompts (original behavior).
    let Some(mut state) = setup_and_prefill(
        model,
        &prompts,
        &configs,
        callbacks,
        max_batch,
        m,
        &error_senders,
        &pending_joins,
    ) else {
        return;
    };

    // Phase 2: Decode loop with mid-batch joins (PMAT-073/099) and slot recycling (PMAT-074)
    // Lock per step (~19ms per acquire vs ~660ms total).
    // PMAT-099: Pending staggered joins are processed one-per-step for progressive ramp-up.
    // PP-24: the initial batch is live from here, joins or not.
    in_flight.sync(state.m);
    while !state.all_done() && state.gen_idx < state.max_tokens_max {
        let Some(token_ids) = decode_step(
            model,
            &mut state,
            rx,
            &mut pending_joins,
            &mut error_senders,
            &in_flight,
        ) else {
            return;
        };

        // Token distribution runs WITHOUT model lock — SSE callbacks only
        state.distribute_tokens(&token_ids);
    }

    // Phase 3: Cleanup (under lock)
    {
        let mut cuda_model = model.write().expect("PMAT-072: model lock poisoned");
        cuda_model.batched_cleanup(&state);
    }
}

/// PMAT-044: the m=1 fast path, holding the model lock for the whole turn.
#[cfg(feature = "cuda")]
fn run_fast_path(model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>, req: CudaBatchRequest) {
    ttft_trace("queue_latency", &req);
    let mut cuda_model = model.write().expect("PMAT-044: model lock poisoned");
    ttft_trace("lock_acquired", &req);
    generate_single_request(&mut cuda_model, req);
}

/// TTFT_TRACE: how long `req` has waited by `stage`.
#[cfg(feature = "cuda")]
fn ttft_trace(stage: &str, req: &CudaBatchRequest) {
    if std::env::var("TTFT_TRACE").is_ok() {
        eprintln!(
            "[TTFT] {:>20}: {:>7.2}ms",
            stage,
            req.enqueue_time.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// PMAT-072/099: what Phase 1 prefills, who hears about its tokens and
/// errors, and (staggered) who joins later.
#[cfg(feature = "cuda")]
struct Phase1 {
    prompts: Vec<Vec<u32>>,
    configs: Vec<QuantizedGenerateConfig>,
    error_senders: Vec<TokenSender>,
    callbacks: Vec<TokenCallback>,
    pending_joins: std::collections::VecDeque<CudaBatchRequest>,
}

#[cfg(feature = "cuda")]
fn phase1_inputs(batch: Vec<CudaBatchRequest>, staggered: bool) -> Phase1 {
    if staggered {
        // Split batch: first → immediate prefill, rest → pending joins
        let mut batch_iter = batch.into_iter();
        let first_req = batch_iter.next().unwrap();
        let pending_joins: std::collections::VecDeque<CudaBatchRequest> = batch_iter.collect();

        eprintln!(
            "[PMAT-099] Staggered prefill: 1 immediate + {} pending joins",
            pending_joins.len()
        );

        Phase1 {
            prompts: vec![first_req.prompt_ids.clone()],
            configs: vec![first_req.config.clone()],
            error_senders: vec![first_req.token_tx.clone()],
            callbacks: vec![token_callback(first_req.token_tx)],
            pending_joins,
        }
    } else {
        // All prompts prefilled together in Phase 1 (original PMAT-072 behavior)
        Phase1 {
            prompts: batch.iter().map(|r| r.prompt_ids.clone()).collect(),
            configs: batch.iter().map(|r| r.config.clone()).collect(),
            error_senders: batch.iter().map(|r| r.token_tx.clone()).collect(),
            callbacks: batch
                .into_iter()
                .map(|req| token_callback(req.token_tx))
                .collect(),
            pending_joins: std::collections::VecDeque::new(),
        }
    }
}

/// Phase 1 under the model lock. `None` when it failed: every caller has
/// been told, and the batched state reset.
#[cfg(feature = "cuda")]
fn setup_and_prefill(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    prompts: &[Vec<u32>],
    configs: &[QuantizedGenerateConfig],
    callbacks: Vec<TokenCallback>,
    max_batch: usize,
    m: usize,
    error_senders: &[TokenSender],
    pending_joins: &std::collections::VecDeque<CudaBatchRequest>,
) -> Option<crate::gguf::BatchedDecodeState> {
    let mut cuda_model = model.write().expect("PMAT-072: model lock poisoned");
    match cuda_model.batched_setup_and_prefill(prompts, configs, callbacks, max_batch) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("[PMAT-072] Setup+prefill ERROR (m={m}): {e}");
            // Every slot, and the pending joins too
            fail_all(error_senders, pending_joins, &e.to_string());
            // PMAT-765: reset stale batched state (batched_kv_stride) before returning, so
            // the NEXT batch doesn't reuse buffers with a stale stride → KV corruption.
            cuda_model.reset_batched_state();
            None
        },
    }
}

/// Send `error` to every slot's caller and every pending join's.
#[cfg(feature = "cuda")]
fn fail_all(
    error_senders: &[TokenSender],
    pending_joins: &std::collections::VecDeque<CudaBatchRequest>,
    error: &str,
) {
    for tx in error_senders {
        let _ = tx.try_send(Err(error.to_string()));
    }
    for req in pending_joins {
        let _ = req.token_tx.try_send(Err(error.to_string()));
    }
}

/// One Phase 2 step under the model lock: joins (one pending staggered
/// slot, then waiting requests), slot recycling, then the decode itself.
/// `None` when the step failed: every caller has been told, and the batch
/// cleaned up.
#[cfg(feature = "cuda")]
fn decode_step(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    state: &mut crate::gguf::BatchedDecodeState,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    pending_joins: &mut std::collections::VecDeque<CudaBatchRequest>,
    error_senders: &mut Vec<TokenSender>,
    in_flight: &BatchInFlight<'_>,
) -> Option<Vec<u32>> {
    let mut cuda_model = model.write().expect("PMAT-072: model lock poisoned");

    // PMAT-099: Join one pending staggered slot per step (progressive ramp-up).
    // This limits decode stall to one prefill per step instead of blocking all at once.
    if !pending_joins.is_empty() && state.m < state.max_kv_slots {
        let req = pending_joins.pop_front().unwrap();
        join_slot(
            &mut cuda_model,
            state,
            req,
            error_senders,
            "[PMAT-099] Staggered join FAILED",
        );
    }

    // PMAT-073: Check for pending requests to join mid-batch (fill empty slots).
    while state.m < state.max_kv_slots {
        match rx.try_recv() {
            Ok(req) => join_slot(
                &mut cuda_model,
                state,
                req,
                error_senders,
                "[PMAT-073] Mid-batch join FAILED",
            ),
            Err(_) => break, // No pending requests
        }
    }

    // PP-24: publish what this batch actually holds, after every join
    // and recycle path above has run and before the step decodes it.
    in_flight.sync(state.m);

    recycle_done_slots(&mut cuda_model, state, rx, pending_joins, error_senders);

    match cuda_model.batched_decode_step(state) {
        Ok(ids) => Some(ids),
        Err(e) => {
            eprintln!(
                "[PMAT-074] Decode step ERROR (m={}, step={}): {e}",
                state.m, state.gen_idx
            );
            // Every slot, and any remaining pending joins
            fail_all(error_senders, pending_joins, &e.to_string());
            // Still need cleanup under lock
            cuda_model.batched_cleanup(state);
            None
        },
    }
}

/// PMAT-073/099: a new slot in the running batch for `req`; on failure its
/// caller hears why, and `failed` heads the log line.
#[cfg(feature = "cuda")]
fn join_slot(
    cuda_model: &mut OwnedQuantizedModelCuda,
    state: &mut crate::gguf::BatchedDecodeState,
    req: CudaBatchRequest,
    error_senders: &mut Vec<TokenSender>,
    failed: &str,
) {
    let error_tx = req.token_tx.clone();
    let on_token = token_callback(req.token_tx);
    match cuda_model.add_slot_to_batch(state, req.prompt_ids, req.config, on_token) {
        Ok(()) => {
            error_senders.push(error_tx);
        },
        Err(e) => {
            eprintln!("{failed}: {e}");
            let _ = error_tx.try_send(Err(e.to_string()));
        },
    }
}

/// PMAT-074: Slot recycling — reuse finished slots for pending requests.
/// Staggered pending joins go first (they arrived with the initial batch),
/// then the channel.
#[cfg(feature = "cuda")]
fn recycle_done_slots(
    cuda_model: &mut OwnedQuantizedModelCuda,
    state: &mut crate::gguf::BatchedDecodeState,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    pending_joins: &mut std::collections::VecDeque<CudaBatchRequest>,
    error_senders: &mut [TokenSender],
) {
    for slot_idx in 0..state.m {
        if !state.done[slot_idx] {
            continue;
        }
        let Some(req) = pending_joins.pop_front().or_else(|| rx.try_recv().ok()) else {
            break; // No pending requests
        };
        let error_tx = req.token_tx.clone();
        let on_token = token_callback(req.token_tx);
        match cuda_model.recycle_slot(state, slot_idx, req.prompt_ids, req.config, on_token) {
            Ok(()) => {
                error_senders[slot_idx] = error_tx;
            },
            Err(e) => {
                eprintln!("[PMAT-074] Slot recycle FAILED (slot {slot_idx}): {e}");
                let _ = error_tx.try_send(Err(e.to_string()));
            },
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "cuda_batch_scheduler_tests.rs"]
mod tests;
