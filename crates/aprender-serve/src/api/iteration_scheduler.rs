//! PMAT-088: Iteration-level scheduler for continuous batching.
//!
//! Replaces the batch-then-wait scheduler (cuda_batch_scheduler) with
//! iteration-level scheduling inspired by Orca (Yu et al., OSDI 2022)
//! and Sarathi-Serve (Agrawal et al., OSDI 2024).
//!
//! Key differences from cuda_batch_scheduler:
//! - Decode-maximal: always schedules ALL running decode tokens first
//! - Chunked prefill: splits long prompts across iterations (no generation stalls)
//! - Token budget: caps total tokens per forward pass
//! - Per-iteration scheduling decisions (not per-batch)
//!
//! Enabled via `ITERATION_SCHEDULER=1` env var (opt-in during development).

#[cfg(feature = "cuda")]
use crate::api::cuda_batch_scheduler::{
    token_callback, CudaBatchRequest, TokenCallback, TokenSender,
};
#[cfg(feature = "cuda")]
use crate::gguf::OwnedQuantizedModelCuda;
use renacer_core::PhaseTimer;
use std::collections::VecDeque;
use std::sync::Arc;

/// Iteration scheduler configuration.
#[cfg(feature = "cuda")]
pub struct IterationSchedulerConfig {
    /// Maximum concurrent decode slots (default 4, env CUDA_MAX_BATCH)
    pub max_slots: usize,
    /// Prefill chunk size in tokens — tile-aligned for sm_89 (default 256)
    pub prefill_chunk_size: usize,
    /// Token budget per forward pass (0 = unlimited, env ITERATION_TOKEN_BUDGET)
    pub token_budget: usize,
}

#[cfg(feature = "cuda")]
impl IterationSchedulerConfig {
    /// PP-13/PP-24: this scheduler's identity and admission ceiling, for
    /// `/v1/effective-config`.
    ///
    /// `in_flight_now`/`peak_in_flight` stay `null`: this scheduler is not
    /// instrumented with an [`InFlightCounter`](crate::api::InFlightCounter),
    /// and reporting `0` would be indistinguishable from an idle instrumented
    /// scheduler. It is opt-in (`ITERATION_SCHEDULER=1`) and not the path §5.2's
    /// argv reaches.
    #[must_use]
    pub fn report(&self, admission_ceiling_reason: &'static str) -> crate::api::SchedulerReport {
        crate::api::SchedulerReport {
            kind: "iteration",
            max_in_flight: self.max_slots,
            // PMAT-088: this scheduler forms an initial batch of 1 and joins
            // the rest between decode steps; there is no batch window.
            window_ms: 0,
            prefill_chunk_size: Some(self.prefill_chunk_size),
            // 0 means unlimited in the config; on the wire that is `null`, not
            // a budget of zero tokens.
            token_budget: (self.token_budget > 0).then_some(self.token_budget),
            slots_admitted: self.max_slots,
            admission_ceiling_reason,
            in_flight_now: None,
            peak_in_flight: None,
        }
    }
}

#[cfg(feature = "cuda")]
impl Default for IterationSchedulerConfig {
    fn default() -> Self {
        let max_slots = std::env::var("CUDA_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let prefill_chunk_size = std::env::var("PREFILL_CHUNK_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let token_budget = std::env::var("ITERATION_TOKEN_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0); // 0 = unlimited (for now)
        Self {
            max_slots,
            prefill_chunk_size,
            token_budget,
        }
    }
}

/// Per-iteration scheduling decision.
#[cfg(feature = "cuda")]
struct SchedulerOutput {
    /// Number of running slots with decode tokens this iteration
    num_decode: usize,
    /// If Some, a prefill chunk to run this iteration: (slot_idx, chunk_start, chunk_end)
    prefill_chunk: Option<PrefillChunk>,
}

/// A prefill chunk to process in one iteration.
#[cfg(feature = "cuda")]
struct PrefillChunk {
    /// Which request in the waiting queue
    waiting_idx: usize,
    /// Token range [start, end) within the prompt
    start_token: usize,
    end_token: usize,
    /// True if this is the last chunk (moves request to running)
    is_final: bool,
}

/// Spawn the iteration-level scheduler.
///
/// Drop-in replacement for `spawn_cuda_batch_scheduler` when
/// `ITERATION_SCHEDULER=1` is set.
#[cfg(feature = "cuda")]
pub fn spawn_iteration_scheduler(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: IterationSchedulerConfig,
) -> tokio::sync::mpsc::Sender<CudaBatchRequest> {
    let (tx, rx) = tokio::sync::mpsc::channel::<CudaBatchRequest>(256);

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("PMAT-088: failed to create scheduler runtime");

        rt.block_on(async move {
            iteration_scheduler_loop(model, config, rx).await;
        });
    });

    tx
}

/// State tracking for a request being prefilled in chunks.
#[cfg(feature = "cuda")]
struct ChunkedPrefillState {
    request: CudaBatchRequest,
    /// How many prompt tokens have been prefilled so far
    tokens_prefilled: usize,
    /// Total prompt tokens to prefill
    total_tokens: usize,
}

#[cfg(feature = "cuda")]
async fn iteration_scheduler_loop(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: IterationSchedulerConfig,
    mut rx: tokio::sync::mpsc::Receiver<CudaBatchRequest>,
) {
    eprintln!(
        "[PMAT-088] Iteration scheduler started: max_slots={}, prefill_chunk={}, token_budget={}",
        config.max_slots,
        config.prefill_chunk_size,
        if config.token_budget == 0 {
            "unlimited".to_string()
        } else {
            config.token_budget.to_string()
        }
    );

    // Waiting queue: requests pending prefill
    let mut waiting: VecDeque<CudaBatchRequest> = VecDeque::new();

    // Running state: managed by BatchedDecodeState inside the model lock
    // We track whether a batch is active here
    let mut batch_active = false;
    let mut total_batches: u64 = 0;
    let mut total_iterations: u64 = 0;

    loop {
        // Drain incoming requests into waiting queue
        if !batch_active && waiting.is_empty() {
            // Nothing running — block until a request arrives
            match rx.recv().await {
                Some(req) => waiting.push_back(req),
                None => {
                    eprintln!("[PMAT-088] Iteration scheduler shutting down");
                    return;
                },
            }
        }

        // Non-blocking drain of any additional queued requests
        tokio::task::yield_now().await;
        while let Ok(req) = rx.try_recv() {
            waiting.push_back(req);
        }

        if waiting.is_empty() && !batch_active {
            continue;
        }

        // === SCHEDULING DECISION ===
        //
        // PMAT-088c: Decode-maximal policy (Sarathi-Serve / Orca):
        // 1. Form initial batch with just 1 request (start decode ASAP)
        // 2. Remaining waiting requests join via mid-batch add_slot_to_batch()
        //    between decode steps — interleaved prefill, no decode stalls
        // 3. Slot recycling for finished slots with pending requests
        //
        // This gives decode-maximal scheduling: slot 0 starts generating tokens
        // immediately (~21ms TTFT) while slots 1-3 are prefilled one at a time
        // between decode steps (~14ms prefill per slot, interleaved with ~13ms decode).
        //
        // Expected TTFT improvement at c=4:
        //   Before: 82ms (all 4 prefilled upfront before any decode)
        //   After:  ~21ms for slot 0 (14ms prefill + 6.6ms first decode)
        //           ~35ms for slot 1, ~50ms for slot 2, ~65ms for slot 3
        //           P50 TTFT ≈ 42ms (49% improvement)

        if !batch_active {
            // PMAT-088c: Form initial batch with 1 request. Remaining stay in
            // waiting queue for mid-batch joins during decode loop.
            let batch: Vec<CudaBatchRequest> = vec![waiting.pop_front().unwrap()];

            total_batches += 1;
            let batch_start = std::time::Instant::now();

            eprintln!(
                "[PMAT-088c] Batch #{}: starting m=1 (decode-maximal), waiting={} for mid-batch join",
                total_batches,
                waiting.len(),
            );

            // Process using existing infrastructure (PMAT-072/073/074).
            // Pass rx for mid-batch joins and recycling.
            process_iteration_batch(
                &model,
                batch,
                &mut rx,
                &mut waiting,
                config.max_slots,
                &mut total_iterations,
            );

            let elapsed = batch_start.elapsed();
            eprintln!(
                "[PMAT-088] Batch #{} done in {:.1}ms, {} iterations",
                total_batches,
                elapsed.as_secs_f64() * 1000.0,
                total_iterations,
            );
        }
    }
}

/// A running batch: its decode state, each slot's error sender, and the
/// sub-phase timer of its iterations.
#[cfg(feature = "cuda")]
struct IterationBatch {
    state: crate::gguf::BatchedDecodeState,
    /// PMAT-088c: Option<Sender> so we can drop individual senders when slots finish.
    /// Both the callback's sender AND this sender must be dropped for the channel to close.
    error_senders: Vec<Option<TokenSender>>,
    /// PMAT-284: Uniform sub-phase timing via renacer-core PhaseTimer
    phase_timer: PhaseTimer,
}

/// Process a batch using iteration-level scheduling.
///
/// Uses the existing PMAT-072/073/074 infrastructure but adds:
/// - Waiting queue integration (new requests from waiting queue, not just rx)
/// - Per-iteration metrics
/// - Prefill interleaving preparation
#[cfg(feature = "cuda")]
fn process_iteration_batch(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    batch: Vec<CudaBatchRequest>,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    waiting: &mut VecDeque<CudaBatchRequest>,
    max_slots: usize,
    total_iterations: &mut u64,
) {
    // Single request — fast M=1 path (CUDA graph replay)
    // realizr#212: non_streaming accumulates via Vec::push then bulk-sends
    // PERF-041: same predicate, via the one ungated statement of it, so the
    // F-BATCH-004 knob applies here too. Off, this is `m == 1 && waiting.is_empty()`
    // unchanged.
    if crate::api::batch_admission::fast_path_eligible(
        batch.len(),
        waiting.is_empty(),
        crate::api::batch_admission::force_batched_path(),
    ) {
        let req = batch.into_iter().next().unwrap();
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        crate::api::cuda_batch_scheduler::generate_single_request(&mut cuda_model, req);
        *total_iterations += 1;
        return;
    }

    // Multi-request batch — PMAT-072/073/074 step-wise decode
    // Phase 1: Setup + Prefill
    let Some(mut running) = iteration_setup(model, batch, max_slots) else {
        return;
    };

    // Phase 2: Iteration-level decode loop
    //
    // PMAT-088c: Continuous batch — don't restart when all slots finish if there
    // are pending requests. Recycle done slots instead of exiting → restarting.
    // Without this, every batch restart costs ~20ms setup + 3×30ms slot adds = ~110ms,
    // causing TTFT P50 = 116ms. With continuous recycling, TTFT ≈ 27ms (recycle + decode).
    while !iteration_batch_complete(&running.state, rx, waiting) {
        *total_iterations += 1;
        if !iteration_step(model, &mut running, rx, waiting, *total_iterations) {
            return;
        }
    }

    // Phase 3: Cleanup
    {
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        cuda_model.batched_cleanup(&running.state);
    }
}

/// Phase 1: prefill the batch's prompts into a new decode state. If that
/// fails, every request gets the error and there is no batch.
#[cfg(feature = "cuda")]
fn iteration_setup(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    batch: Vec<CudaBatchRequest>,
    max_slots: usize,
) -> Option<IterationBatch> {
    use crate::gguf::QuantizedGenerateConfig;

    let m = batch.len();
    let prompts: Vec<Vec<u32>> = batch.iter().map(|r| r.prompt_ids.clone()).collect();
    let configs: Vec<QuantizedGenerateConfig> = batch.iter().map(|r| r.config.clone()).collect();
    let error_senders: Vec<Option<TokenSender>> =
        batch.iter().map(|r| Some(r.token_tx.clone())).collect();
    let callbacks: Vec<TokenCallback> = batch
        .into_iter()
        .map(|req| token_callback(req.token_tx))
        .collect();

    let setup = {
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        cuda_model.batched_setup_and_prefill(&prompts, &configs, callbacks, max_slots)
    };
    match setup {
        Ok(state) => Some(IterationBatch {
            state,
            error_senders,
            phase_timer: PhaseTimer::from_env("PMAT_283_TIMING", "PMAT-283"),
        }),
        Err(e) => {
            eprintln!("[PMAT-088] Setup+prefill ERROR (m={m}): {e}");
            for tx in error_senders.iter().flatten() {
                let _ = tx.try_send(Err(e.to_string()));
            }
            None
        },
    }
}

/// Whether the batch is over. Exit conditions:
/// 1. All slots done AND no pending requests (batch truly complete)
/// 2. gen_idx exceeded AND all slots done (safety limit)
///
/// With every slot done, the channel is drained into the waiting queue first:
/// a pending request keeps the batch going, to recycle a done slot.
#[cfg(feature = "cuda")]
fn iteration_batch_complete(
    state: &crate::gguf::BatchedDecodeState,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    waiting: &mut VecDeque<CudaBatchRequest>,
) -> bool {
    if !state.all_done() {
        return false;
    }
    while let Ok(req) = rx.try_recv() {
        waiting.push_back(req);
    }
    waiting.is_empty() || state.gen_idx >= state.max_tokens_max
}

/// One iteration: under the model lock, give waiting requests slots and
/// decode a step; then hand out its tokens without the lock and close the
/// channels of the slots it finished. False when the decode failed: every
/// slot has the error, and the batch is cleaned up.
#[cfg(feature = "cuda")]
fn iteration_step(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    running: &mut IterationBatch,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    waiting: &mut VecDeque<CudaBatchRequest>,
    iteration: u64,
) -> bool {
    let iter_start = std::time::Instant::now();

    running.phase_timer.start();

    let token_ids = {
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");

        running.phase_timer.mark("lock");

        schedule_waiting(&mut cuda_model, running, rx, waiting);

        running.phase_timer.mark("sched");

        // Decode step
        match cuda_model.batched_decode_step(&mut running.state) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!(
                    "[PMAT-088] Decode step ERROR (m={}, step={}): {e}",
                    running.state.m, running.state.gen_idx
                );
                for tx in running.error_senders.iter().flatten() {
                    let _ = tx.try_send(Err(e.to_string()));
                }
                cuda_model.batched_cleanup(&running.state);
                return false;
            },
        }
    };

    running.phase_timer.mark("decode");

    // Token distribution WITHOUT lock
    running.state.distribute_tokens(&token_ids);

    running.phase_timer.mark("dist");

    close_done_slots(running);

    // PMAT-284: Uniform sub-phase timing via renacer-core PhaseTimer
    running.phase_timer.emit(iteration, running.state.m);

    // Per-iteration metrics (first 3 only to avoid log spam)
    if iteration <= 3 || running.state.gen_idx % 50 == 0 {
        let iter_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
        let active_slots = running.state.done.iter().filter(|&&d| !d).count();
        eprintln!(
            "[PMAT-088] iter={}, m={}, active={}, step_ms={:.1}",
            iteration, running.state.m, active_slots, iter_ms,
        );
    }
    true
}

/// PMAT-088: Check waiting queue FIRST, then rx channel.
/// This ensures requests that arrived during previous iteration's
/// token distribution get scheduled before new channel arrivals.
///
/// Priority: RECYCLE done slots first, then ADD new slots.
#[cfg(feature = "cuda")]
fn schedule_waiting(
    cuda_model: &mut OwnedQuantizedModelCuda,
    running: &mut IterationBatch,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    waiting: &mut VecDeque<CudaBatchRequest>,
) {
    if running.state.done.iter().any(|&d| d) {
        // 1. BATCH RECYCLE done slots (multi-prompt prefill, one weight read)
        recycle_done_slots(cuda_model, running, rx, waiting);
    } else if running.state.m < running.state.max_kv_slots {
        // 2. ADD new slots only when no done slots to recycle
        join_waiting_slot(cuda_model, running, rx, waiting);
    }
}

/// PMAT-088d: Batch recycle — collect ALL done slots with waiting
/// requests and recycle them in one prefill_multi_prompt call (~14ms
/// total regardless of count, vs N×17ms sequential). This eliminates
/// recycle serialization when multiple slots finish simultaneously.
#[cfg(feature = "cuda")]
fn recycle_done_slots(
    cuda_model: &mut OwnedQuantizedModelCuda,
    running: &mut IterationBatch,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    waiting: &mut VecDeque<CudaBatchRequest>,
) {
    let mut recycle_pairs: Vec<(
        usize,
        Vec<u32>,
        crate::gguf::QuantizedGenerateConfig,
        TokenCallback,
    )> = Vec::new();
    let mut recycle_error_txs: Vec<(usize, TokenSender)> = Vec::new();

    for slot_idx in 0..running.state.m {
        if !running.state.done[slot_idx] {
            continue;
        }
        let Some(req) = waiting.pop_front().or_else(|| rx.try_recv().ok()) else {
            break; // No more waiting requests
        };
        recycle_error_txs.push((slot_idx, req.token_tx.clone()));
        recycle_pairs.push((
            slot_idx,
            req.prompt_ids,
            req.config,
            token_callback(req.token_tx),
        ));
    }

    if recycle_pairs.is_empty() {
        return;
    }
    match cuda_model.recycle_slots_batch(&mut running.state, recycle_pairs) {
        Ok(()) => {
            for (slot_idx, error_tx) in recycle_error_txs {
                running.error_senders[slot_idx] = Some(error_tx);
            }
        },
        Err(e) => {
            eprintln!("[PMAT-088d] Batch recycle FAILED: {e}");
            for (_, error_tx) in &recycle_error_txs {
                let _ = error_tx.try_send(Err(e.to_string()));
            }
        },
    }
}

/// PMAT-088c: the next waiting request joins the batch in a new slot.
#[cfg(feature = "cuda")]
fn join_waiting_slot(
    cuda_model: &mut OwnedQuantizedModelCuda,
    running: &mut IterationBatch,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    waiting: &mut VecDeque<CudaBatchRequest>,
) {
    let Some(req) = waiting.pop_front().or_else(|| rx.try_recv().ok()) else {
        return;
    };
    let error_tx = req.token_tx.clone();
    match cuda_model.add_slot_to_batch(
        &mut running.state,
        req.prompt_ids,
        req.config,
        token_callback(req.token_tx),
    ) {
        Ok(()) => running.error_senders.push(Some(error_tx)),
        Err(e) => {
            eprintln!("[PMAT-088c] Mid-batch join FAILED: {e}");
            let _ = error_tx.try_send(Err(e.to_string()));
        },
    }
}

/// PMAT-088c: Drop callbacks AND error senders for done slots to close channels.
/// The SSE handler waits for channel closure (ALL senders dropped) to send [DONE].
/// Without this, continuous batching keeps senders alive → SSE never ends
/// → probador never sends new requests → recycling never gets requests.
#[cfg(feature = "cuda")]
fn close_done_slots(running: &mut IterationBatch) {
    let IterationBatch {
        state,
        error_senders,
        ..
    } = running;
    for slot_idx in 0..state.m {
        if state.done[slot_idx]
            && error_senders
                .get(slot_idx)
                .and_then(|o| o.as_ref())
                .is_some()
        {
            // Replace callback → drops old closure → drops one sender
            state.on_tokens[slot_idx] = Box::new(|_| false);
            // Drop error sender → drops second sender → channel closes
            error_senders[slot_idx] = None;
        }
    }
}
