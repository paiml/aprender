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
use crate::api::cuda_batch_scheduler::CudaBatchRequest;
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
    use crate::gguf::QuantizedGenerateConfig;

    let m = batch.len();

    // Single request — fast M=1 path (CUDA graph replay)
    // realizr#212: non_streaming accumulates via Vec::push then bulk-sends
    if m == 1 && waiting.is_empty() {
        let req = batch.into_iter().next().unwrap();
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        crate::api::cuda_batch_scheduler::generate_single_request(&mut cuda_model, req);
        *total_iterations += 1;
        return;
    }

    // Multi-request batch — PMAT-072/073/074 step-wise decode
    let prompts: Vec<Vec<u32>> = batch.iter().map(|r| r.prompt_ids.clone()).collect();
    let configs: Vec<QuantizedGenerateConfig> = batch.iter().map(|r| r.config.clone()).collect();
    // PMAT-088c: Option<Sender> so we can drop individual senders when slots finish.
    // Both the callback's sender AND this sender must be dropped for the channel to close.
    let mut error_senders: Vec<Option<tokio::sync::mpsc::Sender<Result<u32, String>>>> =
        batch.iter().map(|r| Some(r.token_tx.clone())).collect();

    let callbacks: Vec<Box<dyn FnMut(u32) -> bool + Send>> = batch
        .into_iter()
        .map(|req| {
            Box::new(move |token_id: u32| -> bool { req.token_tx.try_send(Ok(token_id)).is_ok() })
                as Box<dyn FnMut(u32) -> bool + Send>
        })
        .collect();

    // Phase 1: Setup + Prefill
    let mut state = {
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        match cuda_model.batched_setup_and_prefill(&prompts, &configs, callbacks, max_slots) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[PMAT-088] Setup+prefill ERROR (m={m}): {e}");
                for tx in error_senders.iter().flatten() {
                    let _ = tx.try_send(Err(e.to_string()));
                }
                return;
            },
        }
    };

    // Phase 2: Iteration-level decode loop
    //
    // PMAT-088c: Continuous batch — don't restart when all slots finish if there
    // are pending requests. Recycle done slots instead of exiting → restarting.
    // Without this, every batch restart costs ~20ms setup + 3×30ms slot adds = ~110ms,
    // causing TTFT P50 = 116ms. With continuous recycling, TTFT ≈ 27ms (recycle + decode).

    // PMAT-284: Uniform sub-phase timing via renacer-core PhaseTimer
    let mut phase_timer = PhaseTimer::from_env("PMAT_283_TIMING", "PMAT-283");

    loop {
        // Exit conditions:
        // 1. All slots done AND no pending requests (batch truly complete)
        // 2. gen_idx exceeded AND all slots done (safety limit)
        if state.all_done() {
            // Drain channel into waiting queue before deciding to exit
            while let Ok(req) = rx.try_recv() {
                waiting.push_back(req);
            }
            if waiting.is_empty() {
                break; // No pending requests — batch complete
            }
            // Pending requests exist — continue loop to recycle done slots
        }
        if state.gen_idx >= state.max_tokens_max && state.all_done() {
            break; // Safety limit reached
        }

        *total_iterations += 1;
        let iter_start = std::time::Instant::now();

        phase_timer.start();

        let token_ids = {
            let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");

            phase_timer.mark("lock");

            // PMAT-088: Check waiting queue FIRST, then rx channel.
            // This ensures requests that arrived during previous iteration's
            // token distribution get scheduled before new channel arrivals.

            // PMAT-088d: Batch recycle — collect ALL done slots with waiting
            // requests and recycle them in one prefill_multi_prompt call (~14ms
            // total regardless of count, vs N×17ms sequential). This eliminates
            // recycle serialization when multiple slots finish simultaneously.
            //
            // Priority: RECYCLE done slots first, then ADD new slots.
            let mut joined_this_step = false;

            // 1. BATCH RECYCLE done slots (multi-prompt prefill, one weight read)
            let has_done_slots = state.done.iter().any(|&d| d);
            if has_done_slots {
                let mut recycle_pairs: Vec<(
                    usize,
                    Vec<u32>,
                    crate::gguf::QuantizedGenerateConfig,
                    Box<dyn FnMut(u32) -> bool + Send>,
                )> = Vec::new();
                let mut recycle_error_txs: Vec<(
                    usize,
                    tokio::sync::mpsc::Sender<Result<u32, String>>,
                )> = Vec::new();

                for slot_idx in 0..state.m {
                    if !state.done[slot_idx] {
                        continue;
                    }
                    let next_req = waiting.pop_front().or_else(|| rx.try_recv().ok());
                    if let Some(req) = next_req {
                        let error_tx = req.token_tx.clone();
                        let on_token: Box<dyn FnMut(u32) -> bool + Send> = {
                            let token_tx = req.token_tx;
                            Box::new(move |token_id: u32| -> bool {
                                token_tx.try_send(Ok(token_id)).is_ok()
                            })
                        };
                        recycle_error_txs.push((slot_idx, error_tx));
                        recycle_pairs.push((slot_idx, req.prompt_ids, req.config, on_token));
                    } else {
                        break; // No more waiting requests
                    }
                }

                if !recycle_pairs.is_empty() {
                    match cuda_model.recycle_slots_batch(&mut state, recycle_pairs) {
                        Ok(()) => {
                            for (slot_idx, error_tx) in recycle_error_txs {
                                error_senders[slot_idx] = Some(error_tx);
                            }
                            joined_this_step = true;
                        },
                        Err(e) => {
                            eprintln!("[PMAT-088d] Batch recycle FAILED: {e}");
                            for (_, error_tx) in &recycle_error_txs {
                                let _ = error_tx.try_send(Err(e.to_string()));
                            }
                        },
                    }
                }
            }

            // 2. ADD new slots only when no done slots to recycle
            if !joined_this_step && !has_done_slots && state.m < state.max_kv_slots {
                let next_req = waiting.pop_front().or_else(|| rx.try_recv().ok());
                if let Some(req) = next_req {
                    let error_tx = req.token_tx.clone();
                    let on_token: Box<dyn FnMut(u32) -> bool + Send> = {
                        let token_tx = req.token_tx;
                        Box::new(move |token_id: u32| -> bool {
                            token_tx.try_send(Ok(token_id)).is_ok()
                        })
                    };
                    match cuda_model.add_slot_to_batch(
                        &mut state,
                        req.prompt_ids,
                        req.config,
                        on_token,
                    ) {
                        Ok(()) => {
                            error_senders.push(Some(error_tx));
                            joined_this_step = true;
                        },
                        Err(e) => {
                            eprintln!("[PMAT-088c] Mid-batch join FAILED: {e}");
                            let _ = error_tx.try_send(Err(e.to_string()));
                        },
                    }
                }
            }

            let _ = joined_this_step; // suppress unused warning

            phase_timer.mark("sched");

            // Decode step
            match cuda_model.batched_decode_step(&mut state) {
                Ok(ids) => ids,
                Err(e) => {
                    eprintln!(
                        "[PMAT-088] Decode step ERROR (m={}, step={}): {e}",
                        state.m, state.gen_idx
                    );
                    for tx in error_senders.iter().flatten() {
                        let _ = tx.try_send(Err(e.to_string()));
                    }
                    cuda_model.batched_cleanup(&state);
                    return;
                },
            }
        };

        phase_timer.mark("decode");

        // Token distribution WITHOUT lock
        state.distribute_tokens(&token_ids);

        phase_timer.mark("dist");

        // PMAT-088c: Drop callbacks AND error senders for done slots to close channels.
        // The SSE handler waits for channel closure (ALL senders dropped) to send [DONE].
        // Without this, continuous batching keeps senders alive → SSE never ends
        // → probador never sends new requests → recycling never gets requests.
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

        // PMAT-284: Uniform sub-phase timing via renacer-core PhaseTimer
        phase_timer.emit(*total_iterations as u64, state.m);

        // Per-iteration metrics (first 3 only to avoid log spam)
        if *total_iterations <= 3 || state.gen_idx % 50 == 0 {
            let iter_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
            let active_slots = state.done.iter().filter(|&&d| !d).count();
            eprintln!(
                "[PMAT-088] iter={}, m={}, active={}, step_ms={:.1}",
                total_iterations, state.m, active_slots, iter_ms,
            );
        }
    }

    // Phase 3: Cleanup
    {
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        cuda_model.batched_cleanup(&state);
    }
}
