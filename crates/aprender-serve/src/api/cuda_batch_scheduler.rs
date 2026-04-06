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
    if req.non_streaming {
        let mut tokens = Vec::new();
        let result =
            cuda_model.generate_gpu_resident_streaming(&req.prompt_ids, &req.config, |tid| {
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
        let result =
            cuda_model.generate_gpu_resident_streaming(&req.prompt_ids, &req.config, |tid| {
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
) -> tokio::sync::mpsc::Sender<CudaBatchRequest> {
    let (tx, rx) = tokio::sync::mpsc::channel::<CudaBatchRequest>(256);

    // Run the scheduler in a blocking thread (CUDA ops are synchronous)
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("PMAT-044: failed to create scheduler runtime");

        rt.block_on(async move {
            cuda_batch_scheduler_loop(model, config, rx).await;
        });
    });

    tx
}

#[cfg(feature = "cuda")]
async fn cuda_batch_scheduler_loop(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: CudaBatchConfig,
    mut rx: tokio::sync::mpsc::Receiver<CudaBatchRequest>,
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
        let mut batch = vec![first];
        if config.window_ms == 0 {
            // PMAT-086/095: Zero-latency drain with cooperative yield + adaptive wait.
            tokio::task::yield_now().await;
            while batch.len() < config.max_batch {
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
            let should_wait = if batch.len() > 1 && batch.len() < config.max_batch {
                true // Phase 2a: found peers, wait for more
            } else if batch.len() == 1 && recent_batch_gt1 && config.max_batch > 1 {
                true // Phase 2b: singleton but concurrent traffic detected
            } else {
                false
            };
            if should_wait {
                let wait_ms = if batch.len() > 1 { 3 } else { 2 };
                let deadline =
                    tokio::time::Instant::now() + tokio::time::Duration::from_millis(wait_ms);
                while batch.len() < config.max_batch {
                    match tokio::time::timeout_at(deadline, rx.recv()).await {
                        Ok(Some(req)) => batch.push(req),
                        Ok(None) => break,
                        Err(_timeout) => break,
                    }
                }
            }
        } else {
            let deadline =
                tokio::time::Instant::now() + tokio::time::Duration::from_millis(config.window_ms);

            while batch.len() < config.max_batch {
                match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Some(req)) => batch.push(req),
                    Ok(None) => {
                        eprintln!("[PMAT-044] Channel closed during accumulation");
                        if !batch.is_empty() {
                            process_cuda_batch(&model, batch, &mut rx, config.max_batch);
                        }
                        return;
                    },
                    Err(_timeout) => break, // Window expired
                }
            }
        }

        let batch_size = batch.len();
        let batch_start = std::time::Instant::now();

        // PMAT-097: Update concurrency hint for next batch's adaptive wait.
        recent_batch_gt1 = batch_size > 1;

        // Process the batch (PMAT-073: pass rx for mid-batch joins)
        process_cuda_batch(&model, batch, &mut rx, config.max_batch);

        let elapsed = batch_start.elapsed();
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
}

#[cfg(feature = "cuda")]
fn process_cuda_batch(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    batch: Vec<CudaBatchRequest>,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    max_batch: usize,
) {
    let m = batch.len();

    if m == 1 {
        // Single request — use the optimized single-request path (138 tok/s vs 46 batched).
        // PMAT-073: Mid-batch joins only work for initial batches >1. For c=1→c=2
        // staggered arrivals, the second request queues until the first completes.
        // This is acceptable because the fast path's 3x better ITL outweighs the
        // latency benefit of mid-batch join for the second request.
        let req = batch.into_iter().next().unwrap();
        if std::env::var("TTFT_TRACE").is_ok() {
            eprintln!(
                "[TTFT] {:>20}: {:>7.2}ms",
                "queue_latency",
                req.enqueue_time.elapsed().as_secs_f64() * 1000.0
            );
        }
        let mut cuda_model = model.write().expect("PMAT-044: model lock poisoned");
        if std::env::var("TTFT_TRACE").is_ok() {
            eprintln!(
                "[TTFT] {:>20}: {:>7.2}ms",
                "lock_acquired",
                req.enqueue_time.elapsed().as_secs_f64() * 1000.0
            );
        }
        generate_single_request(&mut cuda_model, req);
        return;
    }

    // PMAT-099: Staggered prefill — prefill first prompt only, join rest during decode.
    // FALSIFIED for short prompts: per-slot join overhead (14ms×3) exceeds batched prefill (20ms).
    // c=4 short prompts: staggered 241 vs batched 260 aggregate (-7.3%), TTFT 87 vs 40ms (+118%).
    // May win for long prompts (>500 tokens) — not yet tested.
    // Default OFF. Enable: STAGGERED_PREFILL=1.
    let staggered = std::env::var("STAGGERED_PREFILL").as_deref() == Ok("1") && m > 1;

    let mut pending_joins: std::collections::VecDeque<CudaBatchRequest> =
        std::collections::VecDeque::new();

    // Build Phase 1 inputs: first request only (staggered) or all requests (batched)
    let (phase1_prompts, phase1_configs, mut error_senders, phase1_callbacks) = if staggered {
        // Split batch: first → immediate prefill, rest → pending joins
        let mut batch_iter = batch.into_iter();
        let first_req = batch_iter.next().unwrap();
        pending_joins = batch_iter.collect();

        let prompts = vec![first_req.prompt_ids.clone()];
        let configs = vec![first_req.config.clone()];
        let error_senders = vec![first_req.token_tx.clone()];
        let first_tx = first_req.token_tx;
        let callbacks: Vec<Box<dyn FnMut(u32) -> bool + Send>> =
            vec![Box::new(move |token_id: u32| -> bool {
                first_tx.try_send(Ok(token_id)).is_ok()
            })];

        eprintln!(
            "[PMAT-099] Staggered prefill: 1 immediate + {} pending joins",
            pending_joins.len()
        );

        (prompts, configs, error_senders, callbacks)
    } else {
        // All prompts prefilled together in Phase 1 (original PMAT-072 behavior)
        let prompts: Vec<Vec<u32>> = batch.iter().map(|r| r.prompt_ids.clone()).collect();
        let configs: Vec<QuantizedGenerateConfig> =
            batch.iter().map(|r| r.config.clone()).collect();
        let error_senders: Vec<tokio::sync::mpsc::Sender<Result<u32, String>>> =
            batch.iter().map(|r| r.token_tx.clone()).collect();
        let callbacks: Vec<Box<dyn FnMut(u32) -> bool + Send>> = batch
            .into_iter()
            .map(|req| {
                Box::new(move |token_id: u32| -> bool {
                    req.token_tx.try_send(Ok(token_id)).is_ok()
                }) as Box<dyn FnMut(u32) -> bool + Send>
            })
            .collect();
        (prompts, configs, error_senders, callbacks)
    };

    // Phase 1: Setup + Prefill (under lock)
    // Staggered: prefills 1 prompt, pre-allocates max_batch KV slots.
    // Non-staggered: prefills all M prompts (original behavior).
    let mut state = {
        let mut cuda_model = model.write().expect("PMAT-072: model lock poisoned");
        match cuda_model.batched_setup_and_prefill(
            &phase1_prompts,
            &phase1_configs,
            phase1_callbacks,
            max_batch,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[PMAT-072] Setup+prefill ERROR (m={m}): {e}");
                for tx in &error_senders {
                    let _ = tx.try_send(Err(e.to_string()));
                }
                // Also notify pending joins
                for req in &pending_joins {
                    let _ = req.token_tx.try_send(Err(e.to_string()));
                }
                return;
            },
        }
    };

    // Phase 2: Decode loop with mid-batch joins (PMAT-073/099) and slot recycling (PMAT-074)
    // Lock per step (~19ms per acquire vs ~660ms total).
    // PMAT-099: Pending staggered joins are processed one-per-step for progressive ramp-up.
    while !state.all_done() && state.gen_idx < state.max_tokens_max {
        let token_ids = {
            let mut cuda_model = model.write().expect("PMAT-072: model lock poisoned");

            // PMAT-099: Join one pending staggered slot per step (progressive ramp-up).
            // This limits decode stall to one prefill per step instead of blocking all at once.
            if !pending_joins.is_empty() && state.m < state.max_kv_slots {
                let req = pending_joins.pop_front().unwrap();
                let error_tx = req.token_tx.clone();
                let prompt_ids = req.prompt_ids;
                let config = req.config;
                let token_tx = req.token_tx;
                let on_token: Box<dyn FnMut(u32) -> bool + Send> =
                    Box::new(move |token_id: u32| -> bool {
                        token_tx.try_send(Ok(token_id)).is_ok()
                    });
                match cuda_model.add_slot_to_batch(&mut state, prompt_ids, config, on_token) {
                    Ok(()) => {
                        error_senders.push(error_tx);
                    },
                    Err(e) => {
                        eprintln!("[PMAT-099] Staggered join FAILED: {e}");
                        let _ = error_tx.try_send(Err(e.to_string()));
                    },
                }
            }

            // PMAT-073: Check for pending requests to join mid-batch (fill empty slots).
            while state.m < state.max_kv_slots {
                match rx.try_recv() {
                    Ok(req) => {
                        let error_tx = req.token_tx.clone();
                        let prompt_ids = req.prompt_ids;
                        let config = req.config;
                        let token_tx = req.token_tx;
                        let on_token: Box<dyn FnMut(u32) -> bool + Send> =
                            Box::new(move |token_id: u32| -> bool {
                                token_tx.try_send(Ok(token_id)).is_ok()
                            });
                        match cuda_model.add_slot_to_batch(&mut state, prompt_ids, config, on_token)
                        {
                            Ok(()) => {
                                error_senders.push(error_tx);
                            },
                            Err(e) => {
                                eprintln!("[PMAT-073] Mid-batch join FAILED: {e}");
                                let _ = error_tx.try_send(Err(e.to_string()));
                            },
                        }
                    },
                    Err(_) => break, // No pending requests
                }
            }

            // PMAT-074: Slot recycling — reuse finished slots for pending requests.
            // Check staggered pending joins first, then external channel.
            for slot_idx in 0..state.m {
                if !state.done[slot_idx] {
                    continue;
                }
                // Try pending staggered joins first (they arrived with the initial batch)
                let req = if !pending_joins.is_empty() {
                    Some(pending_joins.pop_front().unwrap())
                } else {
                    rx.try_recv().ok()
                };
                match req {
                    Some(req) => {
                        let error_tx = req.token_tx.clone();
                        let prompt_ids = req.prompt_ids;
                        let config = req.config;
                        let token_tx = req.token_tx;
                        let on_token: Box<dyn FnMut(u32) -> bool + Send> =
                            Box::new(move |token_id: u32| -> bool {
                                token_tx.try_send(Ok(token_id)).is_ok()
                            });
                        match cuda_model
                            .recycle_slot(&mut state, slot_idx, prompt_ids, config, on_token)
                        {
                            Ok(()) => {
                                error_senders[slot_idx] = error_tx;
                            },
                            Err(e) => {
                                eprintln!("[PMAT-074] Slot recycle FAILED (slot {slot_idx}): {e}");
                                let _ = error_tx.try_send(Err(e.to_string()));
                            },
                        }
                    },
                    None => break, // No pending requests
                }
            }

            match cuda_model.batched_decode_step(&mut state) {
                Ok(ids) => ids,
                Err(e) => {
                    eprintln!(
                        "[PMAT-074] Decode step ERROR (m={}, step={}): {e}",
                        state.m, state.gen_idx
                    );
                    for tx in &error_senders {
                        let _ = tx.try_send(Err(e.to_string()));
                    }
                    // Notify any remaining pending joins
                    for req in &pending_joins {
                        let _ = req.token_tx.try_send(Err(e.to_string()));
                    }
                    // Still need cleanup under lock
                    cuda_model.batched_cleanup(&state);
                    return;
                },
            }
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
