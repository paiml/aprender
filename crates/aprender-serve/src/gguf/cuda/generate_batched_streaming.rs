// Batched streaming token generation with per-step lock release for concurrent scheduling.

/// PMAT-764/792b: per-slot token selection for batched decode — greedy argmax when
/// `temperature == 0.0`, else temperature/top-k/top-p sampling. Defined in the cuda module
/// so the batched loop can honor each request's OWN sampling params instead of the old forced
/// on-GPU greedy argmax.
///
/// PMAT-792b: the stochastic branch now applies the per-request `top_p` (nucleus) cutoff.
/// Previously it called `sample_topk`, which silently dropped `top_p` — so a batched request
/// asking for e.g. `top_p=0.1` got the full top_k distribution sampled. That is the same
/// per-request-param-drop class as the PMAT-764 temperature bug (FALSIFY-CB-010), just the next
/// sampling parameter, which the #2027 fix and contract both missed.
///
/// The pipeline (temperature → top-k truncate → top-p nucleus → softmax → sample) matches the
/// canonical `sample_advanced` nucleus logic. With `top_p >= 1.0` the nucleus step is a no-op,
/// so all-default and top_k-only requests behave exactly as the pre-fix `sample_topk` path.
fn select_batched_token(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> u32 {
    use rand::Rng;

    if temperature == 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32);
    }

    // 1. Temperature scaling.
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // 2. Top-k: sort descending, truncate (top_k == 0 keeps the full vocabulary).
    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if top_k > 0 {
        indexed.truncate(top_k);
    }

    // 3. Top-p (nucleus, PMAT-792b): keep the smallest prefix whose cumulative probability
    //    reaches top_p. Skipped when top_p is out of (0,1) — a no-op preserving the top_k path.
    if top_p > 0.0 && top_p < 1.0 {
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_vals: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let total: f32 = exp_vals.iter().sum();
        let mut cumulative = 0.0;
        let mut cutoff = indexed.len();
        for (i, &ev) in exp_vals.iter().enumerate() {
            cumulative += ev / total;
            if cumulative >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        indexed.truncate(cutoff);
    }

    // 4. Softmax over the surviving set.
    let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|(i, v)| (*i, (v - max_val).exp() / exp_sum))
        .collect();

    // 5. Inverse-CDF sample.
    let mut rng = rand::rng();
    let r: f32 = rng.random();
    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if cumulative >= r {
            return idx as u32;
        }
    }
    probs.last().map_or(0, |(idx, _)| *idx as u32)
}

/// FALSIFY-CB-008 (`contracts/continuous-batching-v1.yaml`): "All M slots produce distinct
/// tokens per decode step (not constant)". The contract named `BATCHED_DECODE_TRACE` as its
/// evidence; the variable did not exist, so nothing could ever have read that log. It does now.
///
/// `BATCHED_DECODE_TRACE=1` emits one line per batched decode step:
///
/// ```text
/// [CB-008] step=0 m=4 token_ids=[3156, 3156, 3156, 3156] positions=[16, 16, 16, 16] done=[..]
/// ```
///
/// Read once per process. The existing `[PMAT-072] decode step` lines stop at step 3, which is
/// too early to see a slot freeze; CB-008 is a property of the whole generation.
#[cfg(feature = "cuda")]
fn batched_decode_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BATCHED_DECODE_TRACE").as_deref() == Ok("1"))
}

/// PMAT-072: Decode state for step-wise batched generation.
///
/// Extracted from `generate_batched_streaming` to allow lock release between
/// decode steps. The scheduler acquires the model lock for ~19ms per step
/// instead of ~660ms for the entire batch.
pub struct BatchedDecodeState {
    /// Number of concurrent slots (batch size)
    pub m: usize,
    /// Maximum KV cache slots pre-allocated (for PMAT-073 mid-batch joins)
    pub max_kv_slots: usize,
    /// Maximum tokens to generate across all slots
    pub max_tokens_max: usize,
    /// Current decode step index (incremented by `distribute_tokens`)
    pub gen_idx: usize,
    /// Model hidden dimension (for embedding buffer sizing)
    pub hidden_dim: usize,
    /// Model intermediate dimension (for workspace cleanup)
    pub intermediate_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Layer norm epsilon
    pub eps: f32,
    /// Original prompts per slot (for generated-length tracking)
    pub prompts: Vec<Vec<u32>>,
    /// Generation config per slot (stop tokens, max tokens)
    pub configs: Vec<QuantizedGenerateConfig>,
    /// Per-slot streaming callbacks (SSE token delivery)
    pub on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>>,
    /// Accumulated sequences per slot (prompt + generated tokens)
    pub sequences: Vec<Vec<u32>>,
    /// Current position per slot (for RoPE)
    pub positions: Vec<usize>,
    /// Last generated token per slot (input for next decode step)
    pub last_tokens: Vec<u32>,
    /// Whether each slot has finished generating
    pub done: Vec<bool>,
    /// Pre-allocated embedding buffer `[m × hidden_dim]`
    pub embed_buf: Vec<f32>,
    /// PMAT-086: Pre-allocated position buffer `[m]` (avoids Vec alloc per decode step)
    pub pos_buf: Vec<u32>,
}

impl BatchedDecodeState {
    /// Distribute generated tokens to SSE callbacks. No model lock needed.
    ///
    /// Returns true if all slots are done.
    pub fn distribute_tokens(&mut self, token_ids: &[u32]) -> bool {
        for slot_idx in 0..self.m {
            if self.done[slot_idx] {
                continue;
            }

            let next_token = token_ids[slot_idx];

            if self.configs[slot_idx].stop_tokens.contains(&next_token) {
                self.done[slot_idx] = true;
                continue;
            }

            self.sequences[slot_idx].push(next_token);

            if !self.on_tokens[slot_idx](next_token) {
                self.done[slot_idx] = true;
                continue;
            }

            self.last_tokens[slot_idx] = next_token;
            self.positions[slot_idx] += 1;

            let generated = self.sequences[slot_idx].len() - self.prompts[slot_idx].len();
            if generated >= self.configs[slot_idx].max_tokens {
                self.done[slot_idx] = true;
            }
        }
        self.gen_idx += 1;
        self.done.iter().all(|&d| d)
    }

    /// Check if all slots have finished generating.
    pub fn all_done(&self) -> bool {
        self.done.iter().all(|&d| d)
    }
}

/// PMAT-044: Streaming batched generation for continuous batching
///
/// Processes M requests concurrently using batched GEMV (weight sharing):
/// - Weights read ONCE per decode step for all M sequences
/// - Per-slot GPU KV caches (PAR-119)
/// - Per-slot streaming callbacks for SSE token delivery
///
/// # Performance Target
///
/// Jetson Orin Nano Super (c=4):
/// - Current (sequential): 34.3 tok/s aggregate
/// - Target (batched): 50-80 tok/s via weight amortization
impl OwnedQuantizedModelCuda {
    /// Generate tokens for M requests with per-request streaming callbacks.
    ///
    /// Backward-compatible wrapper around the PMAT-072 step-wise API.
    /// For lock-releasing batched decode, use `batched_setup_and_prefill`,
    /// `batched_decode_step`, and `batched_cleanup` directly.
    pub fn generate_batched_streaming(
        &mut self,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>>,
    ) -> Result<Vec<Vec<u32>>> {
        contract_pre_correctness_under_batching!();
        let m = prompts.len();
        eprintln!("[PMAT-044] generate_batched_streaming ENTRY m={m}");
        if m == 0 {
            return Ok(Vec::new());
        }

        let mut state = self.batched_setup_and_prefill(prompts, configs, on_tokens, m)?;

        while !state.all_done() && state.gen_idx < state.max_tokens_max {
            let token_ids = self.batched_decode_step(&mut state)?;
            state.distribute_tokens(&token_ids);
        }

        self.batched_cleanup(&state);

        Ok(state.sequences)
    }

    /// PMAT-072: Setup and prefill phase for step-wise batched generation.
    ///
    /// Performs validation, graph clearing, workspace init, KV cache allocation,
    /// and prefill. Returns a `BatchedDecodeState` for use with `batched_decode_step`.
    ///
    /// Caller must hold model write lock for the duration of this call.
    pub fn batched_setup_and_prefill(
        &mut self,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>>,
        max_kv_slots: usize,
    ) -> Result<BatchedDecodeState> {
        let m = prompts.len();
        eprintln!("[PMAT-072] batched_setup_and_prefill ENTRY m={m}");
        self.validate_batch_args(m, configs.len(), on_tokens.len())?;

        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // PMAT-044 FIX: Clear decode graph from prior M=1 requests. prefill_and_scatter
        // uses forward_gpu_resident which replays the M=1 decode graph. During graph
        // replay, CPU-side kv_cache_lengths is NOT updated (only GPU kernels run),
        // causing KV cache state mismatch. Force eager path for prefill by clearing graph.
        self.executor.clear_decode_graph();

        // PMAT-075: Do NOT clear batched decode graphs here. Workspace buffer
        // addresses are now stable across batches (batched_cleanup preserves them
        // via init_workspace skip path). If init_batched_workspace or
        // init_prefill_workspace must reallocate (longer prompt), they clear
        // graphs at the reallocation site (poka-yoke at source).

        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;
        let max_tokens_max = configs.iter().map(|c| c.max_tokens).max().unwrap_or(128);
        let max_prompt_len = prompts.iter().map(Vec::len).max().unwrap_or(0);

        // Create CPU KV caches for each slot (needed for prefill path)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let max_seq_len = max_prompt_len + max_tokens_max;
        let mut caches: Vec<OwnedQuantizedKVCache> = (0..m)
            .map(|_| OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq_len))
            .collect();

        // PMAT-045: Pre-allocate workspace for max_prompt_len BEFORE prefill.
        // buffer_capacity high-water mark ensures init_batched_workspace(m) skips
        // reallocation since buffer_capacity >= max_prompt_len >= m.
        self.executor
            .init_prefill_workspace(max_prompt_len, hidden_dim, intermediate_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_prefill_workspace".to_string(),
                reason: format!("Failed to pre-allocate workspace for seq_len={max_prompt_len}: {e}"),
            })?;

        // PMAT-060: Allocate batched KV alongside FP16 weight cache (both fit in VRAM).
        // Five-Whys: c=4 TTFT = 659ms (29.9x gap vs llama.cpp 22ms).
        // Why? SGEMM prefill at ~165ms/slot instead of HGEMM ~58ms/slot.
        // Why? FP16 weight cache (2944 MB) cleared before prefill (GH-141).
        // Why? Assumed FP16 + batched KV + Q4K > 8 GB VRAM.
        // Reality: FP16 (2944) + batched KV (56-896) + Q4K (850) = 3850-4700 MB.
        //          RTX 4060L has ~7.5 GB usable → fits with 2.8-3.6 GB headroom.
        // Fix: Keep FP16 during prefill, allocate batched KV, D2D scatter, THEN clear FP16.
        // GH-141 ILLEGAL_ADDRESS was from a different root cause (stale graph pointers).
        // PMAT-073: Pre-allocate batched KV for max_kv_slots to enable mid-batch joins.
        // Extra slots cost ~224 MB/slot but avoid reallocation when new requests join.
        let kv_alloc = max_kv_slots.max(m);
        self.executor
            .init_batched_kv_cache_gpu(num_layers, kv_alloc)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_kv_cache_gpu".to_string(),
                reason: format!("Failed to init batched KV cache for M={kv_alloc}: {e}"),
            })?;
        self.executor.reset_batched_kv_cache_gpu();

        // Prefill each slot with HGEMM (FP16 present) + D2D scatter to batched KV.
        let (sequences, positions, last_tokens) =
            self.prefill_and_scatter(prompts, &mut caches)?;

        // PMAT-062: DP4A GEMV for batched decode (default), not cuBLAS HGEMM.
        // PMAT-088b: Re-testing HGEMM crossover. At M=4, Q4K GEMV is compute-bound
        // (9.7ms DP4A chains) while HGEMM is BW-bound (9.1ms FP16 reads). Tensor cores
        // may win when compute dominates. PMAT-062 confound: fused gate+up was disabled.
        // Override: HGEMM_BATCHED_DECODE=1 to force cuBLAS tensor core path.
        let use_hgemm_decode = std::env::var("HGEMM_BATCHED_DECODE").as_deref() == Ok("1");
        if use_hgemm_decode {
            // PMAT-088b: FP16 cache may be empty if FP8 prefill is active (sm_89+).
            // Clear FP8 cache to free VRAM (1472 MB), then warm FP16 cache (2944 MB).
            // FP8 cache will be re-warmed lazily on next prefill if needed.
            if !self.executor.has_fp16_weight_cache() {
                self.executor.clear_fp8_weight_cache();
                self.executor.ensure_cublas().map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "ensure_cublas".to_string(),
                    reason: format!("HGEMM batched decode: cuBLAS init failed: {e}"),
                })?;
                self.executor
                    .warmup_hgemm_cache(
                        num_layers,
                        hidden_dim as u32,
                        intermediate_dim as u32,
                        vocab_size as u32,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "warmup_hgemm_cache".to_string(),
                        reason: format!("HGEMM batched decode: FP16 cache warmup failed: {e}"),
                    })?;
            }
            self.executor.hgemm_batched_decode_active = true;
            eprintln!("[PMAT-088b] HGEMM batched decode active (FP16 cache: {} matrices)",
                if self.executor.has_fp16_weight_cache() { "ready" } else { "EMPTY" });
        }
        self.executor.clear_prefill_graphs();

        // PMAT-045: Set workspace.batch_size = m for decode kernels (they check == m).
        // buffer_capacity stays at max_prompt_len (high-water mark), so this call
        // skips reallocation and just sets the logical batch_size.
        self.executor
            .init_batched_workspace(hidden_dim, intermediate_dim, m)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_workspace".to_string(),
                reason: format!("Failed to set decode batch_size={m}: {e}"),
            })?;

        // PMAT-044 FIX: Clear M=1 decode graph — prefill_and_scatter may have
        // captured an M=1 graph with stale addresses. init_batched_workspace skips
        // reallocation if batch_size matches (PMAT-045), preserving batched graphs.
        self.executor.clear_decode_graph();

        let embed_buf = vec![0.0f32; m * hidden_dim];
        let pos_buf: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let done = vec![false; m];

        Ok(BatchedDecodeState {
            m,
            max_kv_slots: kv_alloc,
            max_tokens_max,
            gen_idx: 0,
            hidden_dim,
            intermediate_dim,
            num_layers,
            vocab_size,
            eps,
            prompts: prompts.to_vec(),
            configs: configs.to_vec(),
            on_tokens,
            sequences,
            positions,
            last_tokens,
            done,
            embed_buf,
            pos_buf,
        })
    }

    /// PMAT-072: Single decode step for step-wise batched generation.
    ///
    /// Embeds current tokens, runs one forward pass, returns raw token IDs.
    /// Call `state.distribute_tokens(&token_ids)` after releasing the model lock
    /// to deliver tokens to SSE callbacks without holding the lock.
    ///
    /// Caller must hold model write lock for the duration of this call.
    pub fn batched_decode_step(
        &mut self,
        state: &mut BatchedDecodeState,
    ) -> Result<Vec<u32>> {
        // PERF-050 round 3: in-process A/B against the M=1 oracle (APR_PARITY_PROBE=1).
        self.cb006_parity_probe(state);

        // PMAT-286: Sub-phase timing inside decode step
        static DECODE_PHASE_TIMER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let timing = *DECODE_PHASE_TIMER.get_or_init(|| {
            std::env::var("PMAT_286_TIMING").map(|v| v == "1").unwrap_or(false)
        });
        let mut timer = renacer_core::PhaseTimer::from_env("PMAT_286_TIMING", "PMAT-286-decode");
        timer.start();

        // Embed all tokens into state's pre-allocated buffer
        for slot_idx in 0..state.m {
            let offset = slot_idx * state.hidden_dim;
            if state.done[slot_idx] {
                state.embed_buf[offset..offset + state.hidden_dim].fill(0.0);
            } else {
                self.model.embed_into(
                    state.last_tokens[slot_idx],
                    &mut state.embed_buf[offset..offset + state.hidden_dim],
                );
            }
        }

        timer.mark("embed");

        // PMAT-086: Reuse pre-allocated position buffer (no Vec alloc per step)
        for (i, &p) in state.positions.iter().enumerate() {
            state.pos_buf[i] = p as u32;
        }

        if state.gen_idx < 3 {
            eprintln!(
                "[PMAT-072] decode step {}: positions={:?}, last_tokens={:?}",
                state.gen_idx, &state.pos_buf[..state.m], state.last_tokens
            );
        }

        // PMAT-076: Set dead slot mask before forward pass so attention kernel
        // can skip KV iteration for done slots (seq_lens=0 → early exit).
        self.executor.batched_done_mask.clone_from(&state.done);

        timer.mark("prep");

        // PMAT-056: Multi-stream root cause fixed (scatter moved to self.stream),
        // but graph replay still 25% slower than eager due to capture overhead.
        // Keep eager by default until graph replay is optimized.
        // Enable with BATCHED_GRAPH=1 for testing.
        let use_graph = std::env::var("BATCHED_GRAPH").as_deref() == Ok("1");

        // PMAT-764: if any live request asked for stochastic sampling (temperature > 0),
        // download per-slot logits and sample each by its OWN config (temperature/top_k) —
        // the batched path otherwise forces on-GPU greedy argmax for EVERY request, silently
        // ignoring temperature/top_k/seed. All-greedy batches keep the faster on-GPU argmax.
        let any_sampling = state
            .configs
            .iter()
            .take(state.m)
            .any(|c| c.temperature > 0.0);

        let token_ids: Vec<u32> = if any_sampling {
            let vocab = state.vocab_size;
            let logits = self
                .executor
                .forward_batched_to_logits(
                    &state.embed_buf,
                    &state.pos_buf,
                    state.num_layers,
                    state.hidden_dim as u32,
                    state.intermediate_dim as u32,
                    vocab as u32,
                    state.eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_batched_to_logits".to_string(),
                    reason: format!("Batched forward failed: {e}"),
                })?;
            (0..state.m)
                .map(|slot| {
                    let cfg = &state.configs[slot];
                    let base = slot * vocab;
                    let slot_logits = &logits[base..base + vocab];
                    select_batched_token(slot_logits, cfg.temperature, cfg.top_k, cfg.top_p)
                })
                .collect()
        } else if use_graph {
            self.executor
                .forward_batched_to_token_ids_graphed(
                    &state.embed_buf,
                    &state.pos_buf,
                    state.num_layers,
                    state.hidden_dim as u32,
                    state.intermediate_dim as u32,
                    state.vocab_size as u32,
                    state.eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_batched_to_token_ids_graphed".to_string(),
                    reason: format!("Batched forward failed: {e}"),
                })?
        } else {
            self.executor
                .forward_batched_to_token_ids(
                    &state.embed_buf,
                    &state.pos_buf,
                    state.num_layers,
                    state.hidden_dim as u32,
                    state.intermediate_dim as u32,
                    state.vocab_size as u32,
                    state.eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_batched_to_token_ids".to_string(),
                    reason: format!("Batched forward failed: {e}"),
                })?
        };

        timer.mark("fwd+sync+argmax");
        timer.emit(state.gen_idx as u64, state.m);

        if state.gen_idx < 3 {
            eprintln!(
                "[PMAT-072] decode step {}: token_ids={token_ids:?}, done={:?}",
                state.gen_idx, state.done
            );
        }

        // FALSIFY-CB-008 evidence, every step rather than the first three.
        if batched_decode_trace() {
            eprintln!(
                "[CB-008] step={} m={} token_ids={token_ids:?} positions={:?} done={:?}",
                state.gen_idx,
                state.m,
                &state.pos_buf[..state.m],
                state.done
            );
        }

        Ok(token_ids)
    }

    /// PMAT-072: Cleanup after step-wise batched generation.
    ///
    /// PMAT-075: Preserves workspace AND KV cache buffer addresses for CUDA
    /// graph reuse across batches. Only resets logical state (batch_size=1,
    /// kv_stride=0) without freeing GPU buffers.
    ///
    /// VRAM budget: FP16(2944)+KV(896)+Q4K(850)+WS(40) = 4730 MB fits 7.5 GB.
    ///
    /// Caller must hold model write lock for the duration of this call.
    /// PMAT-765: reset batched logical state after an ABORTED setup/prefill — before any
    /// `BatchedDecodeState` exists (so the full `batched_cleanup` can't run).
    /// `batched_setup_and_prefill` may have already called `init_batched_kv_cache_gpu` (which
    /// sets `batched_kv_stride != 0`) before failing; if the error path returns WITHOUT
    /// resetting it, the next batch reuses buffers (PMAT-075 high-water mark) with a STALE
    /// stride and prefill/scatter writes K/V to the wrong slots → cross-request KV corruption.
    /// Mirrors the stride + decode-graph resets that `batched_cleanup` performs.
    pub fn reset_batched_state(&mut self) {
        self.executor.batched_kv_stride = 0;
        self.executor.clear_decode_graph();
    }

    /// PMAT-072: Cleanup after step-wise batched generation. Preserves workspace + batched KV
    /// buffers (PMAT-075 high-water mark for graph reuse) and resets the batched logical state
    /// (batched_kv_stride, decode graph) so the next batch starts clean.
    pub fn batched_cleanup(&mut self, state: &BatchedDecodeState) {
        // PMAT-061: Disable HGEMM batched decode flag before cleanup.
        self.executor.hgemm_batched_decode_active = false;

        // PMAT-075: Keep workspace buffers alive for graph reuse across batches.
        // PAR-200: init_workspace sets batch_size=1 without reallocation when
        // buffer_capacity >= 1 (preserves GPU buffer addresses → cached batched
        // decode graphs remain valid). Previously clear_workspace() forced
        // reallocation with new addresses → stale graph pointers → ILLEGAL_ADDRESS.
        let _ = self
            .executor
            .init_workspace(state.hidden_dim, state.intermediate_dim);

        // PMAT-044 FIX: Reset batched KV stride so subsequent M=1 prefill doesn't
        // take the batched attention path (batched_ffn.rs line 37). Without this,
        // M=1 prefill writes K/V to batched caches while M=1 decode reads from the
        // single KV cache (empty) → EOS on first token → 0 output tokens.
        // Note: batched_kv_stride is restored by init_batched_kv_cache_gpu on next batch.
        self.executor.batched_kv_stride = 0;

        // PMAT-075: Keep batched KV caches alive for graph reuse.
        // The captured batched decode graph holds pointers to KV cache buffers
        // (k_ptrs_per_layer, v_ptrs_per_layer, seq_lens_gpu). Freeing them and
        // reallocating in the next batch gives new addresses → stale graph
        // pointers → CUDA_ERROR_ILLEGAL_ADDRESS on graph replay.
        // init_batched_kv_cache_gpu has high-water mark (batched_kv_allocated_batch)
        // and skips reallocation when capacity is sufficient.
        // VRAM cost: ~896 MB retained between batches (fits in 7.5 GB budget).
        // reset_batched_kv_cache_gpu zeroes contents at next batch start.

        // PMAT-088: Clear M=1 decode graph after batched decode. Batched decode may have
        // resized workspace.logits_buf to M*vocab_size, invalidating the M=1 graph's
        // captured logits pointer. Without this, M=1 generate_gpu_resident_streaming
        // replays a graph with stale buffer pointers → "Length mismatch" error.
        self.executor.clear_decode_graph();

        // PMAT-061: Weight cache is now kept during decode (not cleared after prefill).
        // PMAT-067: Prefer FP8 cache on sm_89+ — skip FP16 cache rebuild.
        if self.executor.gpu_profile.fp8_prefill {
            // FP8 cache handles its own lazy population — no rebuild needed.
        } else if !self.executor.has_fp16_weight_cache() {
            // FP16 was never populated (HGEMM_PREFILL=0) — rebuild for c=1 HGEMM prefill.
            let _ = self.executor.warmup_hgemm_cache(
                state.num_layers,
                state.hidden_dim as u32,
                state.intermediate_dim as u32,
                state.vocab_size as u32,
            );
        }
    }

    /// PMAT-073: Add a new request to a running batch mid-generation.
    ///
    /// Prefills the new slot using M=1 forward, scatters KV to the batched cache
    /// at the next available slot index, and extends the decode state.
    ///
    /// Caller must hold model write lock for the duration of this call.
    /// The prefill takes ~40-130ms depending on prompt length (one-time cost
    /// that briefly increases ITL for existing slots).
    pub fn add_slot_to_batch(
        &mut self,
        state: &mut BatchedDecodeState,
        prompt: Vec<u32>,
        config: QuantizedGenerateConfig,
        on_token: Box<dyn FnMut(u32) -> bool + Send>,
    ) -> Result<()> {
        let new_slot = state.m;
        if new_slot >= state.max_kv_slots {
            return Err(RealizarError::UnsupportedOperation {
                operation: "add_slot_to_batch".to_string(),
                reason: format!(
                    "PMAT-073: batch full (m={}, max_kv_slots={})",
                    state.m, state.max_kv_slots
                ),
            });
        }

        let seq_len = prompt.len().saturating_sub(1);
        let prefill_start = std::time::Instant::now();

        // Clear M=1 decode graph — prefill uses different kernel config.
        // PMAT-075: Do NOT clear batched decode graphs unconditionally.
        // M is changing (growing), but the old M's graph entry stays in the
        // HashMap (unused). The new M will trigger a capture on first use.
        // If workspace reallocation is needed, init_prefill_workspace clears
        // batched graphs at the source (poka-yoke).
        self.executor.clear_decode_graph();

        // Init workspace for prefill (high-water mark, usually no realloc)
        self.executor
            .init_prefill_workspace(seq_len, state.hidden_dim, state.intermediate_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "add_slot_prefill_workspace".to_string(),
                reason: format!("PMAT-073: workspace init for slot {new_slot}: {e}"),
            })?;

        // Prefill using M=1 path (writes to single-slot KV cache)
        let kv_dim = self.model.config.num_kv_heads
            * (self.model.config.hidden_dim / self.model.config.num_heads);
        let max_seq_len = seq_len + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::new(state.num_layers, kv_dim, max_seq_len);

        let saved_stride = self.executor.batched_kv_stride;
        self.executor.batched_kv_stride = 0;
        self.executor.reset_kv_cache_gpu();
        self.run_prefill(&prompt, &mut cache, seq_len, false, false)?;
        self.executor.batched_kv_stride = saved_stride;

        // Scatter KV from single-slot cache to batched cache at new_slot
        self.executor
            .scatter_single_kv_to_batched(new_slot, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "scatter_single_kv_to_batched".to_string(),
                reason: format!("PMAT-073: scatter for slot {new_slot}: {e}"),
            })?;

        // Switch back to batched decode mode with M+1
        let new_m = new_slot + 1;
        self.executor.clear_prefill_graphs();
        self.executor
            .init_batched_workspace(state.hidden_dim, state.intermediate_dim, new_m)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_workspace".to_string(),
                reason: format!("PMAT-073: workspace resize for M={new_m}: {e}"),
            })?;
        self.executor.clear_decode_graph();

        // Update state
        let last_token = prompt[prompt.len() - 1];
        state.sequences.push(prompt.clone());
        state.prompts.push(prompt);
        state.configs.push(config);
        state.on_tokens.push(on_token);
        state.positions.push(seq_len);
        state.last_tokens.push(last_token);
        state.done.push(false);
        state.m = new_m;
        state.embed_buf.resize(new_m * state.hidden_dim, 0.0);
        state.pos_buf.resize(new_m, 0);
        // PMAT-765: a slot joining mid-batch at step `gen_idx` with `max_tokens` M must run
        // until gen_idx + M, else it terminates ~gen_idx steps early. recycle_slot and
        // recycle_slots_batch already add the gen_idx offset; add_slot_to_batch did not.
        state.max_tokens_max = state.max_tokens_max.max(
            state.gen_idx
                + state
                    .configs
                    .last()
                    .expect("configs must not be empty when updating max_tokens")
                    .max_tokens,
        );

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[PMAT-073] Mid-batch join: slot {} added (now M={}), prefill {:.1}ms ({} tokens)",
            new_slot, new_m, prefill_ms, seq_len,
        );

        Ok(())
    }

    /// PMAT-074: Recycle a finished slot for a new request.
    ///
    /// Similar to `add_slot_to_batch` but reuses an existing slot index instead
    /// of appending. The slot's KV cache is overwritten with the new request's
    /// prefill data. `state.m` stays the same (no growth).
    ///
    /// Caller must hold model write lock for the duration of this call.
    pub fn recycle_slot(
        &mut self,
        state: &mut BatchedDecodeState,
        slot_idx: usize,
        prompt: Vec<u32>,
        config: QuantizedGenerateConfig,
        on_token: Box<dyn FnMut(u32) -> bool + Send>,
    ) -> Result<()> {
        if !state.done[slot_idx] {
            return Err(RealizarError::UnsupportedOperation {
                operation: "recycle_slot".to_string(),
                reason: format!("PMAT-074: slot {slot_idx} is not done"),
            });
        }

        let seq_len = prompt.len().saturating_sub(1);
        let prefill_start = std::time::Instant::now();

        // Clear M=1 decode graph — prefill uses different kernel config.
        // PMAT-075: Do NOT clear batched decode graphs. M is unchanged and
        // workspace addresses remain stable (init_prefill_workspace skips
        // realloc when buffer_capacity >= seq_len). If realloc IS needed,
        // init_prefill_workspace clears batched graphs at the source.
        self.executor.clear_decode_graph();

        // Init workspace for prefill (high-water mark, usually no realloc)
        self.executor
            .init_prefill_workspace(seq_len, state.hidden_dim, state.intermediate_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "recycle_slot_prefill_workspace".to_string(),
                reason: format!("PMAT-074: workspace init for slot {slot_idx}: {e}"),
            })?;

        // Prefill using M=1 path (writes to single-slot KV cache)
        let kv_dim = self.model.config.num_kv_heads
            * (self.model.config.hidden_dim / self.model.config.num_heads);
        let max_seq_len = seq_len + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::new(state.num_layers, kv_dim, max_seq_len);

        let saved_stride = self.executor.batched_kv_stride;
        self.executor.batched_kv_stride = 0;
        self.executor.reset_kv_cache_gpu();
        self.run_prefill(&prompt, &mut cache, seq_len, false, false)?;
        self.executor.batched_kv_stride = saved_stride;

        // Scatter KV to the RECYCLED slot index (overwrites old data)
        self.executor
            .scatter_single_kv_to_batched(slot_idx, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "scatter_single_kv_to_batched".to_string(),
                reason: format!("PMAT-074: scatter for recycled slot {slot_idx}: {e}"),
            })?;

        // Restore batched decode workspace (M unchanged)
        self.executor.clear_prefill_graphs();
        self.executor
            .init_batched_workspace(state.hidden_dim, state.intermediate_dim, state.m)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_workspace".to_string(),
                reason: format!("PMAT-074: workspace restore for M={}: {e}", state.m),
            })?;
        self.executor.clear_decode_graph();

        // Replace state at recycled slot index
        let last_token = prompt[prompt.len() - 1];
        state.sequences[slot_idx].clone_from(&prompt);
        state.prompts[slot_idx] = prompt;
        state.configs[slot_idx] = config;
        state.on_tokens[slot_idx] = on_token;
        state.positions[slot_idx] = seq_len;
        state.last_tokens[slot_idx] = last_token;
        state.done[slot_idx] = false;

        // Extend max_tokens_max to cover gen_idx + new request's max_tokens
        state.max_tokens_max = state
            .max_tokens_max
            .max(state.gen_idx + state.configs[slot_idx].max_tokens);

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[PMAT-074] Slot recycled: slot {} reused (M={}), prefill {:.1}ms ({} tokens)",
            slot_idx, state.m, prefill_ms, seq_len,
        );

        Ok(())
    }

    /// PMAT-088d: Batch-recycle multiple done slots using multi-prompt prefill.
    ///
    /// Processes all recycled prompts in a single `prefill_multi_prompt` call
    /// (~14ms total regardless of count) instead of N sequential `recycle_slot`
    /// calls (N × 17ms). KV is scattered directly to target batched cache slots.
    ///
    /// Caller must hold model write lock for the duration of this call.
    pub fn recycle_slots_batch(
        &mut self,
        state: &mut BatchedDecodeState,
        slots_and_requests: Vec<(
            usize,
            Vec<u32>,
            QuantizedGenerateConfig,
            Box<dyn FnMut(u32) -> bool + Send>,
        )>,
    ) -> Result<()> {
        if slots_and_requests.is_empty() {
            return Ok(());
        }

        // PMAT-088d: Always use prefill_multi_prompt path, even for N=1.
        // This is faster than recycle_slot because:
        // 1. KV scatter integrated into attention (no separate D2D copy, saves ~1ms)
        // 2. No force_workspace_reinit (avoids CUDA malloc churn, saves ~0.5ms)
        // 3. No CPU KV cache allocation (saves ~0.5ms)
        // Net: ~17ms → ~15ms per recycle for N=1.

        // Validate all target slots are done
        for (slot_idx, _, _, _) in &slots_and_requests {
            if *slot_idx >= state.m || !state.done[*slot_idx] {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "recycle_slots_batch".to_string(),
                    reason: format!(
                        "PMAT-088d: slot {} not done or out of range (m={})",
                        slot_idx, state.m
                    ),
                });
            }
        }

        let prefill_start = std::time::Instant::now();
        let num_recycles = slots_and_requests.len();

        // Build multi-prompt inputs
        let slot_indices: Vec<usize> = slots_and_requests.iter().map(|(s, _, _, _)| *s).collect();
        let prompt_lengths: Vec<usize> = slots_and_requests
            .iter()
            .map(|(_, p, _, _)| p.len().saturating_sub(1))
            .collect();
        let m_total: usize = prompt_lengths.iter().sum();
        let prompt_offsets: Vec<usize> = prompt_lengths
            .iter()
            .scan(0, |acc, &len| {
                let off = *acc;
                *acc += len;
                Some(off)
            })
            .collect();

        // Clear decode graph for prefill
        self.executor.clear_decode_graph();

        // Init workspace for M_total tokens (high-water mark, may realloc)
        self.executor
            .init_prefill_workspace(m_total, state.hidden_dim, state.intermediate_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "recycle_slots_batch_workspace".to_string(),
                reason: format!("PMAT-088d: workspace init for M_total={m_total}: {e}"),
            })?;

        // Embed all prompts concatenated
        let hidden_dim = state.hidden_dim;
        let mut all_embeddings = Vec::with_capacity(m_total * hidden_dim);
        for (i, (_, prompt, _, _)) in slots_and_requests.iter().enumerate() {
            let embeddings = self.model.embed(&prompt[..prompt_lengths[i]]);
            all_embeddings.extend_from_slice(&embeddings);
        }

        // Build concatenated positions: [0..S0-1, 0..S1-1, ...]
        let mut all_positions = Vec::with_capacity(m_total);
        for &len in &prompt_lengths {
            all_positions.extend(0..len as u32);
        }

        // Initialize cuBLAS for multi-prompt prefill
        self.executor.ensure_cublas().map_err(|e| RealizarError::UnsupportedOperation {
            operation: "recycle_slots_batch_cublas".to_string(),
            reason: format!("PMAT-088d: cuBLAS init failed: {e}"),
        })?;

        // Run multi-prompt prefill with slot mapping — weights read ONCE,
        // KV scattered directly to target batched cache slots
        self.executor
            .prefill_multi_prompt(
                &all_embeddings,
                &all_positions,
                &prompt_offsets,
                &prompt_lengths,
                state.num_layers,
                state.hidden_dim as u32,
                state.intermediate_dim as u32,
                state.eps,
                Some(&slot_indices),
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "recycle_slots_batch_prefill".to_string(),
                reason: format!("PMAT-088d: multi-prompt prefill failed: {e}"),
            })?;

        // Restore decode workspace
        self.executor.clear_prefill_graphs();
        self.executor
            .init_batched_workspace(state.hidden_dim, state.intermediate_dim, state.m)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "recycle_slots_batch_workspace_restore".to_string(),
                reason: format!("PMAT-088d: workspace restore for M={}: {e}", state.m),
            })?;
        self.executor.clear_decode_graph();

        // Update state for each recycled slot
        for (slot_idx, prompt, config, on_token) in slots_and_requests {
            let last_token = prompt[prompt.len() - 1];
            let seq_len = prompt.len().saturating_sub(1);
            state.sequences[slot_idx].clone_from(&prompt);
            state.prompts[slot_idx] = prompt;
            state.configs[slot_idx] = config;
            state.on_tokens[slot_idx] = on_token;
            state.positions[slot_idx] = seq_len;
            state.last_tokens[slot_idx] = last_token;
            state.done[slot_idx] = false;
            state.max_tokens_max = state
                .max_tokens_max
                .max(state.gen_idx + state.configs[slot_idx].max_tokens);
        }

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[PMAT-088d] Batch recycled: {} slots {:?} (M={}), prefill {:.1}ms ({} total tokens)",
            num_recycles, slot_indices, state.m, prefill_ms, m_total,
        );

        Ok(())
    }

    fn validate_batch_args(&self, m: usize, configs_len: usize, callbacks_len: usize) -> Result<()> {
        if m > 32 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batched_streaming".to_string(),
                reason: format!("PMAT-044: batch size {m} exceeds max 32"),
            });
        }
        if configs_len != m || callbacks_len != m {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batched_streaming".to_string(),
                reason: format!(
                    "PMAT-044: prompts({m})/configs({configs_len})/callbacks({callbacks_len}) length mismatch"
                ),
            });
        }
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batched_streaming".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }
        Ok(())
    }

    fn prefill_and_scatter(
        &mut self,
        prompts: &[Vec<u32>],
        caches: &mut [OwnedQuantizedKVCache],
    ) -> Result<(Vec<Vec<u32>>, Vec<usize>, Vec<u32>)> {
        let m = prompts.len();
        let sequences: Vec<Vec<u32>> = prompts.to_vec();
        let mut positions: Vec<usize> = Vec::with_capacity(m);
        let mut last_tokens: Vec<u32> = Vec::with_capacity(m);

        let prefill_start = std::time::Instant::now();

        // PMAT-051: Multi-prompt batched prefill — read weights once for all prompts.
        // Concatenate all prompts' tokens, run GEMM once per layer (M_total tokens),
        // only attention is per-prompt (prompt-local causal mask).
        let use_multi_prompt = m > 1
            && std::env::var("MULTI_PROMPT_PREFILL").as_deref() != Ok("0");

        if use_multi_prompt {
            let hidden_dim = self.model.config.hidden_dim;
            let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
            let num_layers = self.model.config.num_layers;
            let eps = self.model.config.eps;

            // Calculate per-prompt lengths (prefill_count = prompt.len() - 1)
            let prompt_lengths: Vec<usize> = prompts.iter()
                .map(|p| p.len().saturating_sub(1))
                .collect();
            let m_total: usize = prompt_lengths.iter().sum();

            // Build offsets into the packed buffer
            let prompt_offsets: Vec<usize> = prompt_lengths.iter()
                .scan(0, |acc, &len| { let off = *acc; *acc += len; Some(off) })
                .collect();

            // Embed all prompts, concatenate into packed [M_total × hidden_dim]
            let mut all_embeddings = Vec::with_capacity(m_total * hidden_dim);
            for (i, prompt) in prompts.iter().enumerate() {
                let embeddings = self.model.embed(&prompt[..prompt_lengths[i]]);
                all_embeddings.extend_from_slice(&embeddings);
            }

            // Build concatenated positions: [0..S0-1, 0..S1-1, ...]
            let mut all_positions = Vec::with_capacity(m_total);
            for &len in &prompt_lengths {
                all_positions.extend(0..len as u32);
            }

            // Ensure workspace sized for M_total
            self.executor
                .init_prefill_workspace(m_total, hidden_dim, intermediate_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_prefill_workspace".to_string(),
                    reason: format!("PMAT-051: workspace init failed for M_total={m_total}: {e}"),
                })?;

            eprintln!(
                "[PMAT-051] Multi-prompt prefill: {} prompts, M_total={}, reading weights once...",
                m, m_total
            );

            // Run multi-prompt prefill — weights read once, per-prompt attention
            self.executor
                .prefill_multi_prompt(
                    &all_embeddings,
                    &all_positions,
                    &prompt_offsets,
                    &prompt_lengths,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    eps,
                    None, // PMAT-088d: initial batch uses slot_idx == prompt_idx
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "prefill_multi_prompt".to_string(),
                    reason: format!("PMAT-051: multi-prompt prefill failed: {e}"),
                })?;

            // Build return values
            for (i, prompt) in prompts.iter().enumerate() {
                positions.push(prompt_lengths[i]);
                last_tokens.push(prompt[prompt.len() - 1]);
            }
        } else {
            // Sequential fallback (M=1 or MULTI_PROMPT_PREFILL=0)
            let saved_stride = self.executor.batched_kv_stride;
            eprintln!("[PMAT-044] Batched streaming: {m} slots, prefilling sequentially...");

            for (slot_idx, prompt) in prompts.iter().enumerate() {
                self.executor.reset_kv_cache_gpu();
                let seq_len = prompt.len().saturating_sub(1);

                self.executor.batched_kv_stride = 0;
                self.run_prefill(prompt, &mut caches[slot_idx], seq_len, false, false)?;

                self.executor.batched_kv_stride = saved_stride;
                self.executor
                    .scatter_single_kv_to_batched(slot_idx, seq_len)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "scatter_single_kv_to_batched".to_string(),
                        reason: format!("Failed to scatter KV for slot {slot_idx}: {e}"),
                    })?;

                positions.push(seq_len);
                last_tokens.push(prompt[prompt.len() - 1]);
            }
        }

        let prefill_elapsed = prefill_start.elapsed();
        eprintln!(
            "[PMAT-051] Prefill done in {:.1}ms ({:.0} tok/s), starting batched decode...",
            prefill_elapsed.as_secs_f64() * 1000.0,
            prompts.iter().map(|p| p.len() as f64).sum::<f64>() / prefill_elapsed.as_secs_f64(),
        );
        Ok((sequences, positions, last_tokens))
    }
}

#[cfg(test)]
mod pmat764_select_batched_token_tests {
    use super::select_batched_token;

    #[test]
    fn temperature_zero_is_greedy_argmax() {
        // temp==0 must pick the max-logit index (deterministic), NOT sample.
        assert_eq!(select_batched_token(&[1.0, 5.0, 2.0], 0.0, 40, 1.0), 1);
        assert_eq!(select_batched_token(&[9.0, 5.0, 2.0], 0.0, 40, 1.0), 0);
    }

    #[test]
    fn top_k_one_is_greedy_at_any_temperature() {
        // top_k==1 keeps only the single best token → deterministic argmax even when
        // temperature > 0 (the sampling branch with a 1-token nucleus).
        assert_eq!(select_batched_token(&[1.0, 5.0, 2.0], 1.5, 1, 1.0), 1);
        assert_eq!(select_batched_token(&[9.0, 5.0, 2.0], 0.9, 1, 1.0), 0);
    }

    #[test]
    fn temperature_zero_is_greedy_at_any_top_p() {
        // temp==0 must stay deterministic argmax regardless of top_p (greedy short-circuit).
        assert_eq!(select_batched_token(&[1.0, 5.0, 2.0], 0.0, 40, 0.1), 1);
        assert_eq!(select_batched_token(&[9.0, 5.0, 2.0], 0.0, 40, 0.9), 0);
    }

    #[test]
    fn top_p_restricts_nucleus_to_dominant_token() {
        // PMAT-792b FALSIFIER (the load-bearing test). Logits [3.0, 0, 0, 0] at temp 1.0
        // softmax to P(token0) ≈ 0.870, P(each other) ≈ 0.043. With top_p = 0.85 the nucleus
        // (smallest prefix with cumulative prob >= top_p) is exactly {token0} (0.870 >= 0.85),
        // so EVERY draw must return index 0. top_k = 4 keeps the whole vocabulary, so the only
        // thing that can restrict the sample is the per-request top_p.
        //
        // BEFORE the fix, select_batched_token called sample_topk, which silently dropped top_p:
        // the full 4-way distribution stayed samplable, P(non-token0) ≈ 0.13 per draw, and over
        // 512 draws a non-zero token is effectively certain (0.87^512 ≈ 0) → this loop fails.
        // AFTER the fix, the nucleus collapses to {token0} and every draw is deterministic.
        let logits = [3.0_f32, 0.0, 0.0, 0.0];
        for _ in 0..512 {
            let tok = select_batched_token(&logits, 1.0, 4, 0.85);
            assert_eq!(
                tok, 0,
                "top_p=0.85 must collapse the nucleus to the dominant token 0, got {tok} \
                 (top_p silently dropped → low-prob token leaked through)"
            );
        }
    }

    #[test]
    fn top_p_one_is_a_noop_nucleus() {
        // top_p==1.0 (and >1.0) skips the nucleus cutoff → identical to the pure top_k path.
        // top_k==1 always returns the argmax, verifying the no-regression equivalence.
        assert_eq!(select_batched_token(&[1.0, 5.0, 2.0], 1.5, 1, 1.0), 1);
        assert_eq!(select_batched_token(&[1.0, 5.0, 2.0], 1.5, 1, 2.0), 1);
    }
}

include!("cb006_parity_probe.rs");
