impl OwnedQuantizedModelCuda {
    /// Generate tokens using CUDA acceleration
    ///
    /// Uses `forward_cuda` for each token generation step.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration (max_tokens, temperature, etc.)
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_cuda(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        let mut tokens = prompt.to_vec();

        for _ in 0..config.max_tokens {
            // aprender#2376(3): CANCELLATION POLL. The HTTP client may be gone;
            // stop here instead of burning a core to max_tokens for nobody.
            if config.cancel.is_cancelled() {
                break;
            }
            let logits = self.forward_cuda(&tokens)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);

                // Apply temperature and sample (simplified - take max after temperature)
                let max_logit = indexed[0].1;
                let _exp_sum: f32 = indexed
                    .iter()
                    .map(|(_, l)| ((l - max_logit) / config.temperature).exp())
                    .sum();

                // Take argmax (proper probabilistic sampling would use exp_sum for normalization)
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Generate tokens using CUDA with KV cache
    ///
    /// Uses `forward_single_cuda_with_cache` for incremental decoding with KV cache.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.head_dim();
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        let mut tokens = prompt.to_vec();

        // Process prompt tokens
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                // Just populate the cache
                let _ = self.forward_single_cuda_with_cache(token_id, &mut cache, pos)?;
            }
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _ in 0..config.max_tokens {
            // aprender#2376(3): CANCELLATION POLL. The HTTP client may be gone;
            // stop here instead of burning a core to max_tokens for nobody.
            if config.cancel.is_cancelled() {
                break;
            }
            let logits = self.forward_single_cuda_with_cache(last_token, &mut cache, position)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// IMP-1010: Full GPU-accelerated token generation
    ///
    /// Uses `forward_single_full_cuda_with_cache` for maximum GPU utilization.
    /// All matmul operations (5 per layer) run on GPU.
    ///
    /// # Performance Target
    ///
    /// - CPU path: ~5 tok/s (limited by memory bandwidth)
    /// - Full GPU path: ~200 tok/s (matching Ollama)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_full_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.head_dim();
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill) - use full GPU path
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                // Just populate the cache
                let _ = self.forward_single_full_cuda_with_cache(token_id, &mut cache, pos)?;
            }
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _ in 0..config.max_tokens {
            // aprender#2376(3): CANCELLATION POLL. The HTTP client may be gone;
            // stop here instead of burning a core to max_tokens for nobody.
            if config.cancel.is_cancelled() {
                break;
            }
            let logits =
                self.forward_single_full_cuda_with_cache(last_token, &mut cache, position)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// Run prefill phase: process prompt tokens through all layers to populate KV cache.
    ///
    /// GH-94: Batched prefill is now default (2x throughput vs serial).
    /// Set `BATCHED_PREFILL=0` for serial fallback.
    fn run_prefill(
        &mut self,
        prompt: &[u32],
        cache: &mut OwnedQuantizedKVCache,
        prefill_count: usize,
        trace: bool,
        extract_first_token: bool,
    ) -> Result<Option<u32>> {
        if prefill_count == 0 {
            // No prefill PHASE ran. `None`, not `Some(0.0)`: a zero would be
            // read as "prefill was instantaneous" and would enter a ratio.
            self.last_phase_timings.prefill_ms = None;
            if trace {
                eprintln!("[TRACE-PREFILL] Single token prompt, no prefill needed");
            }
            return Ok(None);
        }

        // GH-94: Batched prefill is default (36% throughput improvement).
        // Set BATCHED_PREFILL=0 to disable (serial fallback).
        //
        // PMAT-810 (Blackwell batched-prefill KV-cache corruption): on Blackwell
        // (cc>=120, e.g. GB10 sm_121) the batched prefill path writes a CORRUPT
        // KV cache. The extracted first token is ~correct (near-tie vs CPU), but
        // every subsequent decode step reads poisoned K/V and the output collapses
        // to a single repeated token (measured: CPU/serial-prefill emit
        // "Certainly! Below is a Rust function that", batched prefill emits
        // "CertainlyCertainlyCertainly..."). The decode-path parity probe
        // (PMAT-806 / F2-VALIDATION) accepts the model and the corruption ships
        // silently. NOTE the reason is not the one this comment used to give:
        // since PMAT-919 the probe checks EVERY prompt position, not just the
        // first. It still cannot see this bug for a different and more basic
        // reason - it executes ZERO decode steps and never calls run_prefill at
        // all, so it exercises neither the batched prefill kernels nor any
        // cached decode. Extending it is PMAT-F2-DECODE-PHASE-001. The bug is
        // structural to
        // batched prefill on Blackwell (it reproduces with FP8_PREFILL=0 / HGEMM
        // too, so it is NOT the PMAT-806 activation-quant outlier), while serial
        // prefill (per-token forward_gpu_resident, the PMAT-806 fp32-MWV decode
        // GEMV) is byte-for-byte correct on-device. Default Blackwell to serial
        // prefill so quantized models stay coherent; discrete GPUs (sm_89 etc.)
        // keep the fast batched path unchanged. Explicit BATCHED_PREFILL=1 still
        // forces batched for A/B testing the deeper KV-scatter root-cause fix.
        // Contract: contracts/apr-cpu-vs-gpu-output-parity-v1.yaml (FALSIFY-CPU-GPU-009).
        //
        // PP-LLAMA-001 §9 #1: the predicate is no longer inline. It lives in
        // `gpu_profile::select_prefill_path`, is resolved ONCE into the
        // profile, and is the same answer `/v1/effective-config` reports and
        // the multi-prompt guard in `generate_batched_streaming` enforces —
        // so the endpoint cannot say `batched` about a serial run.
        let choice = self.executor.gpu_profile.prefill_path();
        let use_batched = choice.path == crate::cuda::gpu_profile::PrefillPath::Batched;
        announce_prefill_path(choice);

        let prefill_start = std::time::Instant::now();

        if !use_batched {
            for (pos, &token_id) in prompt.iter().enumerate().take(prefill_count) {
                let _ = self.forward_gpu_resident(token_id, cache, pos)?;
            }
            let elapsed = prefill_start.elapsed();
            self.last_phase_timings.prefill_ms = Some(elapsed.as_secs_f64() * 1000.0);
            if trace {
                eprintln!(
                    "[TRACE-PREFILL] Serial prefill: {} tokens in {:?}",
                    prefill_count, elapsed
                );
            }
            return Ok(None);
        }

        // GH-94: Batched prefill (default path)
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.config.num_layers;
        let vocab_size = self.model.config.vocab_size;
        let eps = self.model.config.eps;

        let embeddings = self.model.embed(&prompt[..prefill_count]);
        let positions: Vec<u32> = (0..prefill_count as u32).collect();

        self.executor
            .init_prefill_workspace(prefill_count, hidden_dim, intermediate_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_prefill_workspace".to_string(),
                reason: format!("Prefill workspace init failed: {e}"),
            })?;
        self.executor
            .prefill_all_layers_gpu(
                &embeddings,
                &positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                eps,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "prefill_all_layers_gpu".to_string(),
                reason: format!("Batched prefill failed: {e}"),
            })?;

        // PMAT-083: Extract first predicted token from prefill hidden state.
        // Runs output RMSNorm + LM head GEMV + GPU argmax on the last position.
        // This eliminates the separate first decode step (~7ms savings).
        // Must happen BEFORE force_workspace_reinit (hidden_buf2 still valid).
        let first_token = if extract_first_token {
            let token = self
                .executor
                .prefill_extract_first_token(
                    prefill_count - 1, // last position index
                    hidden_dim as u32,
                    vocab_size as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "prefill_extract_first_token".to_string(),
                    reason: format!("PMAT-083 first token extraction failed: {e}"),
                })?;
            Some(token)
        } else {
            None
        };

        // CORRECTNESS-016: Log KV cache fingerprint after batched prefill.
        // Non-destructive: just reads the KV cache, no serial comparison.
        // Compare fingerprints across requests to detect non-determinism.
        static KV_FINGERPRINT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *KV_FINGERPRINT.get_or_init(|| std::env::var("KV_FINGERPRINT").as_deref() == Ok("1")) {
            if let Ok(sums) = self.executor.kv_cache_l0_k_fingerprint(prefill_count) {
                // Compute a single hash-like value: sum of sums
                let total: f32 = sums.iter().sum();
                // Also report first 4 and last 4 position sums for pattern matching
                let all: Vec<String> = sums.iter().map(|s| format!("{:.2}", s)).collect();
                eprintln!(
                    "[KV-FP] total={:.4} all=[{}] S={}",
                    total,
                    all.join(","),
                    prefill_count
                );
            }
        }

        // PMAT-109: Skip force_workspace_reinit — let PAR-200 preserve buffer addresses.
        // CORRECTNESS-015 forced reallocation after every prefill, destroying the CUDA
        // decode graph (stale pointers). But init_prefill_workspace already clears the
        // graph when it actually reallocates (longer prompt exceeds buffer_capacity).
        // When PAR-200 fires (same prompt length), buffers are stable → graph persists
        // → no cuGraphExecDestroy per request → eliminates bimodal TTFT tail.
        // Replaces: self.executor.force_workspace_reinit();
        self.executor
            .init_workspace(hidden_dim, intermediate_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_workspace".to_string(),
                reason: format!("Workspace restore failed: {e}"),
            })?;

        let elapsed = prefill_start.elapsed();
        // §3: the numerator of `prefill_tok_per_sec`. Recorded on `self` rather
        // than printed, because a stderr line is not on the wire and §7.2 gates
        // `prefill_ratio` at c=1 off a receipt field.
        self.last_phase_timings.prefill_ms = Some(elapsed.as_secs_f64() * 1000.0);
        if trace {
            eprintln!(
                "[TRACE-PREFILL] Batched prefill: {} tokens in {:?} ({:.1} tok/s){}",
                prefill_count,
                elapsed,
                prefill_count as f64 / elapsed.as_secs_f64(),
                if first_token.is_some() { " [+LM head]" } else { "" },
            );
        }
        Ok(first_token)
    }

    /// GPU-resident token generation with minimal CPU↔GPU transfers.
    ///
    /// # Reentrant
    ///
    /// This method creates fresh generation state on each call (new KV cache,
    /// reset GPU positions). It is safe and efficient to call multiple times
    /// on the same `OwnedQuantizedModelCuda` — weights are preloaded once
    /// during construction and reused across calls.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration (max_tokens, temperature, etc.)
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    /// #2697: put this prompt's KV on the device and say where decode starts.
    ///
    /// Returns `(position, last_token, max_decode, prefix_was_hit)`, or `None`
    /// when the prefill's own first token was a stop token and there is nothing
    /// left to generate.
    /// The non-streaming decode loop. Same shape as `decode_loop`, without a
    /// per-token callback.
    fn decode_blocking(
        &mut self,
        config: &QuantizedGenerateConfig,
        tokens: &mut Vec<u32>,
        cache: &mut OwnedQuantizedKVCache,
        mut position: usize,
        mut last_token: u32,
        max_decode: usize,
    ) -> Result<()> {
        let penalty_active = config.repeat_penalty != 1.0 && config.repeat_last_n > 0;
        for _token_num in 0..max_decode {
            let next_token = self.next_token(NextToken {
                config,
                tokens,
                cache,
                last_token,
                position,
                penalty_active,
            })?;
            if config.stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }
        Ok(())
    }

    /// realizr#194: the three things every GPU-resident entry point must check —
    /// KV capacity, a current CUDA context, and a supported architecture.
    fn check_gpu_resident_preconditions(&mut self, prompt: &[u32], op: &str) -> Result<()> {
        let gpu_max_len = self.executor.max_kv_len();
        let effective_max = if gpu_max_len > 0 {
            gpu_max_len.min(self.model.config.context_length)
        } else {
            self.model.config.context_length
        };
        if prompt.len() > effective_max {
            return Err(RealizarError::ContextLimitExceeded {
                provided: prompt.len(),
                maximum: effective_max,
            });
        }
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: op.to_string(),
                reason: "Architecture not supported for GPU-resident path".to_string(),
            });
        }
        Ok(())
    }

    fn establish_kv_blocking(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        cache: &mut OwnedQuantizedKVCache,
        tokens: &mut Vec<u32>,
    ) -> Result<Option<(usize, u32, usize, bool)>> {
        let position: usize;
        let last_token: u32;
        let max_decode: usize;
            // #2697: the same residency question the streaming path asks, on the
            // path my nsys profiles were actually exercising — 62 synchronous 4 MiB
            // host-to-device copies per request, to put back bytes already there.
            #[cfg(feature = "gpu")]
            // #2697 MEASURED REGRESSION — OFF BY DEFAULT.
        //
        // Skipping the restore saves 234 MB of host traffic and wins 2.36x on
        // TTFT when the host is STARVED (41.3 ms vs 97.6 ms at load average
        // 128). On a quiet box it LOSES, reproducibly, two interleaved rounds:
        //
        //     RESIDENCY=1   TTFT 41.18 / 40.96 ms   prefill 2477 / 2491
        //     RESIDENCY=0   TTFT 34.31 / 33.82 ms   prefill 2973 / 3016
        //
        // ~7 ms, which is almost exactly one decode step. That is the
        // mechanism: taking this path means `prefill_first_token = None`, so
        // decode starts at prompt.len()-1 and re-processes the last prompt
        // token, forgoing the fused first token prefill extracts from the LM
        // head (PMAT-083). Prefill over ~100 tokens is cheaper than that
        // whenever the host is not the bottleneck.
        //
        // So it ships OFF: opt in with APR_KV_RESIDENCY=1 on a host under
        // heavy CPU contention. Making it a win everywhere needs the first
        // token cached beside the prefix, which is follow-up work on #2697.
        let already_resident = std::env::var("APR_KV_RESIDENCY").as_deref() == Ok("1")
                && self.executor.kv_prefix_is_resident(
                    crate::gguf::batch_scheduler::PrefixCache::hash_tokens(prompt),
                    prompt.len(),
                );
            #[cfg(not(feature = "gpu"))]
            let already_resident = false;
            #[cfg(feature = "gpu")]
            let prefix_hit = if already_resident {
                None // skip the 234 MB clone entirely
            } else {
                self.prefix_cache.lookup(prompt)
            };
            #[cfg(not(feature = "gpu"))]
            let prefix_hit: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = None;
            #[cfg(feature = "gpu")]
            let prefix_was_hit = prefix_hit.is_some();
            #[cfg(not(feature = "gpu"))]
            let prefix_was_hit = false;

            let mut position;
            let mut last_token;
            let max_decode;

            if already_resident {
                // The bytes are already on the device; only the lengths carry the
                // previous request's generated tokens.
                #[cfg(feature = "gpu")]
                self.executor.truncate_kv_lengths(prompt.len());
                if config.trace {
                    eprintln!("[#2697] KV PREFIX RESIDENT: {} tokens, nothing to restore", prompt.len());
                }
                position = prompt.len() - 1;
                last_token = prompt[prompt.len() - 1];
                max_decode = config.max_tokens;
            } else if let Some((cached_k, cached_v)) = prefix_hit {
                let kv_pairs: Vec<(Vec<f32>, Vec<f32>)> = cached_k.into_iter().zip(cached_v).collect();
                let cached_len = prompt.len();
                self.executor
                    .restore_kv_cache_from_host(&kv_pairs, cached_len)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "restore_kv_cache_from_host".to_string(),
                        reason: format!("Prefix cache restore failed: {e}"),
                    })?;

                if config.trace {
                    eprintln!("[PMAT-450] Prefix cache HIT: skipped prefill for {} tokens", cached_len);
                }

                // After restore, generate first token via decode (no prefill extraction)
                #[cfg(feature = "gpu")]
                self.executor.mark_kv_prefix_resident(Some((
                    crate::gguf::batch_scheduler::PrefixCache::hash_tokens(prompt),
                    prompt.len(),
                )));
                position = prompt.len() - 1;
                last_token = prompt[prompt.len() - 1];
                max_decode = config.max_tokens;
            } else {
                // `is_moe` was a compile-time `false` guarding a serial-prefill
                // branch that therefore never ran: OwnedQuantizedLayer is the
                // GGUF-dense path and never holds MoE experts (MoE dispatch goes
                // through apr_transformer::AprTransformerLayer). Removed rather
                // than kept as dead weight.
                let greedy = config.temperature == 0.0 || config.top_k == 1;
                let prefill_count = if greedy { prompt.len() } else { prompt.len() - 1 };
                let prefill_first_token =
                    self.run_prefill(prompt, cache, prefill_count, config.trace, greedy)?;
                // #2697: the GPU now holds this prompt; a repeat costs a length reset.
                #[cfg(feature = "gpu")]
                self.executor.mark_kv_prefix_resident(Some((
                    crate::gguf::batch_scheduler::PrefixCache::hash_tokens(prompt),
                    prompt.len(),
                )));

                if let Some(first_tok) = prefill_first_token {
                    // PMAT-083: First token from prefill LM head
                    position = prompt.len();
                    last_token = first_tok;
                    tokens.push(first_tok);
                    if config.stop_tokens.contains(&first_tok) {
                        return Ok(None);
                    }
                    max_decode = config.max_tokens.saturating_sub(1);
                } else {
                    position = prompt.len() - 1;
                    last_token = prompt[prompt.len() - 1];
                    max_decode = config.max_tokens;
                }
            }
        Ok(Some((position, last_token, max_decode, prefix_was_hit)))
    }

    /// GPU-resident token generation with minimal CPU↔GPU transfers.
    ///
    /// # Reentrant
    ///
    /// This method creates fresh generation state on each call (new KV cache,
    /// reset GPU positions). It is safe and efficient to call multiple times
    /// on the same `OwnedQuantizedModelCuda` — weights are preloaded once
    /// during construction and reused across calls.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration (max_tokens, temperature, etc.)
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_gpu_resident(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        // PP-LLAMA-001 §3 / PP-2: phase timings belong to THIS request. A
        // prefix-cache hit skips `run_prefill`, so without this reset the hit
        // would inherit the previous request's `prefill_ms` and the server-
        // reported `timings.prompt_ms` that feeds the c=1 `prefill_ratio`
        // would be fabricated. None, never a stale number.
        self.last_phase_timings = crate::api::PhaseTimings::default();
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // GH-167 + realizr#194: Check against GPU KV cache capacity (not model context_length).
        // The GPU KV cache may be smaller than the model's native context window when
        // --context-length is used. Without this, overflow poisons CUDA graph state.
        let gpu_max_len = self.executor.max_kv_len();
        let effective_max = if gpu_max_len > 0 {
            gpu_max_len.min(self.model.config.context_length)
        } else {
            self.model.config.context_length
        };
        if prompt.len() > effective_max {
            return Err(RealizarError::ContextLimitExceeded {
                provided: prompt.len(),
                maximum: effective_max,
            });
        }

        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        // (context may have been created on a different thread, e.g., main vs tokio worker)
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident".to_string(),
                reason: "Model architecture not supported for GPU-resident path (requires separate Q/K/V, SwiGLU, RMSNorm)".to_string(),
            });
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.head_dim();
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        // PAR-055 FIX: Reset GPU KV cache positions before new generation
        // Without this, cache positions accumulate across generate calls causing degradation
        self.executor.reset_kv_cache_gpu();

        // PMAT-032: Graph preserved — workspace pointers stable across requests.

        let mut tokens = prompt.to_vec();

        if config.trace {
            eprintln!(
                "[TRACE-CACHE] GGUF model (GPU): {} layers, hidden_dim={}, vocab={}",
                self.model.config.num_layers,
                self.model.config.hidden_dim,
                self.model.config.vocab_size
            );
            eprintln!(
                "[TRACE-CACHE] Prefill: {} tokens, max_gen={}",
                prompt.len(),
                config.max_tokens
            );
        }

        // realizr#199 (PMAT-450): Check prefix cache before prefill.
        // If prompt was seen before, skip prefill entirely (TTFT ~900ms → ~5ms).
        // #2697 refactor: how this prompt's KV reaches the device, and where
        // decoding therefore starts. `None` means a stop token ended it already.
        let Some((position, last_token, max_decode, prefix_was_hit)) =
            self.establish_kv_blocking(prompt, config, &mut cache, &mut tokens)?
        else {
            return Ok(tokens);
        };
        let mut position = position;
        let mut last_token = last_token;

        // PMAT-814: when a repetition penalty is active we MUST have CPU-side logits to
        // penalize, so the GPU-side fused argmax fast path (forward_gpu_resident_to_token_id)
        // can only be used when no penalty applies. With repeat_penalty == 1.0 (the default)
        // this is false and the greedy fast path is taken unchanged — no perf regression.
        let penalty_active = config.repeat_penalty != 1.0 && config.repeat_last_n > 0;
        self.decode_blocking(config, &mut tokens, &mut cache, position, last_token, max_decode)?;

        // realizr#199 (PMAT-450): Insert into prefix cache after generation.
        // Only cache PROMPT KV if prefill was actually computed (not a cache hit).
        #[cfg(feature = "gpu")]
        if !prefix_was_hit {
            let num_layers = self.model.config.num_layers;
            // Temporarily truncate KV to prompt length for snapshot
            let current_lens: Vec<(usize, usize)> = (0..num_layers)
                .map(|l| (l, self.executor.kv_cache_len(l)))
                .collect();
            for &(l, _) in &current_lens {
                self.executor.set_kv_cache_len(l, prompt.len());
            }
            match self.executor.snapshot_kv_cache_to_host(num_layers) {
                Ok(kv_snapshot) => {
                    let (k_vecs, v_vecs): (Vec<_>, Vec<_>) = kv_snapshot.into_iter().unzip();
                    self.prefix_cache.insert(prompt.to_vec(), k_vecs, v_vecs);
                    if config.trace {
                        eprintln!("[PMAT-450] PREFIX CACHE INSERT: {} prompt tokens ({} layers)", prompt.len(), num_layers);
                    }
                }
                Err(e) => {
                    if config.trace {
                        eprintln!("[PMAT-450] PREFIX CACHE SNAPSHOT ERROR: {}", e);
                    }
                }
            }
            // Restore original KV lengths
            for &(l, len) in &current_lens {
                self.executor.set_kv_cache_len(l, len);
            }
        }

        Ok(tokens)
    }

    /// realizr#191: Generate with per-token log probabilities for perplexity.
    ///
    /// Same as `generate_gpu_resident` but always uses the logits path
    /// (no `forward_gpu_resident_to_token_id` shortcut) so we can extract
    /// log_softmax for each chosen token. ~5% slower due to logits download.
    pub fn generate_gpu_resident_logprobs(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<super::super::logprobs::GenerateResult> {
        use super::super::logprobs::{GenerateResult, TokenLogprob};

        // PP-LLAMA-001 §3 / PP-2: phase timings belong to THIS request. A
        // prefix-cache hit skips `run_prefill`, so without this reset the hit
        // would inherit the previous request's `prefill_ms` and the server-
        // reported `timings.prompt_ms` that feeds the c=1 `prefill_ratio`
        // would be fabricated. None, never a stale number.
        self.last_phase_timings = crate::api::PhaseTimings::default();
        if prompt.is_empty() {
            return Ok(GenerateResult { tokens: Vec::new(), logprobs: Vec::new() });
        }
        self.check_gpu_resident_preconditions(prompt, "generate_gpu_resident_logprobs")?;

        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.head_dim();
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers, kv_dim,
            prompt.len() + config.max_tokens,
        );
        self.executor.reset_kv_cache_gpu();

        let mut tokens = prompt.to_vec();
        let mut token_logprobs = Vec::with_capacity(config.max_tokens);

        let greedy = config.temperature == 0.0 || config.top_k == 1;
        let prefill_count = if greedy { prompt.len() } else { prompt.len() - 1 };
        let prefill_first_token = self.run_prefill(prompt, &mut cache, prefill_count, false, greedy)?;

        let mut position;
        let mut last_token;
        let max_decode;

        if let Some(first_tok) = prefill_first_token {
            position = prompt.len();
            last_token = first_tok;
            tokens.push(first_tok);
            if config.stop_tokens.contains(&first_tok) {
                return Ok(GenerateResult { tokens, logprobs: token_logprobs });
            }
            max_decode = config.max_tokens.saturating_sub(1);
        } else {
            position = prompt.len() - 1;
            last_token = prompt[prompt.len() - 1];
            max_decode = config.max_tokens;
        }

        for _ in 0..max_decode {
            // Always use logits path for logprob extraction
            let mut logits = self.forward_gpu_resident(last_token, &mut cache, position)?;
            // PMAT-814: penalize recently-seen tokens before greedy/sampling AND before
            // logprob extraction, so the reported logprob reflects the penalized
            // distribution that actually selected the token (no-op when penalty == 1.0).
            OwnedQuantizedModel::apply_repeat_penalty(
                &mut logits,
                &tokens,
                config.repeat_penalty,
                config.repeat_last_n,
            );
            let next_token = if greedy {
                OwnedQuantizedModel::argmax(&logits)
            } else {
                OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };
            token_logprobs.push(TokenLogprob {
                token_id: next_token,
                logprob: super::super::logprobs::logprob_of(&logits, next_token),
            });
            if config.stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(GenerateResult { tokens, logprobs: token_logprobs })
    }

    /// realizr#191: Teacher-forcing perplexity on a token sequence.
    ///
    /// Feeds each ground-truth token through the forward pass and records
    /// the log probability of the ACTUAL next token (not the model's
    /// prediction). This is the standard perplexity methodology used by
    /// llama-perplexity and lm-evaluation-harness.
    ///
    /// PPL = exp(-1/N * sum(logprob_of(token[i+1]) at position i))
    pub fn perplexity_gpu_resident(
        &mut self,
        tokens: &[u32],
    ) -> Result<f64> {
        use super::super::logprobs::logprob_of;

        if tokens.len() < 2 {
            return Ok(0.0);
        }
        // realizr#194: Validate against GPU KV cache capacity (not model context_length).
        // The GPU KV cache is pre-allocated at server startup and may be smaller than
        // the model's native context window. Without this check, overflow corrupts CUDA
        // state and poisons all subsequent requests.
        let gpu_max_len = self.executor.max_kv_len();
        let effective_max = if gpu_max_len > 0 {
            gpu_max_len.min(self.model.config.context_length)
        } else {
            self.model.config.context_length
        };
        if tokens.len() > effective_max {
            return Err(RealizarError::ContextLimitExceeded {
                provided: tokens.len(),
                maximum: effective_max,
            });
        }
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "perplexity_gpu_resident".to_string(),
                reason: "Architecture not supported".to_string(),
            });
        }

        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.head_dim();
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers, kv_dim, tokens.len(),
        );
        self.executor.reset_kv_cache_gpu();

        let mut sum_logprob: f64 = 0.0;
        let mut count: usize = 0;

        // Teacher-forcing: feed token[i], get logits, score token[i+1]
        // realizr#194: On error, reset KV cache to prevent CUDA state poisoning.
        for i in 0..tokens.len() - 1 {
            match self.forward_gpu_resident(tokens[i], &mut cache, i) {
                Ok(logits) => {
                    let lp = logprob_of(&logits, tokens[i + 1]);
                    sum_logprob += f64::from(lp);
                    count += 1;
                }
                Err(e) => {
                    // realizr#194: Reset KV cache AND invalidate decode graph
                    // to prevent poisoned CUDA state from affecting subsequent
                    // requests. Without graph invalidation, the stale graph
                    // replays with invalid pointers → CUDA_ERROR_INVALID_VALUE.
                    self.executor.reset_kv_cache_gpu();
                    self.executor.clear_decode_graph();
                    return Err(e);
                }
            }
        }

        // Reset KV cache after measurement (perplexity is stateless)
        self.executor.reset_kv_cache_gpu();

        let ppl = if count > 0 {
            (-sum_logprob / count as f64).exp()
        } else {
            0.0
        };
        Ok(ppl)
    }

    /// realizr#203: Batched prefill perplexity using FP8 GEMM path.
    ///
    /// Processes all tokens in one prefill forward (M=S uses FP8 cuBLASLt),
    /// then extracts per-position logits via batched LM head GEMM.
    /// Expected to close PPL gap vs llama.cpp (24.2 → ~13-16 target).
    ///
    /// Five-whys root cause: `perplexity_gpu_resident` used M=1 DP4A GEMV
    /// per token, accumulating int8 precision loss over 28 layers.
    pub fn perplexity_gpu_batched(
        &mut self,
        tokens: &[u32],
    ) -> Result<f64> {
        use super::super::logprobs::logprob_of;

        if tokens.len() < 2 {
            return Ok(0.0);
        }

        // Validate against GPU KV cache capacity
        let gpu_max_len = self.executor.max_kv_len();
        let effective_max = if gpu_max_len > 0 {
            gpu_max_len.min(self.model.config.context_length)
        } else {
            self.model.config.context_length
        };
        if tokens.len() > effective_max {
            return Err(RealizarError::ContextLimitExceeded {
                provided: tokens.len(),
                maximum: effective_max,
            });
        }

        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "perplexity_gpu_batched".to_string(),
                reason: "Architecture not supported".to_string(),
            });
        }

        let s = tokens.len();
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;

        // 1. Reset KV cache and initialize prefill workspace
        self.executor.reset_kv_cache_gpu();
        self.executor
            .init_prefill_workspace(
                s,
                hidden_dim,
                intermediate_dim,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_prefill_workspace".to_string(),
                reason: format!("Prefill workspace init failed: {e}"),
            })?;

        // 2. Embed all S tokens (CPU — fast, token lookup)
        let mut embeddings = vec![0.0f32; s * hidden_dim];
        for (i, &token_id) in tokens.iter().enumerate() {
            let start = i * hidden_dim;
            let end = start + hidden_dim;
            self.model.embed_into(token_id, &mut embeddings[start..end]);
        }

        // 3. Prefill: process all S tokens through all layers (FP8 GEMM path)
        let positions: Vec<u32> = (0..s as u32).collect();
        self.executor
            .prefill_all_layers_gpu(
                &embeddings,
                &positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                eps,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "prefill_all_layers_gpu".to_string(),
                reason: format!("Prefill failed: {e}"),
            })?;

        // 4. Download hidden_buf2[S × hidden_dim] and extract per-position logits.
        //
        // After prefill, hidden_buf2 has FP8-precision hidden states for all S positions.
        // We apply output norm + LM head per position using the standard M=1 path.
        // The precision improvement comes from layers using FP8 GEMM (prefill) instead
        // of DP4A GEMV (sequential decode).
        //
        // Why per-position instead of batched: batched LM head extraction via
        // batched_gemv_or_gemm has correctness bugs at S>2 (realizr#203).
        // Per-position is ~S×0.1ms but only runs during PPL measurement (offline).
        self.executor.sync_stream()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "stream_sync".to_string(),
                reason: format!("{e}"),
            })?;

        let hidden_size = s * hidden_dim;
        let mut all_hidden = vec![0.0f32; hidden_size];
        self.executor.download_hidden_buf2(&mut all_hidden)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "hidden_download".to_string(),
                reason: format!("{e}"),
            })?;

        let mut sum_logprob: f64 = 0.0;
        let mut count: usize = 0;

        for i in 0..s - 1 {
            let pos_hidden = &all_hidden[i * hidden_dim..(i + 1) * hidden_dim];
            let mut logits = vec![0.0f32; vocab_size];
            self.executor
                .hidden_to_logits(pos_hidden, &mut logits, hidden_dim as u32, vocab_size as u32, eps)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "hidden_to_logits".to_string(),
                    reason: format!("{e}"),
                })?;

            // Add LM head bias if present
            if let Some(ref bias) = self.model.lm_head_bias {
                crate::gguf::ops::add_bias(&mut logits, bias);
            }

            let lp = logprob_of(&logits, tokens[i + 1]);
            sum_logprob += f64::from(lp);
            count += 1;
        }

        // Reset KV cache after measurement
        self.executor.reset_kv_cache_gpu();

        let ppl = if count > 0 {
            (-sum_logprob / count as f64).exp()
        } else {
            0.0
        };
        Ok(ppl)
    }
}

/// PP-LLAMA-001 §9 #1: state, once per process, which prefill path is in force.
///
/// The path is resolved once in `GpuProfile::detect` and cannot change while the
/// process runs, so one line is the whole fact — and a per-request line would be
/// one per request at 100+ requests per band. Unconditional (not behind
/// `trace`), because a receipt that cannot witness which prefill ran cannot
/// distinguish the PMAT-810 Blackwell corruption from a healthy run.
fn announce_prefill_path(choice: crate::cuda::gpu_profile::PrefillPathChoice) {
    static ANNOUNCED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ANNOUNCED.get_or_init(|| {
        eprintln!(
            "[PREFILL-PATH] {} cc={} ({})",
            choice.path.as_str(),
            choice.cc,
            choice.reason
        );
    });
}
