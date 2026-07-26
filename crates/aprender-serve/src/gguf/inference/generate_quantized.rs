impl OwnedQuantizedModel {
    /// Get most likely next token
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;
        Ok(max_idx as u32)
    }

    /// Generate tokens using fused Q4_K operations (IMP-100)
    ///
    /// This is the HTTP serving entry point for quantized inference.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn generate(&self, prompt: &[u32], config: &QuantizedGenerateConfig) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        // GH-167: Check context length before GPU dispatch to avoid cryptic CUDA errors
        if prompt.len() > self.config.context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: prompt.len(),
                maximum: self.config.context_length,
            });
        }

        let mut tokens = prompt.to_vec();
        let max_len = prompt.len() + config.max_tokens;
        // PMAT-819: seed the sampler RNG from config.seed so the OpenAI `seed`
        // API contract holds (same prompt+seed+params => same tokens). Greedy
        // (temperature==0 || top_k==1) never touches the RNG, so the default
        // config is byte-for-byte unchanged.
        let mut rng = StdRng::seed_from_u64(config.seed);

        for _ in 0..config.max_tokens {
            // Forward pass with fused Q4_K ops (1.37x faster)
            let mut logits = self.forward(&tokens)?;

            // PMAT-814: apply repetition penalty in place over the recent context
            // BEFORE both greedy argmax and sampling (no-op when repeat_penalty == 1.0).
            Self::apply_repeat_penalty(
                &mut logits,
                &tokens,
                config.repeat_penalty,
                config.repeat_last_n,
            );

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                // Greedy decoding
                Self::argmax(&logits)
            } else {
                // Temperature + top-k sampling (seeded for reproducibility)
                Self::sample_topk_seeded(&logits, config.temperature, config.top_k, config.top_p, &mut rng)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_len {
                break;
            }
        }

        Ok(tokens)
    }

    /// Greedy argmax over logits
    pub(crate) fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    }

    /// Apply a repetition penalty to `logits` in place (PMAT-814).
    ///
    /// Mirrors the live MoE path (`infer/qwen3_moe_generate.rs::sample_from_logits`)
    /// and Candle's `apply_repeat_penalty`: every token in the recency window has its
    /// logit divided by `penalty` when positive and multiplied by `penalty` when
    /// non-positive, so a larger `penalty` always shrinks the chance of repeating a
    /// recently-seen token regardless of its logit sign.
    ///
    /// The window is the last `last_n` entries of `recent_tokens` (the full decoded
    /// context — prompt + generated — exactly as `repeat_last_n` is interpreted on the
    /// MoE path and by llama.cpp's default), so callers pass the entire `tokens` vector.
    ///
    /// # No-op guarantee (no-regression)
    ///
    /// When `penalty == 1.0` (the default), `last_n == 0`, or `recent_tokens` is empty,
    /// this returns immediately without touching `logits` — every all-default `apr run`
    /// / `apr serve` request is byte-identical to the pre-PMAT-814 path (greedy argmax
    /// and top-k/top-p sampling alike).
    pub(crate) fn apply_repeat_penalty(
        logits: &mut [f32],
        recent_tokens: &[u32],
        penalty: f32,
        last_n: usize,
    ) {
        if penalty == 1.0 || last_n == 0 || recent_tokens.is_empty() {
            return;
        }
        let start = recent_tokens.len().saturating_sub(last_n);
        for &token in &recent_tokens[start..] {
            let idx = token as usize;
            if idx < logits.len() {
                if logits[idx] <= 0.0 {
                    logits[idx] *= penalty;
                } else {
                    logits[idx] /= penalty;
                }
            }
        }
    }

    /// Top-k sampling with temperature, drawing from the given uniform sample `r ∈ [0,1)`.
    ///
    /// Pure: the only source of randomness is the caller-supplied `r`. This lets both the
    /// entropy-seeded [`Self::sample_topk`] and the seeded [`Self::sample_topk_seeded`]
    /// share one inverse-CDF implementation, so a seeded RNG fully determines the token.
    fn sample_topk_with_draw(
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        r: f32,
    ) -> u32 {
        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Get top-k indices
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        // `top_k == 0` means "disabled" in llama.cpp and Ollama, and both the
        // Ollama-compat and OpenAI-compat surfaces pass it straight through. An
        // unguarded `truncate(0)` empties `indexed`, so `probs` is empty and the
        // inverse-CDF loop below falls through to `probs.last().map_or(0, ..)` —
        // returning token 0 on EVERY step (`!!!!!!`). Guard exactly as the live
        // MoE sampler does (infer/qwen3_moe_generate.rs:104).
        if top_k > 0 && top_k < indexed.len() {
            indexed.truncate(top_k);
        }

        // Top-p (nucleus): keep the smallest prefix whose cumulative softmax mass
        // reaches `top_p`. Ported from the live MoE sampler
        // (infer/qwen3_moe_generate.rs:109), which is where the only working
        // implementation lived — the dense path accepted `top_p` in
        // QuantizedGenerateConfig and then silently discarded it, so
        // `--top-p 0.001` was byte-identical to `--top-p 1.0` on every
        // /v1/chat/completions, /api/chat, /api/generate and `apr run` request.
        //
        // The `top_p > 0.0 && top_p < 1.0` guard is what keeps this a BIT-EXACT
        // no-op at the default 1.0: the branch is not entered at all, so the
        // candidate set and the inverse-CDF draw below are unchanged. That
        // matters because the default config sets top_p = 1.0 (runtime.rs:51) —
        // every existing caller must keep its current output token-for-token.
        if top_p > 0.0 && top_p < 1.0 {
            let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
            let exp_vals: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
            let total: f32 = exp_vals.iter().sum();
            if total > 0.0 {
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
        }

        // Softmax over the filtered set
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
        let probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(i, v)| (*i, (v - max_val).exp() / exp_sum))
            .collect();

        // Inverse-CDF draw from the categorical distribution
        let mut cumulative = 0.0;
        for &(idx, prob) in &probs {
            cumulative += prob;
            if cumulative >= r {
                return idx as u32;
            }
        }

        probs.last().map_or(0, |(idx, _)| *idx as u32)
    }

    /// Top-k sampling with temperature (entropy-seeded RNG).
    ///
    /// Backward-compatible: uses a fresh process-entropy RNG, so output is NOT reproducible.
    /// For deterministic / seeded sampling (the OpenAI `seed` API contract), use
    /// [`Self::sample_topk_seeded`].
    pub fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
        let r: f32 = rand::rng().random();
        // top_p = 1.0 => the nucleus branch is skipped entirely: bit-exact no-op,
        // so this public signature stays source-compatible for existing callers.
        Self::sample_topk_with_draw(logits, temperature, top_k, 1.0, r)
    }

    /// Top-k sampling with temperature, drawing from a caller-owned seeded RNG.
    ///
    /// PMAT-819: closes the dense-path seed-determinism gap. The HTTP `/v1/chat/completions`
    /// dense decode loops own one [`StdRng`] seeded from `QuantizedGenerateConfig.seed` and
    /// advance it once per sampled token, so the same `(prompt, seed, temperature, top_k)`
    /// produces byte-identical output across runs — matching the qwen3_moe path
    /// (`infer/qwen3_moe_generate.rs::sample_from_logits`) which already seeds from config.
    ///
    /// Discharges `openai-serve-sampling-determinism-v1` F-SEED-DETERMINISM-001/002.
    pub fn sample_topk_seeded(
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rng: &mut StdRng,
    ) -> u32 {
        let r: f32 = rng.random();
        Self::sample_topk_with_draw(logits, temperature, top_k, top_p, r)
    }

    /// Generate tokens using KV cache for efficient autoregressive decoding (IMP-101)
    ///
    /// This is O(n) per token instead of O(n²) due to KV cache reuse.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        // GH-167: Check context length before processing to avoid cryptic CUDA errors
        if prompt.len() > self.config.context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: prompt.len(),
                maximum: self.config.context_length,
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();
        // PMAT-819: seeded sampler RNG (OpenAI `seed` determinism). This is the
        // production HTTP dense path (try_quantized_backend -> generate_with_cache).
        let mut rng = StdRng::seed_from_u64(config.seed);

        // GH-104: BrickProfiler for per-operation timing in autoregressive path
        let mut profiler = if config.trace {
            BrickProfiler::new()
        } else {
            BrickProfiler::disabled()
        };
        if config.trace {
            profiler.set_num_layers(self.config.num_layers);
        }

        // PMAT-TRACE-GGUF-001: Trace config info
        if config.trace {
            eprintln!(
                "[TRACE-CACHE] GGUF model: {} layers, hidden_dim={}, vocab={}",
                self.config.num_layers, self.config.hidden_dim, self.config.vocab_size
            );
            eprintln!(
                "[TRACE-CACHE] Prefill: {} tokens, max_gen={}",
                prompt.len(),
                config.max_tokens
            );
        }

        // Process prompt tokens (prefill), keeping the logits from the last position
        // The logits from processing token[n-1] at position n-1 predict token[n]
        let prefill_start = std::time::Instant::now();
        let mut logits = Vec::new();
        if config.trace {
            profiler.start_inference();
            for (pos, &token_id) in prompt.iter().enumerate() {
                logits = self.forward_single_with_cache_profiled(
                    token_id, &mut cache, pos, &mut profiler,
                )?;
            }
        } else {
            for (pos, &token_id) in prompt.iter().enumerate() {
                logits = self.forward_single_with_cache(token_id, &mut cache, pos)?;
            }
        }
        if config.trace {
            eprintln!(
                "[TRACE-CACHE] Prefill complete: {} tokens in {:?}",
                prompt.len(),
                prefill_start.elapsed()
            );
        }

        // Generate new tokens
        // First iteration uses logits from prefill, subsequent use logits from forward pass
        for gen_idx in 0..config.max_tokens {
            let token_start = std::time::Instant::now();
            // DEBUG: Print logits info for first generated token
            if gen_idx == 0 && std::env::var("REALIZAR_DEBUG_LOGITS").is_ok() {
                let sum: f32 = logits.iter().sum();
                let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let min_val = logits.iter().copied().fold(f32::INFINITY, f32::min);
                let top_5: Vec<(usize, f32)> = {
                    let mut indexed: Vec<_> =
                        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                    indexed.sort_by(|(_, a), (_, b)| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    indexed.into_iter().take(5).collect()
                };
                eprintln!(
                    "[DEBUG-LOGITS] len={}, sum={:.4}, min={:.4}, max={:.4}",
                    logits.len(),
                    sum,
                    min_val,
                    max_val
                );
                eprintln!("[DEBUG-LOGITS] top 5 token ids and logits: {:?}", top_5);
                eprintln!(
                    "[DEBUG-LOGITS] logits[0..5]: {:?}",
                    &logits[..5.min(logits.len())]
                );
            }

            // PMAT-814: apply repetition penalty in place over the recent context
            // BEFORE both greedy argmax and sampling (no-op when repeat_penalty == 1.0).
            crate::gguf::OwnedQuantizedModel::apply_repeat_penalty(
                &mut logits,
                &tokens,
                config.repeat_penalty,
                config.repeat_last_n,
            );

            // Sample next token (PMAT-819: seeded for OpenAI `seed` determinism)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk_seeded(
                    &logits,
                    config.temperature,
                    config.top_k,
                    config.top_p,
                    &mut rng,
                )
            };

            // DEBUG: Print selected token
            if gen_idx == 0 && std::env::var("REALIZAR_DEBUG_LOGITS").is_ok() {
                eprintln!(
                    "[DEBUG-LOGITS] selected token: {} (logit={:.4})",
                    next_token,
                    logits.get(next_token as usize).copied().unwrap_or(f32::NAN)
                );
            }

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the newly sampled token
            // Position is prompt.len() + gen_idx (where token was just added)
            let position = prompt.len() + gen_idx;
            if config.trace {
                logits = self.forward_single_with_cache_profiled(
                    next_token, &mut cache, position, &mut profiler,
                )?;
            } else {
                logits = self.forward_single_with_cache(next_token, &mut cache, position)?;
            }

            // PMAT-TRACE-GGUF-001: Per-token timing
            if config.trace {
                eprintln!(
                    "[TRACE-CACHE] pos={}: {} layers took {:?}",
                    position,
                    self.config.num_layers,
                    token_start.elapsed()
                );
            }
        }

        // GH-104: Print BrickProfiler report when tracing is enabled
        if config.trace {
            profiler.stop_inference();
            let generated = tokens.len().saturating_sub(prompt.len());
            profiler.set_tokens(prompt.len() + generated);
            let report = profiler.report();
            eprintln!("[BRICK-PROFILE] === Autoregressive Path Profile ===");
            eprintln!(
                "[BRICK-PROFILE] Total: {:.2}ms, {} tokens ({} prefill + {} decode), {:.1} tok/s",
                report.total_inference_us / 1000.0,
                report.tokens_processed,
                prompt.len(),
                generated,
                report.throughput_tok_s,
            );
            let breakdown = report.percentage_breakdown();
            for (name, stats) in report.sorted_by_time() {
                let pct = breakdown.get(name).copied().unwrap_or(0.0);
                eprintln!(
                    "[BRICK-PROFILE]   {:<20} {:>8.2}ms ({:>5.1}%)  avg={:.1}us  count={}",
                    name,
                    stats.total_us / 1000.0,
                    pct,
                    stats.avg_us,
                    stats.count,
                );
            }
        }

        Ok(tokens)
    }

    /// Generate tokens with streaming callback (PMAT-087)
    ///
    /// Same as `generate_with_cache` but calls `on_token` after each token
    /// is generated, enabling true streaming to clients.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    /// * `on_token` - Callback called for each generated token. Return `false` to stop.
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_with_cache_streaming<F>(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        // GH-167: Check context length before processing to avoid cryptic CUDA errors
        if prompt.len() > self.config.context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: prompt.len(),
                maximum: self.config.context_length,
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();
        // PMAT-819: seeded sampler RNG (OpenAI `seed` determinism, streaming dense path).
        let mut rng = StdRng::seed_from_u64(config.seed);

        // GH-104: BrickProfiler for per-operation timing in streaming path
        let mut profiler = if config.trace {
            BrickProfiler::new()
        } else {
            BrickProfiler::disabled()
        };
        if config.trace {
            profiler.set_num_layers(self.config.num_layers);
        }

        // PMAT-TRACE-GGUF-001: Trace config info
        if config.trace {
            eprintln!(
                "[TRACE-CACHE] GGUF streaming: {} layers, hidden_dim={}, vocab={}",
                self.config.num_layers, self.config.hidden_dim, self.config.vocab_size
            );
            eprintln!(
                "[TRACE-CACHE] Prefill: {} tokens, max_gen={}",
                prompt.len(),
                config.max_tokens
            );
        }

        // Process prompt tokens (prefill)
        let prefill_start = std::time::Instant::now();
        let mut logits = Vec::new();
        if config.trace {
            profiler.start_inference();
            for (pos, &token_id) in prompt.iter().enumerate() {
                logits = self.forward_single_with_cache_profiled(
                    token_id, &mut cache, pos, &mut profiler,
                )?;
            }
        } else {
            for (pos, &token_id) in prompt.iter().enumerate() {
                logits = self.forward_single_with_cache(token_id, &mut cache, pos)?;
            }
        }
        if config.trace {
            eprintln!(
                "[TRACE-CACHE] Prefill complete: {} tokens in {:?}",
                prompt.len(),
                prefill_start.elapsed()
            );
        }

        // Generate new tokens with streaming
        for gen_idx in 0..config.max_tokens {
            let token_start = std::time::Instant::now();
            // PMAT-814: apply repetition penalty in place over the recent context
            // BEFORE both greedy argmax and sampling (no-op when repeat_penalty == 1.0).
            crate::gguf::OwnedQuantizedModel::apply_repeat_penalty(
                &mut logits,
                &tokens,
                config.repeat_penalty,
                config.repeat_last_n,
            );
            // Sample next token (PMAT-819: seeded for OpenAI `seed` determinism)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk_seeded(
                    &logits,
                    config.temperature,
                    config.top_k,
                    config.top_p,
                    &mut rng,
                )
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // PMAT-087: Call streaming callback - stop if it returns false
            if !on_token(next_token) {
                break;
            }

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration
            let position = prompt.len() + gen_idx;
            if config.trace {
                logits = self.forward_single_with_cache_profiled(
                    next_token, &mut cache, position, &mut profiler,
                )?;
            } else {
                logits = self.forward_single_with_cache(next_token, &mut cache, position)?;
            }

            // PMAT-TRACE-GGUF-001: Per-token timing
            if config.trace {
                eprintln!(
                    "[TRACE-CACHE] pos={}: {} layers took {:?}",
                    position,
                    self.config.num_layers,
                    token_start.elapsed()
                );
            }
        }

        // GH-104: Print BrickProfiler report when tracing is enabled
        if config.trace {
            profiler.stop_inference();
            let generated = tokens.len().saturating_sub(prompt.len());
            profiler.set_tokens(prompt.len() + generated);
            let report = profiler.report();
            eprintln!("[BRICK-PROFILE] === Streaming Path Profile ===");
            eprintln!(
                "[BRICK-PROFILE] Total: {:.2}ms, {} tokens ({} prefill + {} decode), {:.1} tok/s",
                report.total_inference_us / 1000.0,
                report.tokens_processed,
                prompt.len(),
                generated,
                report.throughput_tok_s,
            );
            let breakdown = report.percentage_breakdown();
            for (name, stats) in report.sorted_by_time() {
                let pct = breakdown.get(name).copied().unwrap_or(0.0);
                eprintln!(
                    "[BRICK-PROFILE]   {:<20} {:>8.2}ms ({:>5.1}%)  avg={:.1}us  count={}",
                    name,
                    stats.total_us / 1000.0,
                    pct,
                    stats.avg_us,
                    stats.count,
                );
            }
        }

        Ok(tokens)
    }
}
