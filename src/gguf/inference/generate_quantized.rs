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

        for _ in 0..config.max_tokens {
            // Forward pass with fused Q4_K ops (1.37x faster)
            let logits = self.forward(&tokens)?;

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                // Greedy decoding
                Self::argmax(&logits)
            } else {
                // Temperature + top-k sampling
                Self::sample_topk(&logits, config.temperature, config.top_k)
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

    /// Top-k sampling with temperature
    pub fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Get top-k indices
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);

        // Softmax over top-k
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
        let probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(i, v)| (*i, (v - max_val).exp() / exp_sum))
            .collect();

        // Sample from probability distribution with proper randomness
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();

        let mut cumulative = 0.0;
        for &(idx, prob) in &probs {
            cumulative += prob;
            if cumulative >= r {
                return idx as u32;
            }
        }

        probs.last().map_or(0, |(idx, _)| *idx as u32)
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

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(
                    &logits,
                    config.temperature,
                    config.top_k,
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
            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(
                    &logits,
                    config.temperature,
                    config.top_k,
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
