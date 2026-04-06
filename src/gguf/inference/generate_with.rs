
impl OwnedQuantizedModel {

    /// Generate tokens with zero-allocation inference (IMP-131)
    ///
    /// This is the highest-performance generation path. Uses pre-allocated
    /// scratch buffers to eliminate per-token allocations, providing ~3-4x
    /// speedup over allocating variants.
    ///
    /// Performance characteristics:
    /// - Single allocation at start (scratch buffer + KV cache)
    /// - Zero allocations per generated token
    /// - ~500KB saved per token for TinyLlama-1.1B
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
    pub fn generate_with_scratch(
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
        let mut scratch = InferenceScratchBuffer::from_config(&self.config);
        let mut tokens = prompt.to_vec();

        // PMAT-TRACE-GGUF-001: Trace config info
        if config.trace {
            eprintln!(
                "[TRACE-CACHE] GGUF scratch: {} layers, hidden_dim={}, vocab={}",
                self.config.num_layers, self.config.hidden_dim, self.config.vocab_size
            );
            eprintln!(
                "[TRACE-CACHE] Prefill: {} tokens, max_gen={}",
                prompt.len(),
                config.max_tokens
            );
        }

        // Process prompt tokens (prefill) - uses scratch buffers
        let prefill_start = std::time::Instant::now();
        for (pos, &token_id) in prompt.iter().enumerate() {
            self.forward_single_with_scratch(token_id, &mut cache, pos, &mut scratch)?;
        }
        if config.trace {
            eprintln!(
                "[TRACE-CACHE] Prefill complete: {} tokens in {:?}",
                prompt.len(),
                prefill_start.elapsed()
            );
        }

        // Generate new tokens - zero allocations per token
        // PAR-126: Fixed loop structure to match generate_with_cache:
        // 1. Sample from current logits (prefill on first iter, previous forward otherwise)
        // 2. Then run forward on the new token to get logits for next iteration
        for gen_idx in 0..config.max_tokens {
            let token_start = std::time::Instant::now();
            // Sample next token from current logits (prefill logits on first iter)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&scratch.logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(
                    &scratch.logits,
                    config.temperature,
                    config.top_k,
                )
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the new token
            let position = prompt.len() + gen_idx;
            self.forward_single_with_scratch(next_token, &mut cache, position, &mut scratch)?;

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

        Ok(tokens)
    }

    /// Generate tokens with adaptive CPU/GPU attention (IMP-125)
    ///
    /// This variant of `generate_with_cache` uses `forward_single_with_cache_adaptive`
    /// to automatically select between CPU and GPU backends based on cache length.
    /// It also records dispatch decisions to the provided metrics tracker.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if forward pass fails
    #[cfg(feature = "gpu")]
    pub fn generate_with_cache_adaptive(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // PMAT-TRACE-GGUF-001: Trace config info
        if config.trace {
            eprintln!(
                "[TRACE-CACHE] GGUF adaptive: {} layers, hidden_dim={}, vocab={}",
                self.config.num_layers, self.config.hidden_dim, self.config.vocab_size
            );
            eprintln!(
                "[TRACE-CACHE] Prefill: {} tokens, max_gen={}",
                prompt.len(),
                config.max_tokens
            );
        }

        // Process prompt tokens (prefill) with adaptive attention
        // Keep the logits from the last position for the first generated token
        let prefill_start = std::time::Instant::now();
        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache_adaptive(token_id, &mut cache, pos, metrics)?;
        }
        if config.trace {
            eprintln!(
                "[TRACE-CACHE] Prefill complete: {} tokens in {:?}",
                prompt.len(),
                prefill_start.elapsed()
            );
        }

        // Generate new tokens with adaptive attention
        for gen_idx in 0..config.max_tokens {
            let token_start = std::time::Instant::now();
            // Sample next token from current logits
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

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the newly sampled token
            let position = prompt.len() + gen_idx;
            logits =
                self.forward_single_with_cache_adaptive(next_token, &mut cache, position, metrics)?;

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

        Ok(tokens)
    }
}

include!("fails.rs");
