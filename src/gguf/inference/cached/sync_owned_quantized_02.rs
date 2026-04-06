impl OwnedQuantizedModelCachedSync {

    /// Batch generation with GPU-accelerated FFN (PARITY-020)
    ///
    /// Processes multiple prompts in parallel using GPU batch operations.
    /// The key optimization is converting MATVEC (single token) to GEMM (batch tokens).
    ///
    /// # Architecture
    /// - Attention: CPU with KV cache (MATVEC is faster on CPU)
    /// - FFN: GPU with batch GEMM (batch_size ≥ 32 uses GPU)
    /// - Sampling: CPU (negligible compared to matmul)
    ///
    /// # Arguments
    /// * `prompts` - Multiple prompts to process in parallel [num_prompts][seq_len]
    /// * `config` - Generation configuration (shared across all prompts)
    ///
    /// # Returns
    /// Generated sequences for each prompt [num_prompts][generated_len]
    ///
    /// # Errors
    /// Returns error if GPU cache not warmed up or generation fails
    ///
    /// # Performance
    /// - Single prompt: ~5 tok/s (CPU-bound, no batching benefit)
    /// - 32 prompts: ~150 tok/s total (~4.7 tok/s per prompt)
    /// - 64 prompts: ~280 tok/s total (~4.4 tok/s per prompt, memory-bound)
    pub fn batch_generate_gpu(
        &self,
        prompts: &[Vec<u32>],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<Vec<u32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        // Verify GPU cache is warmed up
        if !self.is_gpu_cache_warm() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "batch_generate_gpu".to_string(),
                reason: "GPU cache not warmed up. Call warmup_gpu_cache() first.".to_string(),
            });
        }

        let num_prompts = prompts.len();
        let max_seq_len = prompts.iter().map(Vec::len).max().unwrap_or(0) + config.max_tokens;

        // Initialize KV caches for each prompt
        let mut caches: Vec<OwnedQuantizedKVCache> = prompts
            .iter()
            .map(|_| OwnedQuantizedKVCache::from_config(&self.model.config, max_seq_len))
            .collect();

        // Initialize token sequences (copy prompts)
        let mut sequences: Vec<Vec<u32>> = prompts.to_vec();

        // Track generation progress per prompt
        let mut done: Vec<bool> = vec![false; num_prompts];

        // PARITY-097: Parallel prefill across prompts using rayon
        // Each prompt's prefill is independent (different KV cache)
        // Model is shared immutably (&self), caches are mutated independently
        use rayon::prelude::*;

        caches
            .par_iter_mut()
            .zip(prompts.par_iter())
            .try_for_each(|(cache, prompt)| {
                for (pos, &token_id) in prompt.iter().enumerate() {
                    self.model.forward_single_with_cache(token_id, cache, pos)?;
                }
                Ok::<_, RealizarError>(())
            })?;

        // Generation loop with batched FFN (PARITY-021: GPU optimization)
        for gen_idx in 0..config.max_tokens {
            // Collect active prompts for this generation step
            let active_indices: Vec<usize> = (0..num_prompts).filter(|&i| !done[i]).collect();

            if active_indices.is_empty() {
                break;
            }

            let active_count = active_indices.len();

            // Use batched forward when we have enough active prompts for GPU benefit
            // GPU batch threshold is 32 (from IMP-600 analysis)
            const GPU_BATCH_THRESHOLD: usize = 32;

            if active_count >= GPU_BATCH_THRESHOLD {
                // PARITY-021: Batched forward with GPU FFN
                // Collect tokens, positions, and cache slices for active prompts
                let batch_tokens: Vec<u32> = active_indices
                    .iter()
                    .map(|&idx| {
                        *sequences[idx]
                            .last()
                            .expect("sequence must have at least prompt tokens")
                    })
                    .collect();

                let batch_positions: Vec<usize> = active_indices
                    .iter()
                    .map(|&idx| prompts[idx].len() + gen_idx)
                    .collect();

                // PARITY-096: Extract caches without cloning using std::mem::take
                // This avoids expensive cache cloning on every generation step
                let mut batch_caches: Vec<OwnedQuantizedKVCache> = active_indices
                    .iter()
                    .map(|&idx| std::mem::take(&mut caches[idx]))
                    .collect();

                // Forward batch with GPU FFN
                let all_logits = self.forward_batch_with_gpu_ffn(
                    &batch_tokens,
                    &mut batch_caches,
                    &batch_positions,
                )?;

                // PARITY-096: Put caches back (move, not clone)
                for (i, &idx) in active_indices.iter().enumerate() {
                    caches[idx] = std::mem::take(&mut batch_caches[i]);
                }

                // Sample and update sequences
                for (i, &prompt_idx) in active_indices.iter().enumerate() {
                    let logits = &all_logits[i];
                    let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                        OwnedQuantizedModel::argmax(logits)
                    } else {
                        OwnedQuantizedModel::sample_topk(logits, config.temperature, config.top_k)
                    };

                    if config.stop_tokens.contains(&next_token) {
                        done[prompt_idx] = true;
                    } else {
                        sequences[prompt_idx].push(next_token);
                        if sequences[prompt_idx].len() >= max_seq_len {
                            done[prompt_idx] = true;
                        }
                    }
                }
            } else {
                // Sequential forward for small batches (CPU is faster)
                for &prompt_idx in &active_indices {
                    let position = prompts[prompt_idx].len() + gen_idx;
                    let last_token = *sequences[prompt_idx]
                        .last()
                        .expect("sequence must have at least prompt tokens");

                    let logits = self.model.forward_single_with_cache(
                        last_token,
                        &mut caches[prompt_idx],
                        position,
                    )?;

                    let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                        OwnedQuantizedModel::argmax(&logits)
                    } else {
                        OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
                    };

                    if config.stop_tokens.contains(&next_token) {
                        done[prompt_idx] = true;
                    } else {
                        sequences[prompt_idx].push(next_token);
                        if sequences[prompt_idx].len() >= max_seq_len {
                            done[prompt_idx] = true;
                        }
                    }
                }
            }
        }

        Ok(sequences)
    }
}
