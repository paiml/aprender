
impl OwnedQuantizedModelCuda {

    /// PAR-112: True token-by-token streaming generation
    ///
    /// Generates tokens one at a time and calls the callback after each token.
    /// The callback receives the token ID and can return `false` to stop generation early.
    ///
    /// This enables true real-time streaming where each token is delivered
    /// as soon as it's generated, rather than pseudo-streaming where all tokens
    /// are generated first then iterated.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `on_token` - Callback called for each generated token, returns `false` to stop
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.generate_gpu_resident_streaming(&prompt, &config, |token_id| {
    ///     println!("Generated: {}", token_id);
    ///     true // continue generation
    /// })?;
    /// ```
    pub fn generate_gpu_resident_streaming<F>(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        let ttft_trace = std::env::var("TTFT_TRACE").is_ok();
        let t_start = if ttft_trace { Some(std::time::Instant::now()) } else { None };
        macro_rules! ttft_mark {
            ($label:expr) => {
                if let Some(t0) = t_start {
                    eprintln!("[TTFT] {:>20}: {:>7.2}ms", $label, t0.elapsed().as_secs_f64() * 1000.0);
                }
            };
        }

        // GH-167: Check context length BEFORE GPU dispatch to return clean error
        if prompt.len() > self.model.config.context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: prompt.len(),
                maximum: self.model.config.context_length,
            });
        }

        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;
        ttft_mark!("make_current");

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident_streaming".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }

        // Create KV cache with GQA-aware dimensions
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim,
            prompt.len() + config.max_tokens,
        );
        ttft_mark!("kv_cache_alloc");

        // Reset GPU KV cache positions (lengths → 0)
        self.executor.reset_kv_cache_gpu();
        ttft_mark!("reset_gpu");

        let mut tokens = prompt.to_vec();

        // realizr#199 (PMAT-450): Check prefix cache before prefill.
        #[cfg(feature = "gpu")]
        let prefix_hit = self.prefix_cache.lookup(prompt);
        #[cfg(not(feature = "gpu"))]
        let prefix_hit: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = None;
        #[cfg(feature = "gpu")]
        let prefix_was_hit = prefix_hit.is_some();
        #[cfg(not(feature = "gpu"))]
        let prefix_was_hit = false;

        let prefill_first_token;
        if let Some((cached_k, cached_v)) = prefix_hit {
            // PMAT-450: Prefix cache hit — skip prefill, restore GPU KV cache
            eprintln!("[PMAT-450] PREFIX CACHE HIT: {} prompt tokens, skipping prefill", prompt.len());
            let kv_pairs: Vec<(Vec<f32>, Vec<f32>)> = cached_k.into_iter().zip(cached_v).collect();
            self.executor
                .restore_kv_cache_from_host(&kv_pairs, prompt.len())
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "restore_kv_cache_from_host".to_string(),
                    reason: format!("Prefix cache restore failed: {e}"),
                })?;
            prefill_first_token = None; // No prefill extraction — go straight to decode
            ttft_mark!("prefix_cache_hit");
        } else {
            // PMAT-083: Prefill ALL tokens and extract first predicted token from LM head.
            let greedy = config.temperature == 0.0 || config.top_k == 1;
            let prefill_count = if greedy { prompt.len() } else { prompt.len() - 1 };
            prefill_first_token = if prefill_count > 0 {
                self.run_prefill(prompt, &mut cache, prefill_count, ttft_trace, greedy)?
            } else {
                None
            };
            ttft_mark!("prefill");
        }

        // PMAT-109: Graph persistence — do NOT clear decode graph here.
        // init_prefill_workspace clears the graph only when it actually reallocates
        // (longer prompt exceeds buffer_capacity). When PAR-200 fires (same/shorter
        // prompt), workspace buffer addresses are stable → graph replay is valid.
        // This eliminates cuGraphExecDestroy from every request's TTFT critical path,
        // fixing the bimodal tail (95% at 20ms, 5% at 42ms → uniform).
        // Supersedes: PMAT-085, CORRECTNESS-013, PMAT-107 (all addressed by PAR-200).

        // Generate tokens
        let mut position;
        let mut last_token;
        let first_token_offset;

        if let Some(first_tok) = prefill_first_token {
            // PMAT-083: First token came from prefill LM head — skip first decode
            position = prompt.len(); // KV cache has ALL prompt positions
            last_token = first_tok;
            first_token_offset = 0; // First loop iteration generates second output token

            // Emit the first token immediately
            tokens.push(first_tok);
            if config.stop_tokens.contains(&first_tok) {
                return Ok(tokens);
            }
            if !on_token(first_tok) {
                return Ok(tokens);
            }
            ttft_mark!("first_token(prefill)");
        } else {
            // Original path: last prompt token feeds into first decode
            position = prompt.len() - 1;
            last_token = prompt[prompt.len() - 1];
            first_token_offset = 0;
        }

        let max_decode = if prefill_first_token.is_some() {
            config.max_tokens.saturating_sub(1) // Already emitted one token
        } else {
            config.max_tokens
        };
        for _token_num in 0..max_decode {
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                let tok = self.forward_gpu_resident_to_token_id(last_token, &mut cache, position)?;
                if _token_num == first_token_offset && prefill_first_token.is_none() {
                    ttft_mark!("first_decode");
                }
                tok
            } else {
                let logits = self.forward_gpu_resident(last_token, &mut cache, position)?;
                OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // PAR-112: Call the streaming callback IMMEDIATELY after generating each token
            // If callback returns false, stop generation early
            if !on_token(next_token) {
                break;
            }

            last_token = next_token;
            position += 1;
        }

        // realizr#199 (PMAT-450): Insert PROMPT KV into prefix cache.
        // CRITICAL: snapshot prompt.len() positions, NOT the full KV (which includes
        // generated tokens). On cache hit we restore prompt KV and decode from scratch.
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
                    eprintln!("[PMAT-450] PREFIX CACHE INSERT: {} prompt tokens ({} layers)", prompt.len(), num_layers);
                }
                Err(e) => {
                    eprintln!("[PMAT-450] PREFIX CACHE SNAPSHOT ERROR: {}", e);
                }
            }
            // Restore original KV lengths
            for &(l, len) in &current_lens {
                self.executor.set_kv_cache_len(l, len);
            }
        }

        Ok(tokens)
    }

    /// PAR-106: Batched GPU-resident generation for continuous batching
    ///
    /// Processes multiple prompts concurrently with true weight sharing:
    /// - Single weight read produces N tokens (one per active request)
    /// - Target: 400 tok/s (2x Ollama) with 4+ concurrent requests
    ///
    /// Key optimization: Uses `forward_batch_with_cache_cuda_native` which
    /// amortizes memory bandwidth across the batch.
    pub fn generate_batch_gpu_resident(
        &mut self,
        prompts: &[Vec<u32>],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<Vec<u32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batch_gpu_resident".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }

        let num_prompts = prompts.len();
        let max_prompt_len = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let max_seq_len = max_prompt_len + config.max_tokens;

        // PAR-045: Create KV caches with GQA-aware dimensions
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let mut caches: Vec<OwnedQuantizedKVCache> = (0..num_prompts)
            .map(|_| OwnedQuantizedKVCache::new(self.model.config.num_layers, kv_dim, max_seq_len))
            .collect();

        // Reset GPU KV cache positions
        self.executor.reset_kv_cache_gpu();

        // Initialize token sequences
        let mut sequences: Vec<Vec<u32>> = prompts.to_vec();
        let mut done: Vec<bool> = vec![false; num_prompts];

        // Prefill: Process each prompt's tokens (can't batch different lengths easily)
        for (prompt_idx, prompt) in prompts.iter().enumerate() {
            for (pos, &token_id) in prompt.iter().enumerate() {
                if pos < prompt.len() - 1 {
                    // PAR-106: Use single-token forward for prefill
                    // (batched prefill would require padding/masking complexity)
                    let _ = self.forward_gpu_resident(token_id, &mut caches[prompt_idx], pos)?;
                }
            }
        }

        // Track positions per prompt (filter empty prompts)
        let mut positions: Vec<usize> = prompts
            .iter()
            .map(|p| p.len().saturating_sub(1))
            .collect();
        let mut last_tokens: Vec<u32> = prompts
            .iter()
            .map(|p| p.last().copied().unwrap_or(0))
            .collect();

        // PAR-106: Batched decode loop with weight sharing
        for _gen_idx in 0..config.max_tokens {
            // Collect active prompts
            let active_indices: Vec<usize> = (0..num_prompts).filter(|&i| !done[i]).collect();

            if active_indices.is_empty() {
                break;
            }

            // PAR-106/PAR-108: Sequential CUDA graphs outperform batched CPU path.
            // The batched GEMV kernel is 15x faster, but CUDA graphs amortize
            // kernel launch overhead which is more impactful. Batched path achieves
            // ~225 tok/s vs ~360 tok/s for sequential graphs.
            //
            // To achieve 2x Ollama (400 tok/s), need multi-token CUDA graph capture
            // that batches M tokens into a single graph execution.
            for &prompt_idx in &active_indices {
                let next_token = self.forward_gpu_resident_to_token_id(
                    last_tokens[prompt_idx],
                    &mut caches[prompt_idx],
                    positions[prompt_idx],
                )?;

                if config.stop_tokens.contains(&next_token) {
                    done[prompt_idx] = true;
                } else {
                    sequences[prompt_idx].push(next_token);
                    last_tokens[prompt_idx] = next_token;
                    positions[prompt_idx] += 1;

                    if sequences[prompt_idx].len() >= max_seq_len {
                        done[prompt_idx] = true;
                    }
                }
            }
        }

        Ok(sequences)
    }
}

include!("generate_2.rs");
