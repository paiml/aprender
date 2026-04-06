
#[cfg(feature = "cuda")]
impl AprV2ModelCuda {

    /// GPU GEMM helper: C[m, n] = A[m, k] × B[k, n]
    ///
    /// Phase 45: Routes through test_executor when present for testability.
    #[allow(clippy::many_single_char_names)] // Standard matrix notation
    fn gemm_gpu(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Phase 45: Route through test executor if present
        if let Some(ref mut test_exec) = self.test_executor {
            return test_exec.matmul(a, b, m, k, n);
        }

        // Normal CUDA path
        let mut c = vec![0.0f32; m * n];
        self.executor
            .gemm(a, b, &mut c, m as u32, n as u32, k as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "GPU GEMM".to_string(),
                reason: format!("CUDA GEMM failed: {e}"),
            })?;
        Ok(c)
    }

    /// GPU GEMM with cached weight: C[m, n] = A[m, k] × B_cached[k, n]
    ///
    /// Uses pre-cached weight matrix B to avoid repeated GPU uploads.
    /// Dispatches to F32 GEMM or quantized GEMV based on weight cache location.
    ///
    /// PMAT-222: Added quantized dispatch for GGUF-sourced APR models.
    /// Phase 45: When test_executor is present, falls back to returning zeros.
    #[allow(clippy::many_single_char_names)] // Standard matrix notation
    /// Dispatch quantized GEMV on GPU for Q4_K or Q6_K weights.
    fn dispatch_quantized_gemv(
        &mut self,
        weight_name: &str,
        a: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        qtype: u32,
    ) -> Result<()> {
        match qtype {
            12 => {
                if m == 1 {
                    self.executor
                        .q4k_gemv_cached(weight_name, a, c, n as u32, k as u32)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "GPU Q4K GEMV cached".to_string(),
                            reason: format!("CUDA Q4K GEMV '{}' failed: {e}", weight_name),
                        })
                } else {
                    self.executor
                        .batched_q4k_gemv_cached(weight_name, a, c, m as u32, k as u32, n as u32)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "GPU Q4K batched GEMV cached".to_string(),
                            reason: format!("CUDA batched Q4K GEMV '{}' failed: {e}", weight_name),
                        })
                }
            }
            14 => {
                if m == 1 {
                    self.executor
                        .q6k_gemv_cached(weight_name, a, c, n as u32, k as u32)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "GPU Q6K GEMV cached".to_string(),
                            reason: format!("CUDA Q6K GEMV '{}' failed: {e}", weight_name),
                        })
                } else {
                    for row in 0..m {
                        let row_input = &a[row * k..(row + 1) * k];
                        let row_output = &mut c[row * n..(row + 1) * n];
                        self.executor
                            .q6k_gemv_cached(weight_name, row_input, row_output, n as u32, k as u32)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "GPU Q6K GEMV cached (batched)".to_string(),
                                reason: format!("CUDA Q6K GEMV '{}' row {row} failed: {e}", weight_name),
                            })?;
                    }
                    Ok(())
                }
            }
            _ => {
                self.executor
                    .gemm_b_cached(weight_name, a, c, m as u32, n as u32, k as u32)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "GPU GEMM cached (qtype fallback)".to_string(),
                        reason: format!("CUDA GEMM '{}' qtype={qtype} failed: {e}", weight_name),
                    })
            }
        }
    }

    fn gemm_cached_gpu(
        &mut self,
        weight_name: &str,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Phase 45: Test executor can't use cached weights, return zeros
        if self.test_executor.is_some() {
            return Ok(vec![0.0f32; m * n]);
        }

        // PMAT-222: Check if weight is quantized (GGUF-sourced APR) or F32 (SafeTensors APR)
        if self.executor.has_quantized_weights(weight_name) {
            // R-03 (Meyer DbC): Q4_K (GGML type 12) is the default for quantized APR files.
            const GGML_TYPE_Q4_K: u32 = 12;
            let qtype = self
                .executor
                .get_quantized_weight_type(weight_name)
                .unwrap_or(GGML_TYPE_Q4_K);
            let mut c = vec![0.0f32; m * n];
            self.dispatch_quantized_gemv(weight_name, a, &mut c, m, k, n, qtype)?;
            Ok(c)
        } else {
            // F32 path: standard GEMM with cached weights
            let mut c = vec![0.0f32; m * n];
            self.executor
                .gemm_b_cached(weight_name, a, &mut c, m as u32, n as u32, k as u32)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "GPU GEMM cached".to_string(),
                    reason: format!("CUDA GEMM with cached weight '{}' failed: {e}", weight_name),
                })?;
            Ok(c)
        }
    }

    /// Check if a weight is cached on GPU.
    ///
    /// Phase 45: Returns false when test_executor is present, forcing the
    /// uncached GEMM path which routes through the test executor.
    ///
    /// Issue #45 fix: Check BOTH weight_cache (f32) and quantized_weight_cache
    /// (Q4_K/Q5_K/Q6_K). APR models use quantized weights, so checking only
    /// weight_cache was causing cache misses and 278x slowdown.
    fn has_cached_weight(&self, name: &str) -> bool {
        if self.test_executor.is_some() {
            return false; // Force uncached path for testing
        }
        // Check both f32 cache and quantized cache
        self.executor.has_weights(name) || self.executor.has_quantized_weights(name)
    }

    /// GPU-accelerated token generation.
    ///
    /// Generates tokens autoregressively using GPU acceleration.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial prompt token IDs
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `eos_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// Complete token sequence including prompt and generated tokens.
    pub fn generate_cuda(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        // GH-282: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // GH-284: Reset KV cache to prevent cross-request position overflow.
        // Without this, kv_position accumulates across HTTP requests, causing
        // "KV cache overflow - max_len=2048, trying to add position 2049" warnings
        // and degrading TPS (1.37 → 0.91 over successive requests).
        self.reset_kv_cache();

        let mut tokens = prompt.to_vec();

        for _ in 0..max_new_tokens {
            // Forward pass
            let logits = self.forward_cuda(&tokens)?;

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(eos_id, |(idx, _)| idx as u32);

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// GPU-accelerated forward pass for single token with KV cache.
    ///
    /// This is the optimized decode path that reuses cached K/V values
    /// from previous positions for O(1) attention per token.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token ID to process
    /// * `position` - Current position in sequence
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size` for next token prediction.
    pub fn forward_single_cuda(&mut self, token_id: u32, _position: usize) -> Result<Vec<f32>> {
        // Uses full forward pass; KV cache optimization available via GGUF path
        self.forward_cuda(&[token_id])
    }

    /// GPU-accelerated generation with KV cache.
    ///
    /// Uses the optimized single-token decode path after prefill.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial prompt token IDs
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `eos_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// Complete token sequence including prompt and generated tokens.
    pub fn generate_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        // GH-282: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // GH-260: Reset KV cache before each generation.
        // kv_position=0 prevents stale positions from previous request.
        // PMAT-042: Preserve CUDA graph across requests (same pattern as GGUF
        // generate_1/generate_2 which call reset_kv_cache_gpu only).
        // Graph is position-independent: reads position/seq_len from GPU buffers
        // updated via copy_from_host before each replay.
        self.reset_kv_cache();

        // PMAT-113-F: Diagnostic tracing for logit verification
        let trace_enabled = std::env::var("APR_TRACE_LOGITS").is_ok();

        // PMAT-114: Fixed prefill - KEEP logits from last token (like GGUF)
        // The logits from processing token[n-1] at position n-1 predict token[n]
        // This matches the GGUF pattern in generate_with_cache (lines 171-183)
        let mut tokens = prompt.to_vec();
        let mut logits = self.forward_cuda(&tokens)?;

        // Decode: generate one token at a time
        // First iteration uses logits from prefill (no extra forward needed)
        for i in 0..max_new_tokens {
            // For subsequent tokens, run forward pass on the newly generated token
            if i > 0 {
                let position = tokens.len();
                let last_token = *tokens.last().unwrap_or(&1);
                logits = self.forward_single_cuda(last_token, position)?;
            }

            // PMAT-113-F: Diagnostic tracing for Q1-Q3
            if trace_enabled && i < 3 {
                let nan_count = logits.iter().filter(|x| x.is_nan()).count();
                let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
                let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = logits.iter().sum();
                let mean = sum / logits.len() as f32;
                let variance: f32 =
                    logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32;

                eprintln!("[PMAT-113-F] Token {}: logits stats:", i);
                eprintln!(
                    "  NaN: {}, Inf: {}, len: {}",
                    nan_count,
                    inf_count,
                    logits.len()
                );
                eprintln!(
                    "  min: {:.4}, max: {:.4}, mean: {:.4}, var: {:.4}",
                    min, max, mean, variance
                );
                eprintln!(
                    "  kv_position: {}, kv_cache_len[0]: {:?}",
                    self.kv_position,
                    self.executor.kv_cache_len(0)
                );

                // Show top 5 token predictions
                let mut indexed: Vec<_> = logits.iter().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!(
                    "  Top 5 tokens: {:?}",
                    indexed
                        .iter()
                        .take(5)
                        .map(|(i, v)| (*i, **v))
                        .collect::<Vec<_>>()
                );
            }

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(eos_id, |(idx, _)| idx as u32);

            if trace_enabled && i < 3 {
                eprintln!(
                    "  Selected token: {} (logit: {:.4})",
                    next_token,
                    logits.get(next_token as usize).unwrap_or(&0.0)
                );
            }

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

include!("cuda_model_init.rs");
include!("weight.rs");
include!("cuda_streaming_weights.rs");
include!("forward_cuda_to_token.rs");
include!("forward_cuda.rs");
