impl OwnedQuantizedModelCuda {

    /// PAR-014: Fused matmul with explicit cache key
    ///
    /// Same as `fused_matmul_cuda` but accepts an explicit cache key, allowing
    /// the caller to use the original weight pointer for caching even when
    /// working with cloned weight data.
    fn fused_matmul_cuda_with_key(
        &mut self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        cache_key: &str,
    ) -> Result<Vec<f32>> {
        // PMAT-181: Support both Q4_K and Q6_K on GPU.
        // Q4_K_M GGUF files use Q4_K for most weights but Q6_K for
        // V projection, FFN down, and token embedding. Without Q6K GPU
        // support, these fall back to CPU causing precision divergence
        // that produces semi-coherent but wrong output (five-whys root cause).
        const GGUF_TYPE_Q4_K: u32 = 12;
        const GGUF_TYPE_Q6_K: u32 = 14;

        if weight.qtype != GGUF_TYPE_Q4_K && weight.qtype != GGUF_TYPE_Q6_K {
            // Fallback to CPU for unsupported quantization types
            return self.model.fused_matmul(input, weight);
        }

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;

        if input.len() != in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "PAR-014: Input length {} doesn't match weight in_dim {}",
                    input.len(),
                    in_dim
                ),
            });
        }

        let mut output = vec![0.0f32; out_dim];

        // Lazy cache - upload weight on first use
        if !self.executor.has_quantized_weights(cache_key) {
            self.executor
                .load_quantized_weights(cache_key, &weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: format!("cuda_q{}k_cache", if weight.qtype == GGUF_TYPE_Q4_K { 4 } else { 6 }),
                    reason: format!("Failed to cache weights: {e}"),
                })?;
        }

        // Dispatch to appropriate kernel based on quantization type
        if weight.qtype == GGUF_TYPE_Q4_K {
            self.executor
                .q4k_gemv_cached(cache_key, input, &mut output, out_dim as u32, in_dim as u32)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "q4k_gemv_cached".to_string(),
                    reason: format!("CUDA Q4_K GEMV failed: {e}"),
                })?;
        } else {
            // Q6_K path (PMAT-181)
            self.executor
                .q6k_gemv_cached(cache_key, input, &mut output, out_dim as u32, in_dim as u32)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "q6k_gemv_cached".to_string(),
                    reason: format!("CUDA Q6_K GEMV failed: {e}"),
                })?;
        }

        Ok(output)
    }

    /// QKV matmul with CUDA - handles both fused and separate Q/K/V
    ///
    /// Five Whys Root Cause Fix: Supports TinyLlama and other LLaMA-style models
    fn qkv_matmul_cuda(&mut self, input: &[f32], qkv: &OwnedQKVWeights) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul_cuda(input, weight),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Compute Q, K, V separately then concatenate
                let q_out = self.fused_matmul_cuda(input, q)?;
                let k_out = self.fused_matmul_cuda(input, k)?;
                let v_out = self.fused_matmul_cuda(input, v)?;

                // Concatenate Q, K, V
                let mut output = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// PAR-014: QKV matmul with explicit cache key for fused weights
    ///
    /// Same as `qkv_matmul_cuda` but accepts a cache key for the fused case.
    fn qkv_matmul_cuda_with_key(
        &mut self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        cache_key: &str,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => {
                self.fused_matmul_cuda_with_key(input, weight, cache_key)
            },
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // For separate Q/K/V, we still use the cloned pointers
                // (less critical since these are already separate tensors)
                let q_out = self.fused_matmul_cuda(input, q)?;
                let k_out = self.fused_matmul_cuda(input, k)?;
                let v_out = self.fused_matmul_cuda(input, v)?;

                let mut output = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }
}
