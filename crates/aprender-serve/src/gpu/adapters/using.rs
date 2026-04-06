impl GpuModelQ4 {
    /// Execute forward pass using Q4_0 kernels
    ///
    /// # Arguments
    ///
    /// * `executor` - CUDA executor with cached weights
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction
    #[cfg(feature = "cuda")]
    pub fn forward(&self, executor: &mut CudaExecutor, token_ids: &[usize]) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;

        // 1. Embed tokens (CPU - fast lookup)
        let seq_len = token_ids.len();
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let start = token_id * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                // Out of vocab, use zeros
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Upload hidden state to GPU
        let mut hidden_gpu = GpuBuffer::from_host(executor.context(), &hidden).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to upload hidden: {e}"),
            }
        })?;

        // 3. Pass through transformer layers
        for layer_idx in 0..self.num_layers {
            hidden_gpu = self.forward_layer(executor, &hidden_gpu, layer_idx, seq_len)?;
        }

        // 4. Final layer norm (GPU)
        // PAR-023: Use cached gamma pointer to avoid CPU roundtrip
        let (output_gamma_ptr, output_gamma_len) = executor
            .get_rmsnorm_gamma_ptr("apr.output_norm")
            .ok_or_else(|| RealizarError::GpuError {
                reason: "apr.output_norm not cached on GPU".to_string(),
            })?;

        let normed_gpu = executor
            .rmsnorm_gpu_ptr(
                &hidden_gpu,
                output_gamma_ptr,
                output_gamma_len,
                hidden_dim as u32,
                self.config.eps,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Output RMSNorm failed: {e}"),
            })?;

        // 5. LM head projection (GPU, Q4_0)

        let logits_gpu = GpuBuffer::new(executor.context(), vocab_size).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to allocate logits: {e}"),
            }
        })?;

        let lm_head_ptr =
            executor
                .get_quantized_weight_ptr("lm_head")
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("lm_head not cached: {e}"),
                })?;

        executor
            .q4_0_gemv_into(
                lm_head_ptr,
                &normed_gpu,
                &logits_gpu,
                vocab_size as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("LM head GEMV failed: {e}"),
            })?;

        // 6. Sync and return
        executor
            .synchronize()
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Sync failed: {e}"),
            })?;

        gpu_to_host(&logits_gpu)
    }

    /// Execute single transformer layer
    #[cfg(feature = "cuda")]
    fn forward_layer(
        &self,
        executor: &mut CudaExecutor,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<GpuBuffer<f32>> {
        contract_pre_layer_composition!();
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        // GH-479: Use explicit_head_dim if set (Qwen3 head_dim != hidden/heads)
        let head_dim = self.config.explicit_head_dim.unwrap_or(hidden_dim / num_heads);
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = hidden_dim + 2 * kv_dim;

        // 1. Pre-attention RMSNorm (GPU)
        // PAR-023: Use cached gamma pointer to avoid CPU roundtrip
        let attn_norm_name = format!("apr.layer_{layer_idx}.attn_norm");
        let (attn_gamma_ptr, attn_gamma_len) = executor
            .get_rmsnorm_gamma_ptr(&attn_norm_name)
            .ok_or_else(|| RealizarError::GpuError {
                reason: format!("{attn_norm_name} not cached on GPU"),
            })?;

        let normed_gpu = executor
            .rmsnorm_gpu_ptr(
                input,
                attn_gamma_ptr,
                attn_gamma_len,
                hidden_dim as u32,
                self.config.eps,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Attention RMSNorm failed: {e}"),
            })?;

        // 3. QKV projection (Q4_0)
        let qkv_gpu =
            GpuBuffer::new(executor.context(), qkv_dim).map_err(|e| RealizarError::GpuError {
                reason: format!("Failed to allocate QKV: {e}"),
            })?;

        let qkv_name = format!("layer_{layer_idx}.attn.qkv");
        let qkv_ptr =
            executor
                .get_quantized_weight_ptr(&qkv_name)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("{qkv_name} not cached: {e}"),
                })?;

        executor
            .q4_0_gemv_into(
                qkv_ptr,
                &normed_gpu,
                &qkv_gpu,
                qkv_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("QKV GEMV failed: {e}"),
            })?;

        // 4. Attention (CPU for single-token - GPU launch overhead exceeds benefit)
        let qkv = gpu_to_host(&qkv_gpu)?;

        // Apply RoPE to Q and K before attention
        // For seq_len tokens at layer 0, positions are [0, 1, 2, ... seq_len-1]
        let mut qkv_with_rope = qkv.clone();
        self.apply_rope_to_qkv(
            &mut qkv_with_rope,
            seq_len,
            hidden_dim,
            num_heads,
            num_kv_heads,
        );

        let attn_out =
            self.attention_cpu(&qkv_with_rope, seq_len, hidden_dim, num_heads, num_kv_heads);

        // 5. Output projection (Q4_0)
        let attn_out_gpu = GpuBuffer::from_host(executor.context(), &attn_out).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to upload attn_out: {e}"),
            }
        })?;

        let out_gpu = GpuBuffer::new(executor.context(), hidden_dim).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to allocate out: {e}"),
            }
        })?;

        let out_name = format!("layer_{layer_idx}.attn.out");
        let out_ptr =
            executor
                .get_quantized_weight_ptr(&out_name)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("{out_name} not cached: {e}"),
                })?;

        executor
            .q4_0_gemv_into(
                out_ptr,
                &attn_out_gpu,
                &out_gpu,
                hidden_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Out GEMV failed: {e}"),
            })?;

        // 6. Residual connection (GPU)
        // PAR-023: Keep data on GPU to avoid roundtrip
        let residual1 = executor
            .residual_add_gpu(input, &out_gpu, hidden_dim as u32)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Residual add failed: {e}"),
            })?;

        // 7. FFN norm (GPU)
        let ffn_norm_name = format!("apr.layer_{layer_idx}.ffn_norm");
        let (ffn_gamma_ptr, ffn_gamma_len) = executor
            .get_rmsnorm_gamma_ptr(&ffn_norm_name)
            .ok_or_else(|| RealizarError::GpuError {
                reason: format!("{ffn_norm_name} not cached on GPU"),
            })?;

        let ffn_input_gpu = executor
            .rmsnorm_gpu_ptr(
                &residual1,
                ffn_gamma_ptr,
                ffn_gamma_len,
                hidden_dim as u32,
                self.config.eps,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("FFN RMSNorm failed: {e}"),
            })?;

        // 8. FFN (GPU, Q4_0)

        let ffn_out = if self.has_gate {
            // SwiGLU: gate * up * silu
            self.ffn_swiglu_gpu(executor, &ffn_input_gpu, layer_idx)?
        } else {
            // Standard FFN: up -> activation -> down
            self.ffn_standard_gpu(executor, &ffn_input_gpu, layer_idx)?
        };

        // 9. Final residual (GPU)
        // PAR-023: Keep data on GPU - residual1 + ffn_out
        let result = executor
            .residual_add_gpu(&residual1, &ffn_out, hidden_dim as u32)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Final residual add failed: {e}"),
            });
        if result.is_ok() {
            contract_post_layer_composition!(&());
        }
        result
    }

    /// SwiGLU FFN using Q4_0 kernels
    #[cfg(feature = "cuda")]
    fn ffn_swiglu_gpu(
        &self,
        executor: &mut CudaExecutor,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
    ) -> Result<GpuBuffer<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // Gate projection
        let gate_gpu = GpuBuffer::new(executor.context(), intermediate_dim).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to allocate gate: {e}"),
            }
        })?;

        let gate_name = format!("layer_{layer_idx}.ffn.gate");
        let gate_ptr =
            executor
                .get_quantized_weight_ptr(&gate_name)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("{gate_name} not cached: {e}"),
                })?;

        executor
            .q4_0_gemv_into(
                gate_ptr,
                input,
                &gate_gpu,
                intermediate_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Gate GEMV failed: {e}"),
            })?;

        // Up projection
        let up_gpu = GpuBuffer::new(executor.context(), intermediate_dim).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to allocate up: {e}"),
            }
        })?;

        let up_name = format!("layer_{layer_idx}.ffn.up");
        let up_ptr =
            executor
                .get_quantized_weight_ptr(&up_name)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("{up_name} not cached: {e}"),
                })?;

        executor
            .q4_0_gemv_into(
                up_ptr,
                input,
                &up_gpu,
                intermediate_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Up GEMV failed: {e}"),
            })?;

        // SwiGLU activation (GPU - fused kernel PAR-023)
        // Eliminates 2x GPU→CPU transfers + CPU compute + 1x CPU→GPU transfer
        let activated_gpu = executor
            .fused_swiglu_gpu(&gate_gpu, &up_gpu, intermediate_dim as u32)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Fused SwiGLU failed: {e}"),
            })?;

        let down_gpu = GpuBuffer::new(executor.context(), hidden_dim).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to allocate down: {e}"),
            }
        })?;

        let down_name = format!("layer_{layer_idx}.ffn.down");
        let down_ptr =
            executor
                .get_quantized_weight_ptr(&down_name)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("{down_name} not cached: {e}"),
                })?;

        executor
            .q4_0_gemv_into(
                down_ptr,
                &activated_gpu,
                &down_gpu,
                hidden_dim as u32,
                intermediate_dim as u32,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Down GEMV failed: {e}"),
            })?;

        Ok(down_gpu)
    }

    /// Standard FFN (GELU) using Q4_0 kernels
    #[cfg(feature = "cuda")]
    fn ffn_standard_gpu(
        &self,
        executor: &mut CudaExecutor,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
    ) -> Result<GpuBuffer<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // Up projection
        let up_gpu = GpuBuffer::new(executor.context(), intermediate_dim).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to allocate up: {e}"),
            }
        })?;

        let up_name = format!("layer_{layer_idx}.ffn.up");
        let up_ptr =
            executor
                .get_quantized_weight_ptr(&up_name)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("{up_name} not cached: {e}"),
                })?;

        executor
            .q4_0_gemv_into(
                up_ptr,
                input,
                &up_gpu,
                intermediate_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Up GEMV failed: {e}"),
            })?;

        // GELU activation (GPU - in place)
        // Eliminates 2x PCIe transfers (GPU→CPU download + CPU→GPU upload)
        executor
            .gelu_gpu(&up_gpu, intermediate_dim as u32)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("GELU GPU failed: {e}"),
            })?;

        let down_gpu = GpuBuffer::new(executor.context(), hidden_dim).map_err(|e| {
            RealizarError::GpuError {
                reason: format!("Failed to allocate down: {e}"),
            }
        })?;

        let down_name = format!("layer_{layer_idx}.ffn.down");
        let down_ptr =
            executor
                .get_quantized_weight_ptr(&down_name)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("{down_name} not cached: {e}"),
                })?;

        executor
            .q4_0_gemv_into(
                down_ptr,
                &up_gpu, // up_gpu now contains GELU-activated values (in-place)
                &down_gpu,
                hidden_dim as u32,
                intermediate_dim as u32,
            )
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Down GEMV failed: {e}"),
            })?;

        Ok(down_gpu)
    }

    /// RMSNorm in place
    pub(crate) fn rms_norm_inplace(&self, x: &mut [f32], weight: &[f32]) {
        let eps = self.config.eps;
        let n = x.len();

        // Calculate RMS
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let scale = 1.0 / rms;

        // Normalize and apply weight
        for (i, v) in x.iter_mut().enumerate() {
            *v = *v * scale * weight.get(i).copied().unwrap_or(1.0);
        }
    }
}
