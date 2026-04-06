impl OwnedQuantizedModelCached {
    /// Create a new cached model wrapper
    ///
    /// The scheduler is lazily initialized on first GPU operation.
    /// PARITY-103: Also initializes CudaScheduler when CUDA feature is enabled.
    pub fn new(model: OwnedQuantizedModel) -> Self {
        Self {
            model,
            scheduler: std::cell::RefCell::new(None),
            #[cfg(feature = "cuda")]
            cuda_scheduler: std::cell::RefCell::new(None),
        }
    }

    /// Get or create the cached scheduler (wgpu backend)
    ///
    /// # Errors
    /// Returns error if scheduler creation fails
    fn get_scheduler(&self) -> Result<std::cell::RefMut<'_, crate::gpu::HybridScheduler>> {
        use crate::gpu::HybridScheduler;

        let mut scheduler_opt = self.scheduler.borrow_mut();

        // Initialize if not already done
        if scheduler_opt.is_none() {
            let new_scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "HybridScheduler::with_threshold".to_string(),
                    reason: format!("GPU scheduler initialization failed: {e}"),
                }
            })?;
            *scheduler_opt = Some(new_scheduler);
        }

        // Return mutable reference to the scheduler
        Ok(std::cell::RefMut::map(scheduler_opt, |opt| {
            opt.as_mut().expect("scheduler should be initialized")
        }))
    }

    /// PARITY-103: Get or create the cached CUDA scheduler
    ///
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly.
    /// Returns None if CUDA is not available.
    ///
    /// # Errors
    /// Returns error if CUDA scheduler creation fails
    #[cfg(feature = "cuda")]
    fn get_cuda_scheduler(
        &self,
    ) -> Result<Option<std::cell::RefMut<'_, crate::gpu::CudaScheduler>>> {
        use crate::gpu::CudaScheduler;

        let mut scheduler_opt = self.cuda_scheduler.borrow_mut();

        // Initialize if not already done
        if scheduler_opt.is_none() {
            match CudaScheduler::new() {
                Ok(new_scheduler) => {
                    *scheduler_opt = Some(new_scheduler);
                },
                Err(_) => {
                    // CUDA not available, return None (will fallback to wgpu)
                    return Ok(None);
                },
            }
        }

        // Return mutable reference to the scheduler
        Ok(Some(std::cell::RefMut::map(scheduler_opt, |opt| {
            opt.as_mut().expect("cuda_scheduler should be initialized")
        })))
    }

    /// Forward pass with cached scheduler (IMP-112)
    ///
    /// Uses the cached HybridScheduler instead of creating a new one,
    /// eliminating ~300ms initialization overhead per call.
    ///
    /// # Arguments
    /// * `token_ids` - Batch of input token IDs
    ///
    /// # Returns
    /// Logits for all positions [batch_size * vocab_size]
    ///
    /// # Errors
    /// Returns error if GPU operations fail
    /// PARITY-103: Forward pass preferring CUDA over wgpu
    ///
    /// Uses CudaScheduler when available to bypass wgpu 256MB buffer limit.
    /// Falls back to HybridScheduler (wgpu) if CUDA is not available.
    pub fn forward_batch_gpu_cached(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let batch_size = token_ids.len();
        let hidden_dim = self.model.config.hidden_dim;
        let vocab_size = self.model.config.vocab_size;

        // 1. Token embedding lookup
        let mut hidden = self.model.embed(token_ids);

        // 2. Process through transformer layers
        for layer in &self.model.layers {
            // Pre-attention LayerNorm
            let normed = self.model.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.model.config.eps,
            );

            // PARITY-103: QKV projection preferring CUDA
            let qkv =
                self.batch_qkv_matmul_gpu(&normed, &layer.qkv_weight, batch_size, hidden_dim)?;

            // Split Q, K, V
            let qkv_dim = qkv.len() / batch_size;
            let q_dim = hidden_dim;
            let kv_dim = (qkv_dim - q_dim) / 2;

            let mut q_all = Vec::with_capacity(batch_size * q_dim);
            let mut k_all = Vec::with_capacity(batch_size * kv_dim);
            let mut v_all = Vec::with_capacity(batch_size * kv_dim);

            for pos in 0..batch_size {
                let qkv_start = pos * qkv_dim;
                q_all.extend_from_slice(&qkv[qkv_start..qkv_start + q_dim]);
                k_all.extend_from_slice(&qkv[qkv_start + q_dim..qkv_start + q_dim + kv_dim]);
                v_all.extend_from_slice(&qkv[qkv_start + q_dim + kv_dim..qkv_start + qkv_dim]);
            }

            // Attention (still uses HybridScheduler for now - attention is memory-bound)
            let mut scheduler = self.get_scheduler()?;
            let attn_out = self.batched_causal_attention_with_scheduler(
                &q_all,
                &k_all,
                &v_all,
                batch_size,
                &mut scheduler,
            )?;
            drop(scheduler); // Release borrow before next CUDA call

            // PARITY-103: Output projection preferring CUDA
            let projected = self.batch_matmul_gpu_prefer_cuda(
                &attn_out,
                &layer.attn_output_weight,
                batch_size,
                hidden_dim,
                layer.attn_output_weight.out_dim,
            )?;

            // Residual
            for i in 0..hidden.len() {
                hidden[i] += projected[i];
            }

            // FFN
            let ffn_normed = self.model.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.model.config.eps,
            );

            // PARITY-103: FFN up projection preferring CUDA
            let mut ffn_hidden = self.batch_matmul_gpu_prefer_cuda(
                &ffn_normed,
                &layer.ffn_up_weight,
                batch_size,
                hidden_dim,
                layer.ffn_up_weight.out_dim,
            )?;

            self.model.gelu(&mut ffn_hidden);

            // PARITY-103: FFN down projection preferring CUDA
            let ffn_output = self.batch_matmul_gpu_prefer_cuda(
                &ffn_hidden,
                &layer.ffn_down_weight,
                batch_size,
                layer.ffn_up_weight.out_dim,
                hidden_dim,
            )?;

            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        let normed = self.model.layer_norm(
            &hidden,
            &self.model.output_norm_weight,
            self.model.output_norm_bias.as_deref(),
            self.model.config.eps,
        );

        // PARITY-103: LM head projection preferring CUDA
        let logits = self.batch_matmul_gpu_prefer_cuda(
            &normed,
            &self.model.lm_head_weight,
            batch_size,
            hidden_dim,
            vocab_size,
        )?;

        Ok(logits)
    }

    /// Batch matmul with provided scheduler (wgpu backend)
    fn batch_matmul_gpu_with_scheduler(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        // Dequantize weight
        let weight_f32 = self.model.dequantize_weight(weight)?;

        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        // GPU matmul
        scheduler
            .matmul(input, &weight_f32, batch_size, in_dim, out_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "batch_matmul_gpu_with_scheduler".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            })
    }

    /// PARITY-103: Batch matmul preferring CUDA over wgpu
    ///
    /// Tries CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    /// This bypasses the wgpu 256MB buffer limit that was blocking GPU batch inference.
    #[cfg(feature = "cuda")]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        // Dequantize weight
        let weight_f32 = self.model.dequantize_weight(weight)?;

        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        // Try CUDA first (no buffer size limits)
        if let Ok(Some(mut cuda_sched)) = self.get_cuda_scheduler() {
            return cuda_sched
                .matmul(input, &weight_f32, batch_size, in_dim, out_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                    reason: format!("CUDA matmul failed: {e}"),
                });
        }

        // Fallback to wgpu (may hit 256MB limit for large batches)
        let mut scheduler = self.get_scheduler()?;
        scheduler
            .matmul(input, &weight_f32, batch_size, in_dim, out_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            })
    }

    /// PARITY-103: Batch matmul preferring CUDA (non-CUDA fallback)
    #[cfg(not(feature = "cuda"))]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut scheduler = self.get_scheduler()?;
        self.batch_matmul_gpu_with_scheduler(
            input,
            weight,
            batch_size,
            in_dim,
            out_dim,
            &mut scheduler,
        )
    }

    /// Batch QKV matmul for GPU paths - handles both fused and separate Q/K/V
    ///
    /// Five Whys Root Cause Fix: This method handles both tensor layouts for GPU batch ops
    #[cfg(feature = "gpu")]
    fn batch_qkv_matmul_gpu(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        batch_size: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.batch_matmul_gpu_prefer_cuda(
                input,
                weight,
                batch_size,
                hidden_dim,
                weight.out_dim,
            ),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Compute Q, K, V separately then concatenate
                let q_out =
                    self.batch_matmul_gpu_prefer_cuda(input, q, batch_size, hidden_dim, q.out_dim)?;
                let k_out =
                    self.batch_matmul_gpu_prefer_cuda(input, k, batch_size, hidden_dim, k.out_dim)?;
                let v_out =
                    self.batch_matmul_gpu_prefer_cuda(input, v, batch_size, hidden_dim, v.out_dim)?;

                // Interleave Q, K, V for each position in batch
                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(batch_size * qkv_dim);
                for b in 0..batch_size {
                    output.extend_from_slice(&q_out[b * q.out_dim..(b + 1) * q.out_dim]);
                    output.extend_from_slice(&k_out[b * k.out_dim..(b + 1) * k.out_dim]);
                    output.extend_from_slice(&v_out[b * v.out_dim..(b + 1) * v.out_dim]);
                }
                Ok(output)
            },
        }
    }

    /// Batched causal attention with provided scheduler
    fn batched_causal_attention_with_scheduler(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        let num_heads = self.model.config.num_heads;
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let head_dim = self.model.config.head_dim();
        let q_dim = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = vec![0.0f32; seq_len * q_dim];

        for head in 0..num_heads {
            let head_offset = head * head_dim;

            // Extract Q_h, K_h, V_h
            let mut q_h = Vec::with_capacity(seq_len * head_dim);
            let mut k_h = Vec::with_capacity(seq_len * head_dim);
            let mut v_h = Vec::with_capacity(seq_len * head_dim);

            for pos in 0..seq_len {
                let start = pos * q_dim + head_offset;
                q_h.extend_from_slice(&q[start..start + head_dim]);
                k_h.extend_from_slice(&k[start..start + head_dim]);
                v_h.extend_from_slice(&v[start..start + head_dim]);
            }

            // Q @ K^T
            let mut k_t = vec![0.0f32; head_dim * seq_len];
            for i in 0..seq_len {
                for j in 0..head_dim {
                    k_t[j * seq_len + i] = k_h[i * head_dim + j];
                }
            }

            let scores = scheduler
                .matmul(&q_h, &k_t, seq_len, head_dim, seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batched_qk_scores_cached".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Apply scale
            let scaled: Vec<f32> = scores.iter().map(|&s| s * scale).collect();

            // Causal mask + softmax
            let attn_weights = self.model.apply_causal_mask_softmax(&scaled, seq_len);

            // Attn @ V
            let head_output = scheduler
                .matmul(&attn_weights, &v_h, seq_len, seq_len, head_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batched_attn_v_cached".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Copy to output
            for pos in 0..seq_len {
                let out_start = pos * q_dim + head_offset;
                let head_start = pos * head_dim;
                output[out_start..out_start + head_dim]
                    .copy_from_slice(&head_output[head_start..head_start + head_dim]);
            }
        }

        Ok(output)
    }
}
