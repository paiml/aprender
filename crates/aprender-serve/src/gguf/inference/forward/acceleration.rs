impl OwnedQuantizedModel {

    /// Batch matmul with GPU acceleration via HybridScheduler (IMP-107)
    ///
    /// Dequantizes weights and uses GPU for large operations.
    #[cfg(feature = "gpu")]
    fn batch_matmul_gpu(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        m: usize,
        k: usize,
        n: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        // Dequantize weight to f32
        let weight_f32 = self.dequantize_weight(weight)?;

        // Use HybridScheduler for GPU/CPU dispatch
        // A: [m, k], B: [k, n] -> C: [m, n]
        scheduler.matmul(input, &weight_f32, m, k, n).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::matmul".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            }
        })
    }

    /// Batch QKV matmul with GPU acceleration via HybridScheduler
    ///
    /// Five Whys Root Cause Fix: Handles both fused and separate Q/K/V formats
    #[cfg(feature = "gpu")]
    fn batch_qkv_matmul_gpu_with_scheduler(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        batch_size: usize,
        hidden_dim: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.batch_matmul_gpu(
                input,
                weight,
                batch_size,
                hidden_dim,
                weight.out_dim,
                scheduler,
            ),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Compute Q, K, V separately then concatenate
                let q_out =
                    self.batch_matmul_gpu(input, q, batch_size, hidden_dim, q.out_dim, scheduler)?;
                let k_out =
                    self.batch_matmul_gpu(input, k, batch_size, hidden_dim, k.out_dim, scheduler)?;
                let v_out =
                    self.batch_matmul_gpu(input, v, batch_size, hidden_dim, v.out_dim, scheduler)?;

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

    /// Dequantize a weight tensor to f32
    #[cfg(feature = "gpu")]
    pub(crate) fn dequantize_weight(&self, weight: &OwnedQuantizedTensor) -> Result<Vec<f32>> {
        use crate::quantize::{dequantize_q4_k_simd, dequantize_q5_k, dequantize_q6_k, QK_K};

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let total_elements = in_dim * out_dim;

        match weight.qtype {
            GGUF_TYPE_Q4_K => {
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let mut output = Vec::with_capacity(total_elements);
                for row in 0..out_dim {
                    let row_start = row * super_blocks_per_row * 144;
                    let row_end = row_start + super_blocks_per_row * 144;
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q4_k_simd(row_data)?;
                    // Take only in_dim values (may have padding due to super-block alignment)
                    output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
                }
                Ok(output)
            },
            GGUF_TYPE_Q5_K => {
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let mut output = Vec::with_capacity(total_elements);
                for row in 0..out_dim {
                    let row_start = row * super_blocks_per_row * 176;
                    let row_end = row_start + super_blocks_per_row * 176;
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q5_k(row_data)?;
                    output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
                }
                Ok(output)
            },
            GGUF_TYPE_Q6_K => {
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let mut output = Vec::with_capacity(total_elements);
                for row in 0..out_dim {
                    let row_start = row * super_blocks_per_row * 210;
                    let row_end = row_start + super_blocks_per_row * 210;
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q6_k(row_data)?;
                    output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
                }
                Ok(output)
            },
            _ => {
                // F32 or unsupported - interpret raw bytes as f32
                let num_floats = weight.data.len() / 4;
                let mut output = vec![0.0f32; num_floats];
                for (i, chunk) in weight.data.chunks_exact(4).enumerate() {
                    output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(output)
            },
        }
    }

    /// Dequantize QKV weights - handles both fused and separate formats
    ///
    /// Five Whys Root Cause Fix: This method handles both tensor layouts for dequantization
    #[cfg(feature = "gpu")]
    pub fn dequantize_qkv(&self, qkv: &OwnedQKVWeights) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.dequantize_weight(weight),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Dequantize each separately and concatenate
                let q_out = self.dequantize_weight(q)?;
                let k_out = self.dequantize_weight(k)?;
                let v_out = self.dequantize_weight(v)?;

                let mut output = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// Fused batch matmul with GPU acceleration (IMP-109)
    ///
    /// Performs batched matrix multiplication with fused dequantization.
    /// Uses the same weight layout interpretation as `batch_matmul_gpu` for
    /// consistency within the codebase.
    ///
    /// Key optimization: Dequantizes weight matrix once for all batch elements,
    /// reducing memory bandwidth for repeated operations in transformer layers.
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch_size, in_dim]
    /// * `weight` - Quantized weight tensor [out_dim, in_dim]
    /// * `batch_size` - Number of input vectors
    ///
    /// # Returns
    /// Output tensor [batch_size, out_dim]
    ///
    /// # Errors
    /// Returns error if GPU operations fail or dimensions mismatch
    #[cfg(feature = "gpu")]
    pub fn fused_batch_matmul_gpu(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        batch_size: usize,
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;

        // Validate input dimensions
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}={}",
                    input.len(),
                    batch_size,
                    in_dim,
                    batch_size * in_dim
                ),
            });
        }

        // Dequantize weight once (key optimization: reuse across batch elements)
        let weight_f32 = self.dequantize_weight(weight)?;

        // Use HybridScheduler for CPU/GPU dispatch based on workload size
        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        // Use same matmul approach as batch_matmul_gpu for consistency
        scheduler
            .matmul(input, &weight_f32, batch_size, in_dim, out_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::matmul".to_string(),
                reason: format!("GPU batched matmul failed: {e}"),
            })
    }

    /// Batched causal attention with GPU acceleration (IMP-108)
    ///
    /// Computes causal self-attention using matrix multiplications that can be
    /// GPU-accelerated for large sequence lengths. Uses HybridScheduler for
    /// automatic CPU/GPU dispatch.
    ///
    /// Algorithm:
    /// 1. For each head: scores = Q @ K^T / sqrt(head_dim)
    /// 2. Apply causal mask: scores[i,j] = -inf for j > i
    /// 3. Softmax per row
    /// 4. Output = softmax(scores) @ V
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Attention output [seq_len, hidden_dim]
    ///
    /// # Errors
    /// Returns error if GPU operations fail
    #[cfg(feature = "gpu")]
    pub fn batched_causal_attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        let mut output = vec![0.0f32; seq_len * hidden_dim];

        // Process each head
        for head in 0..num_heads {
            let head_offset = head * head_dim;

            // Extract Q_h, K_h, V_h for this head: [seq_len, head_dim]
            let mut q_h = Vec::with_capacity(seq_len * head_dim);
            let mut k_h = Vec::with_capacity(seq_len * head_dim);
            let mut v_h = Vec::with_capacity(seq_len * head_dim);

            for pos in 0..seq_len {
                let start = pos * hidden_dim + head_offset;
                q_h.extend_from_slice(&q[start..start + head_dim]);
                k_h.extend_from_slice(&k[start..start + head_dim]);
                v_h.extend_from_slice(&v[start..start + head_dim]);
            }

            // Compute attention scores: Q_h @ K_h^T -> [seq_len, seq_len]
            // Use GPU for large sequences (seq_len^2 * head_dim ops)
            let scores =
                self.batched_qk_scores(&q_h, &k_h, seq_len, head_dim, scale, &mut scheduler)?;

            // Apply causal mask and softmax
            let attn_weights = self.apply_causal_mask_softmax(&scores, seq_len);

            // Compute output: attn_weights @ V_h -> [seq_len, head_dim]
            let head_output =
                self.batched_attn_v(&attn_weights, &v_h, seq_len, head_dim, &mut scheduler)?;

            // Copy head output to final output
            for pos in 0..seq_len {
                let out_start = pos * hidden_dim + head_offset;
                let head_start = pos * head_dim;
                output[out_start..out_start + head_dim]
                    .copy_from_slice(&head_output[head_start..head_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Compute Q @ K^T attention scores with GPU acceleration
    #[cfg(feature = "gpu")]
    fn batched_qk_scores(
        &self,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        // Q: [seq_len, head_dim], K: [seq_len, head_dim]
        // scores = Q @ K^T -> [seq_len, seq_len]

        // Transpose K: [head_dim, seq_len]
        let mut k_t = vec![0.0f32; head_dim * seq_len];
        for i in 0..seq_len {
            for j in 0..head_dim {
                k_t[j * seq_len + i] = k[i * head_dim + j];
            }
        }

        // Matmul: Q[seq_len, head_dim] @ K_T[head_dim, seq_len] -> [seq_len, seq_len]
        let scores = scheduler
            .matmul(q, &k_t, seq_len, head_dim, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "batched_qk_scores".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            })?;

        // Apply scale
        let scaled: Vec<f32> = scores.iter().map(|&s| s * scale).collect();
        Ok(scaled)
    }

    /// Apply causal mask and softmax to attention scores
    #[cfg(feature = "gpu")]
    pub(crate) fn apply_causal_mask_softmax(&self, scores: &[f32], seq_len: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; seq_len * seq_len];

        for i in 0..seq_len {
            // Apply causal mask: set j > i to -inf
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..=i {
                let idx = i * seq_len + j;
                max_score = max_score.max(scores[idx]);
            }

            // Compute softmax for causal positions only
            let mut exp_sum = 0.0f32;
            for j in 0..=i {
                let idx = i * seq_len + j;
                let exp_val = (scores[idx] - max_score).exp();
                weights[idx] = exp_val;
                exp_sum += exp_val;
            }

            // Normalize
            for j in 0..=i {
                let idx = i * seq_len + j;
                weights[idx] /= exp_sum;
            }
            // j > i remains 0 (masked out)
        }

        weights
    }

    /// Compute attention_weights @ V with GPU acceleration
    #[cfg(feature = "gpu")]
    fn batched_attn_v(
        &self,
        attn_weights: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        // attn_weights: [seq_len, seq_len], V: [seq_len, head_dim]
        // output = attn_weights @ V -> [seq_len, head_dim]
        scheduler
            .matmul(attn_weights, v, seq_len, seq_len, head_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "batched_attn_v".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            })
    }
}
