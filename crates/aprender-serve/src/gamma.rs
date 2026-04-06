
#[cfg(feature = "cuda")]
impl SafeTensorsCudaModel {

    /// Forward pass for a single transformer layer.
    ///
    /// PMAT-120 FIX: RoPE is now applied explicitly here (not in incremental_attention_gpu).
    fn forward_layer(&mut self, layer_idx: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        // GH-201: Load layer weights on-demand if in streaming mode
        self.ensure_layer_weights_loaded(layer_idx)?;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        // 1. Pre-attention norm (CPU for now)
        let normed = self.apply_rms_norm_layer_cpu(hidden, layer_idx, "attn")?;

        // 2. QKV projection: [1, hidden_dim] × [hidden_dim, qkv_out_dim]^T
        let mut qkv = vec![0.0f32; qkv_out_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.attn_qkv"),
                &normed,
                &mut qkv,
                1,
                qkv_out_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "qkv_proj".to_string(),
                reason: format!("Layer {layer_idx} QKV GEMM failed: {e}"),
            })?;

        // PMAT-120 FIX: Add QKV bias (Qwen2 has attention biases!)
        if let Some(bias) = self.qkv_bias_cache.get(&format!("qkv_bias.{layer_idx}")) {
            for (q, b) in qkv.iter_mut().zip(bias.iter()) {
                *q += b;
            }
        }

        // 3. Split Q, K, V
        let mut q = qkv[..hidden_dim].to_vec();
        let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
        let v = qkv[hidden_dim + kv_dim..].to_vec();

        // 3b. GH-279: Per-head QK RMSNorm (Qwen3) — after bias, before RoPE
        if let Some(q_norm) = self.qk_norm_cache.get(&format!("q_norm.{layer_idx}")) {
            crate::gguf::ops::apply_per_head_rms_norm(&mut q, q_norm, num_heads, self.epsilon);
        }
        if let Some(k_norm) = self.qk_norm_cache.get(&format!("k_norm.{layer_idx}")) {
            crate::gguf::ops::apply_per_head_rms_norm(&mut k, k_norm, num_kv_heads, self.epsilon);
        }

        // 4. PMAT-120 FIX: Apply RoPE to Q and K before attention
        // Position is kv_position (number of tokens already processed)
        let position = self.kv_position as usize;
        let rope_theta = self.config.rope_theta;
        let half_dim = head_dim / 2;

        // Apply RoPE to Q (num_heads heads)
        for h in 0..num_heads {
            let head_start = h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx1 = head_start + i;
                let idx2 = head_start + i + half_dim;
                let x1 = q[idx1];
                let x2 = q[idx2];
                q[idx1] = x1 * cos_val - x2 * sin_val;
                q[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }

        // Apply RoPE to K (num_kv_heads heads for GQA)
        for h in 0..num_kv_heads {
            let head_start = h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx1 = head_start + i;
                let idx2 = head_start + i + half_dim;
                let x1 = k[idx1];
                let x2 = k[idx2];
                k[idx1] = x1 * cos_val - x2 * sin_val;
                k[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }

        // 5. Attention with KV cache (GPU)
        let mut attn_output = vec![0.0f32; hidden_dim];
        self.executor
            .incremental_attention_gpu(layer_idx, &q, &k, &v, &mut attn_output)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "attention".to_string(),
                reason: format!("Layer {layer_idx} attention failed: {e}"),
            })?;

        // 5. Output projection
        let mut attn_proj = vec![0.0f32; hidden_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.attn_output"),
                &attn_output,
                &mut attn_proj,
                1,
                hidden_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "attn_output".to_string(),
                reason: format!("Layer {layer_idx} attn output GEMM failed: {e}"),
            })?;

        // PMAT-120 FIX: Add o_proj bias (if present)
        if let Some(bias) = self.o_bias_cache.get(&format!("o_bias.{layer_idx}")) {
            for (p, b) in attn_proj.iter_mut().zip(bias.iter()) {
                *p += b;
            }
        }

        // 6. Residual connection
        let mut residual: Vec<f32> = hidden.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();

        // 7. Post-attention norm (CPU)
        let normed2 = self.apply_rms_norm_layer_cpu(&residual, layer_idx, "ffn")?;

        // 8. FFN gate projection
        let mut gate = vec![0.0f32; intermediate_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.ffn_gate"),
                &normed2,
                &mut gate,
                1,
                intermediate_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "ffn_gate".to_string(),
                reason: format!("Layer {layer_idx} FFN gate GEMM failed: {e}"),
            })?;

        // 9. FFN up projection
        let mut up = vec![0.0f32; intermediate_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.ffn_up"),
                &normed2,
                &mut up,
                1,
                intermediate_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "ffn_up".to_string(),
                reason: format!("Layer {layer_idx} FFN up GEMM failed: {e}"),
            })?;

        // 10. SwiGLU: silu(gate) * up
        let swiglu: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();

        // 11. FFN down projection
        let mut ffn_out = vec![0.0f32; hidden_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.ffn_down"),
                &swiglu,
                &mut ffn_out,
                1,
                hidden_dim as u32,
                intermediate_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "ffn_down".to_string(),
                reason: format!("Layer {layer_idx} FFN down GEMM failed: {e}"),
            })?;

        // 12. Residual connection
        for (r, f) in residual.iter_mut().zip(&ffn_out) {
            *r += f;
        }

        Ok(residual)
    }

    /// Apply RMS normalization with output gamma weights.
    ///
    /// RMS norm formula: (x / sqrt(mean(x^2) + eps)) * gamma
    fn apply_rms_norm_cpu(&self, x: &[f32]) -> Result<Vec<f32>> {
        // RMS norm: x / sqrt(mean(x^2) + eps) * gamma
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + self.epsilon).sqrt();

        // Get output gamma from cache
        let gamma =
            self.gamma_cache
                .get("output")
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "rms_norm".to_string(),
                    reason: "Output gamma not found in cache".to_string(),
                })?;

        // Apply normalization with gamma scaling
        Ok(x.iter()
            .zip(gamma.iter())
            .map(|(xi, gi)| (xi / rms) * gi)
            .collect())
    }

    /// Apply RMS normalization for a specific layer with gamma weights.
    ///
    /// RMS norm formula: (x / sqrt(mean(x^2) + eps)) * gamma
    fn apply_rms_norm_layer_cpu(
        &self,
        x: &[f32],
        layer_idx: usize,
        norm_type: &str,
    ) -> Result<Vec<f32>> {
        // RMS norm: x / sqrt(mean(x^2) + eps) * gamma
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + self.epsilon).sqrt();

        // Get layer gamma from cache
        let cache_key = format!("{norm_type}.{layer_idx}");
        let gamma = self.gamma_cache.get(&cache_key).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "rms_norm".to_string(),
                reason: format!("Layer {layer_idx} {norm_type} gamma not found in cache"),
            }
        })?;

        // Apply normalization with gamma scaling
        Ok(x.iter()
            .zip(gamma.iter())
            .map(|(xi, gi)| (xi / rms) * gi)
            .collect())
    }
}

include!("safetensors_cuda_load_safe_tensors.rs");
include!("safetensors_cuda_upload_weights_safe.rs");
