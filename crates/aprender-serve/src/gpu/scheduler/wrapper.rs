
impl GpuModel {

    /// Forward pass through a single transformer block by index
    pub fn forward_block_idx(
        &mut self,
        input: &[f32],
        seq_len: usize,
        block_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let qkv_dim = self.config.qkv_dim();

        // Get references to block weights (avoid cloning)
        let block = &self.block_weights[block_idx];
        let attn_norm_weight = &block.attn_norm_weight;
        let attn_norm_bias = &block.attn_norm_bias;

        // Pre-norm (uses references, no clone)
        let normed = Self::layer_norm_static(
            input,
            attn_norm_weight,
            attn_norm_bias,
            hidden_dim,
            self.config.eps,
        );

        // IMP-1005: Clone weights to avoid borrow conflict with &mut self in do_matmul
        let qkv_weight = self.block_weights[block_idx].qkv_weight.clone();

        // QKV projection (IMP-1005: use do_matmul for CUDA)
        // [seq_len, hidden_dim] @ [hidden_dim, qkv_dim] -> [seq_len, qkv_dim]
        let mut qkv = self.do_matmul(&normed, &qkv_weight, seq_len, hidden_dim, qkv_dim)?;

        // PMAT-216 FIX: Apply RoPE to Q and K for EACH position
        // Without RoPE, attention has no position information and produces garbage
        // Five Whys root cause: forward_block_idx was missing RoPE that exists in forward_block_refcell
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let kv_dim = self.config.kv_dim();
        let rope_theta = self.config.rope_theta;

        for pos in 0..seq_len {
            let qkv_start = pos * qkv_dim;

            // Apply RoPE to Q portion: [hidden_dim elements starting at qkv_start]
            Self::apply_rope_inline(
                &mut qkv[qkv_start..qkv_start + hidden_dim],
                num_heads,
                head_dim,
                rope_theta,
                pos,
            );

            // Apply RoPE to K portion: [kv_dim elements starting at qkv_start + hidden_dim]
            Self::apply_rope_inline(
                &mut qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim],
                num_kv_heads,
                head_dim,
                rope_theta,
                pos,
            );
        }

        // Optimized GQA attention with GPU matmul for scores
        let attn_out = self.optimized_gqa_attention(&qkv, seq_len)?;

        // IMP-1005: Clone weights to avoid borrow conflict
        let out_weight = self.block_weights[block_idx].out_weight.clone();
        let out_bias = self.block_weights[block_idx].out_bias.clone();

        // Output projection (IMP-1005: use do_matmul for CUDA)
        let projected = self.do_matmul(&attn_out, &out_weight, seq_len, hidden_dim, hidden_dim)?;

        // Residual 1 (vectorized)
        let mut residual1: Vec<f32> = input
            .iter()
            .zip(projected.iter())
            .enumerate()
            .map(|(i, (&inp, &proj))| inp + proj + out_bias[i % hidden_dim])
            .collect();

        // IMP-1005: Clone weights to avoid borrow conflict
        let ffn_norm_weight = self.block_weights[block_idx].ffn_norm_weight.clone();
        let ffn_norm_bias = self.block_weights[block_idx].ffn_norm_bias.clone();

        // FFN pre-norm
        let ffn_normed = Self::layer_norm_static(
            &residual1,
            &ffn_norm_weight,
            &ffn_norm_bias,
            hidden_dim,
            self.config.eps,
        );

        // FFN: MoE dispatch when experts present, SwiGLU/GELU otherwise
        if let Some(ref moe) = self.block_weights[block_idx].moe_experts {
            // ALB-010: MoE forward — route per token, accumulate into residual
            for t in 0..seq_len {
                let token_hidden = &ffn_normed[t * hidden_dim..(t + 1) * hidden_dim];
                let moe_out = super::moe_dispatch::moe_forward_token(token_hidden, moe, hidden_dim);
                for j in 0..hidden_dim {
                    residual1[t * hidden_dim + j] += moe_out[j];
                }
            }
        } else {
            // IMP-1005: Clone weights to avoid borrow conflict
            let ffn_fc1_weight = self.block_weights[block_idx].ffn_fc1_weight.clone();
            let ffn_fc1_bias = self.block_weights[block_idx].ffn_fc1_bias.clone();
            let ffn_gate_weight = self.block_weights[block_idx].ffn_gate_weight.clone();

            let activated: Vec<f32> = if let Some(gate_weight) = ffn_gate_weight {
                let up_out = self.do_matmul(
                    &ffn_normed,
                    &ffn_fc1_weight,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                )?;
                let gate_out = self.do_matmul(
                    &ffn_normed,
                    &gate_weight,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                )?;
                up_out
                    .iter()
                    .zip(gate_out.iter())
                    .map(|(&u, &g)| {
                        let silu_g = g / (1.0 + (-g).exp());
                        silu_g * u
                    })
                    .collect()
            } else {
                let fc1_out = self.do_matmul(
                    &ffn_normed,
                    &ffn_fc1_weight,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                )?;
                fc1_out
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let x = x + ffn_fc1_bias[i % intermediate_dim];
                        0.5 * x
                            * (1.0
                                + ((2.0f32 / std::f32::consts::PI).sqrt()
                                    * (x + 0.044_715 * x.powi(3)))
                                .tanh())
                    })
                    .collect()
            };

            let ffn_fc2_weight = self.block_weights[block_idx].ffn_fc2_weight.clone();
            let ffn_fc2_bias = self.block_weights[block_idx].ffn_fc2_bias.clone();
            let fc2_out = self.do_matmul(
                &activated,
                &ffn_fc2_weight,
                seq_len,
                intermediate_dim,
                hidden_dim,
            )?;
            for (i, x) in residual1.iter_mut().enumerate() {
                *x += fc2_out[i] + ffn_fc2_bias[i % hidden_dim];
            }
        }

        Ok(residual1)
    }

    /// RMSNorm (delegates to ops module)
    pub(crate) fn layer_norm_static(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        hidden_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        super::ops::layer_norm_static(input, weight, bias, hidden_dim, eps)
    }

    /// Layer normalization (instance method)
    fn layer_norm(&self, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        Self::layer_norm_static(input, weight, bias, self.config.hidden_dim, self.config.eps)
    }

    /// Generate tokens using GPU-accelerated forward pass with incremental decoding (wrapper)
    pub fn generate_gpu(&mut self, prompt: &[usize], max_tokens: usize) -> Result<Vec<usize>> {
        super::batch::generate_gpu(self, prompt, max_tokens)
    }

    /// Argmax helper for sampling (wrapper)
    fn argmax(logits: &[f32]) -> usize {
        super::batch::argmax(logits)
    }

    /// Optimized GQA attention using GPU for matmul operations (wrapper)
    fn optimized_gqa_attention(&mut self, qkv: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        super::batch::optimized_gqa_attention(self, qkv, seq_len)
    }
}

include!("matmul.rs");
include!("forward_block.rs");
include!("from_gguf_config.rs");
include!("model_forward_block_gpu.rs");
include!("gpu_forward_pass.rs");
