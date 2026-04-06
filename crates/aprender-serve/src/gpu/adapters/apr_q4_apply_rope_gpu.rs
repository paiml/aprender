
impl GpuModelQ4 {

    /// Apply RoPE to Q and K within fused QKV tensor
    ///
    /// # Arguments
    ///
    /// * `qkv` - Fused QKV tensor [Q | K | V] where Q is [seq_len * hidden_dim]
    /// * `seq_len` - Number of tokens
    /// * `hidden_dim` - Q dimension
    /// * `num_heads` - Number of Q heads
    /// * `num_kv_heads` - Number of KV heads (for GQA)
    pub(crate) fn apply_rope_to_qkv(
        &self,
        qkv: &mut [f32],
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = hidden_dim + 2 * kv_dim;
        let theta = self.config.rope_theta;

        for pos in 0..seq_len {
            let qkv_start = pos * qkv_dim;

            // Apply RoPE to Q
            let q_start = qkv_start;
            self.apply_rope_inplace(
                &mut qkv[q_start..q_start + hidden_dim],
                pos,
                num_heads,
                head_dim,
                theta,
            );

            // Apply RoPE to K
            let k_start = qkv_start + hidden_dim;
            self.apply_rope_inplace(
                &mut qkv[k_start..k_start + kv_dim],
                pos,
                num_kv_heads,
                head_dim,
                theta,
            );
        }
    }

    /// Apply RoPE to a single Q or K tensor at given position
    pub(crate) fn apply_rope_inplace(
        &self,
        x: &mut [f32],
        position: usize,
        num_heads: usize,
        head_dim: usize,
        theta: f32,
    ) {
        let half_dim = head_dim / 2;
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;

        for h in 0..num_heads {
            let head_start = h * head_dim;

            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let idx1 = head_start + i;
                let idx2 = head_start + half_dim + i;

                if idx2 < x.len() {
                    let x1 = x[idx1];
                    let x2 = x[idx2];

                    x[idx1] = x1 * cos_val - x2 * sin_val;
                    x[idx2] = x1 * sin_val + x2 * cos_val;
                }
            }
        }
    }

    /// Simple attention (CPU, single-token)
    pub(crate) fn attention_cpu(
        &self,
        qkv: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Vec<f32> {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Split QKV
        let _q = &qkv[..hidden_dim];
        let _k = &qkv[hidden_dim..hidden_dim + kv_dim];
        let v = &qkv[hidden_dim + kv_dim..];

        // For single token, attention is trivial (softmax of single score = 1.0)
        // Output is just V projected back
        if seq_len == 1 {
            // GQA: repeat KV heads to match Q heads
            let kv_repeat = num_heads / num_kv_heads;
            let mut out = vec![0.0; hidden_dim];

            for h in 0..num_heads {
                let kv_h = h / kv_repeat;
                for d in 0..head_dim {
                    out[h * head_dim + d] = v[kv_h * head_dim + d];
                }
            }

            out
        } else {
            // Multi-token attention (simplified)
            // In practice, we'd use KV cache here
            v[..hidden_dim.min(v.len())].to_vec()
        }
    }
}

include!("using.rs");
