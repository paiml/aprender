/// Accessor function that extracts optional quantized bytes from a Q4K layer.
type Q4KFieldAccessor<'a> = fn(&'a Q4KLayerWeights) -> Option<&'a [u8]>;

impl AprTransformer {
    /// Project a single weight matrix using Q4K, then Q6K, then F32 fallback.
    ///
    /// Tries quantized kernels in order of preference (Q4K > Q6K > F32).
    /// When `force_f32` is true, skips quantized paths entirely.
    fn project_with_q4k_or_f32(
        &self,
        q4k_bytes: Option<&[u8]>,
        q6k_bytes: Option<&[u8]>,
        f32_weight: &[f32],
        input: &[f32],
        out_dim: usize,
        in_dim: usize,
        force_f32: bool,
    ) -> Result<Vec<f32>> {
        if !force_f32 {
            if let Some(q4k) = q4k_bytes {
                return matmul_q4k_rowmajor(q4k, input, out_dim, in_dim);
            }
            if let Some(q6k) = q6k_bytes {
                return matmul_q6k_rowmajor(q6k, input, out_dim, in_dim);
            }
        }
        Ok(self.matmul(input, f32_weight, in_dim, out_dim))
    }

    /// Select quantized weight bytes (Q4K then Q6K) when not forcing F32.
    fn select_q4k_q6k<'a>(
        q4k_layer: Option<&'a Q4KLayerWeights>,
        force_f32: bool,
        q4k_field: Q4KFieldAccessor<'a>,
        q6k_field: Option<Q4KFieldAccessor<'a>>,
    ) -> (Option<&'a [u8]>, Option<&'a [u8]>) {
        if force_f32 {
            return (None, None);
        }
        let q4k = q4k_layer.and_then(q4k_field);
        let q6k = if q4k.is_none() {
            q6k_field.and_then(|f| q4k_layer.and_then(f))
        } else {
            None
        };
        (q4k, q6k)
    }

    /// Project QKV with separate Q4K weights, producing separate Q, K, V vectors.
    #[allow(clippy::too_many_arguments)]
    fn project_qkv_fused(
        &self,
        normed: &[f32],
        layer: &AprTransformerLayer,
        q4k: &Q4KLayerWeights,
        hidden_dim: usize,
        kv_size: usize,
        force_f32: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let q_f32 = &layer.qkv_weight[0..hidden_dim * hidden_dim];
        let (q4k_q, _) = Self::select_q4k_q6k(Some(q4k), force_f32,
            |q| q.attn_q_weight.as_deref(), None);
        let q = self.project_with_q4k_or_f32(q4k_q, None, q_f32, normed, hidden_dim, hidden_dim, force_f32)?;

        let k_start = hidden_dim * hidden_dim;
        let k_f32 = &layer.qkv_weight[k_start..k_start + kv_size * hidden_dim];
        let (q4k_k, _) = Self::select_q4k_q6k(Some(q4k), force_f32,
            |q| q.attn_k_weight.as_deref(), None);
        let k = self.project_with_q4k_or_f32(q4k_k, None, k_f32, normed, kv_size, hidden_dim, force_f32)?;

        let v_start = hidden_dim * hidden_dim + kv_size * hidden_dim;
        let v_f32 = &layer.qkv_weight[v_start..v_start + kv_size * hidden_dim];
        let (q4k_v, q6k_v) = Self::select_q4k_q6k(Some(q4k), force_f32,
            |q| q.attn_v_weight.as_deref(), Some(|q: &Q4KLayerWeights| q.attn_v_weight_q6k.as_deref()));
        let v = self.project_with_q4k_or_f32(q4k_v, q6k_v, v_f32, normed, kv_size, hidden_dim, force_f32)?;

        Ok((q, k, v))
    }

    /// Project QKV with cache-aware logic, handling fused Q4K paths and bias application.
    #[allow(clippy::too_many_arguments)]
    fn project_qkv_with_cache(
        &self,
        normed: &[f32],
        layer: &AprTransformerLayer,
        q4k_layer: Option<&Q4KLayerWeights>,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position: usize,
        force_f32: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let kv_size = num_kv_heads * head_dim;

        let (mut q, mut k, mut v) = if let Some(q4k) = q4k_layer {
            self.project_qkv_fused(normed, layer, q4k, hidden_dim, kv_size, force_f32)?
        } else {
            // Legacy path: combined QKV with F32
            let qkv_out_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = self.matmul(normed, &layer.qkv_weight, hidden_dim, qkv_out_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }
            let q = qkv[0..hidden_dim].to_vec();
            let k = qkv[hidden_dim..hidden_dim + kv_size].to_vec();
            let v = qkv[hidden_dim + kv_size..hidden_dim + 2 * kv_size].to_vec();
            (q, k, v)
        };

        // Apply biases for the fused path
        if q4k_layer.is_some() {
            Self::apply_split_qkv_bias(layer, &mut q, &mut k, &mut v, hidden_dim, kv_size);
        }

        // Apply RoPE to Q and K
        self.apply_rope_f32(&mut q, position, num_heads, head_dim);
        self.apply_rope_f32(&mut k, position, num_kv_heads, head_dim);

        Ok((q, k, v))
    }

    /// Apply split QKV bias from a combined bias vector to separate Q, K, V vectors.
    fn apply_split_qkv_bias(
        layer: &AprTransformerLayer,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        hidden_dim: usize,
        kv_size: usize,
    ) {
        if let Some(ref bias) = layer.qkv_bias {
            for (i, b) in bias[0..hidden_dim].iter().enumerate() {
                q[i] += b;
            }
            for (i, b) in bias[hidden_dim..hidden_dim + kv_size].iter().enumerate() {
                k[i] += b;
            }
            let v_bias_start = hidden_dim + kv_size;
            for (i, b) in bias[v_bias_start..v_bias_start + kv_size].iter().enumerate() {
                v[i] += b;
            }
        }
    }

    /// SwiGLU FFN: `down(SiLU(gate(x)) * up(x))` with parallel gate/up computation.
    #[allow(clippy::too_many_arguments)]
    fn forward_ffn_swiglu(
        &self,
        ffn_input: &[f32],
        gate_weight: &[f32],
        layer: &AprTransformerLayer,
        q4k_layer: Option<&Q4KLayerWeights>,
        hidden_dim: usize,
        intermediate_dim: usize,
        force_f32: bool,
    ) -> Result<Vec<f32>> {
        let (q4k_gate, _) = Self::select_q4k_q6k(q4k_layer, force_f32,
            |q| q.ffn_gate_weight.as_deref(), None);
        let (q4k_up, q6k_up) = Self::select_q4k_q6k(q4k_layer, force_f32,
            |q| q.ffn_up_weight.as_deref(), Some(|q: &Q4KLayerWeights| q.ffn_up_weight_q6k.as_deref()));

        let (gate_result, up_result) = rayon::join(
            || self.project_with_q4k_or_f32(q4k_gate, None, gate_weight, ffn_input, intermediate_dim, hidden_dim, force_f32),
            || self.project_with_q4k_or_f32(q4k_up, q6k_up, &layer.ffn_up_weight, ffn_input, intermediate_dim, hidden_dim, force_f32),
        );
        let gate = gate_result?;
        let up = up_result?;

        // SiLU(gate) * up
        let ffn_hidden: Vec<f32> = gate.iter().zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        // Down projection
        let (q4k_down, q6k_down) = Self::select_q4k_q6k(q4k_layer, force_f32,
            |q| q.ffn_down_weight.as_deref(), Some(|q: &Q4KLayerWeights| q.ffn_down_weight_q6k.as_deref()));
        let mut out = self.project_with_q4k_or_f32(q4k_down, q6k_down, &layer.ffn_down_weight, &ffn_hidden, hidden_dim, intermediate_dim, force_f32)?;
        if let Some(ref bias) = layer.ffn_down_bias {
            self.add_bias(&mut out, bias);
        }
        Ok(out)
    }

    /// Standard MLP FFN: `down(GELU(up(x)))`.
    #[allow(clippy::too_many_arguments)]
    fn forward_ffn_standard(
        &self,
        ffn_input: &[f32],
        layer: &AprTransformerLayer,
        q4k_layer: Option<&Q4KLayerWeights>,
        hidden_dim: usize,
        intermediate_dim: usize,
        force_f32: bool,
    ) -> Result<Vec<f32>> {
        let (q4k_up, _) = Self::select_q4k_q6k(q4k_layer, force_f32,
            |q| q.ffn_up_weight.as_deref(), None);
        let mut ffn_hidden = self.project_with_q4k_or_f32(q4k_up, None, &layer.ffn_up_weight, ffn_input, intermediate_dim, hidden_dim, force_f32)?;
        if let Some(ref bias) = layer.ffn_up_bias {
            self.add_bias(&mut ffn_hidden, bias);
        }
        self.gelu(&mut ffn_hidden);

        let (q4k_down, _) = Self::select_q4k_q6k(q4k_layer, force_f32,
            |q| q.ffn_down_weight.as_deref(), None);
        let mut out = self.project_with_q4k_or_f32(q4k_down, None, &layer.ffn_down_weight, &ffn_hidden, hidden_dim, intermediate_dim, force_f32)?;
        if let Some(ref bias) = layer.ffn_down_bias {
            self.add_bias(&mut out, bias);
        }
        Ok(out)
    }

    /// Forward through the FFN block, dispatching to SwiGLU or standard GELU.
    #[allow(clippy::too_many_arguments)]
    fn forward_ffn_block(
        &self,
        ffn_input: &[f32],
        layer: &AprTransformerLayer,
        q4k_layer: Option<&Q4KLayerWeights>,
        hidden_dim: usize,
        intermediate_dim: usize,
        force_f32: bool,
    ) -> Result<Vec<f32>> {
        if let Some(ref gate_weight) = layer.ffn_gate_weight {
            self.forward_ffn_swiglu(ffn_input, gate_weight, layer, q4k_layer, hidden_dim, intermediate_dim, force_f32)
        } else {
            self.forward_ffn_standard(ffn_input, layer, q4k_layer, hidden_dim, intermediate_dim, force_f32)
        }
    }

    /// Project through the LM head (final vocabulary projection).
    fn project_lm_head(
        &self,
        normed: &[f32],
        hidden_dim: usize,
        force_f32: bool,
    ) -> Result<Vec<f32>> {
        let (q4k, q6k) = if force_f32 {
            (None, None)
        } else {
            (self.lm_head_weight_q4k.as_deref(), self.lm_head_weight_q6k.as_deref())
        };
        let mut logits = self.project_with_q4k_or_f32(q4k, q6k, &self.lm_head_weight, normed, self.config.vocab_size, hidden_dim, force_f32)?;
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }
        Ok(logits)
    }
}
