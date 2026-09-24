// #4228: the layer-major CPU prefill. Included by `forward_qwen35.rs`.
//
// The per-token prefill runs every layer for one token, then the next token, so
// each projection's weights are streamed from DRAM once per prompt token. Here
// every layer runs over ALL the new tokens before the next layer starts: each
// projection is one multi-row matmul (`quantize::multi_row`), and only the
// per-token state updates — the causal conv, the delta rule and attention —
// walk the tokens in order.
//
// Bitwise the per-token path. The multi-row matmul is bitwise the one-row
// kernel per token (see its module docs), every element-wise step below is the
// per-token code on the same token's values, and the recurrences see the same
// tokens in the same order. The layer order changes WHEN a token's layer-L
// values are computed, never WHAT they are computed from: layer L for token r
// reads layer L-1 for token r and layer L's state after tokens 0..r, in both
// orders.

impl Qwen35Model<'_> {
    /// `output[r] = weight · input[r]` for the `t` rows of `input`: one multi-row
    /// call when `quantize::multi_row` reproduces the one-row kernel bitwise for
    /// this qtype, else the one-row path per row.
    fn rows_matmul_into(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        t: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let (in_dim, out_dim) = (weight.in_dim, weight.out_dim);
        if crate::quantize::multi_row::fused_k_rows_matmul_into(
            weight.qtype,
            &weight.data,
            input,
            t,
            in_dim,
            out_dim,
            output,
        )? {
            return Ok(());
        }
        for (x, y) in input.chunks_exact(in_dim).zip(output.chunks_exact_mut(out_dim)) {
            self.base.fused_matmul_into(x, weight, y)?;
        }
        Ok(())
    }

    /// Can [`Self::forward_prefill_qwen35`] take `n` new tokens on `cache` from
    /// `start`? It needs the cache to hold exactly `start` positions and room for
    /// all `n` more: the per-token path's `append` drops a position past
    /// `max_seq_len`, and this path does not reproduce that.
    #[must_use]
    pub fn prefill_fits(&self, cache: &Qwen35State, start: usize, n: usize) -> bool {
        cache.kv_cache.len() == start && start + n <= cache.kv_cache.max_len()
    }

    /// Prefill `tokens` at positions `start..` layer by layer and return the last
    /// position's logits: the state and logits [`Self::forward_single_qwen35`]
    /// leaves after the same tokens one at a time.
    ///
    /// # Errors
    /// An empty `tokens`, a cache that does not satisfy [`Self::prefill_fits`], a
    /// token id past the vocabulary, or a matmul failure. On `Err` the state is
    /// partly advanced and must be discarded.
    pub fn forward_prefill_qwen35(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35State,
        start: usize,
    ) -> Result<Vec<f32>> {
        let t = tokens.len();
        let hd = self.base.config.hidden_dim;
        if t == 0 || !self.prefill_fits(cache, start, t) {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!(
                    "qwen35 prefill: {t} tokens from position {start} on a cache holding {} of {}",
                    cache.kv_cache.len(),
                    cache.kv_cache.max_len()
                ),
            });
        }
        let embed = self.base.token_embedding();
        let mut hidden = Vec::with_capacity(t * hd);
        for &tok in tokens {
            let row = embed.get(tok as usize * hd..(tok as usize + 1) * hd).ok_or_else(|| {
                crate::error::RealizarError::InvalidShape {
                    reason: format!("qwen35 prefill: token {tok} is past the embedding table"),
                }
            })?;
            hidden.extend_from_slice(row);
        }
        for (il, layer) in self.layers.iter().enumerate() {
            match layer {
                Qwen35OwnedLayer::DeltaNet(d) => self.prefill_deltanet(d, &mut hidden, cache, il, t)?,
                Qwen35OwnedLayer::Attention(a) => {
                    self.prefill_attention(a, &mut hidden, cache, il, start, t)?;
                },
            }
        }
        let mut out_normed = vec![0.0; hd];
        crate::gguf::ops::rms_norm_into(
            &hidden[(t - 1) * hd..],
            self.base.output_norm_weight(),
            self.base.config.eps,
            &mut out_normed,
        );
        let mut logits = vec![0.0; self.base.config.vocab_size];
        self.base
            .fused_matmul_into(&out_normed, self.base.lm_head_weight(), &mut logits)?;
        cache.kv_cache.advance_by(t);
        Ok(logits)
    }

    /// `rms_norm(hidden[r], weight)` for every token row.
    fn rows_rms_norm(&self, hidden: &[f32], weight: &[f32]) -> Vec<f32> {
        let hd = self.base.config.hidden_dim;
        let mut normed = vec![0.0; hidden.len()];
        for (x, y) in hidden.chunks_exact(hd).zip(normed.chunks_exact_mut(hd)) {
            crate::gguf::ops::rms_norm_into(x, weight, self.base.config.eps, y);
        }
        normed
    }

    /// The SwiGLU FFN and its residual over every token row, as the tail of
    /// `forward_deltanet` / `forward_attention`.
    fn prefill_ffn(
        &self,
        hidden: &mut [f32],
        post_attention_norm: &[f32],
        ffn_gate: &OwnedQuantizedTensor,
        ffn_up: &OwnedQuantizedTensor,
        ffn_down: &OwnedQuantizedTensor,
        t: usize,
    ) -> Result<()> {
        let normed = self.rows_rms_norm(hidden, post_attention_norm);
        let mut gate = vec![0.0; t * ffn_gate.out_dim];
        self.rows_matmul_into(&normed, ffn_gate, t, &mut gate)?;
        let mut up = vec![0.0; t * ffn_up.out_dim];
        self.rows_matmul_into(&normed, ffn_up, t, &mut up)?;
        for (u, &x) in up.iter_mut().zip(&gate) {
            let silu = x / (1.0 + (-x as f32).exp());
            *u *= silu;
        }
        let mut down = vec![0.0; t * ffn_down.out_dim];
        self.rows_matmul_into(&up, ffn_down, t, &mut down)?;
        for (h, d) in hidden.iter_mut().zip(&down) {
            *h += *d;
        }
        Ok(())
    }

    /// One Gated `DeltaNet` layer over every token row; `forward_deltanet` per token.
    fn prefill_deltanet(
        &self,
        d: &Qwen35OwnedDeltaNetLayer,
        hidden: &mut [f32],
        cache: &mut Qwen35State,
        il: usize,
        t: usize,
    ) -> Result<()> {
        let fn_softplus = |x: f32| -> f32 {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        };
        let hd = self.base.config.hidden_dim;
        let conv_dim = self.head_k_dim * self.num_k_heads * 2 + self.head_v_dim * self.num_v_heads;
        let k_dim = self.head_k_dim * self.num_k_heads;
        let v_dim = self.head_v_dim * self.num_v_heads;
        let nv = self.num_v_heads;

        let normed = self.rows_rms_norm(hidden, &d.attn_norm);
        let mut conv_in = vec![0.0; t * conv_dim];
        self.rows_matmul_into(&normed, &d.attn_qkv, t, &mut conv_in)?;
        let mut dt_raw = vec![0.0; t * nv];
        self.rows_matmul_into(&normed, &d.ssm_alpha, t, &mut dt_raw)?;
        let mut beta_raw = vec![0.0; t * nv];
        self.rows_matmul_into(&normed, &d.ssm_beta, t, &mut beta_raw)?;
        let mut gate = vec![0.0; t * v_dim];
        self.rows_matmul_into(&normed, &d.attn_gate, t, &mut gate)?;

        let mut ssm_out_in = vec![0.0; t * v_dim];
        let mut conv_out = vec![0.0; conv_dim];
        let mut dt = vec![0.0; nv];
        let mut out_h = vec![0.0; v_dim];
        for r in 0..t {
            causal_conv1d(
                &conv_in[r * conv_dim..(r + 1) * conv_dim],
                &mut cache.conv_states[il][..],
                &d.ssm_conv1d_weight,
                self.conv_kernel,
                conv_dim,
                &mut conv_out,
            );
            for x in conv_out.iter_mut() {
                *x = *x / (1.0 + (-*x).exp());
            }
            let mut q = conv_out[0..k_dim].to_vec();
            let mut k = conv_out[k_dim..k_dim * 2].to_vec();
            let v = &conv_out[k_dim * 2..conv_dim];
            l2_norm_per_head(&mut q, self.head_k_dim, self.base.config.eps);
            l2_norm_per_head(&mut k, self.head_k_dim, self.base.config.eps);
            for (i, val) in dt_raw[r * nv..(r + 1) * nv].iter().enumerate() {
                dt[i] = fn_softplus(val + d.ssm_dt_bias[i]) * d.ssm_a[i];
            }
            let beta = &mut beta_raw[r * nv..(r + 1) * nv];
            for x in beta.iter_mut() {
                *x = 1.0 / (1.0 + (-*x).exp());
            }
            out_h.fill(0.0);
            delta_rule_recurrence_gqa(
                &q,
                &k,
                v,
                beta,
                &dt,
                &mut cache.ssm_states[il][..],
                &mut out_h,
                self.num_k_heads,
                self.head_k_dim,
                self.num_v_heads,
                self.head_v_dim,
            );
            gated_rmsnorm(
                &out_h,
                &gate[r * v_dim..(r + 1) * v_dim],
                &d.ssm_norm_weight,
                self.base.config.eps,
                self.head_v_dim,
                &mut ssm_out_in[r * v_dim..(r + 1) * v_dim],
            );
        }
        let mut ssm_out = vec![0.0; t * hd];
        self.rows_matmul_into(&ssm_out_in, &d.ssm_out, t, &mut ssm_out)?;
        for (h, s) in hidden.iter_mut().zip(&ssm_out) {
            *h += *s;
        }
        self.prefill_ffn(hidden, &d.post_attention_norm, &d.ffn_gate, &d.ffn_up, &d.ffn_down, t)
    }

    /// One full-attention layer over every token row; `forward_attention` per token.
    /// Token r sits at position `start + r` and appends its K/V before it attends,
    /// so it sees positions `0..=start + r`, as the per-token path.
    fn prefill_attention(
        &self,
        a: &Qwen35OwnedAttentionLayer,
        hidden: &mut [f32],
        cache: &mut Qwen35State,
        il: usize,
        start: usize,
        t: usize,
    ) -> Result<()> {
        let hd = self.base.config.hidden_dim;
        let normed = self.rows_rms_norm(hidden, &a.attn_norm);
        let (qf_dim, k_dim, v_dim) = (a.attn_q.out_dim, a.attn_k.out_dim, a.attn_v.out_dim);
        let mut q_full = vec![0.0; t * qf_dim];
        self.rows_matmul_into(&normed, &a.attn_q, t, &mut q_full)?;
        let mut k_all = vec![0.0; t * k_dim];
        self.rows_matmul_into(&normed, &a.attn_k, t, &mut k_all)?;
        let mut v_all = vec![0.0; t * v_dim];
        self.rows_matmul_into(&normed, &a.attn_v, t, &mut v_all)?;

        let num_heads = self.base.config.num_heads;
        let num_kv_heads = self.base.config.num_kv_heads;
        let norm_dim = a.attn_q_norm.len();
        let group_size = num_heads / num_kv_heads;
        let head_dim = self.head_dim;
        let n_rot = 2 * self.rope_sections.iter().sum::<usize>();
        let freq_base = self.base.config.rope_theta;
        let kv_stride = num_kv_heads * head_dim;

        let mut attn_out_in = vec![0.0; t * num_heads * norm_dim];
        for r in 0..t {
            let position = start + r;
            let qf = &q_full[r * qf_dim..(r + 1) * qf_dim];
            let mut q = vec![0.0; num_heads * norm_dim];
            let mut gate = vec![0.0; num_heads * norm_dim];
            for h in 0..num_heads {
                let (oqf, oq) = (h * norm_dim * 2, h * norm_dim);
                q[oq..oq + norm_dim].copy_from_slice(&qf[oqf..oqf + norm_dim]);
                gate[oq..oq + norm_dim].copy_from_slice(&qf[oqf + norm_dim..oqf + norm_dim * 2]);
            }
            let mut k = k_all[r * k_dim..(r + 1) * k_dim].to_vec();
            crate::gguf::ops::apply_per_head_rms_norm(
                &mut q,
                &a.attn_q_norm,
                num_heads,
                self.base.config.eps,
            );
            crate::gguf::ops::apply_per_head_rms_norm(
                &mut k,
                &a.attn_k_norm,
                num_kv_heads,
                self.base.config.eps,
            );
            apply_partial_neox_rope(&mut q, num_heads, norm_dim, n_rot, position, freq_base);
            apply_partial_neox_rope(&mut k, num_kv_heads, norm_dim, n_rot, position, freq_base);
            cache
                .kv_cache
                .append(il, &k, &v_all[r * v_dim..(r + 1) * v_dim]);
            let k_cache = cache.kv_cache.get_k(il);
            let v_cache = cache.kv_cache.get_v(il);

            let out = &mut attn_out_in[r * num_heads * norm_dim..(r + 1) * num_heads * norm_dim];
            let mut scores = vec![0.0; position + 1];
            for h in 0..num_heads {
                let kv_h = h / group_size;
                let q_h = &q[h * head_dim..(h + 1) * head_dim];
                for (p, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0;
                    let k_p = &k_cache[p * kv_stride + kv_h * head_dim..p * kv_stride + (kv_h + 1) * head_dim];
                    for i in 0..head_dim {
                        dot += q_h[i] * k_p[i];
                    }
                    *score = dot / (head_dim as f32).sqrt();
                }
                crate::gguf::ops::softmax(&mut scores);
                let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
                for (p, &w) in scores.iter().enumerate() {
                    let v_p = &v_cache[p * kv_stride + kv_h * head_dim..p * kv_stride + (kv_h + 1) * head_dim];
                    for i in 0..head_dim {
                        out_h[i] += w * v_p[i];
                    }
                }
            }
            apply_sigmoid_gate(out, &gate);
        }
        let mut attn_out = vec![0.0; t * hd];
        self.rows_matmul_into(&attn_out_in, &a.attn_output, t, &mut attn_out)?;
        for (h, o) in hidden.iter_mut().zip(&attn_out) {
            *h += *o;
        }
        self.prefill_ffn(hidden, &a.post_attention_norm, &a.ffn_gate, &a.ffn_up, &a.ffn_down, t)
    }
}
