// PREFILL-CPU (#2787): batched CPU prefill.
//
// ## What was wrong
//
// `generate_with_cache*` processed the prompt by calling
// `forward_single_with_cache` once per prompt token, and every batched
// `prefill_*` in the tree lived under `src/cuda/`. So on CPU, prefill WAS
// decode: a 7B Q4_K_M model streams ~4.4 GB of weights from RAM per call, once
// per prompt token. Measured on `lambda` (Threadripper 7960X, 48 threads,
// `qwen2.5-coder-7b-instruct-q4_k_m.gguf`, 513-token prompt, 128 generated):
//
//     prefill 8.609 tok/s   decode 7.760 tok/s   ratio 1.109   t_req 76.1 s
//
// Prefill and decode were indistinguishable, which is the signature of a
// missing GEMM. The issue records 1.07x on `intel` and 1.02x on `mini`.
//
// ## What this does
//
// Processes the prompt in chunks of `PREFILL_CHUNK` tokens. Within a chunk the
// six projections (QKV, O, gate, up, down — and the LM head is left alone) run
// as ONE batched matmul over all chunk rows via
// `quantize::quantized_matmul_batch_into`, so each weight row is loaded from
// RAM once and reused across the chunk instead of once per token.
//
// Attention is NOT batched: each query still runs `attention_with_cache_gqa_into`
// against the cache prefix, in position order, appending K/V as it goes. That is
// the identical call sequence the per-token path makes, and it is not the
// bottleneck — the projections are.
//
// ## Equivalence, and why it is checkable
//
// Every kernel here is the one the per-token path calls, on the same inputs:
// same `ops::rms_norm_into`, same `apply_rope`, same
// `attention_with_cache_gqa_into`, same `ops::silu`, and a batched matmul whose
// per-element result is bit-identical to `fused_q{4,5,6}k_parallel_matvec_into`
// (proved in `quantize::batched_matmul::tests`). So the batched path must
// produce the same KV cache and the same logits — not merely close ones.
// `falsify_batched_prefill.rs` asserts that on a real model.
//
// ## Scope: `supports_batched_prefill` is deliberately narrow
//
// Anything outside the covered class — learned position embeddings, LayerNorm
// architectures, Gemma's `(1+w)` offset or post-norms, a non-gated FFN, a fused
// gate_up tensor, a weight in a quantization format with no batched kernel —
// falls back to the per-token loop. A model that cannot be batched must run
// slowly, never differently.

/// Prompt tokens per batched chunk.
///
/// Chosen by sweep, not by intuition. `lambda` (Threadripper 7960X, 48 threads),
/// `qwen2.5-coder-7b-instruct-q4_k_m.gguf`, 513-token prompt, prefill tok/s:
///
/// | chunk | 8 | 16 | 32 | 64 | 128 | 256 |
/// |---|---|---|---|---|---|---|
/// | tok/s | 22.5 | 34.6 | 33.3 | 37.6 | 40.4 | 41.5 |
///
/// It does NOT scale with the chunk the way pure bandwidth amortisation would,
/// and that is the honest ceiling of this design: looping the existing per-row
/// dot over the chunk's columns amortises the weight LOAD but repeats the Q4_K
/// UNPACK per column. Past ~128 the unpack is the bottleneck and more rows buy
/// almost nothing while the FFN scratch (`chunk * intermediate_dim * 4` bytes,
/// 9.7 MB at 128 on a 7B) grows linearly. A kernel that unpacks a weight row
/// once into a register tile and multiplies it against all columns would lift
/// that ceiling; it is a separate change (see #2787 follow-up).
///
/// Overridable via `APR_PREFILL_CHUNK`.
const PREFILL_CHUNK: usize = 128;

impl OwnedQuantizedModel {
    /// Chunk width for batched prefill, honouring `APR_PREFILL_CHUNK`.
    ///
    /// `APR_PREFILL_CHUNK=1` reduces the batched path to one row per matmul,
    /// which is the arithmetic of the per-token path — useful for isolating the
    /// batching from every other difference when measuring.
    fn prefill_chunk_size() -> usize {
        std::env::var("APR_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(PREFILL_CHUNK)
    }

    /// Kill switch for the batched path, ON by default.
    ///
    /// `APR_BATCHED_PREFILL=0` sends `prefill_prompt` back to the per-token loop.
    /// It exists so the two arms can be interleaved on ONE binary: measuring
    /// before/after with two builds confounds the change with everything else
    /// that differs between them.
    fn batched_prefill_enabled() -> bool {
        !matches!(std::env::var("APR_BATCHED_PREFILL").as_deref(), Ok("0"))
    }

    /// Run `prompt` through the model, populating `cache`, and return the logits
    /// for the last prompt token — the value the sampling loop needs.
    ///
    /// Takes the batched path when this model is in the covered class and the
    /// prompt is longer than one token; otherwise runs the per-token loop that
    /// has always been here. Both return the same logits.
    ///
    /// # Errors
    ///
    /// Propagates any forward-pass error.
    pub fn prefill_prompt(
        &self,
        prompt: &[u32],
        cache: &mut OwnedQuantizedKVCache,
    ) -> Result<Vec<f32>> {
        if prompt.len() > 1 && Self::batched_prefill_enabled() && self.supports_batched_prefill() {
            return self.forward_prefill_batched(prompt, cache, 0);
        }
        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache(token_id, cache, pos)?;
        }
        Ok(logits)
    }

    /// Every weight this path multiplies must have a batched kernel and real data.
    fn weight_is_batchable(t: &crate::gguf::OwnedQuantizedTensor) -> bool {
        crate::quantize::batched_matmul_supports(t.qtype) && !t.data.is_empty()
    }

    /// Architecture-level half of [`Self::supports_batched_prefill`].
    ///
    /// Each clause names a branch `forward_single_with_cache` takes that the
    /// batched pass does not implement.
    fn arch_is_batchable(&self) -> bool {
        let c = &self.config;
        c.constraints.uses_rmsnorm()
            && c.constraints.uses_rope()
            && !c.constraints.uses_absolute_positions()
            && c.constraints.has_gate_ffn()
            && !c.rmsnorm_unit_offset()
            && !c.is_gemma1()
            && !c.is_gemma2()
            && self.position_embedding.is_none()
            && !self.layers.is_empty()
    }

    /// Per-layer half of [`Self::supports_batched_prefill`].
    fn layer_is_batchable(l: &OwnedQuantizedLayer) -> bool {
        let ok = Self::weight_is_batchable;
        let qkv_ok = match &l.qkv_weight {
            crate::gguf::OwnedQKVWeights::Fused(w) => ok(w),
            crate::gguf::OwnedQKVWeights::Separate { q, k, v } => ok(q) && ok(k) && ok(v),
        };
        qkv_ok
            && l.ffn_gate_weight.as_ref().is_some_and(ok)
            && ok(&l.attn_output_weight)
            && ok(&l.ffn_up_weight)
            && ok(&l.ffn_down_weight)
            && l.ffn_norm_weight.is_some()
            && l.post_attn_norm_weight.is_none()
            && l.post_ffw_norm_weight.is_none()
            && l.attn_norm_bias.is_none()
    }

    /// True when [`Self::forward_prefill_batched`] covers this model.
    ///
    /// Conservative by construction: every condition is one the per-token path
    /// branches on, and a `false` sends the caller back to that path.
    #[must_use]
    pub fn supports_batched_prefill(&self) -> bool {
        // CUDA models route `fused_matmul` to the device; this is the CPU GEMM.
        #[cfg(feature = "cuda")]
        if self.cuda_executor.is_some() {
            return false;
        }
        self.arch_is_batchable() && self.layers.iter().all(Self::layer_is_batchable)
    }

    /// Batched prefill over `tokens`, appending to `cache` from `start_pos`.
    ///
    /// Returns the logits for the LAST token — the same value
    /// `forward_single_with_cache` returns for that position, so the caller's
    /// sampling loop is unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `tokens` is empty, when the cache cannot hold
    /// `start_pos + tokens.len()` positions, or when a matmul rejects its
    /// shapes. Callers should check [`Self::supports_batched_prefill`] first;
    /// this function does not silently fall back.
    pub fn forward_prefill_batched(
        &self,
        tokens: &[u32],
        cache: &mut OwnedQuantizedKVCache,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: "batched prefill: tokens must be non-empty".to_string(),
            });
        }
        if start_pos + tokens.len() > cache.capacity() {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!(
                    "batched prefill: {} tokens from {start_pos} exceed cache capacity {}",
                    tokens.len(),
                    cache.capacity()
                ),
            });
        }

        let chunk = Self::prefill_chunk_size();
        let mut last_hidden = Vec::new();
        let mut offset = 0usize;
        while offset < tokens.len() {
            let n = chunk.min(tokens.len() - offset);
            last_hidden = self.prefill_chunk_forward(
                &tokens[offset..offset + n],
                cache,
                start_pos + offset,
            )?;
            offset += n;
        }

        let hidden_dim = self.config.hidden_dim;
        let last = &last_hidden[last_hidden.len() - hidden_dim..];
        self.single_cache_final_output(last, start_pos + tokens.len() - 1, true)
    }

    /// QKV projection for a whole chunk, batched.
    ///
    /// Fused weights are one matmul; separate Q/K/V are three, then interleaved
    /// into the per-row `[q|k|v]` layout the rest of the layer expects — which
    /// is the layout `fused_rmsnorm_qkv_matmul` builds for a single token.
    fn prefill_qkv_batched(
        &self,
        layer: &OwnedQuantizedLayer,
        normed: &[f32],
        n: usize,
        qkv: &mut [f32],
    ) -> Result<()> {
        use crate::quantize::quantized_matmul_batch_into;
        let hidden_dim = self.config.hidden_dim;
        match &layer.qkv_weight {
            crate::gguf::OwnedQKVWeights::Fused(w) => quantized_matmul_batch_into(
                &w.data, w.qtype, normed, hidden_dim, w.out_dim, n, qkv,
            ),
            crate::gguf::OwnedQKVWeights::Separate { q, k, v } => {
                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut qo = vec![0.0f32; n * q.out_dim];
                let mut ko = vec![0.0f32; n * k.out_dim];
                let mut vo = vec![0.0f32; n * v.out_dim];
                quantized_matmul_batch_into(
                    &q.data, q.qtype, normed, hidden_dim, q.out_dim, n, &mut qo,
                )?;
                quantized_matmul_batch_into(
                    &k.data, k.qtype, normed, hidden_dim, k.out_dim, n, &mut ko,
                )?;
                quantized_matmul_batch_into(
                    &v.data, v.qtype, normed, hidden_dim, v.out_dim, n, &mut vo,
                )?;
                for s in 0..n {
                    let row = &mut qkv[s * qkv_dim..(s + 1) * qkv_dim];
                    row[..q.out_dim].copy_from_slice(&qo[s * q.out_dim..(s + 1) * q.out_dim]);
                    row[q.out_dim..q.out_dim + k.out_dim]
                        .copy_from_slice(&ko[s * k.out_dim..(s + 1) * k.out_dim]);
                    row[q.out_dim + k.out_dim..]
                        .copy_from_slice(&vo[s * v.out_dim..(s + 1) * v.out_dim]);
                }
                Ok(())
            },
        }
    }

    /// QKV bias, per-head QK norm and RoPE for every row of the chunk.
    ///
    /// Row `s` is at absolute position `base_pos + s` — the position the
    /// per-token loop would have passed for that token.
    fn prefill_rope_rows(
        &self,
        layer: &OwnedQuantizedLayer,
        qkv: &mut [f32],
        n: usize,
        base_pos: usize,
    ) {
        let (q_dim, kv_dim) = (self.config.q_dim(), self.config.kv_dim());
        let qkv_dim = q_dim + 2 * kv_dim;
        let eps = self.config.eps;
        for s in 0..n {
            let row = &mut qkv[s * qkv_dim..(s + 1) * qkv_dim];
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(row, bias);
            }
            if let Some(ref q_norm) = layer.attn_q_norm_weight {
                ops::apply_per_head_rms_norm(
                    &mut row[0..q_dim],
                    q_norm,
                    self.config.num_heads,
                    eps,
                );
            }
            if let Some(ref k_norm) = layer.attn_k_norm_weight {
                ops::apply_per_head_rms_norm(
                    &mut row[q_dim..q_dim + kv_dim],
                    k_norm,
                    self.config.num_kv_heads,
                    eps,
                );
            }
            self.apply_rope(&mut row[0..q_dim], base_pos + s, self.config.num_heads);
            self.apply_rope(
                &mut row[q_dim..q_dim + kv_dim],
                base_pos + s,
                self.config.num_kv_heads,
            );
        }
    }

    /// Attention for every row of the chunk, in position order, appending K/V
    /// to the cache as it goes.
    ///
    /// NOT batched, deliberately: this is the identical call sequence the
    /// per-token loop makes, and the projections are the bottleneck.
    fn prefill_attention_rows(
        &self,
        layer_idx: usize,
        qkv: &[f32],
        attn_out: &mut [f32],
        n: usize,
        cache: &mut OwnedQuantizedKVCache,
    ) {
        let (q_dim, kv_dim, head_dim) = (
            self.config.q_dim(),
            self.config.kv_dim(),
            self.config.head_dim(),
        );
        let qkv_dim = q_dim + 2 * kv_dim;
        let num_heads = self.config.num_heads;
        let q_per_kv = num_heads / self.config.num_kv_heads;

        for s in 0..n {
            let row = &qkv[s * qkv_dim..(s + 1) * qkv_dim];
            let (q, k, v) = (
                &row[0..q_dim],
                &row[q_dim..q_dim + kv_dim],
                &row[q_dim + kv_dim..q_dim + 2 * kv_dim],
            );
            let out = &mut attn_out[s * q_dim..(s + 1) * q_dim];
            let k_cache = cache.get_k(layer_idx);
            if k_cache.is_empty() {
                // First position of the sequence: no prior keys, so softmax over
                // the single current key is 1.0 and the output is V broadcast
                // across its query-head group. The branch
                // `forward_single_with_cache` takes at position 0.
                for q_head in 0..num_heads {
                    let v_start = (q_head / q_per_kv) * head_dim;
                    let o_start = q_head * head_dim;
                    out[o_start..o_start + head_dim]
                        .copy_from_slice(&v[v_start..v_start + head_dim]);
                }
            } else {
                let v_cache = cache.get_v(layer_idx);
                self.attention_with_cache_gqa_into(q, k_cache, v_cache, k, v, out);
            }
            cache.append(layer_idx, k, v);
        }
    }

    /// SwiGLU FFN for a whole chunk: gate and up batched, activation row-wise.
    ///
    /// Returns the activated `[n * intermediate_dim]` buffer to feed the down
    /// projection — the batched analogue of `single_cache_ffn_block`'s return.
    fn prefill_ffn_activate(
        &self,
        layer: &OwnedQuantizedLayer,
        layer_idx: usize,
        normed: &[f32],
        n: usize,
    ) -> Result<Vec<f32>> {
        use crate::quantize::quantized_matmul_batch_into;
        let hidden_dim = self.config.hidden_dim;
        let gate_w =
            layer
                .ffn_gate_weight
                .as_ref()
                .ok_or_else(|| crate::error::RealizarError::InvalidShape {
                    reason: format!("batched prefill: layer {layer_idx} has no ffn_gate_weight"),
                })?;
        let up_w = &layer.ffn_up_weight;
        let inter = up_w.out_dim;
        let mut gate = vec![0.0f32; n * inter];
        let mut up = vec![0.0f32; n * inter];
        quantized_matmul_batch_into(
            &gate_w.data,
            gate_w.qtype,
            normed,
            hidden_dim,
            inter,
            n,
            &mut gate,
        )?;
        quantized_matmul_batch_into(
            &up_w.data, up_w.qtype, normed, hidden_dim, inter, n, &mut up,
        )?;
        for s in 0..n {
            let g = &mut gate[s * inter..(s + 1) * inter];
            let u = &mut up[s * inter..(s + 1) * inter];
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(u, bias);
            }
            if let Some(ref bias) = layer.ffn_gate_bias {
                ops::add_bias(g, bias);
            }
            ops::silu(g);
            for i in 0..inter {
                g[i] *= u[i];
            }
        }
        Ok(gate)
    }

    /// Add a projection's rows into the residual stream, with optional bias.
    fn prefill_add_residual(proj: &mut [f32], hidden: &mut [f32], n: usize, dim: usize, bias: Option<&[f32]>) {
        for s in 0..n {
            let row = &mut proj[s * dim..(s + 1) * dim];
            if let Some(b) = bias {
                ops::add_bias(row, b);
            }
            for i in 0..dim {
                hidden[s * dim + i] += row[i];
            }
        }
    }

    /// RMSNorm every row of the chunk into `out`.
    fn prefill_norm_rows(&self, hidden: &[f32], weight: &[f32], n: usize, out: &mut [f32]) {
        let dim = self.config.hidden_dim;
        for s in 0..n {
            ops::rms_norm_into(
                &hidden[s * dim..(s + 1) * dim],
                weight,
                self.config.eps,
                &mut out[s * dim..(s + 1) * dim],
            );
        }
    }

    /// One chunk of `n` tokens through every layer. Returns the chunk's hidden
    /// states `[n * hidden_dim]`.
    fn prefill_chunk_forward(
        &self,
        tokens: &[u32],
        cache: &mut OwnedQuantizedKVCache,
        base_pos: usize,
    ) -> Result<Vec<f32>> {
        use crate::quantize::quantized_matmul_batch_into;

        let n = tokens.len();
        let hidden_dim = self.config.hidden_dim;
        let q_dim = self.config.q_dim();
        let qkv_dim = q_dim + 2 * self.config.kv_dim();

        let mut hidden = self.embed(tokens);

        // Buffers reused across all layers — allocated once per chunk.
        let mut normed = vec![0.0f32; n * hidden_dim];
        let mut qkv = vec![0.0f32; n * qkv_dim];
        let mut attn_out = vec![0.0f32; n * q_dim];
        let mut proj = vec![0.0f32; n * hidden_dim];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.prefill_norm_rows(&hidden, &layer.attn_norm_weight, n, &mut normed);
            self.prefill_qkv_batched(layer, &normed, n, &mut qkv)?;
            self.prefill_rope_rows(layer, &mut qkv, n, base_pos);
            self.prefill_attention_rows(layer_idx, &qkv, &mut attn_out, n, cache);

            let ow = &layer.attn_output_weight;
            quantized_matmul_batch_into(
                &ow.data, ow.qtype, &attn_out, q_dim, hidden_dim, n, &mut proj,
            )?;
            Self::prefill_add_residual(
                &mut proj,
                &mut hidden,
                n,
                hidden_dim,
                layer.attn_output_bias.as_deref(),
            );

            let ffn_norm = layer.ffn_norm_weight.as_ref().ok_or_else(|| {
                crate::error::RealizarError::InvalidShape {
                    reason: format!("batched prefill: layer {layer_idx} has no ffn_norm_weight"),
                }
            })?;
            self.prefill_norm_rows(&hidden, ffn_norm, n, &mut normed);
            let activated = self.prefill_ffn_activate(layer, layer_idx, &normed, n)?;

            let dw = &layer.ffn_down_weight;
            quantized_matmul_batch_into(
                &dw.data,
                dw.qtype,
                &activated,
                layer.ffn_up_weight.out_dim,
                hidden_dim,
                n,
                &mut proj,
            )?;
            Self::prefill_add_residual(
                &mut proj,
                &mut hidden,
                n,
                hidden_dim,
                layer.ffn_down_bias.as_deref(),
            );
        }

        cache.advance_by(n);
        Ok(hidden)
    }
}
