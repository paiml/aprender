impl AprTransformer {

    /// Create a new APR transformer with the given configuration
    pub fn new(config: AprTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let intermediate_dim = config.intermediate_dim;

        let layers = (0..config.num_layers)
            .map(|_| AprTransformerLayer::empty(hidden_dim, intermediate_dim))
            .collect();

        Self {
            config,
            token_embedding: vec![0.0; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; hidden_dim * vocab_size],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Generate tokens autoregressively (simplified version without KV cache)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Returns
    ///
    /// Generated token sequence (including prompt)
    pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_tokens {
            let logits = self.forward(&tokens)?;

            // Greedy sampling: take argmax
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32);

            tokens.push(next_token);

            // GH-330: Stop at EOS from model config (Design by Contract)
            if next_token == 0 {
                break;
            }
            if let Some(eos) = self.config.eos_token_id {
                if next_token == eos {
                    break;
                }
            }
        }

        Ok(tokens)
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.token_embedding.len();
        for layer in &self.layers {
            count += layer.num_parameters();
        }
        count += self.output_norm_weight.len();
        count += self.output_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.lm_head_weight.len();
        count += self.lm_head_bias.as_ref().map_or(0, Vec::len);
        count
    }

    /// Get memory size in bytes (F32 = 4 bytes per param)
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.num_parameters() * 4
    }

    /// Look up token embeddings
    #[must_use]
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let debug = std::env::var("REALIZE_DEBUG").is_ok();
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                if debug && token_id < 10 {
                    eprintln!(
                        "[DEBUG] embed token {}: offset={}, first 5: {:?}",
                        token_id,
                        offset,
                        &self.token_embedding[offset..offset + 5.min(hidden_dim)]
                    );
                }
                embeddings.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                // N-09: OOB token → zeros. Contract: embedding-lookup-v1.yaml
                eprintln!(
                    "Warning: AprTransformer::embed token_id {} OOB (offset={offset}, len={}). N-09 escape.",
                    token_id, self.token_embedding.len()
                );
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// RMSNorm (delegates to helpers module)
    fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        helpers::rms_norm(input, weight, bias, self.config.hidden_dim, eps)
    }

    /// Matrix multiplication (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn matmul(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        helpers::f32_matmul(input, weight, in_dim, out_dim)
    }

    /// M-FFN-GGUF-5 / SHIP-007 §22 fix: matvec with Q4K+Q8K dispatch matching GGUF.
    ///
    /// Used by `forward_traced` (inference.rs) to match the production decode
    /// path's Q4K+Q8K semantics. The cascade M91-M101 + M-FFN-GGUF-7 empirically
    /// validated that promoting GGUF-PATH semantics into APR forward closes the
    /// §27 layer-3 ffn_swigl 18.23× APR-vs-GGUF std-ratio.
    ///
    /// Multi-token aware: loops over sequence positions when seq_len > 1.
    /// Falls back to F32 matmul when Q4K bytes are unavailable.
    #[allow(clippy::unused_self)]
    fn matmul_q4k_or_f32_traced(
        &self,
        input: &[f32],
        q4k_bytes: Option<&[u8]>,
        q6k_bytes: Option<&[u8]>,
        f32_weight: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let seq_len = input.len() / in_dim;
        if let Some(q4k) = q4k_bytes {
            if let Ok(out) = Self::seq_matmul_q4k(q4k, input, seq_len, out_dim, in_dim) {
                return out;
            }
        }
        if let Some(q6k) = q6k_bytes {
            if let Ok(out) = Self::seq_matmul_q6k(q6k, input, seq_len, out_dim, in_dim) {
                return out;
            }
        }
        helpers::f32_matmul(input, f32_weight, in_dim, out_dim)
    }

    /// M-FFN-GGUF-5b / SHIP-007 §22 closure: split-Q4K QKV projection for the
    /// multi-token traced/forward paths.
    ///
    /// When `q4k_layer` exposes separate `attn_q_weight` / `attn_k_weight` /
    /// `attn_v_weight{,_q6k}` Q4K bytes (matching the production decode
    /// `forward_with_cache` storage layout), this helper computes Q, K, V
    /// independently across all sequence positions via `seq_matmul_q4k` /
    /// `seq_matmul_q6k`, then re-interleaves per-token to produce the fused
    /// `[Q_pos | K_pos | V_pos]` layout that the downstream RoPE +
    /// attention code expects (mirrors the F32 fused QKV matmul output of
    /// `f32_matmul(normed, qkv_weight, hidden_dim, qkv_dim)`).
    ///
    /// Mirrors `project_qkv_fused`'s semantics (single-token decode) at
    /// sequence granularity. Falls back to fused F32 matmul when any Q
    /// or K bytes are missing (V can be Q6K).
    ///
    /// Closes the 8th `forward_traced` / `forward()` matmul site that M-FFN-GGUF-5
    /// (PR #1550) left as F32 fallback because Q4K storage splits Q/K/V into
    /// separate arrays while APR uses a fused F32 `qkv_weight` array.
    #[allow(clippy::unused_self)]
    #[allow(clippy::too_many_arguments)]
    fn qkv_split_q4k_traced(
        &self,
        normed: &[f32],
        q4k_layer: Option<&Q4KLayerWeights>,
        fused_f32_weight: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        kv_size: usize,
        qkv_dim: usize,
    ) -> Vec<f32> {
        // Try Q4K-split path: requires separate attn_q + attn_k bytes;
        // V may be Q4K or Q6K (Q6K used for high-precision V on some 7B
        // qwen2.5 quantizations — mirrors `select_q4k_q6k` cascade).
        if let Some(q4k) = q4k_layer {
            let q_b = q4k.attn_q_weight.as_deref();
            let k_b = q4k.attn_k_weight.as_deref();
            let v_q4k = q4k.attn_v_weight.as_deref();
            let v_q6k = q4k.attn_v_weight_q6k.as_deref();

            if let (Some(qb), Some(kb)) = (q_b, k_b) {
                let q_out = Self::seq_matmul_q4k(qb, normed, seq_len, hidden_dim, hidden_dim).ok();
                let k_out = Self::seq_matmul_q4k(kb, normed, seq_len, kv_size, hidden_dim).ok();
                // V: prefer Q4K, fall back to Q6K.
                let v_out = if let Some(vb) = v_q4k {
                    Self::seq_matmul_q4k(vb, normed, seq_len, kv_size, hidden_dim).ok()
                } else if let Some(vb) = v_q6k {
                    Self::seq_matmul_q6k(vb, normed, seq_len, kv_size, hidden_dim).ok()
                } else {
                    None
                };
                if let (Some(q), Some(k), Some(v)) = (q_out, k_out, v_out) {
                    let mut qkv = vec![0.0f32; seq_len * qkv_dim];
                    for s in 0..seq_len {
                        let qkv_off = s * qkv_dim;
                        qkv[qkv_off..qkv_off + hidden_dim]
                            .copy_from_slice(&q[s * hidden_dim..(s + 1) * hidden_dim]);
                        qkv[qkv_off + hidden_dim..qkv_off + hidden_dim + kv_size]
                            .copy_from_slice(&k[s * kv_size..(s + 1) * kv_size]);
                        qkv[qkv_off + hidden_dim + kv_size
                            ..qkv_off + hidden_dim + 2 * kv_size]
                            .copy_from_slice(&v[s * kv_size..(s + 1) * kv_size]);
                    }
                    return qkv;
                }
            }
        }
        // F32 fused fallback: matches existing legacy semantics byte-for-byte.
        helpers::f32_matmul(normed, fused_f32_weight, hidden_dim, qkv_dim)
    }

    /// Add bias in-place (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn add_bias(&self, data: &mut [f32], bias: &[f32]) {
        helpers::add_bias_inplace(data, bias);
    }

    /// GELU activation (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn gelu(&self, data: &mut [f32]) {
        helpers::gelu_inplace(data);
    }

    /// Apply RoPE (delegates to helpers module)
    ///
    /// PMAT-797: pass the architecture-correct pairing convention. LLaMA-family
    /// uses NORM (adjacent pairs), Qwen/NeoX/Phi/Gemma use NEOX (split halves).
    fn apply_rope_f32(&self, x: &mut [f32], position: usize, num_heads: usize, head_dim: usize) {
        let rope_type = crate::gguf::infer_rope_type(&self.config.architecture);
        helpers::apply_rope_f32(
            x,
            position,
            num_heads,
            head_dim,
            self.config.rope_theta,
            rope_type,
        );
    }
}
