
/// Add bias vector to each position in a projection buffer.
///
/// `buf` has shape `[seq_len, dim]`, `bias` has shape `[dim]`.
/// Adds `bias[i]` to `buf[pos * dim + i]` for all positions.
fn add_bias_per_position(buf: &mut [f32], bias: &[f32], dim: usize, seq_len: usize) {
    for pos in 0..seq_len {
        let start = pos * dim;
        for (i, &b) in bias.iter().enumerate() {
            buf[start + i] += b;
        }
    }
}

impl AprV2ModelCuda {

    /// Apply QKV bias for a transformer layer (both fused and separate formats).
    #[allow(clippy::too_many_arguments)]
    fn apply_qkv_bias_for_layer(
        &self,
        layer_idx: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        hidden_dim: usize,
        kv_dim: usize,
        seq_len: usize,
        _trace_layers: bool,
    ) -> Result<()> {
        // Try fused bias first (HuggingFace import with fused QKV)
        let fused_bias_name = format!("model.layers.{layer_idx}.self_attn.qkv_proj.bias");
        if let Ok(qkv_bias) = self.model.get_tensor_f32(&fused_bias_name) {
            if qkv_bias.len() >= hidden_dim + kv_dim + kv_dim {
                add_bias_per_position(q, &qkv_bias[..hidden_dim], hidden_dim, seq_len);
                add_bias_per_position(k, &qkv_bias[hidden_dim..hidden_dim + kv_dim], kv_dim, seq_len);
                add_bias_per_position(v, &qkv_bias[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim], kv_dim, seq_len);
                return Ok(());
            }
        }

        // PMAT-113 FIX: Try separate Q/K/V biases (APR converted from GGUF)
        let q_bias_name = format!("model.layers.{layer_idx}.self_attn.q_proj.bias");
        let k_bias_name = format!("model.layers.{layer_idx}.self_attn.k_proj.bias");
        let v_bias_name = format!("model.layers.{layer_idx}.self_attn.v_proj.bias");

        if let Ok(bias) = self.model.get_tensor_f32(&q_bias_name) {
            if bias.len() == hidden_dim {
                add_bias_per_position(q, &bias, hidden_dim, seq_len);
            }
        }
        if let Ok(bias) = self.model.get_tensor_f32(&k_bias_name) {
            if bias.len() == kv_dim {
                add_bias_per_position(k, &bias, kv_dim, seq_len);
            }
        }
        if let Ok(bias) = self.model.get_tensor_f32(&v_bias_name) {
            if bias.len() == kv_dim {
                add_bias_per_position(v, &bias, kv_dim, seq_len);
            }
        }

        Ok(())
    }

    /// Attention computation for a single layer: RoPE + KV cache + output projection + residual.
    #[allow(clippy::too_many_arguments)]
    fn forward_cuda_attention_layer(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        hidden: &mut [f32],
        seq_len: usize,
        hidden_dim: usize,
        kv_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        o_name: &str,
        o_cache_name: &str,
        profiling: bool,
    ) -> Result<()> {
        let timer_attn = if profiling {
            Some(self.executor.profiler_mut().start("apr.Attention"))
        } else {
            None
        };

        // R-02 (Meyer DbC): rope_theta from metadata, or architecture-specific default.
        let rope_theta = self.model.metadata.rope_theta.unwrap_or_else(|| {
            let arch = self.model.metadata.architecture.as_deref().unwrap_or("unknown");
            crate::gguf::default_rope_theta_for_architecture(arch)
        });
        // GH-329: Use shared infer_rope_type() as single source of truth
        let rope_type = self.model.metadata.rope_type.unwrap_or_else(|| {
            let arch = self.model.metadata.architecture.as_deref().unwrap_or("");
            crate::gguf::infer_rope_type(arch)
        });
        let mut attn_out = vec![0.0f32; seq_len * hidden_dim];

        for pos in 0..seq_len {
            let q_pos_start = pos * hidden_dim;
            let k_pos_start = pos * kv_dim;
            let v_pos_start = pos * kv_dim;

            let mut q_pos = q[q_pos_start..q_pos_start + hidden_dim].to_vec();
            let mut k_pos = k[k_pos_start..k_pos_start + kv_dim].to_vec();
            let v_pos = v[v_pos_start..v_pos_start + kv_dim].to_vec();

            // Apply RoPE
            // CORRECTNESS-011: Use correct RoPE style based on rope_type
            let abs_position = self.kv_position as usize + pos;
            if rope_type == 2 {
                // NEOX style: split halves (i, i + half_dim)
                crate::inference::apply_rope(
                    &mut q_pos, hidden_dim, num_heads, abs_position, rope_theta,
                );
                crate::inference::apply_rope(
                    &mut k_pos, kv_dim, num_kv_heads, abs_position, rope_theta,
                );
            } else {
                // NORM style: adjacent pairs (2*i, 2*i+1)
                apply_rope_norm(
                    &mut q_pos, num_heads, head_dim, abs_position, rope_theta, 0,
                );
                apply_rope_norm(
                    &mut k_pos, num_kv_heads, head_dim, abs_position, rope_theta, 0,
                );
            }

            // Use incremental_attention_gpu to append K/V to cache and compute attention
            let mut out_pos = vec![0.0f32; hidden_dim];
            if let Err(e) = self.executor.incremental_attention_gpu(
                layer_idx, &q_pos, &k_pos, &v_pos, &mut out_pos,
            ) {
                // Fallback to simple_attention if GPU KV cache fails
                eprintln!(
                    "PMAT-110 WARNING: incremental_attention_gpu failed: {e}, using fallback"
                );
                let simple_out =
                    simple_attention(q, k, v, seq_len, num_heads, num_kv_heads, head_dim);
                attn_out = simple_out;
                break;
            }

            attn_out[q_pos_start..q_pos_start + hidden_dim].copy_from_slice(&out_pos);
        }

        if let Some(t) = timer_attn {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // Output projection (GPU GEMM)
        let timer_oproj = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.OProj"))
        } else {
            None
        };
        let attn_proj = if self.has_cached_weight(o_cache_name) {
            self.gemm_cached_gpu(o_cache_name, &attn_out, seq_len, hidden_dim, hidden_dim)?
        } else {
            let o_weight = self.model.get_tensor_f32(o_name)?;
            let o_weight_t = transpose_matrix(&o_weight, hidden_dim, hidden_dim);
            self.gemm_gpu(&attn_out, &o_weight_t, seq_len, hidden_dim, hidden_dim)?
        };
        if let Some(t) = timer_oproj {
            let _ = self.executor.synchronize();
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // Residual connection
        let timer_res1 = if profiling {
            Some(self.executor.profiler_mut().start("apr.Residual"))
        } else {
            None
        };
        for (h, &a) in hidden.iter_mut().zip(attn_proj.iter()) {
            *h += a;
        }
        if let Some(t) = timer_res1 {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        Ok(())
    }

    /// FFN computation for a single layer: norm + gate/up projections + SiLU + down + residual.
    #[allow(clippy::too_many_arguments)]
    fn forward_cuda_ffn_layer(
        &mut self,
        layer_idx: usize,
        hidden: &mut [f32],
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        eps: f32,
        num_layers: usize,
        profiling: bool,
        trace_layers: bool,
    ) -> Result<()> {
        let ffn_norm_name = self.model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
            &format!("layers.{layer_idx}.post_attention_layernorm.weight"),
            &format!("transformer.h.{layer_idx}.ln_2.weight"),
            &format!("layers.{layer_idx}.ffn_norm.weight"),
            &format!("blk.{layer_idx}.ffn_norm.weight"),
        ])?;
        let gate_name = self.model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
            &format!("layers.{layer_idx}.mlp.gate_proj.weight"),
            &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
            &format!("layers.{layer_idx}.feed_forward.w1.weight"),
            &format!("blk.{layer_idx}.ffn_gate.weight"),
        ])?;
        let up_name = self.model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
            &format!("layers.{layer_idx}.mlp.up_proj.weight"),
            &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
            &format!("layers.{layer_idx}.feed_forward.w3.weight"),
            &format!("blk.{layer_idx}.ffn_up.weight"),
        ])?;
        let down_name = self.model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
            &format!("layers.{layer_idx}.mlp.down_proj.weight"),
            &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
            &format!("layers.{layer_idx}.feed_forward.w2.weight"),
            &format!("blk.{layer_idx}.ffn_down.weight"),
        ])?;

        // FFN RMSNorm
        let timer_rmsnorm2 = if profiling {
            Some(self.executor.profiler_mut().start("apr.RmsNorm"))
        } else {
            None
        };
        let ffn_norm = self.model.get_tensor_f32(&ffn_norm_name)?;
        let normed = rms_norm(hidden, &ffn_norm, eps);
        if let Some(t) = timer_rmsnorm2 {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // FFN projections - use cached weights if available
        // PMAT-805 FIX: Use GGUF-style cache names to match pre_cache_weights()
        let gate_cache_name = format!("blk.{}.ffn_gate.weight", layer_idx);
        let up_cache_name = format!("blk.{}.ffn_up.weight", layer_idx);
        let down_cache_name = format!("blk.{}.ffn_down.weight", layer_idx);

        let timer_ffn = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.FFN"))
        } else {
            None
        };
        let (gate_out, up_out) = if self.has_cached_weight(&gate_cache_name) {
            let gate_out = self.gemm_cached_gpu(
                &gate_cache_name, &normed, seq_len, hidden_dim, intermediate_dim,
            )?;
            let up_out = self.gemm_cached_gpu(
                &up_cache_name, &normed, seq_len, hidden_dim, intermediate_dim,
            )?;
            (gate_out, up_out)
        } else {
            let gate = self.model.get_tensor_f32(&gate_name)?;
            let up = self.model.get_tensor_f32(&up_name)?;
            let gate_t = transpose_matrix(&gate, intermediate_dim, hidden_dim);
            let up_t = transpose_matrix(&up, intermediate_dim, hidden_dim);
            let gate_out =
                self.gemm_gpu(&normed, &gate_t, seq_len, hidden_dim, intermediate_dim)?;
            let up_out =
                self.gemm_gpu(&normed, &up_t, seq_len, hidden_dim, intermediate_dim)?;
            (gate_out, up_out)
        };

        // SiLU activation and element-wise multiply (CPU - fast)
        let mut ffn_hidden = Vec::with_capacity(seq_len * intermediate_dim);
        for (g, u) in gate_out.iter().zip(up_out.iter()) {
            let silu = g * (1.0 / (1.0 + (-g).exp()));
            ffn_hidden.push(silu * u);
        }

        let ffn_out = if self.has_cached_weight(&down_cache_name) {
            self.gemm_cached_gpu(
                &down_cache_name, &ffn_hidden, seq_len, intermediate_dim, hidden_dim,
            )?
        } else {
            let down = self.model.get_tensor_f32(&down_name)?;
            let down_t = transpose_matrix(&down, hidden_dim, intermediate_dim);
            self.gemm_gpu(&ffn_hidden, &down_t, seq_len, intermediate_dim, hidden_dim)?
        };
        if let Some(t) = timer_ffn {
            let _ = self.executor.synchronize();
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // Residual
        let timer_res2 = if profiling {
            Some(self.executor.profiler_mut().start("apr.Residual"))
        } else {
            None
        };
        for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
            *h += f;
        }
        if let Some(t) = timer_res2 {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // PMAT-114: Layer tracing for Five-Whys analysis
        if trace_layers && (layer_idx < 2 || layer_idx == num_layers - 1) {
            let last_hidden = &hidden[hidden.len() - hidden_dim..];
            let sum: f32 = last_hidden.iter().sum();
            let mean = sum / hidden_dim as f32;
            let min = last_hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = last_hidden
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[PMAT-114] After layer {}: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                layer_idx, mean, min, max,
                &last_hidden[..5.min(hidden_dim)]
            );
        }

        Ok(())
    }

    /// Fast decode path for single token with indexed Q4K weights and CUDA graph.
    ///
    /// Returns logits if applicable, or None if fast path conditions aren't met.
    #[allow(clippy::too_many_arguments)]
    fn forward_cuda_indexed_decode(
        &mut self,
        token_id: u32,
        vocab_size: usize,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let position = self.kv_position;

        let input: Vec<f32> = self
            .get_embedding(token_id)
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!("Token {} out of embedding range", token_id),
            })?
            .to_vec();

        let mut output = vec![0.0f32; vocab_size];
        self.executor
            .forward_all_layers_gpu_to_logits_graphed(
                &input,
                &mut output,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size as u32,
                eps,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "forward_all_layers_gpu_to_logits_graphed".to_string(),
                reason: format!("Q4K graphed fast path failed: {e}"),
            })?;

        self.kv_position += 1;
        Ok(output)
    }

    /// QKV projection for a transformer layer (cached, fused, or separate weights).
    #[allow(clippy::too_many_arguments)]
    fn forward_cuda_qkv_projection(
        &mut self,
        layer_idx: usize,
        normed: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let q_cache_name = format!("blk.{}.attn_q.weight", layer_idx);
        let k_cache_name = format!("blk.{}.attn_k.weight", layer_idx);
        let v_cache_name = format!("blk.{}.attn_v.weight", layer_idx);

        if self.has_cached_weight(&q_cache_name) {
            let q = self.gemm_cached_gpu(&q_cache_name, normed, seq_len, hidden_dim, hidden_dim)?;
            let k = self.gemm_cached_gpu(&k_cache_name, normed, seq_len, hidden_dim, kv_dim)?;
            let v = self.gemm_cached_gpu(&v_cache_name, normed, seq_len, hidden_dim, kv_dim)?;
            return Ok((q, k, v));
        }

        // Check for fused QKV (from APR import)
        let fused_qkv_name = self.model.find_tensor_name(&[&format!(
            "model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        )]);

        if let Ok(fused_name) = fused_qkv_name {
            let qkv_weight = self.model.get_tensor_f32(&fused_name)?;
            let q_size = hidden_dim * hidden_dim;
            let k_size = kv_dim * hidden_dim;
            let v_size = kv_dim * hidden_dim;

            if qkv_weight.len() < q_size + k_size + v_size {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Fused QKV weight too small: {} < {}",
                        qkv_weight.len(),
                        q_size + k_size + v_size
                    ),
                });
            }

            let q_weight_t = transpose_matrix(&qkv_weight[..q_size], hidden_dim, hidden_dim);
            let k_weight_t = transpose_matrix(&qkv_weight[q_size..q_size + k_size], kv_dim, hidden_dim);
            let v_weight_t = transpose_matrix(&qkv_weight[q_size + k_size..q_size + k_size + v_size], kv_dim, hidden_dim);
            let q = self.gemm_gpu(normed, &q_weight_t, seq_len, hidden_dim, hidden_dim)?;
            let k = self.gemm_gpu(normed, &k_weight_t, seq_len, hidden_dim, kv_dim)?;
            let v = self.gemm_gpu(normed, &v_weight_t, seq_len, hidden_dim, kv_dim)?;
            return Ok((q, k, v));
        }

        // Separate Q/K/V weights
        let q_name = self.model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
            &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
            &format!("layers.{layer_idx}.attention.wq.weight"),
            &format!("blk.{layer_idx}.attn_q.weight"),
        ])?;
        let k_name = self.model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
            &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
            &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
            &format!("layers.{layer_idx}.attention.wk.weight"),
            &format!("blk.{layer_idx}.attn_k.weight"),
        ])?;
        let v_name = self.model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
            &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
            &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
            &format!("layers.{layer_idx}.attention.wv.weight"),
            &format!("blk.{layer_idx}.attn_v.weight"),
        ])?;

        let q_weight = self.model.get_tensor_f32(&q_name)?;
        let k_weight = self.model.get_tensor_f32(&k_name)?;
        let v_weight = self.model.get_tensor_f32(&v_name)?;
        let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
        let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
        let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);
        let q = self.gemm_gpu(normed, &q_weight_t, seq_len, hidden_dim, hidden_dim)?;
        let k = self.gemm_gpu(normed, &k_weight_t, seq_len, hidden_dim, kv_dim)?;
        let v = self.gemm_gpu(normed, &v_weight_t, seq_len, hidden_dim, kv_dim)?;
        Ok((q, k, v))
    }

    /// LM head projection: produces logits from the last token's hidden state.
    fn forward_cuda_lm_head(
        &mut self,
        last_hidden: &[f32],
        hidden_dim: usize,
        vocab_size: usize,
    ) -> Result<Vec<f32>> {
        if self.has_cached_weight("lm_head") {
            return self.gemm_cached_gpu("lm_head", last_hidden, 1, hidden_dim, vocab_size);
        }

        let lm_head_name = self.model.find_tensor_name(&[
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight",
        ])?;
        let lm_head = self.model.get_tensor_f32(&lm_head_name)?;

        // Contract: tied-embeddings-v1.yaml — lm_head must have vocab_size * hidden_dim elements
        let expected_len = vocab_size * hidden_dim;
        if lm_head.len() != expected_len {
            eprintln!(
                "Warning: lm_head '{}' has {} elements, expected {} (vocab={} * hidden={}). \
                 PMAT-328 dimension mismatch.",
                lm_head_name, lm_head.len(), expected_len, vocab_size, hidden_dim
            );
        }

        let is_tied_embedding = lm_head_name == "token_embd.weight"
            || lm_head_name.ends_with("embed_tokens.weight");

        let lm_head_for_gemm = if is_tied_embedding && lm_head.len() == expected_len {
            lm_head.clone()
        } else {
            transpose_matrix(&lm_head, vocab_size, hidden_dim)
        };
        self.gemm_gpu(last_hidden, &lm_head_for_gemm, 1, hidden_dim, vocab_size)
    }
}
