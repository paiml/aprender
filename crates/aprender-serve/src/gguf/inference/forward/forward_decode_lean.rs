// PMAT-305: Allocation-free M=1 decode forward pass.
// Eliminates ~364 Vec allocations per token by using pre-allocated workspace.

impl OwnedQuantizedModel {
    /// PMAT-305: Lean forward pass for M=1 decode.
    ///
    /// Same computation as `forward()` but uses pre-allocated workspace buffers.
    /// Only supports seq_len=1 (single token decode). Falls back to `forward()`
    /// for prefill (seq_len > 1).
    #[allow(private_interfaces)]
    pub fn forward_decode_lean(
        &self,
        token_ids: &[u32],
        workspace: &mut CpuWorkspace,
    ) -> Result<Vec<f32>> {
        // Only optimize M=1 decode. Prefill uses the allocating path.
        if token_ids.len() != 1 {
            return self.forward(token_ids);
        }

        let hidden_dim = self.config.hidden_dim;
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();

        // 1. Embed single token
        let mut hidden = self.embed(token_ids);

        if self.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.position_embedding {
                for i in 0..hidden_dim {
                    if i < pos_emb.len() {
                        hidden[i] += pos_emb[i];
                    }
                }
            }
        }

        // 2. Process layers with workspace buffers
        for layer in &self.layers {
            // 2a. Pre-attention RMSNorm → buf_hidden_a
            if use_rmsnorm {
                ops::rms_norm_into(
                    &hidden,
                    &layer.attn_norm_weight,
                    self.config.eps,
                    &mut workspace.buf_hidden_a[..hidden_dim],
                );
            } else {
                // LayerNorm: use allocating path (rare for LLaMA-family)
                let normed = ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                );
                workspace.buf_hidden_a[..hidden_dim].copy_from_slice(&normed);
            }

            // 2b. QKV matmul (allocates — QKV weight is OwnedQKVWeights, not single tensor)
            let q_dim = self.config.num_heads * self.config.head_dim();
            let k_dim = self.config.num_kv_heads * self.config.head_dim();
            let v_dim = k_dim;
            let mut qkv = self.qkv_matmul(&workspace.buf_hidden_a[..hidden_dim], &layer.qkv_weight)?;

            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V + RoPE + attention
            let mut q = qkv[..q_dim].to_vec();
            let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
            let v = qkv[q_dim + k_dim..q_dim + k_dim + v_dim].to_vec();

            if self.config.constraints.uses_rope() {
                self.apply_rope(&mut q, 0, self.config.num_heads);
                self.apply_rope(&mut k, 0, self.config.num_kv_heads);
            }

            let attn_out = self.causal_attention(&q, &k, &v, 1);

            // 2d. O projection → buf_hidden_a
            self.fused_matmul_into(
                &attn_out,
                &layer.attn_output_weight,
                &mut workspace.buf_hidden_a[..hidden_dim],
            )?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut workspace.buf_hidden_a[..hidden_dim], bias);
            }

            // 2e. Residual: hidden += attn_output
            for i in 0..hidden_dim {
                hidden[i] += workspace.buf_hidden_a[i];
            }

            // 2f. Pre-FFN RMSNorm → buf_hidden_a
            if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    ops::rms_norm_into(
                        &hidden,
                        ffn_norm,
                        self.config.eps,
                        &mut workspace.buf_hidden_a[..hidden_dim],
                    );
                } else {
                    let normed = ops::layer_norm(
                        &hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    );
                    workspace.buf_hidden_a[..hidden_dim].copy_from_slice(&normed);
                }
            } else {
                workspace.buf_hidden_a[..hidden_dim].copy_from_slice(&hidden);
            }

            // 2g. FFN (SwiGLU path — LLaMA/Qwen)
            if self.config.constraints.has_gate_ffn() {
                if let Some(ref gate_weight) = layer.ffn_gate_weight {
                    let intermediate = self.config.intermediate_dim;

                    // gate → buf_ffn_gate
                    self.fused_matmul_into(
                        &workspace.buf_hidden_a[..hidden_dim],
                        gate_weight,
                        &mut workspace.buf_ffn_gate[..intermediate],
                    )?;
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut workspace.buf_ffn_gate[..intermediate], bias);
                    }

                    // up → buf_ffn_up
                    self.fused_matmul_into(
                        &workspace.buf_hidden_a[..hidden_dim],
                        &layer.ffn_up_weight,
                        &mut workspace.buf_ffn_up[..intermediate],
                    )?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut workspace.buf_ffn_up[..intermediate], bias);
                    }

                    // SwiGLU: silu(gate) * up → buf_ffn_gate (in-place)
                    ops::silu(&mut workspace.buf_ffn_gate[..intermediate]);
                    for i in 0..intermediate {
                        workspace.buf_ffn_gate[i] *= workspace.buf_ffn_up[i];
                    }

                    // down → buf_hidden_a
                    self.fused_matmul_into(
                        &workspace.buf_ffn_gate[..intermediate],
                        &layer.ffn_down_weight,
                        &mut workspace.buf_hidden_a[..hidden_dim],
                    )?;
                    if let Some(ref bias) = layer.ffn_down_bias {
                        ops::add_bias(&mut workspace.buf_hidden_a[..hidden_dim], bias);
                    }
                }
            } else {
                // GELU path
                let intermediate = self.config.intermediate_dim;
                self.fused_matmul_into(
                    &workspace.buf_hidden_a[..hidden_dim],
                    &layer.ffn_up_weight,
                    &mut workspace.buf_ffn_gate[..intermediate],
                )?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut workspace.buf_ffn_gate[..intermediate], bias);
                }
                ops::gelu(&mut workspace.buf_ffn_gate[..intermediate]);

                self.fused_matmul_into(
                    &workspace.buf_ffn_gate[..intermediate],
                    &layer.ffn_down_weight,
                    &mut workspace.buf_hidden_a[..hidden_dim],
                )?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut workspace.buf_hidden_a[..hidden_dim], bias);
                }
            }

            // Residual: hidden += ffn_output
            for i in 0..hidden_dim {
                hidden[i] += workspace.buf_hidden_a[i];
            }
        }

        // 3. Output norm → buf_hidden_a
        if use_rmsnorm {
            ops::rms_norm_into(
                &hidden,
                &self.output_norm_weight,
                self.config.eps,
                &mut workspace.buf_hidden_a[..hidden_dim],
            );
        } else {
            let normed = ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            );
            workspace.buf_hidden_a[..hidden_dim].copy_from_slice(&normed);
        }

        // 4. LM head
        let mut logits = self.fused_matmul(&workspace.buf_hidden_a[..hidden_dim], &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }
}
