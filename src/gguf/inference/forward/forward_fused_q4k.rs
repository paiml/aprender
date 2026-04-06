impl OwnedQuantizedModel {
    /// Forward pass with fused Q4_K operations (IMP-100)
    ///
    /// This is 1.37x faster than dequantized f32 due to reduced memory bandwidth.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        contract_pre_decode_step!();
        let hidden_dim = self.config.hidden_dim;
        // Note: intermediate_dim is encoded in layer weight tensors (in_dim/out_dim)
        let _ = self.config.intermediate_dim;

        // 1. Token embedding lookup (f32, fast)
        let mut hidden = self.embed(token_ids);

        // GH-278: Add learned position embeddings for absolute encoding (GPT-2, BERT, whisper)
        if self.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.position_embedding {
                let hidden_dim = self.config.hidden_dim;
                for (s, _) in token_ids.iter().enumerate() {
                    let pos_start = s * hidden_dim;
                    let pos_end = pos_start + hidden_dim;
                    if pos_end <= pos_emb.len() {
                        let h_start = s * hidden_dim;
                        for i in 0..hidden_dim {
                            hidden[h_start + i] += pos_emb[pos_start + i];
                        }
                    }
                }
            }
        }

        // GH-278: Use contract-derived norm type instead of runtime heuristics.
        // Source: ArchConstraints::from_architecture() ← aprender/contracts/model-families/*.yaml
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();

        // 2. Process through transformer layers with FUSED Q4_K ops
        let cpu_debug_layers = std::env::var("CPU_DEBUG_LAYERS").is_ok();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm (RMSNorm for LLaMA, LayerNorm for others)
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };

            // CORRECTNESS-011: CPU intermediate debug at L0
            if cpu_debug_layers && layer_idx < 2 {
                let norm_label = if use_rmsnorm { "RMSNorm" } else { "LayerNorm" };
                eprintln!(
                    "[CPU-L{}] {}: first 3 = [{:.4}, {:.4}, {:.4}]",
                    layer_idx, norm_label, normed[0], normed[1], normed[2]
                );
            }

            // 2b. QKV projection with FUSED dequant+dot (1.37x faster)
            // Note: qkv_dim may differ from 3*hidden_dim for GQA models
            let qkv_dim = layer.qkv_weight.out_dim();
            // GQA-aware dimension computation (PARITY-GQA)
            // For GQA: q_dim = hidden_dim, k_dim = v_dim = num_kv_heads * head_dim
            let q_dim = layer.qkv_weight.q_dim_for_config(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.hidden_dim,
                self.config.head_dim(),
            );
            let k_dim = layer.qkv_weight.k_dim_for_config(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.hidden_dim,
                self.config.head_dim(),
            );
            let v_dim = layer.qkv_weight.v_dim_for_config(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.hidden_dim,
                self.config.head_dim(),
            );
            let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // CORRECTNESS-011: Q, K, V before RoPE (after bias)
            if cpu_debug_layers && (layer_idx < 2 || layer_idx == 4 || layer_idx == 5) {
                eprintln!(
                    "[CPU-L{}] Q (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx, qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]
                );
                // K starts at q_dim offset
                eprintln!(
                    "[CPU-L{}] K (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx,
                    qkv[q_dim],
                    qkv[q_dim + 1],
                    qkv[q_dim + 2],
                    qkv[q_dim + 3],
                    qkv[q_dim + 4]
                );
                // V starts at q_dim + k_dim offset
                let v_offset = q_dim + k_dim;
                eprintln!(
                    "[CPU-L{}] V (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx,
                    qkv[v_offset],
                    qkv[v_offset + 1],
                    qkv[v_offset + 2],
                    qkv[v_offset + 3],
                    qkv[v_offset + 4]
                );
            }

            // 2c. Proper attention with RoPE and causal mask (IMP-101)
            let seq_len = token_ids.len();

            // Extract Q, K, V and apply RoPE to Q and K
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * k_dim);
            let mut v_all = Vec::with_capacity(seq_len * v_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V for this position (QKV layout: [Q..., K..., V...])
                let mut q = qkv[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv[qkv_start + q_dim..qkv_start + q_dim + k_dim].to_vec();
                let v = &qkv[qkv_start + q_dim + k_dim..qkv_start + q_dim + k_dim + v_dim];

                // GH-278: Apply RoPE only for architectures that use it (contract-driven).
                // GPT-2/BERT/whisper use absolute position embeddings — no RoPE.
                if self.config.constraints.uses_rope() {
                    self.apply_rope(&mut q, s, self.config.num_heads);
                    self.apply_rope(&mut k, s, self.config.num_kv_heads);
                }

                // CORRECTNESS-011: Q after RoPE at position 0
                if cpu_debug_layers && layer_idx < 2 && s == 0 {
                    eprintln!(
                        "[CPU-L{}] Q (after RoPE): first 3 = [{:.4}, {:.4}, {:.4}]",
                        layer_idx, q[0], q[1], q[2]
                    );
                    eprintln!(
                        "[CPU-L{}] K (after RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                        layer_idx, k[0], k[1], k[2], k[3], k[4]
                    );
                    eprintln!(
                        "[CPU-L{}] V: first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                        layer_idx, v[0], v[1], v[2], v[3], v[4]
                    );
                }

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // Compute scaled dot-product attention with causal mask
            let attn_out = self.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // CORRECTNESS-011: Attention output
            if cpu_debug_layers && layer_idx < 2 {
                eprintln!(
                    "[CPU-L{}] Attn output: first 3 = [{:.4}, {:.4}, {:.4}]",
                    layer_idx, attn_out[0], attn_out[1], attn_out[2]
                );
            }

            // 2d. Attention output projection with FUSED ops
            // Input is q_dim (attention output), projects back to hidden_dim
            let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Pre-FFN layer norm (LLaMA uses separate ffn_norm with RMSNorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                // LLaMA-style: separate FFN layer norm (use RMSNorm for LLaMA)
                if use_rmsnorm {
                    ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                } else {
                    ops::layer_norm(
                        &hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    )
                }
            } else {
                // phi-2 style: no separate FFN norm, use hidden directly
                // (some models apply attn_norm again, but we've already done residual)
                hidden.clone()
            };

            // 2g. FFN with SwiGLU or GELU activation (contract-driven, GH-278)
            let ffn_activated = if self.config.constraints.has_gate_ffn() {
                if let Some(ref gate_weight) = layer.ffn_gate_weight {
                    // SwiGLU path (LLaMA, TinyLlama, Mistral, etc.)
                    let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }

                    let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    // SwiGLU: down(silu(gate(x)) * up(x))
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }

                    ffn_gate
                } else {
                    // GH-306: Fused gate_up weight (Phi-3.5) — single matmul, split in half
                    let fused = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    let half = fused.len() / 2;
                    let mut ffn_gate = fused[..half].to_vec();
                    let mut ffn_up = fused[half..].to_vec();
                    if let Some(ref bias) = layer.ffn_up_bias {
                        let bias_half = bias.len() / 2;
                        ops::add_bias(&mut ffn_gate, &bias[..bias_half]);
                        ops::add_bias(&mut ffn_up, &bias[bias_half..]);
                    }
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                }
            } else {
                // GELU path (GPT-2, BERT, etc.)
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);
                ffn_hidden
            };

            // 2g. FFN down projection with FUSED ops
            let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }

            // CORRECTNESS-011: Per-layer CPU debug output
            if cpu_debug_layers {
                let seq_len = token_ids.len();
                let last_hidden_start = (seq_len - 1) * hidden_dim;
                let last_h = &hidden[last_hidden_start..last_hidden_start + hidden_dim];
                let sum: f32 = last_h.iter().sum();
                let sq_sum: f32 = last_h.iter().map(|x| x * x).sum();
                let rms = (sq_sum / last_h.len() as f32).sqrt();
                eprintln!(
                    "[CPU-L{}] After layer: first 3 = [{:.4}, {:.4}, {:.4}], sum = {:.4}, rms = {:.4}",
                    layer_idx, last_h[0], last_h[1], last_h[2], sum, rms
                );
            }
        }

        // CORRECTNESS-011: CPU hidden state debug output (compare with GPU)
        if std::env::var("CPU_DEBUG").is_ok() {
            let seq_len = token_ids.len();
            let last_hidden_start = (seq_len - 1) * hidden_dim;
            let last_hidden_raw = &hidden[last_hidden_start..last_hidden_start + hidden_dim];

            let sum: f32 = last_hidden_raw.iter().sum();
            let sq_sum: f32 = last_hidden_raw.iter().map(|x| x * x).sum();
            let rms = (sq_sum / last_hidden_raw.len() as f32).sqrt();

            eprintln!("[CORRECTNESS-011] CPU Hidden before output_norm:");
            eprintln!(
                "  first 5 = {:?}",
                &last_hidden_raw[..5.min(last_hidden_raw.len())]
            );
            eprintln!("  sum = {:.4}, rms = {:.4}", sum, rms);
            eprintln!("  (GPU shows: sum=466.2486, rms=39.4793)");
        }

        // 3. Final layer norm (RMSNorm for LLaMA, LayerNorm for others)
        let normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };

        // CORRECTNESS-011: CPU normed hidden state debug output
        if std::env::var("CPU_DEBUG").is_ok() {
            let seq_len = token_ids.len();
            let last_normed_start = (seq_len - 1) * hidden_dim;
            let last_normed = &normed[last_normed_start..last_normed_start + hidden_dim];

            let sum: f32 = last_normed.iter().sum();
            let sq_sum: f32 = last_normed.iter().map(|x| x * x).sum();
            let rms = (sq_sum / last_normed.len() as f32).sqrt();

            eprintln!("[CORRECTNESS-011] CPU Normed hidden:");
            eprintln!("  first 5 = {:?}", &last_normed[..5.min(last_normed.len())]);
            eprintln!("  sum = {:.4}, rms = {:.4}", sum, rms);
            eprintln!("  (GPU shows: sum=107.5945, rms=4.6616)");
        }

        // 4. LM head projection with FUSED ops (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        // Compute logits using fused op
        let mut logits = self.fused_matmul(last_hidden, &self.lm_head_weight)?;

        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        contract_post_decode_step!(&logits);
        Ok(logits)
    }
}
