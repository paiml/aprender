impl OwnedQuantizedModel {
    /// FFN block for single-token cached forward pass
    ///
    /// Handles the match on (ffn_norm_weight, ffn_gate_weight) to select between:
    /// - Fused RMSNorm + SwiGLU path
    /// - Non-fused SwiGLU path (LayerNorm models with gate)
    /// - GELU path (no gate weight)
    ///
    /// Returns the activated FFN output before down projection.
    /// Contract-driven FFN block for single-token cached forward pass (GH-278).
    ///
    /// Uses `constraints.has_gate_ffn()` to select SwiGLU vs GELU path,
    /// with fused RMSNorm optimization when applicable.
    fn single_cache_ffn_block(
        &self,
        hidden: &[f32],
        layer_idx: usize,
        use_rmsnorm: bool,
    ) -> Result<Vec<f32>> {
        let layer = &self.layers[layer_idx];

        if self.config.constraints.has_gate_ffn() {
            // GH-306: Fused path only when separate gate weight exists
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // Fused RMSNorm + SwiGLU (LLaMA, TinyLlama, Mistral, etc.)
                if use_rmsnorm {
                    if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                        let (mut ffn_up, mut ffn_gate) = self.fused_rmsnorm_ffn_up_gate(
                            hidden, ffn_norm, self.config.eps,
                            &layer.ffn_up_weight, gate_weight,
                        )?;
                        if let Some(ref bias) = layer.ffn_up_bias {
                            ops::add_bias(&mut ffn_up, bias);
                        }
                        if let Some(ref bias) = layer.ffn_gate_bias {
                            ops::add_bias(&mut ffn_gate, bias);
                        }
                        ops::silu(&mut ffn_gate);
                        for i in 0..ffn_gate.len() {
                            ffn_gate[i] *= ffn_up[i];
                        }
                        return Ok(ffn_gate);
                    }
                }

                // Non-fused gated path (LayerNorm models or no FFN norm)
                let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                    if use_rmsnorm {
                        ops::rms_norm(hidden, ffn_norm, self.config.eps)
                    } else {
                        ops::layer_norm(
                            hidden, ffn_norm,
                            layer.ffn_norm_bias.as_deref(), self.config.eps,
                        )
                    }
                } else {
                    hidden.to_vec()
                };

                let out_dim = layer.ffn_up_weight.out_dim;
                let mut ffn_up = vec![0.0f32; out_dim];
                let mut ffn_gate = vec![0.0f32; out_dim];
                self.fused_gate_up_matmul_into(
                    &ffn_input, gate_weight, &layer.ffn_up_weight,
                    &mut ffn_gate, &mut ffn_up,
                )?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_up, bias);
                }
                if let Some(ref bias) = layer.ffn_gate_bias {
                    ops::add_bias(&mut ffn_gate, bias);
                }
                ops::silu(&mut ffn_gate);
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }
                Ok(ffn_gate)
            } else {
                // GH-306: Fused gate_up weight (Phi-3.5) — single matmul, split in half
                let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                    if use_rmsnorm {
                        ops::rms_norm(hidden, ffn_norm, self.config.eps)
                    } else {
                        ops::layer_norm(
                            hidden, ffn_norm,
                            layer.ffn_norm_bias.as_deref(), self.config.eps,
                        )
                    }
                } else {
                    hidden.to_vec()
                };

                let fused = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                let half = fused.len() / 2;
                let mut ffn_gate = fused[..half].to_vec();
                let mut ffn_up = fused[half..].to_vec();
                if let Some(ref bias) = layer.ffn_up_bias {
                    // Split bias too if present
                    let bias_half = bias.len() / 2;
                    ops::add_bias(&mut ffn_gate, &bias[..bias_half]);
                    ops::add_bias(&mut ffn_up, &bias[bias_half..]);
                }
                ops::silu(&mut ffn_gate);
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }
                Ok(ffn_gate)
            }
        } else {
            // GELU path (GPT-2, BERT, etc.) - no gate weight
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    ops::rms_norm(hidden, ffn_norm, self.config.eps)
                } else {
                    ops::layer_norm(
                        hidden, ffn_norm,
                        layer.ffn_norm_bias.as_deref(), self.config.eps,
                    )
                }
            } else {
                hidden.to_vec()
            };

            let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(&mut ffn_hidden, bias);
            }
            ops::gelu(&mut ffn_hidden);
            Ok(ffn_hidden)
        }
    }

    /// Final output computation for single-token cached forward pass
    ///
    /// Handles everything after the layer loop: cache advance, debug logging,
    /// final layer norm, LM head projection, debug logits verification,
    /// and LM head bias application.
    fn single_cache_final_output(
        &self,
        hidden: &[f32],
        position: usize,
        use_rmsnorm: bool,
    ) -> Result<Vec<f32>> {
        let debug_forward = std::env::var("REALIZAR_DEBUG_FORWARD").is_ok();

        // DEBUG: Print hidden state before LM head
        if debug_forward {
            let hidden_sum: f32 = hidden.iter().sum();
            let hidden_max = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let hidden_min = hidden.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!(
                "[DEBUG-FORWARD] Hidden after all layers: sum={:.4}, min={:.4}, max={:.4}",
                hidden_sum, hidden_min, hidden_max
            );
            eprintln!(
                "[DEBUG-FORWARD] Hidden[0..8]: {:?}",
                &hidden[..8.min(hidden.len())]
            );
            eprintln!(
                "[DEBUG-LM-HEAD] lm_head_weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
                self.lm_head_weight.in_dim,
                self.lm_head_weight.out_dim,
                self.lm_head_weight.qtype,
                self.lm_head_weight.data.len()
            );
            eprintln!(
                "[DEBUG-LM-HEAD] First 16 bytes of lm_head data: {:02x?}",
                &self.lm_head_weight.data[..16.min(self.lm_head_weight.data.len())]
            );
            eprintln!(
                "[DEBUG-LM-HEAD] output_norm_weight[0..4]: {:?}",
                &self.output_norm_weight[..4.min(self.output_norm_weight.len())]
            );
        }

        // 3+4. Fused final layer norm + LM head projection
        // For RMSNorm models: fuse norm + matmul to eliminate intermediate allocation
        let mut logits = if use_rmsnorm {
            self.fused_rmsnorm_lm_head(hidden)?
        } else {
            let normed = ops::layer_norm(
                hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            );
            self.fused_matmul(&normed, &self.lm_head_weight)?
        };

        // DEBUG: Verify Q8_0 matmul by manual computation
        if debug_forward {
            self.debug_verify_lm_head(hidden, &logits, position);
        }

        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Debug verification of LM head output by manual Q8_0 dequantization
    ///
    /// Manually dequantizes row 0 of the LM head weight matrix and computes
    /// a dot product to verify the fused matmul result is correct.
    fn debug_verify_lm_head(&self, hidden: &[f32], logits: &[f32], _position: usize) {
        // Get the normalized hidden state
        let normed = ops::rms_norm(hidden, &self.output_norm_weight, self.config.eps);
        eprintln!(
            "[DEBUG-VERIFY] Normed hidden[0..8]: {:?}",
            &normed[..8.min(normed.len())]
        );

        // Manual dequantize row 0 of LM head weight
        const Q8_0_BLOCK_BYTES: usize = 34;
        const Q8_0_BLOCK_SIZE: usize = 32;
        let blocks_per_row = self.lm_head_weight.in_dim.div_ceil(Q8_0_BLOCK_SIZE);
        let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

        // Dequantize row 0 (token 0's projection weights)
        let row0_data = &self.lm_head_weight.data[0..bytes_per_row];
        let mut row0_f32 = vec![0.0f32; self.lm_head_weight.in_dim];
        for block_idx in 0..blocks_per_row {
            let block_start = block_idx * Q8_0_BLOCK_BYTES;
            let block = &row0_data[block_start..block_start + Q8_0_BLOCK_BYTES];
            let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            for j in 0..32 {
                let idx = block_idx * 32 + j;
                if idx >= self.lm_head_weight.in_dim {
                    break;
                }
                row0_f32[idx] = (block[2 + j] as i8 as f32) * scale;
            }
        }
        eprintln!(
            "[DEBUG-VERIFY] LM head row 0 (dequantized) first 8: {:?}",
            &row0_f32[..8.min(row0_f32.len())]
        );

        // Compute dot product manually
        let manual_logit0: f32 = normed.iter().zip(row0_f32.iter()).map(|(a, b)| a * b).sum();
        eprintln!("[DEBUG-VERIFY] Manual logits[0] = {:.6}", manual_logit0);
        eprintln!("[DEBUG-VERIFY] Computed logits[0] = {:.6}", logits[0]);
        eprintln!(
            "[DEBUG-VERIFY] Difference = {:.6}",
            (manual_logit0 - logits[0]).abs()
        );

        // Check top tokens
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "[DEBUG-VERIFY] Top 5 tokens: {:?}",
            &indexed[..5.min(indexed.len())]
        );
    }

    /// Debug trace after token embedding (PMAT-260)
    ///
    /// Consolidates three environment-variable-gated debug logging blocks
    /// (`REALIZAR_DEBUG_FORWARD`, `CPU_DEBUG`, `APR_TRACE_LAYERS`) that fire
    /// immediately after `embed()`.
    fn debug_trace_embedding(&self, hidden: &[f32], token_id: u32, position: usize) {
        let debug_forward = std::env::var("REALIZAR_DEBUG_FORWARD").is_ok();
        if debug_forward {
            let hidden_sum: f32 = hidden.iter().sum();
            eprintln!("[DEBUG-FORWARD] Token={}, Position={}", token_id, position);
            eprintln!(
                "[DEBUG-FORWARD] After embed: sum={:.6}, hidden[0..4]={:?}",
                hidden_sum,
                &hidden[..4.min(hidden.len())]
            );
        }

        if std::env::var("CPU_DEBUG").is_ok() {
            let embed_sum: f32 = hidden.iter().sum();
            let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden.len() as f32).sqrt();
            eprintln!(
                "[GQA-DEBUG-CPU-EMBED] Embedding before L0: first 5 = {:?}, sum={:.4}, rms={:.4}",
                &hidden[..5.min(hidden.len())],
                embed_sum,
                rms
            );
        }

        if std::env::var("APR_TRACE_LAYERS").is_ok() {
            let hidden_dim = self.config.hidden_dim;
            eprintln!(
                "[PMAT-114-GGUF] Token ID: {}, position: {}",
                token_id, position
            );
            let sum: f32 = hidden.iter().sum();
            let mean = sum / hidden_dim as f32;
            let min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[PMAT-114-GGUF] After embed: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                mean,
                min,
                max,
                &hidden[..5.min(hidden.len())]
            );
        }
    }

    /// Debug trace QKV projection for layer 0 (PMAT-260)
    ///
    /// Prints K-vector mean before bias addition when `APR_TRACE_LAYERS` is set
    /// and this is layer 0.
    fn debug_trace_qkv(&self, qkv: &[f32], layer_idx: usize, _hidden_dim: usize) {
        if layer_idx != 0 {
            return;
        }
        if std::env::var("APR_TRACE_LAYERS").is_err() {
            return;
        }
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let q_dim = self.config.q_dim();
        let kv_dim = self.config.kv_dim();

        let k = &qkv[q_dim..q_dim + kv_dim];
        let k_mean: f32 = k.iter().sum::<f32>() / kv_dim as f32;
        eprintln!("[PMAT-114-GGUF] L0 K BEFORE bias: mean={:.6}", k_mean);
    }

    /// Debug trace QKV after bias for layer 0 (PMAT-260)
    ///
    /// Prints bias stats and Q/K/V means after bias addition (pre-RoPE)
    /// when `APR_TRACE_LAYERS` is set and this is layer 0.
    fn debug_trace_qkv_after_bias(
        &self,
        qkv: &[f32],
        layer: &crate::gguf::OwnedQuantizedLayer,
        layer_idx: usize,
        _hidden_dim: usize,
    ) {
        if layer_idx != 0 || std::env::var("APR_TRACE_LAYERS").is_err() {
            return;
        }
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let q_dim = self.config.q_dim();
        let kv_dim = self.config.kv_dim();

        eprintln!(
            "[PMAT-114-GGUF] L0 has_qkv_bias={}",
            layer.qkv_bias.is_some()
        );
        if let Some(ref bias) = layer.qkv_bias {
            let k_bias = &bias[q_dim..q_dim + kv_dim];
            let k_bias_mean: f32 = k_bias.iter().sum::<f32>() / kv_dim as f32;
            eprintln!(
                "[PMAT-114-GGUF] L0 K bias mean={:.6}, first5={:?}",
                k_bias_mean,
                &k_bias[..5.min(kv_dim)]
            );
        }

        let q = &qkv[0..q_dim];
        let k = &qkv[q_dim..q_dim + kv_dim];
        let v = &qkv[q_dim + kv_dim..q_dim + 2 * kv_dim];
        let q_mean: f32 = q.iter().sum::<f32>() / q_dim as f32;
        let k_mean: f32 = k.iter().sum::<f32>() / kv_dim as f32;
        let v_mean: f32 = v.iter().sum::<f32>() / kv_dim as f32;
        eprintln!(
            "[PMAT-114-GGUF] L0 after QKV (pre-RoPE): Q mean={:.6}, K mean={:.6}, V mean={:.6}",
            q_mean, k_mean, v_mean
        );
        eprintln!(
            "[PMAT-114-GGUF] L0 Q first5={:?}",
            q.get(..5).unwrap_or(&[])
        );
    }

    /// Debug CPU attention output for layer 0 (PMAT-260)
    ///
    /// Prints per-head attention output for CORRECTNESS-013 validation
    /// when `CPU_DEBUG` is set and position >= 1 for layer 0.
    fn debug_trace_attention_output(
        attn_out: &[f32],
        layer_idx: usize,
        position: usize,
        head_dim: usize,
    ) {
        if layer_idx != 0 || position < 1 || std::env::var("CPU_DEBUG").is_err() {
            return;
        }
        eprintln!(
            "[CORRECTNESS-013-CPU] Layer 0 attention output at pos={}, first 10: {:?}",
            position,
            &attn_out[..10.min(attn_out.len())]
        );
        for h in 0..3 {
            let start = h * head_dim;
            eprintln!(
                "[CORRECTNESS-013-CPU] Head {} first 5: {:?}",
                h,
                &attn_out[start..start + 5]
            );
        }
    }

    /// Debug trace after processing a layer (PMAT-260)
    ///
    /// Consolidates three environment-variable-gated debug logging blocks
    /// (`REALIZAR_DEBUG_FORWARD`, `CPU_DEBUG`, `APR_TRACE_LAYERS`) that fire
    /// after each transformer layer's residual connections.
    fn debug_trace_layer_output(&self, hidden: &[f32], layer_idx: usize) {
        let hidden_dim = self.config.hidden_dim;

        if std::env::var("REALIZAR_DEBUG_FORWARD").is_ok() && layer_idx == 0 {
            let hidden_sum: f32 = hidden.iter().sum();
            eprintln!(
                "[DEBUG-FORWARD] After layer 0: sum={:.6}, hidden[0..4]={:?}",
                hidden_sum,
                &hidden[..4.min(hidden.len())]
            );
        }

        if std::env::var("CPU_DEBUG").is_ok() && layer_idx == 0 {
            let hidden_sum: f32 = hidden.iter().sum();
            let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden.len() as f32).sqrt();
            eprintln!(
                "[GQA-DEBUG-CPU-L0] After layer 0: first 5 = {:?}, sum={:.4}, rms={:.4}",
                &hidden[..5.min(hidden.len())],
                hidden_sum,
                rms
            );
        }

        if std::env::var("APR_TRACE_LAYERS").is_ok()
            && (layer_idx < 2 || layer_idx == self.layers.len() - 1)
        {
            let sum: f32 = hidden.iter().sum();
            let mean = sum / hidden_dim as f32;
            let min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[PMAT-114-GGUF] After layer {}: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                layer_idx,
                mean,
                min,
                max,
                &hidden[..5.min(hidden.len())]
            );
        }
    }

    /// Forward pass for a single token using KV cache (IMP-101c)
    ///
    /// This is O(n) per token instead of O(n^2) due to KV cache reuse.
    ///
    /// # Arguments
    /// * `token_id` - Single input token ID
    /// * `cache` - Mutable reference to KV cache
    /// * `position` - Position in sequence for RoPE
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    pub fn forward_single_with_cache(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // GH-278: Add learned position embedding for absolute encoding (GPT-2, BERT, whisper)
        if self.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.position_embedding {
                let start = position * hidden_dim;
                let end = start + hidden_dim;
                if end <= pos_emb.len() {
                    for i in 0..hidden_dim {
                        hidden[i] += pos_emb[start + i];
                    }
                }
            }
        }

        // GH-278: Use contract-derived norm type.
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();

        // PMAT-305/307: Pre-allocate workspace buffers — reused across all layers.
        let mut attn_out_buffer = vec![0.0f32; self.config.q_dim()];
        let mut o_proj_buffer = vec![0.0f32; hidden_dim];
        let mut ffn_down_buffer = vec![0.0f32; hidden_dim];
        // PMAT-307: QKV workspace — eliminates 28 Vec allocs per token
        let qkv_dim = self.config.q_dim() + 2 * self.config.kv_dim();
        let mut qkv_buffer = vec![0.0f32; qkv_dim];

        // DEBUG: Consolidated embedding trace (PMAT-260)
        self.debug_trace_embedding(&hidden, token_id, position);

        // GH-559 DIAGNOSTIC: Dump CPU hidden state per layer for GPU comparison
        static CPU_LAYER_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let cpu_layer_debug = *CPU_LAYER_DEBUG.get_or_init(|| {
            std::env::var("CPU_LAYER_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // GH-559: Dump CPU RMSNorm output for Layer 0 comparison with GPU
        if cpu_layer_debug {
            let gamma = &self.layers[0].attn_norm_weight;
            let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden.len() as f32 + self.config.eps).sqrt();
            let normed: Vec<f32> = hidden.iter().zip(gamma.iter())
                .map(|(x, g)| (x / rms) * g)
                .collect();
            eprintln!(
                "[GH-559-CPU] Layer 0 RMSNorm: rms={:.6}, first5={:?}",
                rms, &normed[..5.min(normed.len())]
            );
        }

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a+2b. Fused attention layer norm + QKV projection
            // For RMSNorm models: fuse norm + matmul to eliminate intermediate allocation
            // For LayerNorm models: use separate operations (has bias)
            // PMAT-307: Use workspace for QKV to eliminate per-layer Vec alloc.
            // For fused QKV weights: norm_into + matmul_into → qkv_buffer
            let qkv_actual_dim;
            let mut qkv = if use_rmsnorm {
                match &layer.qkv_weight {
                    crate::gguf::quantized::OwnedQKVWeights::Fused(ref w) => {
                        // RMSNorm → o_proj_buffer (reuse as temp), matmul → qkv_buffer
                        ops::rms_norm_into(&hidden, &layer.attn_norm_weight, self.config.eps,
                            &mut o_proj_buffer[..hidden_dim]);
                        qkv_actual_dim = w.out_dim;
                        self.fused_matmul_into(&o_proj_buffer[..hidden_dim], w,
                            &mut qkv_buffer[..w.out_dim])?;
                        &mut qkv_buffer[..w.out_dim]
                    },
                    _ => {
                        // Separate Q/K/V: use allocating path (rayon::join needs ownership)
                        qkv_actual_dim = 0; // unused
                        let v = self.fused_rmsnorm_qkv_matmul(
                            &hidden, &layer.attn_norm_weight, self.config.eps, &layer.qkv_weight)?;
                        // Copy to qkv_buffer for uniform handling below
                        qkv_buffer[..v.len()].copy_from_slice(&v);
                        &mut qkv_buffer[..v.len()]
                    },
                }
            } else {
                let normed = ops::layer_norm(
                    &hidden, &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(), self.config.eps);
                let v = self.qkv_matmul(&normed, &layer.qkv_weight)?;
                qkv_actual_dim = 0;
                qkv_buffer[..v.len()].copy_from_slice(&v);
                &mut qkv_buffer[..v.len()]
            };

            // PMAT-114: Trace QKV BEFORE bias (PMAT-260)
            self.debug_trace_qkv(&qkv, layer_idx, hidden_dim);

            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            // Q: [hidden_dim] = [num_heads * head_dim]
            // K: [kv_dim] = [num_kv_heads * head_dim]
            // V: [kv_dim] = [num_kv_heads * head_dim]
            // Optimization: apply RoPE in-place to avoid Q/K copies
            let num_kv_heads = self.config.num_kv_heads;
            // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
            let head_dim = self.config.head_dim();
            let q_dim = self.config.q_dim();
            let kv_dim = self.config.kv_dim();

            // PMAT-114: Trace QKV after bias for layer 0 (PMAT-260)
            self.debug_trace_qkv_after_bias(&qkv, layer, layer_idx, hidden_dim);

            // GH-479: Per-head QK RMSNorm (Qwen3) — after bias, before RoPE
            if let Some(ref q_norm) = layer.attn_q_norm_weight {
                ops::apply_per_head_rms_norm(
                    &mut qkv[0..q_dim],
                    q_norm,
                    self.config.num_heads,
                    self.config.eps,
                );
            }
            if let Some(ref k_norm) = layer.attn_k_norm_weight {
                ops::apply_per_head_rms_norm(
                    &mut qkv[q_dim..q_dim + kv_dim],
                    k_norm,
                    num_kv_heads,
                    self.config.eps,
                );
            }

            // GH-278: Skip RoPE for models with learned position embeddings (GPT-2)
            if self.config.constraints.uses_rope() {
                self.apply_rope(&mut qkv[0..q_dim], position, self.config.num_heads);
                self.apply_rope(
                    &mut qkv[q_dim..q_dim + kv_dim],
                    position,
                    num_kv_heads,
                );
            }

            // Use slices to avoid copies (only copy K for cache storage)
            // GH-479: Use q_dim (may differ from hidden_dim for Qwen3)
            let q = &qkv[0..q_dim];
            let k = &qkv[q_dim..q_dim + kv_dim];
            let v = &qkv[q_dim + kv_dim..q_dim + 2 * kv_dim];

            // 2d. Get cached K/V and compute attention with GQA support
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            // Use pre-allocated attention output buffer (reused across layers)
            if k_cache.is_empty() {
                // First token - no cache yet, output is just weighted V
                // With single query and single K/V, need to expand V for all Q heads
                let q_per_kv = self.config.num_heads / num_kv_heads;
                for q_head in 0..self.config.num_heads {
                    let kv_head = q_head / q_per_kv;
                    let v_start = kv_head * head_dim;
                    let out_start = q_head * head_dim;
                    attn_out_buffer[out_start..out_start + head_dim]
                        .copy_from_slice(&v[v_start..v_start + head_dim]);
                }
            } else {
                // Use cached K/V for attention with GQA
                // Uses pre-allocated buffer to avoid 704 Vec allocations per token
                self.attention_with_cache_gqa_into(q, k_cache, v_cache, k, v, &mut attn_out_buffer);

                // CORRECTNESS-013: Debug CPU attention output (PMAT-260)
                Self::debug_trace_attention_output(&attn_out_buffer, layer_idx, position, head_dim);
            }

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, k, v);

            // 2f. Attention output projection → o_proj_buffer (PMAT-305: no alloc)
            self.fused_matmul_into(&attn_out_buffer, &layer.attn_output_weight, &mut o_proj_buffer)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut o_proj_buffer, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += o_proj_buffer[i];
            }

            // 2h+2i. FFN with optional layer norm and SwiGLU/GELU activation
            let ffn_activated = self.single_cache_ffn_block(&hidden, layer_idx, use_rmsnorm)?;

            // 2j. FFN down projection → ffn_down_buffer (PMAT-305: no alloc)
            self.fused_matmul_into(&ffn_activated, &layer.ffn_down_weight, &mut ffn_down_buffer)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_down_buffer, bias);
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_down_buffer[i];
            }

            // DEBUG: Consolidated per-layer output trace (PMAT-260)
            self.debug_trace_layer_output(&hidden, layer_idx);

            // GH-559 DIAGNOSTIC: Dump CPU hidden state per layer
            if cpu_layer_debug {
                let sum: f32 = hidden.iter().sum();
                let rms: f32 = (hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32).sqrt();
                eprintln!(
                    "[GH-559-CPU] Layer {}/{} output: sum={:.6}, rms={:.6}, first5={:?}",
                    layer_idx, self.layers.len(), sum, rms,
                    &hidden[..5.min(hidden.len())]
                );
                // GH-559: For Layer 0, dump elements at Q4K super-block boundaries (every 256)
                if layer_idx == 0 {
                    for sb in 0..(hidden.len() / 256) {
                        let idx = sb * 256;
                        let end = (idx + 5).min(hidden.len());
                        let sb_sum: f32 = hidden[idx..idx+256.min(hidden.len()-idx)].iter().sum();
                        eprintln!(
                            "[GH-559-CPU] L0 sb{}: idx={}, sum={:.4}, vals={:?}",
                            sb, idx, sb_sum, &hidden[idx..end]
                        );
                    }
                }
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // Final output: norm + LM head + debug verification + bias
        self.single_cache_final_output(&hidden, position, use_rmsnorm)
    }
}
