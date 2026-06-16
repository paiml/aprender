impl AprTransformer {

    /// Forward pass with layer-by-layer activation tracing.
    ///
    /// This is identical to `forward()` but collects statistics at each layer
    /// for debugging inference divergence issues.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// `ForwardTrace` containing logits and per-layer activation statistics
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn forward_traced(&self, token_ids: &[u32]) -> Result<ForwardTrace> {
        self.forward_traced_with_plan(token_ids, None)
    }

    /// SHIP-007 PR-C-real step 3: forward_traced + optional per-stage
    /// `SaveTensorPlan` emission.
    ///
    /// Identical to [`Self::forward_traced`] when `plan == None`. When
    /// `plan == Some(&plan)`, emits selected stage tensors to disk via
    /// [`crate::inference_trace::save_tensor_emit::maybe_save_stage`] at
    /// each natural capture point (Embedding, AttnNorm, QkvMatmul, AttnOut,
    /// PostAttnResidual, FfnNorm, FfnGate, FfnUp, FfnSilu, FfnSwigl, FfnOut,
    /// PostFfnResidual per-layer, plus FinalNorm + LmHead whole-model).
    ///
    /// Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] step-3
    /// per-layer threading.
    ///
    /// # Errors
    ///
    /// - Propagates [`RealizarError`](crate::error::RealizarError) from the
    ///   underlying forward pass.
    /// - Returns [`RealizarError::IoError`] if any
    ///   [`maybe_save_stage`](crate::inference_trace::save_tensor_emit::maybe_save_stage)
    ///   write fails.
    pub fn forward_traced_with_plan(
        &self,
        token_ids: &[u32],
        plan: Option<&crate::inference_trace::save_tensor_plan::SaveTensorPlan>,
    ) -> Result<ForwardTrace> {
        use crate::inference_trace::save_tensor::WHOLE_MODEL_LAYER;
        use crate::inference_trace::save_tensor_emit::maybe_save_stage;
        use crate::inference_trace::save_tensor_stage::SaveTensorStage;

        // Wraps `maybe_save_stage` IO error → `RealizarError::IoError`.
        let emit = |stage: SaveTensorStage, layer: u32, values: &[f32]| -> Result<()> {
            maybe_save_stage(plan, stage, layer, values).map_err(|e| RealizarError::IoError {
                message: format!("save_tensor::{stage:?} L{layer}: {e}"),
            })
        };

        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);
        emit(SaveTensorStage::Embedding, 0, &hidden)?;
        let embed_stats = ActivationStats::from_slice(&hidden);

        let mut layer_activations = Vec::with_capacity(self.layers.len());

        // 2. Process through transformer layers with tracing
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // M-FFN-GGUF-5 / SHIP-007 §22 fix: use Q4K layers when available
            // to match GGUF's Q4K+Q8K matvec semantics (per M91-M101 cascade
            // + M-FFN-GGUF-7 empirical validation that Option-A is the correct
            // fix path). Falls back to F32 when Q4K bytes are missing.
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );
            emit(SaveTensorStage::AttnNorm, layer_idx as u32, &normed)?;
            let attn_norm_stats = ActivationStats::from_slice(&normed);
            // Last-token-only slice for parity with GGUF (§37 / FALSIFY-APR-GGUF-PARITY-007)
            let seq_len_for_last = token_ids.len();
            let last_token_attn_norm_stats = ActivationStats::from_slice(
                &normed[(seq_len_for_last - 1) * hidden_dim..],
            );

            // 2b. QKV projection
            // M-FFN-GGUF-5b / SHIP-007 §22 closure: split-Q4K QKV when the
            // q4k_layer has separate attn_q/k/v Q4K bytes (matches GGUF
            // Q4K+Q8K matvec semantics like the rest of `forward_traced` and
            // mirrors `project_qkv_fused`). F32 fallback otherwise.
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let head_dim_pre = hidden_dim / self.config.num_heads;
            let kv_size_pre = self.config.num_kv_heads * head_dim_pre;
            let seq_len_pre = token_ids.len();
            let mut qkv = self.qkv_split_q4k_traced(
                &normed,
                q4k_layer,
                &layer.qkv_weight,
                seq_len_pre,
                hidden_dim,
                kv_size_pre,
                qkv_dim,
            );
            emit(SaveTensorStage::QkvMatmul, layer_idx as u32, &qkv)?;
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
                emit(SaveTensorStage::QkvBias, layer_idx as u32, &qkv)?;
            }
            let qkv_stats = ActivationStats::from_slice(&qkv);
            let last_token_qkv_stats =
                ActivationStats::from_slice(&qkv[(seq_len_for_last - 1) * qkv_dim..]);

            // 2c. Attention computation (simplified for trace - same logic as forward)
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let kv_dim = num_kv_heads * head_dim;
            let group_size = self.config.num_heads / num_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q_pos = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k_pos =
                    qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v_pos =
                    &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                // PMAT-799b: Per-head QK RMSNorm (Qwen3) — applied AFTER projection
                // + bias, BEFORE RoPE (Qwen3 ordering). The trace path previously
                // skipped this entirely, while the live cached path
                // (`project_qkv_with_cache`, PMAT-799), the GGUF reference
                // (`gguf/inference/forward/forward_cached.rs`), and the sibling
                // `AprTransformer::forward` (pmat-260.rs, GH-279) all apply it.
                // No-op when the layer lacks the weights (LLaMA/Qwen2/Mistral) →
                // byte-identical for non-Qwen3 models. Contract: qk-norm-v1
                // §QKN-INV-007.
                if let Some(ref q_norm) = layer.attn_q_norm_weight {
                    crate::gguf::ops::apply_per_head_rms_norm(
                        &mut q_pos,
                        q_norm,
                        self.config.num_heads,
                        self.config.eps,
                    );
                }
                if let Some(ref k_norm) = layer.attn_k_norm_weight {
                    crate::gguf::ops::apply_per_head_rms_norm(
                        &mut k_pos,
                        k_norm,
                        num_kv_heads,
                        self.config.eps,
                    );
                }

                self.apply_rope_f32(&mut q_pos, s, self.config.num_heads, head_dim);
                self.apply_rope_f32(&mut k_pos, s, num_kv_heads, head_dim);

                q_all.extend_from_slice(&q_pos);
                k_all.extend_from_slice(&k_pos);
                v_all.extend_from_slice(v_pos);
            }

            // Capture Q/K post-RoPE tensors per `trace-attn-sub-stages-v1.yaml`.
            // Closes pre-existing capture gap in parent contract per
            // `evidence/ship-007-layer0-attn-bisection-2026-05-04/forward-traced-research.md`.
            emit(SaveTensorStage::QPostRope, layer_idx as u32, &q_all)?;
            emit(SaveTensorStage::KPostRope, layer_idx as u32, &k_all)?;

            // Allocate accumulators for attn_scores + attn_softmax ONLY when
            // the plan requests them. Per `trace-attn-sub-stages-v1.yaml`
            // v1.1.0 FALSIFY-ATTN-SUB-005 (additive purity): no allocation
            // unless capture is selected.
            let want_scores = plan
                .is_some_and(|p| p.should_save(SaveTensorStage::AttnScores, layer_idx as u32));
            let want_softmax = plan
                .is_some_and(|p| p.should_save(SaveTensorStage::AttnSoftmax, layer_idx as u32));
            let mut scores_all: Option<Vec<f32>> = if want_scores {
                Some(vec![0.0f32; self.config.num_heads * seq_len * seq_len])
            } else {
                None
            };
            let mut softmax_all: Option<Vec<f32>> = if want_softmax {
                Some(vec![0.0f32; self.config.num_heads * seq_len * seq_len])
            } else {
                None
            };

            // Attention output
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
            for head in 0..self.config.num_heads {
                let kv_head = head / group_size;
                let q_head_offset = head * head_dim;
                let kv_head_offset = kv_head * head_dim;

                for i in 0..seq_len {
                    let mut scores = Vec::with_capacity(i + 1);
                    let q_start = i * hidden_dim + q_head_offset;

                    for j in 0..=i {
                        let k_start = j * kv_dim + kv_head_offset;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_all[q_start + d] * k_all[k_start + d];
                        }
                        scores.push(score * scale);
                    }

                    // Capture pre-softmax scores into [num_heads × seq × seq]
                    // buffer (zero-padded for non-causal positions).
                    if let Some(ref mut buf) = scores_all {
                        let row_base = head * seq_len * seq_len + i * seq_len;
                        for (j, &s) in scores.iter().enumerate() {
                            buf[row_base + j] = s;
                        }
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                    // Capture post-softmax probabilities into the same
                    // [num_heads × seq × seq] shape.
                    if let Some(ref mut buf) = softmax_all {
                        let row_base = head * seq_len * seq_len + i * seq_len;
                        for (j, &p) in probs.iter().enumerate() {
                            buf[row_base + j] = p;
                        }
                    }

                    // Weighted sum of values
                    let out_start = i * hidden_dim + q_head_offset;
                    for (j, &p) in probs.iter().enumerate() {
                        let v_start = j * kv_dim + kv_head_offset;
                        for d in 0..head_dim {
                            attn_out[out_start + d] += p * v_all[v_start + d];
                        }
                    }
                }
            }

            // Emit captured attention sub-stages (no-op if accumulator is None).
            if let Some(buf) = scores_all.as_deref() {
                emit(SaveTensorStage::AttnScores, layer_idx as u32, buf)?;
            }
            if let Some(buf) = softmax_all.as_deref() {
                emit(SaveTensorStage::AttnSoftmax, layer_idx as u32, buf)?;
            }

            // Capture pre-O-proj attention output (Q@Kᵀ@V combined).
            emit(SaveTensorStage::Attention, layer_idx as u32, &attn_out)?;

            // Output projection (M-FFN-GGUF-5: Q4K when available)
            let mut attn_output = self.matmul_q4k_or_f32_traced(
                &attn_out,
                q4k_layer.and_then(|q| q.attn_output_weight.as_deref()),
                None,
                &layer.attn_output_weight,
                hidden_dim,
                hidden_dim,
            );
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }
            emit(SaveTensorStage::AttnOut, layer_idx as u32, &attn_output)?;
            let attn_out_stats = ActivationStats::from_slice(&attn_output);
            let last_token_attn_out_stats = ActivationStats::from_slice(
                &attn_output[(seq_len_for_last - 1) * hidden_dim..],
            );

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }
            emit(SaveTensorStage::PostAttnResidual, layer_idx as u32, &hidden)?;

            // 2f. FFN layer norm (if present)
            let ffn_input = if let Some(ref norm_weight) = layer.ffn_norm_weight {
                let normed = self.layer_norm(
                    &hidden,
                    norm_weight,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                );
                normed
            } else {
                hidden.clone()
            };
            emit(SaveTensorStage::FfnNorm, layer_idx as u32, &ffn_input)?;
            let ffn_norm_stats = ActivationStats::from_slice(&ffn_input);
            let last_token_ffn_norm_stats = ActivationStats::from_slice(
                &ffn_input[(seq_len_for_last - 1) * hidden_dim..],
            );

            // 2g. FFN - check if gated MLP (SwiGLU) by presence of gate weight
            // Per contracts/trace-ffn-sub-block-v1.yaml: capture sub-FFN
            // intermediate stats (gate, up, silu(gate), silu(gate)*up)
            // for SHIP-007 layer-3 bisection (§17.4).
            let mut ffn_gate_stats = ActivationStats::default();
            let mut ffn_up_stats = ActivationStats::default();
            let mut ffn_silu_gate_stats = ActivationStats::default();
            let mut ffn_swiglu_inner_stats = ActivationStats::default();
            let mut last_token_ffn_gate_stats = ActivationStats::default();
            let mut last_token_ffn_up_stats = ActivationStats::default();
            let mut last_token_ffn_silu_gate_stats = ActivationStats::default();
            let mut last_token_ffn_swiglu_inner_stats = ActivationStats::default();

            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // M-FFN-GGUF-5: gate / up projections use Q4K when available
                let gate = self.matmul_q4k_or_f32_traced(
                    &ffn_input,
                    q4k_layer.and_then(|q| q.ffn_gate_weight.as_deref()),
                    None,
                    gate_weight,
                    hidden_dim,
                    intermediate_dim,
                );
                emit(SaveTensorStage::FfnGate, layer_idx as u32, &gate)?;
                let up = self.matmul_q4k_or_f32_traced(
                    &ffn_input,
                    q4k_layer.and_then(|q| q.ffn_up_weight.as_deref()),
                    q4k_layer.and_then(|q| q.ffn_up_weight_q6k.as_deref()),
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                );
                emit(SaveTensorStage::FfnUp, layer_idx as u32, &up)?;

                ffn_gate_stats = ActivationStats::from_slice(&gate);
                ffn_up_stats = ActivationStats::from_slice(&up);
                last_token_ffn_gate_stats = ActivationStats::from_slice(
                    &gate[(seq_len_for_last - 1) * intermediate_dim..],
                );
                last_token_ffn_up_stats = ActivationStats::from_slice(
                    &up[(seq_len_for_last - 1) * intermediate_dim..],
                );

                let mut silu_gate = Vec::with_capacity(gate.len());
                let mut ffn_hidden = Vec::with_capacity(gate.len());
                for (g, u) in gate.iter().zip(up.iter()) {
                    let silu_g = g / (1.0 + (-g).exp());
                    silu_gate.push(silu_g);
                    ffn_hidden.push(silu_g * u);
                }
                emit(SaveTensorStage::FfnSilu, layer_idx as u32, &silu_gate)?;
                emit(SaveTensorStage::FfnSwigl, layer_idx as u32, &ffn_hidden)?;

                ffn_silu_gate_stats = ActivationStats::from_slice(&silu_gate);
                ffn_swiglu_inner_stats = ActivationStats::from_slice(&ffn_hidden);
                last_token_ffn_silu_gate_stats = ActivationStats::from_slice(
                    &silu_gate[(seq_len_for_last - 1) * intermediate_dim..],
                );
                last_token_ffn_swiglu_inner_stats = ActivationStats::from_slice(
                    &ffn_hidden[(seq_len_for_last - 1) * intermediate_dim..],
                );

                // M-FFN-GGUF-5: ffn_down projection uses Q4K when available
                let mut out = self.matmul_q4k_or_f32_traced(
                    &ffn_hidden,
                    q4k_layer.and_then(|q| q.ffn_down_weight.as_deref()),
                    q4k_layer.and_then(|q| q.ffn_down_weight_q6k.as_deref()),
                    &layer.ffn_down_weight,
                    intermediate_dim,
                    hidden_dim,
                );
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            } else {
                // Standard MLP without gating (no SwiGLU sub-stats apply;
                // ffn_gate_stats / ffn_silu_gate_stats / ffn_swiglu_inner_stats
                // remain default-zero; ffn_up_stats reflects pre-GELU values).
                // M-FFN-GGUF-5: ffn_up projection uses Q4K when available
                let mut ffn_hidden = self.matmul_q4k_or_f32_traced(
                    &ffn_input,
                    q4k_layer.and_then(|q| q.ffn_up_weight.as_deref()),
                    q4k_layer.and_then(|q| q.ffn_up_weight_q6k.as_deref()),
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                );
                if let Some(ref bias) = layer.ffn_up_bias {
                    self.add_bias(&mut ffn_hidden, bias);
                }
                ffn_up_stats = ActivationStats::from_slice(&ffn_hidden);
                last_token_ffn_up_stats = ActivationStats::from_slice(
                    &ffn_hidden[(seq_len_for_last - 1) * intermediate_dim..],
                );
                for h in &mut ffn_hidden {
                    let gelu_approx =
                        0.5 * *h * (1.0 + (0.797_884_6 * (*h + 0.044_715 * *h * *h * *h)).tanh());
                    *h = gelu_approx;
                }
                // M-FFN-GGUF-5: ffn_down projection uses Q4K when available
                let mut out = self.matmul_q4k_or_f32_traced(
                    &ffn_hidden,
                    q4k_layer.and_then(|q| q.ffn_down_weight.as_deref()),
                    q4k_layer.and_then(|q| q.ffn_down_weight_q6k.as_deref()),
                    &layer.ffn_down_weight,
                    intermediate_dim,
                    hidden_dim,
                );
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            };
            emit(SaveTensorStage::FfnOut, layer_idx as u32, &ffn_output)?;
            let ffn_out_stats = ActivationStats::from_slice(&ffn_output);
            let last_token_ffn_out_stats = ActivationStats::from_slice(
                &ffn_output[(seq_len_for_last - 1) * hidden_dim..],
            );

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
            emit(SaveTensorStage::PostFfnResidual, layer_idx as u32, &hidden)?;
            let output_stats = ActivationStats::from_slice(&hidden);
            let last_token_output_stats = ActivationStats::from_slice(
                &hidden[(seq_len_for_last - 1) * hidden_dim..],
            );

            // §37 / FALSIFY-APR-GGUF-PARITY-007: emit last-token-only stats for
            // sample-size parity with GGUF's forward_traced (which traces only
            // the last token). When seq_len == 1, last_token == all-tokens.
            let last_token = Some(crate::apr_transformer::LastTokenStats {
                attn_norm_stats: last_token_attn_norm_stats,
                qkv_stats: last_token_qkv_stats,
                attn_out_stats: last_token_attn_out_stats,
                ffn_norm_stats: last_token_ffn_norm_stats,
                ffn_gate_stats: last_token_ffn_gate_stats,
                ffn_up_stats: last_token_ffn_up_stats,
                ffn_silu_gate_stats: last_token_ffn_silu_gate_stats,
                ffn_swiglu_inner_stats: last_token_ffn_swiglu_inner_stats,
                ffn_out_stats: last_token_ffn_out_stats,
                output_stats: last_token_output_stats,
            });

            layer_activations.push(LayerActivation {
                layer_idx,
                attn_norm_stats,
                qkv_stats,
                attn_out_stats,
                ffn_norm_stats,
                ffn_gate_stats,
                ffn_up_stats,
                ffn_silu_gate_stats,
                ffn_swiglu_inner_stats,
                ffn_out_stats,
                output_stats,
                last_token,
            });
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );
        emit(SaveTensorStage::FinalNorm, WHOLE_MODEL_LAYER, &normed)?;
        let final_norm_stats = ActivationStats::from_slice(&normed);

        // 4. LM head projection (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        // M-FFN-GGUF-5: lm_head uses Q4K when available (matches production project_lm_head)
        let mut logits = self.matmul_q4k_or_f32_traced(
            last_hidden,
            self.lm_head_weight_q4k.as_deref(),
            self.lm_head_weight_q6k.as_deref(),
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }
        emit(SaveTensorStage::LmHead, WHOLE_MODEL_LAYER, &logits)?;
        let logits_stats = ActivationStats::from_slice(&logits);

        Ok(ForwardTrace {
            input_tokens: token_ids.to_vec(),
            embed_stats,
            layer_activations,
            final_norm_stats,
            logits_stats,
            logits,
        })
    }

    /// Predict next token (greedy decoding)
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Token ID with highest probability
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;

        // Argmax
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;

        Ok(max_idx as u32)
    }
}
