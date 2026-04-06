impl GpuModel {

    /// IMP-1008: Forward single block without weight cloning
    ///
    /// Uses interior mutability pattern to avoid cloning weights on each matmul.
    /// This method takes `&self` instead of `&mut self`.
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails.
    #[cfg(feature = "cuda")]
    pub fn forward_block_refcell(
        &self,
        input: &[f32],
        block_idx: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Phase 21 Debug: trace first forward call only
        static DEBUG_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let call_count = DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let debug_this_call = block_idx == 0 && call_count == 0; // Only first call to block 0

        // Extract config values (Copy types, no borrow conflict)
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim();
        let kv_dim = self.config.kv_dim();
        let qkv_dim = self.config.qkv_dim();
        let intermediate_dim = self.config.intermediate_dim;
        let eps = self.config.eps;
        let num_kv_heads = self.config.num_kv_heads;

        if debug_this_call {
            eprintln!(
                "[PHASE21] forward_block_refcell START block_idx={}",
                block_idx
            );
            eprintln!(
                "[PHASE21] input L2: {:.4}",
                input.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
        }

        // IMP-1008: No cloning! Direct reference to weights
        // Pre-attention layer norm (static function avoids &self borrow)
        let normed = Self::layer_norm_static(
            input,
            &self.block_weights[block_idx].attn_norm_weight,
            &self.block_weights[block_idx].attn_norm_bias,
            hidden_dim,
            eps,
        );

        // QKV projection - NO CLONE!
        let mut qkv = self.matmul_refcell(
            &normed,
            &self.block_weights[block_idx].qkv_weight,
            1,
            hidden_dim,
            qkv_dim,
        )?;

        // F-REGR-231 FIX: Add QKV bias (critical for correct attention)
        // The GGUF path applies bias after matmul, APR must do the same
        let qkv_bias = &self.block_weights[block_idx].qkv_bias;
        if debug_this_call {
            eprintln!(
                "[PHASE21-BIAS] qkv_bias len: {}, qkv len: {}, bias first 5: {:?}",
                qkv_bias.len(),
                qkv.len(),
                &qkv_bias[..5.min(qkv_bias.len())]
            );
        }
        if !qkv_bias.is_empty() && qkv_bias.len() == qkv.len() {
            for (q, b) in qkv.iter_mut().zip(qkv_bias.iter()) {
                *q += *b;
            }
        }

        if debug_this_call {
            eprintln!(
                "[PHASE21] QKV L2: {:.4}",
                qkv.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
            // F-REGR-231 DEBUG: Show Q values after bias
            eprintln!(
                "[PHASE21] Q after bias first 5: {:?}",
                &qkv[..5.min(qkv.len())]
            );
        }

        // Get current position BEFORE caching (Phase 21)
        let (cached_k_ref, _) = kv_cache.get_valid(block_idx);
        let current_pos = cached_k_ref.len() / kv_dim;

        // Phase 21: Apply RoPE to Q and K BEFORE caching
        // Without RoPE, attention has no position information and produces garbage
        let rope_theta = self.config.rope_theta;
        Self::apply_rope_inline(
            &mut qkv[0..hidden_dim],
            num_heads,
            head_dim,
            rope_theta,
            current_pos,
        );
        Self::apply_rope_inline(
            &mut qkv[hidden_dim..hidden_dim + kv_dim],
            num_kv_heads,
            head_dim,
            rope_theta,
            current_pos,
        );

        // Split QKV (GQA: K/V have kv_dim, not hidden_dim) - after RoPE
        let q = qkv[0..hidden_dim].to_vec();
        let k_new = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
        let v_new = qkv[hidden_dim + kv_dim..].to_vec();

        // F-REGR-231 DEBUG: Show K and V values after bias and RoPE
        if debug_this_call {
            eprintln!(
                "[PHASE21] K after RoPE first 5: {:?}",
                &k_new[..5.min(k_new.len())]
            );
            eprintln!("[PHASE21] V first 5: {:?}", &v_new[..5.min(v_new.len())]);
        }

        // Get cached K/V and clone to avoid borrow issues with kv_cache
        let (cached_k, cached_v) = kv_cache.get_valid(block_idx);
        let keys_cached = cached_k.to_vec();
        let vals_cached = cached_v.to_vec();

        // Append new K/V (with RoPE applied) to cache
        kv_cache.append(block_idx, &k_new, &v_new);

        // Build full K/V (cached + new)
        let kv_len = keys_cached.len() / kv_dim + 1;
        let mut full_k = keys_cached;
        full_k.extend_from_slice(&k_new);
        let mut full_v = vals_cached;
        full_v.extend_from_slice(&v_new);

        // GQA attention (IMP-089): static method to avoid borrow conflicts
        let attn_output = Self::gqa_multihead_attention(
            &q,
            &full_k,
            &full_v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        if debug_this_call {
            eprintln!(
                "[PHASE21] attn_output L2: {:.4}",
                attn_output.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
        }

        // Output projection - NO CLONE!
        let attn_proj = self.matmul_refcell(
            &attn_output,
            &self.block_weights[block_idx].out_weight,
            1,
            hidden_dim,
            hidden_dim,
        )?;

        // Add residual and bias
        let out_bias = &self.block_weights[block_idx].out_bias;
        let post_attn: Vec<f32> = input
            .iter()
            .zip(attn_proj.iter())
            .zip(out_bias.iter())
            .map(|((&i, &a), &b)| i + a + b)
            .collect();

        // FFN with layer norm (static function)
        let ffn_normed = Self::layer_norm_static(
            &post_attn,
            &self.block_weights[block_idx].ffn_norm_weight,
            &self.block_weights[block_idx].ffn_norm_bias,
            hidden_dim,
            eps,
        );

        // FFN: MoE dispatch when experts present, SwiGLU/GELU otherwise
        let output: Vec<f32> = if let Some(ref moe) = self.block_weights[block_idx].moe_experts {
            // ALB-010: MoE forward — route to top-k experts + shared expert
            let moe_out = super::moe_dispatch::moe_forward_token(&ffn_normed, moe, hidden_dim);
            // Residual connection (MoE has no separate bias)
            post_attn
                .iter()
                .zip(moe_out.iter())
                .map(|(&h, &m)| h + m)
                .collect()
        } else {
            // Dense FFN: SwiGLU when gate weight exists, otherwise GELU
            let fc1_activated: Vec<f32> = if let Some(ref gate_weight) =
                self.block_weights[block_idx].ffn_gate_weight
            {
                // SwiGLU: silu(gate(x)) * up(x)
                let up_out = self.matmul_refcell(
                    &ffn_normed,
                    &self.block_weights[block_idx].ffn_fc1_weight,
                    1,
                    hidden_dim,
                    intermediate_dim,
                )?;
                let gate_out =
                    self.matmul_refcell(&ffn_normed, gate_weight, 1, hidden_dim, intermediate_dim)?;
                up_out
                    .iter()
                    .zip(gate_out.iter())
                    .map(|(&u, &g)| {
                        let silu_g = g / (1.0 + (-g).exp());
                        silu_g * u
                    })
                    .collect()
            } else {
                // Standard GELU FFN
                let fc1_out = self.matmul_refcell(
                    &ffn_normed,
                    &self.block_weights[block_idx].ffn_fc1_weight,
                    1,
                    hidden_dim,
                    intermediate_dim,
                )?;
                let ffn_fc1_bias = &self.block_weights[block_idx].ffn_fc1_bias;
                fc1_out
                    .iter()
                    .zip(ffn_fc1_bias.iter())
                    .map(|(&x, &b)| {
                        let x_b = x + b;
                        x_b * 0.5 + x_b * 0.5 * (0.797_884_6 * (x_b + 0.044_715 * x_b.powi(3))).tanh()
                    })
                    .collect()
            };

            // FFN FC2 (down projection) - NO CLONE!
            let fc2_out = self.matmul_refcell(
                &fc1_activated,
                &self.block_weights[block_idx].ffn_fc2_weight,
                1,
                intermediate_dim,
                hidden_dim,
            )?;

            // Add bias and residual
            let ffn_fc2_bias = &self.block_weights[block_idx].ffn_fc2_bias;
            post_attn
                .iter()
                .zip(fc2_out.iter())
                .zip(ffn_fc2_bias.iter())
                .map(|((&h, &f), &b)| h + f + b)
                .collect()
        };

        if debug_this_call {
            eprintln!(
                "[PHASE21] block output L2: {:.4}",
                output.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
        }

        Ok(output)
    }

    /// IMP-1008: Full incremental forward pass without weight cloning
    ///
    /// Uses interior mutability pattern throughout for zero-clone operation.
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails.
    #[cfg(feature = "cuda")]
    pub fn forward_refcell(
        &self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Phase 21: Debug first call only
        static FWD_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let fwd_count = FWD_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let debug_this_fwd = fwd_count == 0;

        if token_id >= self.config.vocab_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Token ID {} out of bounds (vocab_size={})",
                    token_id, self.config.vocab_size
                ),
            });
        }

        let hidden_dim = self.config.hidden_dim;

        // Embed single token
        let offset = token_id * hidden_dim;
        let mut hidden = self.embedding_weights[offset..offset + hidden_dim].to_vec();

        // Process through all blocks - NO CLONE!
        for block_idx in 0..self.config.num_layers {
            hidden = self.forward_block_refcell(&hidden, block_idx, kv_cache)?;
        }

        // Final layer norm
        hidden = self.layer_norm_refcell(&hidden, &self.final_norm_weight, &self.final_norm_bias);

        // LM head projection
        let lm_head_elements = hidden_dim * self.config.vocab_size;
        let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
            // CPU path with transposed weights + SIMD + fused bias
            cpu_matmul_transposed_simd(
                &hidden,
                &self.lm_head_weight_t,
                &self.lm_head_bias,
                hidden_dim,
                self.config.vocab_size,
            )
        } else {
            // GPU path - NO CLONE!
            let vocab_size = self.config.vocab_size;
            let logits =
                self.matmul_refcell(&hidden, &self.lm_head_weight, 1, hidden_dim, vocab_size)?;
            // Add bias
            logits
                .into_iter()
                .zip(self.lm_head_bias.iter())
                .map(|(l, &b)| l + b)
                .collect()
        };

        if debug_this_fwd {
            // Find argmax
            let (argmax_idx, argmax_val) = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            eprintln!(
                "[PHASE21] forward_refcell: final hidden L2: {:.4}",
                hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
            eprintln!(
                "[PHASE21] forward_refcell: logits argmax: {} (val: {:.4})",
                argmax_idx, argmax_val
            );
        }

        Ok(output)
    }

    /// IMP-1008: Layer norm with RefCell pattern (takes &self)
    #[cfg(feature = "cuda")]
    fn layer_norm_refcell(&self, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        Self::layer_norm_static(input, weight, bias, self.config.hidden_dim, self.config.eps)
    }

    /// IMP-1008: Generate tokens without weight cloning
    ///
    /// Uses interior mutability pattern for zero-clone inference.
    ///
    /// # Errors
    ///
    /// Returns error if generation fails.
    #[cfg(feature = "cuda")]
    pub fn generate_refcell(
        &self,
        prompt: &[usize],
        config: &GpuGenerateConfig,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let max_seq_len = prompt.len() + config.max_tokens;

        // Initialize KV cache
        let mut kv_cache =
            StreamingKVCache::new(self.config.num_layers, max_seq_len, num_kv_heads, head_dim);

        let mut tokens = prompt.to_vec();

        // F-REGR-231 FIX: Process prefill correctly
        // Process all but last prompt token to populate KV cache (discard logits)
        // Then process last token to get logits for first generation
        let prompt_len = prompt.len();
        for &token_id in &prompt[..prompt_len.saturating_sub(1)] {
            let _ = self.forward_refcell(token_id, &mut kv_cache)?;
        }

        // Process last prompt token to get logits for first generated token
        let last_prompt_token = prompt[prompt_len - 1];
        let mut current_logits = self.forward_refcell(last_prompt_token, &mut kv_cache)?;

        // F-REGR-231 DEBUG: Show logits from last prompt token
        let argmax = current_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);
        let argmax_val = current_logits.get(argmax).copied().unwrap_or(0.0);
        eprintln!(
            "[PHASE21-GEN] Last prompt token: {}, logits argmax: {} (val: {:.4}), top5 logits: {:?}",
            last_prompt_token,
            argmax,
            argmax_val,
            {
                let mut indexed: Vec<(usize, f32)> = current_logits.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.into_iter().take(5).collect::<Vec<_>>()
            }
        );

        // Generate new tokens
        for _ in 0..config.max_tokens {
            // Sample next token (greedy when temperature=0, otherwise top-k)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                // Greedy decoding
                current_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx)
            } else {
                // Top-k sampling with temperature
                Self::sample_topk_generate(&current_logits, config.temperature, config.top_k)
            };

            tokens.push(next_token);

            // Check for stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            // F-REGR-231: Get logits for next iteration by processing the new token
            current_logits = self.forward_refcell(next_token, &mut kv_cache)?;
        }

        Ok(tokens)
    }
}
