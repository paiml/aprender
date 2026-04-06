
impl OwnedQuantizedModel {

    /// Forward pass with KV cache for efficient autoregressive decoding
    ///
    /// This method properly handles both architectures:
    /// - LLaMA-style: RMSNorm, SwiGLU FFN, GQA attention
    /// - phi-2 style: LayerNorm, GELU FFN, MHA attention
    ///
    /// Uses O(n) per-token cost instead of O(n²) by caching K/V.
    pub fn forward_cached(
        &self,
        token_id: u32,
        cache: &mut crate::gguf::OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // GH-278: Use contract-derived norm type instead of runtime heuristics.
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();

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

        // Cache debug flags (avoid repeated env var lookups)
        let cpu_debug = std::env::var("CPU_DEBUG").is_ok();
        let debug_forward = std::env::var("REALIZAR_DEBUG_FORWARD").is_ok();

        // GQA-DEBUG: Print embedding for CPU/GPU comparison
        if cpu_debug && position == 4 {
            let embed_sum: f32 = hidden.iter().sum();
            let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden.len() as f32).sqrt();
            eprintln!(
                "[GQA-DEBUG-CPU-EMBED] Token {} at pos {}: first 5 = {:?}, sum={:.4}, rms={:.4}",
                token_id, position, &hidden[..5.min(hidden.len())], embed_sum, rms
            );
        }

        // PAR-052: Debug output for OwnedQuantizedModel forward path
        if debug_forward && position == 0 {
            eprintln!("[PAR-052] OwnedQuantizedModel::forward_cached");
            eprintln!("[PAR-052] Token ID: {}, Position: {}", token_id, position);
            eprintln!("[PAR-052] use_rmsnorm: {}", use_rmsnorm);
            eprintln!("[PAR-052] Embedding[0..8]: {:?}", &hidden[..8.min(hidden.len())]);
            eprintln!("[PAR-052] Embedding sum: {:.6}", hidden.iter().sum::<f32>());
        }

        // 2. Process through transformer layers
        for (layer_idx, _layer) in self.layers.iter().enumerate() {
            // F-REGR-231: Debug hidden state at start of each layer
            if cpu_debug && layer_idx == 1 && position == 0 {
                eprintln!(
                    "[GQA-DEBUG-CPU-L1] Hidden at layer start: first 5 = {:?}",
                    &hidden[..5.min(hidden.len())]
                );
            }

            // Attention block (norm → QKV → RoPE → attention → output projection)
            let attn_output = self.cached_attention_block(
                &hidden, layer_idx, cache, position, use_rmsnorm,
                debug_forward, cpu_debug,
            )?;
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // FFN block (norm → SwiGLU/GELU → down projection)
            let ffn_output = self.cached_ffn_block(
                &hidden, layer_idx, use_rmsnorm,
            )?;
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }

            // GQA-DEBUG: Print hidden state after layer 0 for CPU/GPU comparison
            if cpu_debug && layer_idx == 0 && (position == 0 || position == 4) {
                let hidden_sum: f32 = hidden.iter().sum();
                let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden.len() as f32).sqrt();
                eprintln!(
                    "[GQA-DEBUG-CPU-L0] After layer 0 (pos={}): first 5 = {:?}, sum={:.4}, rms={:.4}",
                    position, &hidden[..5.min(hidden.len())], hidden_sum, rms
                );
            }
        }

        // Advance cache position
        cache.advance();

        // 3. Final layer norm + 4. LM head
        self.cached_final_output(
            &hidden, position, use_rmsnorm, debug_forward, cpu_debug,
        )
    }

    /// Attention block for cached forward pass: norm → QKV → RoPE → attention → output proj
    #[allow(clippy::too_many_arguments)]
    fn cached_attention_block(
        &self,
        hidden: &[f32],
        layer_idx: usize,
        cache: &mut crate::gguf::OwnedQuantizedKVCache,
        position: usize,
        use_rmsnorm: bool,
        debug_forward: bool,
        cpu_debug: bool,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let layer = &self.layers[layer_idx];

        // Attention layer norm (RMSNorm for LLaMA, LayerNorm for phi-2)
        let normed = if use_rmsnorm {
            ops::rms_norm(hidden, &layer.attn_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                hidden, &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(), self.config.eps,
            )
        };

        // QKV projection (GQA-aware dimension computation)
        let _qkv_dim = layer.qkv_weight.out_dim();
        let q_dim = layer.qkv_weight.q_dim_for_config(
            self.config.num_heads, self.config.num_kv_heads, hidden_dim, self.config.head_dim(),
        );
        let k_dim = layer.qkv_weight.k_dim_for_config(
            self.config.num_heads, self.config.num_kv_heads, hidden_dim, self.config.head_dim(),
        );
        let v_dim = layer.qkv_weight.v_dim_for_config(
            self.config.num_heads, self.config.num_kv_heads, hidden_dim, self.config.head_dim(),
        );

        // PAR-052 + GQA debug: normed input
        if (debug_forward || cpu_debug) && layer_idx == 0 && position == 0 {
            if debug_forward {
                eprintln!("[PAR-052-L0] attn_norm[0..8]: {:?}", &layer.attn_norm_weight[..8.min(layer.attn_norm_weight.len())]);
                eprintln!("[PAR-052-L0] normed[0..8]: {:?}", &normed[..8.min(normed.len())]);
                eprintln!("[PAR-052-L0] normed sum: {:.6}", normed.iter().sum::<f32>());
            }
            if cpu_debug {
                eprintln!("[GQA-DEBUG-CPU-L0] Normed input (after attn_norm): first 5 = {:?}", &normed[..5.min(normed.len())]);
            }
        }
        if cpu_debug && layer_idx == 1 && position == 0 {
            eprintln!("[GQA-DEBUG-CPU-L1] Normed input (after attn_norm): first 5 = {:?}", &normed[..5.min(normed.len())]);
        }

        let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
        if let Some(ref bias) = layer.qkv_bias {
            ops::add_bias(&mut qkv, bias);
        }

        if debug_forward && layer_idx == 0 && position == 0 {
            eprintln!("[PAR-052-L0] QKV dims: q={}, k={}, v={}, total={}", q_dim, k_dim, v_dim, qkv.len());
            eprintln!("[PAR-052-L0] QKV sum: {:.6}", qkv.iter().sum::<f32>());
            eprintln!("[PAR-052-L0] Q[0..8]: {:?}", &qkv[..8.min(q_dim)]);
        }

        // Extract Q, K, V and apply RoPE
        let mut q = qkv[0..q_dim].to_vec();
        let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
        let v = qkv[q_dim + k_dim..q_dim + k_dim + v_dim].to_vec();

        // GQA-DEBUG: K before/after RoPE
        if cpu_debug && (layer_idx == 0 || layer_idx == 1) && position == 0 {
            eprintln!(
                "[GQA-DEBUG-CPU-L{}] K weight info: qkv_type={:?}, q_dim={}, k_dim={}, v_dim={}",
                layer_idx,
                match &layer.qkv_weight {
                    crate::gguf::OwnedQKVWeights::Fused(_) => "Fused",
                    crate::gguf::OwnedQKVWeights::Separate { .. } => "Separate",
                },
                q_dim, k_dim, v_dim
            );
            eprintln!("[GQA-DEBUG-CPU-L{}] K after bias (before RoPE): first 5 = {:?}", layer_idx, &k[..5.min(k.len())]);
        }

        // Apply RoPE with correct head counts for GQA
        // GH-278: Skip RoPE for models with learned position embeddings (GPT-2)
        if self.config.constraints.uses_rope() {
            self.apply_rope(&mut q, position, self.config.num_heads);
            self.apply_rope(&mut k, position, self.config.num_kv_heads);
        }

        // GQA-DEBUG: K after RoPE
        if cpu_debug && (layer_idx == 0 || layer_idx == 1) && position == 0 {
            eprintln!("[GQA-DEBUG-CPU-L{}] K after RoPE: first 5 = {:?}", layer_idx, &k[..5.min(k.len())]);
        }
        if debug_forward && layer_idx == 0 && position == 0 {
            eprintln!("[PAR-052-L0] Q after RoPE[0..8]: {:?}", &q[..8.min(q.len())]);
            eprintln!("[PAR-052-L0] K after RoPE[0..4]: {:?}", &k[..4.min(k.len())]);
        }

        // Compute attention using cached K/V
        let k_cache = cache.get_k(layer_idx);
        let v_cache = cache.get_v(layer_idx);

        let attn_out = if k_cache.is_empty() {
            // First token - just use V directly (self-attention with single token)
            if self.config.num_kv_heads < self.config.num_heads {
                let head_dim = hidden_dim / self.config.num_heads;
                let group_size = self.config.num_heads / self.config.num_kv_heads;
                (0..self.config.num_heads)
                    .flat_map(|h| {
                        let kv_head = h / group_size;
                        let start = kv_head * head_dim;
                        v[start..start + head_dim].iter().copied()
                    })
                    .collect()
            } else {
                v.clone()
            }
        } else {
            self.attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
        };

        // Store K and V in cache (store original size, not expanded)
        cache.append(layer_idx, &k, &v);

        // Attention output projection
        let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
        if let Some(ref bias) = layer.attn_output_bias {
            ops::add_bias(&mut attn_output, bias);
        }

        Ok(attn_output)
    }

    /// FFN block for cached forward pass: norm → SwiGLU/GELU → down projection
    fn cached_ffn_block(
        &self,
        hidden: &[f32],
        layer_idx: usize,
        use_rmsnorm: bool,
    ) -> Result<Vec<f32>> {
        let layer = &self.layers[layer_idx];

        // Pre-FFN layer norm (LLaMA has separate ffn_norm)
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

        // FFN with SwiGLU or GELU (contract-driven, GH-278)
        if self.config.constraints.has_gate_ffn() {
            // SwiGLU path (LLaMA, Qwen2, Mistral, etc.)
            let (mut ffn_gate, mut ffn_up) = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // Separate gate/up weights (LLaMA, Qwen2, Mistral)
                let ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                let ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                (ffn_gate, ffn_up)
            } else {
                // GH-306: Fused gate_up weight (Phi-3.5) — single matmul, split output
                let fused = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                let half = fused.len() / 2;
                let ffn_gate = fused[..half].to_vec();
                let ffn_up = fused[half..].to_vec();
                (ffn_gate, ffn_up)
            };
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

            let mut output = self.fused_matmul(&ffn_gate, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut output, bias);
            }
            Ok(output)
        } else {
            // GELU path (GPT-2, BERT, etc.)
            let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(&mut ffn_hidden, bias);
            }
            ops::gelu(&mut ffn_hidden);

            let mut output = self.fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut output, bias);
            }
            Ok(output)
        }
    }

    /// Final output for cached forward pass: final norm → LM head → debug
    #[allow(clippy::too_many_arguments)]
    fn cached_final_output(
        &self,
        hidden: &[f32],
        position: usize,
        use_rmsnorm: bool,
        debug_forward: bool,
        cpu_debug: bool,
    ) -> Result<Vec<f32>> {
        // Final layer norm
        let normed = if use_rmsnorm {
            ops::rms_norm(hidden, &self.output_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                hidden, &self.output_norm_weight,
                self.output_norm_bias.as_deref(), self.config.eps,
            )
        };

        // PAR-052: Debug final hidden state
        if debug_forward && position == 0 {
            eprintln!("[PAR-052] Final hidden sum: {:.6}", hidden.iter().sum::<f32>());
            eprintln!("[PAR-052] Final hidden[0..8]: {:?}", &hidden[..8.min(hidden.len())]);
            eprintln!("[PAR-052] After output_norm sum: {:.6}", normed.iter().sum::<f32>());
            eprintln!("[PAR-052] output_norm_weight[0..4]: {:?}", &self.output_norm_weight[..4.min(self.output_norm_weight.len())]);
            eprintln!("[PAR-052] LM head weight dims: in={}, out={}", self.lm_head_weight.in_dim, self.lm_head_weight.out_dim);
        }

        // LM head projection
        let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        // PAR-052: Debug final logits
        if debug_forward && position == 0 {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[PAR-052] Top-5 logits:");
            for (idx, val) in indexed.iter().take(5) {
                eprintln!("  Token {}: {:.6}", idx, val);
            }
            eprintln!("[PAR-052] Logits sum: {:.6}", logits.iter().sum::<f32>());
        }

        // GQA-DEBUG: CPU logits comparison with GPU
        if cpu_debug {
            let sum: f32 = normed.iter().sum();
            let sq_sum: f32 = normed.iter().map(|x| x * x).sum();
            let rms = (sq_sum / normed.len() as f32).sqrt();
            eprintln!(
                "[GQA-DEBUG-CPU] Normed hidden: first 5 = {:?}, sum={:.4}, rms={:.4}",
                &normed[..5.min(normed.len())], sum, rms
            );

            let (argmax_idx, argmax_val) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i, *v))
                .expect("empty logits");
            eprintln!("[GQA-DEBUG-CPU] Logits argmax: idx={}, val={:.4}", argmax_idx, argmax_val);
            eprintln!("[GQA-DEBUG-CPU] Logits[0..20]: {:?}", &logits[..20.min(logits.len())]);
        }

        Ok(logits)
    }
}

include!("forward_fused_q4k.rs");
