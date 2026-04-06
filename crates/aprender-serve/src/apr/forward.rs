impl AprV2Model {

    /// Resolve RoPE parameters from metadata with architecture-specific fallback.
    fn resolve_rope_params(&self) -> (f32, u32) {
        let arch = self.metadata.architecture.as_deref().unwrap_or("unknown");
        let rope_theta = self.metadata.rope_theta.unwrap_or_else(||
            crate::gguf::default_rope_theta_for_architecture(arch));
        let rope_type = self.metadata.rope_type.unwrap_or_else(||
            crate::gguf::infer_rope_type(arch));
        (rope_theta, rope_type)
    }

    /// Look up token embeddings for a sequence of token IDs.
    fn embed_tokens(
        &self,
        token_ids: &[u32],
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        let embed_name = self.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight", // SafeTensors (no model. prefix)
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
            "token_embd.weight", // GGUF naming convention
        ])?;

        let embeddings = self.get_tensor_f32(&embed_name)?;
        let mut hidden = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }
        Ok(hidden)
    }

    /// Project hidden state to vocabulary logits via LM head.
    ///
    /// Handles both regular lm_head [vocab_size, hidden_dim] and tied
    /// embeddings [hidden_dim, vocab_size] with transposed access.
    fn project_lm_head(
        last_hidden: &[f32],
        lm_head: &[f32],
        hidden_dim: usize,
        vocab_size: usize,
        is_tied: bool,
    ) -> Vec<f32> {
        let mut logits = vec![0.0; vocab_size];
        if is_tied && lm_head.len() == hidden_dim * vocab_size {
            // Tied embedding: token_embd is [hidden_dim, vocab_size]
            for (i, logit) in logits.iter_mut().enumerate() {
                for (j, &h) in last_hidden.iter().enumerate() {
                    *logit += h * lm_head.get(j * vocab_size + i).copied().unwrap_or(0.0);
                }
            }
        } else {
            // Regular lm_head: [vocab_size, hidden_dim]
            for (i, logit) in logits.iter_mut().enumerate() {
                for (j, &h) in last_hidden.iter().enumerate() {
                    *logit += h * lm_head.get(i * hidden_dim + j).copied().unwrap_or(0.0);
                }
            }
        }
        logits
    }

    /// Apply RoPE to Q and K tensors for all positions in the sequence.
    fn apply_rope_to_qk(
        q: &mut [f32],
        k: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        rope_theta: f32,
        rope_type: u32,
    ) {
        let q_dim_per_token = hidden_dim;
        let k_dim_per_token = num_kv_heads * head_dim;
        for pos in 0..seq_len {
            let q_start = pos * q_dim_per_token;
            let q_end = q_start + q_dim_per_token;
            if q_end <= q.len() {
                apply_rope_norm(&mut q[q_start..q_end], num_heads, head_dim, pos, rope_theta, rope_type);
            }
            let k_start = pos * k_dim_per_token;
            let k_end = k_start + k_dim_per_token;
            if k_end <= k.len() {
                apply_rope_norm(&mut k[k_start..k_end], num_kv_heads, head_dim, pos, rope_theta, rope_type);
            }
        }
    }

    /// Extract model dimensions from metadata.
    fn resolve_model_dims(&self) -> (usize, usize, usize, usize, usize, usize, f32) {
        let hidden_dim = self.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.metadata.num_layers.unwrap_or(0);
        let num_heads = self.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.metadata.num_kv_heads.unwrap_or(num_heads);
        let vocab_size = self.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self.metadata.intermediate_size.unwrap_or(hidden_dim * 4);
        let eps = self.metadata.rms_norm_eps.unwrap_or(1e-6);
        (hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, intermediate_dim, eps)
    }

    /// Run transformer forward pass on token IDs
    ///
    /// Returns logits for the next token prediction.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size`
    ///
    /// # Errors
    ///
    /// Returns error if model is not a transformer or tensors are missing
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        if !self.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let (hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, intermediate_dim, eps) =
            self.resolve_model_dims();

        // 1. Token embedding lookup
        let mut hidden = self.embed_tokens(token_ids, hidden_dim)?;

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Try common naming patterns (HuggingFace, SafeTensors, GPT-2, LLaMA, GGUF)
            let attn_norm_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
                &format!("blk.{layer_idx}.attn_norm.weight"), // GGUF naming
            ])?;

            let q_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                &format!("layers.{layer_idx}.attention.wq.weight"),
                &format!("blk.{layer_idx}.attn_q.weight"), // GGUF naming
            ])?;

            let k_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                &format!("layers.{layer_idx}.attention.wk.weight"),
                &format!("blk.{layer_idx}.attn_k.weight"), // GGUF naming
            ])?;

            let v_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                &format!("layers.{layer_idx}.attention.wv.weight"),
                &format!("blk.{layer_idx}.attn_v.weight"), // GGUF naming
            ])?;

            let o_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
                &format!("blk.{layer_idx}.attn_output.weight"), // GGUF naming
            ])?;

            // Load tensors
            let norm_weight = self.get_tensor_f32(&attn_norm_name)?;
            let q_weight = self.get_tensor_f32(&q_name)?;
            let k_weight = self.get_tensor_f32(&k_name)?;
            let v_weight = self.get_tensor_f32(&v_name)?;
            let o_weight = self.get_tensor_f32(&o_name)?;

            // RMSNorm
            let normed = rms_norm(&hidden, &norm_weight, eps);

            // Attention: Q, K, V projections
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / num_heads;

            let mut q = matmul(&normed, &q_weight, seq_len, hidden_dim, hidden_dim);
            let mut k = matmul(
                &normed,
                &k_weight,
                seq_len,
                hidden_dim,
                num_kv_heads * head_dim,
            );
            let v = matmul(
                &normed,
                &v_weight,
                seq_len,
                hidden_dim,
                num_kv_heads * head_dim,
            );

            // BUG-2 FIX: Apply RoPE (Rotary Position Embedding) to Q and K
            let (rope_theta, rope_type) = self.resolve_rope_params();
            Self::apply_rope_to_qk(
                &mut q, &mut k, seq_len, num_heads, num_kv_heads, head_dim,
                hidden_dim, rope_theta, rope_type,
            );

            let attn_out = simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);

            // Output projection
            let attn_proj = matmul(&attn_out, &o_weight, seq_len, hidden_dim, hidden_dim);

            // Residual connection
            for (h, &a) in hidden.iter_mut().zip(attn_proj.iter()) {
                *h += a;
            }

            // FFN
            let ffn_norm_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("layers.{layer_idx}.post_attention_layernorm.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.ln_2.weight"),
                &format!("layers.{layer_idx}.ffn_norm.weight"),
                &format!("blk.{layer_idx}.ffn_norm.weight"), // GGUF naming
            ])?;

            let gate_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
                &format!("blk.{layer_idx}.ffn_gate.weight"), // GGUF naming
            ])?;

            let up_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
                &format!("blk.{layer_idx}.ffn_up.weight"), // GGUF naming
            ])?;

            let down_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
                &format!("blk.{layer_idx}.ffn_down.weight"), // GGUF naming
            ])?;

            let ffn_norm = self.get_tensor_f32(&ffn_norm_name)?;
            let gate = self.get_tensor_f32(&gate_name)?;
            let up = self.get_tensor_f32(&up_name)?;
            let down = self.get_tensor_f32(&down_name)?;

            let normed = rms_norm(&hidden, &ffn_norm, eps);
            let gate_out = matmul(&normed, &gate, seq_len, hidden_dim, intermediate_dim);
            let up_out = matmul(&normed, &up, seq_len, hidden_dim, intermediate_dim);

            // SiLU activation and element-wise multiply
            let mut ffn_hidden = Vec::with_capacity(seq_len * intermediate_dim);
            for (g, u) in gate_out.iter().zip(up_out.iter()) {
                let silu = g * (1.0 / (1.0 + (-g).exp()));
                ffn_hidden.push(silu * u);
            }

            let ffn_out = matmul(&ffn_hidden, &down, seq_len, intermediate_dim, hidden_dim);

            // Residual
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f;
            }
        }

        // 3. Final layer norm
        let final_norm_name = self.find_tensor_name(&[
            "model.norm.weight",
            "norm.weight", // SafeTensors
            "transformer.ln_f.weight",
            "output_norm.weight", // GGUF naming
        ])?;
        let final_norm = self.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);

        // 4. LM head (last token only for generation)
        // BUG-APR-001: Add token_embd.weight for GGUF weight tying
        let lm_head_name = self.find_tensor_name(&[
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight", // Tied embeddings
            "embed_tokens.weight",       // SafeTensors tied embeddings
            "token_embd.weight",         // GGUF tied embeddings
        ])?;
        let lm_head = self.get_tensor_f32(&lm_head_name)?;
        let last_hidden = &hidden[hidden.len() - hidden_dim..];

        // BUG-APR-001-FIX: Detect weight tying and handle transposed layout
        let is_tied = lm_head_name == "token_embd.weight"
            || lm_head_name.ends_with("embed_tokens.weight");
        let logits = Self::project_lm_head(last_hidden, &lm_head, hidden_dim, vocab_size, is_tied);

        Ok(logits)
    }

    /// Autoregressive text generation.
    ///
    /// Generates tokens one at a time using greedy decoding (argmax sampling).
    ///
    /// # Arguments
    ///
    /// * `input_tokens` - Initial token sequence (prompt)
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `eos_token_id` - End-of-sequence token ID (stops generation early)
    ///
    /// # Returns
    ///
    /// Complete token sequence including input and generated tokens
    ///
    /// # Errors
    ///
    /// Returns error if model is not a transformer or forward pass fails
    pub fn generate(
        &self,
        input_tokens: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        if input_tokens.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tokens cannot be empty".to_string(),
            });
        }

        let mut tokens = input_tokens.to_vec();
        let vocab_size = self.metadata.vocab_size.unwrap_or(0);

        for _ in 0..max_new_tokens {
            // Forward pass to get logits for next token
            let logits = self.forward(&tokens)?;

            // Greedy sampling: pick token with highest logit
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32);

            // Check for EOS
            if let Some(eos) = eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            // Sanity check: don't append invalid tokens
            if (next_token as usize) >= vocab_size && vocab_size > 0 {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Find first matching tensor name from candidates.
    /// Handles GGUF/SafeTensors/HuggingFace naming variations.
    pub fn find_tensor_name(&self, candidates: &[&str]) -> Result<String> {
        for &name in candidates {
            if self.get_tensor(name).is_some() {
                return Ok(name.to_string());
            }
        }
        Err(RealizarError::FormatError {
            reason: format!("No matching tensor found. Tried: {:?}", candidates),
        })
    }

    /// Load tokenizer from sibling tokenizer.json file
    ///
    /// GAP-UX-002: Tries hash-prefixed companion first (`{stem}.tokenizer.json`),
    /// then falls back to non-prefixed (`tokenizer.json`) for backwards compatibility.
    ///
    /// Looks for tokenizer.json in the same directory as the model file.
    /// Returns (vocab, bos_token_id, eos_token_id) if found.
    pub fn load_tokenizer_from_sibling(
        model_path: &Path,
    ) -> Option<(Vec<String>, Option<u32>, Option<u32>)> {
        let tokenizer_path = find_sibling_file(model_path, "tokenizer.json")?;

        let content = fs::read_to_string(&tokenizer_path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Extract vocabulary from model.vocab
        let vocab_obj = json.get("model")?.get("vocab")?;
        let vocab_map = vocab_obj.as_object()?;

        // Build vocab vector (sorted by ID)
        let mut vocab_vec: Vec<(String, u32)> = vocab_map
            .iter()
            .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
            .collect();
        vocab_vec.sort_by_key(|(_, id)| *id);

        // Extract special tokens AND merge added_tokens into vocab
        let mut bos_id = None;
        let mut eos_id = None;

        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let content = token.get("content").and_then(|v| v.as_str());
                let id = token
                    .get("id")
                    .and_then(serde_json::Value::as_u64)
                    .map(|v| v as u32);

                if let (Some(content), Some(id)) = (content, id) {
                    if content == "<|endoftext|>" || content == "</s>" || content == "<eos>" {
                        eos_id = Some(id);
                    }
                    if content == "<s>" || content == "<bos>" {
                        bos_id = Some(id);
                    }
                    // ALB-109: Merge added_tokens into vocab so BPETokenizer can decode them.
                    // Without this, special tokens (EOS, im_start, im_end) cause decode errors.
                    vocab_vec.push((content.to_string(), id));
                }
            }
        }

        // Sort by ID after merging added_tokens
        vocab_vec.sort_by_key(|(_, id)| *id);
        let vocab: Vec<String> = vocab_vec.into_iter().map(|(token, _)| token).collect();

        Some((vocab, bos_id, eos_id))
    }

    /// Decode token IDs to text using vocabulary
    ///
    /// If vocab is not available, returns formatted token IDs.
    pub fn decode_tokens(vocab: &[String], token_ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            if let Some(token) = vocab.get(id as usize) {
                // Handle byte-level BPE encoding (Ġ = space prefix)
                let decoded = token
                    .replace("Ġ", " ")
                    .replace("Ċ", "\n")
                    .replace("ĉ", "\t");
                result.push_str(&decoded);
            } else {
                result.push_str(&format!("[{}]", id));
            }
        }
        result
    }
}
