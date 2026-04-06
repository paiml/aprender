impl AprTransformer {

    /// Parse APR metadata JSON into an `AprTransformerConfig`.
    ///
    /// Reads 13+ metadata fields with fallback aliases matching aprender's naming conventions.
    /// Returns a config struct with all architecture parameters.
    fn parse_apr_metadata_config(metadata: &serde_json::Value) -> AprTransformerConfig {
        // ALB-106: entrenar checkpoints store model config as a nested JSON string
        // in "model_config". If top-level fields are null/missing, parse and use it.
        let model_config_parsed: Option<serde_json::Value> = metadata
            .get("model_config")
            .and_then(serde_json::Value::as_str)
            .and_then(|s| serde_json::from_str(s).ok());
        let fallback = |key: &str| -> Option<&serde_json::Value> {
            model_config_parsed.as_ref().and_then(|mc| mc.get(key))
        };

        let hidden_dim = metadata
            .get("hidden_size")
            .or_else(|| metadata.get("hidden_dim"))
            .filter(|v| !v.is_null())
            .or_else(|| fallback("hidden_size"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(64) as usize;

        let num_layers = metadata
            .get("num_hidden_layers")
            .or_else(|| metadata.get("num_layers"))
            .filter(|v| !v.is_null())
            .or_else(|| fallback("num_hidden_layers"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as usize;

        let num_heads = metadata
            .get("num_attention_heads")
            .or_else(|| metadata.get("num_heads"))
            .filter(|v| !v.is_null())
            .or_else(|| fallback("num_attention_heads"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(4) as usize;

        let num_kv_heads = metadata
            .get("num_key_value_heads")
            .or_else(|| metadata.get("num_kv_heads"))
            .filter(|v| !v.is_null())
            .or_else(|| fallback("num_key_value_heads").or_else(|| fallback("num_kv_heads")))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(num_heads as u64) as usize;

        // C-12 (Meyer DbC): 0 = unknown, inferred from embedding tensor dims downstream.
        let vocab_size = metadata
            .get("vocab_size")
            .filter(|v| !v.is_null())
            .or_else(|| fallback("vocab_size"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;

        let intermediate_dim = metadata
            .get("intermediate_size")
            .or_else(|| metadata.get("intermediate_dim"))
            .filter(|v| !v.is_null())
            .or_else(|| fallback("intermediate_size"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or((hidden_dim * 4) as u64) as usize;

        // PMAT-125: Extract architecture from metadata (was missing, defaulted to "unknown")
        // Check "architecture" first (APR v2 standard), then "model_type" (fallback)
        // Moved before rope_theta so architecture-specific default can be used (R-02).
        let architecture = metadata
            .get("architecture")
            .or_else(|| metadata.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_lowercase)
            .filter(|s| s != "auto" && !s.is_empty())
            .or_else(|| {
                fallback("hf_architecture")
                    .or_else(|| fallback("architecture"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_lowercase)
                    .filter(|s| s != "auto" && !s.is_empty())
            })
            .unwrap_or_else(|| "unknown".to_string());

        // R-02 (Meyer DbC): rope_theta from metadata, or architecture-specific default.
        let rope_theta = metadata
            .get("rope_theta")
            .filter(|v| !v.is_null())
            .or_else(|| fallback("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(||
                crate::gguf::default_rope_theta_for_architecture(&architecture) as f64) as f32;

        let rms_norm_eps = metadata
            .get("rms_norm_eps")
            .filter(|v| !v.is_null())
            .or_else(|| fallback("rms_norm_eps"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;

        // PMAT-238 FIX: APR files use "context_length" (not "max_position_embeddings").
        // Check all aliases matching aprender's AprMetadata serde aliases.
        let max_position = metadata
            .get("max_position_embeddings")
            .or_else(|| metadata.get("context_length"))
            .or_else(|| metadata.get("max_seq_len"))
            .or_else(|| metadata.get("n_ctx"))
            .filter(|v| !v.is_null())
            .or_else(|| fallback("max_position_embeddings"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;

        // ALB-094: MoE metadata fields
        let head_dim = metadata
            .get("head_dim")
            .filter(|v| !v.is_null())
            .or_else(|| fallback("head_dim_override"))
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize);

        let num_experts = metadata
            .get("num_experts")
            .filter(|v| !v.is_null())
            .or_else(|| fallback("num_experts"))
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize);

        let num_experts_per_tok = metadata
            .get("num_experts_per_tok")
            .filter(|v| !v.is_null())
            .or_else(|| fallback("num_experts_per_tok"))
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize);

        let expert_intermediate_size = metadata
            .get("moe_intermediate_size")
            .filter(|v| !v.is_null())
            .or_else(|| fallback("moe_intermediate_size"))
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize);

        AprTransformerConfig {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: max_position,
            rope_theta,
            eps: rms_norm_eps,
            eos_token_id: None, // Not yet in APR metadata; defaults to tokenizer config
            explicit_head_dim: head_dim,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            ..Default::default()
        }
    }

    /// Parse the APR v2 tensor index into a `BTreeMap`.
    ///
    /// Reads variable-length tensor entries: name, dtype, ndim, dims, offset, size.
    /// Returns map of tensor name -> (absolute_offset, size, dims, dtype).
    fn parse_apr_tensor_index(
        data: &[u8],
        tensor_index_offset: usize,
        data_offset: usize,
        tensor_count: usize,
    ) -> std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)> {
        let mut tensors: std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)> =
            std::collections::BTreeMap::new();

        let mut pos = tensor_index_offset;
        for _ in 0..tensor_count {
            if pos + 4 > data.len() {
                break;
            }

            // Read tensor name length and name
            let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;

            if pos + name_len + 18 > data.len() {
                break;
            }

            let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
            pos += name_len;

            // Read dtype (1 byte)
            let dtype = data[pos];
            pos += 1;

            // Read ndim (1 byte)
            let ndim = data[pos] as usize;
            pos += 1;

            // Read dimensions (8 bytes each)
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                if pos + 8 > data.len() {
                    break;
                }
                let dim = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize;
                dims.push(dim);
                pos += 8;
            }

            // Read offset (8 bytes)
            if pos + 16 > data.len() {
                break;
            }
            let offset = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;

            // Read size (8 bytes)
            let size = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;

            tensors.insert(name, (data_offset + offset, size, dims, dtype));
        }

        tensors
    }

    /// Search for embedding tensor name and return its dimensions
    fn find_embedding_dims(
        tensors: &std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)>,
        vocab_size: usize,
        hidden_dim: usize,
        debug_enabled: bool,
    ) -> Option<Vec<usize>> {
        for name in &[
            "model.embed_tokens.weight",
            "token_embd.weight",
            "tok_embeddings.weight",
        ] {
            if let Some((_offset, _size, dims, _dtype)) = tensors.get(*name) {
                if debug_enabled {
                    eprintln!(
                        "[APR-LOAD] Embedding tensor '{}': dims={:?}, expected [vocab={}, hidden={}]",
                        name, dims, vocab_size, hidden_dim
                    );
                }
                return Some(dims.clone());
            }
        }
        None
    }

    /// Validate token 0 embedding for corruption (all-zero, NaN, Inf)
    fn validate_embedding_token0(
        token_embedding: &[f32],
        hidden_dim: usize,
        debug_enabled: bool,
    ) {
        if token_embedding.len() < hidden_dim {
            return;
        }
        let first_embed = &token_embedding[0..hidden_dim];
        let all_zero = first_embed.iter().all(|&x| x == 0.0);
        let has_nan = first_embed.iter().any(|x| x.is_nan());
        let has_inf = first_embed.iter().any(|x| x.is_infinite());
        if all_zero {
            eprintln!(
                "[APR-LOAD] WARNING: Token 0 embedding is all zeros - possible load failure"
            );
        }
        if has_nan || has_inf {
            eprintln!(
                "[APR-LOAD] ERROR: Token 0 embedding contains NaN/Inf - data corruption!"
            );
        }
        if debug_enabled {
            eprintln!(
                "[APR-LOAD] Token 0 embedding sample: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                first_embed[0],
                first_embed.get(1).unwrap_or(&0.0),
                first_embed.get(2).unwrap_or(&0.0),
                first_embed.get(3).unwrap_or(&0.0),
                first_embed.get(4).unwrap_or(&0.0)
            );
        }
    }

    /// Resolve LM head F32 weights with tied-weight fallback
    ///
    /// Tries lm_head.weight, output.weight, then falls back to embedding weights.
    /// Returns (weights, used_tied_weights).
    fn resolve_lm_head_f32(
        lookup: &AprTensorLookup<'_>,
        debug_enabled: bool,
    ) -> Result<(Vec<f32>, bool)> {
        if let Some(lm_head) = lookup.get_f32("lm_head.weight").or_else(|| lookup.get_f32("output.weight")) {
            return Ok((lm_head, false));
        }
        // Weight tying: use embedding weights for lm_head
        if debug_enabled {
            eprintln!("[APR-LOAD] No lm_head found, trying tied embedding weights");
        }
        let tied = lookup.get_f32("model.embed_tokens.weight")
            .or_else(|| lookup.get_f32("token_embd.weight"));
        if let Some(t) = tied {
            Ok((t, true))
        } else {
            Err(RealizarError::FormatError {
                reason: "FATAL: No lm_head tensor found and no embedding for weight tying. \
                        Tried: lm_head.weight, output.weight, model.embed_tokens.weight, \
                        token_embd.weight. APR file may be corrupt."
                    .to_string(),
            })
        }
    }

    /// Log LM head quantization path (debug only)
    fn debug_log_lm_head_quant(
        lm_head_weight_q4k: Option<&[u8]>,
        lm_head_weight_q6k: Option<&[u8]>,
    ) {
        if let Some(bytes) = lm_head_weight_q4k {
            eprintln!(
                "[APR-LOAD] LM head using Q4K fused kernel ({} bytes)",
                bytes.len()
            );
        } else if let Some(bytes) = lm_head_weight_q6k {
            eprintln!(
                "[APR-LOAD] LM head using Q6K fused kernel ({} bytes)",
                bytes.len()
            );
        } else {
            eprintln!("[APR-LOAD] LM head using F32 matmul (no Q4K/Q6K found)");
        }
    }

    /// Load token embedding tensor with fallback names and sanity checks.
    ///
    /// Tries `model.embed_tokens.weight`, `token_embd.weight`, `tok_embeddings.weight`.
    /// Validates the embedding for token 0 (all-zero, NaN, Inf checks).
    fn load_apr_embedding(
        lookup: &AprTensorLookup<'_>,
        tensors: &std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)>,
        vocab_size: usize,
        hidden_dim: usize,
        debug_enabled: bool,
    ) -> Result<Vec<f32>> {
        let embed_dims = Self::find_embedding_dims(tensors, vocab_size, hidden_dim, debug_enabled);

        // Try to load token embedding - FAIL FAST if not found (no silent zeros)
        let token_embedding = lookup.get_f32("model.embed_tokens.weight")
            .or_else(|| lookup.get_f32("token_embd.weight"))
            .or_else(|| lookup.get_f32("tok_embeddings.weight"))
            .ok_or_else(|| RealizarError::FormatError {
                reason: "FATAL: No embedding tensor found. Tried: model.embed_tokens.weight, \
                        token_embd.weight, tok_embeddings.weight. APR file may be corrupt or \
                        use unsupported tensor naming convention."
                    .to_string(),
            })?;

        // GH-208 FIX: Do NOT transpose embedding data
        if debug_enabled {
            if let Some(ref dims) = embed_dims {
                eprintln!(
                    "[APR-LOAD] Embedding dims={:?}, using raw data (no transpose needed)",
                    dims
                );
            }
        }

        // GH-187: Sanity check - verify embedding produces non-garbage for token 0
        Self::validate_embedding_token0(&token_embedding, hidden_dim, debug_enabled);

        // GH-253: Only log embedding size in debug mode
        if debug_enabled {
            eprintln!(
                "[APR-LOAD] Embedding loaded: {} elements (vocab={} x hidden={})",
                token_embedding.len(),
                vocab_size,
                hidden_dim
            );
        }

        Ok(token_embedding)
    }

    /// Load LM head tensor with tied-weights fallback and quantized variants.
    ///
    /// Tries `lm_head.weight`, `output.weight`, then falls back to embedding weights (tied).
    /// Also loads Q4K/Q6K raw bytes for fused kernel inference.
    /// Returns (lm_head_weight_f32, q4k_bytes, q6k_bytes).
    #[allow(clippy::type_complexity)]
    fn load_apr_lm_head(
        lookup: &AprTensorLookup<'_>,
        tensors: &std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)>,
        vocab_size: usize,
        hidden_dim: usize,
        debug_enabled: bool,
    ) -> Result<(Vec<f32>, Option<Vec<u8>>, Option<Vec<u8>>)> {
        if debug_enabled {
            for name in &["lm_head.weight", "output.weight"] {
                if let Some((_offset, _size, dims, dtype)) = tensors.get(*name) {
                    eprintln!(
                        "[APR-LOAD] LM head tensor '{}': dims={:?}, dtype={}, expected [vocab={}, hidden={}]",
                        name, dims, dtype, vocab_size, hidden_dim
                    );
                    break;
                }
            }
        }

        let (lm_head_weight, used_tied_weights) = Self::resolve_lm_head_f32(lookup, debug_enabled)?;
        if used_tied_weights && debug_enabled {
            eprintln!("[APR-LOAD] Using tied weights: embedding -> lm_head");
        }

        if debug_enabled {
            eprintln!(
                "[APR-LOAD] LM head loaded: {} elements (hidden={} x vocab={})",
                lm_head_weight.len(),
                hidden_dim,
                vocab_size
            );
        }

        let lm_head_weight_q4k =
            lookup.get_q4k("lm_head.weight").or_else(|| lookup.get_q4k("output.weight"));
        let lm_head_weight_q6k =
            lookup.get_q6k("lm_head.weight").or_else(|| lookup.get_q6k("output.weight"));
        if debug_enabled {
            Self::debug_log_lm_head_quant(lm_head_weight_q4k.as_deref(), lm_head_weight_q6k.as_deref());
        }

        Ok((lm_head_weight, lm_head_weight_q4k, lm_head_weight_q6k))
    }

    /// Load APR transformer from bytes
    ///
    /// Parses APR v2 format from memory buffer.
    pub fn from_apr_bytes(data: &[u8]) -> Result<Self> {
        // Check minimum size for header
        if data.len() < 64 {
            return Err(RealizarError::FormatError {
                reason: format!("APR file too small: {} bytes (need 64)", data.len()),
            });
        }

        // Check magic - first 3 bytes must be "APR", 4th byte is version (0, '1', or '2')
        let magic = data.get(0..4).expect("APR file validated to have at least 64 bytes above");
        if magic[0..3] != *b"APR" || (magic[3] != 0 && magic[3] != b'1' && magic[3] != b'2') {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid APR magic: {:?}, expected APR followed by version byte",
                    String::from_utf8_lossy(magic)
                ),
            });
        }

        // Parse header
        // APR header layout:
        //   0-3: Magic "APR\0"
        //   4-5: Version major.minor
        //   6-7: Flags
        //   8-11: Tensor count
        //   12-19: Metadata offset
        //   20-23: Metadata size
        //   24-31: Tensor index offset
        //   32-39: Data offset
        //   40-43: Checksum
        //   44-63: Reserved

        let tensor_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let metadata_offset = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]) as usize;
        let metadata_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;
        let tensor_index_offset = u64::from_le_bytes([
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
        ]) as usize;
        let data_offset = u64::from_le_bytes([
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        ]) as usize;

        // Parse metadata (JSON)
        let metadata_end = metadata_offset + metadata_size;
        if metadata_end > data.len() {
            return Err(RealizarError::FormatError {
                reason: "Metadata extends beyond file".to_string(),
            });
        }

        let metadata_json = &data[metadata_offset..metadata_end];
        let metadata: serde_json::Value = serde_json::from_slice(metadata_json).unwrap_or_default();

        // Extract architecture info from metadata
        let config = Self::parse_apr_metadata_config(&metadata);

        let debug_enabled = std::env::var("REALIZE_DEBUG").is_ok();
        if debug_enabled {
            eprintln!("[DEBUG] AprTransformerConfig: hidden_dim={}, num_layers={}, num_heads={}, num_kv_heads={}, vocab_size={}, intermediate_dim={}",
                config.hidden_dim, config.num_layers, config.num_heads, config.num_kv_heads, config.vocab_size, config.intermediate_dim);
        }

        // GH-478: Contract gate — validate architecture and dimensions before loading weights.
        // AprTransformer was the only loading path missing this gate (GH-279 gap).
        let _proof = crate::contract_gate::validate_model_load_basic(
            &config.architecture,
            config.num_layers,
            config.hidden_dim,
            config.num_heads,
            config.num_kv_heads,
            config.intermediate_dim,
            config.vocab_size,
        ).map_err(crate::contract_gate::gate_error)?;

        // Parse tensor index
        let tensors = Self::parse_apr_tensor_index(data, tensor_index_offset, data_offset, tensor_count);

        // GH-478: Resource limit gate — check if F32 dequant would exceed system RAM.
        // This fires BEFORE any weight allocation, preventing OOM.
        let tensor_entries: Vec<(usize, u8)> = tensors
            .values()
            .map(|&(_, size, _, dtype)| (size, dtype))
            .collect();
        crate::contract_gate::validate_f32_dequant_limits(
            &tensor_entries,
            data.len() as u64,
        ).map_err(crate::contract_gate::gate_error)?;

        // ALB-106: entrenar checkpoints store linear weights in cuBLAS column-major
        // convention. Detect via "format": "entrenar-checkpoint" metadata.
        let is_entrenar = metadata
            .get("format")
            .and_then(serde_json::Value::as_str)
            == Some("entrenar-checkpoint");
        if debug_enabled && is_entrenar {
            eprintln!("[DEBUG] Detected entrenar checkpoint — will transpose linear weights");
        }

        // Tensor lookup helper (replaces closures for decomposition)
        let lookup = AprTensorLookup { data, tensors: &tensors, transpose_cublas_weights: is_entrenar };

        // Debug: print available tensor names (only when REALIZE_DEBUG is set)
        if debug_enabled {
            eprintln!("[DEBUG] APR v2 tensor count: {tensor_count}");
            eprintln!("[DEBUG] Available tensor names (first 10):");
            for (i, (name, (offset, size, dims, dtype))) in tensors.iter().enumerate() {
                if i < 10 {
                    eprintln!(
                        "  {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}"
                    );
                }
            }
        }

        // Load token embedding with sanity checks
        let token_embedding = Self::load_apr_embedding(
            &lookup, &tensors, config.vocab_size, config.hidden_dim, debug_enabled,
        )?;

        // Load output norm
        let output_norm_weight = lookup.get_f32("model.norm.weight")
            .or_else(|| lookup.get_f32("output_norm.weight"))
            .unwrap_or_else(|| vec![1.0; config.hidden_dim]);

        // Load LM head with tied-weights fallback and quantized variants
        let (lm_head_weight, lm_head_weight_q4k, lm_head_weight_q6k) = Self::load_apr_lm_head(
            &lookup, &tensors, config.vocab_size, config.hidden_dim, debug_enabled,
        )?;

        // ALB-094: Use explicit head_dim when available (MoE models like Qwen3.5)
        let head_dim = config.explicit_head_dim.unwrap_or(config.hidden_dim / config.num_heads);
        let kv_dim = config.num_kv_heads * head_dim;

        // Load per-layer weights (extracted for decomposition)
        let (layers, q4k_layers) = Self::build_apr_layers(
            &lookup, config.num_layers, config.hidden_dim, kv_dim, config.intermediate_dim,
            config.num_experts, debug_enabled,
        );

        Ok(Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            q4k_layers,
            lm_head_weight_q6k,
            lm_head_weight_q4k,
        })
    }
}

include!("mod_dequant_q4k_apr.rs");
