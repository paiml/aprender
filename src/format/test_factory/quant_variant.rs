
// ============================================================================
// Shared tensor data generation helpers (reduces DataTransformation entropy)
// ============================================================================

/// Generate embedding-style tensor data: values in [-0.05, 0.05] range
fn gen_embed_data(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect()
}

/// Generate weight-style tensor data: values in [-0.05, 0.05] range
fn gen_weight_data(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
        .collect()
}

/// Create standard llama-style APR metadata with a writer
fn new_llama_apr_writer(name: &str, hidden_size: usize, vocab_size: usize) -> AprV2Writer {
    let mut metadata = AprV2Metadata::new(name);
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(hidden_size);
    metadata.vocab_size = Some(vocab_size);
    AprV2Writer::new(metadata)
}

/// Descriptor for a single-layer quantized APR model variant.
///
/// Captures the differences between Q8/Q4/F16 builders so the shared
/// `build_single_layer_apr` helper can emit the right tensor types.
enum QuantVariant {
    Q8 { hidden: usize },
    Q4 { block_size: usize },
    F16 { hidden: usize },
}

/// Shared builder for single-layer quantized APR models (Q8, Q4, F16).
///
/// All three variants follow the same structure: embed (F32) + attention (quant) +
/// norms (same dtype as variant) + final norm. Only the tensor-add method differs.
fn build_single_layer_apr(variant: QuantVariant) -> Vec<u8> {
    let config = PygmyConfig::default();

    match variant {
        QuantVariant::Q8 { hidden } => {
            let mut writer = new_llama_apr_writer("pygmy", hidden, config.vocab_size);

            // Embedding (F32)
            writer.add_f32_tensor(
                "model.embed_tokens.weight",
                vec![config.vocab_size, hidden],
                &gen_embed_data(config.vocab_size * hidden),
            );

            // Layer 0 attention (Q8)
            let qkvo_data = gen_weight_data(hidden * hidden);
            for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                writer.add_q8_tensor(
                    format!("model.layers.0.self_attn.{suffix}.weight"),
                    vec![hidden, hidden],
                    &qkvo_data,
                );
            }

            // Norms (F32)
            let norm_data: Vec<f32> = vec![1.0; hidden];
            writer.add_f32_tensor(
                "model.layers.0.input_layernorm.weight",
                vec![hidden],
                &norm_data,
            );
            writer.add_f32_tensor("model.norm.weight", vec![hidden], &norm_data);

            writer.write().unwrap_or_default()
        }
        QuantVariant::Q4 { block_size } => {
            let mut writer = new_llama_apr_writer("pygmy", block_size, config.vocab_size);

            // Embedding (F32) — uses default hidden_size for embed shape
            writer.add_f32_tensor(
                "model.embed_tokens.weight",
                vec![config.vocab_size, config.hidden_size],
                &gen_embed_data(config.vocab_size * config.hidden_size),
            );

            // Layer 0 attention (Q4)
            let q4_data = gen_weight_data(block_size * block_size);
            for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                writer.add_q4_tensor(
                    format!("model.layers.0.self_attn.{suffix}.weight"),
                    vec![block_size, block_size],
                    &q4_data,
                );
            }

            // Norms (F32)
            let norm_data: Vec<f32> = vec![1.0; block_size];
            writer.add_f32_tensor(
                "model.layers.0.input_layernorm.weight",
                vec![block_size],
                &norm_data,
            );
            writer.add_f32_tensor("model.norm.weight", vec![block_size], &norm_data);

            writer.write().unwrap_or_default()
        }
        QuantVariant::F16 { hidden } => {
            let mut writer = new_llama_apr_writer("pygmy", hidden, config.vocab_size);

            // Token embedding (F16)
            writer.add_f16_tensor(
                "model.embed_tokens.weight",
                vec![config.vocab_size, hidden],
                &gen_embed_data(config.vocab_size * hidden),
            );

            // Layer 0 attention (F16)
            let qkvo_data = gen_weight_data(hidden * hidden);
            for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                writer.add_f16_tensor(
                    format!("model.layers.0.self_attn.{suffix}.weight"),
                    vec![hidden, hidden],
                    &qkvo_data,
                );
            }

            // Norms (F16)
            let norm_data: Vec<f32> = vec![1.0; hidden];
            writer.add_f16_tensor(
                "model.layers.0.input_layernorm.weight",
                vec![hidden],
                &norm_data,
            );
            writer.add_f16_tensor("model.norm.weight", vec![hidden], &norm_data);

            writer.write().unwrap_or_default()
        }
    }
}

// ============================================================================
// GH-205: F16 SafeTensors Builder for Passthrough Testing
// ============================================================================

/// Build a minimal valid F16 SafeTensors file in memory
///
/// Creates a SafeTensors file with F16 (half-precision) tensors for testing
/// the GH-205 F16 passthrough fix. The F32 values are converted to F16.
#[must_use]
pub fn build_pygmy_safetensors_f16() -> Vec<u8> {
    build_pygmy_safetensors_f16_with_config(PygmyConfig::minimal())
}

/// Build F16 SafeTensors with custom config
#[must_use]
pub fn build_pygmy_safetensors_f16_with_config(config: PygmyConfig) -> Vec<u8> {
    // Build tensor metadata and data (same as F32 version, but stored as F16)
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

    // Token embedding: [vocab_size, hidden_size]
    if config.include_embedding {
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            gen_embed_data(config.vocab_size * config.hidden_size),
        ));
    }

    // LM head: [vocab_size, hidden_size]
    if config.include_embedding && !config.tied_embeddings {
        tensors.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            gen_embed_data(config.vocab_size * config.hidden_size),
        ));
    }

    build_safetensors_bytes_f16(&tensors)
}

/// Build F16 SafeTensors bytes from tensor list
///
/// Converts F32 values to F16 format for storage.
fn build_safetensors_bytes_f16(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    use std::collections::BTreeMap;

    // Calculate tensor data offsets
    let mut current_offset = 0usize;
    let mut tensor_info: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut all_data = Vec::new();

    for (name, shape, data) in tensors {
        let byte_size = data.len() * 2; // f16 = 2 bytes

        // SafeTensors format: {"dtype": "F16", "shape": [...], "data_offsets": [start, end]}
        tensor_info.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F16",
                "shape": shape,
                "data_offsets": [current_offset, current_offset + byte_size]
            }),
        );

        // Append tensor data (little-endian f16)
        for &val in data {
            all_data.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
        }

        current_offset += byte_size;
    }

    // Add __metadata__ for completeness
    tensor_info.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt", "pygmy": "true", "dtype": "F16"}),
    );

    // Serialize header
    let header_json = serde_json::to_string(&tensor_info).unwrap_or_default();
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    // Build final file: [header_len: u64] + [header: JSON] + [tensor_data]
    let mut result = Vec::with_capacity(8 + header_bytes.len() + all_data.len());
    result.extend_from_slice(&header_len.to_le_bytes());
    result.extend_from_slice(header_bytes);
    result.extend_from_slice(&all_data);

    result
}

/// Convert f32 to f16 bits (IEEE 754 half-precision)
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007fffff;

    if exp == 255 {
        // Inf or NaN
        if mant != 0 {
            sign | 0x7e00 // NaN
        } else {
            sign | 0x7c00 // Inf
        }
    } else if exp > 142 {
        // Overflow to Inf
        sign | 0x7c00
    } else if exp < 113 {
        // Underflow to zero (or denormal, which we skip for simplicity)
        sign
    } else {
        // Normal number
        let new_exp = ((exp - 127 + 15) as u16) << 10;
        let new_mant = (mant >> 13) as u16;
        sign | new_exp | new_mant
    }
}

// ============================================================================
// APR v2 Pygmy Builder
// ============================================================================

/// Build a minimal valid APR v2 file in memory
///
/// Creates an APR v2 file with:
/// - Valid header with magic "APR2"
/// - Metadata with architecture info
/// - F32 tensors (embedding + weight)
/// - 64-byte alignment
#[must_use]
pub fn build_pygmy_apr() -> Vec<u8> {
    build_pygmy_apr_with_config(PygmyConfig::default())
}

/// Build APR v2 with custom config
#[must_use]
pub fn build_pygmy_apr_with_config(config: PygmyConfig) -> Vec<u8> {
    let mut metadata = AprV2Metadata::new("pygmy");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(config.hidden_size);
    metadata.vocab_size = Some(config.vocab_size);
    metadata.num_layers = Some(config.num_layers);
    metadata
        .custom
        .insert("pygmy".to_string(), serde_json::json!(true));

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding
    if config.include_embedding {
        writer.add_f32_tensor(
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_size],
            &gen_embed_data(config.vocab_size * config.hidden_size),
        );
    }

    // Layer tensors
    for layer_idx in 0..config.num_layers {
        if config.include_norms {
            let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
                &norm_data,
            );
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
                &norm_data,
            );
        }

        if config.include_attention {
            let qkvo_data = gen_weight_data(config.hidden_size * config.hidden_size);
            for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                writer.add_f32_tensor(
                    format!("model.layers.{layer_idx}.self_attn.{suffix}.weight"),
                    vec![config.hidden_size, config.hidden_size],
                    &qkvo_data,
                );
            }
        }

        if config.include_mlp {
            let intermediate = config.hidden_size * 2;
            let gate_up_data = gen_weight_data(intermediate * config.hidden_size);
            let down_data = gen_weight_data(config.hidden_size * intermediate);

            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![intermediate, config.hidden_size],
                &gate_up_data,
            );
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![intermediate, config.hidden_size],
                &gate_up_data,
            );
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![config.hidden_size, intermediate],
                &down_data,
            );
        }
    }

    // Final norm
    if config.include_norms && config.num_layers > 0 {
        let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
        writer.add_f32_tensor("model.norm.weight", vec![config.hidden_size], &norm_data);
    }

    // LM head
    if config.include_embedding {
        writer.add_f32_tensor(
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_size],
            &gen_embed_data(config.vocab_size * config.hidden_size),
        );
    }

    writer.write().unwrap_or_default()
}

/// Build APR with Q8 quantized tensors
#[must_use]
pub fn build_pygmy_apr_q8() -> Vec<u8> {
    build_single_layer_apr(QuantVariant::Q8 {
        hidden: PygmyConfig::default().hidden_size,
    })
}

/// Build APR with Q4 quantized tensors
#[must_use]
pub fn build_pygmy_apr_q4() -> Vec<u8> {
    build_single_layer_apr(QuantVariant::Q4 { block_size: 32 })
}

/// Build APR with F16 tensors
#[must_use]
pub fn build_pygmy_apr_f16() -> Vec<u8> {
    build_single_layer_apr(QuantVariant::F16 {
        hidden: PygmyConfig::default().hidden_size,
    })
}

// ============================================================================
// K-Quant Pygmy Builders (Q4K / Q6K)
// ============================================================================

/// Shared builder for GGUF-style K-quant APR models (Q4K, Q6K).
///
/// Both Q4K and Q6K follow the same structure with GGUF naming:
/// embed (F32) + attention (raw quant blocks) + norms (F32) + output (F32).
/// Only the block size and tensor-add method differ.
fn build_kquant_gguf_apr(
    name: &str,
    block_bytes: usize,
    add_raw_fn: fn(&mut AprV2Writer, String, Vec<usize>, Vec<u8>),
) -> Vec<u8> {
    let (vocab, hidden) = (8, 256);

    let mut metadata = AprV2Metadata::new(name);
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(hidden);
    metadata.vocab_size = Some(vocab);
    metadata.num_layers = Some(1);
    metadata
        .custom
        .insert("naming".to_string(), serde_json::json!("gguf"));

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding (F32 -- embedding lookup tables stay unquantized)
    writer.add_f32_tensor("token_embd.weight", vec![vocab, hidden], &gen_embed_data(vocab * hidden));

    // Layer 0 attention weights (K-quant): 256 elements = 1 super-block
    let raw_block = vec![0u8; block_bytes];
    for suffix in &["attn_q", "attn_k", "attn_v", "attn_output"] {
        add_raw_fn(
            &mut writer,
            format!("blk.0.{suffix}.weight"),
            vec![hidden, 1],
            raw_block.clone(),
        );
    }

    // Norms (F32)
    let norm_data: Vec<f32> = vec![1.0; hidden];
    writer.add_f32_tensor("blk.0.attn_norm.weight", vec![hidden], &norm_data);
    writer.add_f32_tensor("output_norm.weight", vec![hidden], &norm_data);

    // Output (F32)
    writer.add_f32_tensor("output.weight", vec![vocab, hidden], &gen_embed_data(vocab * hidden));

    writer.write().unwrap_or_default()
}

/// Build APR with Q4_K quantized tensors (GGUF-style names)
///
/// Q4_K format: 256-element super-blocks, 144 bytes each.
/// Layout: d (f16, 2B) + dmin (f16, 2B) + scales (12B) + qs (128B) = 144 bytes.
/// Uses GGUF naming: `token_embd.weight`, `blk.0.attn_q.weight`, etc.
#[must_use]
pub fn build_pygmy_apr_q4k() -> Vec<u8> {
    build_kquant_gguf_apr("pygmy-q4k", 144, |w, name, shape, data| {
        w.add_q4k_raw_tensor(name, shape, data);
    })
}
