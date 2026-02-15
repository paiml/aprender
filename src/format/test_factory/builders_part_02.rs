
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
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            embed_data,
        ));
    }

    // LM head: [vocab_size, hidden_size]
    if config.include_embedding && !config.tied_embeddings {
        let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            lm_head_data,
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
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        writer.add_f32_tensor(
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_size],
            &embed_data,
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
            let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
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
            let gate_up_data: Vec<f32> = (0..intermediate * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            let down_data: Vec<f32> = (0..config.hidden_size * intermediate)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();

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
        let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        writer.add_f32_tensor(
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_size],
            &lm_head_data,
        );
    }

    writer.write().unwrap_or_default()
}

/// Build APR with Q8 quantized tensors
#[must_use]
pub fn build_pygmy_apr_q8() -> Vec<u8> {
    let config = PygmyConfig::default();
    let mut metadata = AprV2Metadata::new("pygmy");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(config.hidden_size);
    metadata.vocab_size = Some(config.vocab_size);

    let mut writer = AprV2Writer::new(metadata);

    // Embedding (F32 - lookup table)
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Layer 0 attention (Q8)
    let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
        .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
        .collect();
    for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        writer.add_q8_tensor(
            format!("model.layers.0.self_attn.{suffix}.weight"),
            vec![config.hidden_size, config.hidden_size],
            &qkvo_data,
        );
    }

    // Norms (F32)
    let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
    writer.add_f32_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![config.hidden_size],
        &norm_data,
    );
    writer.add_f32_tensor("model.norm.weight", vec![config.hidden_size], &norm_data);

    writer.write().unwrap_or_default()
}

/// Build APR with Q4 quantized tensors
#[must_use]
pub fn build_pygmy_apr_q4() -> Vec<u8> {
    let config = PygmyConfig::default();
    let mut metadata = AprV2Metadata::new("pygmy");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(32); // Q4 block size alignment
    metadata.vocab_size = Some(config.vocab_size);

    let mut writer = AprV2Writer::new(metadata);

    // Embedding (F32)
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Layer 0 attention (Q4) - need at least 32 elements for Q4 blocks
    let q4_size = 32; // Q4 block size
    let q4_data: Vec<f32> = (0..q4_size * q4_size)
        .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
        .collect();
    for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        writer.add_q4_tensor(
            format!("model.layers.0.self_attn.{suffix}.weight"),
            vec![q4_size, q4_size],
            &q4_data,
        );
    }

    // Norms (F32)
    let norm_data: Vec<f32> = vec![1.0; q4_size];
    writer.add_f32_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![q4_size],
        &norm_data,
    );
    writer.add_f32_tensor("model.norm.weight", vec![q4_size], &norm_data);

    writer.write().unwrap_or_default()
}

/// Build APR with F16 tensors
#[must_use]
pub fn build_pygmy_apr_f16() -> Vec<u8> {
    let config = PygmyConfig::default();
    let mut metadata = AprV2Metadata::new("pygmy");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(config.hidden_size);
    metadata.vocab_size = Some(config.vocab_size);

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding (F16)
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f16_tensor(
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Layer 0 attention (F16)
    let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
        .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
        .collect();
    for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        writer.add_f16_tensor(
            format!("model.layers.0.self_attn.{suffix}.weight"),
            vec![config.hidden_size, config.hidden_size],
            &qkvo_data,
        );
    }

    // Norms (F16)
    let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
    writer.add_f16_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![config.hidden_size],
        &norm_data,
    );
    writer.add_f16_tensor("model.norm.weight", vec![config.hidden_size], &norm_data);

    writer.write().unwrap_or_default()
}

// ============================================================================
// K-Quant Pygmy Builders (Q4K / Q6K)
// ============================================================================

/// Build APR with Q4_K quantized tensors (GGUF-style names)
///
/// Q4_K format: 256-element super-blocks, 144 bytes each.
/// Layout: d (f16, 2B) + dmin (f16, 2B) + scales (12B) + qs (128B) = 144 bytes.
/// Uses GGUF naming: `token_embd.weight`, `blk.0.attn_q.weight`, etc.
#[must_use]
pub fn build_pygmy_apr_q4k() -> Vec<u8> {
    let mut metadata = AprV2Metadata::new("pygmy-q4k");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(256);
    metadata.vocab_size = Some(8);
    metadata.num_layers = Some(1);
    metadata
        .custom
        .insert("naming".to_string(), serde_json::json!("gguf"));

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding (F32 -- embedding lookup tables stay unquantized)
    let embed_data: Vec<f32> = (0..8 * 256)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor("token_embd.weight", vec![8, 256], &embed_data);

    // Layer 0 attention weights (Q4K): 256 elements = 1 super-block = 144 bytes
    let q4k_block = vec![0u8; 144];
    for suffix in &["attn_q", "attn_k", "attn_v", "attn_output"] {
        writer.add_q4k_raw_tensor(
            format!("blk.0.{suffix}.weight"),
            vec![256, 1],
            q4k_block.clone(),
        );
    }

    // Norms (F32)
    let norm_data: Vec<f32> = vec![1.0; 256];
    writer.add_f32_tensor("blk.0.attn_norm.weight", vec![256], &norm_data);
    writer.add_f32_tensor("output_norm.weight", vec![256], &norm_data);

    // Output (F32)
    let output_data: Vec<f32> = (0..8 * 256)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor("output.weight", vec![8, 256], &output_data);

    writer.write().unwrap_or_default()
}
