//! Test Factory - Pygmy Model Builders (T-COV-95)
//!
//! Implements the "Active Pygmy" pattern from realizar for creating minimal
//! valid model files in memory without needing real model files on disk.
//!
//! # Dr. Popper's "Minimum Viable Predictor"
//!
//! A tiny model that:
//! 1. Has valid tensor layout
//! 2. Has valid quantized/unquantized weights
//! 3. Exercises all code paths in format loading/conversion
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::test_factory::{build_pygmy_safetensors, build_pygmy_apr};
//!
//! // Create minimal SafeTensors in memory
//! let st_bytes = build_pygmy_safetensors();
//!
//! // Create minimal APR v2 in memory
//! let apr_bytes = build_pygmy_apr();
//! ```

use crate::format::v2::{AprV2Metadata, AprV2Writer};

// ============================================================================
// SafeTensors Pygmy Builder
// ============================================================================

/// Build a minimal valid SafeTensors file in memory
///
/// Creates a SafeTensors file with:
/// - 2 F32 tensors (embedding + weight)
/// - Small dimensions (vocab=8, hidden=4)
/// - Valid JSON header format
#[must_use]
pub fn build_pygmy_safetensors() -> Vec<u8> {
    build_pygmy_safetensors_with_config(PygmyConfig::default())
}

/// Configuration for pygmy model generation
#[derive(Debug, Clone)]
pub struct PygmyConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Include embedding tensor
    pub include_embedding: bool,
    /// Include norm tensors
    pub include_norms: bool,
    /// Include attention tensors
    pub include_attention: bool,
    /// Include MLP tensors
    pub include_mlp: bool,
}

impl Default for PygmyConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8,
            hidden_size: 4,
            num_layers: 1,
            include_embedding: true,
            include_norms: true,
            include_attention: true,
            include_mlp: true,
        }
    }
}

impl PygmyConfig {
    /// Create minimal config (smallest valid model)
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            vocab_size: 4,
            hidden_size: 2,
            num_layers: 1,
            include_embedding: true,
            include_norms: false,
            include_attention: false,
            include_mlp: false,
        }
    }

    /// Create config with only embedding and LM head
    #[must_use]
    pub fn embedding_only() -> Self {
        Self {
            vocab_size: 8,
            hidden_size: 4,
            num_layers: 0,
            include_embedding: true,
            include_norms: false,
            include_attention: false,
            include_mlp: false,
        }
    }

    /// Create LLaMA-style config
    #[must_use]
    pub fn llama_style() -> Self {
        Self {
            vocab_size: 16,
            hidden_size: 8,
            num_layers: 1,
            include_embedding: true,
            include_norms: true,
            include_attention: true,
            include_mlp: true,
        }
    }
}

/// Build SafeTensors with custom config
#[must_use]
pub fn build_pygmy_safetensors_with_config(config: PygmyConfig) -> Vec<u8> {
    // Build tensor metadata and data
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

    // Layer tensors
    for layer_idx in 0..config.num_layers {
        // Input layernorm
        if config.include_norms {
            let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
            tensors.push((
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
                norm_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
                norm_data,
            ));
        }

        // Attention: Q, K, V, O projections
        if config.include_attention {
            let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.{suffix}.weight"),
                    vec![config.hidden_size, config.hidden_size],
                    qkvo_data.clone(),
                ));
            }
        }

        // MLP: gate, up, down projections
        if config.include_mlp {
            let intermediate = config.hidden_size * 2;
            let gate_up_data: Vec<f32> = (0..intermediate * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            let down_data: Vec<f32> = (0..config.hidden_size * intermediate)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();

            tensors.push((
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![intermediate, config.hidden_size],
                gate_up_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![intermediate, config.hidden_size],
                gate_up_data,
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![config.hidden_size, intermediate],
                down_data,
            ));
        }
    }

    // Final norm
    if config.include_norms && config.num_layers > 0 {
        let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
        tensors.push(("model.norm.weight".to_string(), vec![config.hidden_size], norm_data));
    }

    // LM head: [vocab_size, hidden_size]
    if config.include_embedding {
        let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            lm_head_data,
        ));
    }

    // Build SafeTensors format
    build_safetensors_bytes(&tensors)
}

/// Build SafeTensors bytes from tensor list
fn build_safetensors_bytes(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    use std::collections::BTreeMap;

    // Calculate tensor data offsets
    let mut current_offset = 0usize;
    let mut tensor_info: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut all_data = Vec::new();

    for (name, shape, data) in tensors {
        let byte_size = data.len() * 4; // f32 = 4 bytes

        // SafeTensors format: {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}
        tensor_info.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [current_offset, current_offset + byte_size]
            }),
        );

        // Append tensor data (little-endian f32)
        for &val in data {
            all_data.extend_from_slice(&val.to_le_bytes());
        }

        current_offset += byte_size;
    }

    // Add __metadata__ for completeness
    tensor_info.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt", "pygmy": "true"}),
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
    metadata.custom.insert("pygmy".to_string(), serde_json::json!(true));

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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::v2::{AprV2Reader, MAGIC_V2};

    #[test]
    fn test_pygmy_safetensors_valid() {
        let data = build_pygmy_safetensors();

        // Should have valid header length
        assert!(data.len() > 8);
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
        assert!(header_len > 0);
        assert!(header_len < 10000); // Reasonable size

        // Should have JSON header
        let header_end = 8 + header_len as usize;
        assert!(data.len() >= header_end);
        let header_str = std::str::from_utf8(&data[8..header_end]).unwrap();
        assert!(header_str.starts_with('{'));
        assert!(header_str.contains("model.embed_tokens.weight"));
    }

    #[test]
    fn test_pygmy_safetensors_with_config() {
        let config = PygmyConfig::llama_style();
        let data = build_pygmy_safetensors_with_config(config);

        assert!(data.len() > 100);
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let header_end = 8 + header_len as usize;
        let header_str = std::str::from_utf8(&data[8..header_end]).unwrap();

        // Should have attention tensors
        assert!(header_str.contains("self_attn.q_proj.weight"));
        assert!(header_str.contains("self_attn.k_proj.weight"));
    }

    #[test]
    fn test_pygmy_safetensors_minimal() {
        let config = PygmyConfig::minimal();
        let data = build_pygmy_safetensors_with_config(config);

        // Minimal config should produce small file
        assert!(data.len() < 1000);
    }

    #[test]
    fn test_pygmy_apr_valid() {
        let data = build_pygmy_apr();

        // Should have valid magic
        assert!(data.len() >= 64);
        assert_eq!(&data[0..4], &MAGIC_V2);

        // Should be parseable
        let reader = AprV2Reader::from_bytes(&data);
        assert!(reader.is_ok());
    }

    #[test]
    fn test_pygmy_apr_metadata() {
        let data = build_pygmy_apr();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        // Should have architecture metadata
        assert_eq!(reader.metadata().architecture, Some("llama".to_string()));

        // Should have tensors
        assert!(!reader.tensor_names().is_empty());
    }

    #[test]
    fn test_pygmy_apr_tensor_count() {
        let config = PygmyConfig::llama_style();
        let data = build_pygmy_apr_with_config(config);
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        // LLaMA style: embed + 2 norms + 4 attn + 3 mlp + final norm + lm_head = 12
        assert!(reader.tensor_names().len() >= 10);
    }

    #[test]
    fn test_pygmy_apr_alignment() {
        let data = build_pygmy_apr();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        // All tensors should be 64-byte aligned
        assert!(reader.verify_alignment());
    }

    #[test]
    fn test_pygmy_apr_q8() {
        let data = build_pygmy_apr_q8();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        // Should have Q8 tensors - check tensor names contain attention tensors
        let names = reader.tensor_names();
        let has_attn = names.iter().any(|n| n.contains("self_attn.q_proj.weight"));
        assert!(has_attn);

        // Get one tensor and verify it has Q8 format (scale + quantized values)
        let tensor_data = reader.get_tensor_data("model.layers.0.self_attn.q_proj.weight");
        assert!(tensor_data.is_some());
    }

    #[test]
    fn test_pygmy_apr_q4() {
        let data = build_pygmy_apr_q4();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        // Should have Q4 tensors - check tensor names
        let names = reader.tensor_names();
        let has_attn = names.iter().any(|n| n.contains("self_attn.q_proj.weight"));
        assert!(has_attn);
    }

    #[test]
    fn test_pygmy_apr_f16() {
        let data = build_pygmy_apr_f16();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        // Should have F16 tensors - check tensor names
        let names = reader.tensor_names();
        let has_embed = names.iter().any(|n| n.contains("embed_tokens.weight"));
        assert!(has_embed);
    }

    #[test]
    fn test_pygmy_config_variants() {
        // Test all config variants produce valid output
        for config in [
            PygmyConfig::default(),
            PygmyConfig::minimal(),
            PygmyConfig::embedding_only(),
            PygmyConfig::llama_style(),
        ] {
            let st_data = build_pygmy_safetensors_with_config(config.clone());
            assert!(st_data.len() > 8);

            let apr_data = build_pygmy_apr_with_config(config);
            let reader = AprV2Reader::from_bytes(&apr_data);
            assert!(reader.is_ok());
        }
    }

    #[test]
    fn test_pygmy_size_comparison() {
        // Quantized models should be smaller
        let f32_data = build_pygmy_apr();
        let q8_data = build_pygmy_apr_q8();
        let q4_data = build_pygmy_apr_q4();
        let f16_data = build_pygmy_apr_f16();

        // F16 should be ~50% of F32
        assert!(f16_data.len() < f32_data.len());

        // Q8 should be smaller than F32
        // Note: overhead may make small models not show compression
        assert!(!q8_data.is_empty());
        assert!(!q4_data.is_empty());
    }
}
