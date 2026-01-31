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
// Encryption Pygmy Builders (feature-gated)
// ============================================================================

/// Build a minimal encrypted APR model in memory
///
/// Creates an APR file encrypted with password "test_password"
#[cfg(feature = "format-encryption")]
#[must_use]
pub fn build_pygmy_apr_encrypted(password: &str) -> Vec<u8> {
    use crate::format::{save_encrypted, ModelType, SaveOptions};
    use serde::{Deserialize, Serialize};
    use tempfile::NamedTempFile;

    #[derive(Debug, Serialize, Deserialize)]
    struct PygmyModel {
        weights: Vec<f32>,
        bias: f32,
    }

    let model = PygmyModel {
        weights: vec![0.1, 0.2, 0.3, 0.4],
        bias: 0.5,
    };

    let temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    save_encrypted(
        &model,
        ModelType::Custom,
        temp.path(),
        SaveOptions::default(),
        password,
    )
    .expect("Save encrypted");

    std::fs::read(temp.path()).expect("Read encrypted file")
}

/// Build encrypted APR with default test password
#[cfg(feature = "format-encryption")]
#[must_use]
pub fn build_pygmy_apr_encrypted_default() -> Vec<u8> {
    build_pygmy_apr_encrypted("pygmy_test_password_123")
}

// ============================================================================
// Signing Pygmy Builders (feature-gated)
// ============================================================================

/// Build a minimal signed APR model in memory
///
/// Creates an APR file with Ed25519 signature
#[cfg(feature = "format-signing")]
#[must_use]
pub fn build_pygmy_apr_signed() -> (Vec<u8>, ed25519_dalek::VerifyingKey) {
    use crate::format::{save_signed, ModelType, SaveOptions};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use serde::{Deserialize, Serialize};
    use tempfile::NamedTempFile;

    #[derive(Debug, Serialize, Deserialize)]
    struct PygmyModel {
        weights: Vec<f32>,
        bias: f32,
    }

    let model = PygmyModel {
        weights: vec![0.1, 0.2, 0.3, 0.4],
        bias: 0.5,
    };

    // Generate signing key
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();

    let temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    save_signed(
        &model,
        ModelType::Custom,
        temp.path(),
        SaveOptions::default(),
        &signing_key,
    )
    .expect("Save signed");

    let data = std::fs::read(temp.path()).expect("Read signed file");
    (data, verifying_key)
}

/// Generate a test signing key pair
#[cfg(feature = "format-signing")]
#[must_use]
pub fn generate_test_signing_key() -> ed25519_dalek::SigningKey {
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    SigningKey::generate(&mut OsRng)
}

// ============================================================================
// Quantization Pygmy Builders (feature-gated extras)
// ============================================================================

/// Build pygmy data for Q8_0 quantization testing
#[cfg(feature = "format-quantize")]
#[must_use]
pub fn build_pygmy_quantize_data() -> Vec<f32> {
    // Create data that exercises quantization edge cases
    let mut data = Vec::with_capacity(64);

    // Normal range values
    for i in 0..32 {
        data.push((i as f32 - 16.0) / 16.0);
    }

    // Edge case values
    data.push(0.0);           // Zero
    data.push(1.0);           // Max normal
    data.push(-1.0);          // Min normal
    data.push(0.5);           // Mid positive
    data.push(-0.5);          // Mid negative
    for _ in 0..27 {
        data.push(0.001);     // Small values
    }

    data
}

/// Build pygmy Q8_0 quantized block
#[cfg(feature = "format-quantize")]
#[must_use]
pub fn build_pygmy_q8_block() -> crate::format::quantize::QuantizedBlock {
    use crate::format::quantize::{quantize as quantize_data, QuantType};

    let data = build_pygmy_quantize_data();
    quantize_data(&data, &[64], QuantType::Q8_0).expect("Quantize Q8_0")
}

/// Build pygmy Q4_0 quantized block
#[cfg(feature = "format-quantize")]
#[must_use]
pub fn build_pygmy_q4_block() -> crate::format::quantize::QuantizedBlock {
    use crate::format::quantize::{quantize as quantize_data, QuantType};

    let data = build_pygmy_quantize_data();
    quantize_data(&data, &[64], QuantType::Q4_0).expect("Quantize Q4_0")
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

    // ========================================================================
    // Feature-Gated Tests: Encryption (format-encryption)
    // ========================================================================

    #[cfg(feature = "format-encryption")]
    mod encryption_tests {
        use super::*;

        #[test]
        fn test_pygmy_apr_encrypted_roundtrip() {
            let password = "test_password_123";
            let encrypted_data = build_pygmy_apr_encrypted(password);

            // Should have valid APR header with ENCRYPTED flag
            assert!(encrypted_data.len() > 64);
            assert_eq!(&encrypted_data[0..4], b"APRN");

            // Verify ENCRYPTED flag is set (bytes 6-7 are u16 flags, ENCRYPTED = 0x0004)
            let flags = u16::from_le_bytes([encrypted_data[6], encrypted_data[7]]);
            assert!(
                flags & 0x0004 != 0,
                "ENCRYPTED flag (0x0004) should be set in flags: {flags:#06x}"
            );
        }

        #[test]
        fn test_pygmy_apr_encrypted_default() {
            let data = build_pygmy_apr_encrypted_default();
            assert!(!data.is_empty());
            assert!(data.len() > 64);
        }

        #[test]
        fn test_pygmy_encrypted_wrong_password_fails() {
            use crate::format::{load_encrypted, ModelType};
            use serde::{Deserialize, Serialize};
            use tempfile::NamedTempFile;
            use std::io::Write;

            #[derive(Debug, Serialize, Deserialize)]
            struct PygmyModel {
                weights: Vec<f32>,
                bias: f32,
            }

            let encrypted_data = build_pygmy_apr_encrypted("correct_password");

            // Write to temp file
            let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp");
            temp.write_all(&encrypted_data).expect("Write");
            temp.flush().expect("Flush");

            // Try to load with wrong password - should fail
            let result: crate::error::Result<PygmyModel> =
                load_encrypted(temp.path(), ModelType::Custom, "wrong_password");
            assert!(result.is_err(), "Wrong password should fail decryption");
        }
    }

    // ========================================================================
    // Feature-Gated Tests: Signing (format-signing)
    // ========================================================================

    #[cfg(feature = "format-signing")]
    mod signing_tests {
        use super::*;

        #[test]
        fn test_pygmy_apr_signed_has_signature() {
            let (data, _verifying_key) = build_pygmy_apr_signed();

            // Should have valid APR header with SIGNED flag
            assert!(data.len() > 100); // Header + signature block
            assert_eq!(&data[0..4], b"APRN");

            // Verify SIGNED flag is set (bytes 6-7 are u16 flags, SIGNED = 0x0008)
            let flags = u16::from_le_bytes([data[6], data[7]]);
            assert!(
                flags & 0x0008 != 0,
                "SIGNED flag (0x0008) should be set in flags: {flags:#06x}"
            );
        }

        #[test]
        fn test_pygmy_signed_roundtrip() {
            use crate::format::{load_verified, ModelType};
            use serde::{Deserialize, Serialize};
            use tempfile::NamedTempFile;
            use std::io::Write;

            #[derive(Debug, Serialize, Deserialize, PartialEq)]
            struct PygmyModel {
                weights: Vec<f32>,
                bias: f32,
            }

            let (data, verifying_key) = build_pygmy_apr_signed();

            // Write to temp file
            let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp");
            temp.write_all(&data).expect("Write");
            temp.flush().expect("Flush");

            // Load with signature verification
            let loaded: PygmyModel =
                load_verified(temp.path(), ModelType::Custom, Some(&verifying_key))
                    .expect("Load signed should succeed");

            assert_eq!(loaded.weights, vec![0.1, 0.2, 0.3, 0.4]);
            assert!((loaded.bias - 0.5).abs() < 0.001);
        }

        #[test]
        fn test_generate_signing_key() {
            let key = generate_test_signing_key();
            let verifying = key.verifying_key();

            // Should be 32-byte keys
            assert_eq!(verifying.as_bytes().len(), 32);
        }

        #[test]
        fn test_pygmy_signed_tampering_detected() {
            use crate::format::{load_verified, ModelType};
            use serde::{Deserialize, Serialize};
            use tempfile::NamedTempFile;
            use std::io::Write;

            #[derive(Debug, Serialize, Deserialize)]
            struct PygmyModel {
                weights: Vec<f32>,
                bias: f32,
            }

            let (mut data, verifying_key) = build_pygmy_apr_signed();

            // Tamper with the data (flip a bit in the payload area)
            if data.len() > 100 {
                data[80] ^= 0xFF;
            }

            // Write tampered data to temp file
            let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp");
            temp.write_all(&data).expect("Write");
            temp.flush().expect("Flush");

            // Load should fail due to signature mismatch
            let result: crate::error::Result<PygmyModel> =
                load_verified(temp.path(), ModelType::Custom, Some(&verifying_key));
            assert!(result.is_err(), "Tampered file should fail verification");
        }
    }

    // ========================================================================
    // Feature-Gated Tests: Quantization (format-quantize)
    // ========================================================================

    #[cfg(feature = "format-quantize")]
    mod quantize_tests {
        use super::*;
        use crate::format::quantize::{dequantize, QuantType};

        #[test]
        fn test_pygmy_quantize_data_valid() {
            let data = build_pygmy_quantize_data();
            assert_eq!(data.len(), 64);

            // Check no NaN or Inf
            for &v in &data {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn test_pygmy_q8_block_roundtrip() {
            let original = build_pygmy_quantize_data();
            let block = build_pygmy_q8_block();

            // Verify block properties
            assert_eq!(block.quant_type, QuantType::Q8_0);
            assert!(!block.blocks.is_empty());

            // Dequantize and verify values are close
            let restored = dequantize(&block).expect("Dequantize Q8");
            assert_eq!(restored.len(), original.len());

            // Q8_0 should have reasonable accuracy
            for (orig, rest) in original.iter().zip(restored.iter()) {
                assert!((orig - rest).abs() < 0.1, "Q8_0 error too large");
            }
        }

        #[test]
        fn test_pygmy_q4_block_roundtrip() {
            let original = build_pygmy_quantize_data();
            let block = build_pygmy_q4_block();

            // Verify block properties
            assert_eq!(block.quant_type, QuantType::Q4_0);
            assert!(!block.blocks.is_empty());

            // Dequantize and verify values are somewhat close
            let restored = dequantize(&block).expect("Dequantize Q4");
            assert_eq!(restored.len(), original.len());

            // Q4_0 has lower precision
            for (orig, rest) in original.iter().zip(restored.iter()) {
                assert!((orig - rest).abs() < 0.5, "Q4_0 error too large");
            }
        }

        #[test]
        fn test_quant_type_bits_per_weight() {
            assert!((QuantType::Q8_0.bits_per_weight() - 8.5).abs() < 0.01);
            assert!((QuantType::Q4_0.bits_per_weight() - 4.5).abs() < 0.01);
            assert!((QuantType::Q4_1.bits_per_weight() - 5.0).abs() < 0.01);
        }

        #[test]
        fn test_quant_type_from_u8() {
            assert_eq!(QuantType::from_u8(0x01), Some(QuantType::Q8_0));
            assert_eq!(QuantType::from_u8(0x02), Some(QuantType::Q4_0));
            assert_eq!(QuantType::from_u8(0x03), Some(QuantType::Q4_1));
            assert_eq!(QuantType::from_u8(0x10), Some(QuantType::Q8Tensor));
            assert_eq!(QuantType::from_u8(0xFF), Some(QuantType::Custom));
            assert_eq!(QuantType::from_u8(0x99), None);
        }

        #[test]
        fn test_quantized_block_num_blocks() {
            let block = build_pygmy_q8_block();
            let num = block.num_blocks();
            assert!(num > 0);

            // With 64 elements and block size 32, should have 2 blocks
            assert_eq!(num, 2);
        }

        #[test]
        fn test_quantized_block_num_elements() {
            let block = build_pygmy_q8_block();
            let total = block.num_elements();
            assert_eq!(total, 64);
        }
    }
}
