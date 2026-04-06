//! T-COV-95 Active APR Pygmy: Cross-Format Dynamic Falsification
//!
//! This module implements Dr. Popper's "Cross-Format Offensive" -
//! extending the Active Pygmy architecture to the APR format.
//!
//! Key insight: The APR format uses F32 tensors directly (no quantization),
//! making it simpler to create executable models that can survive forward().
//!
//! These tests exercise the entire APR inference pipeline:
//! - Token embedding lookup
//! - RMSNorm layer normalization
//! - QKV projection and attention
//! - FFN gate/up/down (SiLU activation)
//! - Final layer norm and LM head projection

use crate::apr::{AprV2Model, HEADER_SIZE, MAGIC};

// Note: APR v2 tensor entries are variable-size (name + dtype + shape + offset + size)

/// Build an **executable** Pygmy APR model (F32 tensors, minimal dimensions)
///
/// This creates a minimal APR v2 model with:
/// - hidden_size: 8
/// - num_layers: 1
/// - num_heads: 2
/// - num_kv_heads: 2
/// - vocab_size: 10
/// - intermediate_size: 16
///
/// All tensors use F32 format for simplicity.
pub fn build_executable_pygmy_apr() -> Vec<u8> {
    // Active Pygmy dimensions (minimal but valid)
    let hidden_size = 8;
    let num_layers = 1;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 10;
    let intermediate_size = 16;

    let metadata = format!(
        r#"{{
        "architecture": "llama",
        "hidden_size": {hidden_size},
        "num_layers": {num_layers},
        "num_heads": {num_heads},
        "num_kv_heads": {num_kv_heads},
        "vocab_size": {vocab_size},
        "intermediate_size": {intermediate_size},
        "rms_norm_eps": 1e-6
    }}"#
    );
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // GH-317: Contract-compliant tensor names (model.layers.N. prefix for llama)
    // - model.embed_tokens.weight [vocab=10, hidden=8] = 80 floats
    // - model.layers.0.input_layernorm.weight [hidden=8] = 8 floats
    // - model.layers.0.self_attn.q_proj.weight [hidden=8, hidden=8] = 64 floats
    // - model.layers.0.self_attn.k_proj.weight [kv_dim=8, hidden=8] = 64 floats
    // - model.layers.0.self_attn.v_proj.weight [kv_dim=8, hidden=8] = 64 floats
    // - model.layers.0.self_attn.o_proj.weight [hidden=8, hidden=8] = 64 floats
    // - model.layers.0.post_attention_layernorm.weight [hidden=8] = 8 floats
    // - model.layers.0.mlp.gate_proj.weight [inter=16, hidden=8] = 128 floats
    // - model.layers.0.mlp.up_proj.weight [inter=16, hidden=8] = 128 floats
    // - model.layers.0.mlp.down_proj.weight [hidden=8, inter=16] = 128 floats
    // - model.norm.weight [hidden=8] = 8 floats
    // - lm_head.weight [vocab=10, hidden=8] = 80 floats

    let tensor_defs: Vec<(&str, Vec<u64>, usize)> = vec![
        (
            "model.embed_tokens.weight",
            vec![vocab_size as u64, hidden_size as u64],
            vocab_size * hidden_size * 4,
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![intermediate_size as u64, hidden_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![intermediate_size as u64, hidden_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![hidden_size as u64, intermediate_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        ("model.norm.weight", vec![hidden_size as u64], hidden_size * 4),
        (
            "lm_head.weight",
            vec![vocab_size as u64, hidden_size as u64],
            vocab_size * hidden_size * 4,
        ),
    ];

    // Build tensor index entries
    let mut tensor_entries = Vec::new();
    let mut current_offset = 0u64;

    for (name, shape, byte_size) in &tensor_defs {
        let entry = create_tensor_entry(name, 0, shape, current_offset, *byte_size as u64);
        tensor_entries.push(entry);
        current_offset += *byte_size as u64;
    }

    let tensor_index: Vec<u8> = tensor_entries
        .iter()
        .flat_map(|e| e.iter().copied())
        .collect();
    let tensor_count = tensor_defs.len() as u32;
    let total_data_size = current_offset as usize;

    // Calculate offsets
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_index.len() as u64;
    let total_size = data_offset as usize + total_data_size;

    let mut data = vec![0u8; total_size];

    // Write header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // version major
    data[5] = 0; // version minor
    data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Write metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Write tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

    // Write tensor data with small values
    let data_start = data_offset as usize;
    let num_floats = total_data_size / 4;
    for i in 0..num_floats {
        let val = ((i % 10) as f32 - 5.0) * 0.1; // Small values between -0.5 and 0.4
        data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    // Set layernorm weights to 1.0 (they need to be non-zero for proper normalization)
    // Offsets: input_layernorm, post_attention_layernorm, norm.weight
    let norm_offsets = vec![
        vocab_size * hidden_size * 4, // after embed_tokens
        vocab_size * hidden_size * 4 + hidden_size * 4 + 4 * hidden_size * hidden_size * 4, // after o_proj
        vocab_size * hidden_size * 4
            + hidden_size * 4
            + 4 * hidden_size * hidden_size * 4
            + hidden_size * 4
            + 3 * hidden_size * intermediate_size * 4, // norm.weight
    ];

    for offset in norm_offsets {
        for i in 0..hidden_size {
            let val = 1.0f32;
            let pos = data_start + offset + i * 4;
            if pos + 4 <= data.len() {
                data[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
    }

    data
}

/// Create a binary tensor entry for the APR v2 tensor index (variable-size format)
///
/// Format: name_len(u16) + name + dtype(u8) + shape_len(u8) + dims(u64 each) + offset(u64) + size(u64)
fn create_tensor_entry(name: &str, dtype: u8, shape: &[u64], offset: u64, size: u64) -> Vec<u8> {
    let mut data = Vec::new();

    // Name length (u16) + name bytes
    data.extend_from_slice(&(name.len() as u16).to_le_bytes());
    data.extend_from_slice(name.as_bytes());

    // dtype (u8)
    data.push(dtype);

    // shape_len (u8) + dims (u64 each)
    data.push(shape.len() as u8);
    for &dim in shape {
        data.extend_from_slice(&dim.to_le_bytes());
    }

    // offset (u64)
    data.extend_from_slice(&offset.to_le_bytes());

    // size (u64)
    data.extend_from_slice(&size.to_le_bytes());

    data
}

// ============================================================================
// Active APR Pygmy Tests - Cross-Format Dynamic Falsification
// ============================================================================

#[test]
fn test_active_apr_pygmy_parses() {
    let data = build_executable_pygmy_apr();
    let model = AprV2Model::from_bytes(data);

    assert!(
        model.is_ok(),
        "Active APR Pygmy should parse: {:?}",
        model.err()
    );

    let model = model.unwrap();
    assert!(model.metadata().is_transformer());
    assert_eq!(model.metadata().hidden_size, Some(8));
    assert_eq!(model.metadata().num_layers, Some(1));
    assert_eq!(model.metadata().vocab_size, Some(10));
}

#[test]
fn test_active_apr_pygmy_has_all_tensors() {
    let data = build_executable_pygmy_apr();
    let model = AprV2Model::from_bytes(data).expect("Should parse");

    // GH-317: Check all required tensors exist (contract-compliant names)
    assert!(
        model.get_tensor("model.embed_tokens.weight").is_some(),
        "Missing embed_tokens"
    );
    assert!(
        model
            .get_tensor("model.layers.0.input_layernorm.weight")
            .is_some(),
        "Missing input_layernorm"
    );
    assert!(
        model
            .get_tensor("model.layers.0.self_attn.q_proj.weight")
            .is_some(),
        "Missing q_proj"
    );
    assert!(
        model
            .get_tensor("model.layers.0.self_attn.k_proj.weight")
            .is_some(),
        "Missing k_proj"
    );
    assert!(
        model
            .get_tensor("model.layers.0.self_attn.v_proj.weight")
            .is_some(),
        "Missing v_proj"
    );
    assert!(
        model
            .get_tensor("model.layers.0.self_attn.o_proj.weight")
            .is_some(),
        "Missing o_proj"
    );
    assert!(
        model
            .get_tensor("model.layers.0.post_attention_layernorm.weight")
            .is_some(),
        "Missing post_attention_layernorm"
    );
    assert!(
        model.get_tensor("model.layers.0.mlp.gate_proj.weight").is_some(),
        "Missing gate_proj"
    );
    assert!(
        model.get_tensor("model.layers.0.mlp.up_proj.weight").is_some(),
        "Missing up_proj"
    );
    assert!(
        model.get_tensor("model.layers.0.mlp.down_proj.weight").is_some(),
        "Missing down_proj"
    );
    assert!(model.get_tensor("model.norm.weight").is_some(), "Missing norm");
    assert!(
        model.get_tensor("lm_head.weight").is_some(),
        "Missing lm_head"
    );
}

/// T-COV-95 CRITICAL: Test full APR forward() execution
#[test]
fn test_active_apr_pygmy_forward() {
    let data = build_executable_pygmy_apr();
    let model = AprV2Model::from_bytes(data).expect("Should parse");

    // Run forward pass with a single token
    let result = model.forward(&[1]);

    assert!(result.is_ok(), "forward() failed: {:?}", result.err());

    let logits = result.unwrap();

    // Verify logits shape (vocab_size = 10)
    assert_eq!(logits.len(), 10, "Logits should have vocab_size elements");

    // Verify logits are finite
    assert!(
        logits.iter().all(|&v| v.is_finite()),
        "Logits contain NaN/Inf"
    );
}

#[test]
fn test_active_apr_pygmy_forward_multiple_tokens() {
    let data = build_executable_pygmy_apr();
    let model = AprV2Model::from_bytes(data).expect("Should parse");

    // Run forward pass with multiple tokens (simulating prefill)
    let result = model.forward(&[1, 2, 3]);

    assert!(
        result.is_ok(),
        "forward() with multiple tokens failed: {:?}",
        result.err()
    );

    let logits = result.unwrap();
    assert_eq!(logits.len(), 10);
    assert!(logits.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_active_apr_pygmy_edge_tokens() {
    let data = build_executable_pygmy_apr();
    let model = AprV2Model::from_bytes(data).expect("Should parse");

    // Test token 0 (first valid)
    let result0 = model.forward(&[0]);
    assert!(result0.is_ok(), "Token 0 should work");

    // Test token 9 (last valid, vocab_size=10)
    let result9 = model.forward(&[9]);
    assert!(result9.is_ok(), "Token 9 should work");
}

#[test]
fn test_active_apr_pygmy_generate() {
    let data = build_executable_pygmy_apr();
    let model = AprV2Model::from_bytes(data).expect("Should parse");

    // Test generate with small max_tokens
    let result = model.generate(&[1], 3, None);

    assert!(result.is_ok(), "generate() failed: {:?}", result.err());

    let tokens = result.unwrap();
    // Should have at least the original token plus some generated
    assert!(!tokens.is_empty());
}

#[test]
fn test_active_apr_pygmy_size() {
    let data = build_executable_pygmy_apr();

    // Should be small (Active Pygmy property)
    assert!(
        data.len() < 10_000,
        "APR Pygmy should be < 10KB, got {} bytes",
        data.len()
    );

    // But non-trivial
    assert!(
        data.len() > 500,
        "APR Pygmy should be > 500 bytes, got {} bytes",
        data.len()
    );
}

#[test]
fn test_active_apr_pygmy_metadata() {
    let data = build_executable_pygmy_apr();
    let model = AprV2Model::from_bytes(data).expect("Should parse");

    let meta = model.metadata();
    assert_eq!(meta.architecture, Some("llama".to_string()));
    assert_eq!(meta.hidden_size, Some(8));
    assert_eq!(meta.num_layers, Some(1));
    assert_eq!(meta.num_heads, Some(2));
    assert_eq!(meta.num_kv_heads, Some(2));
    assert_eq!(meta.vocab_size, Some(10));
    assert_eq!(meta.intermediate_size, Some(16));
}
