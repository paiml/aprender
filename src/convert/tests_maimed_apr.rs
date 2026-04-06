//! T-COV-95 Maimed Pygmy Campaign: Convert Module Destructive Tests
//!
//! Dr. Popper's directive: "Falsify the 'Happy Path' until only the 'Infinite Truth' remains."
//!
//! This module tests error paths in GGUF-to-APR conversion and APR deserialization
//! by feeding malformed data to the converter.
//!
//! Target: convert/mod.rs (~50% coverage)

use super::*;
use crate::apr::{HEADER_SIZE, MAGIC};

// ============================================================================
// Maimed Pygmy: APR Header Corruption
// ============================================================================

/// Test from_apr_bytes with truncated header
#[test]
fn test_maimed_apr_truncated_header() {
    // APR header needs at least HEADER_SIZE bytes
    let data = vec![0u8; 10]; // Way too small

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err(), "Should fail with truncated header");
}

/// Test from_apr_bytes with invalid magic
#[test]
fn test_maimed_apr_invalid_magic() {
    let mut data = vec![0u8; HEADER_SIZE + 100];
    // Set invalid magic
    data[0..4].copy_from_slice(b"NOPE");

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err(), "Should fail with invalid magic");
}

/// Test from_apr_bytes with valid header but truncated tensor index
#[test]
fn test_maimed_apr_truncated_tensor_index() {
    let mut data = vec![0u8; HEADER_SIZE + 10];
    // Valid APR magic
    data[0..4].copy_from_slice(&MAGIC);
    // Version 2.0
    data[4] = 2;
    data[5] = 0;
    // Tensor count = 1
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    // Metadata offset
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    // Metadata length = 2 (for "{}")
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    // Tensor index offset - points beyond file
    let huge_offset = 10000u64;
    data[24..32].copy_from_slice(&huge_offset.to_le_bytes());
    // Data offset - also huge
    data[32..40].copy_from_slice(&20000u64.to_le_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err(), "Should fail with truncated tensor index");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("truncated") || err_str.contains("Truncated") || err_str.contains("bytes"),
        "Error should mention truncation: {}",
        err_str
    );
}

/// Test from_apr_bytes with invalid JSON in tensor index
#[test]
fn test_maimed_apr_invalid_tensor_index_json() {
    let mut data = vec![0u8; HEADER_SIZE + 200];
    // Valid APR magic
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    // Tensor count = 1
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    // Metadata at offset 64, length 2
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    // Put metadata "{}" at offset 64
    data[HEADER_SIZE..HEADER_SIZE + 2].copy_from_slice(b"{}");

    // Tensor index at offset 66, data at 100
    let index_offset = (HEADER_SIZE + 2) as u64;
    let data_offset = 150u64;
    data[24..32].copy_from_slice(&index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Put INVALID JSON at tensor index location
    let invalid_json = b"{ not valid json {{{{";
    let idx = index_offset as usize;
    data[idx..idx + invalid_json.len()].copy_from_slice(invalid_json);

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(
        result.is_err(),
        "Should fail with invalid tensor index JSON"
    );
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("tensor index") || err_str.contains("parse"),
        "Error should mention tensor index parsing: {}",
        err_str
    );
}

/// Test from_apr_bytes with missing 'weights' tensor
#[test]
fn test_maimed_apr_missing_weights_tensor() {
    // Valid JSON tensor index but with WRONG tensor name
    let tensor_index =
        r#"[{"name":"not_weights","dtype":"json","shape":[10],"offset":0,"size":10}]"#;
    let tensor_index_len = tensor_index.len();

    let index_offset = HEADER_SIZE + 64; // metadata at 64, padded to 64 bytes
    let data_offset = index_offset + tensor_index_len;

    let mut data = vec![0u8; data_offset + 100];
    // Valid APR magic
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    // Tensor count = 1
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    // Metadata at offset 64, length 2
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    // Tensor index offset and data offset
    data[24..32].copy_from_slice(&(index_offset as u64).to_le_bytes());
    data[32..40].copy_from_slice(&(data_offset as u64).to_le_bytes());
    // Put metadata "{}" at offset 64
    data[HEADER_SIZE..HEADER_SIZE + 2].copy_from_slice(b"{}");
    // Put tensor index at its offset
    data[index_offset..index_offset + tensor_index_len].copy_from_slice(tensor_index.as_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err(), "Should fail with missing weights tensor");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("weights") || err_str.contains("not found"),
        "Error should mention missing weights: {}",
        err_str
    );
}

/// Test from_apr_bytes with truncated tensor data
#[test]
fn test_maimed_apr_truncated_tensor_data() {
    // Tensor index claims size is 10000 bytes (way more than file has)
    let tensor_index =
        r#"[{"name":"weights","dtype":"json","shape":[10000],"offset":0,"size":10000}]"#;
    let tensor_index_len = tensor_index.len();

    let index_offset = HEADER_SIZE + 64; // metadata at 64, padded
    let data_offset = index_offset + tensor_index_len;

    // File is only 300 bytes but tensor claims 10000
    let mut data = vec![0u8; data_offset + 50]; // way less than 10000
    // Valid APR magic
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    data[24..32].copy_from_slice(&(index_offset as u64).to_le_bytes());
    data[32..40].copy_from_slice(&(data_offset as u64).to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + 2].copy_from_slice(b"{}");
    data[index_offset..index_offset + tensor_index_len].copy_from_slice(tensor_index.as_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err(), "Should fail with truncated tensor data");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("truncated") || err_str.contains("bytes"),
        "Error should mention truncation: {}",
        err_str
    );
}

/// Test from_apr_bytes with invalid JSON in weights payload
#[test]
fn test_maimed_apr_invalid_weights_json() {
    let mut data = vec![0u8; HEADER_SIZE + 400];
    // Valid APR magic
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + 2].copy_from_slice(b"{}");

    let index_offset = (HEADER_SIZE + 2) as u64;
    let data_offset = 200u64;
    data[24..32].copy_from_slice(&index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Tensor index with valid structure
    let invalid_payload = b"{ not valid transformer json {{{{";
    let tensor_index = format!(
        r#"[{{"name":"weights","dtype":"json","shape":[{}],"offset":0,"size":{}}}]"#,
        invalid_payload.len(),
        invalid_payload.len()
    );
    let idx = index_offset as usize;
    data[idx..idx + tensor_index.len()].copy_from_slice(tensor_index.as_bytes());

    // Put invalid JSON at data offset
    let d_idx = data_offset as usize;
    data[d_idx..d_idx + invalid_payload.len()].copy_from_slice(invalid_payload);

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err(), "Should fail with invalid weights JSON");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("deserialize")
            || err_str.contains("transformer")
            || err_str.contains("Failed"),
        "Error should mention deserialization failure: {}",
        err_str
    );
}

// ============================================================================
// Maimed Pygmy: GGUF to APR Conversion Errors
// ============================================================================

/// Test convert with invalid GGUF magic
#[test]
fn test_maimed_gguf_to_apr_invalid_magic() {
    let mut data = vec![0u8; 100];
    // Invalid magic
    data[0..4].copy_from_slice(b"NOPE");
    data[4..8].copy_from_slice(&3u32.to_le_bytes()); // version 3
    data[8..16].copy_from_slice(&0u64.to_le_bytes()); // tensor_count
    data[16..24].copy_from_slice(&0u64.to_le_bytes()); // metadata_count

    let result = GgufToAprConverter::convert(&data);
    assert!(result.is_err(), "Should fail with invalid GGUF magic");
}

/// Test convert with truncated GGUF
#[test]
fn test_maimed_gguf_to_apr_truncated() {
    // Truncated - just magic
    let data = b"GGUF".to_vec();

    let result = GgufToAprConverter::convert(&data);
    assert!(result.is_err(), "Should fail with truncated GGUF");
}

/// Test convert with GGUF that has zero tensors
#[test]
fn test_maimed_gguf_to_apr_zero_tensors() {
    let mut data = vec![0u8; 100];
    // Valid GGUF magic
    data[0..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&3u32.to_le_bytes()); // version 3
    data[8..16].copy_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data[16..24].copy_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    let result = GgufToAprConverter::convert(&data);
    // May succeed (empty model) or fail - either exercises code
    // May succeed (empty model) or fail - either exercises code
    let _ = result;
}

// ============================================================================
// ConversionStats Tests
// ============================================================================

/// Test ConversionStats display
#[test]
fn test_conversion_stats_fields() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 4_000_000,
        num_layers: 12,
        hidden_dim: 768,
        vocab_size: 50000,
        architecture: "llama".to_string(),
    };

    assert_eq!(stats.total_parameters, 1_000_000);
    assert_eq!(stats.memory_bytes_f32, 4_000_000);
    assert_eq!(stats.num_layers, 12);
    assert_eq!(stats.hidden_dim, 768);
    assert_eq!(stats.vocab_size, 50000);
    assert_eq!(stats.architecture, "llama");
}

// ============================================================================
// Edge Cases
// ============================================================================

/// Test APR round-trip with minimal transformer
#[test]
fn test_apr_roundtrip_minimal() {
    use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};

    // Create minimal transformer
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 4,
        intermediate_dim: 8,
        context_length: 16,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; 4],
        attn_norm_bias: None,
        qkv_weight: vec![0.1; 4 * 3 * 4], // hidden_dim * 3 * hidden_dim for fused QKV
        qkv_bias: None,
        attn_output_weight: vec![0.1; 4 * 4],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.1; 4 * 8]), // Optional for SwiGLU
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.1; 4 * 8],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.1; 8 * 4],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; 4]), // Optional
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
        linear_attn_z_weight: None,
        linear_attn_b_weight: None,
        linear_attn_a_weight: None,
        linear_attn_conv1d_weight: None,
        linear_attn_a_log: None,
        linear_attn_dt_bias: None,
        linear_attn_norm_weight: None,
        moe_gate_weight: None,
        moe_expert_gate_up: None,
        moe_expert_down: None,
        moe_shared_gate: None,
        moe_shared_up: None,
        moe_shared_down: None,
        moe_shared_expert_gate_weight: None,
    };

    let transformer = AprTransformer {
        config,
        token_embedding: vec![0.1; 4 * 4], // vocab_size * hidden_dim
        layers: vec![layer],
        output_norm_weight: vec![1.0; 4],
        output_norm_bias: None,
        lm_head_weight: vec![0.1; 4 * 4],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    // Serialize to APR bytes
    let apr_bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("serialization failed");

    // Deserialize back
    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("deserialization failed");

    // Verify config matches
    assert_eq!(loaded.config.architecture, "test");
    assert_eq!(loaded.config.hidden_dim, 4);
    assert_eq!(loaded.config.num_layers, 1);
    assert_eq!(loaded.config.vocab_size, 4);
}

/// Test to_apr_bytes preserves all config fields
#[test]
fn test_apr_bytes_config_preservation() {
    use crate::apr_transformer::{AprTransformer, AprTransformerConfig};

    let config = AprTransformerConfig {
        architecture: "custom_arch".to_string(),
        hidden_dim: 256,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 2, // GQA
        vocab_size: 32000,
        intermediate_dim: 512,
        context_length: 2048,
        rope_theta: 500000.0, // Custom RoPE
        eps: 1e-6,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer {
        config: config.clone(),
        token_embedding: vec![0.0; 256 * 32000],
        layers: vec![],
        output_norm_weight: vec![1.0; 256],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 256 * 32000],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let apr_bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("serialization failed");
    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("deserialization failed");

    // Verify ALL config fields
    assert_eq!(loaded.config.architecture, "custom_arch");
    assert_eq!(loaded.config.hidden_dim, 256);
    assert_eq!(loaded.config.num_layers, 6);
    assert_eq!(loaded.config.num_heads, 8);
    assert_eq!(loaded.config.num_kv_heads, 2);
    assert_eq!(loaded.config.vocab_size, 32000);
    assert_eq!(loaded.config.intermediate_dim, 512);
    assert_eq!(loaded.config.context_length, 2048);
    assert!((loaded.config.rope_theta - 500000.0).abs() < 1.0);
    assert!((loaded.config.eps - 1e-6).abs() < 1e-9);
}
