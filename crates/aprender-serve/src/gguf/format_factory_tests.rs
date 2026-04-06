//! Comprehensive tests for format_factory.rs
//!
//! These tests target the uncovered 9% of code paths:
//! - SafetensorsBuilder::add_f16_tensor
//! - SafetensorsBuilder::add_bf16_tensor
//! - SafetensorsBuilder::default()
//! - AprBuilder::add_q4_0_tensor
//! - AprBuilder::add_q8_0_tensor
//! - AprBuilder::default()
//! - FormatType::Apr detection with "APR2" magic
//! - Edge cases in format detection

use super::format_factory::{
    create_f32_embedding_data, create_f32_norm_weights, create_q4_0_data, create_q8_0_data,
    AprBuilder, FormatType, SafetensorsBuilder, APR_DTYPE_F16, APR_DTYPE_F32, APR_DTYPE_Q4_0,
    APR_DTYPE_Q8_0,
};

// =============================================================================
// SafetensorsBuilder Default Tests
// =============================================================================

#[test]
fn test_safetensors_builder_default() {
    // Test Default trait implementation
    let builder = SafetensorsBuilder::default();
    let data = builder.build();

    // Should produce valid empty SafeTensors
    assert!(data.len() >= 10);
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
    assert_eq!(header_len, 2); // "{}"
}

#[test]
fn test_safetensors_default_eq_new() {
    // Verify Default::default() and new() produce identical results
    let from_default = SafetensorsBuilder::default().build();
    let from_new = SafetensorsBuilder::new().build();
    assert_eq!(from_default, from_new);
}

// =============================================================================
// SafetensorsBuilder F16 Tensor Tests
// =============================================================================

#[test]
fn test_safetensors_add_f16_tensor() {
    // Create F16 data (2 bytes per element)
    let f16_data: Vec<u8> = [0x00, 0x3C].repeat(16); // 16 F16 values

    let data = SafetensorsBuilder::new()
        .add_f16_tensor("test.f16_weight", &[4, 4], &f16_data)
        .build();

    assert!(data.len() > 10);
    assert_eq!(FormatType::from_magic(&data), FormatType::SafeTensors);

    // Verify JSON header contains F16 dtype
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let json_str = std::str::from_utf8(&data[8..8 + header_len]).expect("valid UTF-8");
    assert!(json_str.contains("test.f16_weight"));
    assert!(json_str.contains("F16"));
}

#[test]
fn test_safetensors_f16_tensor_data_integrity() {
    // Verify raw F16 bytes are preserved
    let f16_data: Vec<u8> = (0..64).collect(); // 32 F16 values with unique bytes
    let shape = [8, 4]; // 32 elements

    let data = SafetensorsBuilder::new()
        .add_f16_tensor("f16_tensor", &shape, &f16_data)
        .build();

    // F16 data should be present after header
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let tensor_start = 8 + header_len;
    let tensor_data = &data[tensor_start..tensor_start + f16_data.len()];
    assert_eq!(tensor_data, &f16_data[..]);
}

#[test]
fn test_safetensors_multiple_f16_tensors() {
    let f16_data1: Vec<u8> = [0x00, 0x3C].repeat(8);
    let f16_data2: Vec<u8> = [0x00, 0x40].repeat(16);

    let data = SafetensorsBuilder::new()
        .add_f16_tensor("layer.0.weight", &[4, 2], &f16_data1)
        .add_f16_tensor("layer.1.weight", &[8, 2], &f16_data2)
        .build();

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let json_str = std::str::from_utf8(&data[8..8 + header_len]).expect("valid UTF-8");

    assert!(json_str.contains("layer.0.weight"));
    assert!(json_str.contains("layer.1.weight"));
    // Both should be F16
    assert_eq!(json_str.matches("F16").count(), 2);
}

// =============================================================================
// SafetensorsBuilder BF16 Tensor Tests
// =============================================================================

#[test]
fn test_safetensors_add_bf16_tensor() {
    // Create BF16 data (2 bytes per element)
    let bf16_data: Vec<u8> = [0x00, 0x3F].repeat(16); // 16 BF16 values (1.0 in BF16)

    let data = SafetensorsBuilder::new()
        .add_bf16_tensor("test.bf16_weight", &[4, 4], &bf16_data)
        .build();

    assert!(data.len() > 10);
    assert_eq!(FormatType::from_magic(&data), FormatType::SafeTensors);

    // Verify JSON header contains BF16 dtype
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let json_str = std::str::from_utf8(&data[8..8 + header_len]).expect("valid UTF-8");
    assert!(json_str.contains("test.bf16_weight"));
    assert!(json_str.contains("BF16"));
}

#[test]
fn test_safetensors_bf16_tensor_data_integrity() {
    // Verify raw BF16 bytes are preserved
    let bf16_data: Vec<u8> = (0..128).collect(); // 64 BF16 values
    let shape = [8, 8];

    let data = SafetensorsBuilder::new()
        .add_bf16_tensor("bf16_tensor", &shape, &bf16_data)
        .build();

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let tensor_start = 8 + header_len;
    let tensor_data = &data[tensor_start..tensor_start + bf16_data.len()];
    assert_eq!(tensor_data, &bf16_data[..]);
}

#[test]
fn test_safetensors_mixed_dtypes() {
    // Test all three dtypes in one file
    let f32_data = vec![1.0f32; 8];
    let f16_data: Vec<u8> = [0x00, 0x3C].repeat(8);
    let bf16_data: Vec<u8> = [0x00, 0x3F].repeat(8);

    let data = SafetensorsBuilder::new()
        .add_f32_tensor("weight.f32", &[2, 4], &f32_data)
        .add_f16_tensor("weight.f16", &[4, 2], &f16_data)
        .add_bf16_tensor("weight.bf16", &[4, 2], &bf16_data)
        .build();

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let json_str = std::str::from_utf8(&data[8..8 + header_len]).expect("valid UTF-8");

    assert!(json_str.contains("\"F32\""));
    assert!(json_str.contains("\"F16\""));
    assert!(json_str.contains("\"BF16\""));
}

// =============================================================================
// AprBuilder Default Tests
// =============================================================================

#[test]
fn test_apr_builder_default() {
    // Test Default trait implementation
    let builder = AprBuilder::default();
    let data = builder.build();

    // Should produce valid empty APR file
    assert!(data.len() >= 64);
    assert_eq!(&data[0..4], b"APR\0");
}

#[test]
fn test_apr_default_eq_new() {
    // Verify Default::default() and new() produce identical results
    let from_default = AprBuilder::default().build();
    let from_new = AprBuilder::new().build();
    assert_eq!(from_default, from_new);
}

// =============================================================================
// AprBuilder Q4_0 Tensor Tests
// =============================================================================

#[test]
fn test_apr_add_q4_0_tensor() {
    let q4_0_data = create_q4_0_data(64);

    let data = AprBuilder::new()
        .add_q4_0_tensor("blk.0.attn_q.weight", &[64, 64], &q4_0_data)
        .build();

    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

    // Verify tensor count in header
    let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
    assert_eq!(tensor_count, 1);
}

#[test]
fn test_apr_q4_0_multiple_tensors() {
    let q4_0_data1 = create_q4_0_data(32);
    let q4_0_data2 = create_q4_0_data(64);

    let data = AprBuilder::new()
        .architecture("llama")
        .add_q4_0_tensor("blk.0.attn_q.weight", &[32, 32], &q4_0_data1)
        .add_q4_0_tensor("blk.0.attn_k.weight", &[64, 64], &q4_0_data2)
        .build();

    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

    let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
    assert_eq!(tensor_count, 2);
}

#[test]
fn test_apr_q4_0_with_f32() {
    // Mix Q4_0 and F32 tensors
    let embed_data = create_f32_embedding_data(10, 8);
    let q4_0_data = create_q4_0_data(64);

    let data = AprBuilder::new()
        .architecture("llama")
        .add_f32_tensor("token_embd.weight", &[10, 8], &embed_data)
        .add_q4_0_tensor("blk.0.ffn_gate.weight", &[64, 64], &q4_0_data)
        .build();

    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

    let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
    assert_eq!(tensor_count, 2);
}

// =============================================================================
// AprBuilder Q8_0 Tensor Tests
// =============================================================================

#[test]
fn test_apr_add_q8_0_tensor() {
    let q8_0_data = create_q8_0_data(64);

    let data = AprBuilder::new()
        .add_q8_0_tensor("blk.0.attn_v.weight", &[64, 64], &q8_0_data)
        .build();

    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

    let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
    assert_eq!(tensor_count, 1);
}

#[test]
fn test_apr_q8_0_multiple_tensors() {
    let q8_0_data1 = create_q8_0_data(32);
    let q8_0_data2 = create_q8_0_data(64);
    let q8_0_data3 = create_q8_0_data(128);

    let data = AprBuilder::new()
        .architecture("llama")
        .hidden_dim(64)
        .add_q8_0_tensor("blk.0.attn_q.weight", &[32, 32], &q8_0_data1)
        .add_q8_0_tensor("blk.0.attn_k.weight", &[64, 64], &q8_0_data2)
        .add_q8_0_tensor("blk.0.attn_v.weight", &[128, 128], &q8_0_data3)
        .build();

    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

    let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
    assert_eq!(tensor_count, 3);
}

#[test]
fn test_apr_mixed_quantization() {
    // Mix all tensor types
    let f32_data = create_f32_norm_weights(64);
    let q4_0_data = create_q4_0_data(64);
    let q8_0_data = create_q8_0_data(64);

    let data = AprBuilder::new()
        .architecture("llama")
        .num_layers(1)
        .add_f32_tensor("output_norm.weight", &[64], &f32_data)
        .add_q4_0_tensor("blk.0.ffn_gate.weight", &[64, 64], &q4_0_data)
        .add_q8_0_tensor("blk.0.ffn_up.weight", &[64, 64], &q8_0_data)
        .build();

    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

    let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
    assert_eq!(tensor_count, 3);
}

// =============================================================================
// FormatType Detection Edge Cases
// =============================================================================

#[test]
fn test_format_detection_apr2_magic() {
    // APR2 magic should also be detected as Apr format
    let mut data = vec![0u8; 100];
    data[0..4].copy_from_slice(b"APR2");
    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);
}

#[test]
fn test_format_detection_exactly_8_bytes() {
    // Exactly 8 bytes - minimum for GGUF/APR detection, not enough for SafeTensors
    let mut data = vec![0u8; 8];
    data[0..4].copy_from_slice(b"GGUF");
    assert_eq!(FormatType::from_magic(&data), FormatType::Gguf);

    data[0..4].copy_from_slice(b"APR\0");
    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

    // Non-magic should be unknown
    data[0..4].copy_from_slice(b"XXXX");
    assert_eq!(FormatType::from_magic(&data), FormatType::Unknown);
}

#[test]
fn test_format_detection_7_bytes() {
    // 7 bytes - too short
    let data = vec![0u8; 7];
    assert_eq!(FormatType::from_magic(&data), FormatType::Unknown);
}

#[test]
fn test_format_detection_empty() {
    // Empty data
    let data: Vec<u8> = vec![];
    assert_eq!(FormatType::from_magic(&data), FormatType::Unknown);
}

#[test]
fn test_format_detection_safetensors_edge_cases() {
    // SafeTensors with exactly 10 bytes (minimum for detection)
    let mut data = vec![0u8; 10];
    // Header length = 2 (for "{}")
    data[0..8].copy_from_slice(&2u64.to_le_bytes());
    data[8..10].copy_from_slice(b"{\"");
    assert_eq!(FormatType::from_magic(&data), FormatType::SafeTensors);
}

#[test]
fn test_format_detection_safetensors_huge_header() {
    // SafeTensors with header length > 100MB should be unknown
    let mut data = vec![0u8; 16];
    // Header length = 200MB
    data[0..8].copy_from_slice(&200_000_000u64.to_le_bytes());
    data[8..10].copy_from_slice(b"{\"");
    // Should be Unknown because header_len >= 100_000_000
    assert_eq!(FormatType::from_magic(&data), FormatType::Unknown);
}

#[test]
fn test_format_detection_safetensors_not_json() {
    // Valid header length but not JSON start
    let mut data = vec![0u8; 12];
    data[0..8].copy_from_slice(&2u64.to_le_bytes());
    data[8..10].copy_from_slice(b"AB"); // Not "{\"
    assert_eq!(FormatType::from_magic(&data), FormatType::Unknown);
}

#[test]
fn test_format_detection_9_bytes() {
    // 9 bytes - enough for GGUF but not SafeTensors
    let mut data = vec![0u8; 9];
    data[0..4].copy_from_slice(b"GGUF");
    assert_eq!(FormatType::from_magic(&data), FormatType::Gguf);

    // 9 bytes is not enough for SafeTensors detection (needs 10)
    data[0..8].copy_from_slice(&2u64.to_le_bytes());
    data[8] = b'{';
    assert_eq!(FormatType::from_magic(&data), FormatType::Unknown);
}

// =============================================================================
// APR Dtype Constants Tests
// =============================================================================

#[test]
fn test_apr_dtype_constants() {
    // Verify dtype constants are correct
    assert_eq!(APR_DTYPE_F32, 0);
    assert_eq!(APR_DTYPE_F16, 1);
    assert_eq!(APR_DTYPE_Q4_0, 2);
    assert_eq!(APR_DTYPE_Q8_0, 8);
}

// =============================================================================
// Builder Chaining Tests
// =============================================================================

#[test]
fn test_safetensors_builder_chaining() {
    // Test fluent API chaining
    let f32_data = vec![1.0f32; 4];
    let f16_data = vec![0u8; 8];
    let bf16_data = vec![0u8; 8];

    let data = SafetensorsBuilder::new()
        .add_f32_tensor("a", &[2, 2], &f32_data)
        .add_f16_tensor("b", &[2, 2], &f16_data)
        .add_bf16_tensor("c", &[2, 2], &bf16_data)
        .add_f32_tensor("d", &[2, 2], &f32_data)
        .build();

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let json_str = std::str::from_utf8(&data[8..8 + header_len]).expect("valid UTF-8");

    // All 4 tensors should be in the header
    assert!(json_str.contains("\"a\""));
    assert!(json_str.contains("\"b\""));
    assert!(json_str.contains("\"c\""));
    assert!(json_str.contains("\"d\""));
}

#[test]
fn test_apr_builder_chaining() {
    // Test fluent API chaining
    let f32_data = create_f32_norm_weights(32);
    let q4_0_data = create_q4_0_data(32);
    let q8_0_data = create_q8_0_data(32);

    let data = AprBuilder::new()
        .architecture("llama")
        .hidden_dim(64)
        .num_layers(2)
        .add_f32_tensor("norm", &[32], &f32_data)
        .add_q4_0_tensor("q4", &[32, 32], &q4_0_data)
        .add_q8_0_tensor("q8", &[32, 32], &q8_0_data)
        .add_f32_tensor("norm2", &[32], &f32_data)
        .build();

    assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

    let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
    assert_eq!(tensor_count, 4);
}

include!("format_factory_tests_safetensors_data.rs");
