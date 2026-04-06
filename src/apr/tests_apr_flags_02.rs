//! T-COV-95 Coverage Bridge: apr/mod.rs Part 05
//!
//! Targets uncovered lines: AprFlags methods (is_compressed, is_lz4, is_zstd,
//! is_encrypted, is_quantized, has_vocab), AprHeader::from_bytes edge cases,
//! TensorEntry::from_binary edge cases, TensorEntry::element_count,
//! AprMetadata methods, AprV2Model::from_bytes, decode_tokens,
//! MappedAprModel::dtype_to_qtype, extract_special_tokens_from_vocab.

use std::collections::HashMap;

use crate::apr::*;

// ============================================================================
// AprFlags tests
// ============================================================================

#[test]
fn test_apr_flags_new() {
    let flags = AprFlags::new(0);
    assert!(!flags.is_compressed());
    assert!(!flags.is_lz4());
    assert!(!flags.is_zstd());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_quantized());
    assert!(!flags.has_vocab());
}

#[test]
fn test_apr_flags_lz4() {
    let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
    assert!(flags.is_lz4());
    assert!(flags.is_compressed());
    assert!(!flags.is_zstd());
    assert!(!flags.is_encrypted());
}

#[test]
fn test_apr_flags_zstd() {
    let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
    assert!(flags.is_zstd());
    assert!(flags.is_compressed());
    assert!(!flags.is_lz4());
}

#[test]
fn test_apr_flags_both_compression() {
    let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED | AprFlags::ZSTD_COMPRESSED);
    assert!(flags.is_compressed());
    assert!(flags.is_lz4());
    assert!(flags.is_zstd());
}

#[test]
fn test_apr_flags_encrypted() {
    let flags = AprFlags::new(AprFlags::ENCRYPTED);
    assert!(flags.is_encrypted());
    assert!(!flags.is_compressed());
}

#[test]
fn test_apr_flags_quantized() {
    let flags = AprFlags::new(AprFlags::QUANTIZED);
    assert!(flags.is_quantized());
}

#[test]
fn test_apr_flags_has_vocab() {
    let flags = AprFlags::new(AprFlags::HAS_VOCAB);
    assert!(flags.has_vocab());
}

#[test]
fn test_apr_flags_all_set() {
    let flags = AprFlags::new(
        AprFlags::LZ4_COMPRESSED
            | AprFlags::ZSTD_COMPRESSED
            | AprFlags::ENCRYPTED
            | AprFlags::QUANTIZED
            | AprFlags::HAS_VOCAB
            | AprFlags::SIGNED
            | AprFlags::SHARDED,
    );
    assert!(flags.is_compressed());
    assert!(flags.is_lz4());
    assert!(flags.is_zstd());
    assert!(flags.is_encrypted());
    assert!(flags.is_quantized());
    assert!(flags.has_vocab());
}

#[test]
fn test_apr_flags_default() {
    let flags = AprFlags::default();
    assert!(!flags.is_compressed());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_quantized());
    assert!(!flags.has_vocab());
}

#[test]
fn test_apr_flags_debug() {
    let flags = AprFlags::new(0x0021);
    let debug = format!("{:?}", flags);
    assert!(debug.contains("AprFlags"));
}

#[test]
fn test_apr_flags_clone_copy() {
    let flags = AprFlags::new(0x0005);
    let cloned = flags;
    assert_eq!(cloned.is_encrypted(), flags.is_encrypted());
}

// ============================================================================
// AprHeader::from_bytes tests
// ============================================================================

#[test]
fn test_apr_header_too_small() {
    let data = vec![0u8; 32]; // Less than 64 bytes
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("too small") || err.contains("header"));
}

#[test]
fn test_apr_header_wrong_magic() {
    let mut data = vec![0u8; 64];
    data[0] = 0x00; // Not APR
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("magic") || err.contains("Invalid"));
}

#[test]
fn test_apr_header_v1_rejected() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]); // APR1
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("v1") || err.contains("not supported"));
}

#[test]
fn test_apr_header_v2_accepted() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x32]); // APR2
    data[4] = 2; // version major
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_apr_header_legacy_magic_accepted() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x00]); // APR\0
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_apr_header_invalid_version_byte() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x33]); // APR3 (invalid)
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_apr_header_parses_fields() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // version major
    data[5] = 1; // version minor
    data[6..8].copy_from_slice(&0x0020u16.to_le_bytes()); // flags: QUANTIZED
    data[8..12].copy_from_slice(&10u32.to_le_bytes()); // tensor_count
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&100u32.to_le_bytes()); // metadata_size
    data[24..32].copy_from_slice(&200u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&500u64.to_le_bytes()); // data_offset
    data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes()); // checksum

    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.version, (2, 1));
    assert!(header.flags.is_quantized());
    assert_eq!(header.tensor_count, 10);
    assert_eq!(header.metadata_offset, 64);
    assert_eq!(header.metadata_size, 100);
    assert_eq!(header.tensor_index_offset, 200);
    assert_eq!(header.data_offset, 500);
    assert_eq!(header.checksum, 0xDEADBEEF);
}

#[test]
fn test_apr_header_debug() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&MAGIC);
    let header = AprHeader::from_bytes(&data).expect("should parse");
    let debug = format!("{:?}", header);
    assert!(debug.contains("AprHeader"));
}

#[test]
fn test_apr_header_clone() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&MAGIC);
    let header = AprHeader::from_bytes(&data).expect("should parse");
    let cloned = header.clone();
    assert_eq!(cloned.tensor_count, header.tensor_count);
}

// ============================================================================
// TensorEntry::from_binary tests
// ============================================================================

fn make_tensor_entry_binary(
    name: &str,
    dtype: u8,
    ndim: u8,
    dims: &[u64],
    offset: u64,
    size: u64,
) -> Vec<u8> {
    let mut data = Vec::new();
    let name_bytes = name.as_bytes();
    data.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
    data.extend_from_slice(name_bytes);
    data.push(dtype);
    data.push(ndim);
    for &dim in dims {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&offset.to_le_bytes());
    data.extend_from_slice(&size.to_le_bytes());
    data
}

#[test]
fn test_tensor_entry_from_binary_f32() {
    let data = make_tensor_entry_binary("test.weight", 0, 2, &[100, 64], 0, 25600);
    let (entry, consumed) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.name, "test.weight");
    assert_eq!(entry.dtype, "F32");
    assert_eq!(entry.shape, vec![100, 64]);
    assert_eq!(entry.offset, 0);
    assert_eq!(entry.size, 25600);
    assert_eq!(consumed, data.len());
}

#[test]
fn test_tensor_entry_from_binary_f16() {
    let data = make_tensor_entry_binary("embed", 1, 2, &[32000, 512], 0, 32768000);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "F16");
}

#[test]
fn test_tensor_entry_from_binary_q4k() {
    let data = make_tensor_entry_binary("layer.0.qkv", 12, 2, &[1024, 512], 100, 50000);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q4_K");
}

#[test]
fn test_tensor_entry_from_binary_q6k() {
    let data = make_tensor_entry_binary("lm_head", 14, 2, &[32000, 512], 0, 100000);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q6_K");
}

#[test]
fn test_tensor_entry_from_binary_q8_0() {
    // GH-438: Byte 8 now maps to APR-native "q4" (was GGML "Q8_0" before GH-438)
    let data = make_tensor_entry_binary("layer.0.ffn", 8, 2, &[256, 64], 0, 5000);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "q4");
}

#[test]
fn test_tensor_entry_from_binary_bf16() {
    let data = make_tensor_entry_binary("bf16_tensor", 30, 1, &[1024], 0, 2048);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "BF16");
}

#[test]
fn test_tensor_entry_from_binary_unknown_dtype() {
    let data = make_tensor_entry_binary("unknown", 255, 1, &[100], 0, 400);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "F32"); // Unknown defaults to F32
}

#[test]
fn test_tensor_entry_from_binary_1d() {
    let data = make_tensor_entry_binary("bias", 0, 1, &[512], 0, 2048);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.shape, vec![512]);
}

#[test]
fn test_tensor_entry_from_binary_truncated_too_short() {
    let data = vec![0u8; 3]; // Less than 4 bytes
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_from_binary_truncated_at_name() {
    let data = vec![0x10, 0x00]; // name_len=16 but no name data
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_from_binary_truncated_at_shape() {
    // name_len=1, name="a", dtype=0, ndim=2, but no dim data
    let data = vec![0x01, 0x00, b'a', 0x00, 0x02];
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_from_binary_all_qtypes() {
    let qtypes = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q5_0"),
        (7, "Q5_1"),
        // GH-438: Bytes 8/9 are APR-native types (not GGML Q8_0/Q8_1)
        (8, "q4"),   // APR-native Q4 (legacy ID, collided with GGML Q8_0)
        (9, "q8"),   // APR-native Q8 (legacy ID, collided with GGML Q8_1)
        (10, "Q2_K"),
        (11, "Q3_K"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
        (16, "IQ2_XXS"),
        (17, "IQ2_XS"),
        (30, "BF16"),
    ];

    for (dtype_byte, expected_name) in qtypes {
        let data = make_tensor_entry_binary("t", dtype_byte, 1, &[100], 0, 400);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(
            entry.dtype, expected_name,
            "dtype byte {} should map to {}",
            dtype_byte, expected_name
        );
    }
}

/// GH-438: Test that new APR-native dtype IDs (128, 129) are correctly parsed
#[test]
fn test_tensor_entry_from_binary_apr_native_new_ids() {
    // New canonical IDs for APR-native types (>= 128)
    let data_128 = make_tensor_entry_binary("apr_q4", 128, 1, &[100], 0, 400);
    let (entry, _) = TensorEntry::from_binary(&data_128).expect("should parse");
    assert_eq!(entry.dtype, "q4", "APR-native Q4 (new ID 128)");

    let data_129 = make_tensor_entry_binary("apr_q8", 129, 1, &[100], 0, 400);
    let (entry, _) = TensorEntry::from_binary(&data_129).expect("should parse");
    assert_eq!(entry.dtype, "q8", "APR-native Q8 (new ID 129)");
}

// ============================================================================
// TensorEntry::element_count
// ============================================================================

#[test]
fn test_tensor_entry_element_count_2d() {
    let entry = TensorEntry {
        name: "test".to_string(),
        dtype: "F32".to_string(),
        shape: vec![100, 64],
        offset: 0,
        size: 25600,
    };
    assert_eq!(entry.element_count(), 6400);
}

#[test]
fn test_tensor_entry_element_count_1d() {
    let entry = TensorEntry {
        name: "bias".to_string(),
        dtype: "F32".to_string(),
        shape: vec![512],
        offset: 0,
        size: 2048,
    };
    assert_eq!(entry.element_count(), 512);
}

#[test]
fn test_tensor_entry_element_count_empty_shape() {
    let entry = TensorEntry {
        name: "scalar".to_string(),
        dtype: "F32".to_string(),
        shape: vec![],
        offset: 0,
        size: 4,
    };
    assert_eq!(entry.element_count(), 1); // Product of empty = 1
}

#[test]
fn test_tensor_entry_element_count_3d() {
    let entry = TensorEntry {
        name: "3d".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 3, 4],
        offset: 0,
        size: 96,
    };
    assert_eq!(entry.element_count(), 24);
}

// ============================================================================
// AprMetadata tests
// ============================================================================

#[test]
fn test_apr_metadata_default() {
    let m = AprMetadata::default();
    assert!(m.model_type.is_none());
    assert!(m.name.is_none());
    assert!(m.architecture.is_none());
    assert!(m.hidden_size.is_none());
    assert!(m.num_layers.is_none());
    assert!(m.num_heads.is_none());
    assert!(m.vocab_size.is_none());
}

#[test]
fn test_apr_metadata_is_transformer_true() {
    let m = AprMetadata {
        hidden_size: Some(512),
        num_layers: Some(6),
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(m.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_false_missing_hidden() {
    let m = AprMetadata {
        num_layers: Some(6),
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!m.is_transformer());
}

include!("tests_apr_metadata.rs");
include!("tests_apr_metadata_02.rs");
