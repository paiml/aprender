//! GGUF Error Path Fuzzing Tests (Spec 1.5.1)
//!
//! Systematic testing of all error paths in the GGUF parser.
//! Uses malformed byte sequences to trigger every `return Err(...)`.
//!
//! Coverage targets:
//! - InvalidMagic: Wrong magic bytes
//! - UnsupportedVersion: Non-v3 versions
//! - TruncatedHeader: Incomplete header reads
//! - InvalidTensorMapping: Bad tensor info
//! - InvalidMetadata: Malformed metadata values

use realizar::gguf::GGUFModel;

/// GGUF magic number
const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION_V3: u32 = 3;

// ============================================================================
// A. Invalid Magic Tests
// ============================================================================

#[test]
fn test_invalid_magic_all_zeros() {
    let data = vec![0u8; 24];
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Invalid GGUF magic") || err.contains("magic"));
}

#[test]
fn test_invalid_magic_wrong_bytes() {
    // "FUGG" instead of "GGUF"
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&[0x47, 0x47, 0x55, 0x46]); // Wrong order
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_invalid_magic_ggml_format() {
    // Old GGML magic (not GGUF)
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(b"GGML");
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_invalid_magic_partial() {
    // Just "GGU" - missing last byte
    let data = b"GGU";
    let result = GGUFModel::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_invalid_magic_random_bytes() {
    let malicious_magics: &[&[u8]] = &[
        &[0xFF, 0xFF, 0xFF, 0xFF],
        &[0x00, 0x00, 0x00, 0x01],
        &[0x89, 0x50, 0x4E, 0x47], // PNG magic
        &[0x50, 0x4B, 0x03, 0x04], // ZIP magic
        &[0x7F, 0x45, 0x4C, 0x46], // ELF magic
    ];

    for magic in malicious_magics {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(magic);
        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err(), "Should reject magic: {:?}", magic);
    }
}

// ============================================================================
// B. Unsupported Version Tests
// ============================================================================

#[test]
fn test_unsupported_version_v1() {
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&1u32.to_le_bytes()); // Version 1
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("version") || err.contains("Unsupported"),
        "Error should mention version: {}",
        err
    );
}

#[test]
fn test_unsupported_version_v2() {
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&2u32.to_le_bytes()); // Version 2
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_unsupported_version_v4_future() {
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&4u32.to_le_bytes()); // Future version 4
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_unsupported_version_max() {
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_unsupported_version_zero() {
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&0u32.to_le_bytes()); // Version 0
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// C. Truncated Header Tests
// ============================================================================

#[test]
fn test_truncated_header_empty() {
    let data: &[u8] = &[];
    let result = GGUFModel::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_truncated_header_1_byte() {
    let data = vec![0x47u8]; // Just 'G'
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_truncated_header_3_bytes() {
    let data = vec![0x47, 0x47, 0x55]; // "GGU"
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_truncated_header_magic_only() {
    // Valid magic but nothing else
    let data = GGUF_MAGIC.to_le_bytes().to_vec();
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_truncated_header_no_counts() {
    // Magic + version but no tensor/metadata counts
    let mut data = vec![0u8; 8];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_truncated_header_partial_tensor_count() {
    // Magic + version + partial tensor count
    let mut data = vec![0u8; 12];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    // Only 4 bytes of 8-byte tensor count
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_truncated_header_no_metadata_count() {
    // Magic + version + tensor_count but no metadata_count
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data[8..16].copy_from_slice(&0u64.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// D. Invalid Tensor/Metadata Count Tests
// ============================================================================

#[test]
fn test_huge_tensor_count_no_data() {
    // Valid header claiming 1 million tensors but no tensor data
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data[8..16].copy_from_slice(&1_000_000u64.to_le_bytes()); // 1M tensors
    data[16..24].copy_from_slice(&0u64.to_le_bytes()); // 0 metadata
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_huge_metadata_count_no_data() {
    // Valid header claiming 1 million metadata entries but no data
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data[8..16].copy_from_slice(&0u64.to_le_bytes()); // 0 tensors
    data[16..24].copy_from_slice(&1_000_000u64.to_le_bytes()); // 1M metadata
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_max_u64_tensor_count() {
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    data[16..24].copy_from_slice(&0u64.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// E. Malformed Metadata Tests
// ============================================================================

/// Helper to create a minimal valid header
fn create_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data[8..16].copy_from_slice(&tensor_count.to_le_bytes());
    data[16..24].copy_from_slice(&metadata_count.to_le_bytes());
    data
}

#[test]
fn test_metadata_truncated_key_length() {
    // Header says 1 metadata, but key length is truncated
    let mut data = create_header(0, 1);
    // Append only 4 bytes (key length is u64, needs 8)
    data.extend_from_slice(&[0x05, 0x00, 0x00, 0x00]);
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_metadata_truncated_key_string() {
    // Header says 1 metadata, key length says 10, but only 5 bytes provided
    let mut data = create_header(0, 1);
    data.extend_from_slice(&10u64.to_le_bytes()); // key length = 10
    data.extend_from_slice(b"hello"); // only 5 bytes
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_metadata_invalid_value_type() {
    // Valid key, but invalid value type (255)
    let mut data = create_header(0, 1);
    let key = b"test_key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&255u32.to_le_bytes()); // Invalid type
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_metadata_string_value_truncated() {
    // Valid key, string type, but string value truncated
    let mut data = create_header(0, 1);
    let key = b"test_key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&8u32.to_le_bytes()); // Type 8 = STRING
    data.extend_from_slice(&100u64.to_le_bytes()); // String length = 100
    data.extend_from_slice(b"short"); // Only 5 bytes
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_metadata_array_truncated() {
    // Valid key, array type, but array elements truncated
    let mut data = create_header(0, 1);
    let key = b"array_key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&9u32.to_le_bytes()); // Type 9 = ARRAY
    data.extend_from_slice(&4u32.to_le_bytes()); // Element type = U32
    data.extend_from_slice(&1000u64.to_le_bytes()); // Array length = 1000
                                                    // No array data
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// F. Malformed Tensor Info Tests
// ============================================================================

#[test]
fn test_tensor_info_truncated_name() {
    // Header says 1 tensor, but tensor name is truncated
    let mut data = create_header(1, 0);
    data.extend_from_slice(&10u64.to_le_bytes()); // name length = 10
    data.extend_from_slice(b"short"); // only 5 bytes
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_info_truncated_dimensions() {
    // Valid tensor name, but n_dims is truncated
    let mut data = create_header(1, 0);
    let name = b"weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    // n_dims needs 4 bytes, only provide 2
    data.extend_from_slice(&[0x02, 0x00]);
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_info_invalid_qtype() {
    // Valid tensor header but invalid quantization type
    let mut data = create_header(1, 0);
    let name = b"weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    data.extend_from_slice(&64u64.to_le_bytes()); // dim[0] = 64
    data.extend_from_slice(&64u64.to_le_bytes()); // dim[1] = 64
    data.extend_from_slice(&255u32.to_le_bytes()); // Invalid qtype = 255
    data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
    let result = GGUFModel::from_bytes(&data);
    // This may or may not error depending on how qtype is validated
    // But we want to ensure no panic
    let _ = result;
}

#[test]
fn test_tensor_info_zero_dimensions() {
    // Tensor with 0 dimensions (scalar) - edge case
    let mut data = create_header(1, 0);
    let name = b"scalar";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&0u32.to_le_bytes()); // n_dims = 0
    data.extend_from_slice(&0u32.to_le_bytes()); // qtype = F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
                                                 // This is technically valid for a scalar
    let _ = GGUFModel::from_bytes(&data);
}

#[test]
fn test_tensor_info_excessive_dimensions() {
    // Tensor claiming 100 dimensions - should fail or handle gracefully
    let mut data = create_header(1, 0);
    let name = b"huge_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&100u32.to_le_bytes()); // n_dims = 100
                                                   // Not enough dimension data
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// G. Byte-Level Corruption Tests
// ============================================================================

#[test]
fn test_single_bit_flip_in_magic() {
    for bit in 0..32 {
        let mut data = vec![0u8; 24];
        let corrupted_magic = GGUF_MAGIC ^ (1 << bit);
        data[0..4].copy_from_slice(&corrupted_magic.to_le_bytes());
        data[4..8].copy_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err(), "Bit flip {} should cause error", bit);
    }
}

#[test]
fn test_null_bytes_injection() {
    // Valid header but with null bytes injected in metadata key
    let mut data = create_header(0, 1);
    let key_with_nulls = b"test\x00\x00key";
    data.extend_from_slice(&(key_with_nulls.len() as u64).to_le_bytes());
    data.extend_from_slice(key_with_nulls);
    data.extend_from_slice(&4u32.to_le_bytes()); // Type = U32
    data.extend_from_slice(&42u32.to_le_bytes()); // Value
                                                  // Should either parse (null bytes in key) or error gracefully
    let _ = GGUFModel::from_bytes(&data);
}

#[test]
fn test_invalid_utf8_in_key() {
    // Metadata key with invalid UTF-8 sequences
    let mut data = create_header(0, 1);
    let invalid_utf8 = &[0xFF, 0xFE, 0x00, 0x01];
    data.extend_from_slice(&(invalid_utf8.len() as u64).to_le_bytes());
    data.extend_from_slice(invalid_utf8);
    data.extend_from_slice(&4u32.to_le_bytes()); // Type = U32
    data.extend_from_slice(&42u32.to_le_bytes()); // Value
    let result = GGUFModel::from_bytes(&data);
    // Should error on invalid UTF-8
    assert!(result.is_err());
}

// ============================================================================
// H. Edge Case Alignment Tests
// ============================================================================

#[test]
fn test_misaligned_tensor_offset() {
    // Tensor with offset that's not 32-byte aligned
    let mut data = create_header(1, 0);
    let name = b"misaligned";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
    data.extend_from_slice(&64u64.to_le_bytes()); // dim[0] = 64
    data.extend_from_slice(&0u32.to_le_bytes()); // qtype = F32
    data.extend_from_slice(&17u64.to_le_bytes()); // offset = 17 (not aligned)
                                                  // Should either handle or error - no panic
    let _ = GGUFModel::from_bytes(&data);
}

// ============================================================================
// I. Stress Tests (still fast, but more thorough)
// ============================================================================

#[test]
fn test_many_small_corruptions() {
    let base = create_header(0, 0);

    // Test corruption at each byte position
    for pos in 0..base.len() {
        for corrupt_byte in [0x00, 0xFF, 0x42] {
            let mut data = base.clone();
            data[pos] = corrupt_byte;
            // Should not panic
            let _ = GGUFModel::from_bytes(&data);
        }
    }
}

#[test]
fn test_random_length_truncations() {
    let full_header = create_header(0, 0);

    // Test all truncation lengths
    for len in 0..full_header.len() {
        let truncated = &full_header[..len];
        let result = GGUFModel::from_bytes(truncated);
        if len < 24 {
            assert!(result.is_err(), "Truncated to {} bytes should fail", len);
        }
    }
}
