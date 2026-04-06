//! APR Format Boundary Tests (Spec 1.5.1)
//!
//! Tests edge cases and boundary conditions for APR format validation.
//! Constructs valid/invalid APR structures with edge-case values.
//!
//! Coverage targets:
//! - Zero dimensions
//! - Massive dimensions
//! - Shape mismatches
//! - Header validation
//! - Tensor entry parsing

use realizar::apr::{AprHeader, AprV2Model, TensorEntry};

/// APR v2 magic number
const APR_V2_MAGIC: u32 = 0x41505232; // "APR2" in little-endian

// ============================================================================
// A. Header Validation Tests
// ============================================================================

#[test]
fn test_header_invalid_magic() {
    let mut data = vec![0u8; 64];
    // Wrong magic
    data[0..4].copy_from_slice(&0x12345678u32.to_le_bytes());
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_header_empty_data() {
    let data: &[u8] = &[];
    let result = AprHeader::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_header_truncated_1_byte() {
    let data = vec![0x32u8]; // Just 'R' from APR2
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_header_truncated_3_bytes() {
    let data = vec![0x32, 0x52, 0x50]; // "2RP" (partial magic)
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_header_magic_only() {
    let data = APR_V2_MAGIC.to_le_bytes().to_vec();
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_header_gguf_magic_rejected() {
    // Try GGUF magic instead of APR
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_header_zero_tensor_count() {
    // Valid header with 0 tensors should be acceptable
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&APR_V2_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
                                                      // May succeed or fail depending on implementation
    let _ = AprHeader::from_bytes(&data);
}

// ============================================================================
// B. Tensor Entry Parsing Tests
// ============================================================================

#[test]
fn test_tensor_entry_empty_data() {
    let data: &[u8] = &[];
    let result = TensorEntry::from_binary(data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_truncated_name_length() {
    // Only 2 bytes (name length needs 4)
    let data = vec![0x05, 0x00];
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_truncated_name() {
    let mut data = vec![0u8; 8];
    data[0..4].copy_from_slice(&10u32.to_le_bytes()); // name length = 10
    data[4..8].copy_from_slice(b"test"); // only 4 bytes of name
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_zero_name_length() {
    let mut data = vec![0u8; 32];
    data[0..4].copy_from_slice(&0u32.to_le_bytes()); // name length = 0
                                                     // This may be valid (empty name) or invalid
    let _ = TensorEntry::from_binary(&data);
}

#[test]
fn test_tensor_entry_huge_name_length() {
    let mut data = vec![0u8; 8];
    data[0..4].copy_from_slice(&u32::MAX.to_le_bytes()); // huge name length
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

// ============================================================================
// C. Model Loading Edge Cases
// ============================================================================

#[test]
fn test_model_from_empty_bytes() {
    let data = vec![];
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_model_from_single_byte() {
    let data = vec![0x00];
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_model_from_random_garbage() {
    let data: Vec<u8> = (0..100).map(|i| (i * 17) as u8).collect();
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_model_from_gguf_data() {
    // GGUF header should be rejected by APR parser
    let mut data = vec![0u8; 100];
    data[0..4].copy_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF magic
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_model_nonexistent_path() {
    let result = AprV2Model::load("/nonexistent/path/to/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_model_directory_path() {
    // Try to load a directory as a model
    let result = AprV2Model::load("/tmp");
    assert!(result.is_err());
}

// ============================================================================
// D. Dimension Boundary Tests
// ============================================================================

/// Helper to create a minimal APR-like header
fn create_apr_header(tensor_count: u32) -> Vec<u8> {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&APR_V2_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
    data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
    data
}

#[test]
fn test_tensor_zero_dimensions() {
    // Tensor with shape [] (scalar)
    let mut data = create_apr_header(1);

    // Tensor entry: name_len(4) + name + ndims(4) + dtype(4) + offset(8) + size(8)
    let name = b"scalar";
    data.extend_from_slice(&(name.len() as u32).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&0u32.to_le_bytes()); // ndims = 0 (scalar)
    data.extend_from_slice(&0u32.to_le_bytes()); // dtype = f32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    data.extend_from_slice(&4u64.to_le_bytes()); // size = 4 bytes (1 float)

    // May or may not be valid
    let _ = AprV2Model::from_bytes(data);
}

#[test]
fn test_tensor_single_dimension_zero() {
    // Tensor with shape [0]
    let mut data = create_apr_header(1);

    let name = b"empty";
    data.extend_from_slice(&(name.len() as u32).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // ndims = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // dim[0] = 0
    data.extend_from_slice(&0u32.to_le_bytes()); // dtype
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    data.extend_from_slice(&0u64.to_le_bytes()); // size = 0

    let _ = AprV2Model::from_bytes(data);
}

#[test]
fn test_tensor_huge_single_dimension() {
    // Tensor with shape [u64::MAX]
    let mut data = create_apr_header(1);

    let name = b"huge";
    data.extend_from_slice(&(name.len() as u32).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // ndims = 1
    data.extend_from_slice(&u64::MAX.to_le_bytes()); // dim[0] = MAX
    data.extend_from_slice(&0u32.to_le_bytes()); // dtype
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    data.extend_from_slice(&0u64.to_le_bytes()); // size

    let result = AprV2Model::from_bytes(data);
    // Should fail due to impossible size
    assert!(result.is_err());
}

#[test]
fn test_tensor_many_dimensions() {
    // Tensor with 10 dimensions
    let mut data = create_apr_header(1);

    let name = b"manydim";
    data.extend_from_slice(&(name.len() as u32).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&10u32.to_le_bytes()); // ndims = 10

    for _ in 0..10 {
        data.extend_from_slice(&2u64.to_le_bytes()); // each dim = 2
    }

    data.extend_from_slice(&0u32.to_le_bytes()); // dtype
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    data.extend_from_slice(&(4 * 1024u64).to_le_bytes()); // size = 2^10 * 4

    // May be valid or rejected
    let _ = AprV2Model::from_bytes(data);
}

// ============================================================================
// E. Data Type Validation Tests
// ============================================================================

#[test]
fn test_tensor_invalid_dtype() {
    let mut data = create_apr_header(1);

    let name = b"badtype";
    data.extend_from_slice(&(name.len() as u32).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // ndims = 1
    data.extend_from_slice(&10u64.to_le_bytes()); // dim[0] = 10
    data.extend_from_slice(&255u32.to_le_bytes()); // invalid dtype = 255
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    data.extend_from_slice(&40u64.to_le_bytes()); // size

    let _ = AprV2Model::from_bytes(data);
}

// ============================================================================
// F. Offset/Size Validation Tests
// ============================================================================

#[test]
fn test_tensor_offset_beyond_file() {
    let mut data = create_apr_header(1);

    let name = b"beyond";
    data.extend_from_slice(&(name.len() as u32).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // ndims
    data.extend_from_slice(&10u64.to_le_bytes()); // dim
    data.extend_from_slice(&0u32.to_le_bytes()); // dtype
    data.extend_from_slice(&u64::MAX.to_le_bytes()); // offset beyond file
    data.extend_from_slice(&40u64.to_le_bytes()); // size

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_size_mismatch() {
    let mut data = create_apr_header(1);

    let name = b"mismatch";
    data.extend_from_slice(&(name.len() as u32).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // ndims = 1
    data.extend_from_slice(&10u64.to_le_bytes()); // dim = 10
    data.extend_from_slice(&0u32.to_le_bytes()); // dtype = f32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    data.extend_from_slice(&100u64.to_le_bytes()); // size = 100 (should be 40)

    // May succeed (size is just a hint) or fail
    let _ = AprV2Model::from_bytes(data);
}

// ============================================================================
// G. Metadata Edge Cases
// ============================================================================

#[test]
fn test_model_methods_with_no_tensors() {
    // Create minimal valid APR with 0 tensors
    let mut data = create_apr_header(0);
    // Add metadata section (empty)
    data.extend_from_slice(&0u32.to_le_bytes()); // metadata_size = 0

    // Align to 64 bytes for tensor data
    while data.len() < 64 {
        data.push(0);
    }

    let result = AprV2Model::from_bytes(data);
    if let Ok(model) = result {
        assert_eq!(model.tensor_count(), 0);
        assert!(model.tensor_names().is_empty());
        assert!(model.get_tensor("nonexistent").is_none());
    }
}

// ============================================================================
// H. UTF-8 Validation in Names
// ============================================================================

#[test]
fn test_tensor_name_invalid_utf8() {
    let mut data = create_apr_header(1);

    // Invalid UTF-8 sequence
    let invalid_name = &[0xFF, 0xFE, 0x00, 0x01];
    data.extend_from_slice(&(invalid_name.len() as u32).to_le_bytes());
    data.extend_from_slice(invalid_name);
    data.extend_from_slice(&1u32.to_le_bytes()); // ndims
    data.extend_from_slice(&10u64.to_le_bytes()); // dim
    data.extend_from_slice(&0u32.to_le_bytes()); // dtype
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    data.extend_from_slice(&40u64.to_le_bytes()); // size

    let result = AprV2Model::from_bytes(data);
    // Should fail on invalid UTF-8
    assert!(result.is_err());
}

#[test]
fn test_tensor_name_with_null_bytes() {
    let mut data = create_apr_header(1);

    let name_with_null = b"test\x00name";
    data.extend_from_slice(&(name_with_null.len() as u32).to_le_bytes());
    data.extend_from_slice(name_with_null);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&10u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&40u64.to_le_bytes());

    // May accept or reject null bytes in name
    let _ = AprV2Model::from_bytes(data);
}

// ============================================================================
// I. BPE Tokenizer Edge Cases
// ============================================================================

#[test]
fn test_tokenizer_encode_empty() {
    // Can't easily test without a real tokenizer, but we can test the path
    let result = AprV2Model::encode_text(std::path::Path::new("/nonexistent"), "");
    // Should return None since tokenizer doesn't exist
    assert!(result.is_none());
}

#[test]
fn test_tokenizer_decode_empty() {
    let vocab: Vec<String> = vec![];
    let result = AprV2Model::decode_tokens(&vocab, &[]);
    assert!(result.is_empty());
}

#[test]
fn test_tokenizer_decode_out_of_bounds() {
    let vocab = vec!["hello".to_string(), "world".to_string()];
    // Token ID 100 is out of bounds
    let result = AprV2Model::decode_tokens(&vocab, &[100]);
    // Should handle gracefully (empty or error token)
    let _ = result;
}

#[test]
fn test_tokenizer_decode_valid() {
    let vocab = vec!["hello".to_string(), " ".to_string(), "world".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "hello world");
}

// ============================================================================
// J. Concurrent Safety (No GPU/Network)
// ============================================================================

#[test]
fn test_header_parsing_is_deterministic() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&APR_V2_MAGIC.to_le_bytes());
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    data[8..12].copy_from_slice(&5u32.to_le_bytes()); // 5 tensors

    // Parse multiple times - should get same result
    let r1 = AprHeader::from_bytes(&data);
    let r2 = AprHeader::from_bytes(&data);

    match (r1, r2) {
        (Ok(h1), Ok(h2)) => {
            assert_eq!(h1.tensor_count, h2.tensor_count);
        },
        (Err(_), Err(_)) => {
            // Both errors is also consistent
        },
        _ => panic!("Inconsistent parsing results"),
    }
}
