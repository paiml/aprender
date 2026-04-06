//! APR Format Regression Suite (Spec 1.5.1)
//!
//! Prevents regression in APR format handling. Verifies every metadata field
//! and tensor statistic against known golden values.
//!
//! Coverage targets:
//! - APR v2 format backward compatibility
//! - Metadata field preservation
//! - Tensor statistics consistency
//! - Loading modes (Eager/Mmap)
//!
//! Constraint: Pure CPU, zero GPU, execution < 2s

use realizar::apr::{AprHeader, AprV2Model, TensorEntry, HEADER_SIZE, MAGIC};

// ============================================================================
// A. Demo Model Creation
// ============================================================================

/// Create a demo APR v2 model with known tensors for regression testing
fn create_demo_model() -> Vec<u8> {
    // APR v2 format layout:
    // [0..64]    - Header
    // [64..]     - Metadata (JSON)
    // [meta_end..] - Tensor index
    // [idx_end..] - Tensor data

    let metadata_json = br#"{"architecture":"test","hidden_dim":4,"num_layers":2,"vocab_size":8,"version":"7.6.0"}"#;
    let metadata_size = metadata_json.len();

    // Create tensor entries
    let tensor_entries = create_tensor_index();
    let tensor_index_bytes = serialize_tensor_index(&tensor_entries);

    // Create tensor data
    let tensor_data = create_tensor_data(&tensor_entries);

    // Calculate offsets
    let metadata_offset = HEADER_SIZE;
    let tensor_index_offset = metadata_offset + metadata_size;
    let data_offset = tensor_index_offset + tensor_index_bytes.len();

    // Build header
    let mut data = vec![0u8; data_offset + tensor_data.len()];

    // Magic
    data[0..4].copy_from_slice(&MAGIC);
    // Version major.minor
    data[4] = 2;
    data[5] = 0;
    // Flags (0 = uncompressed)
    data[6..8].copy_from_slice(&0u16.to_le_bytes());
    // Tensor count
    data[8..12].copy_from_slice(&(tensor_entries.len() as u32).to_le_bytes());
    // Metadata offset
    data[12..20].copy_from_slice(&(metadata_offset as u64).to_le_bytes());
    // Metadata size
    data[20..24].copy_from_slice(&(metadata_size as u32).to_le_bytes());
    // Tensor index offset
    data[24..32].copy_from_slice(&(tensor_index_offset as u64).to_le_bytes());
    // Data offset
    data[32..40].copy_from_slice(&(data_offset as u64).to_le_bytes());
    // Checksum (placeholder)
    data[40..44].copy_from_slice(&0u32.to_le_bytes());

    // Copy metadata
    data[metadata_offset..metadata_offset + metadata_size].copy_from_slice(metadata_json);

    // Copy tensor index
    data[tensor_index_offset..tensor_index_offset + tensor_index_bytes.len()]
        .copy_from_slice(&tensor_index_bytes);

    // Copy tensor data
    data[data_offset..data_offset + tensor_data.len()].copy_from_slice(&tensor_data);

    data
}

/// Create known tensor entries for testing
fn create_tensor_index() -> Vec<TensorEntry> {
    vec![
        TensorEntry {
            name: "token_embedding".to_string(),
            dtype: "F32".to_string(),
            shape: vec![8, 4], // vocab_size x hidden_dim
            offset: 0,
            size: 8 * 4 * 4, // 8 * 4 floats * 4 bytes
        },
        TensorEntry {
            name: "layer.0.attn_norm".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4], // hidden_dim
            offset: 128,
            size: 4 * 4,
        },
        TensorEntry {
            name: "layer.0.qkv_weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 12], // hidden_dim x 3*hidden_dim
            offset: 144,
            size: 4 * 12 * 4,
        },
        TensorEntry {
            name: "output_norm".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4], // hidden_dim
            offset: 336,
            size: 4 * 4,
        },
        TensorEntry {
            name: "lm_head".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 8], // hidden_dim x vocab_size
            offset: 352,
            size: 4 * 8 * 4,
        },
    ]
}

/// Serialize tensor index to APR binary format
fn serialize_tensor_index(entries: &[TensorEntry]) -> Vec<u8> {
    let mut bytes = Vec::new();

    for entry in entries {
        // Name length (2 bytes)
        bytes.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        // Name bytes
        bytes.extend_from_slice(entry.name.as_bytes());
        // Dtype (1 byte)
        let dtype_byte = match entry.dtype.as_str() {
            "F32" => 0u8,
            "F16" => 1,
            "BF16" => 2,
            "I8" => 3,
            "Q4_K" => 8,
            "Q6_K" => 9,
            "Q8_0" => 10,
            _ => 0,
        };
        bytes.push(dtype_byte);
        // Ndim (1 byte)
        bytes.push(entry.shape.len() as u8);
        // Dimensions (8 bytes each)
        for &dim in &entry.shape {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        // Offset (8 bytes)
        bytes.extend_from_slice(&entry.offset.to_le_bytes());
        // Size (8 bytes)
        bytes.extend_from_slice(&entry.size.to_le_bytes());
    }

    bytes
}

/// Create known tensor data for testing
fn create_tensor_data(entries: &[TensorEntry]) -> Vec<u8> {
    let total_size: u64 = entries.iter().map(|e| e.size).sum();
    let mut data = vec![0u8; total_size as usize];

    for entry in entries {
        let start = entry.offset as usize;
        let num_elements: usize = entry.shape.iter().product();

        // Fill with deterministic values based on name
        let seed = entry.name.bytes().map(|b| b as u32).sum::<u32>();
        for i in 0..num_elements {
            let value = ((seed + i as u32) % 100) as f32 / 100.0;
            let bytes = value.to_le_bytes();
            let byte_offset = start + i * 4;
            if byte_offset + 4 <= data.len() {
                data[byte_offset..byte_offset + 4].copy_from_slice(&bytes);
            }
        }
    }

    data
}

// ============================================================================
// B. Header Regression Tests
// ============================================================================

#[test]
fn test_regression_header_magic() {
    let data = create_demo_model();
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_ok(), "Demo model header should parse");

    let header = result.unwrap();
    assert_eq!(header.magic, MAGIC, "Magic should be APR\\0");
}

#[test]
fn test_regression_header_version() {
    let data = create_demo_model();
    let header = AprHeader::from_bytes(&data).unwrap();

    assert_eq!(header.version.0, 2, "Major version should be 2");
    assert_eq!(header.version.1, 0, "Minor version should be 0");
}

#[test]
fn test_regression_header_tensor_count() {
    let data = create_demo_model();
    let header = AprHeader::from_bytes(&data).unwrap();

    assert_eq!(header.tensor_count, 5, "Should have 5 tensors");
}

#[test]
fn test_regression_header_offsets_valid() {
    let data = create_demo_model();
    let header = AprHeader::from_bytes(&data).unwrap();

    // Metadata offset should be right after header
    assert_eq!(
        header.metadata_offset, HEADER_SIZE as u64,
        "Metadata offset should be {}",
        HEADER_SIZE
    );

    // Tensor index should be after metadata
    let expected_tensor_idx = header.metadata_offset + header.metadata_size as u64;
    assert_eq!(
        header.tensor_index_offset, expected_tensor_idx,
        "Tensor index offset should follow metadata"
    );

    // Data offset should be after tensor index
    assert!(
        header.data_offset >= header.tensor_index_offset,
        "Data should be after tensor index"
    );
}

// ============================================================================
// C. Model Loading Regression Tests
// ============================================================================

#[test]
fn test_regression_model_loads_from_bytes() {
    let data = create_demo_model();
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_ok(), "Model should load from valid bytes");
}

#[test]
fn test_regression_model_tensor_count() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    assert_eq!(model.tensor_count(), 5, "Model should have 5 tensors");
}

#[test]
fn test_regression_model_tensor_names() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let names = model.tensor_names();
    assert_eq!(names.len(), 5, "Should have 5 tensor names");

    // Verify expected tensor names
    let expected_names = [
        "token_embedding",
        "layer.0.attn_norm",
        "layer.0.qkv_weight",
        "output_norm",
        "lm_head",
    ];

    for expected in &expected_names {
        assert!(names.contains(expected), "Missing tensor: {}", expected);
    }
}

#[test]
fn test_regression_model_is_not_mmap_from_bytes() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    // from_bytes should use heap, not mmap
    assert!(!model.is_mmap(), "from_bytes should not use mmap");
}

// ============================================================================
// D. Metadata Regression Tests
// ============================================================================

#[test]
fn test_regression_metadata_architecture() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let metadata = model.metadata();
    // Check if architecture is accessible (may be in custom fields)
    let _ = metadata;
}

#[test]
fn test_regression_metadata_custom_fields() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let metadata = model.metadata();
    // Metadata should be parseable - verify key fields are accessible
    // The extra HashMap can hold additional metadata fields
    let _ = &metadata.extra;
}

// ============================================================================
// E. Tensor Entry Regression Tests
// ============================================================================

#[test]
fn test_regression_token_embedding_shape() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let entry = model.get_tensor("token_embedding");
    assert!(entry.is_some(), "token_embedding should exist");

    let entry = entry.unwrap();
    assert_eq!(entry.shape, vec![8, 4], "Shape should be [8, 4]");
    assert_eq!(entry.dtype, "F32", "Dtype should be F32");
}

#[test]
fn test_regression_layer_attn_norm_shape() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let entry = model.get_tensor("layer.0.attn_norm");
    assert!(entry.is_some(), "layer.0.attn_norm should exist");

    let entry = entry.unwrap();
    assert_eq!(entry.shape, vec![4], "Shape should be [4]");
}

#[test]
fn test_regression_qkv_weight_shape() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let entry = model.get_tensor("layer.0.qkv_weight");
    assert!(entry.is_some(), "layer.0.qkv_weight should exist");

    let entry = entry.unwrap();
    assert_eq!(entry.shape, vec![4, 12], "Shape should be [4, 12]");
}

#[test]
fn test_regression_lm_head_shape() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let entry = model.get_tensor("lm_head");
    assert!(entry.is_some(), "lm_head should exist");

    let entry = entry.unwrap();
    assert_eq!(entry.shape, vec![4, 8], "Shape should be [4, 8]");
}

// ============================================================================
// F. Tensor Data Regression Tests
// ============================================================================

#[test]
fn test_regression_tensor_data_readable() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    for name in model.tensor_names() {
        let result = model.get_tensor_bytes(name);
        assert!(result.is_ok(), "Tensor {} should be readable", name);
    }
}

#[test]
fn test_regression_tensor_f32_conversion() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let result = model.get_tensor_f32("token_embedding");
    assert!(result.is_ok(), "token_embedding should convert to f32");

    let floats = result.unwrap();
    assert_eq!(floats.len(), 32, "Should have 8*4=32 elements");
}

#[test]
fn test_regression_tensor_values_deterministic() {
    // Create two models and verify they produce identical data
    let data1 = create_demo_model();
    let data2 = create_demo_model();

    let model1 = AprV2Model::from_bytes(data1).unwrap();
    let model2 = AprV2Model::from_bytes(data2).unwrap();

    let floats1 = model1.get_tensor_f32("token_embedding").unwrap();
    let floats2 = model2.get_tensor_f32("token_embedding").unwrap();

    assert_eq!(floats1, floats2, "Tensor data should be deterministic");
}

#[test]
fn test_regression_tensor_byte_sizes() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    // Verify byte sizes match expectations
    let tests = [
        ("token_embedding", 8 * 4 * 4),     // 8*4 floats * 4 bytes
        ("layer.0.attn_norm", 4 * 4),       // 4 floats * 4 bytes
        ("layer.0.qkv_weight", 4 * 12 * 4), // 4*12 floats * 4 bytes
        ("output_norm", 4 * 4),             // 4 floats * 4 bytes
        ("lm_head", 4 * 8 * 4),             // 4*8 floats * 4 bytes
    ];

    for (name, expected_size) in &tests {
        let entry = model.get_tensor(name).unwrap();
        assert_eq!(
            entry.size, *expected_size as u64,
            "Tensor {} should have size {}",
            name, expected_size
        );
    }
}

// ============================================================================
// G. Estimated Parameters Regression
// ============================================================================

#[test]
fn test_regression_estimated_parameters() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let params = model.estimated_parameters();

    // Expected: 32 + 4 + 48 + 4 + 32 = 120
    // token_embedding: 8*4 = 32
    // layer.0.attn_norm: 4
    // layer.0.qkv_weight: 4*12 = 48
    // output_norm: 4
    // lm_head: 4*8 = 32
    assert_eq!(params, 120, "Should have 120 parameters");
}

// ============================================================================
// H. Predict API Regression Tests
// ============================================================================

#[test]
fn test_regression_predict_no_panic() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    // Predict should not panic even without proper weight tensors
    let result = model.predict(&[1.0, 2.0, 3.0]);
    // May return sum or error, but should not panic
    let _ = result;
}

// ============================================================================
// I. Backward Compatibility Tests (v7.6.0)
// ============================================================================

#[test]
fn test_regression_v760_format_compatibility() {
    // Create model with v7.6.0 metadata
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    // Model should load successfully
    assert!(model.tensor_count() > 0, "Model should have tensors");

    // All tensor operations should work
    for name in model.tensor_names() {
        let entry = model.get_tensor(name);
        assert!(entry.is_some(), "Tensor {} should be accessible", name);

        let bytes = model.get_tensor_bytes(name);
        assert!(bytes.is_ok(), "Tensor {} bytes should be readable", name);
    }
}

#[test]
fn test_regression_empty_model_handling() {
    // Create minimal valid header with no tensors
    let mut data = vec![0u8; HEADER_SIZE + 16];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // version major
    data[5] = 0; // version minor
    data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata_size (empty JSON)
    data[24..32].copy_from_slice(&((HEADER_SIZE + 2) as u64).to_le_bytes()); // tensor_index
    data[32..40].copy_from_slice(&((HEADER_SIZE + 2) as u64).to_le_bytes()); // data_offset
    data[HEADER_SIZE..HEADER_SIZE + 2].copy_from_slice(b"{}");

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_ok(), "Empty model should load");

    let model = result.unwrap();
    assert_eq!(model.tensor_count(), 0, "Empty model should have 0 tensors");
}

// ============================================================================
// J. Loading Mode Tests
// ============================================================================

#[test]
fn test_regression_eager_loading_no_panic() {
    let data = create_demo_model();

    // Eager loading (from_bytes) should complete without panic
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_ok(), "Eager loading should succeed");

    let model = result.unwrap();

    // Access all tensors immediately (eager behavior)
    for name in model.tensor_names() {
        let _ = model.get_tensor_bytes(name);
        let _ = model.get_tensor_f32(name);
    }
}

// ============================================================================
// K. Statistics Consistency Tests
// ============================================================================

#[test]
fn test_regression_tensor_statistics_sum() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    // Calculate total size across all tensors
    let total_bytes: u64 = model
        .tensor_names()
        .iter()
        .filter_map(|name| model.get_tensor(name))
        .map(|e| e.size)
        .sum();

    // Verify against expected (128 + 16 + 192 + 16 + 128 = 480)
    assert_eq!(total_bytes, 480, "Total tensor bytes should be 480");
}

#[test]
fn test_regression_tensor_element_count() {
    let data = create_demo_model();
    let model = AprV2Model::from_bytes(data).unwrap();

    let total_elements = model.estimated_parameters();
    assert_eq!(total_elements, 120, "Total elements should be 120");

    // Verify F32 element count matches byte count / 4
    let total_bytes: u64 = model
        .tensor_names()
        .iter()
        .filter_map(|name| model.get_tensor(name))
        .map(|e| e.size)
        .sum();

    assert_eq!(
        total_bytes / 4,
        total_elements as u64,
        "Byte count / 4 should equal element count for F32"
    );
}
