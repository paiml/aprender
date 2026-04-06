//! T-COV-95 Coverage Bridge: apr/mod.rs Part 06
//!
//! Targets uncovered lines: AprV2Model from_bytes edge cases,
//! from_model_data truncation, predict(), get_tensor_f32/get_tensor_bytes,
//! estimated_parameters, tensor_names, get_tensor, is_mmap,
//! AprV2Model::from_bytes with compressed flag (no feature),
//! from_model_data with tensor index.

use crate::apr::*;

// ============================================================================
// Helper: Build a valid APR v2 byte buffer with tensors
// ============================================================================

/// Construct a minimal APR v2 file in memory with optional tensors and metadata.
fn build_apr_bytes(
    metadata_json: &str,
    tensor_entries: &[(String, u8, Vec<usize>, Vec<u8>)], // (name, dtype_byte, shape, data)
) -> Vec<u8> {
    let meta_bytes = metadata_json.as_bytes();

    // Header is 64 bytes
    // Metadata starts at offset 64
    let metadata_offset = HEADER_SIZE as u64;
    let metadata_size = meta_bytes.len() as u32;

    // Tensor index starts right after metadata (padded to 64-byte boundary)
    let meta_padded = ((meta_bytes.len() + 63) / 64) * 64;
    let tensor_index_offset = (HEADER_SIZE + meta_padded) as u64;

    // Build tensor index binary data
    let mut index_data = Vec::new();
    let mut tensor_data_parts: Vec<&[u8]> = Vec::new();
    let mut current_data_offset: u64 = 0;

    for (name, dtype_byte, shape, data) in tensor_entries {
        let name_bytes = name.as_bytes();
        index_data.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_data.extend_from_slice(name_bytes);
        index_data.push(*dtype_byte);
        index_data.push(shape.len() as u8);
        for &dim in shape {
            index_data.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        index_data.extend_from_slice(&current_data_offset.to_le_bytes());
        index_data.extend_from_slice(&(data.len() as u64).to_le_bytes());
        current_data_offset += data.len() as u64;
        tensor_data_parts.push(data);
    }

    // Data section starts after tensor index (padded)
    let index_padded = ((index_data.len() + 63) / 64) * 64;
    let data_offset = tensor_index_offset + index_padded as u64;

    // Build complete file
    let total_size = HEADER_SIZE
        + meta_padded
        + index_padded
        + tensor_data_parts.iter().map(|d| d.len()).sum::<usize>();
    let mut file = vec![0u8; total_size];

    // Header
    file[0..4].copy_from_slice(&MAGIC);
    file[4] = 2; // version major
    file[5] = 0; // version minor
                 // flags = 0 (no compression, no encryption)
    file[8..12].copy_from_slice(&(tensor_entries.len() as u32).to_le_bytes());
    file[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
    file[20..24].copy_from_slice(&metadata_size.to_le_bytes());
    file[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    file[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Metadata
    file[HEADER_SIZE..HEADER_SIZE + meta_bytes.len()].copy_from_slice(meta_bytes);

    // Tensor index
    let idx_start = tensor_index_offset as usize;
    file[idx_start..idx_start + index_data.len()].copy_from_slice(&index_data);

    // Tensor data
    let data_start = data_offset as usize;
    let mut offset = 0;
    for data in &tensor_data_parts {
        file[data_start + offset..data_start + offset + data.len()].copy_from_slice(data);
        offset += data.len();
    }

    file
}

// ============================================================================
// AprV2Model::from_bytes additional tests
// ============================================================================

#[test]
fn test_from_bytes_with_tensor_index() {
    // Create a model with 2 tensors
    let weight_data = vec![0u8; 64 * 4]; // 64 F32 values
    let bias_data = vec![0u8; 4 * 4]; // 4 F32 values

    let data = build_apr_bytes(
        "{}",
        &[
            ("weight".to_string(), 0, vec![4, 16], weight_data),
            ("bias".to_string(), 0, vec![4], bias_data),
        ],
    );

    let model = AprV2Model::from_bytes(data).expect("should parse");
    assert_eq!(model.tensor_count(), 2);
    let names = model.tensor_names();
    assert!(names.contains(&"weight"));
    assert!(names.contains(&"bias"));
}

#[test]
fn test_from_bytes_get_tensor_by_name() {
    let weight_data = vec![0u8; 32 * 4]; // 32 F32 values

    let data = build_apr_bytes(
        "{}",
        &[("test.weight".to_string(), 0, vec![4, 8], weight_data)],
    );

    let model = AprV2Model::from_bytes(data).expect("should parse");
    let tensor = model.get_tensor("test.weight");
    assert!(tensor.is_some());
    let entry = tensor.unwrap();
    assert_eq!(entry.name, "test.weight");
    assert_eq!(entry.dtype, "F32");
    assert_eq!(entry.shape, vec![4, 8]);

    // Non-existent tensor
    assert!(model.get_tensor("nonexistent").is_none());
}

#[test]
fn test_from_bytes_metadata_parsing() {
    let data = build_apr_bytes(
        r#"{"hidden_size": 256, "num_layers": 4, "num_heads": 8, "vocab_size": 1000}"#,
        &[],
    );

    let model = AprV2Model::from_bytes(data).expect("should parse");
    let meta = model.metadata();
    assert!(meta.is_transformer());
    assert_eq!(meta.hidden_size, Some(256));
    assert_eq!(meta.num_layers, Some(4));
    assert_eq!(meta.num_heads, Some(8));
    assert_eq!(meta.vocab_size, Some(1000));
}

#[test]
fn test_from_bytes_compressed_without_feature() {
    // Set compression flag but no apr-compression feature
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    // Set LZ4 flag
    data[6..8].copy_from_slice(&AprFlags::LZ4_COMPRESSED.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&0u32.to_le_bytes());
    data[24..32].copy_from_slice(&64u64.to_le_bytes());
    data[32..40].copy_from_slice(&64u64.to_le_bytes());

    let result = AprV2Model::from_bytes(data);
    // Without apr-compression feature, this should error
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("compress") || err.contains("feature"),
        "Error should mention compression: {err}"
    );
}

#[test]
fn test_from_bytes_truncated_metadata() {
    let mut data = vec![0u8; 64]; // Only header, no room for metadata
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset = 64
    data[20..24].copy_from_slice(&100u32.to_le_bytes()); // metadata_size = 100 (but file only 64 bytes)
    data[24..32].copy_from_slice(&200u64.to_le_bytes());
    data[32..40].copy_from_slice(&200u64.to_le_bytes());

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("truncated") || err.contains("metadata"));
}

#[test]
fn test_from_bytes_zero_metadata_size() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
    data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

    let model = AprV2Model::from_bytes(data).expect("should parse with empty metadata");
    assert_eq!(model.tensor_count(), 0);
    // Metadata should be default
    assert!(!model.metadata().is_transformer());
}

// ============================================================================
// estimated_parameters
// ============================================================================

#[test]
fn test_estimated_parameters_empty() {
    let data = build_apr_bytes("{}", &[]);
    let model = AprV2Model::from_bytes(data).expect("should parse");
    assert_eq!(model.estimated_parameters(), 0);
}

#[test]
fn test_estimated_parameters_with_tensors() {
    let weight_data = vec![0u8; 512 * 4]; // 512 F32 values
    let bias_data = vec![0u8; 64 * 4]; // 64 F32 values

    let data = build_apr_bytes(
        "{}",
        &[
            ("w".to_string(), 0, vec![32, 16], weight_data), // 32*16 = 512 params
            ("b".to_string(), 0, vec![64], bias_data),       // 64 params
        ],
    );

    let model = AprV2Model::from_bytes(data).expect("should parse");
    assert_eq!(model.estimated_parameters(), 512 + 64);
}

// ============================================================================
// is_mmap
// ============================================================================

#[test]
fn test_is_mmap_from_bytes() {
    let data = build_apr_bytes("{}", &[]);
    let model = AprV2Model::from_bytes(data).expect("should parse");
    // Models loaded from bytes should NOT be mmap
    assert!(!model.is_mmap());
}

// ============================================================================
// predict() - simple linear model
// ============================================================================

#[test]
fn test_predict_no_tensors() {
    let data = build_apr_bytes("{}", &[]);
    let model = AprV2Model::from_bytes(data).expect("should parse");

    // With no tensors, predict returns sum of features
    let result = model
        .predict(&[1.0, 2.0, 3.0])
        .expect("predict should work");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 6.0).abs() < 1e-6);
}

#[test]
fn test_predict_no_tensors_empty_features() {
    let data = build_apr_bytes("{}", &[]);
    let model = AprV2Model::from_bytes(data).expect("should parse");

    let result = model.predict(&[]).expect("predict should work");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_predict_with_weight_tensor() {
    // Create a 2x3 F32 weight matrix: [[1,0,0],[0,1,0]]
    // Result: y = W * x for x=[1,2,3] -> y=[1,2]
    let mut weight_data = vec![0u8; 6 * 4]; // 6 floats
    // Row 0: [1.0, 0.0, 0.0]
    weight_data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
    // Row 1: [0.0, 1.0, 0.0]
    weight_data[16..20].copy_from_slice(&1.0f32.to_le_bytes());

    let data = build_apr_bytes("{}", &[("weight".to_string(), 0, vec![2, 3], weight_data)]);

    let model = AprV2Model::from_bytes(data).expect("should parse");
    let result = model
        .predict(&[1.0, 2.0, 3.0])
        .expect("predict should work");
    assert_eq!(result.len(), 2);
    // Row 0: 1*1 + 0*2 + 0*3 = 1
    assert!((result[0] - 1.0).abs() < 1e-6);
    // Row 1: 0*1 + 1*2 + 0*3 = 2
    assert!((result[1] - 2.0).abs() < 1e-6);
}

// ============================================================================
// get_tensor_f32 with various dtypes
// ============================================================================

#[test]
fn test_get_tensor_f32_basic() {
    // F32 tensor: [1.0, 2.0, 3.0, 4.0]
    let mut tensor_data = Vec::new();
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        tensor_data.extend_from_slice(&v.to_le_bytes());
    }

    let data = build_apr_bytes("{}", &[("t".to_string(), 0, vec![4], tensor_data)]);

    let model = AprV2Model::from_bytes(data).expect("should parse");
    let values = model.get_tensor_f32("t").expect("should load F32");
    assert_eq!(values.len(), 4);
    assert!((values[0] - 1.0).abs() < 1e-6);
    assert!((values[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_get_tensor_f32_not_found() {
    let data = build_apr_bytes("{}", &[]);
    let model = AprV2Model::from_bytes(data).expect("should parse");
    let result = model.get_tensor_f32("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_get_tensor_f32_unsupported_dtype() {
    // BF16 dtype (30) - not in the supported match arms for get_tensor_f32
    let tensor_data = vec![0u8; 32];
    let data = build_apr_bytes("{}", &[("bf16_t".to_string(), 30, vec![16], tensor_data)]);

    let model = AprV2Model::from_bytes(data).expect("should parse");
    let result = model.get_tensor_f32("bf16_t");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Unsupported") || err.contains("BF16"));
}

// ============================================================================
// get_tensor_bytes
// ============================================================================

#[test]
fn test_get_tensor_bytes_basic() {
    let tensor_data = vec![0xAA, 0xBB, 0xCC, 0xDD];
    let data = build_apr_bytes(
        "{}",
        &[("raw".to_string(), 0, vec![1], tensor_data.clone())],
    );

    let model = AprV2Model::from_bytes(data).expect("should parse");
    let bytes = model.get_tensor_bytes("raw").expect("should get bytes");
    assert_eq!(bytes, &tensor_data[..]);
}

#[test]
fn test_get_tensor_bytes_not_found() {
    let data = build_apr_bytes("{}", &[]);
    let model = AprV2Model::from_bytes(data).expect("should parse");
    let result = model.get_tensor_bytes("missing");
    assert!(result.is_err());
}

// ============================================================================
// forward() error paths
// ============================================================================

#[test]
fn test_forward_empty_tokens() {
    let data = build_apr_bytes(
        r#"{"hidden_size": 64, "num_layers": 1, "num_heads": 4, "vocab_size": 100}"#,
        &[],
    );
    let model = AprV2Model::from_bytes(data).expect("should parse");
    let result = model.forward(&[]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("empty"));
}

#[test]
fn test_forward_not_transformer() {
    let data = build_apr_bytes("{}", &[]);
    let model = AprV2Model::from_bytes(data).expect("should parse");
    let result = model.forward(&[1]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("transformer") || err.contains("config"));
}

// ============================================================================
// generate() error paths
// ============================================================================

#[test]
fn test_generate_empty_tokens() {
    let data = build_apr_bytes(
        r#"{"hidden_size": 64, "num_layers": 1, "num_heads": 4, "vocab_size": 100}"#,
        &[],
    );
    let model = AprV2Model::from_bytes(data).expect("should parse");
    let result = model.generate(&[], 5, None);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("empty"));
}

// ============================================================================
// find_tensor_name error path
// ============================================================================

#[test]
fn test_find_tensor_name_no_match() {
    let data = build_apr_bytes("{}", &[]);
    let model = AprV2Model::from_bytes(data).expect("should parse");
    // forward() will fail when it can't find embedding tensor
    let result = model.forward(&[1]);
    // Error should contain the candidates tried
    assert!(result.is_err());
}

// ============================================================================
// TensorEntry::element_count additional
// ============================================================================

#[test]
fn test_tensor_entry_element_count_4d() {
    let entry = TensorEntry {
        name: "4d".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 3, 4, 5],
        offset: 0,
        size: 480,
    };
    assert_eq!(entry.element_count(), 120);
}

include!("tests_apr_metadata_03.rs");
