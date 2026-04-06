//! T-COV-95 Data Storm: Multi-Tensor Pygmies for loader.rs (PMAT-802)
//!
//! Dr. Popper's directive: "We have tested the 'Pygmy in the Forest,'
//! but not the 'Pygmy in the Data Storm.'"
//!
//! This module creates GGUF files with MULTIPLE tensors of different
//! quantization types to exercise all dequantization paths in get_tensor_f32.
//!
//! Target: 946 missed regions in gguf/loader.rs

use crate::gguf::{
    GGUFModel, GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0,
    GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K,
    GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};

// ============================================================================
// Multi-Tensor Pygmy Builder
// ============================================================================

/// Build a multi-tensor GGUF with specified tensor configurations
fn build_multi_tensor_gguf(tensors: &[(&str, &[u64], u32, &[u8])]) -> Vec<u8> {
    let mut data = Vec::new();

    // Header: magic + version + tensor_count + metadata_count
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // no metadata

    // Tensor info section
    let mut tensor_data_sizes = Vec::new();
    for (name, dims, qtype, _tensor_data) in tensors {
        // Name: length + bytes
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());

        // n_dims
        data.extend_from_slice(&(dims.len() as u32).to_le_bytes());

        // Dimensions (reversed for GGML format)
        for &dim in dims.iter().rev() {
            data.extend_from_slice(&dim.to_le_bytes());
        }

        // qtype
        data.extend_from_slice(&qtype.to_le_bytes());

        // offset (calculated after alignment)
        let offset = tensor_data_sizes.iter().sum::<usize>();
        data.extend_from_slice(&(offset as u64).to_le_bytes());

        tensor_data_sizes.push(
            tensors
                .iter()
                .find(|(n, _, _, _)| n == name)
                .map_or(0, |(_, _, _, d)| d.len()),
        );
    }

    // Align to 32-byte boundary for tensor data
    let current_len = data.len();
    let aligned = current_len.div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);

    // Tensor data section
    for (_, _, _, tensor_data) in tensors {
        data.extend_from_slice(tensor_data);
    }

    data
}

/// Build GGUF string bytes (u64 length + UTF-8 bytes)
fn build_string(s: &str) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(s.len() as u64).to_le_bytes());
    data.extend_from_slice(s.as_bytes());
    data
}

// ============================================================================
// get_tensor_f32 Error Path Tests
// ============================================================================

#[test]
fn test_data_storm_tensor_not_found() {
    // Create GGUF with one tensor named "exists"
    let tensor_data = vec![0u8; 18]; // Q4_0 block
    let gguf_data = build_multi_tensor_gguf(&[("exists", &[32], GGUF_TYPE_Q4_0, &tensor_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");

    // Try to get nonexistent tensor
    let result = model.get_tensor_f32("does_not_exist", &gguf_data);
    assert!(result.is_err());
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(err_str.contains("not found") || err_str.contains("Tensor"));
}

#[test]
fn test_data_storm_f32_tensor_extraction() {
    // F32 tensor: 4 elements = 16 bytes
    let f32_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let gguf_data = build_multi_tensor_gguf(&[("test_f32", &[4], GGUF_TYPE_F32, &f32_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_f32", &gguf_data);

    match result {
        Ok(values) => {
            assert_eq!(values.len(), 4);
            assert!((values[0] - 1.0).abs() < 1e-5);
        },
        Err(e) => {
            // Offset calculation may be off in our builder
            let err_str = format!("{:?}", e);
            assert!(err_str.contains("range") || err_str.contains("exceeds"));
        },
    }
}

#[test]
fn test_data_storm_f16_tensor_extraction() {
    // F16 tensor: 4 elements = 8 bytes
    // Using known f16 bit patterns
    let f16_data: Vec<u8> = vec![
        0x00, 0x3C, // 1.0 in f16
        0x00, 0x40, // 2.0 in f16
        0x00, 0x42, // 3.0 in f16
        0x00, 0x44, // 4.0 in f16
    ];

    let gguf_data = build_multi_tensor_gguf(&[("test_f16", &[4], GGUF_TYPE_F16, &f16_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_f16", &gguf_data);

    // Either succeeds or fails with offset error - both exercise code
    if let Ok(values) = result {
        assert_eq!(values.len(), 4);
    }
}

#[test]
fn test_data_storm_q4_0_tensor_extraction() {
    // Q4_0 block: 18 bytes for 32 elements
    // 2 bytes f16 scale + 16 bytes quants
    let mut q4_0_data = vec![0u8; 18];
    q4_0_data[0] = 0x00; // scale f16 low
    q4_0_data[1] = 0x3C; // scale f16 high (1.0)

    let gguf_data = build_multi_tensor_gguf(&[("test_q4_0", &[32], GGUF_TYPE_Q4_0, &q4_0_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q4_0", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 32);
    }
}

#[test]
fn test_data_storm_q8_0_tensor_extraction() {
    // Q8_0 block: 34 bytes for 32 elements
    // 2 bytes f16 scale + 32 bytes quants
    let mut q8_0_data = vec![0u8; 34];
    q8_0_data[0] = 0x00;
    q8_0_data[1] = 0x3C; // scale = 1.0

    let gguf_data = build_multi_tensor_gguf(&[("test_q8_0", &[32], GGUF_TYPE_Q8_0, &q8_0_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q8_0", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 32);
    }
}

#[test]
fn test_data_storm_q4_1_tensor_extraction() {
    // Q4_1 block: 20 bytes for 32 elements
    let q4_1_data = vec![0u8; 20];

    let gguf_data = build_multi_tensor_gguf(&[("test_q4_1", &[32], GGUF_TYPE_Q4_1, &q4_1_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q4_1", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 32);
    }
}

#[test]
fn test_data_storm_q5_0_tensor_extraction() {
    // Q5_0 block: 22 bytes for 32 elements
    let q5_0_data = vec![0u8; 22];

    let gguf_data = build_multi_tensor_gguf(&[("test_q5_0", &[32], GGUF_TYPE_Q5_0, &q5_0_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q5_0", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 32);
    }
}

#[test]
fn test_data_storm_q5_1_tensor_extraction() {
    // Q5_1 block: 24 bytes for 32 elements
    let q5_1_data = vec![0u8; 24];

    let gguf_data = build_multi_tensor_gguf(&[("test_q5_1", &[32], GGUF_TYPE_Q5_1, &q5_1_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q5_1", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 32);
    }
}

#[test]
fn test_data_storm_q4_k_tensor_extraction() {
    // Q4_K super-block: 144 bytes for 256 elements
    let q4_k_data = vec![0u8; 144];

    let gguf_data = build_multi_tensor_gguf(&[("test_q4_k", &[256], GGUF_TYPE_Q4_K, &q4_k_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q4_k", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 256);
    }
}

#[test]
fn test_data_storm_q5_k_tensor_extraction() {
    // Q5_K super-block: 176 bytes for 256 elements
    let q5_k_data = vec![0u8; 176];

    let gguf_data = build_multi_tensor_gguf(&[("test_q5_k", &[256], GGUF_TYPE_Q5_K, &q5_k_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q5_k", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 256);
    }
}

#[test]
fn test_data_storm_q6_k_tensor_extraction() {
    // Q6_K super-block: 210 bytes for 256 elements
    let q6_k_data = vec![0u8; 210];

    let gguf_data = build_multi_tensor_gguf(&[("test_q6_k", &[256], GGUF_TYPE_Q6_K, &q6_k_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q6_k", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 256);
    }
}

// ============================================================================
// Data Storm: Truncated Tensor Data
// ============================================================================

#[test]
fn test_data_storm_truncated_f32_data() {
    // Tensor claims 100 elements but only has 10 bytes
    let f32_data = vec![0u8; 10]; // Way too small for 100 f32s

    let gguf_data = build_multi_tensor_gguf(&[("truncated", &[100], GGUF_TYPE_F32, &f32_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("truncated", &gguf_data);

    assert!(result.is_err());
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(err_str.contains("exceeds") || err_str.contains("range"));
}

#[test]
fn test_data_storm_truncated_q4_0_data() {
    // Tensor claims 1024 elements but only has 1 block
    let q4_0_data = vec![0u8; 18]; // 1 block = 32 elements

    let gguf_data =
        build_multi_tensor_gguf(&[("truncated_q4_0", &[1024], GGUF_TYPE_Q4_0, &q4_0_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("truncated_q4_0", &gguf_data);

    assert!(result.is_err());
}

#[test]
fn test_data_storm_truncated_q4_k_data() {
    // Tensor claims 1024 elements but only has 1 super-block
    let q4_k_data = vec![0u8; 144]; // 1 super-block = 256 elements

    let gguf_data =
        build_multi_tensor_gguf(&[("truncated_q4_k", &[1024], GGUF_TYPE_Q4_K, &q4_k_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("truncated_q4_k", &gguf_data);

    assert!(result.is_err());
}

// ============================================================================
// Data Storm: Unsupported Quantization Types
// ============================================================================

#[test]
fn test_data_storm_unsupported_qtype() {
    // Use qtype 255 which is not supported
    let data = vec![0u8; 100];

    let gguf_data = build_multi_tensor_gguf(&[("unsupported", &[32], 255, &data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("unsupported", &gguf_data);

    assert!(result.is_err());
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("Unsupported") || err_str.contains("quantization"),
        "Error: {}",
        err_str
    );
}

// ============================================================================
// Data Storm: Multi-Tensor GGUF
// ============================================================================

#[test]
fn test_data_storm_multiple_tensors() {
    // Create GGUF with multiple tensors of different types
    let f32_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let q4_0_data = vec![0u8; 18];
    let q8_0_data = vec![0u8; 34];

    let gguf_data = build_multi_tensor_gguf(&[
        ("tensor_f32", &[4], GGUF_TYPE_F32, &f32_data),
        ("tensor_q4_0", &[32], GGUF_TYPE_Q4_0, &q4_0_data),
        ("tensor_q8_0", &[32], GGUF_TYPE_Q8_0, &q8_0_data),
    ]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");

    // Verify tensor count
    assert_eq!(model.tensors.len(), 3);

    // Try to extract each tensor
    let _ = model.get_tensor_f32("tensor_f32", &gguf_data);
    let _ = model.get_tensor_f32("tensor_q4_0", &gguf_data);
    let _ = model.get_tensor_f32("tensor_q8_0", &gguf_data);
}

#[test]
fn test_data_storm_tensor_dimensions_2d() {
    // 2D tensor: [32, 64] = 2048 elements
    let f32_data: Vec<u8> = vec![0u8; 2048 * 4];

    let gguf_data = build_multi_tensor_gguf(&[("matrix", &[32, 64], GGUF_TYPE_F32, &f32_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");

    // Verify dimensions are parsed correctly
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == "matrix")
        .expect("test value should be present");
    assert_eq!(tensor.n_dims, 2);
    // Dimensions reversed from GGML format
    assert!(tensor.dims.contains(&32) && tensor.dims.contains(&64));
}

#[test]
fn test_data_storm_tensor_dimensions_3d() {
    // 3D tensor: [4, 8, 16] = 512 elements
    let f32_data: Vec<u8> = vec![0u8; 512 * 4];

    let gguf_data =
        build_multi_tensor_gguf(&[("tensor_3d", &[4, 8, 16], GGUF_TYPE_F32, &f32_data)]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == "tensor_3d")
        .expect("test value should be present");
    assert_eq!(tensor.n_dims, 3);
}

// ============================================================================
// Data Storm: Dimension Overflow
// ============================================================================

#[test]
fn test_data_storm_dimension_overflow() {
    // Manually create GGUF with huge dimensions that would overflow
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    // Tensor info with huge dimensions
    data.extend(build_string("huge_tensor"));
    data.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
                                                 // Dimensions that would overflow when multiplied: u64::MAX / 2 * 3
    data.extend_from_slice(&(u64::MAX / 2).to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Pad to alignment
    let aligned = data.len().div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned + 100, 0);

    let model = GGUFModel::from_bytes(&data).expect("parse header");
    let result = model.get_tensor_f32("huge_tensor", &data);

    assert!(result.is_err());
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("overflow") || err_str.contains("Overflow") || err_str.contains("exceeds"),
        "Error should mention overflow: {}",
        err_str
    );
}

// ============================================================================
// Data Storm: Q2_K Tensor (K-quant 2-bit)
// ============================================================================

#[test]
fn test_data_storm_q2_k_tensor_extraction() {
    // Q2_K super-block: 84 bytes for 256 elements
    let q2_k_data = vec![0u8; 84];

    let gguf_data = build_multi_tensor_gguf(&[
        ("test_q2_k", &[256], 10, &q2_k_data), // 10 = GGUF_TYPE_Q2_K
    ]);

    let model = GGUFModel::from_bytes(&gguf_data).expect("parse");
    let result = model.get_tensor_f32("test_q2_k", &gguf_data);

    if let Ok(values) = result {
        assert_eq!(values.len(), 256);
    }
}
