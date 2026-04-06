//! Part 24: GGUF Loader State-Space Exhaustion Tests (Phase 52)
//!
//! Targets the ~57% uncovered code paths in `src/gguf/loader.rs`:
//! - All 13 metadata value types (0-12)
//! - Error paths (invalid magic, invalid version, truncated data)
//! - All quantization types in get_tensor_f32
//! - Helper methods (architecture, embedding_dim, num_layers)

use crate::gguf::test_factory::{
    create_q4_0_data, create_q4_k_data, create_q5_k_data, create_q6_k_data, create_q8_0_data,
    GGUFBuilder,
};
use crate::gguf::types::{GGUF_MAGIC, GGUF_VERSION_V3};
use crate::gguf::{GGUFModel, GGUFValue};

// =============================================================================
// Metadata Value Type Coverage (Types 0-12)
// =============================================================================

/// Build a GGUF with a specific metadata value type for testing
fn build_gguf_with_metadata(key: &str, value_type: u32, value_bytes: Vec<u8>) -> Vec<u8> {
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Metadata entry
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&value_type.to_le_bytes());
    data.extend_from_slice(&value_bytes);

    // Align to 32 bytes
    let aligned = data.len().div_ceil(32) * 32;
    data.resize(aligned, 0);

    data
}

#[test]
fn test_metadata_type_0_uint8() {
    let data = build_gguf_with_metadata("test.uint8", 0, vec![42u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.uint8"),
        Some(GGUFValue::UInt8(42))
    ));
}

#[test]
fn test_metadata_type_1_int8() {
    let data = build_gguf_with_metadata("test.int8", 1, vec![0xFEu8]); // -2 as i8
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.int8"),
        Some(GGUFValue::Int8(-2))
    ));
}

#[test]
fn test_metadata_type_2_uint16() {
    let data = build_gguf_with_metadata("test.uint16", 2, 1000u16.to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.uint16"),
        Some(GGUFValue::UInt16(1000))
    ));
}

#[test]
fn test_metadata_type_3_int16() {
    let data = build_gguf_with_metadata("test.int16", 3, (-500i16).to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.int16"),
        Some(GGUFValue::Int16(-500))
    ));
}

#[test]
fn test_metadata_type_4_uint32() {
    let data = build_gguf_with_metadata("test.uint32", 4, 100000u32.to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.uint32"),
        Some(GGUFValue::UInt32(100000))
    ));
}

#[test]
fn test_metadata_type_5_int32() {
    let data = build_gguf_with_metadata("test.int32", 5, (-50000i32).to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.int32"),
        Some(GGUFValue::Int32(-50000))
    ));
}

#[test]
fn test_metadata_type_6_float32() {
    let data = build_gguf_with_metadata("test.float32", 6, 3.14f32.to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test.float32") {
        assert!((v - 3.14).abs() < 0.001);
    } else {
        panic!("Expected Float32");
    }
}

#[test]
fn test_metadata_type_7_bool_true() {
    let data = build_gguf_with_metadata("test.bool", 7, vec![1u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.bool"),
        Some(GGUFValue::Bool(true))
    ));
}

#[test]
fn test_metadata_type_7_bool_false() {
    let data = build_gguf_with_metadata("test.bool_false", 7, vec![0u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.bool_false"),
        Some(GGUFValue::Bool(false))
    ));
}

#[test]
fn test_metadata_type_8_string() {
    let s = "hello world";
    let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(s.as_bytes());
    let data = build_gguf_with_metadata("test.string", 8, bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.string"),
        Some(GGUFValue::String(v)) if v == "hello world"
    ));
}

#[test]
fn test_metadata_type_9_array_u32() {
    // Array of 3 u32 values
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&4u32.to_le_bytes()); // element_type = u32
    bytes.extend_from_slice(&3u64.to_le_bytes()); // array_len = 3
    bytes.extend_from_slice(&10u32.to_le_bytes());
    bytes.extend_from_slice(&20u32.to_le_bytes());
    bytes.extend_from_slice(&30u32.to_le_bytes());

    let data = build_gguf_with_metadata("test.array", 9, bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test.array") {
        assert_eq!(arr.len(), 3);
        assert!(matches!(arr[0], GGUFValue::UInt32(10)));
        assert!(matches!(arr[1], GGUFValue::UInt32(20)));
        assert!(matches!(arr[2], GGUFValue::UInt32(30)));
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_metadata_type_10_uint64() {
    let data = build_gguf_with_metadata("test.uint64", 10, u64::MAX.to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.uint64"),
        Some(GGUFValue::UInt64(u64::MAX))
    ));
}

#[test]
fn test_metadata_type_11_int64() {
    let data = build_gguf_with_metadata("test.int64", 11, i64::MIN.to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.int64"),
        Some(GGUFValue::Int64(i64::MIN))
    ));
}

#[test]
fn test_metadata_type_12_float64() {
    let data = build_gguf_with_metadata(
        "test.float64",
        12,
        std::f64::consts::PI.to_le_bytes().to_vec(),
    );
    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Float64(v)) = model.metadata.get("test.float64") {
        assert!((v - std::f64::consts::PI).abs() < 1e-10);
    } else {
        panic!("Expected Float64");
    }
}

// =============================================================================
// Error Path Coverage
// =============================================================================

#[test]
fn test_error_invalid_magic() {
    let mut data = Vec::new();
    data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // Invalid magic
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("magic"));
}

#[test]
fn test_error_unsupported_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("version"));
}

#[test]
fn test_error_truncated_header() {
    // Only 8 bytes - truncated before tensor_count
    let data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00];
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_error_truncated_metadata_key() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1
    data.extend_from_slice(&100u64.to_le_bytes()); // key length = 100 (but no data)

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_error_unsupported_metadata_type() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Valid key
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(b"test");
    // Invalid type = 99
    data.extend_from_slice(&99u32.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Unsupported value type"));
}

// =============================================================================
// Quantization Type Coverage via get_tensor_f32
// =============================================================================

#[test]
fn test_get_tensor_f32_type_f32() {
    let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let data = GGUFBuilder::new()
        .architecture("test")
        .add_f32_tensor("weights", &[4], &f32_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let weights = model.get_tensor_f32("weights", &data).expect("get tensor");
    assert_eq!(weights.len(), 4);
    assert!((weights[0] - 1.0).abs() < 0.001);
    assert!((weights[3] - 4.0).abs() < 0.001);
}

#[test]
fn test_get_tensor_f32_type_q4_0() {
    let q4_0_data = create_q4_0_data(64); // 64 elements = 2 blocks
    let data = GGUFBuilder::new()
        .architecture("test")
        .add_q4_0_tensor("weights", &[64], &q4_0_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let weights = model.get_tensor_f32("weights", &data).expect("get tensor");
    assert_eq!(weights.len(), 64);
}

#[test]
fn test_get_tensor_f32_type_q8_0() {
    let q8_0_data = create_q8_0_data(64);
    let data = GGUFBuilder::new()
        .architecture("test")
        .add_q8_0_tensor("weights", &[64], &q8_0_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let weights = model.get_tensor_f32("weights", &data).expect("get tensor");
    assert_eq!(weights.len(), 64);
}

#[test]
fn test_get_tensor_f32_type_q4_k() {
    let q4_k_data = create_q4_k_data(256); // 1 super-block
    let data = GGUFBuilder::new()
        .architecture("test")
        .add_q4_k_tensor("weights", &[256], &q4_k_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let weights = model.get_tensor_f32("weights", &data).expect("get tensor");
    assert_eq!(weights.len(), 256);
}

#[test]
fn test_get_tensor_f32_type_q5_k() {
    let q5_k_data = create_q5_k_data(256);
    let data = GGUFBuilder::new()
        .architecture("test")
        .add_q5_k_tensor("weights", &[256], &q5_k_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let weights = model.get_tensor_f32("weights", &data).expect("get tensor");
    assert_eq!(weights.len(), 256);
}

#[test]
fn test_get_tensor_f32_type_q6_k() {
    let q6_k_data = create_q6_k_data(256);
    let data = GGUFBuilder::new()
        .architecture("test")
        .add_q6_k_tensor("weights", &[256], &q6_k_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let weights = model.get_tensor_f32("weights", &data).expect("get tensor");
    assert_eq!(weights.len(), 256);
}

#[test]
fn test_get_tensor_not_found() {
    let data = GGUFBuilder::new().architecture("test").build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let result = model.get_tensor_f32("nonexistent", &data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

// =============================================================================
// Helper Method Coverage
// =============================================================================

#[test]
fn test_architecture_helper() {
    let data = GGUFBuilder::new().architecture("llama").build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.architecture(), Some("llama"));
}

#[test]
fn test_architecture_missing() {
    let data = GGUFBuilder::new().build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.architecture(), None);
}

#[test]
fn test_embedding_dim_helper() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 4096)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.embedding_dim(), Some(4096));
}

#[test]
fn test_embedding_dim_missing_arch() {
    let data = GGUFBuilder::new().build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.embedding_dim(), None);
}

#[test]
fn test_num_layers_helper() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .num_layers("llama", 32)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.num_layers(), Some(32));
}

#[test]
fn test_num_layers_missing() {
    let data = GGUFBuilder::new().architecture("llama").build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.num_layers(), None);
}

// =============================================================================
// Tensor Info Parsing Edge Cases
// =============================================================================

#[test]
fn test_tensor_3d_dimensions() {
    let f32_data: Vec<f32> = vec![1.0; 2 * 3 * 4];
    let data = GGUFBuilder::new()
        .architecture("test")
        .add_f32_tensor("tensor3d", &[4, 3, 2], &f32_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == "tensor3d")
        .expect("test value should be present");
    // Dimensions are reversed for GGML order
    assert_eq!(tensor.dims.len(), 3);
}

include!("multiple_tensors.rs");
