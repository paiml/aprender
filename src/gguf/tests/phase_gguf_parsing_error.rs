//! Phase 36: GGUF Parsing Edge Cases and Error Handling
//!
//! This module adds test coverage for edge cases and error paths in GGUF parsing:
//!
//! - Invalid magic number handling
//! - Unsupported version detection
//! - Invalid value type handling
//! - Int32 and Bool metadata values
//! - Tensor extraction error cases
//! - GGUFConfig extraction
//! - RoPE scaling types
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::{
    GGUFConfig, GGUFHeader, GGUFModel, GGUFValue, TensorInfo, GGUF_ALIGNMENT, GGUF_MAGIC,
    GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1,
    GGUF_VERSION_V3,
};

// =============================================================================
// Test Data Builders
// =============================================================================

fn build_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&metadata_count.to_le_bytes());
    data
}

fn build_gguf_string(s: &str) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(s.len() as u64).to_le_bytes());
    data.extend_from_slice(s.as_bytes());
    data
}

fn build_gguf_metadata(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(build_gguf_string(key));
    data.extend_from_slice(&value_type.to_le_bytes());
    data.extend_from_slice(value_bytes);
    data
}

fn build_tensor_info(name: &str, dims: &[u64], qtype: u32, offset: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(build_gguf_string(name));
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&qtype.to_le_bytes());
    data.extend_from_slice(&offset.to_le_bytes());
    data
}

// =============================================================================
// Invalid Header Tests
// =============================================================================

#[test]
fn test_phase36_invalid_magic_number() {
    let mut data = Vec::new();
    data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Invalid magic should fail");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(err_str.contains("Invalid GGUF magic"), "Error: {}", err_str);
}

#[test]
fn test_phase36_unsupported_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&99u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Unsupported version should fail");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("Unsupported GGUF version"),
        "Error: {}",
        err_str
    );
}

#[test]
fn test_phase36_truncated_header() {
    let data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00];
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated header should fail");
}

#[test]
fn test_phase36_unsupported_value_type() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_string("test_key"));
    data.extend_from_slice(&255u32.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Unknown value type should fail");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("Unsupported value type"),
        "Error: {}",
        err_str
    );
}

// =============================================================================
// Metadata Type Tests
// =============================================================================

#[test]
fn test_phase36_int32_metadata_values() {
    let mut data = build_gguf_header(0, 3);
    data.extend(build_gguf_metadata("test_pos", 5, &100i32.to_le_bytes()));
    data.extend(build_gguf_metadata("test_neg", 5, &(-50i32).to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 5, &i32::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_pos"),
        Some(GGUFValue::Int32(100))
    ));
    assert!(matches!(
        model.metadata.get("test_neg"),
        Some(GGUFValue::Int32(-50))
    ));
    assert!(matches!(model.metadata.get("test_max"), Some(GGUFValue::Int32(v)) if *v == i32::MAX));
}

#[test]
fn test_phase36_bool_metadata_nonzero_is_true() {
    let mut data = build_gguf_header(0, 3);
    data.extend(build_gguf_metadata("test_false", 7, &[0u8]));
    data.extend(build_gguf_metadata("test_true", 7, &[1u8]));
    data.extend(build_gguf_metadata("test_nonzero", 7, &[255u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_false"),
        Some(GGUFValue::Bool(false))
    ));
    assert!(matches!(
        model.metadata.get("test_true"),
        Some(GGUFValue::Bool(true))
    ));
    assert!(matches!(
        model.metadata.get("test_nonzero"),
        Some(GGUFValue::Bool(true))
    ));
}

// =============================================================================
// Tensor Extraction Tests
// =============================================================================

#[test]
fn test_phase36_f16_tensor_extraction() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[8], GGUF_TYPE_F16, 0));
    let aligned = data.len().div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);

    for i in 0..8 {
        let val = half::f16::from_f32(i as f32 * 0.5);
        data.extend_from_slice(&val.to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 8);
    assert!((tensor[2] - 1.0).abs() < 0.01);
}

#[test]
fn test_phase36_q4_1_tensor_extraction() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q4_1, 0));
    let aligned = data.len().div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);

    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0x88u8; 16]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase36_q5_0_tensor_extraction() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_0, 0));
    let aligned = data.len().div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);

    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend([0u8; 4]);
    data.extend([0x88u8; 16]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase36_q5_1_tensor_extraction() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_1, 0));
    let aligned = data.len().div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);

    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0u8; 4]);
    data.extend([0x88u8; 16]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

// =============================================================================
// Tensor Error Cases
// =============================================================================

#[test]
fn test_phase36_tensor_not_found() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");

    let result = model.get_tensor_f32("nonexistent", &data);
    assert!(result.is_err(), "Nonexistent tensor should fail");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(err_str.contains("not found"), "Error: {}", err_str);
}

#[test]
fn test_phase36_tensor_unsupported_qtype() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], 99, 0));
    let aligned = data.len().div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);
    data.extend([0u8; 128]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_err(), "Unsupported qtype should fail");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("Unsupported quantization type"),
        "Error: {}",
        err_str
    );
}

#[test]
fn test_phase36_tensor_data_out_of_bounds() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[1024], GGUF_TYPE_F32, 10000));
    let aligned = data.len().div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);
    data.extend([0u8; 64]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_err(), "Out of bounds tensor should fail");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(err_str.contains("exceeds file size"), "Error: {}", err_str);
}

// =============================================================================
// GGUFConfig Tests
// =============================================================================

#[test]
fn test_phase36_config_missing_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = GGUFConfig::from_gguf(&model);
    assert!(result.is_err(), "Missing architecture should fail");
}

// =============================================================================
// RoPE and Structure Tests
// =============================================================================

#[test]
fn test_phase36_rope_scaling_linear() {
    let mut data = build_gguf_header(0, 2);
    let arch = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch));
    let linear = build_gguf_string("linear");
    data.extend(build_gguf_metadata("custom.rope.scaling.type", 8, &linear));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(0));
}

#[test]
fn test_phase36_gguf_header_equality() {
    let h1 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 10,
        metadata_count: 5,
    };
    let h2 = h1.clone();
    assert_eq!(h1, h2);

    let h3 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 20,
        metadata_count: 5,
    };
    assert_ne!(h1, h3);
}

#[test]
fn test_phase36_tensor_info_equality() {
    let t1 = TensorInfo {
        name: "test.weight".to_string(),
        n_dims: 2,
        dims: vec![64, 128],
        qtype: GGUF_TYPE_Q4_0,
        offset: 0,
    };
    assert_eq!(t1.clone(), t1);
}

#[test]
fn test_phase36_gguf_value_all_numeric_types() {
    let values = vec![
        GGUFValue::UInt8(255),
        GGUFValue::Int8(-128),
        GGUFValue::UInt16(65535),
        GGUFValue::Int16(-32768),
        GGUFValue::UInt32(u32::MAX),
        GGUFValue::Int32(i32::MIN),
        GGUFValue::UInt64(u64::MAX),
        GGUFValue::Int64(i64::MIN),
        GGUFValue::Float32(std::f32::consts::PI),
        GGUFValue::Float64(std::f64::consts::E),
    ];

    for v in &values {
        let cloned = v.clone();
        assert_eq!(v, &cloned);
        let debug = format!("{:?}", v);
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_phase36_tensor_data_alignment() {
    let mut data = build_gguf_header(1, 1);
    data.extend(build_gguf_metadata("x", 4, &1u32.to_le_bytes()));
    data.extend(build_tensor_info("tensor", &[4], GGUF_TYPE_F32, 0));
    let aligned = data.len().div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);

    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);

    let tensor = model
        .get_tensor_f32("tensor", &data)
        .expect("Should extract");
    assert_eq!(tensor.len(), 4);
    assert!((tensor[0] - 0.0).abs() < 0.001);
    assert!((tensor[3] - 3.0).abs() < 0.001);
}
