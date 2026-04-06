//! Phase 33: GGUF Loader coverage tests
//!
//! These lib tests illuminate gguf/loader.rs:
//! - from_bytes() - GGUF parsing
//! - parse_header() - Header validation
//! - parse_metadata() - Metadata parsing
//! - parse_tensor_info() - Tensor info extraction
//! - get_tensor_f32() - Tensor extraction with dequantization
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::{GGUFModel, GGUFValue, GGUF_MAGIC, GGUF_VERSION_V3};

// =============================================================================
// GGUF Test Data Builder
// =============================================================================

/// Build valid GGUF header bytes
fn build_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    // Tensor count
    data.extend_from_slice(&tensor_count.to_le_bytes());
    // Metadata count
    data.extend_from_slice(&metadata_count.to_le_bytes());
    data
}

/// Build a GGUF string: u64 length + UTF-8 bytes
fn build_gguf_string(s: &str) -> Vec<u8> {
    let mut data = Vec::new();
    let len = s.len() as u64;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(s.as_bytes());
    data
}

/// Build a GGUF metadata key-value pair
fn build_gguf_metadata(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(build_gguf_string(key));
    data.extend_from_slice(&value_type.to_le_bytes());
    data.extend_from_slice(value_bytes);
    data
}

/// Build a GGUF tensor info entry
fn build_tensor_info(name: &str, dims: &[u64], qtype: u32, offset: u64) -> Vec<u8> {
    let mut data = Vec::new();
    // Name string
    data.extend(build_gguf_string(name));
    // n_dims
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    // Dimensions (reversed for GGML order)
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    // Quantization type
    data.extend_from_slice(&qtype.to_le_bytes());
    // Offset
    data.extend_from_slice(&offset.to_le_bytes());
    data
}

// =============================================================================
// Header Parsing Tests
// =============================================================================

#[test]
fn test_phase33_loader_valid_header() {
    // Minimal valid GGUF with no tensors, no metadata
    let data = build_gguf_header(0, 0);
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "Valid header should parse");
    let model = result.expect("test value should be present");
    assert_eq!(model.header.magic, GGUF_MAGIC);
    assert_eq!(model.header.version, GGUF_VERSION_V3);
    assert_eq!(model.header.tensor_count, 0);
    assert_eq!(model.header.metadata_count, 0);
}

#[test]
fn test_phase33_loader_invalid_magic() {
    let mut data = build_gguf_header(0, 0);
    // Corrupt magic
    data[0] = 0xFF;
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("magic") || err.contains("Magic"),
        "Error should mention magic: {}",
        err
    );
}

#[test]
fn test_phase33_loader_wrong_version() {
    let mut data = Vec::new();
    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version 99 (truly unsupported; v2 is valid per GH-310)
    data.extend_from_slice(&99u32.to_le_bytes());
    // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes());
    // Metadata count
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("version") || err.contains("Version"),
        "Error should mention version: {}",
        err
    );
}

#[test]
fn test_phase33_loader_truncated_header() {
    // Only magic, missing rest
    let data = GGUF_MAGIC.to_le_bytes();
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated header should fail");
}

#[test]
fn test_phase33_loader_empty_data() {
    let result = GGUFModel::from_bytes(&[]);
    assert!(result.is_err(), "Empty data should fail");
}

// =============================================================================
// Metadata Parsing Tests
// =============================================================================

#[test]
fn test_phase33_loader_metadata_uint8() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_u8", 0, &[42u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_u8").expect("Key should exist");
    assert!(matches!(value, GGUFValue::UInt8(42)));
}

#[test]
fn test_phase33_loader_metadata_int8() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata(
        "test_i8",
        1,
        &[(-10i8).to_le_bytes()[0]],
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_i8").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Int8(-10)));
}

#[test]
fn test_phase33_loader_metadata_uint16() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_u16", 2, &1000u16.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_u16").expect("Key should exist");
    assert!(matches!(value, GGUFValue::UInt16(1000)));
}

#[test]
fn test_phase33_loader_metadata_int16() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_i16", 3, &(-500i16).to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_i16").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Int16(-500)));
}

#[test]
fn test_phase33_loader_metadata_uint32() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_u32", 4, &100000u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_u32").expect("Key should exist");
    assert!(matches!(value, GGUFValue::UInt32(100000)));
}

#[test]
fn test_phase33_loader_metadata_int32() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata(
        "test_i32",
        5,
        &(-50000i32).to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_i32").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Int32(-50000)));
}

#[test]
fn test_phase33_loader_metadata_float32() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_f32", 6, &3.14f32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_f32").expect("Key should exist");
    if let GGUFValue::Float32(v) = value {
        assert!((v - 3.14).abs() < 0.001);
    } else {
        panic!("Expected Float32");
    }
}

#[test]
fn test_phase33_loader_metadata_bool_true() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_bool", 7, &[1u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_bool").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Bool(true)));
}

#[test]
fn test_phase33_loader_metadata_bool_false() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_bool", 7, &[0u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_bool").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Bool(false)));
}

#[test]
fn test_phase33_loader_metadata_string() {
    let mut data = build_gguf_header(0, 1);
    let string_value = build_gguf_string("hello world");
    data.extend(build_gguf_metadata("test_string", 8, &string_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_string").expect("Key should exist");
    if let GGUFValue::String(s) = value {
        assert_eq!(s, "hello world");
    } else {
        panic!("Expected String");
    }
}

#[test]
fn test_phase33_loader_metadata_uint64() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata(
        "test_u64",
        10,
        &999999999u64.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_u64").expect("Key should exist");
    assert!(matches!(value, GGUFValue::UInt64(999999999)));
}

#[test]
fn test_phase33_loader_metadata_int64() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata(
        "test_i64",
        11,
        &(-999999999i64).to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_i64").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Int64(-999999999)));
}

#[test]
fn test_phase33_loader_metadata_float64() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata(
        "test_f64",
        12,
        &2.71828f64.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_f64").expect("Key should exist");
    if let GGUFValue::Float64(v) = value {
        assert!((v - 2.71828).abs() < 0.00001);
    } else {
        panic!("Expected Float64");
    }
}

#[test]
fn test_phase33_loader_metadata_array_u32() {
    let mut data = build_gguf_header(0, 1);
    // Array: element_type (u32=4) + array_len (u64=3) + elements
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
    array_bytes.extend_from_slice(&3u64.to_le_bytes()); // array length: 3
    array_bytes.extend_from_slice(&10u32.to_le_bytes());
    array_bytes.extend_from_slice(&20u32.to_le_bytes());
    array_bytes.extend_from_slice(&30u32.to_le_bytes());
    data.extend(build_gguf_metadata("test_array", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_array").expect("Key should exist");
    if let GGUFValue::Array(arr) = value {
        assert_eq!(arr.len(), 3);
        assert!(matches!(arr[0], GGUFValue::UInt32(10)));
        assert!(matches!(arr[1], GGUFValue::UInt32(20)));
        assert!(matches!(arr[2], GGUFValue::UInt32(30)));
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_phase33_loader_metadata_unsupported_type() {
    let mut data = build_gguf_header(0, 1);
    // Type 99 doesn't exist
    data.extend(build_gguf_metadata("test_bad", 99, &[0u8; 8]));

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("99") || err.contains("type"),
        "Error should mention bad type: {}",
        err
    );
}

#[test]
fn test_phase33_loader_multiple_metadata() {
    let mut data = build_gguf_header(0, 3);
    data.extend(build_gguf_metadata("hidden_dim", 4, &256u32.to_le_bytes()));
    data.extend(build_gguf_metadata("num_layers", 4, &12u32.to_le_bytes()));
    data.extend(build_gguf_metadata(
        "rope_theta",
        6,
        &10000.0f32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.metadata.len(), 3);
    assert!(matches!(
        model.metadata.get("hidden_dim"),
        Some(GGUFValue::UInt32(256))
    ));
    assert!(matches!(
        model.metadata.get("num_layers"),
        Some(GGUFValue::UInt32(12))
    ));
}

// =============================================================================
// Tensor Info Parsing Tests
// =============================================================================

#[test]
fn test_phase33_loader_tensor_info_basic() {
    let mut data = build_gguf_header(1, 0);
    // Add tensor info: 2D tensor [256, 512], F32, offset 0
    data.extend(build_tensor_info("weights", &[256, 512], 0, 0)); // qtype 0 = F32

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].name, "weights");
    assert_eq!(model.tensors[0].n_dims, 2);
    // Dims should be reversed from input (GGML order handling)
    assert_eq!(model.tensors[0].dims, vec![256, 512]);
    assert_eq!(model.tensors[0].qtype, 0);
    assert_eq!(model.tensors[0].offset, 0);
}

#[test]
fn test_phase33_loader_tensor_info_multiple() {
    let mut data = build_gguf_header(3, 0);
    data.extend(build_tensor_info("embed", &[100, 64], 0, 0));
    data.extend(build_tensor_info("attn.q", &[64, 64], 2, 25600)); // Q4_0
    data.extend(build_tensor_info("ffn.w1", &[64, 256], 6, 27648)); // Q8_0

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors.len(), 3);
    assert_eq!(model.tensors[0].name, "embed");
    assert_eq!(model.tensors[1].name, "attn.q");
    assert_eq!(model.tensors[2].name, "ffn.w1");
}

#[test]
fn test_phase33_loader_tensor_info_1d() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("bias", &[512], 0, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].n_dims, 1);
    assert_eq!(model.tensors[0].dims, vec![512]);
}

#[test]
fn test_phase33_loader_tensor_info_4d() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("conv", &[3, 3, 64, 128], 0, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].n_dims, 4);
    assert_eq!(model.tensors[0].dims.len(), 4);
}

// =============================================================================
// get_tensor_f32 Tests (requires tensor data)
// =============================================================================

#[test]
fn test_phase33_loader_get_tensor_f32_not_found() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");

    let result = model.get_tensor_f32("nonexistent", &data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("nonexistent") || err.contains("not found"),
        "Error: {}",
        err
    );
}

include!("phase33_loader.rs");
include!("phase33_loader_02.rs");
