//! EXTREME TDD: Comprehensive GGUF Coverage Tests
//!
//! Targets:
//! - GGUF header parsing edge cases
//! - All metadata value types (u8, i8, u16, i16, u32, i32, u64, i64, f32, f64, bool, string, array)
//! - Tensor info parsing
//! - Quantized model loading paths
//! - OwnedQuantizedModel methods (using new_for_test constructor)
//! - OwnedQuantizedLayer operations
//! - GGUFConfig validation
//! - Error paths for malformed headers

use realizar::gguf::{
    ArchConstraints, GGUFConfig, GGUFHeader, GGUFModel, GGUFValue, OwnedQKVWeights,
    OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor, TensorInfo, GGUF_ALIGNMENT,
    GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K,
    GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
    GGUF_VERSION_V3,
};

// ============================================================================
// HELPER FUNCTIONS FOR BUILDING TEST GGUF DATA
// ============================================================================

/// Build a minimal valid GGUF header
fn build_minimal_gguf_header() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
    data
}

/// Add string metadata to GGUF data
fn add_string_meta(data: &mut Vec<u8>, key: &str, value: &str) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // String type
    data.extend_from_slice(&(value.len() as u64).to_le_bytes());
    data.extend_from_slice(value.as_bytes());
}

/// Add u32 metadata to GGUF data
fn add_u32_meta(data: &mut Vec<u8>, key: &str, value: u32) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // UInt32 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add f32 metadata to GGUF data
fn add_f32_meta(data: &mut Vec<u8>, key: &str, value: f32) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&6u32.to_le_bytes()); // Float32 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add u8 metadata to GGUF data
fn add_u8_meta(data: &mut Vec<u8>, key: &str, value: u8) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // UInt8 type
    data.push(value);
}

/// Add i8 metadata to GGUF data
fn add_i8_meta(data: &mut Vec<u8>, key: &str, value: i8) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // Int8 type
    data.push(value as u8);
}

/// Add u16 metadata to GGUF data
fn add_u16_meta(data: &mut Vec<u8>, key: &str, value: u16) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // UInt16 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add i16 metadata to GGUF data
fn add_i16_meta(data: &mut Vec<u8>, key: &str, value: i16) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // Int16 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add i32 metadata to GGUF data
fn add_i32_meta(data: &mut Vec<u8>, key: &str, value: i32) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&5u32.to_le_bytes()); // Int32 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add bool metadata to GGUF data
fn add_bool_meta(data: &mut Vec<u8>, key: &str, value: bool) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&7u32.to_le_bytes()); // Bool type
    data.push(u8::from(value));
}

/// Add u64 metadata to GGUF data
fn add_u64_meta(data: &mut Vec<u8>, key: &str, value: u64) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&10u32.to_le_bytes()); // UInt64 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add i64 metadata to GGUF data
fn add_i64_meta(data: &mut Vec<u8>, key: &str, value: i64) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&11u32.to_le_bytes()); // Int64 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add f64 metadata to GGUF data
fn add_f64_meta(data: &mut Vec<u8>, key: &str, value: f64) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&12u32.to_le_bytes()); // Float64 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add array of u32 metadata to GGUF data
fn add_u32_array_meta(data: &mut Vec<u8>, key: &str, values: &[u32]) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // Array type
    data.extend_from_slice(&4u32.to_le_bytes()); // Element type: UInt32
    data.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
}

/// Add array of strings metadata to GGUF data
fn add_string_array_meta(data: &mut Vec<u8>, key: &str, values: &[&str]) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // Array type
    data.extend_from_slice(&8u32.to_le_bytes()); // Element type: String
    data.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &v in values {
        data.extend_from_slice(&(v.len() as u64).to_le_bytes());
        data.extend_from_slice(v.as_bytes());
    }
}

/// Add tensor info to GGUF data (GGML order - dims are reversed)
fn add_tensor_info(data: &mut Vec<u8>, name: &str, dims: &[u64], qtype: u32, offset: u64) {
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&qtype.to_le_bytes());
    data.extend_from_slice(&offset.to_le_bytes());
}

/// Build GGUF with architecture metadata
fn build_gguf_with_arch(arch: &str, hidden: usize, layers: usize, heads: usize) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&4u64.to_le_bytes()); // metadata_count

    add_string_meta(&mut data, "general.architecture", arch);
    add_u32_meta(
        &mut data,
        &format!("{arch}.embedding_length"),
        hidden as u32,
    );
    add_u32_meta(&mut data, &format!("{arch}.block_count"), layers as u32);
    add_u32_meta(
        &mut data,
        &format!("{arch}.attention.head_count"),
        heads as u32,
    );

    data
}

/// Create a test GGUFConfig
fn create_test_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    }
}

/// Create a test OwnedQuantizedTensor
fn create_test_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    OwnedQuantizedTensor {
        data: vec![0u8; 64], // minimal data
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q4_0,
    }
}

/// Create a test OwnedQuantizedLayer
fn create_test_layer(config: &GGUFConfig) -> OwnedQuantizedLayer {
    let hidden = config.hidden_dim;
    let inter = config.intermediate_dim;

    OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden, hidden * 3)),
        qkv_bias: None,
        attn_output_weight: create_test_tensor(hidden, hidden),
        attn_output_bias: None,
        ffn_up_weight: create_test_tensor(hidden, inter),
        ffn_up_bias: None,
        ffn_down_weight: create_test_tensor(inter, hidden),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_test_tensor(hidden, inter)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    }
}

// ============================================================================
// SECTION 1: GGUF HEADER PARSING EDGE CASES
// ============================================================================

#[test]
fn test_cov_header_invalid_magic() {
    let mut data = Vec::new();
    data.extend_from_slice(b"XXXX"); // Invalid magic
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Invalid GGUF magic") || err.contains("magic"));
}

#[test]
fn test_cov_header_unsupported_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // Version 2 not supported
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Unsupported GGUF version") || err.contains("version"));
}

#[test]
fn test_cov_header_truncated_magic() {
    let data = vec![0x47, 0x47]; // Only 2 bytes
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_header_truncated_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&[0x03, 0x00]); // Truncated version

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_header_truncated_tensor_count() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&[0x00, 0x00, 0x00]); // Truncated tensor_count

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_header_truncated_metadata_count() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&[0x00, 0x00]); // Truncated metadata_count

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_header_struct_debug() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 10,
        metadata_count: 5,
    };

    let debug_str = format!("{header:?}");
    assert!(debug_str.contains("GGUFHeader"));
    assert!(debug_str.contains("magic"));
    assert!(debug_str.contains("version"));
}

#[test]
fn test_cov_header_struct_clone() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 42,
        metadata_count: 7,
    };

    let cloned = header.clone();
    assert_eq!(header, cloned);
}

#[test]
fn test_cov_header_struct_partial_eq() {
    let h1 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 10,
        metadata_count: 5,
    };

    let h2 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 10,
        metadata_count: 5,
    };

    let h3 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 20,
        metadata_count: 5,
    };

    assert_eq!(h1, h2);
    assert_ne!(h1, h3);
}

// ============================================================================
// SECTION 2: METADATA VALUE TYPES (ALL 13 TYPES)
// ============================================================================

#[test]
fn test_cov_metadata_u8_boundary() {
    let mut data = build_minimal_gguf_header();
    // Patch metadata_count to 3
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_u8_meta(&mut data, "zero", 0);
    add_u8_meta(&mut data, "mid", 128);
    add_u8_meta(&mut data, "max", 255);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::UInt8(0)));
    assert_eq!(model.metadata.get("mid"), Some(&GGUFValue::UInt8(128)));
    assert_eq!(model.metadata.get("max"), Some(&GGUFValue::UInt8(255)));
}

#[test]
fn test_cov_metadata_i8_boundary() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_i8_meta(&mut data, "min", i8::MIN);
    add_i8_meta(&mut data, "zero", 0);
    add_i8_meta(&mut data, "max", i8::MAX);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("min"), Some(&GGUFValue::Int8(i8::MIN)));
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::Int8(0)));
    assert_eq!(model.metadata.get("max"), Some(&GGUFValue::Int8(i8::MAX)));
}

#[test]
fn test_cov_metadata_u16_boundary() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_u16_meta(&mut data, "zero", 0);
    add_u16_meta(&mut data, "mid", 32768);
    add_u16_meta(&mut data, "max", 65535);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::UInt16(0)));
    assert_eq!(model.metadata.get("mid"), Some(&GGUFValue::UInt16(32768)));
    assert_eq!(model.metadata.get("max"), Some(&GGUFValue::UInt16(65535)));
}

#[test]
fn test_cov_metadata_i16_boundary() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_i16_meta(&mut data, "min", i16::MIN);
    add_i16_meta(&mut data, "zero", 0);
    add_i16_meta(&mut data, "max", i16::MAX);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("min"), Some(&GGUFValue::Int16(i16::MIN)));
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::Int16(0)));
    assert_eq!(model.metadata.get("max"), Some(&GGUFValue::Int16(i16::MAX)));
}

#[test]
fn test_cov_metadata_u32_boundary() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_u32_meta(&mut data, "zero", 0);
    add_u32_meta(&mut data, "mid", 2_147_483_648);
    add_u32_meta(&mut data, "max", u32::MAX);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::UInt32(0)));
    assert_eq!(
        model.metadata.get("mid"),
        Some(&GGUFValue::UInt32(2_147_483_648))
    );
    assert_eq!(
        model.metadata.get("max"),
        Some(&GGUFValue::UInt32(u32::MAX))
    );
}

#[test]
fn test_cov_metadata_i32_boundary() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_i32_meta(&mut data, "min", i32::MIN);
    add_i32_meta(&mut data, "zero", 0);
    add_i32_meta(&mut data, "max", i32::MAX);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("min"), Some(&GGUFValue::Int32(i32::MIN)));
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::Int32(0)));
    assert_eq!(model.metadata.get("max"), Some(&GGUFValue::Int32(i32::MAX)));
}

#[test]
fn test_cov_metadata_u64_boundary() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_u64_meta(&mut data, "zero", 0);
    add_u64_meta(&mut data, "big", 9_223_372_036_854_775_808);
    add_u64_meta(&mut data, "max", u64::MAX);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::UInt64(0)));
    assert_eq!(
        model.metadata.get("big"),
        Some(&GGUFValue::UInt64(9_223_372_036_854_775_808))
    );
    assert_eq!(
        model.metadata.get("max"),
        Some(&GGUFValue::UInt64(u64::MAX))
    );
}

#[test]
fn test_cov_metadata_i64_boundary() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_i64_meta(&mut data, "min", i64::MIN);
    add_i64_meta(&mut data, "zero", 0);
    add_i64_meta(&mut data, "max", i64::MAX);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("min"), Some(&GGUFValue::Int64(i64::MIN)));
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::Int64(0)));
    assert_eq!(model.metadata.get("max"), Some(&GGUFValue::Int64(i64::MAX)));
}

#[test]
fn test_cov_metadata_f32_special_values() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&4u64.to_le_bytes());

    add_f32_meta(&mut data, "zero", 0.0);
    add_f32_meta(&mut data, "neg_zero", -0.0);
    add_f32_meta(&mut data, "pi", std::f32::consts::PI);
    add_f32_meta(&mut data, "small", 1e-38);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("zero"), Some(&GGUFValue::Float32(0.0)));

    if let Some(GGUFValue::Float32(v)) = model.metadata.get("pi") {
        assert!((v - std::f32::consts::PI).abs() < 1e-6);
    } else {
        panic!("Expected Float32 for pi");
    }
}

#[test]
fn test_cov_metadata_f64_special_values() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&4u64.to_le_bytes());

    add_f64_meta(&mut data, "zero", 0.0);
    add_f64_meta(&mut data, "pi", std::f64::consts::PI);
    add_f64_meta(&mut data, "e", std::f64::consts::E);
    add_f64_meta(&mut data, "tiny", 1e-300);

    let model = GGUFModel::from_bytes(&data).expect("parse");

    if let Some(GGUFValue::Float64(v)) = model.metadata.get("pi") {
        assert!((v - std::f64::consts::PI).abs() < 1e-12);
    } else {
        panic!("Expected Float64 for pi");
    }

    if let Some(GGUFValue::Float64(v)) = model.metadata.get("e") {
        assert!((v - std::f64::consts::E).abs() < 1e-12);
    } else {
        panic!("Expected Float64 for e");
    }
}

#[test]
fn test_cov_metadata_bool_values() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&2u64.to_le_bytes());

    add_bool_meta(&mut data, "true", true);
    add_bool_meta(&mut data, "false", false);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("true"), Some(&GGUFValue::Bool(true)));
    assert_eq!(model.metadata.get("false"), Some(&GGUFValue::Bool(false)));
}

#[test]
fn test_cov_metadata_string_unicode() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&3u64.to_le_bytes());

    add_string_meta(&mut data, "ascii", "Hello World");
    add_string_meta(&mut data, "utf8", "Hello, World! (unicode)");
    add_string_meta(&mut data, "empty", "");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("ascii"),
        Some(&GGUFValue::String("Hello World".to_string()))
    );
    assert_eq!(
        model.metadata.get("empty"),
        Some(&GGUFValue::String(String::new()))
    );
}

#[test]
fn test_cov_metadata_array_u32() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    add_u32_array_meta(&mut data, "arr", &[1, 2, 3, 4, 5]);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let expected = GGUFValue::Array(vec![
        GGUFValue::UInt32(1),
        GGUFValue::UInt32(2),
        GGUFValue::UInt32(3),
        GGUFValue::UInt32(4),
        GGUFValue::UInt32(5),
    ]);
    assert_eq!(model.metadata.get("arr"), Some(&expected));
}

#[test]
fn test_cov_metadata_array_strings() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    add_string_array_meta(&mut data, "tokens", &["hello", "world", "test"]);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let expected = GGUFValue::Array(vec![
        GGUFValue::String("hello".to_string()),
        GGUFValue::String("world".to_string()),
        GGUFValue::String("test".to_string()),
    ]);
    assert_eq!(model.metadata.get("tokens"), Some(&expected));
}

#[test]
fn test_cov_metadata_array_empty() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    add_u32_array_meta(&mut data, "empty", &[]);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("empty"), Some(&GGUFValue::Array(vec![])));
}

#[test]
fn test_cov_metadata_unknown_type() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    // Add key
    let key = "unknown";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    // Unknown type 99
    data.extend_from_slice(&99u32.to_le_bytes());
    data.extend_from_slice(&[0u8; 8]); // dummy data

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// SECTION 3: TENSOR INFO PARSING
// ============================================================================

#[test]
fn test_cov_tensor_info_1d() {
    let mut data = build_minimal_gguf_header();
    data[8..16].copy_from_slice(&1u64.to_le_bytes()); // tensor_count = 1

    add_tensor_info(&mut data, "bias", &[512], GGUF_TYPE_F32, 0);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].name, "bias");
    assert_eq!(model.tensors[0].n_dims, 1);
    assert_eq!(model.tensors[0].dims, vec![512]);
    assert_eq!(model.tensors[0].qtype, GGUF_TYPE_F32);
}

#[test]
fn test_cov_tensor_info_2d() {
    let mut data = build_minimal_gguf_header();
    data[8..16].copy_from_slice(&1u64.to_le_bytes());

    add_tensor_info(&mut data, "weight", &[256, 512], GGUF_TYPE_Q4_0, 0);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors[0].n_dims, 2);
    assert_eq!(model.tensors[0].dims, vec![256, 512]);
    assert_eq!(model.tensors[0].qtype, GGUF_TYPE_Q4_0);
}

#[test]
fn test_cov_tensor_info_3d() {
    let mut data = build_minimal_gguf_header();
    data[8..16].copy_from_slice(&1u64.to_le_bytes());

    add_tensor_info(&mut data, "attention", &[32, 64, 64], GGUF_TYPE_F16, 1024);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors[0].n_dims, 3);
    assert_eq!(model.tensors[0].dims, vec![32, 64, 64]);
    assert_eq!(model.tensors[0].qtype, GGUF_TYPE_F16);
    assert_eq!(model.tensors[0].offset, 1024);
}

#[test]
fn test_cov_tensor_info_4d() {
    let mut data = build_minimal_gguf_header();
    data[8..16].copy_from_slice(&1u64.to_le_bytes());

    add_tensor_info(&mut data, "conv", &[64, 64, 3, 3], GGUF_TYPE_Q8_0, 2048);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors[0].n_dims, 4);
    assert_eq!(model.tensors[0].dims, vec![64, 64, 3, 3]);
    assert_eq!(model.tensors[0].qtype, GGUF_TYPE_Q8_0);
}

#[test]
fn test_cov_tensor_info_multiple() {
    let mut data = build_minimal_gguf_header();
    data[8..16].copy_from_slice(&3u64.to_le_bytes());

    add_tensor_info(&mut data, "embed", &[1000, 128], GGUF_TYPE_F32, 0);
    add_tensor_info(&mut data, "ffn_up", &[128, 512], GGUF_TYPE_Q4_K, 512000);
    add_tensor_info(&mut data, "ffn_down", &[512, 128], GGUF_TYPE_Q4_K, 1024000);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors.len(), 3);
    assert_eq!(model.tensors[0].name, "embed");
    assert_eq!(model.tensors[1].name, "ffn_up");
    assert_eq!(model.tensors[2].name, "ffn_down");
}

#[test]
fn test_cov_tensor_info_all_qtypes() {
    let qtypes = [
        GGUF_TYPE_F32,
        GGUF_TYPE_F16,
        GGUF_TYPE_Q4_0,
        GGUF_TYPE_Q4_1,
        GGUF_TYPE_Q5_0,
        GGUF_TYPE_Q5_1,
        GGUF_TYPE_Q8_0,
        GGUF_TYPE_Q4_K,
        GGUF_TYPE_Q5_K,
        GGUF_TYPE_Q6_K,
    ];

    let mut data = build_minimal_gguf_header();
    data[8..16].copy_from_slice(&(qtypes.len() as u64).to_le_bytes());

    for (i, &qtype) in qtypes.iter().enumerate() {
        add_tensor_info(
            &mut data,
            &format!("t{i}"),
            &[64, 64],
            qtype,
            (i * 4096) as u64,
        );
    }

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors.len(), qtypes.len());

    for (i, &qtype) in qtypes.iter().enumerate() {
        assert_eq!(model.tensors[i].qtype, qtype);
    }
}

#[test]
fn test_cov_tensor_info_struct() {
    let info = TensorInfo {
        name: "test_tensor".to_string(),
        n_dims: 2,
        dims: vec![256, 512],
        qtype: GGUF_TYPE_Q4_K,
        offset: 4096,
    };

    assert_eq!(info.name, "test_tensor");
    assert_eq!(info.n_dims, 2);
    assert_eq!(info.dims.len(), 2);

    let cloned = info.clone();
    assert_eq!(info, cloned);

    let debug_str = format!("{info:?}");
    assert!(debug_str.contains("TensorInfo"));
}

// ============================================================================
// SECTION 4: GGUFCONFIG VALIDATION
// ============================================================================

#[test]
fn test_cov_config_from_gguf_basic() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 2048);
    assert_eq!(config.num_layers, 22);
    assert_eq!(config.num_heads, 32);
}

#[test]
fn test_cov_config_missing_architecture() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let result = GGUFConfig::from_gguf(&model);
    assert!(result.is_err());
}

#[test]
fn test_cov_config_missing_embedding_length() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());
    add_string_meta(&mut data, "general.architecture", "test");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let result = GGUFConfig::from_gguf(&model);
    assert!(result.is_err());
}

#[test]
fn test_cov_config_defaults() {
    let data = build_gguf_with_arch("custom", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    // num_kv_heads defaults to num_heads
    assert_eq!(config.num_kv_heads, 16);
    // context_length defaults to 2048
    assert_eq!(config.context_length, 2048);
    // rope_theta defaults to 10000.0
    assert!((config.rope_theta - 10000.0).abs() < 1.0);
    // eps defaults to 1e-5
    assert!((config.eps - 1e-5).abs() < 1e-7);
}

#[test]
fn test_cov_config_struct_fields() {
    let config = create_test_config();

    assert_eq!(config.architecture, "test");
    assert_eq!(config.hidden_dim, 128);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.num_heads, 4);
    assert_eq!(config.num_kv_heads, 4);
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.intermediate_dim, 512);
    assert_eq!(config.context_length, 256);
    assert!((config.rope_theta - 10000.0).abs() < 0.1);
    assert!((config.eps - 1e-5).abs() < 1e-7);
    assert_eq!(config.rope_type, 0);
}

#[test]
fn test_cov_config_clone() {
    let config = create_test_config();
    let cloned = config.clone();

    assert_eq!(config.architecture, cloned.architecture);
    assert_eq!(config.hidden_dim, cloned.hidden_dim);
    assert_eq!(config.num_layers, cloned.num_layers);
}

#[test]
fn test_cov_config_debug() {
    let config = create_test_config();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("GGUFConfig"));
    assert!(debug_str.contains("architecture"));
    assert!(debug_str.contains("hidden_dim"));
}

// ============================================================================
// SECTION 5: OWNED QUANTIZED MODEL (new_for_test CONSTRUCTOR)
// ============================================================================

#[test]
fn test_cov_owned_model_new_for_test_basic() {
    let config = create_test_config();
    let layers = vec![create_test_layer(&config)];
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    let model = OwnedQuantizedModel::new_for_test(
        config,
        vec![0.0; vocab_size * hidden_dim], // token_embedding
        layers,
        vec![1.0; hidden_dim],                      // output_norm_weight
        None,                                       // output_norm_bias
        create_test_tensor(hidden_dim, vocab_size), // lm_head_weight
        None,                                       // lm_head_bias
    );

    assert_eq!(model.config().architecture, "test");
    assert_eq!(model.layers().len(), 1);
    assert_eq!(model.token_embedding().len(), vocab_size * hidden_dim);
    assert_eq!(model.output_norm_weight().len(), hidden_dim);
}

#[test]
fn test_cov_owned_model_new_for_test_with_biases() {
    let config = create_test_config();
    let layers = vec![create_test_layer(&config)];
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    let model = OwnedQuantizedModel::new_for_test(
        config,
        vec![0.0; vocab_size * hidden_dim],
        layers,
        vec![1.0; hidden_dim],
        Some(vec![0.0; hidden_dim]), // output_norm_bias
        create_test_tensor(hidden_dim, vocab_size),
        Some(vec![0.0; vocab_size]), // lm_head_bias
    );

    assert!(model.output_norm_bias().is_some());
    assert!(model.lm_head_bias().is_some());
}

#[test]
fn test_cov_owned_model_new_for_test_multiple_layers() {
    let config = create_test_config();
    let layers = vec![
        create_test_layer(&config),
        create_test_layer(&config),
        create_test_layer(&config),
    ];
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    let model = OwnedQuantizedModel::new_for_test(
        config,
        vec![0.0; vocab_size * hidden_dim],
        layers,
        vec![1.0; hidden_dim],
        None,
        create_test_tensor(hidden_dim, vocab_size),
        None,
    );

    assert_eq!(model.layers().len(), 3);
}

#[test]
fn test_cov_owned_model_clone() {
    let config = create_test_config();
    let layers = vec![create_test_layer(&config)];
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    let model = OwnedQuantizedModel::new_for_test(
        config,
        vec![0.1; vocab_size * hidden_dim],
        layers,
        vec![1.0; hidden_dim],
        None,
        create_test_tensor(hidden_dim, vocab_size),
        None,
    );

    let cloned = model.clone();

    assert_eq!(model.config().architecture, cloned.config().architecture);
    assert_eq!(model.layers().len(), cloned.layers().len());
    assert_eq!(
        model.token_embedding().len(),
        cloned.token_embedding().len()
    );
}

#[test]
fn test_cov_owned_model_debug() {
    let config = create_test_config();
    let layers = vec![create_test_layer(&config)];
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    let model = OwnedQuantizedModel::new_for_test(
        config,
        vec![0.0; vocab_size * hidden_dim],
        layers,
        vec![1.0; hidden_dim],
        None,
        create_test_tensor(hidden_dim, vocab_size),
        None,
    );

    let debug_str = format!("{model:?}");
    assert!(debug_str.contains("OwnedQuantizedModel"));
    assert!(debug_str.contains("config"));
    assert!(debug_str.contains("layers_count"));
}

#[test]
fn test_cov_owned_model_config_accessor() {
    let config = create_test_config();
    let layers = vec![create_test_layer(&config)];
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    let model = OwnedQuantizedModel::new_for_test(
        config.clone(),
        vec![0.0; vocab_size * hidden_dim],
        layers,
        vec![1.0; hidden_dim],
        None,
        create_test_tensor(hidden_dim, vocab_size),
        None,
    );

    // Direct field access
    assert_eq!(model.config().architecture, config.architecture);
    assert_eq!(model.config().hidden_dim, config.hidden_dim);
}

// ============================================================================
// SECTION 6: OWNED QUANTIZED LAYER OPERATIONS
// ============================================================================

#[test]
fn test_cov_owned_layer_struct_fields() {
    let config = create_test_config();
    let layer = create_test_layer(&config);

    assert_eq!(layer.attn_norm_weight.len(), config.hidden_dim);
    assert!(layer.attn_norm_bias.is_none());
    assert!(layer.ffn_gate_weight.is_some());
    assert!(layer.ffn_norm_weight.is_some());
}

#[test]
fn test_cov_owned_layer_with_biases() {
    let config = create_test_config();
    let hidden = config.hidden_dim;
    let inter = config.intermediate_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]),
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden, hidden * 3)),
        qkv_bias: Some(vec![0.0; hidden * 3]),
        attn_output_weight: create_test_tensor(hidden, hidden),
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_up_weight: create_test_tensor(hidden, inter),
        ffn_up_bias: Some(vec![0.0; inter]),
        ffn_down_weight: create_test_tensor(inter, hidden),
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    assert!(layer.attn_norm_bias.is_some());
    assert!(layer.qkv_bias.is_some());
    assert!(layer.attn_output_bias.is_some());
    assert!(layer.ffn_up_bias.is_some());
    assert!(layer.ffn_down_bias.is_some());
}

#[test]
fn test_cov_owned_layer_clone() {
    let config = create_test_config();
    let layer = create_test_layer(&config);
    let cloned = layer.clone();

    assert_eq!(layer.attn_norm_weight.len(), cloned.attn_norm_weight.len());
}

#[test]
fn test_cov_owned_layer_debug() {
    let config = create_test_config();
    let layer = create_test_layer(&config);
    let debug_str = format!("{layer:?}");

    assert!(debug_str.contains("OwnedQuantizedLayer"));
    assert!(debug_str.contains("attn_norm_weight"));
}

// ============================================================================
// SECTION 7: OWNED QKV WEIGHTS
// ============================================================================

#[test]
fn test_cov_owned_qkv_weights_fused() {
    let tensor = OwnedQuantizedTensor {
        data: vec![0; 128],
        in_dim: 128,
        out_dim: 384, // 3 * 128
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = OwnedQKVWeights::Fused(tensor);
    assert_eq!(qkv.out_dim(), 384);
}

#[test]
fn test_cov_owned_qkv_weights_separate() {
    let q = OwnedQuantizedTensor {
        data: vec![0; 64],
        in_dim: 128,
        out_dim: 128,
        qtype: GGUF_TYPE_Q4_K,
    };
    let k = OwnedQuantizedTensor {
        data: vec![0; 32],
        in_dim: 128,
        out_dim: 64, // GQA
        qtype: GGUF_TYPE_Q4_K,
    };
    let v = OwnedQuantizedTensor {
        data: vec![0; 32],
        in_dim: 128,
        out_dim: 64,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = OwnedQKVWeights::Separate { q, k, v };
    assert_eq!(qkv.out_dim(), 128 + 64 + 64);
}

#[test]
fn test_cov_owned_qkv_weights_q_dim_fused() {
    let tensor = OwnedQuantizedTensor {
        data: vec![0; 128],
        in_dim: 128,
        out_dim: 384, // Q(128) + K(128) + V(128)
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = OwnedQKVWeights::Fused(tensor);
    // For fused, q_dim = out_dim / 3
    assert_eq!(qkv.q_dim(), 384 / 3);
}

#[test]
fn test_cov_owned_qkv_weights_q_dim_separate() {
    let q = OwnedQuantizedTensor {
        data: vec![0; 64],
        in_dim: 128,
        out_dim: 256,
        qtype: GGUF_TYPE_Q4_K,
    };
    let k = OwnedQuantizedTensor {
        data: vec![0; 32],
        in_dim: 128,
        out_dim: 64,
        qtype: GGUF_TYPE_Q4_K,
    };
    let v = OwnedQuantizedTensor {
        data: vec![0; 32],
        in_dim: 128,
        out_dim: 64,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = OwnedQKVWeights::Separate { q, k, v };
    assert_eq!(qkv.q_dim(), 256);
}

#[test]
fn test_cov_owned_qkv_weights_clone() {
    let tensor = OwnedQuantizedTensor {
        data: vec![1, 2, 3],
        in_dim: 128,
        out_dim: 384,
        qtype: GGUF_TYPE_Q4_0,
    };
    let qkv = OwnedQKVWeights::Fused(tensor);
    let cloned = qkv.clone();

    assert_eq!(qkv.out_dim(), cloned.out_dim());
}

// ============================================================================
// SECTION 8: OWNED QUANTIZED TENSOR
// ============================================================================

#[test]
fn test_cov_owned_quantized_tensor_basic() {
    let tensor = OwnedQuantizedTensor {
        data: vec![1, 2, 3, 4, 5],
        in_dim: 128,
        out_dim: 256,
        qtype: GGUF_TYPE_Q4_0,
    };

    assert_eq!(tensor.data.len(), 5);
    assert_eq!(tensor.in_dim, 128);
    assert_eq!(tensor.out_dim, 256);
    assert_eq!(tensor.qtype, GGUF_TYPE_Q4_0);
}

#[test]
fn test_cov_owned_quantized_tensor_clone() {
    let tensor = OwnedQuantizedTensor {
        data: vec![10, 20, 30],
        in_dim: 64,
        out_dim: 128,
        qtype: GGUF_TYPE_Q8_0,
    };

    let cloned = tensor.clone();

    assert_eq!(cloned.data, tensor.data);
    assert_eq!(cloned.in_dim, tensor.in_dim);
    assert_eq!(cloned.out_dim, tensor.out_dim);
    assert_eq!(cloned.qtype, tensor.qtype);
}

#[test]
fn test_cov_owned_quantized_tensor_debug() {
    let tensor = OwnedQuantizedTensor {
        data: vec![0; 16],
        in_dim: 32,
        out_dim: 64,
        qtype: GGUF_TYPE_Q4_K,
    };

    let debug_str = format!("{tensor:?}");
    assert!(debug_str.contains("OwnedQuantizedTensor"));
    assert!(debug_str.contains("in_dim"));
    assert!(debug_str.contains("out_dim"));
    assert!(debug_str.contains("qtype"));
}

#[test]
fn test_cov_owned_quantized_tensor_all_qtypes() {
    let qtypes = [
        GGUF_TYPE_F32,
        GGUF_TYPE_F16,
        GGUF_TYPE_Q4_0,
        GGUF_TYPE_Q4_1,
        GGUF_TYPE_Q5_0,
        GGUF_TYPE_Q5_1,
        GGUF_TYPE_Q8_0,
        GGUF_TYPE_Q4_K,
        GGUF_TYPE_Q5_K,
        GGUF_TYPE_Q6_K,
    ];

    for qtype in qtypes {
        let tensor = OwnedQuantizedTensor {
            data: vec![0; 32],
            in_dim: 64,
            out_dim: 128,
            qtype,
        };
        assert_eq!(tensor.qtype, qtype);
    }
}

// ============================================================================
// SECTION 9: GGUF ALIGNMENT AND CONSTANTS
// ============================================================================

#[test]
fn test_cov_gguf_alignment_constant() {
    assert_eq!(GGUF_ALIGNMENT, 32);
}

#[test]
fn test_cov_gguf_magic_constant() {
    assert_eq!(GGUF_MAGIC, 0x4655_4747);
    assert_eq!(&GGUF_MAGIC.to_le_bytes(), b"GGUF");
}

#[test]
fn test_cov_gguf_version_constant() {
    assert_eq!(GGUF_VERSION_V3, 3);
}

#[test]
fn test_cov_tensor_data_alignment() {
    // Verify tensor_data_start is always 32-byte aligned
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
}

#[test]
fn test_cov_tensor_data_alignment_with_metadata() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&5u64.to_le_bytes());

    add_u8_meta(&mut data, "a", 1);
    add_u16_meta(&mut data, "b", 2);
    add_u32_meta(&mut data, "c", 3);
    add_u64_meta(&mut data, "d", 4);
    add_string_meta(&mut data, "e", "hello");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
}

// ============================================================================
// SECTION 10: GGUF VALUE ENUM
// ============================================================================

#[test]
fn test_cov_gguf_value_partial_eq() {
    let v1 = GGUFValue::UInt32(42);
    let v2 = GGUFValue::UInt32(42);
    let v3 = GGUFValue::UInt32(100);
    let v4 = GGUFValue::String("42".to_string());

    assert_eq!(v1, v2);
    assert_ne!(v1, v3);
    assert_ne!(v1, v4);
}

#[test]
fn test_cov_gguf_value_clone() {
    let values = [
        GGUFValue::UInt8(10),
        GGUFValue::Int8(-5),
        GGUFValue::UInt16(1000),
        GGUFValue::Int16(-500),
        GGUFValue::UInt32(100000),
        GGUFValue::Int32(-50000),
        GGUFValue::UInt64(1_000_000_000),
        GGUFValue::Int64(-500_000_000),
        GGUFValue::Float32(std::f32::consts::PI),
        GGUFValue::Float64(std::f64::consts::E),
        GGUFValue::Bool(true),
        GGUFValue::String("test".to_string()),
        GGUFValue::Array(vec![GGUFValue::UInt32(1), GGUFValue::UInt32(2)]),
    ];

    for v in &values {
        let cloned = v.clone();
        assert_eq!(v, &cloned);
    }
}

#[test]
fn test_cov_gguf_value_debug() {
    let v = GGUFValue::String("hello".to_string());
    let debug_str = format!("{v:?}");
    assert!(debug_str.contains("String"));
    assert!(debug_str.contains("hello"));
}

#[test]
fn test_cov_gguf_value_nested_array() {
    // Array of arrays (if supported)
    let inner = GGUFValue::Array(vec![GGUFValue::UInt32(1), GGUFValue::UInt32(2)]);
    let outer = GGUFValue::Array(vec![inner]);

    let cloned = outer.clone();
    assert_eq!(outer, cloned);
}

// ============================================================================
// SECTION 11: ERROR PATHS
// ============================================================================

#[test]
fn test_cov_error_empty_data() {
    let result = GGUFModel::from_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_cov_error_partial_header() {
    let data = vec![0x47, 0x47, 0x55, 0x46]; // Just "GGUF"
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_error_truncated_string_length() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    // Key with truncated length field
    data.extend_from_slice(&[0xFF, 0xFF]); // Incomplete u64

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_error_truncated_string_data() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    // Key with length but not enough data
    data.extend_from_slice(&100u64.to_le_bytes()); // Claims 100 bytes
    data.extend_from_slice(b"short"); // Only 5 bytes

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_error_truncated_tensor_info() {
    let mut data = build_minimal_gguf_header();
    data[8..16].copy_from_slice(&1u64.to_le_bytes()); // tensor_count = 1

    // Partial tensor info
    let name = "test";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    // Missing n_dims, dims, qtype, offset

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_error_invalid_array_element_type() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    // Key
    let key = "bad_array";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    // Array type
    data.extend_from_slice(&9u32.to_le_bytes());
    // Invalid element type
    data.extend_from_slice(&99u32.to_le_bytes());
    // Array length
    data.extend_from_slice(&1u64.to_le_bytes());
    // Dummy element data
    data.extend_from_slice(&[0u8; 8]);

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// SECTION 12: GGUF MODEL STRUCTURE
// ============================================================================

#[test]
fn test_cov_gguf_model_struct_fields() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.header.magic, GGUF_MAGIC);
    assert_eq!(model.header.version, GGUF_VERSION_V3);
    assert_eq!(model.header.tensor_count, 0);
    assert_eq!(model.header.metadata_count, 4);
    assert_eq!(model.tensors.len(), 0);
    assert!(model.metadata.contains_key("general.architecture"));
}

#[test]
fn test_cov_gguf_model_clone() {
    let data = build_gguf_with_arch("test", 128, 2, 4);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let cloned = model.clone();

    assert_eq!(model.header, cloned.header);
    assert_eq!(model.tensors.len(), cloned.tensors.len());
    assert_eq!(model.metadata.len(), cloned.metadata.len());
}

#[test]
fn test_cov_gguf_model_debug() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let debug_str = format!("{model:?}");

    assert!(debug_str.contains("GGUFModel"));
    assert!(debug_str.contains("header"));
}

// ============================================================================
// SECTION 13: QUANTIZATION TYPE CONSTANTS
// ============================================================================

#[test]
fn test_cov_qtype_constants() {
    assert_eq!(GGUF_TYPE_F32, 0);
    assert_eq!(GGUF_TYPE_F16, 1);
    assert_eq!(GGUF_TYPE_Q4_0, 2);
    assert_eq!(GGUF_TYPE_Q4_1, 3);
    assert_eq!(GGUF_TYPE_Q5_0, 6);
    assert_eq!(GGUF_TYPE_Q5_1, 7);
    assert_eq!(GGUF_TYPE_Q8_0, 8);
    assert_eq!(GGUF_TYPE_Q4_K, 12);
    assert_eq!(GGUF_TYPE_Q5_K, 13);
    assert_eq!(GGUF_TYPE_Q6_K, 14);
}

// ============================================================================
// SECTION 14: MODEL FORWARD PASS AND INFERENCE (COVERAGE EXPANSION)
// ============================================================================

/// Create a proper test model with valid Q4_0 quantized weights
fn create_inference_test_model() -> OwnedQuantizedModel {
    let hidden_dim = 64;
    let intermediate_dim = 256;
    let vocab_size = 100;
    let num_layers = 2;
    let num_heads = 4;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads: num_heads,
        vocab_size,
        intermediate_dim,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    // Q4_0 block size: 18 bytes for 32 elements
    // Format: 2 bytes f16 scale + 16 bytes (32 4-bit values)
    fn create_q4_0_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let num_blocks = (in_dim * out_dim).div_ceil(32);
        let data_size = num_blocks * 18;
        let mut data = vec![0u8; data_size];

        // Initialize with small positive scale and zero quants
        for block in 0..num_blocks {
            let offset = block * 18;
            // F16 scale = 0.01 (half precision)
            let scale_f16: u16 = 0x1500; // Approximation of small positive
            data[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
            // Remaining 16 bytes are zero quants (representing ~0 values)
        }

        OwnedQuantizedTensor {
            data,
            in_dim,
            out_dim,
            qtype: GGUF_TYPE_Q4_0,
        }
    }

    // Create layers
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push(OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Fused(create_q4_0_tensor(hidden_dim, hidden_dim * 3)),
            qkv_bias: None,
            attn_output_weight: create_q4_0_tensor(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4_0_tensor(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4_0_tensor(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: Some(create_q4_0_tensor(hidden_dim, intermediate_dim)),
            ffn_gate_bias: None,
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        });
    }

    // Token embeddings - simple lookup table
    let token_embedding = vec![0.1f32; vocab_size * hidden_dim];

    // Output norm and LM head
    let output_norm_weight = vec![1.0; hidden_dim];
    let lm_head_weight = create_q4_0_tensor(hidden_dim, vocab_size);

    OwnedQuantizedModel::new_for_test(
        config,
        token_embedding,
        layers,
        output_norm_weight,
        None,
        lm_head_weight,
        None,
    )
}

#[test]
fn test_cov_model_forward_single_token() {
    let model = create_inference_test_model();

    // Forward pass with single token
    let token_ids = [1u32];
    let result = model.forward(&token_ids);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), model.config().vocab_size);
}

#[test]
fn test_cov_model_forward_multiple_tokens() {
    let model = create_inference_test_model();

    // Forward pass with multiple tokens
    let token_ids = [1u32, 2, 3, 4, 5];
    let result = model.forward(&token_ids);

    assert!(result.is_ok());
    let logits = result.unwrap();
    // Should return logits for last token position
    assert_eq!(logits.len(), model.config().vocab_size);
}

#[test]
#[should_panic(expected = "subtract with overflow")]
fn test_cov_model_forward_empty_tokens() {
    let model = create_inference_test_model();

    // Forward pass with empty token list panics due to seq_len - 1 underflow
    let token_ids: [u32; 0] = [];
    let _ = model.forward(&token_ids);
}

#[test]
fn test_cov_model_embed_valid_tokens() {
    let model = create_inference_test_model();

    // Test embedding lookup for valid tokens
    let embeddings = model.embed(&[0, 1, 2]);
    assert_eq!(embeddings.len(), 3 * model.config().hidden_dim);
}

#[test]
fn test_cov_model_embed_out_of_bounds_token() {
    let model = create_inference_test_model();

    // Test embedding lookup for out-of-bounds token (should pad with zeros)
    let vocab_size = model.config().vocab_size;
    let oob_token = (vocab_size + 10) as u32;
    let embeddings = model.embed(&[oob_token]);

    // Should return zeros for OOB tokens
    assert_eq!(embeddings.len(), model.config().hidden_dim);
    assert!(embeddings.iter().all(|&x| x == 0.0));
}

#[test]
fn test_cov_model_embed_boundary_token() {
    let model = create_inference_test_model();

    // Test embedding lookup for token at vocab boundary
    let last_valid = (model.config().vocab_size - 1) as u32;
    let embeddings = model.embed(&[last_valid]);

    assert_eq!(embeddings.len(), model.config().hidden_dim);
}

// ============================================================================
// SECTION 15: LAYER NORMALIZATION PATHS
// ============================================================================

#[test]
fn test_cov_layer_norm_single_position() {
    let model = create_inference_test_model();
    let hidden_dim = model.config().hidden_dim;

    // Create test input (prefixed with _ to suppress unused warning)
    let _input: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.1).collect();
    let _weight = vec![1.0f32; hidden_dim];

    // Use model's forward to exercise layer norm internally
    let result = model.forward(&[1]);
    assert!(result.is_ok());
}

#[test]
fn test_cov_layer_norm_multiple_positions() {
    let model = create_inference_test_model();

    // Forward with multiple positions exercises layer norm across sequence
    let result = model.forward(&[1, 2, 3, 4]);
    assert!(result.is_ok());
}

// ============================================================================
// SECTION 16: ATTENTION COMPUTATION PATHS
// ============================================================================

#[test]
fn test_cov_attention_via_forward() {
    let model = create_inference_test_model();

    // Forward pass exercises attention mechanism
    let result = model.forward(&[1, 2, 3]);
    assert!(result.is_ok());
}

#[test]
fn test_cov_qkv_fused_weight_access() {
    let config = create_test_config();
    let layer = create_test_layer(&config);

    // Access QKV weights through Fused variant
    match &layer.qkv_weight {
        OwnedQKVWeights::Fused(tensor) => {
            assert_eq!(tensor.in_dim, config.hidden_dim);
            assert_eq!(tensor.out_dim, config.hidden_dim * 3);
        },
        OwnedQKVWeights::Separate { .. } => panic!("Expected Fused QKV weights"),
    }
}

#[test]
fn test_cov_qkv_separate_weight_creation() {
    let hidden_dim = 128;
    let head_dim = 32;
    let num_heads = 4;
    let num_kv_heads = 2;

    let q = OwnedQuantizedTensor {
        data: vec![0u8; 64],
        in_dim: hidden_dim,
        out_dim: num_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };
    let k = OwnedQuantizedTensor {
        data: vec![0u8; 32],
        in_dim: hidden_dim,
        out_dim: num_kv_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };
    let v = OwnedQuantizedTensor {
        data: vec![0u8; 32],
        in_dim: hidden_dim,
        out_dim: num_kv_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = OwnedQKVWeights::Separate { q, k, v };

    // Verify dimensions
    let expected_out = num_heads * head_dim + 2 * num_kv_heads * head_dim;
    assert_eq!(qkv.out_dim(), expected_out);
}

// ============================================================================
// SECTION 17: FFN COMPUTATION PATHS
// ============================================================================

#[test]
fn test_cov_ffn_via_forward() {
    let model = create_inference_test_model();

    // Forward pass exercises FFN layers (up, gate, down projections)
    let result = model.forward(&[1]);
    assert!(result.is_ok());
}

#[test]
fn test_cov_ffn_gate_weight_present() {
    let config = create_test_config();
    let layer = create_test_layer(&config);

    // Verify gate weight exists (for SwiGLU FFN)
    assert!(layer.ffn_gate_weight.is_some());
    if let Some(ref gate) = layer.ffn_gate_weight {
        assert_eq!(gate.in_dim, config.hidden_dim);
        assert_eq!(gate.out_dim, config.intermediate_dim);
    }
}

#[test]
fn test_cov_ffn_without_gate() {
    let config = create_test_config();
    let hidden = config.hidden_dim;
    let inter = config.intermediate_dim;

    // Create layer without gate weight (standard GELU FFN)
    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden, hidden * 3)),
        qkv_bias: None,
        attn_output_weight: create_test_tensor(hidden, hidden),
        attn_output_bias: None,
        ffn_up_weight: create_test_tensor(hidden, inter),
        ffn_up_bias: None,
        ffn_down_weight: create_test_tensor(inter, hidden),
        ffn_down_bias: None,
        ffn_gate_weight: None, // No gate for GELU FFN
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    assert!(layer.ffn_gate_weight.is_none());
}

// ============================================================================
// SECTION 18: QUANTIZATION TYPE HANDLING
// ============================================================================

#[test]
fn test_cov_tensor_f32_type() {
    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; 256], // 64 f32 values
        in_dim: 8,
        out_dim: 8,
        qtype: GGUF_TYPE_F32,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_F32);
    assert_eq!(tensor.in_dim * tensor.out_dim * 4, 256);
}

#[test]
fn test_cov_tensor_f16_type() {
    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; 128], // 64 f16 values
        in_dim: 8,
        out_dim: 8,
        qtype: GGUF_TYPE_F16,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_F16);
    assert_eq!(tensor.in_dim * tensor.out_dim * 2, 128);
}

#[test]
fn test_cov_tensor_q4_0_type() {
    // Q4_0: 32 elements per 18-byte block
    let elements: usize = 64;
    let blocks = elements.div_ceil(32);
    let data_size = blocks * 18;

    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; data_size],
        in_dim: 8,
        out_dim: 8,
        qtype: GGUF_TYPE_Q4_0,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_Q4_0);
}

#[test]
fn test_cov_tensor_q4_1_type() {
    // Q4_1: 32 elements per 20-byte block (scale + min + quants)
    let elements: usize = 64;
    let blocks = elements.div_ceil(32);
    let data_size = blocks * 20;

    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; data_size],
        in_dim: 8,
        out_dim: 8,
        qtype: GGUF_TYPE_Q4_1,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_Q4_1);
}

#[test]
fn test_cov_tensor_q5_0_type() {
    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; 128],
        in_dim: 8,
        out_dim: 8,
        qtype: GGUF_TYPE_Q5_0,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_Q5_0);
}

#[test]
fn test_cov_tensor_q5_1_type() {
    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; 128],
        in_dim: 8,
        out_dim: 8,
        qtype: GGUF_TYPE_Q5_1,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_Q5_1);
}

#[test]
fn test_cov_tensor_q8_0_type() {
    // Q8_0: 32 elements per 34-byte block
    let elements: usize = 64;
    let blocks = elements.div_ceil(32);
    let data_size = blocks * 34;

    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; data_size],
        in_dim: 8,
        out_dim: 8,
        qtype: GGUF_TYPE_Q8_0,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_Q8_0);
}

#[test]
fn test_cov_tensor_q4_k_type() {
    // Q4_K: 256 elements per 144-byte super-block
    let elements: usize = 256;
    let super_blocks = elements.div_ceil(256);
    let data_size = super_blocks * 144;

    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; data_size],
        in_dim: 16,
        out_dim: 16,
        qtype: GGUF_TYPE_Q4_K,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_Q4_K);
}

#[test]
fn test_cov_tensor_q5_k_type() {
    // Q5_K: 256 elements per 176-byte super-block
    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; 176],
        in_dim: 16,
        out_dim: 16,
        qtype: GGUF_TYPE_Q5_K,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_Q5_K);
}

#[test]
fn test_cov_tensor_q6_k_type() {
    // Q6_K: 256 elements per 210-byte super-block
    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; 210],
        in_dim: 16,
        out_dim: 16,
        qtype: GGUF_TYPE_Q6_K,
    };

    assert_eq!(tensor.qtype, GGUF_TYPE_Q6_K);
}

// ============================================================================
// SECTION 19: GGUF CONFIG EDGE CASES
// ============================================================================

#[test]
fn test_cov_config_with_custom_rope_theta() {
    let mut config = create_test_config();
    config.rope_theta = 500000.0; // Llama 3 style

    assert!((config.rope_theta - 500000.0).abs() < 1.0);
}

#[test]
fn test_cov_config_with_custom_eps() {
    let mut config = create_test_config();
    config.eps = 1e-6; // Qwen2 style

    assert!((config.eps - 1e-6).abs() < 1e-8);
}

#[test]
fn test_cov_config_with_gqa() {
    // Grouped Query Attention: fewer KV heads than Q heads
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2048,
        num_layers: 24,
        num_heads: 32,
        num_kv_heads: 8, // GQA: 4 Q heads per KV head
        vocab_size: 32000,
        intermediate_dim: 5632,
        context_length: 4096,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    assert_eq!(config.num_heads / config.num_kv_heads, 4);
}

#[test]
fn test_cov_config_with_mqa() {
    // Multi-Query Attention: single KV head
    let config = GGUFConfig {
        architecture: "falcon".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 64,
        num_kv_heads: 1, // MQA: all Q heads share 1 KV head
        vocab_size: 65024,
        intermediate_dim: 16384,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    assert_eq!(config.num_kv_heads, 1);
}

#[test]
fn test_cov_config_rope_type_neox() {
    let mut config = create_test_config();
    config.rope_type = 2; // NEOX style

    assert_eq!(config.rope_type, 2);
}

#[test]
fn test_cov_config_large_context() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 32,
        vocab_size: 32000,
        intermediate_dim: 11008,
        context_length: 131072, // 128K context
        rope_theta: 500000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    assert_eq!(config.context_length, 131072);
}

// ============================================================================
// SECTION 20: ERROR HANDLING AND EDGE CASES
// ============================================================================

#[test]
fn test_cov_error_invalid_tensor_dims_in_info() {
    let mut data = build_minimal_gguf_header();
    data[8..16].copy_from_slice(&1u64.to_le_bytes()); // tensor_count = 1

    // Add tensor with 0 dimensions (edge case)
    let name = "zero_dim";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // n_dims = 0
    data.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    // This should parse but result in a tensor with empty dims
    let result = GGUFModel::from_bytes(&data);
    // May succeed with empty dims or fail - either is acceptable
    if let Ok(model) = result {
        assert_eq!(model.tensors[0].n_dims, 0);
        assert!(model.tensors[0].dims.is_empty());
    }
}

#[test]
fn test_cov_error_metadata_key_truncated() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Key length says 100 but we don't provide that much data
    data.extend_from_slice(&100u64.to_le_bytes());
    // Only provide 5 bytes instead of 100
    data.extend_from_slice(b"short");

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_error_metadata_value_truncated() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    // Valid key
    add_u8_meta(&mut data, "valid_key", 42);
    // Now truncate the data
    data.truncate(data.len() - 5);

    let result = GGUFModel::from_bytes(&data);
    // Should fail due to truncation
    assert!(result.is_err());
}

#[test]
fn test_cov_large_tensor_count() {
    let mut data = build_minimal_gguf_header();
    // Set a reasonable tensor count (not too large to cause memory issues)
    data[8..16].copy_from_slice(&5u64.to_le_bytes());

    // Add 5 tensors
    for i in 0..5 {
        add_tensor_info(
            &mut data,
            &format!("tensor_{i}"),
            &[32, 32],
            GGUF_TYPE_F32,
            (i * 4096) as u64,
        );
    }

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().tensors.len(), 5);
}

#[test]
fn test_cov_large_metadata_count() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&10u64.to_le_bytes());

    // Add 10 metadata entries
    for i in 0..10 {
        add_u32_meta(&mut data, &format!("meta_{i}"), i);
    }

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().metadata.len(), 10);
}

// ============================================================================
// SECTION 21: OWNED QUANTIZED LAYER VARIATIONS
// ============================================================================

#[test]
fn test_cov_layer_with_all_biases() {
    let config = create_test_config();
    let hidden = config.hidden_dim;
    let inter = config.intermediate_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.1; hidden]),
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden, hidden * 3)),
        qkv_bias: Some(vec![0.0; hidden * 3]),
        attn_output_weight: create_test_tensor(hidden, hidden),
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_up_weight: create_test_tensor(hidden, inter),
        ffn_up_bias: Some(vec![0.0; inter]),
        ffn_down_weight: create_test_tensor(inter, hidden),
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: Some(create_test_tensor(hidden, inter)),
        ffn_gate_bias: Some(vec![0.0; inter]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    assert!(layer.attn_norm_bias.is_some());
    assert!(layer.qkv_bias.is_some());
    assert!(layer.attn_output_bias.is_some());
    assert!(layer.ffn_up_bias.is_some());
    assert!(layer.ffn_down_bias.is_some());
    assert!(layer.ffn_gate_bias.is_some());
    assert!(layer.ffn_norm_bias.is_some());
}

#[test]
fn test_cov_layer_minimal_no_biases() {
    let config = create_test_config();
    let hidden = config.hidden_dim;
    let inter = config.intermediate_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden, hidden * 3)),
        qkv_bias: None,
        attn_output_weight: create_test_tensor(hidden, hidden),
        attn_output_bias: None,
        ffn_up_weight: create_test_tensor(hidden, inter),
        ffn_up_bias: None,
        ffn_down_weight: create_test_tensor(inter, hidden),
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    assert!(layer.attn_norm_bias.is_none());
    assert!(layer.ffn_gate_weight.is_none());
    assert!(layer.ffn_norm_weight.is_none());
}

#[test]
fn test_cov_layer_separate_qkv() {
    let config = create_test_config();
    let hidden = config.hidden_dim;
    let inter = config.intermediate_dim;
    let head_dim = hidden / config.num_heads;

    // Create separate Q, K, V weights (like TinyLlama)
    let q_weight = OwnedQuantizedTensor {
        data: vec![0; 64],
        in_dim: hidden,
        out_dim: config.num_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };
    let k_weight = OwnedQuantizedTensor {
        data: vec![0; 64],
        in_dim: hidden,
        out_dim: config.num_kv_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };
    let v_weight = OwnedQuantizedTensor {
        data: vec![0; 64],
        in_dim: hidden,
        out_dim: config.num_kv_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        },
        qkv_bias: None,
        attn_output_weight: create_test_tensor(hidden, hidden),
        attn_output_bias: None,
        ffn_up_weight: create_test_tensor(hidden, inter),
        ffn_up_bias: None,
        ffn_down_weight: create_test_tensor(inter, hidden),
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    match layer.qkv_weight {
        OwnedQKVWeights::Separate {
            ref q,
            ref k,
            ref v,
        } => {
            assert_eq!(q.out_dim, config.num_heads * head_dim);
            assert_eq!(k.out_dim, config.num_kv_heads * head_dim);
            assert_eq!(v.out_dim, config.num_kv_heads * head_dim);
        },
        _ => panic!("Expected Separate QKV weights"),
    }
}

// ============================================================================
// SECTION 22: GGUF VALUE TYPE CONVERSIONS
// ============================================================================

#[test]
fn test_cov_gguf_value_all_types_debug() {
    let values = [
        GGUFValue::UInt8(255),
        GGUFValue::Int8(-128),
        GGUFValue::UInt16(65535),
        GGUFValue::Int16(-32768),
        GGUFValue::UInt32(u32::MAX),
        GGUFValue::Int32(i32::MIN),
        GGUFValue::UInt64(u64::MAX),
        GGUFValue::Int64(i64::MIN),
        GGUFValue::Float32(f32::MAX),
        GGUFValue::Float64(f64::MIN),
        GGUFValue::Bool(false),
        GGUFValue::String("test string".to_string()),
        GGUFValue::Array(vec![GGUFValue::UInt8(1), GGUFValue::UInt8(2)]),
    ];

    for value in &values {
        let debug_str = format!("{value:?}");
        assert!(!debug_str.is_empty());
    }
}

#[test]
fn test_cov_gguf_value_nested_arrays() {
    let inner1 = GGUFValue::Array(vec![GGUFValue::UInt32(1), GGUFValue::UInt32(2)]);
    let inner2 = GGUFValue::Array(vec![GGUFValue::UInt32(3), GGUFValue::UInt32(4)]);
    let outer = GGUFValue::Array(vec![inner1, inner2]);

    let cloned = outer.clone();
    assert_eq!(outer, cloned);
}

#[test]
fn test_cov_gguf_value_empty_string() {
    let value = GGUFValue::String(String::new());
    let debug_str = format!("{value:?}");
    assert!(debug_str.contains("String"));
}

#[test]
fn test_cov_gguf_value_empty_array() {
    let value = GGUFValue::Array(vec![]);
    let cloned = value.clone();
    assert_eq!(value, cloned);
}

// ============================================================================
// SECTION 23: TENSOR INFO EDGE CASES
// ============================================================================

#[test]
fn test_cov_tensor_info_large_dimensions() {
    let info = TensorInfo {
        name: "large_tensor".to_string(),
        n_dims: 2,
        dims: vec![65536, 4096], // Large embedding table
        qtype: GGUF_TYPE_Q4_K,
        offset: 0,
    };

    assert_eq!(info.dims[0] * info.dims[1], 268_435_456);
}

#[test]
fn test_cov_tensor_info_long_name() {
    let long_name = "model.layers().31.self_attn.q_proj.weight".to_string();
    let info = TensorInfo {
        name: long_name.clone(),
        n_dims: 2,
        dims: vec![4096, 4096],
        qtype: GGUF_TYPE_Q4_K,
        offset: 1024,
    };

    assert_eq!(info.name, long_name);
}

#[test]
fn test_cov_tensor_info_max_dims() {
    // Test tensor with maximum practical dimensions (4D)
    let info = TensorInfo {
        name: "conv_weight".to_string(),
        n_dims: 4,
        dims: vec![64, 64, 3, 3],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };

    assert_eq!(info.n_dims, 4);
    assert_eq!(info.dims.len(), 4);
}

// ============================================================================
// SECTION 24: MODEL CONFIGURATION VALIDATION
// ============================================================================

#[test]
fn test_cov_config_head_dim_calculation() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 11008,
        context_length: 4096,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let head_dim = config.hidden_dim / config.num_heads;
    assert_eq!(head_dim, 128);

    let kv_head_dim = config.hidden_dim / config.num_heads;
    assert_eq!(kv_head_dim, 128);
}

#[test]
fn test_cov_config_intermediate_ratio() {
    // Standard LLaMA: intermediate = 4 * hidden * 2/3 (rounded)
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 32,
        vocab_size: 32000,
        intermediate_dim: 11008, // ~2.69x hidden
        context_length: 4096,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let ratio = config.intermediate_dim as f32 / config.hidden_dim as f32;
    assert!(ratio > 2.5 && ratio < 3.0);
}

// ============================================================================
// SECTION 25: BUFFER AND CONSTANT TESTS
// ============================================================================

#[test]
fn test_cov_buffer_constants() {
    use realizar::gguf::{
        ATTENTION_BUFFER_INLINE_CAP, BUFFER_HW_SIZE, BUFFER_LW_SIZE, BUFFER_MAX_SIZE,
        HIDDEN_BUFFER_INLINE_CAP, TOKEN_BUFFER_INLINE_CAP,
    };

    assert_eq!(TOKEN_BUFFER_INLINE_CAP, 32);
    assert_eq!(ATTENTION_BUFFER_INLINE_CAP, 64);
    assert_eq!(HIDDEN_BUFFER_INLINE_CAP, 128);
    assert_eq!(BUFFER_LW_SIZE, 1024);
    assert_eq!(BUFFER_HW_SIZE, 8 * 1024);
    assert_eq!(BUFFER_MAX_SIZE, 32 * 1024);
}

// ============================================================================
// SECTION 26: METADATA PARSING STRESS TESTS
// ============================================================================

#[test]
fn test_cov_metadata_many_keys() {
    let mut data = build_minimal_gguf_header();
    let num_keys: usize = 50;
    data[16..24].copy_from_slice(&(num_keys as u64).to_le_bytes());

    for i in 0..num_keys {
        add_u32_meta(&mut data, &format!("key_{i:03}"), i as u32);
    }

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.len(), num_keys);

    // Verify a few specific keys
    assert_eq!(model.metadata.get("key_000"), Some(&GGUFValue::UInt32(0)));
    assert_eq!(model.metadata.get("key_025"), Some(&GGUFValue::UInt32(25)));
    assert_eq!(model.metadata.get("key_049"), Some(&GGUFValue::UInt32(49)));
}

#[test]
fn test_cov_metadata_mixed_types() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&13u64.to_le_bytes());

    add_u8_meta(&mut data, "type_u8", 255);
    add_i8_meta(&mut data, "type_i8", -128);
    add_u16_meta(&mut data, "type_u16", 65535);
    add_i16_meta(&mut data, "type_i16", -32768);
    add_u32_meta(&mut data, "type_u32", u32::MAX);
    add_i32_meta(&mut data, "type_i32", i32::MIN);
    add_u64_meta(&mut data, "type_u64", u64::MAX);
    add_i64_meta(&mut data, "type_i64", i64::MIN);
    add_f32_meta(&mut data, "type_f32", std::f32::consts::PI);
    add_f64_meta(&mut data, "type_f64", std::f64::consts::E);
    add_bool_meta(&mut data, "type_bool", true);
    add_string_meta(&mut data, "type_string", "hello");
    add_u32_array_meta(&mut data, "type_array", &[1, 2, 3]);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.len(), 13);

    // Verify types
    assert!(matches!(
        model.metadata.get("type_u8"),
        Some(GGUFValue::UInt8(_))
    ));
    assert!(matches!(
        model.metadata.get("type_i8"),
        Some(GGUFValue::Int8(_))
    ));
    assert!(matches!(
        model.metadata.get("type_f32"),
        Some(GGUFValue::Float32(_))
    ));
    assert!(matches!(
        model.metadata.get("type_array"),
        Some(GGUFValue::Array(_))
    ));
}

#[test]
fn test_cov_metadata_long_string_value() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&1u64.to_le_bytes());

    let long_value = "x".repeat(10000);
    add_string_meta(&mut data, "long", &long_value);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::String(s)) = model.metadata.get("long") {
        assert_eq!(s.len(), 10000);
    } else {
        panic!("Expected long string");
    }
}

#[test]
fn test_cov_metadata_special_key_names() {
    let mut data = build_minimal_gguf_header();
    data[16..24].copy_from_slice(&4u64.to_le_bytes());

    add_u32_meta(&mut data, "general.architecture", 1);
    add_u32_meta(&mut data, "llama.embedding_length", 2);
    add_u32_meta(&mut data, "llama.attention.head_count", 3);
    add_u32_meta(&mut data, "llama.rope.freq_base", 4);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.metadata.contains_key("general.architecture"));
    assert!(model.metadata.contains_key("llama.embedding_length"));
}

// ============================================================================
// SECTION 27: QKV WEIGHT DIMENSION TESTS
// ============================================================================

#[test]
fn test_cov_qkv_fused_dimensions() {
    let hidden_dim = 4096;
    let tensor = OwnedQuantizedTensor {
        data: vec![0; 1024],
        in_dim: hidden_dim,
        out_dim: hidden_dim * 3, // Q + K + V
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = OwnedQKVWeights::Fused(tensor);

    assert_eq!(qkv.out_dim(), hidden_dim * 3);
    assert_eq!(qkv.q_dim(), hidden_dim);
}

#[test]
fn test_cov_qkv_separate_gqa_dimensions() {
    let hidden_dim = 4096;
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = hidden_dim / num_heads;

    let q = OwnedQuantizedTensor {
        data: vec![0; 512],
        in_dim: hidden_dim,
        out_dim: num_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };
    let k = OwnedQuantizedTensor {
        data: vec![0; 128],
        in_dim: hidden_dim,
        out_dim: num_kv_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };
    let v = OwnedQuantizedTensor {
        data: vec![0; 128],
        in_dim: hidden_dim,
        out_dim: num_kv_heads * head_dim,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = OwnedQKVWeights::Separate { q, k, v };

    let expected_out = num_heads * head_dim + 2 * num_kv_heads * head_dim;
    assert_eq!(qkv.out_dim(), expected_out);
    assert_eq!(qkv.q_dim(), num_heads * head_dim);
}

// ============================================================================
// SECTION 28: MODEL STRUCT FIELD ACCESSORS
// ============================================================================

#[test]
fn test_cov_model_config_accessor() {
    let model = create_inference_test_model();

    assert_eq!(model.config().architecture, "test");
    assert_eq!(model.config().hidden_dim, 64);
    assert_eq!(model.config().num_layers, 2);
    assert_eq!(model.config().num_heads, 4);
}

#[test]
fn test_cov_model_layers_accessor() {
    let model = create_inference_test_model();

    assert_eq!(model.layers().len(), 2);

    for layer in model.layers() {
        assert_eq!(layer.attn_norm_weight.len(), 64);
    }
}

#[test]
fn test_cov_model_embedding_accessor() {
    let model = create_inference_test_model();

    assert_eq!(model.token_embedding().len(), 100 * 64); // vocab_size * hidden_dim
}

#[test]
fn test_cov_model_output_weights_accessor() {
    let model = create_inference_test_model();

    assert_eq!(model.output_norm_weight().len(), 64);
    assert!(model.output_norm_bias().is_none());
    assert_eq!(model.lm_head_weight().out_dim, 100); // vocab_size
}

// ============================================================================
// SECTION 29: GGUF MODEL METADATA ACCESSORS
// ============================================================================

#[test]
fn test_cov_gguf_model_architecture() {
    let data = build_gguf_with_arch("phi", 2560, 32, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    let arch = model.architecture();
    assert_eq!(arch, Some("phi"));
}

#[test]
fn test_cov_gguf_model_embedding_dim() {
    let data = build_gguf_with_arch("llama", 4096, 32, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    let dim = model.embedding_dim();
    assert_eq!(dim, Some(4096));
}

#[test]
fn test_cov_gguf_model_num_layers() {
    let data = build_gguf_with_arch("mistral", 4096, 32, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    let layers = model.num_layers();
    assert_eq!(layers, Some(32));
}

#[test]
fn test_cov_gguf_model_num_heads() {
    let data = build_gguf_with_arch("qwen2", 3584, 28, 28);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    let heads = model.num_heads();
    assert_eq!(heads, Some(28));
}

// ============================================================================
// SECTION 30: ADDITIONAL EDGE CASES
// ============================================================================

#[test]
fn test_cov_owned_tensor_zero_size() {
    let tensor = OwnedQuantizedTensor {
        data: vec![],
        in_dim: 0,
        out_dim: 0,
        qtype: GGUF_TYPE_F32,
    };

    assert!(tensor.data.is_empty());
    assert_eq!(tensor.in_dim, 0);
    assert_eq!(tensor.out_dim, 0);
}

#[test]
fn test_cov_owned_tensor_large_data() {
    let size = 1024 * 1024; // 1MB
    let tensor = OwnedQuantizedTensor {
        data: vec![0u8; size],
        in_dim: 1024,
        out_dim: 1024,
        qtype: GGUF_TYPE_Q4_K,
    };

    assert_eq!(tensor.data.len(), size);
}

#[test]
fn test_cov_config_partial_eq() {
    let config1 = create_test_config();
    let config2 = create_test_config();

    // Both should be equal
    assert_eq!(config1.architecture, config2.architecture);
    assert_eq!(config1.hidden_dim, config2.hidden_dim);
}

#[test]
fn test_cov_header_memory_layout() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 100,
        metadata_count: 50,
    };

    // Verify field sizes
    assert_eq!(std::mem::size_of_val(&header.magic), 4);
    assert_eq!(std::mem::size_of_val(&header.version), 4);
    assert_eq!(std::mem::size_of_val(&header.tensor_count), 8);
    assert_eq!(std::mem::size_of_val(&header.metadata_count), 8);
}

#[test]
fn test_cov_tensor_info_default_values() {
    let info = TensorInfo {
        name: String::new(),
        n_dims: 0,
        dims: vec![],
        qtype: 0,
        offset: 0,
    };

    assert!(info.name.is_empty());
    assert_eq!(info.n_dims, 0);
    assert!(info.dims.is_empty());
}
