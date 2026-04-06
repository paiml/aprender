//! Phase 35: GGUF Module Coverage Enhancement
//!
//! This module adds comprehensive test coverage for edge cases and error paths
//! in the GGUF parsing code:
//!
//! - Error handling in loader.rs (truncated data, corrupted structures)
//! - MappedGGUFModel edge cases (memory mapping, tensor slicing)
//! - GGUFValue edge cases (all variants, cloning, equality)
//! - RoPE type inference for all supported architectures
//! - Tokenizer edge cases (GPT-2 style, SentencePiece, byte tokens)
//! - Array and nested metadata parsing
//! - Tensor dimension handling and overflow
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::{GGUFHeader, GGUFModel, GGUFValue, TensorInfo, GGUF_MAGIC, GGUF_VERSION_V3};

// =============================================================================
// Test Data Builders (shared utilities)
// =============================================================================

/// Build valid GGUF header bytes
fn build_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
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
// GGUFValue Tests - Edge Cases and All Variants
// =============================================================================

#[test]
fn test_phase35_gguf_value_uint8_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 0, &[0u8]));
    data.extend(build_gguf_metadata("test_max", 0, &[255u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_min"),
        Some(GGUFValue::UInt8(0))
    ));
    assert!(matches!(
        model.metadata.get("test_max"),
        Some(GGUFValue::UInt8(255))
    ));
}

#[test]
fn test_phase35_gguf_value_int8_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 1, &i8::MIN.to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 1, &i8::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_min"),
        Some(GGUFValue::Int8(-128))
    ));
    assert!(matches!(
        model.metadata.get("test_max"),
        Some(GGUFValue::Int8(127))
    ));
}

#[test]
fn test_phase35_gguf_value_uint16_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 2, &0u16.to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 2, &u16::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_min"),
        Some(GGUFValue::UInt16(0))
    ));
    assert!(matches!(
        model.metadata.get("test_max"),
        Some(GGUFValue::UInt16(65535))
    ));
}

#[test]
fn test_phase35_gguf_value_int16_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 3, &i16::MIN.to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 3, &i16::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_min"),
        Some(GGUFValue::Int16(-32768))
    ));
    assert!(matches!(
        model.metadata.get("test_max"),
        Some(GGUFValue::Int16(32767))
    ));
}

#[test]
fn test_phase35_gguf_value_uint64_large() {
    let mut data = build_gguf_header(0, 1);
    let large_value = u64::MAX;
    data.extend(build_gguf_metadata(
        "test_large",
        10,
        &large_value.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(
        matches!(model.metadata.get("test_large"), Some(GGUFValue::UInt64(v)) if *v == u64::MAX)
    );
}

#[test]
fn test_phase35_gguf_value_int64_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 11, &i64::MIN.to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 11, &i64::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(model.metadata.get("test_min"), Some(GGUFValue::Int64(v)) if *v == i64::MIN));
    assert!(matches!(model.metadata.get("test_max"), Some(GGUFValue::Int64(v)) if *v == i64::MAX));
}

#[test]
fn test_phase35_gguf_value_float32_special() {
    let mut data = build_gguf_header(0, 3);
    data.extend(build_gguf_metadata("test_zero", 6, &0.0f32.to_le_bytes()));
    data.extend(build_gguf_metadata("test_neg", 6, &(-1.5f32).to_le_bytes()));
    data.extend(build_gguf_metadata(
        "test_inf",
        6,
        &f32::INFINITY.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test_zero") {
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test_neg") {
        assert!((v - (-1.5)).abs() < f32::EPSILON);
    }
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test_inf") {
        assert!(v.is_infinite());
    }
}

#[test]
fn test_phase35_gguf_value_float64_special() {
    let mut data = build_gguf_header(0, 3);
    data.extend(build_gguf_metadata(
        "test_pi",
        12,
        &std::f64::consts::PI.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "test_e",
        12,
        &std::f64::consts::E.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "test_neg_inf",
        12,
        &f64::NEG_INFINITY.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Float64(v)) = model.metadata.get("test_pi") {
        assert!((v - std::f64::consts::PI).abs() < 1e-10);
    }
    if let Some(GGUFValue::Float64(v)) = model.metadata.get("test_neg_inf") {
        assert!(v.is_infinite() && *v < 0.0);
    }
}

#[test]
fn test_phase35_gguf_value_clone_equality() {
    let value1 = GGUFValue::String("test".to_string());
    let value2 = value1.clone();
    assert_eq!(value1, value2);

    let value3 = GGUFValue::Array(vec![GGUFValue::UInt32(1), GGUFValue::UInt32(2)]);
    let value4 = value3.clone();
    assert_eq!(value3, value4);
}

#[test]
fn test_phase35_gguf_value_debug_format() {
    let value = GGUFValue::String("debug test".to_string());
    let debug_str = format!("{:?}", value);
    assert!(debug_str.contains("String"));
    assert!(debug_str.contains("debug test"));
}

// =============================================================================
// GGUFHeader and TensorInfo Tests
// =============================================================================

#[test]
fn test_phase35_gguf_header_struct() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 100,
        metadata_count: 50,
    };

    assert_eq!(header.magic, 0x4655_4747);
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 100);
    assert_eq!(header.metadata_count, 50);

    // Test Clone
    let cloned = header.clone();
    assert_eq!(header, cloned);

    // Test Debug
    let debug_str = format!("{:?}", header);
    assert!(debug_str.contains("GGUFHeader"));
}

#[test]
fn test_phase35_tensor_info_struct() {
    let info = TensorInfo {
        name: "blk.0.attn_q.weight".to_string(),
        n_dims: 2,
        dims: vec![4096, 4096],
        qtype: 12, // Q4_K
        offset: 0,
    };

    assert_eq!(info.name, "blk.0.attn_q.weight");
    assert_eq!(info.n_dims, 2);
    assert_eq!(info.dims.len(), 2);

    // Test Clone
    let cloned = info.clone();
    assert_eq!(info, cloned);

    // Test Debug
    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("TensorInfo"));
    assert!(debug_str.contains("attn_q"));
}

// =============================================================================
// Error Handling - Truncated Data
// =============================================================================

#[test]
fn test_phase35_truncated_metadata_key() {
    let mut data = build_gguf_header(0, 1);
    // Add truncated key string (length says 100 but only 5 bytes)
    data.extend_from_slice(&100u64.to_le_bytes());
    data.extend_from_slice(b"short");

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated metadata key should fail");
}

#[test]
fn test_phase35_truncated_metadata_value() {
    let mut data = build_gguf_header(0, 1);
    // Add valid key
    data.extend(build_gguf_string("test_key"));
    // Add value type
    data.extend_from_slice(&4u32.to_le_bytes()); // u32
                                                 // Missing value bytes

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated metadata value should fail");
}

#[test]
fn test_phase35_truncated_tensor_name() {
    let mut data = build_gguf_header(1, 0);
    // Add truncated tensor name
    data.extend_from_slice(&50u64.to_le_bytes()); // says 50 bytes
    data.extend_from_slice(b"short"); // only 5 bytes

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated tensor name should fail");
}

#[test]
fn test_phase35_truncated_tensor_dims() {
    let mut data = build_gguf_header(1, 0);
    // Valid tensor name
    data.extend(build_gguf_string("tensor"));
    // Say 5 dimensions
    data.extend_from_slice(&5u32.to_le_bytes());
    // Only provide 2 dimensions
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated tensor dimensions should fail");
}

#[test]
fn test_phase35_truncated_array_elements() {
    let mut data = build_gguf_header(0, 1);
    // Array metadata: element_type + array_len + elements
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
    array_bytes.extend_from_slice(&10u64.to_le_bytes()); // array length: 10
                                                         // Only provide 2 elements
    array_bytes.extend_from_slice(&1u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u32.to_le_bytes());

    data.extend(build_gguf_metadata("test_array", 9, &array_bytes));

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated array should fail");
}

// =============================================================================
// RoPE Type Inference - All Architectures
// =============================================================================

#[test]
fn test_phase35_rope_type_neox_architectures() {
    // Test all NEOX-style architectures
    let neox_archs = [
        "qwen",
        "qwen2",
        "qwen3",
        "stablelm",
        "phi2",
        "phi3",
        "gemma",
        "gemma2",
        "gemma3",
        "starcoder2",
        "gptneox",
        "falcon",
        "codeshell",
        "orion",
        "bert",
        "nomic-bert",
        "dbrx",
        "olmo2",
        "olmoe",
        "plamo",
        "plamo2",
        "openelm",
        "exaone",
        "minicpm3",
        "nemotron",
        "internlm2",
        "deepseek2",
    ];

    for arch in neox_archs {
        let mut data = build_gguf_header(0, 1);
        let arch_value = build_gguf_string(arch);
        data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

        let model =
            GGUFModel::from_bytes(&data).unwrap_or_else(|_| panic!("Should parse {}", arch));
        assert_eq!(
            model.rope_type(),
            Some(2),
            "Architecture {} should use NEOX RoPE",
            arch
        );
    }
}

#[test]
fn test_phase35_rope_type_norm_architectures() {
    // Test NORM-style architectures (LLaMA family)
    let norm_archs = ["llama", "mistral", "tinyllama", "codellama", "unknown_arch"];

    for arch in norm_archs {
        let mut data = build_gguf_header(0, 1);
        let arch_value = build_gguf_string(arch);
        data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

        let model =
            GGUFModel::from_bytes(&data).unwrap_or_else(|_| panic!("Should parse {}", arch));
        assert_eq!(
            model.rope_type(),
            Some(0),
            "Architecture {} should use NORM RoPE",
            arch
        );
    }
}

#[test]
fn test_phase35_rope_type_scaling_none() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let none_value = build_gguf_string("none");
    data.extend(build_gguf_metadata(
        "custom.rope.scaling.type",
        8,
        &none_value,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(0)); // NORM from "none" scaling
}

include!("phase35_rope.rs");
include!("phase35_array.rs");
