//! GGUF parsing tests extracted from inline tests
//!
//! Tests for GGUF header, metadata, tensor info, and data parsing.

use realizar::error::RealizarError;
use realizar::gguf::{
    GGUFModel, GGUFValue, GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q8_0,
};

#[test]
fn test_gguf_magic_constant() {
    // "GGUF" in little-endian
    assert_eq!(GGUF_MAGIC, 0x4655_4747);
    // Verify it spells "GGUF"
    let bytes = GGUF_MAGIC.to_le_bytes();
    assert_eq!(&bytes, b"GGUF");
}

#[test]
fn test_parse_valid_header() {
    // Minimal valid GGUF v3 header
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF"); // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.header.magic, GGUF_MAGIC);
    assert_eq!(model.header.version, 3);
    assert_eq!(model.header.tensor_count, 0);
    assert_eq!(model.header.metadata_count, 0);
}

#[test]
fn test_invalid_magic() {
    let mut data = Vec::new();
    data.extend_from_slice(b"BAAD"); // Invalid magic
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RealizarError::InvalidShape { .. }
    ));
}

#[test]
fn test_unsupported_version() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RealizarError::UnsupportedOperation { .. }
    ));
}

#[test]
fn test_truncated_data() {
    // Only 4 bytes (magic only)
    let data = b"GGUF";
    let result = GGUFModel::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_empty_file() {
    let data = &[];
    let result = GGUFModel::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_parse_uint32_metadata() {
    // GGUF header with 1 metadata item (UInt32)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF"); // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Metadata: key = "test.value", value_type = UInt32 (4), value = 42
    let key = "test.value";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes()); // key length
    data.extend_from_slice(key.as_bytes()); // key string
    data.extend_from_slice(&4u32.to_le_bytes()); // value_type = UInt32
    data.extend_from_slice(&42u32.to_le_bytes()); // value = 42

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.metadata.len(), 1);
    assert_eq!(
        model.metadata.get("test.value"),
        Some(&GGUFValue::UInt32(42))
    );
}

#[test]
fn test_parse_string_metadata() {
    // GGUF header with 1 metadata item (String)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Metadata: key = "model.name", value_type = String (8), value = "TestModel"
    let key = "model.name";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // value_type = String
    let value = "TestModel";
    data.extend_from_slice(&(value.len() as u64).to_le_bytes()); // string length
    data.extend_from_slice(value.as_bytes()); // string data

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.metadata.len(), 1);
    assert_eq!(
        model.metadata.get("model.name"),
        Some(&GGUFValue::String("TestModel".to_string()))
    );
}

#[test]
fn test_parse_multiple_metadata() {
    // GGUF header with 2 metadata items
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes()); // metadata_count = 2

    // First: key = "version", value = UInt32(1)
    data.extend_from_slice(&7u64.to_le_bytes());
    data.extend_from_slice(b"version");
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // Second: key = "arch", value = String("llama")
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(b"arch");
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"llama");

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.metadata.len(), 2);
    assert_eq!(model.metadata.get("version"), Some(&GGUFValue::UInt32(1)));
    assert_eq!(
        model.metadata.get("arch"),
        Some(&GGUFValue::String("llama".to_string()))
    );
}

#[test]
fn test_parse_single_tensor_info() {
    // GGUF header with 1 tensor
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // Tensor: name = "weight", n_dims = 2, dims = [128, 256], qtype = 0, offset = 1024
    let name = "weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    data.extend_from_slice(&128u64.to_le_bytes()); // dim[0] = 128
    data.extend_from_slice(&256u64.to_le_bytes()); // dim[1] = 256
    data.extend_from_slice(&0u32.to_le_bytes()); // qtype = 0
    data.extend_from_slice(&1024u64.to_le_bytes()); // offset = 1024

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 1);
    let tensor = &model.tensors[0];
    assert_eq!(tensor.name, "weight");
    assert_eq!(tensor.n_dims, 2);
    // GGUF stores dims in row-major order, parser returns them reversed
    assert_eq!(tensor.dims, vec![256, 128]);
    assert_eq!(tensor.qtype, 0);
    assert_eq!(tensor.offset, 1024);
}

#[test]
fn test_parse_tensor_3d() {
    // GGUF header with 1 tensor (3D)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // Tensor: name = "conv.weight", n_dims = 3, dims = [64, 64, 3]
    let name = "conv.weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // n_dims = 3
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // qtype = 2 (quantized)
    data.extend_from_slice(&2048u64.to_le_bytes()); // offset = 2048

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 1);
    let tensor = &model.tensors[0];
    assert_eq!(tensor.name, "conv.weight");
    assert_eq!(tensor.n_dims, 3);
    // GGUF stores dims in row-major order, parser returns them reversed
    assert_eq!(tensor.dims, vec![3, 64, 64]);
    assert_eq!(tensor.qtype, 2);
    assert_eq!(tensor.offset, 2048);
}

#[test]
fn test_parse_metadata_and_tensors() {
    // GGUF with both metadata and tensors
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Metadata: model.type = String("llama")
    data.extend_from_slice(&10u64.to_le_bytes());
    data.extend_from_slice(b"model.type");
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"llama");

    // Tensor: embedding
    data.extend_from_slice(&9u64.to_le_bytes());
    data.extend_from_slice(b"embedding");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&32000u64.to_le_bytes());
    data.extend_from_slice(&4096u64.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.metadata.len(), 1);
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(
        model.metadata.get("model.type"),
        Some(&GGUFValue::String("llama".to_string()))
    );
    assert_eq!(model.tensors[0].name, "embedding");
}

#[test]
fn test_parse_uint8_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Metadata: key = "byte_val", value_type = UInt8 (0), value = 255
    let key = "byte_val";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // value_type = UInt8
    data.push(255u8); // value = 255

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.metadata.get("byte_val"), Some(&GGUFValue::UInt8(255)));
}

#[test]
fn test_parse_int8_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "signed_byte";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // value_type = Int8
    data.extend_from_slice(&(-42i8).to_le_bytes()); // value = -42

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(
        model.metadata.get("signed_byte"),
        Some(&GGUFValue::Int8(-42))
    );
}

#[test]
fn test_parse_uint16_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "short_val";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // value_type = UInt16
    data.extend_from_slice(&65535u16.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(
        model.metadata.get("short_val"),
        Some(&GGUFValue::UInt16(65535))
    );
}

#[test]
fn test_parse_int16_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "signed_short";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // value_type = Int16
    data.extend_from_slice(&(-1000i16).to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(
        model.metadata.get("signed_short"),
        Some(&GGUFValue::Int16(-1000))
    );
}

#[test]
fn test_parse_int32_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "signed_int";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&5u32.to_le_bytes()); // value_type = Int32
    data.extend_from_slice(&(-100_000_i32).to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(
        model.metadata.get("signed_int"),
        Some(&GGUFValue::Int32(-100_000))
    );
}

#[test]
fn test_parse_float32_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "float_val";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&6u32.to_le_bytes()); // value_type = Float32
    data.extend_from_slice(&1.25f32.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    if let Some(GGUFValue::Float32(val)) = model.metadata.get("float_val") {
        assert!((val - 1.25).abs() < 1e-5);
    } else {
        panic!("Expected Float32 value");
    }
}

#[test]
fn test_parse_bool_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "is_enabled";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&7u32.to_le_bytes()); // value_type = Bool
    data.push(1u8); // true

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(
        model.metadata.get("is_enabled"),
        Some(&GGUFValue::Bool(true))
    );
}

#[test]
fn test_parse_bool_false_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "is_disabled";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&7u32.to_le_bytes()); // value_type = Bool
    data.push(0u8); // false

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(
        model.metadata.get("is_disabled"),
        Some(&GGUFValue::Bool(false))
    );
}

#[test]
fn test_parse_uint64_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "big_uint";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&10u32.to_le_bytes()); // value_type = UInt64
    data.extend_from_slice(&(u64::MAX).to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(
        model.metadata.get("big_uint"),
        Some(&GGUFValue::UInt64(u64::MAX))
    );
}

#[test]
fn test_parse_int64_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "big_int";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&11u32.to_le_bytes()); // value_type = Int64
    data.extend_from_slice(&(i64::MIN).to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(
        model.metadata.get("big_int"),
        Some(&GGUFValue::Int64(i64::MIN))
    );
}

#[test]
fn test_parse_float64_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "double_val";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&12u32.to_le_bytes()); // value_type = Float64
    data.extend_from_slice(&1.125f64.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    if let Some(GGUFValue::Float64(val)) = model.metadata.get("double_val") {
        assert!((val - 1.125).abs() < 1e-10);
    } else {
        panic!("Expected Float64 value");
    }
}

#[test]
fn test_parse_unsupported_value_type() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "unknown";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&99u32.to_le_bytes()); // Invalid value_type

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RealizarError::UnsupportedOperation { .. }
    ));
}

#[test]
fn test_parse_all_value_types() {
    // Test file with all supported value types
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&12u64.to_le_bytes()); // metadata_count = 12

    // UInt8
    data.extend_from_slice(&2u64.to_le_bytes());
    data.extend_from_slice(b"u8");
    data.extend_from_slice(&0u32.to_le_bytes());
    data.push(100u8);

    // Int8
    data.extend_from_slice(&2u64.to_le_bytes());
    data.extend_from_slice(b"i8");
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&(-50i8).to_le_bytes());

    // UInt16
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"u16");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1000u16.to_le_bytes());

    // Int16
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"i16");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&(-500i16).to_le_bytes());

    // UInt32
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"u32");
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&100_000_u32.to_le_bytes());

    // Int32
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"i32");
    data.extend_from_slice(&5u32.to_le_bytes());
    data.extend_from_slice(&(-50000i32).to_le_bytes());

    // Float32
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"f32");
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&1.5f32.to_le_bytes());

    // Bool
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(b"bool");
    data.extend_from_slice(&7u32.to_le_bytes());
    data.push(1u8);

    // String
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"str");
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(b"test");

    // UInt64
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"u64");
    data.extend_from_slice(&10u32.to_le_bytes());
    data.extend_from_slice(&1_000_000u64.to_le_bytes());

    // Int64
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"i64");
    data.extend_from_slice(&11u32.to_le_bytes());
    data.extend_from_slice(&(-500_000_i64).to_le_bytes());

    // Float64
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(b"f64");
    data.extend_from_slice(&12u32.to_le_bytes());
    data.extend_from_slice(&2.5f64.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.metadata.len(), 12);
    assert_eq!(model.metadata.get("u8"), Some(&GGUFValue::UInt8(100)));
    assert_eq!(model.metadata.get("i8"), Some(&GGUFValue::Int8(-50)));
    assert_eq!(model.metadata.get("u16"), Some(&GGUFValue::UInt16(1000)));
    assert_eq!(model.metadata.get("i16"), Some(&GGUFValue::Int16(-500)));
    assert_eq!(model.metadata.get("u32"), Some(&GGUFValue::UInt32(100_000)));
    assert_eq!(model.metadata.get("i32"), Some(&GGUFValue::Int32(-50000)));
    assert_eq!(model.metadata.get("bool"), Some(&GGUFValue::Bool(true)));
    assert_eq!(
        model.metadata.get("str"),
        Some(&GGUFValue::String("test".to_string()))
    );
    assert_eq!(
        model.metadata.get("u64"),
        Some(&GGUFValue::UInt64(1_000_000))
    );
    assert_eq!(model.metadata.get("i64"), Some(&GGUFValue::Int64(-500_000)));

    // Check floats with tolerance
    if let Some(GGUFValue::Float32(val)) = model.metadata.get("f32") {
        assert!((val - 1.5).abs() < 1e-5);
    } else {
        panic!("Expected f32");
    }
    if let Some(GGUFValue::Float64(val)) = model.metadata.get("f64") {
        assert!((val - 2.5).abs() < 1e-10);
    } else {
        panic!("Expected f64");
    }
}

#[test]
fn test_parse_array_uint32() {
    // GGUF header with 1 metadata item (Array of UInt32)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF"); // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Metadata: key = "test.array", value_type = Array (9)
    let key = "test.array";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes()); // key length
    data.extend_from_slice(key.as_bytes()); // key string
    data.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
    data.extend_from_slice(&4u32.to_le_bytes()); // element_type = UInt32
    data.extend_from_slice(&3u64.to_le_bytes()); // array_len = 3
    data.extend_from_slice(&1u32.to_le_bytes()); // element 0
    data.extend_from_slice(&2u32.to_le_bytes()); // element 1
    data.extend_from_slice(&3u32.to_le_bytes()); // element 2

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.metadata.len(), 1);
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test.array") {
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], GGUFValue::UInt32(1));
        assert_eq!(arr[1], GGUFValue::UInt32(2));
        assert_eq!(arr[2], GGUFValue::UInt32(3));
    } else {
        panic!("Expected Array value");
    }
}

#[test]
fn test_parse_array_string() {
    // GGUF header with 1 metadata item (Array of strings)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // Metadata: key = "test.strings", value_type = Array (9)
    let key = "test.strings";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
    data.extend_from_slice(&8u32.to_le_bytes()); // element_type = String
    data.extend_from_slice(&2u64.to_le_bytes()); // array_len = 2

    // String element 0: "hello"
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"hello");

    // String element 1: "world"
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"world");

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert_eq!(model.metadata.len(), 1);
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test.strings") {
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], GGUFValue::String("hello".to_string()));
        assert_eq!(arr[1], GGUFValue::String("world".to_string()));
    } else {
        panic!("Expected Array value");
    }
}

#[test]
fn test_parse_empty_array() {
    // GGUF header with empty array
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "empty";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
    data.extend_from_slice(&4u32.to_le_bytes()); // element_type = UInt32
    data.extend_from_slice(&0u64.to_le_bytes()); // array_len = 0

    let model = GGUFModel::from_bytes(&data).expect("test");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("empty") {
        assert_eq!(arr.len(), 0);
    } else {
        panic!("Expected empty Array");
    }
}

#[test]
fn test_get_tensor_f32_unquantized() {
    // Create a GGUF file with F32 tensor
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version = 3
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // Tensor: name = "weights", dims = [2, 3], qtype = F32 (0)
    let tensor_name = "weights";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    data.extend_from_slice(&2u64.to_le_bytes()); // dim[0] = 2
    data.extend_from_slice(&3u64.to_le_bytes()); // dim[1] = 3
    data.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes()); // qtype = F32

    // Tensor offset is 0 (relative to tensor data section start)
    data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

    // Pad to 32-byte alignment
    while data.len() % GGUF_ALIGNMENT != 0 {
        data.push(0);
    }

    // Add F32 tensor data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for val in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        data.extend_from_slice(&val.to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("test");
    let values = model.get_tensor_f32("weights", &data).expect("test");

    assert_eq!(values.len(), 6);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_get_tensor_f32_not_found() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&0u64.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f32("nonexistent", &data);

    assert!(result.is_err());
    if let Err(RealizarError::UnsupportedOperation { reason, .. }) = result {
        assert!(reason.contains("not found"));
    }
}

#[test]
fn test_get_tensor_f32_q4_0() {
    // Create a GGUF file with Q4_0 tensor
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&0u64.to_le_bytes());

    // Tensor: name = "quant_weights", dims = [64] (2 blocks), qtype = Q4_0 (2)
    let tensor_name = "quant_weights";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
    data.extend_from_slice(&64u64.to_le_bytes()); // dim[0] = 64 (2 blocks of 32)
    data.extend_from_slice(&GGUF_TYPE_Q4_0.to_le_bytes());

    // Tensor offset is 0 (relative to tensor data section start)
    data.extend_from_slice(&0u64.to_le_bytes());

    // Pad to 32-byte alignment
    while data.len() % GGUF_ALIGNMENT != 0 {
        data.push(0);
    }

    // Add Q4_0 data: 2 blocks (18 bytes each)
    // Q4_0 layout: 1×f16 scale (2 bytes) + 16 bytes (32×4-bit quants) = 18 bytes/block
    // Block 1: scale = 1.0 as f16, quants = 16 bytes
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&[0x10; 16]); // 4-bit values

    // Block 2: scale = 2.0 as f16, quants = 16 bytes
    data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    data.extend_from_slice(&[0x21; 16]);

    let model = GGUFModel::from_bytes(&data).expect("test");
    let values = model.get_tensor_f32("quant_weights", &data).expect("test");

    // Verify size is correct
    assert_eq!(values.len(), 64);

    // Values should be dequantized (non-zero)
    assert!(values.iter().any(|&v| v != 0.0));
}

#[test]
fn test_get_tensor_f32_q8_0() {
    // Create a GGUF file with Q8_0 tensor
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Tensor: dims = [32] (1 block), qtype = Q8_0 (8)
    let tensor_name = "q8_weights";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&32u64.to_le_bytes()); // dim[0] = 32 (1 block)
    data.extend_from_slice(&GGUF_TYPE_Q8_0.to_le_bytes());

    // Tensor offset is 0 (relative to tensor data section start)
    data.extend_from_slice(&0u64.to_le_bytes());

    // Pad to 32-byte alignment
    while data.len() % GGUF_ALIGNMENT != 0 {
        data.push(0);
    }

    // Add Q8_0 data: 1 block (36 bytes: 4 for scale + 32 for quants)
    data.extend_from_slice(&0.5f32.to_le_bytes());
    for i in 0i32..32 {
        // Test data uses i8 range [0, 31] - safe to convert
        data.push(u8::try_from(i).expect("test"));
    }

    let model = GGUFModel::from_bytes(&data).expect("test");
    let values = model.get_tensor_f32("q8_weights", &data).expect("test");

    assert_eq!(values.len(), 32);
    // First value should be approximately 0.5 * 0 = 0.0
    assert!((values[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_get_tensor_f32_unsupported_qtype() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Tensor with unsupported qtype
    let tensor_name = "bad_tensor";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&999u32.to_le_bytes()); // Invalid qtype

    // Calculate offset
    let tensor_offset = (data.len() + 8) as u64;
    data.extend_from_slice(&tensor_offset.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f32("bad_tensor", &data);

    assert!(result.is_err());
    if let Err(RealizarError::UnsupportedOperation { reason, .. }) = result {
        assert!(reason.contains("Unsupported quantization type"));
    }
}
