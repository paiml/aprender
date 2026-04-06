//! EXTREME TDD: GGUFValue Coverage Tests
//!
//! Targets low-coverage areas in gguf.rs GGUFValue enum parsing.

use realizar::gguf::{GGUFModel, GGUFValue, GGUF_MAGIC};

/// Helper to build minimal GGUF data with one metadata item
fn build_gguf_with_metadata(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Key
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    // Value type
    data.extend_from_slice(&value_type.to_le_bytes());
    // Value
    data.extend_from_slice(value_bytes);

    data
}

// ===== GGUFValue UINT8 (type 0) =====

#[test]
fn test_cov_gguf_value_uint8() {
    let data = build_gguf_with_metadata("test.uint8", 0, &[42u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.uint8"),
        Some(&GGUFValue::UInt8(42))
    );
}

#[test]
fn test_cov_gguf_value_uint8_zero() {
    let data = build_gguf_with_metadata("test.zero", 0, &[0u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("test.zero"), Some(&GGUFValue::UInt8(0)));
}

#[test]
fn test_cov_gguf_value_uint8_max() {
    let data = build_gguf_with_metadata("test.max", 0, &[255u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("test.max"), Some(&GGUFValue::UInt8(255)));
}

// ===== GGUFValue INT8 (type 1) =====

#[test]
fn test_cov_gguf_value_int8_positive() {
    let data = build_gguf_with_metadata("test.int8", 1, &[100i8 as u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("test.int8"), Some(&GGUFValue::Int8(100)));
}

#[test]
fn test_cov_gguf_value_int8_negative() {
    let data = build_gguf_with_metadata("test.neg", 1, &(-50i8).to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("test.neg"), Some(&GGUFValue::Int8(-50)));
}

#[test]
fn test_cov_gguf_value_int8_min() {
    let data = build_gguf_with_metadata("test.min", 1, &(-128i8).to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("test.min"), Some(&GGUFValue::Int8(-128)));
}

#[test]
fn test_cov_gguf_value_int8_max() {
    let data = build_gguf_with_metadata("test.max", 1, &127i8.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.get("test.max"), Some(&GGUFValue::Int8(127)));
}

// ===== GGUFValue UINT16 (type 2) =====

#[test]
fn test_cov_gguf_value_uint16() {
    let data = build_gguf_with_metadata("test.u16", 2, &1000u16.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.u16"),
        Some(&GGUFValue::UInt16(1000))
    );
}

#[test]
fn test_cov_gguf_value_uint16_max() {
    let data = build_gguf_with_metadata("test.max", 2, &65535u16.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.max"),
        Some(&GGUFValue::UInt16(65535))
    );
}

// ===== GGUFValue INT16 (type 3) =====

#[test]
fn test_cov_gguf_value_int16_positive() {
    let data = build_gguf_with_metadata("test.i16", 3, &5000i16.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.i16"),
        Some(&GGUFValue::Int16(5000))
    );
}

#[test]
fn test_cov_gguf_value_int16_negative() {
    let data = build_gguf_with_metadata("test.neg", 3, &(-5000i16).to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.neg"),
        Some(&GGUFValue::Int16(-5000))
    );
}

#[test]
fn test_cov_gguf_value_int16_min() {
    let data = build_gguf_with_metadata("test.min", 3, &i16::MIN.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.min"),
        Some(&GGUFValue::Int16(i16::MIN))
    );
}

// ===== GGUFValue UINT32 (type 4) =====

#[test]
fn test_cov_gguf_value_uint32() {
    let data = build_gguf_with_metadata("test.u32", 4, &123456u32.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.u32"),
        Some(&GGUFValue::UInt32(123456))
    );
}

#[test]
fn test_cov_gguf_value_uint32_max() {
    let data = build_gguf_with_metadata("test.max", 4, &u32::MAX.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.max"),
        Some(&GGUFValue::UInt32(u32::MAX))
    );
}

// ===== GGUFValue INT32 (type 5) =====

#[test]
fn test_cov_gguf_value_int32_positive() {
    let data = build_gguf_with_metadata("test.i32", 5, &1000000i32.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.i32"),
        Some(&GGUFValue::Int32(1000000))
    );
}

#[test]
fn test_cov_gguf_value_int32_negative() {
    let data = build_gguf_with_metadata("test.neg", 5, &(-1000000i32).to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.neg"),
        Some(&GGUFValue::Int32(-1000000))
    );
}

// ===== GGUFValue FLOAT32 (type 6) =====

#[test]
fn test_cov_gguf_value_float32() {
    let data = build_gguf_with_metadata("test.f32", 6, &std::f32::consts::PI.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test.f32") {
        assert!((v - std::f32::consts::PI).abs() < 1e-5);
    } else {
        panic!("Expected Float32");
    }
}

#[test]
fn test_cov_gguf_value_float32_zero() {
    let data = build_gguf_with_metadata("test.zero", 6, &0.0f32.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.zero"),
        Some(&GGUFValue::Float32(0.0))
    );
}

#[test]
fn test_cov_gguf_value_float32_negative() {
    let data = build_gguf_with_metadata("test.neg", 6, &(-1.5f32).to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.neg"),
        Some(&GGUFValue::Float32(-1.5))
    );
}

// ===== GGUFValue BOOL (type 7) =====

#[test]
fn test_cov_gguf_value_bool_true() {
    let data = build_gguf_with_metadata("test.bool", 7, &[1u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.bool"),
        Some(&GGUFValue::Bool(true))
    );
}

#[test]
fn test_cov_gguf_value_bool_false() {
    let data = build_gguf_with_metadata("test.bool", 7, &[0u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.bool"),
        Some(&GGUFValue::Bool(false))
    );
}

#[test]
fn test_cov_gguf_value_bool_nonzero_is_true() {
    // Any non-zero byte should be treated as true
    let data = build_gguf_with_metadata("test.bool", 7, &[42u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.bool"),
        Some(&GGUFValue::Bool(true))
    );
}

// ===== GGUFValue STRING (type 8) =====

#[test]
fn test_cov_gguf_value_string() {
    let s = "hello world";
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    value_bytes.extend_from_slice(s.as_bytes());

    let data = build_gguf_with_metadata("test.str", 8, &value_bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.str"),
        Some(&GGUFValue::String("hello world".to_string()))
    );
}

#[test]
fn test_cov_gguf_value_string_empty() {
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&0u64.to_le_bytes()); // empty string

    let data = build_gguf_with_metadata("test.empty", 8, &value_bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.empty"),
        Some(&GGUFValue::String(String::new()))
    );
}

#[test]
fn test_cov_gguf_value_string_unicode() {
    let s = "hello \u{1F600} world"; // with emoji
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    value_bytes.extend_from_slice(s.as_bytes());

    let data = build_gguf_with_metadata("test.unicode", 8, &value_bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.unicode"),
        Some(&GGUFValue::String(s.to_string()))
    );
}

// ===== GGUFValue ARRAY (type 9) =====

#[test]
fn test_cov_gguf_value_array_uint32() {
    // Array: element_type=4 (uint32), array_len=3, elements=[1, 2, 3]
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&4u32.to_le_bytes()); // element_type = UInt32
    value_bytes.extend_from_slice(&3u64.to_le_bytes()); // array_len = 3
    value_bytes.extend_from_slice(&1u32.to_le_bytes());
    value_bytes.extend_from_slice(&2u32.to_le_bytes());
    value_bytes.extend_from_slice(&3u32.to_le_bytes());

    let data = build_gguf_with_metadata("test.array", 9, &value_bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    let expected = GGUFValue::Array(vec![
        GGUFValue::UInt32(1),
        GGUFValue::UInt32(2),
        GGUFValue::UInt32(3),
    ]);
    assert_eq!(model.metadata.get("test.array"), Some(&expected));
}

#[test]
fn test_cov_gguf_value_array_strings() {
    // Array of strings
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&8u32.to_le_bytes()); // element_type = String
    value_bytes.extend_from_slice(&2u64.to_le_bytes()); // array_len = 2

    let s1 = "foo";
    value_bytes.extend_from_slice(&(s1.len() as u64).to_le_bytes());
    value_bytes.extend_from_slice(s1.as_bytes());

    let s2 = "bar";
    value_bytes.extend_from_slice(&(s2.len() as u64).to_le_bytes());
    value_bytes.extend_from_slice(s2.as_bytes());

    let data = build_gguf_with_metadata("test.strs", 9, &value_bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    let expected = GGUFValue::Array(vec![
        GGUFValue::String("foo".to_string()),
        GGUFValue::String("bar".to_string()),
    ]);
    assert_eq!(model.metadata.get("test.strs"), Some(&expected));
}

#[test]
fn test_cov_gguf_value_array_empty() {
    // Empty array
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&4u32.to_le_bytes()); // element_type = UInt32
    value_bytes.extend_from_slice(&0u64.to_le_bytes()); // array_len = 0

    let data = build_gguf_with_metadata("test.empty", 9, &value_bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(
        model.metadata.get("test.empty"),
        Some(&GGUFValue::Array(vec![]))
    );
}

// ===== GGUFValue UINT64 (type 10) =====

#[test]
fn test_cov_gguf_value_uint64() {
    let data = build_gguf_with_metadata("test.u64", 10, &9999999999u64.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.u64"),
        Some(&GGUFValue::UInt64(9999999999))
    );
}

#[test]
fn test_cov_gguf_value_uint64_max() {
    let data = build_gguf_with_metadata("test.max", 10, &u64::MAX.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.max"),
        Some(&GGUFValue::UInt64(u64::MAX))
    );
}

// ===== GGUFValue INT64 (type 11) =====

#[test]
fn test_cov_gguf_value_int64_positive() {
    let data = build_gguf_with_metadata("test.i64", 11, &9999999999i64.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.i64"),
        Some(&GGUFValue::Int64(9999999999))
    );
}

#[test]
fn test_cov_gguf_value_int64_negative() {
    let data = build_gguf_with_metadata("test.neg", 11, &(-9999999999i64).to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(
        model.metadata.get("test.neg"),
        Some(&GGUFValue::Int64(-9999999999))
    );
}

// ===== GGUFValue FLOAT64 (type 12) =====

#[test]
fn test_cov_gguf_value_float64() {
    let data = build_gguf_with_metadata("test.f64", 12, &std::f64::consts::PI.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Float64(v)) = model.metadata.get("test.f64") {
        assert!((v - std::f64::consts::PI).abs() < 1e-10);
    } else {
        panic!("Expected Float64");
    }
}

#[test]
fn test_cov_gguf_value_float64_negative() {
    let neg_e = -std::f64::consts::E;
    let data = build_gguf_with_metadata("test.neg", 12, &neg_e.to_le_bytes());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Float64(v)) = model.metadata.get("test.neg") {
        assert!((v - neg_e).abs() < 1e-10);
    } else {
        panic!("Expected Float64");
    }
}

// ===== Error paths =====

#[test]
fn test_cov_gguf_unsupported_value_type() {
    // Type 99 is not supported
    let data = build_gguf_with_metadata("test.bad", 99, &[0u8; 8]);
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_gguf_truncated_metadata_value() {
    // Build GGUF with metadata that claims u32 but doesn't have enough bytes
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count

    let key = "truncated";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // type = UInt32
    data.extend_from_slice(&[1u8, 2u8]); // Only 2 bytes, need 4

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_gguf_multiple_metadata_entries() {
    // Test parsing multiple metadata entries
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&3u64.to_le_bytes()); // metadata_count = 3

    // Entry 1: uint8
    let key1 = "key1";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // UInt8
    data.push(10);

    // Entry 2: bool
    let key2 = "key2";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&7u32.to_le_bytes()); // Bool
    data.push(1);

    // Entry 3: float32
    let key3 = "key3";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3.as_bytes());
    data.extend_from_slice(&6u32.to_le_bytes()); // Float32
    data.extend_from_slice(&1.5f32.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.metadata.len(), 3);
    assert_eq!(model.metadata.get("key1"), Some(&GGUFValue::UInt8(10)));
    assert_eq!(model.metadata.get("key2"), Some(&GGUFValue::Bool(true)));
    assert_eq!(model.metadata.get("key3"), Some(&GGUFValue::Float32(1.5)));
}

// ===== Tensor info parsing =====

#[test]
fn test_cov_gguf_1d_tensor() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // 1D tensor
    let name = "bias";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
    data.extend_from_slice(&512u64.to_le_bytes()); // dim[0] = 512
    data.extend_from_slice(&0u32.to_le_bytes()); // qtype = 0 (F32)
    data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].name, "bias");
    assert_eq!(model.tensors[0].n_dims, 1);
    assert_eq!(model.tensors[0].dims, vec![512]);
}

#[test]
fn test_cov_gguf_4d_tensor() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // 4D tensor (conv weights)
    let name = "conv.weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // n_dims = 4
    data.extend_from_slice(&64u64.to_le_bytes()); // dim[0]
    data.extend_from_slice(&64u64.to_le_bytes()); // dim[1]
    data.extend_from_slice(&3u64.to_le_bytes()); // dim[2]
    data.extend_from_slice(&3u64.to_le_bytes()); // dim[3]
    data.extend_from_slice(&2u32.to_le_bytes()); // qtype = 2 (Q4_0)
    data.extend_from_slice(&1024u64.to_le_bytes()); // offset

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].name, "conv.weight");
    assert_eq!(model.tensors[0].n_dims, 4);
    // Dims are reversed from GGML order
    assert_eq!(model.tensors[0].dims, vec![3, 3, 64, 64]);
}

#[test]
fn test_cov_gguf_multiple_tensors() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes()); // tensor_count = 2
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // Tensor 1
    let name1 = "w1";
    data.extend_from_slice(&(name1.len() as u64).to_le_bytes());
    data.extend_from_slice(name1.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&128u64.to_le_bytes());
    data.extend_from_slice(&256u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // qtype
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Tensor 2
    let name2 = "w2";
    data.extend_from_slice(&(name2.len() as u64).to_le_bytes());
    data.extend_from_slice(name2.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&256u64.to_le_bytes());
    data.extend_from_slice(&512u64.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // qtype = Q8_0
    data.extend_from_slice(&131072u64.to_le_bytes()); // offset

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors.len(), 2);
    assert_eq!(model.tensors[0].name, "w1");
    assert_eq!(model.tensors[1].name, "w2");
    assert_eq!(model.tensors[1].qtype, 8);
}

// ===== Magic and constants =====

#[test]
fn test_cov_gguf_magic_bytes() {
    let bytes = GGUF_MAGIC.to_le_bytes();
    assert_eq!(&bytes, b"GGUF");
}

// ===== Alignment =====

#[test]
fn test_cov_gguf_tensor_data_alignment() {
    // Verify tensor_data_start is 32-byte aligned
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    // Header is 24 bytes, should align to 32

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensor_data_start % 32, 0);
}
