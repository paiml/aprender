//! Advanced GGUF Tests - Read tests, dequant tests, property-based tests
//! PMAT-085: Split from tests.rs for file health (lines 1090-2213)

use super::super::*;

// ========================================================================
// read_u32/read_u64/read_string tests (Coverage)
// ========================================================================

#[test]
fn test_read_u32_success() {
    let data = 0x12345678u32.to_le_bytes();
    let result = read_u32(&data, 0).expect("should read u32");
    assert_eq!(result, 0x12345678);
}

#[test]
fn test_read_u32_eof() {
    let data = [1, 2, 3]; // Only 3 bytes
    let result = read_u32(&data, 0);
    assert!(result.is_err());
    assert!(format!("{:?}", result.unwrap_err()).contains("EOF"));
}

#[test]
fn test_read_u32_offset() {
    let data = [0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12];
    let result = read_u32(&data, 4).expect("should read u32 at offset");
    assert_eq!(result, 0x12345678);
}

#[test]
fn test_read_u64_success() {
    let data = 0x123456789ABCDEF0u64.to_le_bytes();
    let result = read_u64(&data, 0).expect("should read u64");
    assert_eq!(result, 0x123456789ABCDEF0);
}

#[test]
fn test_read_u64_eof() {
    let data = [1, 2, 3, 4, 5, 6, 7]; // Only 7 bytes
    let result = read_u64(&data, 0);
    assert!(result.is_err());
    assert!(format!("{:?}", result.unwrap_err()).contains("EOF"));
}

#[test]
fn test_read_string_success() {
    // Length (8 bytes) + "hello" (5 bytes)
    let mut data = Vec::new();
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"hello");
    let (s, consumed) = read_string(&data, 0).expect("should read string");
    assert_eq!(s, "hello");
    assert_eq!(consumed, 13); // 8 + 5
}

#[test]
fn test_read_string_eof() {
    // Length says 100 but only 5 bytes
    let mut data = Vec::new();
    data.extend_from_slice(&100u64.to_le_bytes());
    data.extend_from_slice(b"hello");
    let result = read_string(&data, 0);
    assert!(result.is_err());
}

#[test]
fn test_read_string_empty() {
    let mut data = Vec::new();
    data.extend_from_slice(&0u64.to_le_bytes());
    let (s, consumed) = read_string(&data, 0).expect("should read empty string");
    assert_eq!(s, "");
    assert_eq!(consumed, 8);
}

// ========================================================================
// read_metadata_value tests (Coverage for all type branches)
// ========================================================================

#[test]
fn test_read_metadata_value_uint8() {
    let data = [42u8];
    let (value, consumed) = read_metadata_value(&data, 0, 0).expect("type 0 = uint8");
    assert!(matches!(value, GgufValue::Uint8(42)));
    assert_eq!(consumed, 1);
}

#[test]
fn test_read_metadata_value_uint8_eof() {
    let data: [u8; 0] = [];
    let result = read_metadata_value(&data, 0, 0);
    assert!(result.is_err());
}

#[test]
fn test_read_metadata_value_int8() {
    let data = [0xFE]; // -2 as i8
    let (value, consumed) = read_metadata_value(&data, 0, 1).expect("type 1 = int8");
    assert!(matches!(value, GgufValue::Int8(-2)));
    assert_eq!(consumed, 1);
}

#[test]
fn test_read_metadata_value_int8_eof() {
    let data: [u8; 0] = [];
    let result = read_metadata_value(&data, 0, 1);
    assert!(result.is_err());
}

#[test]
fn test_read_metadata_value_uint16() {
    let data = 1000u16.to_le_bytes();
    let (value, consumed) = read_metadata_value(&data, 0, 2).expect("type 2 = uint16");
    assert!(matches!(value, GgufValue::Uint16(1000)));
    assert_eq!(consumed, 2);
}

#[test]
fn test_read_metadata_value_uint16_eof() {
    let data = [1u8];
    let result = read_metadata_value(&data, 0, 2);
    assert!(result.is_err());
}

#[test]
fn test_read_metadata_value_int16() {
    let data = (-1000i16).to_le_bytes();
    let (value, consumed) = read_metadata_value(&data, 0, 3).expect("type 3 = int16");
    assert!(matches!(value, GgufValue::Int16(-1000)));
    assert_eq!(consumed, 2);
}

#[test]
fn test_read_metadata_value_int16_eof() {
    let data = [1u8];
    let result = read_metadata_value(&data, 0, 3);
    assert!(result.is_err());
}

#[test]
fn test_read_metadata_value_uint32() {
    let data = 100_000u32.to_le_bytes();
    let (value, consumed) = read_metadata_value(&data, 0, 4).expect("type 4 = uint32");
    assert!(matches!(value, GgufValue::Uint32(100_000)));
    assert_eq!(consumed, 4);
}

#[test]
fn test_read_metadata_value_int32() {
    let data = (-100_000i32).to_le_bytes();
    let (value, consumed) = read_metadata_value(&data, 0, 5).expect("type 5 = int32");
    assert!(matches!(value, GgufValue::Int32(-100_000)));
    assert_eq!(consumed, 4);
}

#[test]
fn test_read_metadata_value_int32_eof() {
    let data = [1, 2, 3];
    let result = read_metadata_value(&data, 0, 5);
    assert!(result.is_err());
}

#[test]
fn test_read_metadata_value_float32() {
    let data = 3.14f32.to_le_bytes();
    let (value, consumed) = read_metadata_value(&data, 0, 6).expect("type 6 = float32");
    if let GgufValue::Float32(v) = value {
        assert!((v - 3.14).abs() < 0.001);
    } else {
        panic!("expected Float32");
    }
    assert_eq!(consumed, 4);
}

#[test]
fn test_read_metadata_value_float32_eof() {
    let data = [1, 2, 3];
    let result = read_metadata_value(&data, 0, 6);
    assert!(result.is_err());
}

#[test]
fn test_read_metadata_value_bool_true() {
    let data = [1u8];
    let (value, consumed) = read_metadata_value(&data, 0, 7).expect("type 7 = bool");
    assert!(matches!(value, GgufValue::Bool(true)));
    assert_eq!(consumed, 1);
}

#[test]
fn test_read_metadata_value_bool_false() {
    let data = [0u8];
    let (value, consumed) = read_metadata_value(&data, 0, 7).expect("type 7 = bool");
    assert!(matches!(value, GgufValue::Bool(false)));
    assert_eq!(consumed, 1);
}

#[test]
fn test_read_metadata_value_bool_eof() {
    let data: [u8; 0] = [];
    let result = read_metadata_value(&data, 0, 7);
    assert!(result.is_err());
}

#[test]
fn test_read_metadata_value_string() {
    let mut data = Vec::new();
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"hello");
    let (value, consumed) = read_metadata_value(&data, 0, 8).expect("type 8 = string");
    assert!(matches!(value, GgufValue::String(s) if s == "hello"));
    assert_eq!(consumed, 13);
}

#[test]
fn test_read_metadata_value_array_string() {
    // Array: element_type (4) + count (8) + elements
    let mut data = Vec::new();
    data.extend_from_slice(&8u32.to_le_bytes()); // element type 8 = string
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
                                                 // First string "ab"
    data.extend_from_slice(&2u64.to_le_bytes());
    data.extend_from_slice(b"ab");
    // Second string "cd"
    data.extend_from_slice(&2u64.to_le_bytes());
    data.extend_from_slice(b"cd");
    let (value, consumed) = read_metadata_value(&data, 0, 9).expect("type 9 = array");
    if let GgufValue::ArrayString(arr) = value {
        assert_eq!(arr, vec!["ab".to_string(), "cd".to_string()]);
    } else {
        panic!("expected ArrayString");
    }
    // 4 (type) + 8 (count) + 10 (s1) + 10 (s2) = 32
    assert_eq!(consumed, 32);
}

#[test]
fn test_read_metadata_value_array_uint32() {
    let mut data = Vec::new();
    data.extend_from_slice(&4u32.to_le_bytes()); // element type 4 = uint32
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 elements
    data.extend_from_slice(&10u32.to_le_bytes());
    data.extend_from_slice(&20u32.to_le_bytes());
    data.extend_from_slice(&30u32.to_le_bytes());
    let (value, consumed) = read_metadata_value(&data, 0, 9).expect("type 9 = array");
    if let GgufValue::ArrayUint32(arr) = value {
        assert_eq!(arr, vec![10, 20, 30]);
    } else {
        panic!("expected ArrayUint32");
    }
    // 4 (type) + 8 (count) + 12 (3 * 4) = 24
    assert_eq!(consumed, 24);
}

#[test]
fn test_read_metadata_value_array_int32() {
    let mut data = Vec::new();
    data.extend_from_slice(&5u32.to_le_bytes()); // element type 5 = int32
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
    data.extend_from_slice(&(-10i32).to_le_bytes());
    data.extend_from_slice(&20i32.to_le_bytes());
    let (value, consumed) = read_metadata_value(&data, 0, 9).expect("type 9 = array");
    if let GgufValue::ArrayInt32(arr) = value {
        assert_eq!(arr, vec![-10, 20]);
    } else {
        panic!("expected ArrayInt32");
    }
    // 4 (type) + 8 (count) + 8 (2 * 4) = 20
    assert_eq!(consumed, 20);
}

#[test]
fn test_read_metadata_value_array_float32() {
    let mut data = Vec::new();
    data.extend_from_slice(&6u32.to_le_bytes()); // element type 6 = float32
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
    data.extend_from_slice(&1.5f32.to_le_bytes());
    data.extend_from_slice(&2.5f32.to_le_bytes());
    let (value, consumed) = read_metadata_value(&data, 0, 9).expect("type 9 = array");
    if let GgufValue::ArrayFloat32(arr) = value {
        assert!((arr[0] - 1.5).abs() < 0.001);
        assert!((arr[1] - 2.5).abs() < 0.001);
    } else {
        panic!("expected ArrayFloat32");
    }
    // 4 (type) + 8 (count) + 8 (2 * 4) = 20
    assert_eq!(consumed, 20);
}

// ========================================================================
// f16_to_f32 tests
// ========================================================================

#[test]
fn test_f16_to_f32_zero() {
    assert_eq!(f16_to_f32(0x0000), 0.0);
}

#[test]
fn test_f16_to_f32_one() {
    // F16 1.0 = 0x3C00
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_negative_one() {
    // F16 -1.0 = 0xBC00
    let result = f16_to_f32(0xBC00);
    assert!((result + 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_subnormal() {
    // Very small subnormal number
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0 && result < 0.001);
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_gguf_value_type_enum_values() {
    assert_eq!(GgufValueType::Uint8 as u32, 0);
    assert_eq!(GgufValueType::Int8 as u32, 1);
    assert_eq!(GgufValueType::Uint16 as u32, 2);
    assert_eq!(GgufValueType::Int16 as u32, 3);
    assert_eq!(GgufValueType::Uint32 as u32, 4);
    assert_eq!(GgufValueType::Int32 as u32, 5);
    assert_eq!(GgufValueType::Float32 as u32, 6);
    assert_eq!(GgufValueType::Bool as u32, 7);
    assert_eq!(GgufValueType::String as u32, 8);
    assert_eq!(GgufValueType::Array as u32, 9);
    assert_eq!(GgufValueType::Uint64 as u32, 10);
    assert_eq!(GgufValueType::Int64 as u32, 11);
    assert_eq!(GgufValueType::Float64 as u32, 12);
}

#[test]
fn test_ggml_type_enum_values() {
    assert_eq!(GgmlType::F32 as u32, 0);
    assert_eq!(GgmlType::F16 as u32, 1);
    assert_eq!(GgmlType::Q4_0 as u32, 2);
    assert_eq!(GgmlType::Q4_1 as u32, 3);
    assert_eq!(GgmlType::Q8_0 as u32, 8);
    assert_eq!(GgmlType::I8 as u32, 24);
    assert_eq!(GgmlType::I16 as u32, 25);
    assert_eq!(GgmlType::I32 as u32, 26);
    assert_eq!(GgmlType::I64 as u32, 27);
    assert_eq!(GgmlType::F64 as u32, 28);
}

#[test]
fn test_gguf_value_type_debug() {
    let t = GgufValueType::String;
    assert!(format!("{:?}", t).contains("String"));
}

#[test]
fn test_ggml_type_debug() {
    let t = GgmlType::F32;
    assert!(format!("{:?}", t).contains("F32"));
}

#[test]
fn test_gguf_value_type_clone_copy() {
    let t1 = GgufValueType::Uint32;
    let t2 = t1; // Copy
    let t3 = t1.clone();
    assert_eq!(t1, t2);
    assert_eq!(t1, t3);
}

#[test]
fn test_ggml_type_clone_copy() {
    let t1 = GgmlType::F16;
    let t2 = t1;
    let t3 = t1.clone();
    assert_eq!(t1, t2);
    assert_eq!(t1, t3);
}

#[test]
fn test_gguf_header_debug() {
    let header = GgufHeader {
        version: GGUF_VERSION,
        tensor_count: 10,
        metadata_kv_count: 5,
    };
    assert!(format!("{:?}", header).contains("GgufHeader"));
}

#[test]
fn test_gguf_tensor_info_debug() {
    let info = GgufTensorInfo {
        name: "test".to_string(),
        n_dims: 2,
        dims: vec![10, 20],
        dtype: GgmlType::F32,
        offset: 0,
    };
    assert!(format!("{:?}", info).contains("GgufTensorInfo"));
}

#[test]
fn test_gguf_tensor_debug() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![10],
        dtype: GgmlType::F32,
        data: vec![0u8; 40],
    };
    assert!(format!("{:?}", tensor).contains("GgufTensor"));
}

#[test]
fn test_gguf_value_debug() {
    let v = GgufValue::Uint32(42);
    assert!(format!("{:?}", v).contains("Uint32"));
}

#[test]
fn test_f16_to_f32_max() {
    // F16 max ~65504
    let result = f16_to_f32(0x7BFF);
    assert!((result - 65504.0).abs() < 100.0);
}

#[test]
fn test_f16_to_f32_negative_zero() {
    // Negative zero in f16 = 0x8000
    let result = f16_to_f32(0x8000);
    assert!((result - 0.0).abs() < 0.001 || result == -0.0);
}

#[test]
fn test_f16_to_f32_half() {
    // F16 0.5 = 0x3800
    let result = f16_to_f32(0x3800);
    assert!((result - 0.5).abs() < 0.001);
}

include!("advanced_part_02.rs");
include!("advanced_part_03.rs");
