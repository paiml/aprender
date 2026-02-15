//! Core GGUF Tests - Value types, write tests, accessor tests
//! PMAT-085: Split from tests.rs for file health (lines 1-1089)

use super::super::*;

#[test]
fn test_magic_constant() {
    assert_eq!(GGUF_MAGIC, 0x4655_4747);
    assert_eq!(&GGUF_MAGIC.to_le_bytes(), b"GGUF");
}

#[test]
fn test_header_size() {
    let mut buffer = Vec::new();
    let header = GgufHeader {
        version: GGUF_VERSION,
        tensor_count: 0,
        metadata_kv_count: 0,
    };
    header.write_to(&mut buffer).expect("write");
    // magic (4) + version (4) + tensor_count (8) + kv_count (8) = 24
    assert_eq!(buffer.len(), 24);
}

#[test]
fn test_padding_calculation() {
    assert_eq!(padding_for_alignment(0, 32), 0);
    assert_eq!(padding_for_alignment(1, 32), 31);
    assert_eq!(padding_for_alignment(32, 32), 0);
    assert_eq!(padding_for_alignment(33, 32), 31);
    assert_eq!(padding_for_alignment(64, 32), 0);
}

#[test]
fn test_value_types() {
    assert_eq!(GgufValue::Uint32(42).value_type(), GgufValueType::Uint32);
    assert_eq!(
        GgufValue::String("test".to_string()).value_type(),
        GgufValueType::String
    );
    assert_eq!(GgufValue::Bool(true).value_type(), GgufValueType::Bool);
}

// ========================================================================
// GgufValue type coverage tests
// ========================================================================

#[test]
fn test_all_value_types() {
    assert_eq!(GgufValue::Uint8(1).value_type(), GgufValueType::Uint8);
    assert_eq!(GgufValue::Int8(-1).value_type(), GgufValueType::Int8);
    assert_eq!(GgufValue::Uint16(1).value_type(), GgufValueType::Uint16);
    assert_eq!(GgufValue::Int16(-1).value_type(), GgufValueType::Int16);
    assert_eq!(GgufValue::Uint32(1).value_type(), GgufValueType::Uint32);
    assert_eq!(GgufValue::Int32(-1).value_type(), GgufValueType::Int32);
    assert_eq!(GgufValue::Float32(1.0).value_type(), GgufValueType::Float32);
    assert_eq!(GgufValue::Bool(true).value_type(), GgufValueType::Bool);
    assert_eq!(
        GgufValue::String("s".into()).value_type(),
        GgufValueType::String
    );
    assert_eq!(GgufValue::Uint64(1).value_type(), GgufValueType::Uint64);
    assert_eq!(GgufValue::Int64(-1).value_type(), GgufValueType::Int64);
    assert_eq!(GgufValue::Float64(1.0).value_type(), GgufValueType::Float64);
    assert_eq!(
        GgufValue::ArrayUint32(vec![1]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayInt32(vec![1]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayFloat32(vec![1.0]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayString(vec!["s".into()]).value_type(),
        GgufValueType::Array
    );
}

// ========================================================================
// write_metadata_kv tests
// ========================================================================

#[test]
fn test_write_metadata_kv_uint8() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "test", &GgufValue::Uint8(42)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_int8() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "key", &GgufValue::Int8(-5)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_uint16() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Uint16(1000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_int16() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Int16(-1000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_uint32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Uint32(100_000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_int32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Int32(-100_000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_float32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Float32(3.14)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_bool() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Bool(true)).unwrap();
    assert!(!buf.is_empty());

    let mut buf2 = Vec::new();
    write_metadata_kv(&mut buf2, "k", &GgufValue::Bool(false)).unwrap();
    assert!(!buf2.is_empty());
}

#[test]
fn test_write_metadata_kv_string() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::String("hello".into())).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_uint64() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Uint64(1_000_000_000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_int64() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Int64(-1_000_000_000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_float64() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Float64(3.14159265359)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_array_uint32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::ArrayUint32(vec![1, 2, 3])).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_array_int32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::ArrayInt32(vec![-1, 0, 1])).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_array_float32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::ArrayFloat32(vec![1.0, 2.0, 3.0])).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_array_string() {
    let mut buf = Vec::new();
    write_metadata_kv(
        &mut buf,
        "k",
        &GgufValue::ArrayString(vec!["a".into(), "b".into()]),
    )
    .unwrap();
    assert!(!buf.is_empty());
}

// ========================================================================
// GgufTensorInfo tests
// ========================================================================

#[test]
fn test_tensor_info_write() {
    let info = GgufTensorInfo {
        name: "test.weight".to_string(),
        n_dims: 2,
        dims: vec![10, 20],
        dtype: GgmlType::F32,
        offset: 0,
    };
    let mut buf = Vec::new();
    info.write_to(&mut buf).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_tensor_info_all_dtypes() {
    // Test write_to works for various dtypes
    let dtypes = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
    ];
    for dtype in dtypes {
        let info = GgufTensorInfo {
            name: "w".to_string(),
            n_dims: 2,
            dims: vec![10, 20],
            dtype,
            offset: 0,
        };
        let mut buf = Vec::new();
        info.write_to(&mut buf).unwrap();
        assert!(!buf.is_empty());
    }
}

#[test]
fn test_ggml_type_all_variants() {
    // Test all GgmlType variants for Debug and Eq
    let types = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q8_0,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
    ];
    for t in types {
        assert_eq!(t, t);
        assert!(!format!("{t:?}").is_empty());
    }
}

#[test]
fn test_gguf_value_type_all_variants() {
    // Test all GgufValueType variants
    let types = [
        GgufValueType::Uint8,
        GgufValueType::Int8,
        GgufValueType::Uint16,
        GgufValueType::Int16,
        GgufValueType::Uint32,
        GgufValueType::Int32,
        GgufValueType::Float32,
        GgufValueType::Bool,
        GgufValueType::String,
        GgufValueType::Array,
        GgufValueType::Uint64,
        GgufValueType::Int64,
        GgufValueType::Float64,
    ];
    for t in types {
        assert_eq!(t, t);
        assert!(!format!("{t:?}").is_empty());
    }
}

// ========================================================================
// Enum Debug/Clone tests
// ========================================================================

#[test]
fn test_gguf_value_type_enum() {
    let t = GgufValueType::Uint8;
    assert_eq!(t, GgufValueType::Uint8);
    let cloned = t;
    assert_eq!(t, cloned);
    assert!(format!("{t:?}").contains("Uint8"));
}

#[test]
fn test_ggml_type_enum() {
    let t = GgmlType::F32;
    assert_eq!(t, GgmlType::F32);
    let cloned = t;
    assert_eq!(t, cloned);
    assert!(format!("{t:?}").contains("F32"));
}

#[test]
fn test_gguf_value_clone() {
    let v = GgufValue::String("test".to_string());
    let cloned = v.clone();
    assert!(format!("{cloned:?}").contains("test"));
}

#[test]
fn test_gguf_header_clone() {
    let h = GgufHeader {
        version: 3,
        tensor_count: 10,
        metadata_kv_count: 5,
    };
    let cloned = h.clone();
    assert_eq!(cloned.version, 3);
    assert_eq!(cloned.tensor_count, 10);
    assert!(format!("{cloned:?}").contains("GgufHeader"));
}

// ========================================================================
// GgufTensor tests
// ========================================================================

#[test]
fn test_gguf_tensor_byte_size_f32() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::F32,
        data: vec![0; 800], // 10 * 20 * 4
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_f16() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::F16,
        data: vec![0; 400], // 10 * 20 * 2
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_i8() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::I8,
        data: vec![0; 200], // 10 * 20 * 1
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_i16() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::I16,
        data: vec![0; 400],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_i32() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::I32,
        data: vec![0; 800],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_i64() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::I64,
        data: vec![0; 1600],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_f64() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::F64,
        data: vec![0; 1600],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_q4_0() {
    let tensor = GgufTensor {
        name: "quantized".to_string(),
        shape: vec![64],
        dtype: GgmlType::Q4_0,
        data: vec![0; 100],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_q4_1() {
    let tensor = GgufTensor {
        name: "quantized".to_string(),
        shape: vec![64],
        dtype: GgmlType::Q4_1,
        data: vec![0; 100],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

include!("core_part_02.rs");
include!("core_part_03.rs");
