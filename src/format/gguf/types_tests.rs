pub(crate) use super::*;

// ============================================================================
// Constants Tests
// ============================================================================

#[test]
fn test_gguf_magic_is_correct() {
    // "GGUF" in little-endian
    assert_eq!(GGUF_MAGIC, 0x4655_4747);
    let bytes = GGUF_MAGIC.to_le_bytes();
    assert_eq!(&bytes, b"GGUF");
}

#[test]
fn test_gguf_version_is_v3() {
    assert_eq!(GGUF_VERSION, 3);
}

#[test]
fn test_gguf_default_alignment() {
    assert_eq!(GGUF_DEFAULT_ALIGNMENT, 32);
}

// ============================================================================
// GgufValueType Tests
// ============================================================================

#[test]
fn test_gguf_value_type_discriminants() {
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
fn test_gguf_value_type_clone_eq() {
    let t1 = GgufValueType::Float32;
    let t2 = t1;
    assert_eq!(t1, t2);
}

#[test]
fn test_gguf_value_type_debug() {
    let s = format!("{:?}", GgufValueType::String);
    assert!(s.contains("String"));
}

// ============================================================================
// GgmlType Tests
// ============================================================================

#[test]
fn test_ggml_type_discriminants() {
    assert_eq!(GgmlType::F32 as u32, 0);
    assert_eq!(GgmlType::F16 as u32, 1);
    assert_eq!(GgmlType::Q4_0 as u32, 2);
    assert_eq!(GgmlType::Q4_1 as u32, 3);
    assert_eq!(GgmlType::Q8_0 as u32, 8);
    assert_eq!(GgmlType::Q4K as u32, 12);
    assert_eq!(GgmlType::Q6K as u32, 14);
    assert_eq!(GgmlType::I8 as u32, 24);
    assert_eq!(GgmlType::I16 as u32, 25);
    assert_eq!(GgmlType::I32 as u32, 26);
    assert_eq!(GgmlType::I64 as u32, 27);
    assert_eq!(GgmlType::F64 as u32, 28);
}

#[test]
fn test_ggml_type_clone_eq() {
    let t1 = GgmlType::Q4K;
    let t2 = t1;
    assert_eq!(t1, t2);
}

#[test]
fn test_ggml_type_debug() {
    let s = format!("{:?}", GgmlType::Q6K);
    assert!(s.contains("Q6K"));
}

// ============================================================================
// GgufValue Tests
// ============================================================================

#[test]
fn test_gguf_value_type_mapping() {
    assert_eq!(GgufValue::Uint8(42).value_type(), GgufValueType::Uint8);
    assert_eq!(GgufValue::Int8(-1).value_type(), GgufValueType::Int8);
    assert_eq!(GgufValue::Uint16(1000).value_type(), GgufValueType::Uint16);
    assert_eq!(GgufValue::Int16(-500).value_type(), GgufValueType::Int16);
    assert_eq!(
        GgufValue::Uint32(100_000).value_type(),
        GgufValueType::Uint32
    );
    assert_eq!(
        GgufValue::Int32(-100_000).value_type(),
        GgufValueType::Int32
    );
    assert_eq!(
        GgufValue::Float32(3.14).value_type(),
        GgufValueType::Float32
    );
    assert_eq!(GgufValue::Bool(true).value_type(), GgufValueType::Bool);
    assert_eq!(
        GgufValue::String("test".to_string()).value_type(),
        GgufValueType::String
    );
    assert_eq!(
        GgufValue::Uint64(u64::MAX).value_type(),
        GgufValueType::Uint64
    );
    assert_eq!(
        GgufValue::Int64(i64::MIN).value_type(),
        GgufValueType::Int64
    );
    assert_eq!(
        GgufValue::Float64(2.718).value_type(),
        GgufValueType::Float64
    );
}

#[test]
fn test_gguf_value_array_types() {
    assert_eq!(
        GgufValue::ArrayUint32(vec![1, 2, 3]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayInt32(vec![-1, 0, 1]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayFloat32(vec![1.0, 2.0]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayString(vec!["a".to_string()]).value_type(),
        GgufValueType::Array
    );
}

#[test]
fn test_gguf_value_clone() {
    let v1 = GgufValue::String("hello".to_string());
    let v2 = v1.clone();
    if let (GgufValue::String(s1), GgufValue::String(s2)) = (&v1, &v2) {
        assert_eq!(s1, s2);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_gguf_value_debug() {
    let v = GgufValue::Float32(1.5);
    let s = format!("{v:?}");
    assert!(s.contains("Float32"));
    assert!(s.contains("1.5"));
}

// ============================================================================
// GgufHeader Tests
// ============================================================================

#[test]
fn test_gguf_header_write_to() {
    let header = GgufHeader {
        version: 3,
        tensor_count: 10,
        metadata_kv_count: 5,
    };
    let mut buf = Vec::new();
    header.write_to(&mut buf).expect("write header");

    // Check magic (4 bytes)
    assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
    // Check version (4 bytes)
    assert_eq!(&buf[4..8], &3u32.to_le_bytes());
    // Check tensor_count (8 bytes)
    assert_eq!(&buf[8..16], &10u64.to_le_bytes());
    // Check metadata_kv_count (8 bytes)
    assert_eq!(&buf[16..24], &5u64.to_le_bytes());
}

#[test]
fn test_gguf_header_clone_debug() {
    let header = GgufHeader {
        version: 3,
        tensor_count: 1,
        metadata_kv_count: 2,
    };
    let cloned = header.clone();
    assert_eq!(header.version, cloned.version);
    let s = format!("{header:?}");
    assert!(s.contains("GgufHeader"));
}

// ============================================================================
// GgufTensorInfo Tests
// ============================================================================

#[test]
fn test_gguf_tensor_info_write_to() {
    let info = GgufTensorInfo {
        name: "test".to_string(),
        n_dims: 2,
        dims: vec![3, 4],
        dtype: GgmlType::F32,
        offset: 128,
    };
    let mut buf = Vec::new();
    info.write_to(&mut buf).expect("write tensor info");

    // Name is length-prefixed: 8 bytes length + "test" (4 bytes)
    assert_eq!(&buf[0..8], &4u64.to_le_bytes());
    assert_eq!(&buf[8..12], b"test");
    // n_dims (4 bytes)
    assert_eq!(&buf[12..16], &2u32.to_le_bytes());
    // dims[0] (8 bytes)
    assert_eq!(&buf[16..24], &3u64.to_le_bytes());
    // dims[1] (8 bytes)
    assert_eq!(&buf[24..32], &4u64.to_le_bytes());
    // dtype (4 bytes)
    assert_eq!(&buf[32..36], &(GgmlType::F32 as u32).to_le_bytes());
    // offset (8 bytes)
    assert_eq!(&buf[36..44], &128u64.to_le_bytes());
}

#[test]
fn test_gguf_tensor_info_clone_debug() {
    let info = GgufTensorInfo {
        name: "layer.0.weight".to_string(),
        n_dims: 3,
        dims: vec![2, 3, 4],
        dtype: GgmlType::F16,
        offset: 0,
    };
    let cloned = info.clone();
    assert_eq!(info.name, cloned.name);
    let s = format!("{info:?}");
    assert!(s.contains("GgufTensorInfo"));
}

// ============================================================================
// GgufTensor Tests
// ============================================================================

#[test]
fn test_gguf_tensor_byte_size_f32() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::F32,
        data: vec![],
    };
    // 10 * 20 = 200 elements * 4 bytes = 800
    assert_eq!(tensor.byte_size(), 800);
}

#[test]
fn test_gguf_tensor_byte_size_f16() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![100],
        dtype: GgmlType::F16,
        data: vec![],
    };
    // 100 elements * 2 bytes = 200
    assert_eq!(tensor.byte_size(), 200);
}

#[test]
fn test_gguf_tensor_byte_size_i8() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![256],
        dtype: GgmlType::I8,
        data: vec![],
    };
    // 256 elements * 1 byte = 256
    assert_eq!(tensor.byte_size(), 256);
}

#[test]
fn test_gguf_tensor_byte_size_i16() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![128],
        dtype: GgmlType::I16,
        data: vec![],
    };
    // 128 elements * 2 bytes = 256
    assert_eq!(tensor.byte_size(), 256);
}

#[test]
fn test_gguf_tensor_byte_size_i32() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![64],
        dtype: GgmlType::I32,
        data: vec![],
    };
    // 64 elements * 4 bytes = 256
    assert_eq!(tensor.byte_size(), 256);
}

#[test]
fn test_gguf_tensor_byte_size_i64_f64() {
    let tensor_i64 = GgufTensor {
        name: "test".to_string(),
        shape: vec![32],
        dtype: GgmlType::I64,
        data: vec![],
    };
    // 32 elements * 8 bytes = 256
    assert_eq!(tensor_i64.byte_size(), 256);

    let tensor_f64 = GgufTensor {
        name: "test".to_string(),
        shape: vec![16],
        dtype: GgmlType::F64,
        data: vec![],
    };
    // 16 elements * 8 bytes = 128
    assert_eq!(tensor_f64.byte_size(), 128);
}

#[test]
fn test_gguf_tensor_byte_size_q4_0() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![64],
        dtype: GgmlType::Q4_0,
        data: vec![],
    };
    // 64 elements / 32 = 2 blocks * 18 bytes = 36
    assert_eq!(tensor.byte_size(), 36);
}

#[test]
fn test_gguf_tensor_byte_size_q4_1() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![96],
        dtype: GgmlType::Q4_1,
        data: vec![],
    };
    // (96 + 31) / 32 = 3 blocks * 18 bytes = 54
    assert_eq!(tensor.byte_size(), 54);
}

#[test]
fn test_gguf_tensor_byte_size_q8_0() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![64],
        dtype: GgmlType::Q8_0,
        data: vec![],
    };
    // 64 elements / 32 = 2 blocks * 34 bytes = 68
    assert_eq!(tensor.byte_size(), 68);
}

#[test]
fn test_gguf_tensor_byte_size_q4k() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![512],
        dtype: GgmlType::Q4K,
        data: vec![],
    };
    // 512 elements / 256 = 2 super-blocks * 144 bytes = 288
    assert_eq!(tensor.byte_size(), 288);
}

#[test]
fn test_gguf_tensor_byte_size_q6k() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![256],
        dtype: GgmlType::Q6K,
        data: vec![],
    };
    // 256 elements / 256 = 1 super-block * 210 bytes = 210
    assert_eq!(tensor.byte_size(), 210);
}

#[test]
fn test_gguf_tensor_clone_debug() {
    let tensor = GgufTensor {
        name: "weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data: vec![0u8; 64],
    };
    let cloned = tensor.clone();
    assert_eq!(tensor.name, cloned.name);
    let s = format!("{tensor:?}");
    assert!(s.contains("GgufTensor"));
}

// ============================================================================
// padding_for_alignment Tests
// ============================================================================

#[test]
fn test_padding_for_alignment_already_aligned() {
    assert_eq!(padding_for_alignment(0, 32), 0);
    assert_eq!(padding_for_alignment(32, 32), 0);
    assert_eq!(padding_for_alignment(64, 32), 0);
}

#[test]
fn test_padding_for_alignment_needs_padding() {
    assert_eq!(padding_for_alignment(1, 32), 31);
    assert_eq!(padding_for_alignment(16, 32), 16);
    assert_eq!(padding_for_alignment(31, 32), 1);
    assert_eq!(padding_for_alignment(33, 32), 31);
}

#[test]
fn test_padding_for_alignment_different_alignments() {
    assert_eq!(padding_for_alignment(5, 8), 3);
    assert_eq!(padding_for_alignment(7, 16), 9);
    assert_eq!(padding_for_alignment(100, 64), 28);
}

#[path = "types_tests_write_metadata.rs"]
mod types_tests_write_metadata;
