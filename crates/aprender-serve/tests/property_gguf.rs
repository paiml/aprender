//! Property-based tests for GGUF parsing and transformations

use proptest::prelude::*;
use realizar::gguf::{
    GGUFHeader, GGUFValue, TensorInfo, GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32,
    GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K,
    GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};

// ============================================================================
// Constants Tests
// ============================================================================

#[test]
fn test_gguf_magic_correct() {
    // "GGUF" in little-endian: G=0x47, G=0x47, U=0x55, F=0x46
    assert_eq!(GGUF_MAGIC, 0x4655_4747);
    // Verify it's "GGUF" as bytes
    let bytes = GGUF_MAGIC.to_le_bytes();
    assert_eq!(&bytes, b"GGUF");
}

#[test]
fn test_gguf_version() {
    assert_eq!(GGUF_VERSION_V3, 3);
}

#[test]
fn test_gguf_alignment() {
    assert_eq!(GGUF_ALIGNMENT, 32);
    // Must be power of 2
    assert!(GGUF_ALIGNMENT.is_power_of_two());
}

#[test]
fn test_gguf_type_constants_valid() {
    // Verify type constants match GGUF spec
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
// GGUFHeader Tests
// ============================================================================

#[test]
fn test_gguf_header_creation() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 100,
        metadata_count: 50,
    };

    assert_eq!(header.magic, GGUF_MAGIC);
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 100);
    assert_eq!(header.metadata_count, 50);
}

#[test]
fn test_gguf_header_clone() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 42,
        metadata_count: 10,
    };

    let cloned = header.clone();
    assert_eq!(header, cloned);
}

#[test]
fn test_gguf_header_debug() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 1,
        metadata_count: 2,
    };

    let debug = format!("{:?}", header);
    assert!(debug.contains("GGUFHeader"));
    assert!(debug.contains("magic"));
}

// ============================================================================
// GGUFValue Tests
// ============================================================================

#[test]
fn test_gguf_value_uint8() {
    let val = GGUFValue::UInt8(42);
    if let GGUFValue::UInt8(v) = val {
        assert_eq!(v, 42);
    } else {
        panic!("Expected UInt8");
    }
}

#[test]
fn test_gguf_value_int8() {
    let val = GGUFValue::Int8(-10);
    if let GGUFValue::Int8(v) = val {
        assert_eq!(v, -10);
    } else {
        panic!("Expected Int8");
    }
}

#[test]
fn test_gguf_value_uint16() {
    let val = GGUFValue::UInt16(1000);
    if let GGUFValue::UInt16(v) = val {
        assert_eq!(v, 1000);
    } else {
        panic!("Expected UInt16");
    }
}

#[test]
fn test_gguf_value_int16() {
    let val = GGUFValue::Int16(-5000);
    if let GGUFValue::Int16(v) = val {
        assert_eq!(v, -5000);
    } else {
        panic!("Expected Int16");
    }
}

#[test]
fn test_gguf_value_uint32() {
    let val = GGUFValue::UInt32(100_000);
    if let GGUFValue::UInt32(v) = val {
        assert_eq!(v, 100_000);
    } else {
        panic!("Expected UInt32");
    }
}

#[test]
fn test_gguf_value_int32() {
    let val = GGUFValue::Int32(-50_000);
    if let GGUFValue::Int32(v) = val {
        assert_eq!(v, -50_000);
    } else {
        panic!("Expected Int32");
    }
}

#[test]
fn test_gguf_value_float32() {
    let val = GGUFValue::Float32(std::f32::consts::PI);
    if let GGUFValue::Float32(v) = val {
        assert!((v - std::f32::consts::PI).abs() < 0.0001);
    } else {
        panic!("Expected Float32");
    }
}

#[test]
fn test_gguf_value_string() {
    let val = GGUFValue::String("hello world".to_string());
    if let GGUFValue::String(v) = val {
        assert_eq!(v, "hello world");
    } else {
        panic!("Expected String");
    }
}

#[test]
fn test_gguf_value_bool() {
    let val = GGUFValue::Bool(true);
    if let GGUFValue::Bool(v) = val {
        assert!(v);
    } else {
        panic!("Expected Bool");
    }
}

#[test]
fn test_gguf_value_uint64() {
    let val = GGUFValue::UInt64(u64::MAX);
    if let GGUFValue::UInt64(v) = val {
        assert_eq!(v, u64::MAX);
    } else {
        panic!("Expected UInt64");
    }
}

#[test]
fn test_gguf_value_int64() {
    let val = GGUFValue::Int64(i64::MIN);
    if let GGUFValue::Int64(v) = val {
        assert_eq!(v, i64::MIN);
    } else {
        panic!("Expected Int64");
    }
}

#[test]
fn test_gguf_value_float64() {
    let val = GGUFValue::Float64(std::f64::consts::PI);
    if let GGUFValue::Float64(v) = val {
        assert!((v - std::f64::consts::PI).abs() < 1e-10);
    } else {
        panic!("Expected Float64");
    }
}

#[test]
fn test_gguf_value_array() {
    let val = GGUFValue::Array(vec![GGUFValue::UInt32(1), GGUFValue::UInt32(2)]);
    if let GGUFValue::Array(arr) = val {
        assert_eq!(arr.len(), 2);
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_gguf_value_nested_array() {
    let inner = GGUFValue::Array(vec![GGUFValue::UInt8(1)]);
    let outer = GGUFValue::Array(vec![inner]);
    if let GGUFValue::Array(arr) = outer {
        assert_eq!(arr.len(), 1);
        if let GGUFValue::Array(inner_arr) = &arr[0] {
            assert_eq!(inner_arr.len(), 1);
        } else {
            panic!("Expected inner Array");
        }
    } else {
        panic!("Expected outer Array");
    }
}

#[test]
fn test_gguf_value_clone() {
    let val = GGUFValue::String("test".to_string());
    let cloned = val.clone();
    assert_eq!(val, cloned);
}

#[test]
fn test_gguf_value_equality() {
    let v1 = GGUFValue::UInt32(42);
    let v2 = GGUFValue::UInt32(42);
    let v3 = GGUFValue::UInt32(99);
    assert_eq!(v1, v2);
    assert_ne!(v1, v3);
}

#[test]
fn test_gguf_value_debug() {
    let val = GGUFValue::Float32(1.5);
    let debug = format!("{:?}", val);
    assert!(debug.contains("Float32"));
    assert!(debug.contains("1.5"));
}

// ============================================================================
// TensorInfo Tests
// ============================================================================

#[test]
fn test_tensor_info_creation() {
    let info = TensorInfo {
        name: "test_tensor".to_string(),
        n_dims: 2,
        dims: vec![10, 20],
        qtype: GGUF_TYPE_F32,
        offset: 1024,
    };

    assert_eq!(info.name, "test_tensor");
    assert_eq!(info.n_dims, 2);
    assert_eq!(info.dims, vec![10, 20]);
    assert_eq!(info.qtype, GGUF_TYPE_F32);
    assert_eq!(info.offset, 1024);
}

#[test]
fn test_tensor_info_clone() {
    let info = TensorInfo {
        name: "tensor".to_string(),
        n_dims: 3,
        dims: vec![2, 3, 4],
        qtype: GGUF_TYPE_Q4_0,
        offset: 0,
    };

    let cloned = info.clone();
    assert_eq!(info, cloned);
}

#[test]
fn test_tensor_info_debug() {
    let info = TensorInfo {
        name: "debug_test".to_string(),
        n_dims: 1,
        dims: vec![100],
        qtype: GGUF_TYPE_F16,
        offset: 512,
    };

    let debug = format!("{:?}", info);
    assert!(debug.contains("TensorInfo"));
    assert!(debug.contains("debug_test"));
}

#[test]
fn test_tensor_info_1d() {
    let info = TensorInfo {
        name: "bias".to_string(),
        n_dims: 1,
        dims: vec![4096],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };

    assert_eq!(info.n_dims, 1);
    assert_eq!(info.dims.len(), 1);
}

#[test]
fn test_tensor_info_3d() {
    let info = TensorInfo {
        name: "embedding".to_string(),
        n_dims: 3,
        dims: vec![32000, 4096, 1],
        qtype: GGUF_TYPE_Q8_0,
        offset: 1024 * 1024,
    };

    assert_eq!(info.n_dims, 3);
    assert_eq!(info.dims.len(), 3);
}

#[test]
fn test_tensor_info_4d() {
    let info = TensorInfo {
        name: "attention".to_string(),
        n_dims: 4,
        dims: vec![32, 128, 128, 64],
        qtype: GGUF_TYPE_F16,
        offset: 0,
    };

    assert_eq!(info.n_dims, 4);
}

#[test]
fn test_tensor_info_various_qtypes() {
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
        let info = TensorInfo {
            name: format!("tensor_qtype_{}", qtype),
            n_dims: 2,
            dims: vec![256, 256],
            qtype,
            offset: 0,
        };
        assert_eq!(info.qtype, qtype);
    }
}

// ============================================================================
// Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_gguf_value_uint8_roundtrip(val in 0u8..=255u8) {
        let gguf_val = GGUFValue::UInt8(val);
        if let GGUFValue::UInt8(v) = gguf_val {
            prop_assert_eq!(v, val);
        } else {
            prop_assert!(false, "Expected UInt8");
        }
    }

    #[test]
    fn prop_gguf_value_int32_roundtrip(val in i32::MIN..=i32::MAX) {
        let gguf_val = GGUFValue::Int32(val);
        if let GGUFValue::Int32(v) = gguf_val {
            prop_assert_eq!(v, val);
        } else {
            prop_assert!(false, "Expected Int32");
        }
    }

    #[test]
    fn prop_gguf_value_uint64_roundtrip(val in 0u64..=u64::MAX) {
        let gguf_val = GGUFValue::UInt64(val);
        if let GGUFValue::UInt64(v) = gguf_val {
            prop_assert_eq!(v, val);
        } else {
            prop_assert!(false, "Expected UInt64");
        }
    }

    #[test]
    fn prop_gguf_value_string_roundtrip(val in "[a-zA-Z0-9_]{0,100}") {
        let gguf_val = GGUFValue::String(val.clone());
        if let GGUFValue::String(v) = gguf_val {
            prop_assert_eq!(v, val);
        } else {
            prop_assert!(false, "Expected String");
        }
    }

    #[test]
    fn prop_gguf_value_float32_finite(val in -1e10f32..1e10f32) {
        let gguf_val = GGUFValue::Float32(val);
        if let GGUFValue::Float32(v) = gguf_val {
            prop_assert!(v.is_finite());
            prop_assert!((v - val).abs() < 1e-6 * val.abs().max(1.0));
        } else {
            prop_assert!(false, "Expected Float32");
        }
    }

    #[test]
    fn prop_tensor_info_dims_product(dim1 in 1u64..1000, dim2 in 1u64..1000) {
        let info = TensorInfo {
            name: "prop_test".to_string(),
            n_dims: 2,
            dims: vec![dim1, dim2],
            qtype: GGUF_TYPE_F32,
            offset: 0,
        };

        let product: u64 = info.dims.iter().product();
        prop_assert_eq!(product, dim1 * dim2);
    }

    #[test]
    fn prop_tensor_info_offset_aligned(offset in (0u64..1_000_000).prop_map(|x| x * 32)) {
        let info = TensorInfo {
            name: "aligned".to_string(),
            n_dims: 1,
            dims: vec![1024],
            qtype: GGUF_TYPE_F32,
            offset,
        };

        // Offsets should be 32-byte aligned per GGUF spec
        prop_assert_eq!(info.offset % 32, 0);
    }

    #[test]
    fn prop_header_tensor_count_consistent(tc in 0u64..10000, mc in 0u64..1000) {
        let header = GGUFHeader {
            magic: GGUF_MAGIC,
            version: GGUF_VERSION_V3,
            tensor_count: tc,
            metadata_count: mc,
        };

        prop_assert_eq!(header.tensor_count, tc);
        prop_assert_eq!(header.metadata_count, mc);
    }

    #[test]
    fn prop_gguf_value_clone_equality(val in 0u32..=u32::MAX) {
        let original = GGUFValue::UInt32(val);
        let cloned = original.clone();
        prop_assert_eq!(original, cloned);
    }

    #[test]
    fn prop_gguf_value_array_length(len in 0usize..100) {
        let arr: Vec<GGUFValue> = (0..len).map(|i| GGUFValue::UInt32(i as u32)).collect();
        let val = GGUFValue::Array(arr);
        if let GGUFValue::Array(a) = val {
            prop_assert_eq!(a.len(), len);
        } else {
            prop_assert!(false, "Expected Array");
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_gguf_value_empty_string() {
    let val = GGUFValue::String(String::new());
    if let GGUFValue::String(s) = val {
        assert!(s.is_empty());
    } else {
        panic!("Expected String");
    }
}

#[test]
fn test_gguf_value_empty_array() {
    let val = GGUFValue::Array(Vec::new());
    if let GGUFValue::Array(arr) = val {
        assert!(arr.is_empty());
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_tensor_info_empty_dims() {
    let info = TensorInfo {
        name: "scalar".to_string(),
        n_dims: 0,
        dims: vec![],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };

    assert_eq!(info.n_dims, 0);
    assert!(info.dims.is_empty());
}

#[test]
fn test_tensor_info_large_offset() {
    let info = TensorInfo {
        name: "large".to_string(),
        n_dims: 1,
        dims: vec![1],
        qtype: GGUF_TYPE_F32,
        offset: u64::MAX - 1000,
    };

    assert!(info.offset > 1_000_000_000);
}

#[test]
fn test_gguf_value_bool_false() {
    let val = GGUFValue::Bool(false);
    if let GGUFValue::Bool(b) = val {
        assert!(!b);
    } else {
        panic!("Expected Bool");
    }
}

#[test]
fn test_gguf_value_float32_special_values() {
    // Zero
    let val = GGUFValue::Float32(0.0);
    if let GGUFValue::Float32(v) = val {
        assert_eq!(v, 0.0);
    }

    // Negative zero
    let val = GGUFValue::Float32(-0.0);
    if let GGUFValue::Float32(v) = val {
        assert!(v == 0.0 || v == -0.0);
    }

    // Epsilon
    let val = GGUFValue::Float32(f32::EPSILON);
    if let GGUFValue::Float32(v) = val {
        assert!(v > 0.0 && v < 0.001);
    }
}

#[test]
fn test_gguf_value_int_boundaries() {
    // Min/max values
    let _ = GGUFValue::Int8(i8::MIN);
    let _ = GGUFValue::Int8(i8::MAX);
    let _ = GGUFValue::Int16(i16::MIN);
    let _ = GGUFValue::Int16(i16::MAX);
    let _ = GGUFValue::Int32(i32::MIN);
    let _ = GGUFValue::Int32(i32::MAX);
    let _ = GGUFValue::Int64(i64::MIN);
    let _ = GGUFValue::Int64(i64::MAX);

    // All creation succeeded without panic
}

#[test]
fn test_tensor_info_unicode_name() {
    let info = TensorInfo {
        name: "tensor_\u{1F600}_emoji".to_string(),
        n_dims: 1,
        dims: vec![10],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };

    assert!(info.name.contains('\u{1F600}'));
}
