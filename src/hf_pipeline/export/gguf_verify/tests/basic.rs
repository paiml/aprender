use super::*;

#[test]
fn test_verify_empty_gguf() {
    let data = write_gguf(&[], &[]);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.version, 3);
    assert_eq!(summary.tensor_count, 0);
    assert_eq!(summary.metadata_count, 0);
}

#[test]
fn test_verify_with_metadata() {
    let metadata = vec![("general.name".into(), GgufValue::String("test".into()))];
    let data = write_gguf(&[], &metadata);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_with_tensors() {
    let tensors = vec![
        GgufTensor {
            name: "w1".into(),
            shape: vec![3],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec(),
        },
        GgufTensor {
            name: "w2".into(),
            shape: vec![2],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&[4.0f32, 5.0]).to_vec(),
        },
    ];
    let data = write_gguf(&tensors, &[]);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensor_count, 2);
    assert_eq!(summary.tensors.len(), 2);
    assert_eq!(summary.tensors[0].name, "w1");
    assert_eq!(summary.tensors[1].name, "w2");
}

#[test]
fn test_roundtrip_f32() {
    let metadata = vec![("general.architecture".into(), GgufValue::String("llama".into()))];
    let tensors = vec![GgufTensor {
        name: "layer.0.weight".into(),
        shape: vec![8, 16],
        dtype: GgmlType::F32,
        data: bytemuck::cast_slice(&[1.0f32; 128]).to_vec(),
    }];
    let data = write_gguf(&tensors, &metadata);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensor_count, 1);
    assert_eq!(summary.metadata_count, 1);
    assert_eq!(summary.tensors[0].shape, vec![8, 16]);
    assert_eq!(summary.tensors[0].dtype, 0); // F32
}

#[test]
fn test_roundtrip_q4_0() {
    let (bytes, dtype) = quantize_to_gguf_bytes(&[0.5; 64], GgufQuantization::Q4_0);
    let tensors =
        vec![GgufTensor { name: "quantized".into(), shape: vec![64], dtype, data: bytes }];
    let data = write_gguf(&tensors, &[]);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensors[0].dtype, 2); // Q4_0
}

#[test]
fn test_roundtrip_q8_0() {
    let (bytes, dtype) = quantize_to_gguf_bytes(&[0.5; 32], GgufQuantization::Q8_0);
    let tensors =
        vec![GgufTensor { name: "quantized".into(), shape: vec![32], dtype, data: bytes }];
    let data = write_gguf(&tensors, &[]);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensors[0].dtype, 8); // Q8_0
}

#[test]
fn test_verify_invalid_magic() {
    let data =
        b"GGML\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let result = verify_gguf(data);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("Invalid GGUF magic"));
}

#[test]
fn test_verify_wrong_version() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes()); // version 2
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = verify_gguf(&data);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("Unsupported GGUF version"));
}

#[test]
fn test_verify_too_small() {
    let result = verify_gguf(&[0u8; 10]);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("too small"));
}

#[test]
fn test_verify_truncated_metadata() {
    // Valid header claiming 1 metadata but no actual metadata bytes
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata (but missing)

    let result = verify_gguf(&data);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("truncated"));
}

#[test]
fn test_verify_u32_metadata() {
    let metadata = vec![("layers".into(), GgufValue::Uint32(32))];
    let (bytes, dtype) = quantize_to_gguf_bytes(&[1.0], GgufQuantization::None);
    let tensors = vec![GgufTensor { name: "w".into(), shape: vec![1], dtype, data: bytes }];
    let data = write_gguf(&tensors, &metadata);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
    assert_eq!(summary.tensor_count, 1);
}

#[test]
fn test_verify_f32_metadata() {
    let metadata = vec![("loss".into(), GgufValue::Float32(0.42))];
    let data = write_gguf(&[], &metadata);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_u64_metadata() {
    let metadata = vec![("params".into(), GgufValue::Uint64(7_000_000_000))];
    let data = write_gguf(&[], &metadata);

    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

// =====================================================================
// Coverage: skip_gguf_value match arms for all GGUF metadata types
// =====================================================================

#[test]
fn test_verify_uint8_metadata() {
    let metadata = vec![("flag".into(), GgufValue::Uint8(42))];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_int8_metadata() {
    let metadata = vec![("offset".into(), GgufValue::Int8(-1))];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_bool_metadata() {
    let metadata = vec![("enabled".into(), GgufValue::Bool(true))];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_uint16_metadata() {
    let metadata = vec![("vocab_size".into(), GgufValue::Uint16(32000))];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_int16_metadata() {
    let metadata = vec![("temperature".into(), GgufValue::Int16(-100))];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_int64_metadata() {
    let metadata = vec![("timestamp".into(), GgufValue::Int64(-9_000_000))];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_float64_metadata() {
    let metadata = vec![("learning_rate".into(), GgufValue::Float64(1e-5))];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 1);
}

#[test]
fn test_verify_all_metadata_types_combined() {
    let metadata = vec![
        ("u8_val".into(), GgufValue::Uint8(255)),
        ("i8_val".into(), GgufValue::Int8(-128)),
        ("u16_val".into(), GgufValue::Uint16(65535)),
        ("i16_val".into(), GgufValue::Int16(-32768)),
        ("u32_val".into(), GgufValue::Uint32(100)),
        ("i32_val".into(), GgufValue::Int32(-1)),
        ("f32_val".into(), GgufValue::Float32(3.14)),
        ("bool_val".into(), GgufValue::Bool(false)),
        ("str_val".into(), GgufValue::String("test".into())),
        ("u64_val".into(), GgufValue::Uint64(u64::MAX)),
        ("i64_val".into(), GgufValue::Int64(i64::MIN)),
        ("f64_val".into(), GgufValue::Float64(2.718281828)),
    ];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 12);
}

// =====================================================================
// Direct skip_gguf_value match arm coverage (type tags 0-12)
// =====================================================================

#[test]
fn test_skip_gguf_value_type_0_uint8() {
    // Type 0 (UINT8): skip 1 byte -> match arm 0 | 1 | 7
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 0).expect("operation should succeed"), 1);
}

#[test]
fn test_skip_gguf_value_type_1_int8() {
    // Type 1 (INT8): skip 1 byte -> match arm 0 | 1 | 7
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 1).expect("operation should succeed"), 1);
}

#[test]
fn test_skip_gguf_value_type_7_bool() {
    // Type 7 (BOOL): skip 1 byte -> match arm 0 | 1 | 7
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 7).expect("operation should succeed"), 1);
}

#[test]
fn test_skip_gguf_value_type_2_uint16() {
    // Type 2 (UINT16): skip 2 bytes -> match arm 2 | 3
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 2).expect("operation should succeed"), 2);
}

#[test]
fn test_skip_gguf_value_type_3_int16() {
    // Type 3 (INT16): skip 2 bytes -> match arm 2 | 3
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 3).expect("operation should succeed"), 2);
}

#[test]
fn test_skip_gguf_value_type_4_uint32() {
    // Type 4 (UINT32): skip 4 bytes -> match arm 4..=6
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 4).expect("operation should succeed"), 4);
}

#[test]
fn test_skip_gguf_value_type_5_int32() {
    // Type 5 (INT32): skip 4 bytes -> match arm 4..=6
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 5).expect("operation should succeed"), 4);
}

#[test]
fn test_skip_gguf_value_type_6_float32() {
    // Type 6 (FLOAT32): skip 4 bytes -> match arm 4..=6
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 6).expect("operation should succeed"), 4);
}

#[test]
fn test_skip_gguf_value_type_10_uint64() {
    // Type 10 (UINT64): skip 8 bytes -> match arm 10..=12
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 10).expect("operation should succeed"), 8);
}

#[test]
fn test_skip_gguf_value_type_11_int64() {
    // Type 11 (INT64): skip 8 bytes -> match arm 10..=12
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 11).expect("operation should succeed"), 8);
}

#[test]
fn test_skip_gguf_value_type_12_float64() {
    // Type 12 (FLOAT64): skip 8 bytes -> match arm 10..=12
    let data = vec![0u8; 16];
    assert_eq!(skip_gguf_value(&data, 0, 12).expect("operation should succeed"), 8);
}

#[test]
fn test_skip_gguf_value_unknown_type() {
    let data = vec![0u8; 16];
    assert!(skip_gguf_value(&data, 0, 99).is_err());
}
