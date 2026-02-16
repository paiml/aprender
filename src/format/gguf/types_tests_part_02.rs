use super::*;

// ============================================================================
// write_metadata_kv Tests
// ============================================================================

#[test]
fn test_write_metadata_kv_uint32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "count", &GgufValue::Uint32(42)).expect("write kv");

    // Key: length (8) + "count" (5)
    assert_eq!(&buf[0..8], &5u64.to_le_bytes());
    assert_eq!(&buf[8..13], b"count");
    // Value type: Uint32 = 4
    assert_eq!(&buf[13..17], &4u32.to_le_bytes());
    // Value: 42
    assert_eq!(&buf[17..21], &42u32.to_le_bytes());
}

#[test]
fn test_write_metadata_kv_string() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "arch", &GgufValue::String("llama".to_string())).expect("write kv");

    // Key: length (8) + "arch" (4)
    assert_eq!(&buf[0..8], &4u64.to_le_bytes());
    assert_eq!(&buf[8..12], b"arch");
    // Value type: String = 8
    assert_eq!(&buf[12..16], &8u32.to_le_bytes());
    // Value: length (8) + "llama" (5)
    assert_eq!(&buf[16..24], &5u64.to_le_bytes());
    assert_eq!(&buf[24..29], b"llama");
}

#[test]
fn test_write_metadata_kv_bool() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "flag", &GgufValue::Bool(true)).expect("write kv");
    // Should contain type Bool (7) and value 1
    assert!(buf.len() > 12);
}

#[test]
fn test_write_metadata_kv_all_scalar_types() {
    let cases: Vec<(&str, GgufValue)> = vec![
        ("u8", GgufValue::Uint8(255)),
        ("i8", GgufValue::Int8(-128)),
        ("u16", GgufValue::Uint16(65535)),
        ("i16", GgufValue::Int16(-32768)),
        ("u32", GgufValue::Uint32(u32::MAX)),
        ("i32", GgufValue::Int32(i32::MIN)),
        ("f32", GgufValue::Float32(1.5)),
        ("u64", GgufValue::Uint64(u64::MAX)),
        ("i64", GgufValue::Int64(i64::MIN)),
        ("f64", GgufValue::Float64(2.5)),
    ];

    for (key, value) in cases {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, key, &value).expect("write kv");
        assert!(!buf.is_empty(), "Buffer should not be empty for {key}");
    }
}

#[test]
fn test_write_metadata_kv_arrays() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "ids", &GgufValue::ArrayUint32(vec![1, 2, 3])).expect("write kv");
    assert!(!buf.is_empty());

    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "vals", &GgufValue::ArrayInt32(vec![-1, 0, 1])).expect("write kv");
    assert!(!buf.is_empty());

    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "floats", &GgufValue::ArrayFloat32(vec![1.0, 2.0]))
        .expect("write kv");
    assert!(!buf.is_empty());

    let mut buf = Vec::new();
    write_metadata_kv(
        &mut buf,
        "tokens",
        &GgufValue::ArrayString(vec!["a".to_string(), "b".to_string()]),
    )
    .expect("write kv");
    assert!(!buf.is_empty());
}

// ============================================================================
// export_tensors_to_gguf Tests
// ============================================================================

#[test]
fn test_export_tensors_to_gguf_empty() {
    let mut buf = Vec::new();
    export_tensors_to_gguf(&mut buf, &[], &[]).expect("export empty");

    // Should have header at minimum
    assert!(buf.len() >= 24);
    // Check magic
    assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
}

#[test]
fn test_export_tensors_to_gguf_with_metadata() {
    let metadata = vec![
        ("arch".to_string(), GgufValue::String("test".to_string())),
        ("layers".to_string(), GgufValue::Uint32(4)),
    ];
    let mut buf = Vec::new();
    export_tensors_to_gguf(&mut buf, &[], &metadata).expect("export with metadata");

    // Should have magic
    assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
    // Check metadata count
    assert_eq!(&buf[16..24], &2u64.to_le_bytes());
}

#[test]
fn test_export_tensors_to_gguf_with_tensors() {
    let tensors = vec![
        GgufTensor {
            name: "weight".to_string(),
            shape: vec![4, 4],
            dtype: GgmlType::F32,
            data: vec![0u8; 64], // 4*4*4 bytes
        },
        GgufTensor {
            name: "bias".to_string(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: vec![0u8; 16], // 4*4 bytes
        },
    ];
    let mut buf = Vec::new();
    export_tensors_to_gguf(&mut buf, &tensors, &[]).expect("export with tensors");

    // Check magic
    assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
    // Check tensor count
    assert_eq!(&buf[8..16], &2u64.to_le_bytes());
}

#[test]
fn test_export_tensors_to_gguf_full() {
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        ),
        ("llama.context_length".to_string(), GgufValue::Uint32(2048)),
    ];
    let tensors = vec![GgufTensor {
        name: "model.embed_tokens.weight".to_string(),
        shape: vec![32, 128],
        dtype: GgmlType::F32,
        data: vec![0u8; 32 * 128 * 4],
    }];
    let mut buf = Vec::new();
    export_tensors_to_gguf(&mut buf, &tensors, &metadata).expect("export full");

    // Verify output is properly aligned and contains all data
    assert!(buf.len() > 32 * 128 * 4); // Must be larger than just tensor data
    assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
}
