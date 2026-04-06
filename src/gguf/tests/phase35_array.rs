
#[test]
fn test_phase35_array_of_bools() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&7u32.to_le_bytes()); // bool type
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.push(1); // true
    array_bytes.push(0); // false
    array_bytes.push(1); // true
    data.extend(build_gguf_metadata("test_bools", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test_bools") {
        assert_eq!(arr.len(), 3);
        assert!(matches!(arr[0], GGUFValue::Bool(true)));
        assert!(matches!(arr[1], GGUFValue::Bool(false)));
        assert!(matches!(arr[2], GGUFValue::Bool(true)));
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_phase35_array_of_i64() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&11u32.to_le_bytes()); // i64 type
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    array_bytes.extend_from_slice(&i64::MIN.to_le_bytes());
    array_bytes.extend_from_slice(&i64::MAX.to_le_bytes());
    data.extend(build_gguf_metadata("test_i64s", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test_i64s") {
        assert_eq!(arr.len(), 2);
        assert!(matches!(arr[0], GGUFValue::Int64(v) if v == i64::MIN));
        assert!(matches!(arr[1], GGUFValue::Int64(v) if v == i64::MAX));
    } else {
        panic!("Expected Array");
    }
}

// =============================================================================
// Unicode and Special Characters
// =============================================================================

#[test]
fn test_phase35_unicode_tensor_names() {
    let mut data = build_gguf_header(1, 0);
    // Unicode tensor name with emojis and special chars
    data.extend(build_tensor_info("tensor_\u{1F4A1}_light", &[32], 0, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);
    data.extend([0u8; 128]); // tensor data

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].name, "tensor_\u{1F4A1}_light");
}

#[test]
fn test_phase35_unicode_metadata_values() {
    let mut data = build_gguf_header(0, 1);
    let value = build_gguf_string("\u{4E2D}\u{6587}"); // Chinese characters
    data.extend(build_gguf_metadata("chinese_text", 8, &value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::String(s)) = model.metadata.get("chinese_text") {
        assert_eq!(s, "\u{4E2D}\u{6587}");
    }
}

#[test]
fn test_phase35_long_string_metadata() {
    let mut data = build_gguf_header(0, 1);
    // 1KB string
    let long_value = "x".repeat(1024);
    let value = build_gguf_string(&long_value);
    data.extend(build_gguf_metadata("long_string", 8, &value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::String(s)) = model.metadata.get("long_string") {
        assert_eq!(s.len(), 1024);
    }
}

// =============================================================================
// Multi-Tensor and Complex Models
// =============================================================================

#[test]
fn test_phase35_model_with_many_tensors() {
    let mut data = build_gguf_header(50, 0);

    // Add 50 tensor infos
    for i in 0..50 {
        data.extend(build_tensor_info(
            &format!("tensor_{:02}", i),
            &[32],
            0,
            (i * 128) as u64,
        ));
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors.len(), 50);
    assert_eq!(model.tensors[0].name, "tensor_00");
    assert_eq!(model.tensors[49].name, "tensor_49");
}

#[test]
fn test_phase35_model_with_many_metadata() {
    let mut data = build_gguf_header(0, 100);

    // Add 100 metadata entries
    for i in 0..100 {
        data.extend(build_gguf_metadata(
            &format!("key_{:03}", i),
            4,
            &(i as u32).to_le_bytes(),
        ));
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.metadata.len(), 100);
    assert!(matches!(
        model.metadata.get("key_000"),
        Some(GGUFValue::UInt32(0))
    ));
    assert!(matches!(
        model.metadata.get("key_099"),
        Some(GGUFValue::UInt32(99))
    ));
}

#[test]
fn test_phase35_full_model_config() {
    let mut data = build_gguf_header(2, 10);

    // Full set of metadata
    let arch = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch));
    data.extend(build_gguf_metadata(
        "llama.embedding_length",
        4,
        &256u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.block_count",
        4,
        &4u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.attention.head_count",
        4,
        &8u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.attention.head_count_kv",
        4,
        &2u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.context_length",
        4,
        &2048u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.rope.freq_base",
        6,
        &10000.0f32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.attention.layer_norm_rms_epsilon",
        6,
        &1e-5f32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.bos_token_id",
        4,
        &1u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.eos_token_id",
        4,
        &2u32.to_le_bytes(),
    ));

    // Two tensors
    data.extend(build_tensor_info("token_embd.weight", &[100, 256], 0, 0));
    data.extend(build_tensor_info("output_norm.weight", &[256], 0, 102400));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");

    // Verify all accessors
    assert_eq!(model.architecture(), Some("llama"));
    assert_eq!(model.embedding_dim(), Some(256));
    assert_eq!(model.num_layers(), Some(4));
    assert_eq!(model.num_heads(), Some(8));
    assert_eq!(model.num_kv_heads(), Some(2));
    assert_eq!(model.context_length(), Some(2048));
    assert!((model.rope_freq_base().expect("test value should be present") - 10000.0).abs() < 1.0);
    assert!((model.rms_epsilon().expect("test value should be present") - 1e-5).abs() < 1e-7);
    assert_eq!(model.bos_token_id(), Some(1));
    assert_eq!(model.eos_token_id(), Some(2));
    assert_eq!(model.rope_type(), Some(0)); // NORM for llama
}

// =============================================================================
// GGUFModel Clone and Debug
// =============================================================================

#[test]
fn test_phase35_gguf_model_debug() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("GGUFModel"));
    assert!(debug_str.contains("header"));
}

#[test]
fn test_phase35_gguf_model_clone() {
    let mut data = build_gguf_header(1, 1);
    let arch = build_gguf_string("test");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch));
    data.extend(build_tensor_info("test", &[32], 0, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let cloned = model.clone();

    assert_eq!(model.header, cloned.header);
    assert_eq!(model.metadata.len(), cloned.metadata.len());
    assert_eq!(model.tensors.len(), cloned.tensors.len());
}
