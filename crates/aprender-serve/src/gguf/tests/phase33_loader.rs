
#[test]
fn test_phase33_loader_get_tensor_f32_basic() {
    // Build GGUF with one F32 tensor
    let mut data = build_gguf_header(1, 0);
    // Tensor: 4 elements of F32
    data.extend(build_tensor_info("test", &[4], 0, 0)); // F32=0

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add tensor data: 4 f32 values
    let values = [1.0f32, 2.0, 3.0, 4.0];
    for v in &values {
        data.extend_from_slice(&v.to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");

    assert_eq!(tensor.len(), 4);
    assert!((tensor[0] - 1.0).abs() < 0.001);
    assert!((tensor[1] - 2.0).abs() < 0.001);
    assert!((tensor[2] - 3.0).abs() < 0.001);
    assert!((tensor[3] - 4.0).abs() < 0.001);
}

#[test]
fn test_phase33_loader_get_tensor_f32_out_of_bounds() {
    // Build GGUF claiming tensor data that doesn't exist
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[1000000], 0, 0)); // 1M elements
    // No actual tensor data - should fail

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_err());
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_phase33_loader_empty_string_key() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("", 4, &42u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.metadata.contains_key(""));
}

#[test]
fn test_phase33_loader_empty_tensor_name() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("", &[64], 0, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].name, "");
}

#[test]
fn test_phase33_loader_unicode_metadata_key() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata(
        "test_\u{1F600}",
        4,
        &42u32.to_le_bytes(),
    )); // emoji

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.metadata.contains_key("test_\u{1F600}"));
}

#[test]
fn test_phase33_loader_large_tensor_count() {
    // Just header with large tensor count (won't have data)
    let mut data = build_gguf_header(100, 0);
    // Add 100 tensor infos
    for i in 0..100 {
        data.extend(build_tensor_info(
            &format!("t{i}"),
            &[32],
            0,
            i as u64 * 128,
        ));
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors.len(), 100);
}

#[test]
fn test_phase33_loader_large_metadata_count() {
    let mut data = build_gguf_header(0, 50);
    for i in 0..50 {
        data.extend(build_gguf_metadata(
            &format!("key_{i}"),
            4,
            &(i as u32).to_le_bytes(),
        ));
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.metadata.len(), 50);
}

#[test]
fn test_phase33_loader_mixed_content() {
    // Full GGUF with metadata and tensors
    let mut data = build_gguf_header(2, 3);

    // Metadata
    data.extend(build_gguf_metadata("arch", 8, &build_gguf_string("llama")));
    data.extend(build_gguf_metadata("hidden", 4, &128u32.to_le_bytes()));
    data.extend(build_gguf_metadata("layers", 4, &2u32.to_le_bytes()));

    // Tensors
    data.extend(build_tensor_info("embed", &[100, 128], 0, 0));
    data.extend(build_tensor_info("norm", &[128], 0, 51200));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.metadata.len(), 3);
    assert_eq!(model.tensors.len(), 2);

    if let Some(GGUFValue::String(arch)) = model.metadata.get("arch") {
        assert_eq!(arch, "llama");
    }
}

// =============================================================================
// Metadata Accessors Tests
// =============================================================================

#[test]
fn test_phase33_loader_architecture() {
    let mut data = build_gguf_header(0, 1);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.architecture(), Some("llama"));
}

#[test]
fn test_phase33_loader_architecture_missing() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.architecture(), None);
}

#[test]
fn test_phase33_loader_embedding_dim() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata(
        "llama.embedding_length",
        4,
        &256u32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.embedding_dim(), Some(256));
}

#[test]
fn test_phase33_loader_num_layers() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("qwen2");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata(
        "qwen2.block_count",
        4,
        &24u32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_layers(), Some(24));
}

#[test]
fn test_phase33_loader_num_heads() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata(
        "llama.attention.head_count",
        4,
        &32u32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_heads(), Some(32));
}

#[test]
fn test_phase33_loader_context_length() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata(
        "llama.context_length",
        4,
        &4096u32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.context_length(), Some(4096));
}

#[test]
fn test_phase33_loader_num_kv_heads() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata(
        "llama.attention.head_count_kv",
        4,
        &8u32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_kv_heads(), Some(8));
}

#[test]
fn test_phase33_loader_rope_freq_base() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata(
        "llama.rope.freq_base",
        6,
        &10000.0f32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let rope = model.rope_freq_base().expect("Should have rope");
    assert!((rope - 10000.0).abs() < 0.1);
}

#[test]
fn test_phase33_loader_rms_epsilon() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata(
        "llama.attention.layer_norm_rms_epsilon",
        6,
        &1e-5f32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let eps = model.rms_epsilon().expect("Should have epsilon");
    assert!((eps - 1e-5).abs() < 1e-7);
}

// =============================================================================
// Quantized Tensor Extraction Tests
// =============================================================================

#[test]
fn test_phase33_loader_get_tensor_q4_0() {
    use crate::gguf::GGUF_TYPE_Q4_0;

    // Q4_0: 18 bytes per 32 elements (2 f16 scale + 16 bytes quants)
    let mut data = build_gguf_header(1, 0);
    // 32 elements, Q4_0, offset 0
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q4_0, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q4_0 block: f16 scale + 16 bytes
    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    // 16 bytes of quants (each byte has 2 4-bit values)
    data.extend([0x11u8; 16]); // All 1s = small values

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q4_0 extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_q8_0() {
    use crate::gguf::GGUF_TYPE_Q8_0;

    // Q8_0: 34 bytes per 32 elements (2 f16 scale + 32 i8 quants)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q8_0, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q8_0 block: f16 scale + 32 i8 quants
    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend([0i8 as u8; 32]); // All zeros

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q8_0 extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_f16() {
    use crate::gguf::GGUF_TYPE_F16;

    // F16: 2 bytes per element
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[4], GGUF_TYPE_F16, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add F16 data: 4 values
    let values = [1.0f32, 2.0, 3.0, 4.0];
    for v in &values {
        let f16_val = half::f16::from_f32(*v);
        data.extend_from_slice(&f16_val.to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "F16 extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 4);
    assert!((tensor[0] - 1.0).abs() < 0.01);
}

#[test]
fn test_phase33_loader_get_tensor_q4_1() {
    use crate::gguf::GGUF_TYPE_Q4_1;

    // Q4_1: 20 bytes per 32 elements (2 scale + 2 min + 16 quants)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q4_1, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q4_1 block: f16 scale + f16 min + 16 bytes quants
    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0x00u8; 16]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q4_1 extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_q5_0() {
    use crate::gguf::GGUF_TYPE_Q5_0;

    // Q5_0: 22 bytes per 32 elements (2 scale + 4 high bits + 16 quants)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_0, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q5_0 block
    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend([0u8; 4]); // high bits
    data.extend([0x00u8; 16]); // quants

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q5_0 extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_q5_1() {
    use crate::gguf::GGUF_TYPE_Q5_1;

    // Q5_1: 24 bytes per 32 elements (2 scale + 2 min + 4 high bits + 16 quants)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_1, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q5_1 block
    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0u8; 4]); // high bits
    data.extend([0x00u8; 16]); // quants

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q5_1 extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 32);
}
