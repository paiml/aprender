
#[test]
fn test_loader_tensor_info_large_dimension() {
    let mut data = build_header(1, 0);
    // Single very large dimension
    data.extend(build_tensor_info("large", &[1_000_000_000], GGUF_TYPE_F32, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse large dim");
    assert_eq!(model.tensors[0].dims[0], 1_000_000_000);
}

#[test]
fn test_loader_tensor_info_multiple_qtypes() {
    let mut data = build_header(5, 0);

    data.extend(build_tensor_info("f32", &[32], GGUF_TYPE_F32, 0));
    data.extend(build_tensor_info("f16", &[32], GGUF_TYPE_F16, 128));
    data.extend(build_tensor_info("q4_0", &[32], GGUF_TYPE_Q4_0, 192));
    data.extend(build_tensor_info("q8_0", &[32], GGUF_TYPE_Q8_0, 210));
    data.extend(build_tensor_info("q4_k", &[256], GGUF_TYPE_Q4_K, 244));

    let model = GGUFModel::from_bytes(&data).expect("Should parse multiple qtypes");

    assert_eq!(model.tensors.len(), 5);
    assert_eq!(model.tensors[0].qtype, GGUF_TYPE_F32);
    assert_eq!(model.tensors[1].qtype, GGUF_TYPE_F16);
    assert_eq!(model.tensors[2].qtype, GGUF_TYPE_Q4_0);
    assert_eq!(model.tensors[3].qtype, GGUF_TYPE_Q8_0);
    assert_eq!(model.tensors[4].qtype, GGUF_TYPE_Q4_K);
}

// =============================================================================
// Alignment Calculation Tests
// =============================================================================

#[test]
fn test_loader_alignment_at_boundary() {
    // Header + metadata that lands exactly on 32-byte boundary
    let mut data = build_header(0, 0);
    assert_eq!(data.len(), 24);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // tensor_data_start should be aligned to 32
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
    assert_eq!(model.tensor_data_start, 32); // 24 rounded up to 32
}

#[test]
fn test_loader_alignment_with_metadata() {
    let mut data = build_header(0, 1);
    // Add metadata that changes alignment requirement
    data.extend(build_metadata("test", 4, &42u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
}

#[test]
fn test_loader_alignment_with_tensors() {
    let mut data = build_header(2, 0);
    data.extend(build_tensor_info("t1", &[64], GGUF_TYPE_F32, 0));
    data.extend(build_tensor_info("t2", &[64], GGUF_TYPE_F32, 256));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
}

// =============================================================================
// get_tensor_f32 - Data Range Validation
// =============================================================================

#[test]
fn test_loader_get_tensor_f32_exact_boundary() {
    // Tensor data ends exactly at file end
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[4], GGUF_TYPE_F32, 0));
    align_to_boundary(&mut data);

    let tensor_start = data.len();
    // 4 f32 values = 16 bytes
    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 4);
}

#[test]
fn test_loader_get_tensor_f32_one_byte_short() {
    // Tensor data is 1 byte short
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[4], GGUF_TYPE_F32, 0));
    align_to_boundary(&mut data);

    // 4 f32 = 16 bytes, but only provide 15
    data.extend(&[0u8; 15]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_err());
}

#[test]
fn test_loader_get_tensor_f32_2d_matrix() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("matrix", &[3, 4], GGUF_TYPE_F32, 0));
    align_to_boundary(&mut data);

    // 3 * 4 = 12 f32 values = 48 bytes
    for i in 0..12 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("matrix", &data).expect("Should extract");
    assert_eq!(tensor.len(), 12);
    assert!((tensor[5] - 5.0).abs() < f32::EPSILON);
}

#[test]
fn test_loader_get_tensor_q4_0_partial_block() {
    // Q4_0 with elements not divisible by block size (32)
    // Tests the truncation after dequantization
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[48], GGUF_TYPE_Q4_0, 0)); // 48 = 1.5 blocks
    align_to_boundary(&mut data);

    // Need 2 blocks (64 elements worth) for 48 elements
    // Each Q4_0 block: 2 bytes f16 scale + 16 bytes quants = 18 bytes
    for _ in 0..2 {
        let scale = half::f16::from_f32(1.0);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0x00u8; 16]);
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 48); // Truncated to exact size
}

#[test]
fn test_loader_get_tensor_q8_0_partial_block() {
    // Q8_0 with elements not divisible by block size (32)
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[50], GGUF_TYPE_Q8_0, 0)); // 50 = 1.56 blocks
    align_to_boundary(&mut data);

    // Need 2 blocks for 50 elements
    // Each Q8_0 block: 2 bytes f16 scale + 32 bytes quants = 34 bytes
    for _ in 0..2 {
        let scale = half::f16::from_f32(0.5);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0i8 as u8; 32]);
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 50);
}

#[test]
fn test_loader_get_tensor_f16_values() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[8], GGUF_TYPE_F16, 0));
    align_to_boundary(&mut data);

    // 8 f16 values
    let values = [0.0f32, 0.5, 1.0, 1.5, 2.0, -1.0, -0.5, 0.25];
    for v in &values {
        let f16_val = half::f16::from_f32(*v);
        data.extend_from_slice(&f16_val.to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 8);
    assert!((tensor[2] - 1.0).abs() < 0.01);
    assert!((tensor[5] - (-1.0)).abs() < 0.01);
}

#[test]
fn test_loader_get_tensor_q4_1_block() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q4_1, 0));
    align_to_boundary(&mut data);

    // Q4_1: 2 f16 scale + 2 f16 min + 16 bytes quants = 20 bytes
    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0x00u8; 16]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_loader_get_tensor_q5_0_block() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_0, 0));
    align_to_boundary(&mut data);

    // Q5_0: 2 f16 scale + 4 high bits + 16 bytes quants = 22 bytes
    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend([0u8; 4]); // high bits
    data.extend([0x00u8; 16]); // quants

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_loader_get_tensor_q5_1_block() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_1, 0));
    align_to_boundary(&mut data);

    // Q5_1: 2 f16 scale + 2 f16 min + 4 high bits + 16 bytes quants = 24 bytes
    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0u8; 4]); // high bits
    data.extend([0x00u8; 16]); // quants

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_loader_get_tensor_k_quant_partial() {
    // K-quant with partial super-block (elements not divisible by 256)
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[300], GGUF_TYPE_Q4_K, 0)); // 300 = 1.17 super-blocks
    align_to_boundary(&mut data);

    // Need 2 super-blocks (512 elements worth)
    // Q4_K super-block: 144 bytes
    data.extend([0u8; 288]); // 2 * 144

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 300); // Truncated
}

#[test]
fn test_loader_get_tensor_offset_nonzero() {
    // Tensor at non-zero offset
    let mut data = build_header(1, 0);
    // Offset 64 bytes into tensor data section
    data.extend(build_tensor_info("test", &[4], GGUF_TYPE_F32, 64));
    align_to_boundary(&mut data);

    // Padding before tensor data
    data.extend([0xFFu8; 64]);
    // Actual tensor data
    for i in 0..4 {
        data.extend_from_slice(&((i + 100) as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 4);
    assert!((tensor[0] - 100.0).abs() < f32::EPSILON);
}

// =============================================================================
// Error Message Quality Tests
// =============================================================================

#[test]
fn test_loader_error_tensor_not_found_message() {
    let data = build_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");

    let result = model.get_tensor_f32("nonexistent_tensor", &data);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("nonexistent_tensor") || err_str.contains("not found"),
        "Error should mention tensor name: {}",
        err_str
    );
}

#[test]
fn test_loader_error_unsupported_qtype_message() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[32], 200, 0)); // qtype 200 doesn't exist
    align_to_boundary(&mut data);
    data.extend([0u8; 128]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("200") || err_str.contains("Unsupported"),
        "Error should mention unsupported type: {}",
        err_str
    );
}

#[test]
fn test_loader_error_data_range_message() {
    let mut data = build_header(1, 0);
    // Claim huge tensor that exceeds file size
    data.extend(build_tensor_info("huge", &[1_000_000], GGUF_TYPE_F32, 0));
    // No actual tensor data

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("huge", &data);

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("exceeds") || err_str.contains("range"),
        "Error should mention data range issue: {}",
        err_str
    );
}

// =============================================================================
// Edge Cases in String Parsing
// =============================================================================

#[test]
fn test_loader_string_with_null_bytes() {
    let mut data = build_header(0, 1);
    // String containing null bytes
    let s = "hello\0world";
    let str_bytes = build_string(s);
    data.extend(build_metadata("null_str", 8, &str_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::String(val)) = model.metadata.get("null_str") {
        assert_eq!(val, "hello\0world");
    }
}

#[test]
fn test_loader_string_utf8_multibyte() {
    let mut data = build_header(0, 1);
    // Various UTF-8 multibyte sequences
    let s = "\u{00E9}\u{4E2D}\u{1F600}"; // e-acute, Chinese char, emoji
    let str_bytes = build_string(s);
    data.extend(build_metadata("utf8", 8, &str_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::String(val)) = model.metadata.get("utf8") {
        assert_eq!(val, s);
    }
}

#[test]
fn test_loader_tensor_name_with_dots() {
    let mut data = build_header(1, 0);
    // Typical GGUF tensor name with multiple dots
    data.extend(build_tensor_info(
        "blk.15.attn_q.weight",
        &[4096, 4096],
        GGUF_TYPE_Q4_K,
        0,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].name, "blk.15.attn_q.weight");
}
