
#[test]
fn test_f16_to_f32_two() {
    // F16 2.0 = 0x4000
    let result = f16_to_f32(0x4000);
    assert!((result - 2.0).abs() < 0.001);
}

#[test]
fn test_gguf_tensor_clone() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![10],
        dtype: GgmlType::F32,
        data: vec![1, 2, 3, 4],
    };
    let cloned = tensor.clone();
    assert_eq!(tensor.name, cloned.name);
    assert_eq!(tensor.data, cloned.data);
}

#[test]
fn test_gguf_tensor_info_clone() {
    let info = GgufTensorInfo {
        name: "test".to_string(),
        n_dims: 2,
        dims: vec![10, 20],
        dtype: GgmlType::F32,
        offset: 100,
    };
    let cloned = info.clone();
    assert_eq!(info.name, cloned.name);
    assert_eq!(info.offset, cloned.offset);
}

#[test]
fn test_gguf_value_clone_string() {
    let v = GgufValue::String("hello".to_string());
    let cloned = v.clone();
    if let (GgufValue::String(s1), GgufValue::String(s2)) = (&v, &cloned) {
        assert_eq!(s1, s2);
    } else {
        panic!("Expected String values");
    }
}

#[test]
fn test_gguf_header_clone_full() {
    let header = GgufHeader {
        version: 3,
        tensor_count: 5,
        metadata_kv_count: 10,
    };
    let cloned = header.clone();
    assert_eq!(header.version, cloned.version);
}

#[test]
fn test_padding_edge_cases() {
    assert_eq!(padding_for_alignment(0, 1), 0);
    assert_eq!(padding_for_alignment(1, 1), 0);
    assert_eq!(padding_for_alignment(100, 1), 0);
    assert_eq!(padding_for_alignment(31, 32), 1);
    assert_eq!(padding_for_alignment(63, 64), 1);
}

#[test]
fn test_export_empty_tensors() {
    let mut buffer = Vec::new();
    let tensors: Vec<GgufTensor> = vec![];
    let metadata: Vec<(String, GgufValue)> = vec![];
    export_tensors_to_gguf(&mut buffer, &tensors, &metadata).expect("export");
    assert!(!buffer.is_empty());
    assert_eq!(&buffer[0..4], b"GGUF");
}

#[test]
fn test_export_single_tensor() {
    let mut buffer = Vec::new();
    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data: vec![0u8; 64], // 16 floats * 4 bytes
    };
    export_tensors_to_gguf(&mut buffer, &[tensor], &[]).expect("export");
    assert!(!buffer.is_empty());
}

#[test]
fn test_export_with_metadata() {
    let mut buffer = Vec::new();
    let metadata = vec![
        (
            "general.name".to_string(),
            GgufValue::String("test".to_string()),
        ),
        ("general.version".to_string(), GgufValue::Uint32(1)),
    ];
    export_tensors_to_gguf(&mut buffer, &[], &metadata).expect("export");
    assert!(!buffer.is_empty());
}

#[test]
fn test_gguf_tensor_info_write_1d() {
    let info = GgufTensorInfo {
        name: "t".to_string(),
        n_dims: 1,
        dims: vec![100],
        dtype: GgmlType::F32,
        offset: 0,
    };
    let mut buffer = Vec::new();
    info.write_to(&mut buffer).expect("write");
    assert!(!buffer.is_empty());
}

#[test]
fn test_gguf_tensor_info_write_4d() {
    let info = GgufTensorInfo {
        name: "tensor".to_string(),
        n_dims: 4,
        dims: vec![10, 20, 30, 40],
        dtype: GgmlType::F16,
        offset: 1024,
    };
    let mut buffer = Vec::new();
    info.write_to(&mut buffer).expect("write");
    assert!(!buffer.is_empty());
}

#[test]
fn test_constants() {
    assert_eq!(GGUF_VERSION, 3);
    assert_eq!(GGUF_DEFAULT_ALIGNMENT, 32);
}

#[test]
fn test_ggml_type_i8() {
    let t = GgmlType::I8;
    assert_eq!(t as u32, 24);
}

#[test]
fn test_ggml_type_i16() {
    let t = GgmlType::I16;
    assert_eq!(t as u32, 25);
}

#[test]
fn test_ggml_type_i32() {
    let t = GgmlType::I32;
    assert_eq!(t as u32, 26);
}

#[test]
fn test_ggml_type_i64() {
    let t = GgmlType::I64;
    assert_eq!(t as u32, 27);
}

#[test]
fn test_ggml_type_f64() {
    let t = GgmlType::F64;
    assert_eq!(t as u32, 28);
}

// ========================================================================
// Dequantize Function Tests (ROSETTA-ML-001)
// ========================================================================

#[test]
fn test_dequantize_q4_0_basic() {
    // Q4_0 block: 2 bytes d (f16) + 16 bytes qs = 18 bytes for 32 elements
    // Create a minimal valid block
    let mut data = Vec::new();
    // d = 1.0 in f16 (0x3C00)
    data.extend_from_slice(&0x3C00u16.to_le_bytes());
    // 16 quantized bytes (4-bit pairs) = 32 values
    data.extend_from_slice(&[0x00u8; 16]); // All zeros -> output should be around -8*d

    let result = dequantize_q4_0(&data, 0, 32).unwrap();
    assert_eq!(result.len(), 32);
    // Q4_0 subtracts 8 from each 4-bit value, so 0 becomes -8
    // All values should be -8 * d = -8.0
    for &v in &result {
        assert!((v + 8.0).abs() < 0.1);
    }
}

#[test]
fn test_dequantize_q4_0_out_of_bounds() {
    let data = vec![0u8; 10]; // Too small for one block
    let result = dequantize_q4_0(&data, 0, 32);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_0_basic() {
    // Q8_0 block: 2 bytes d (f16) + 32 bytes qs = 34 bytes for 32 elements
    let mut data = Vec::new();
    // d = 1.0 in f16 (0x3C00)
    data.extend_from_slice(&0x3C00u16.to_le_bytes());
    // 32 quantized bytes (signed i8) - all zeros
    data.extend_from_slice(&[0i8 as u8; 32]);

    let result = dequantize_q8_0(&data, 0, 32).unwrap();
    assert_eq!(result.len(), 32);
    // All values should be 0 * d = 0.0
    for &v in &result {
        assert!((v - 0.0).abs() < 0.001);
    }
}

#[test]
fn test_dequantize_q8_0_with_values() {
    // Q8_0 block: 2 bytes d (f16) + 32 bytes qs = 34 bytes for 32 elements
    let mut data = Vec::new();
    // d = 0.5 in f16 (0x3800)
    data.extend_from_slice(&0x3800u16.to_le_bytes());
    // First value = 10, rest = 0
    let mut qs = [0u8; 32];
    qs[0] = 10;
    data.extend_from_slice(&qs);

    let result = dequantize_q8_0(&data, 0, 32).unwrap();
    assert_eq!(result.len(), 32);
    // First value should be 10 * 0.5 = 5.0
    assert!((result[0] - 5.0).abs() < 0.01);
}

#[test]
fn test_dequantize_q8_0_out_of_bounds() {
    let data = vec![0u8; 20]; // Too small for one block (needs 34)
    let result = dequantize_q8_0(&data, 0, 32);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_0_basic() {
    // Q5_0 block size: 2 (d) + 4 (qh) + 16 (ql) = 22 bytes for 32 elements
    let mut data = Vec::new();
    // d = 1.0 in f16 (0x3C00)
    data.extend_from_slice(&0x3C00u16.to_le_bytes());
    // qh (high bits): 4 bytes
    data.extend_from_slice(&[0u8; 4]);
    // ql (low 4 bits): 16 bytes
    data.extend_from_slice(&[0u8; 16]);

    let result = dequantize_q5_0(&data, 0, 32).unwrap();
    assert_eq!(result.len(), 32);
    // Values are finite
    for &v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn test_dequantize_q5_0_out_of_bounds() {
    let data = vec![0u8; 10]; // Too small (needs 22)
    let result = dequantize_q5_0(&data, 0, 32);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_basic() {
    // Q5_1 block size: 2 (d) + 2 (m) + 4 (qh) + 16 (ql) = 24 bytes for 32 elements
    let mut data = Vec::new();
    // d = 1.0 in f16 (0x3C00)
    data.extend_from_slice(&0x3C00u16.to_le_bytes());
    // m = 0.0 in f16 (0x0000)
    data.extend_from_slice(&0x0000u16.to_le_bytes());
    // qh (high bits): 4 bytes
    data.extend_from_slice(&[0u8; 4]);
    // ql (low 4 bits): 16 bytes
    data.extend_from_slice(&[0u8; 16]);

    let result = dequantize_q5_1(&data, 0, 32).unwrap();
    assert_eq!(result.len(), 32);
    // Values are finite
    for &v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn test_dequantize_q5_1_out_of_bounds() {
    let data = vec![0u8; 10]; // Too small (needs 24)
    let result = dequantize_q5_1(&data, 0, 32);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_1_basic() {
    // Q4_1 block: 2 (d) + 2 (m) + 16 (qs) = 20 bytes for 32 elements
    let mut data = Vec::new();
    // d = 1.0 in f16 (0x3C00)
    data.extend_from_slice(&0x3C00u16.to_le_bytes());
    // m = 0.0 in f16 (0x0000)
    data.extend_from_slice(&0x0000u16.to_le_bytes());
    // 16 quantized bytes
    data.extend_from_slice(&[0u8; 16]);

    let result = dequantize_q4_1(&data, 0, 32).unwrap();
    assert_eq!(result.len(), 32);
    // Values are finite
    for &v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn test_dequantize_q4_1_out_of_bounds() {
    let data = vec![0u8; 10]; // Too small (needs 20)
    let result = dequantize_q4_1(&data, 0, 32);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_k_basic() {
    // Q4_K super-block: 144 bytes for 256 elements
    // Structure: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144
    let data = vec![0u8; 144];

    let result = dequantize_q4_k(&data, 0, 256).unwrap();
    assert_eq!(result.len(), 256);
    // All zeros in data -> all zeros in output
    for &v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn test_dequantize_q4_k_out_of_bounds() {
    let data = vec![0u8; 100]; // Too small (needs 144)
    let result = dequantize_q4_k(&data, 0, 256);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_k_basic() {
    // Q5_K super-block: 176 bytes for 256 elements
    let data = vec![0u8; 176];

    let result = dequantize_q5_k(&data, 0, 256).unwrap();
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn test_dequantize_q5_k_out_of_bounds() {
    let data = vec![0u8; 100]; // Too small (needs 176)
    let result = dequantize_q5_k(&data, 0, 256);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q6_k_basic() {
    // Q6_K super-block: 210 bytes for 256 elements
    let data = vec![0u8; 210];

    let result = dequantize_q6_k(&data, 0, 256).unwrap();
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn test_dequantize_q6_k_out_of_bounds() {
    let data = vec![0u8; 100]; // Too small (needs 210)
    let result = dequantize_q6_k(&data, 0, 256);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q2_k_basic() {
    // Q2_K super-block: 84 bytes for 256 elements
    let data = vec![0u8; 84];

    let result = dequantize_q2_k(&data, 0, 256).unwrap();
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn test_dequantize_q2_k_out_of_bounds() {
    let data = vec![0u8; 50]; // Too small (needs 84)
    let result = dequantize_q2_k(&data, 0, 256);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q3_k_basic() {
    // Q3_K super-block: 110 bytes for 256 elements
    let data = vec![0u8; 110];

    let result = dequantize_q3_k(&data, 0, 256).unwrap();
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn test_dequantize_q3_k_out_of_bounds() {
    let data = vec![0u8; 50]; // Too small (needs 110)
    let result = dequantize_q3_k(&data, 0, 256);
    assert!(result.is_err());
}
