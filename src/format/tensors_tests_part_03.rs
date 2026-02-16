use super::*;

// ====================================================================
// Coverage: compute_tensor_stats edge cases
// ====================================================================

#[test]
fn test_compute_stats_all_nan() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![3],
        dtype: "F32".to_string(),
        size_bytes: 12,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[f32::NAN, f32::NAN, f32::NAN]);
    assert_eq!(info.nan_count, Some(3));
    assert!(info.mean.is_none(), "no valid values → no mean");
}

#[test]
fn test_compute_stats_all_inf() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![2],
        dtype: "F32".to_string(),
        size_bytes: 8,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(info.inf_count, Some(2));
    assert!(info.mean.is_none());
}

#[test]
fn test_compute_stats_single_value() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![1],
        dtype: "F32".to_string(),
        size_bytes: 4,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[42.0]);
    assert_eq!(info.mean, Some(42.0));
    assert_eq!(info.std, Some(0.0));
    assert_eq!(info.min, Some(42.0));
    assert_eq!(info.max, Some(42.0));
}

#[test]
fn test_compute_stats_identical_values() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[5.0, 5.0, 5.0, 5.0]);
    assert_eq!(info.mean, Some(5.0));
    assert_eq!(info.std, Some(0.0));
}

// ====================================================================
// Coverage: list_tensors_safetensors error paths
// ====================================================================

#[test]
fn test_safetensors_too_small() {
    let result = list_tensors_from_bytes(&[0u8; 4], TensorListOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_safetensors_truncated_header() {
    // header_len says 1000, but only 20 bytes total
    let mut data = vec![0u8; 20];
    data[0..8].copy_from_slice(&1000u64.to_le_bytes());
    // Make it look like SafeTensors (bytes 8-9 = '{"')
    data[8] = b'{';
    data[9] = b'"';
    let result = super::list_tensors_safetensors(&data, TensorListOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("truncated"));
}

#[test]
fn test_safetensors_invalid_json() {
    let header = b"not json at all";
    let header_len = header.len() as u64;
    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header);
    let result = super::list_tensors_safetensors(&data, TensorListOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("JSON"));
}

#[test]
fn test_safetensors_json_not_object() {
    let header = b"[1, 2, 3]";
    let header_len = header.len() as u64;
    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header);
    let result = super::list_tensors_safetensors(&data, TensorListOptions::default());
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not a JSON object"));
}

#[test]
fn test_safetensors_with_stats() {
    // Build a minimal SafeTensors with one F32 tensor and compute stats
    let tensor_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let header = format!(
        r#"{{"test_tensor":{{"dtype":"F32","shape":[4],"data_offsets":[0,{}]}}}}"#,
        tensor_data.len()
    );
    let header_bytes = header.as_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(header_bytes);
    data.extend_from_slice(&tensor_data);

    let opts = TensorListOptions {
        compute_stats: true,
        ..TensorListOptions::default()
    };
    let result = super::list_tensors_safetensors(&data, opts).expect("parse ok");
    assert_eq!(result.tensor_count, 1);
    let t = &result.tensors[0];
    assert!(t.mean.is_some());
    assert!((t.mean.expect("test value") - 2.5).abs() < 0.01);
}

#[test]
fn test_safetensors_limit() {
    // Two tensors, limit=1
    let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&[0u8; 8]);

    let opts = TensorListOptions {
        limit: 1,
        ..TensorListOptions::default()
    };
    let result = super::list_tensors_safetensors(&data, opts).expect("parse ok");
    // GH-195 FIX: tensor_count reflects true total, not limited count
    assert_eq!(result.tensor_count, 2);
    assert_eq!(result.tensors.len(), 1);
}

#[test]
fn test_safetensors_filter() {
    let header = r#"{"attn.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"mlp.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&[0u8; 16]);

    let opts = TensorListOptions {
        filter: Some("attn".to_string()),
        ..TensorListOptions::default()
    };
    let result = super::list_tensors_safetensors(&data, opts).expect("parse ok");
    assert_eq!(result.tensor_count, 1);
    assert_eq!(result.tensors[0].name, "attn.weight");
}

// ====================================================================
// Coverage: list_tensors_v1 basic path
// ====================================================================

#[test]
fn test_list_tensors_v1_too_small() {
    let data = vec![0u8; 16]; // less than HEADER_SIZE
    let result = super::list_tensors_v1(&data, TensorListOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_list_tensors_v1_metadata_truncated() {
    // Header says metadata is 1000 bytes but file is too small
    let mut data = vec![0u8; 64]; // HEADER_SIZE
                                  // offset 8-11: metadata_size = 1000
    data[8..12].copy_from_slice(&1000u32.to_le_bytes());
    let result = super::list_tensors_v1(&data, TensorListOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_list_tensors_v1_empty_metadata() {
    // Valid header with empty metadata (size=2, content="{}")
    let metadata = b"{}";
    let mut data = vec![0u8; 64];
    data[8..12].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
    data.extend_from_slice(metadata);
    let result = super::list_tensors_v1(&data, TensorListOptions::default());
    assert!(result.is_ok());
    assert_eq!(result.expect("test value").tensor_count, 0);
}

// ====================================================================
// Coverage: ggml_dtype_name all branches
// ====================================================================

#[test]
fn test_ggml_dtype_name_all() {
    assert_eq!(super::ggml_dtype_name(0), "F32");
    assert_eq!(super::ggml_dtype_name(1), "F16");
    assert_eq!(super::ggml_dtype_name(2), "Q4_0");
    assert_eq!(super::ggml_dtype_name(3), "Q4_1");
    assert_eq!(super::ggml_dtype_name(6), "Q5_0");
    assert_eq!(super::ggml_dtype_name(7), "Q5_1");
    assert_eq!(super::ggml_dtype_name(8), "Q8_0");
    assert_eq!(super::ggml_dtype_name(12), "Q4_K");
    assert_eq!(super::ggml_dtype_name(14), "Q6_K");
    assert_eq!(super::ggml_dtype_name(999), "unknown");
}

// ====================================================================
// Coverage: ggml_dtype_name exhaustive (all match arms)
// ====================================================================

#[test]
fn test_ggml_dtype_name_exhaustive_all_arms() {
    // Ensure every documented dtype code has its correct name
    assert_eq!(super::ggml_dtype_name(9), "Q8_1");
    assert_eq!(super::ggml_dtype_name(10), "Q2_K");
    assert_eq!(super::ggml_dtype_name(11), "Q3_K");
    assert_eq!(super::ggml_dtype_name(13), "Q5_K");
    assert_eq!(super::ggml_dtype_name(15), "Q8_K");
    assert_eq!(super::ggml_dtype_name(16), "IQ2_XXS");
    assert_eq!(super::ggml_dtype_name(17), "IQ2_XS");
    assert_eq!(super::ggml_dtype_name(18), "IQ3_XXS");
    assert_eq!(super::ggml_dtype_name(26), "BF16");
    // Newly added I-quant and integer types
    assert_eq!(super::ggml_dtype_name(19), "IQ1_S");
    assert_eq!(super::ggml_dtype_name(20), "IQ4_NL");
    assert_eq!(super::ggml_dtype_name(21), "IQ3_S");
    assert_eq!(super::ggml_dtype_name(22), "IQ2_S");
    assert_eq!(super::ggml_dtype_name(23), "IQ4_XS");
    assert_eq!(super::ggml_dtype_name(24), "I8");
    assert_eq!(super::ggml_dtype_name(25), "I16");
    assert_eq!(super::ggml_dtype_name(27), "I32");
    assert_eq!(super::ggml_dtype_name(28), "I64");
    assert_eq!(super::ggml_dtype_name(29), "F64");
    assert_eq!(super::ggml_dtype_name(30), "IQ1_M");
    // Codes with no mapping (gaps in GGML enum)
    assert_eq!(super::ggml_dtype_name(4), "unknown");
    assert_eq!(super::ggml_dtype_name(5), "unknown");
    assert_eq!(super::ggml_dtype_name(31), "unknown");
    assert_eq!(super::ggml_dtype_name(u32::MAX), "unknown");
}

// ====================================================================
// Coverage: ggml_dtype_element_size exhaustive (all match arms)
// ====================================================================

#[test]
fn test_ggml_dtype_element_size_exhaustive() {
    // F32
    assert!((super::ggml_dtype_element_size(0) - 4.0).abs() < 0.001);
    // F16
    assert!((super::ggml_dtype_element_size(1) - 2.0).abs() < 0.001);
    // Q4_0
    assert!((super::ggml_dtype_element_size(2) - 0.5625).abs() < 0.01);
    // Q4_1
    assert!((super::ggml_dtype_element_size(3) - 0.625).abs() < 0.01);
    // Q5_0
    assert!((super::ggml_dtype_element_size(6) - 0.6875).abs() < 0.01);
    // Q5_1
    assert!((super::ggml_dtype_element_size(7) - 0.75).abs() < 0.01);
    // Q8_0
    assert!((super::ggml_dtype_element_size(8) - 1.0625).abs() < 0.01);
    // Q8_1
    assert!((super::ggml_dtype_element_size(9) - 1.125).abs() < 0.01);
    // Q2_K
    assert!((super::ggml_dtype_element_size(10) - 0.3125).abs() < 0.01);
    // Q3_K
    assert!((super::ggml_dtype_element_size(11) - 0.4375).abs() < 0.01);
    // Q4_K
    assert!((super::ggml_dtype_element_size(12) - 0.5625).abs() < 0.01);
    // Q5_K
    assert!((super::ggml_dtype_element_size(13) - 0.6875).abs() < 0.01);
    // Q6_K
    assert!((super::ggml_dtype_element_size(14) - 0.8125).abs() < 0.01);
    // Q8_K
    assert!((super::ggml_dtype_element_size(15) - 1.0625).abs() < 0.01);
    // BF16
    assert!((super::ggml_dtype_element_size(26) - 2.0).abs() < 0.001);
    // I-quant types
    assert!((super::ggml_dtype_element_size(16) - 0.5625).abs() < 0.01); // IQ2_XXS
    assert!((super::ggml_dtype_element_size(17) - 0.625).abs() < 0.01); // IQ2_XS
    assert!((super::ggml_dtype_element_size(18) - 0.6875).abs() < 0.01); // IQ3_XXS
    assert!((super::ggml_dtype_element_size(19) - 0.4375).abs() < 0.01); // IQ1_S
    assert!((super::ggml_dtype_element_size(20) - 0.5625).abs() < 0.01); // IQ4_NL
    assert!((super::ggml_dtype_element_size(21) - 0.4375).abs() < 0.01); // IQ3_S
    assert!((super::ggml_dtype_element_size(22) - 0.625).abs() < 0.01); // IQ2_S
    assert!((super::ggml_dtype_element_size(23) - 0.5).abs() < 0.01); // IQ4_XS
                                                                      // Integer types
    assert!((super::ggml_dtype_element_size(24) - 1.0).abs() < 0.01); // I8
    assert!((super::ggml_dtype_element_size(25) - 2.0).abs() < 0.01); // I16
    assert!((super::ggml_dtype_element_size(27) - 4.0).abs() < 0.01); // I32
    assert!((super::ggml_dtype_element_size(28) - 8.0).abs() < 0.01); // I64
    assert!((super::ggml_dtype_element_size(29) - 8.0).abs() < 0.01); // F64
    assert!((super::ggml_dtype_element_size(30) - 0.375).abs() < 0.01); // IQ1_M
                                                                        // Unknown defaults to F32 (4.0) — conservative size estimate
    assert!((super::ggml_dtype_element_size(99) - 4.0).abs() < 0.001);
    assert!((super::ggml_dtype_element_size(u32::MAX) - 4.0).abs() < 0.001);
}

// ====================================================================
// Coverage: f16_to_f32 additional edge cases
// ====================================================================

#[test]
fn test_f16_to_f32_positive_zero() {
    // +0: sign=0, exp=0, mantissa=0 -> 0x0000
    let val = super::f16_to_f32(0x0000);
    assert_eq!(val, 0.0);
    assert!(val.is_sign_positive());
}

#[test]
fn test_f16_to_f32_normal_half() {
    // 0.5 in f16 = sign=0, exp=14, mantissa=0 -> 0x3800
    let val = super::f16_to_f32(0x3800);
    assert!((val - 0.5).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_largest_denorm() {
    // Largest denorm: sign=0, exp=0, mantissa=0x3FF -> 0x03FF
    let val = super::f16_to_f32(0x03FF);
    assert!(val > 0.0 && val < 1e-4, "largest denorm: {val}");
}

#[test]
fn test_f16_to_f32_negative_normal() {
    // -2.0 in f16 = sign=1, exp=16, mantissa=0 -> 0xC000
    let val = super::f16_to_f32(0xC000);
    assert!((val - (-2.0)).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_negative_nan() {
    // Negative NaN: sign=1, exp=31, mantissa!=0 -> 0xFE00
    let val = super::f16_to_f32(0xFE00);
    assert!(val.is_nan());
}

// ====================================================================
// Coverage: bf16_to_f32 additional cases
// ====================================================================

#[test]
fn test_bf16_to_f32_negative() {
    // -1.0 in bf16: top 16 bits of f32 -1.0 (0xBF80_0000) -> 0xBF80
    let val = super::bf16_to_f32(0xBF80);
    assert!((val - (-1.0)).abs() < 0.001);
}

#[test]
fn test_bf16_to_f32_large_value() {
    // 256.0 in bf16: top 16 bits of f32 256.0 (0x4380_0000) -> 0x4380
    let val = super::bf16_to_f32(0x4380);
    assert!((val - 256.0).abs() < 1.0);
}

#[test]
fn test_bf16_to_f32_small_positive() {
    // 0.5 in bf16: top 16 bits of f32 0.5 (0x3F00_0000) -> 0x3F00
    let val = super::bf16_to_f32(0x3F00);
    assert!((val - 0.5).abs() < 0.001);
}

// ====================================================================
// Coverage: safetensors_bytes_to_f32 - F16 branch
// ====================================================================

#[test]
fn test_safetensors_bytes_to_f32_f16() {
    // 1.0 in f16 = 0x3C00
    let f16_bytes = 0x3C00u16.to_le_bytes();
    let result = super::safetensors_bytes_to_f32(&f16_bytes, "F16");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 0.001);
}

#[test]
fn test_safetensors_bytes_to_f32_f16_multiple() {
    // Two f16 values: 1.0 (0x3C00), 2.0 (0x4000)
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x3C00u16.to_le_bytes());
    bytes.extend_from_slice(&0x4000u16.to_le_bytes());
    let result = super::safetensors_bytes_to_f32(&bytes, "F16");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[1] - 2.0).abs() < 0.001);
}

#[test]
fn test_safetensors_bytes_to_f32_f32_multiple() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1.5f32.to_le_bytes());
    bytes.extend_from_slice(&(-2.5f32).to_le_bytes());
    let result = super::safetensors_bytes_to_f32(&bytes, "F32");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.5).abs() < 1e-6);
    assert!((result[1] - (-2.5)).abs() < 1e-6);
}
