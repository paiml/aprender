
// =========================================================================
// Quantization and Internal Function Tests (Coverage Boost)
// =========================================================================

#[test]
fn test_calculate_tensor_size() {
    let mut tensors = BTreeMap::new();
    tensors.insert("a".to_string(), (vec![1.0f32; 100], vec![10, 10]));
    tensors.insert("b".to_string(), (vec![2.0f32; 50], vec![50]));
    let size = calculate_tensor_size(&tensors);
    // 100 * 4 + 50 * 4 = 600
    assert_eq!(size, 600);
}

#[test]
fn test_calculate_tensor_size_empty() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    assert_eq!(calculate_tensor_size(&tensors), 0);
}

#[test]
fn test_quantize_fp16_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, -1.0, 0.0, 0.5];
    let quantized = quantize_fp16(&data);
    // Should preserve values with f16 precision
    assert_eq!(quantized.len(), data.len());
    for (orig, quant) in data.iter().zip(quantized.iter()) {
        // f16 has limited precision
        assert!((orig - quant).abs() < 0.01, "fp16 should preserve value");
    }
}

#[test]
fn test_quantize_fp16_large_values() {
    let data = vec![65504.0, -65504.0]; // max f16 values
    let quantized = quantize_fp16(&data);
    assert!((quantized[0] - 65504.0).abs() < 1.0);
}

#[test]
fn test_quantize_int8_roundtrip() {
    let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
    let quantized = quantize_int8(&data);
    assert_eq!(quantized.len(), data.len());
    // int8 quantization scales to -127..127
    for (orig, quant) in data.iter().zip(quantized.iter()) {
        assert!(
            (orig - quant).abs() < 0.05,
            "int8 should preserve value within tolerance"
        );
    }
}

#[test]
fn test_quantize_int8_all_zeros() {
    let data = vec![0.0, 0.0, 0.0];
    let quantized = quantize_int8(&data);
    for v in &quantized {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn test_quantize_int4_roundtrip() {
    let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
    let quantized = quantize_int4(&data);
    assert_eq!(quantized.len(), data.len());
    // int4 has only 16 levels so lower precision
    for (orig, quant) in data.iter().zip(quantized.iter()) {
        assert!(
            (orig - quant).abs() < 0.15,
            "int4 should preserve value within tolerance"
        );
    }
}

#[test]
fn test_quantize_int4_all_zeros() {
    let data = vec![0.0, 0.0, 0.0];
    let quantized = quantize_int4(&data);
    for v in &quantized {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn test_f16_to_f32_zero() {
    assert_eq!(f16_to_f32(0x0000), 0.0);
}

#[test]
fn test_f16_to_f32_one() {
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_negative() {
    let result = f16_to_f32(0xBC00);
    assert!((result + 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_subnormal() {
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0 && result < 0.001);
}

#[test]
fn test_f16_to_f32_max() {
    // Max f16 is 65504
    let result = f16_to_f32(0x7BFF);
    assert!((result - 65504.0).abs() < 1.0);
}

#[test]
fn test_convert_report_zero_sizes() {
    let report = ConvertReport {
        original_size: 0,
        converted_size: 0,
        tensor_count: 0,
        quantization: None,
        compression: None,
        reduction_ratio: 0.0,
    };
    assert_eq!(report.reduction_percent(), "N/A");
}

#[test]
fn test_convert_report_debug() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 500,
        tensor_count: 10,
        quantization: Some(QuantizationType::Int8),
        compression: Some(Compression::Lz4),
        reduction_ratio: 2.0,
    };
    assert!(format!("{:?}", report).contains("ConvertReport"));
}

#[test]
fn test_quantize_tensors_fp16() {
    let mut tensors = BTreeMap::new();
    tensors.insert("w".to_string(), (vec![1.0, 2.0, 3.0], vec![3]));
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Fp16).expect("quantize");
    assert!(result.as_ref().contains_key("w"));
}

#[test]
fn test_quantize_tensors_int8() {
    let mut tensors = BTreeMap::new();
    tensors.insert("w".to_string(), (vec![1.0, -1.0, 0.5], vec![3]));
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8).expect("quantize");
    assert!(result.as_ref().contains_key("w"));
}

#[test]
fn test_quantize_tensors_int4() {
    let mut tensors = BTreeMap::new();
    tensors.insert("w".to_string(), (vec![0.5, -0.5, 0.0], vec![3]));
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4).expect("quantize");
    assert!(result.as_ref().contains_key("w"));
}

#[test]
fn test_dequantize_q4k_to_f32_basic() {
    // Create a minimal Q4K block (144 bytes for 256 elements)
    let mut data = vec![0u8; 144];
    // Set d = 1.0 in f16 (0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    // Set dmin = 0.0
    data[2] = 0x00;
    data[3] = 0x00;
    let result = dequantize_q4_k_to_f32(&data, 256);
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q4k_to_f32_truncated() {
    // Data smaller than one block
    let data = vec![0u8; 50];
    let result = dequantize_q4_k_to_f32(&data, 256);
    // Should produce zero-filled result
    assert_eq!(result.len(), 256);
}

/// PMAT-177: Test that NaN/Inf scale factors are replaced with safe values
#[test]
fn test_dequantize_q4k_nan_inf_protection_pmat177() {
    // Create a Q4K block with NaN d value (f16 NaN = 0x7E00)
    let mut data = vec![0u8; 144];
    // Set d = NaN in f16 (0x7E00)
    data[0] = 0x00;
    data[1] = 0x7E;
    // Set dmin = Inf in f16 (0x7C00)
    data[2] = 0x00;
    data[3] = 0x7C;

    let result = dequantize_q4_k_to_f32(&data, 256);

    // PMAT-177: Result should contain NO NaN or Inf values
    let nan_count = result.iter().filter(|v| v.is_nan()).count();
    let inf_count = result.iter().filter(|v| v.is_infinite()).count();

    assert_eq!(
        nan_count, 0,
        "PMAT-177: dequantize_q4_k should not produce NaN"
    );
    assert_eq!(
        inf_count, 0,
        "PMAT-177: dequantize_q4_k should not produce Inf"
    );
}

/// PMAT-177: Test that subnormal f16 scales are clamped to zero
#[test]
fn test_dequantize_q4k_subnormal_protection_pmat177() {
    // Create a Q4K block with subnormal d value (f16 subnormal = 0x0001)
    let mut data = vec![0u8; 144];
    // Set d = subnormal in f16 (0x0001 - smallest subnormal)
    data[0] = 0x01;
    data[1] = 0x00;
    // Set dmin = 0.0
    data[2] = 0x00;
    data[3] = 0x00;

    let result = dequantize_q4_k_to_f32(&data, 256);

    // PMAT-177: Subnormal should be treated as zero, result should be all zeros
    let non_zero_count = result.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(
        non_zero_count, 0,
        "PMAT-177: subnormal f16 scales should be clamped to zero"
    );
}

#[test]
fn test_calculate_merge_weights_average() {
    let options = MergeOptions {
        strategy: MergeStrategy::Average,
        weights: None,
        ..Default::default()
    };
    let weights = calculate_merge_weights(3, &options).expect("weights");
    assert_eq!(weights.len(), 3);
    for w in &weights {
        assert!((*w - 1.0 / 3.0).abs() < 0.001);
    }
}

#[test]
fn test_calculate_merge_weights_custom() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![0.5, 0.3, 0.2]),
        ..Default::default()
    };
    let weights = calculate_merge_weights(3, &options).expect("weights");
    // Weighted merging always normalizes
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
}

#[test]
fn test_calculate_merge_weights_normalize() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![2.0, 2.0, 1.0]),
        ..Default::default()
    };
    let weights = calculate_merge_weights(3, &options).expect("weights");
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
    // Check relative proportions: 2:2:1
    assert!((weights[0] - 0.4).abs() < 0.001);
    assert!((weights[1] - 0.4).abs() < 0.001);
    assert!((weights[2] - 0.2).abs() < 0.001);
}

#[test]
fn test_calculate_merge_weights_zero_sum() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![0.0, 0.0, 0.0]),
        ..Default::default()
    };
    let result = calculate_merge_weights(3, &options);
    assert!(result.is_err());
}
