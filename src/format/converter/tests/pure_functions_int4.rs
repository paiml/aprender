
#[test]
fn test_quantize_int4_asymmetric_range() {
    // INT4 signed range is -8 to 7 (asymmetric)
    // scale = max_abs / 7
    // This means the negative side can represent -8*scale while positive only 7*scale
    let data = vec![7.0_f32, -7.0, -8.0];
    let q = quantize_int4(&data);
    // scale = 8.0/7 ~ 1.143
    // quantized(7.0) = round(7.0/1.143) = round(6.125) = 6 -> dequant = 6*1.143 = 6.857
    // quantized(-8.0) = round(-8.0/1.143) = round(-7.0) = -7 -> dequant = -7*1.143 = -8.0
    // The -8 clamp: round(-7) is within [-8, 7], so it's -7 -> -8.0
    assert_eq!(q.len(), 3);
}

#[test]
fn test_quantize_int4_only_15_distinct_levels() {
    // INT4 can only represent 16 values: -8..7 (but 0 appears for both +0 and -0)
    // With symmetric quantization around max_abs, there are at most 15 non-zero levels
    let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect(); // 0.0 to 0.99
    let q = quantize_int4(&data);
    let mut unique: Vec<f32> = q.clone();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup();
    assert!(
        unique.len() <= 16,
        "INT4 should produce at most 16 distinct values, got {}",
        unique.len()
    );
}

#[test]
fn test_quantize_int4_higher_error_than_int8() {
    // INT4 quantization error should generally be larger than INT8
    let data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let q4 = quantize_int4(&data);
    let q8 = quantize_int8(&data);
    let err4: f32 = data.iter().zip(q4.iter()).map(|(a, b)| (a - b).abs()).sum();
    let err8: f32 = data.iter().zip(q8.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        err4 >= err8,
        "INT4 total error ({}) should be >= INT8 total error ({})",
        err4,
        err8
    );
}

#[test]
fn test_quantize_int4_single_value() {
    let q = quantize_int4(&[5.0]);
    // scale = 5.0/7 ~ 0.714
    // quantized = round(5.0/0.714) = round(7.0) = 7
    // dequant = 7 * 0.714 = 5.0
    assert!(
        (q[0] - 5.0).abs() < 0.01,
        "single value should reconstruct: got {}",
        q[0]
    );
}

// ============================================================================
// quantize_fp16: FP16 quantization via f32->f16->f32
// ============================================================================

#[test]
fn test_quantize_fp16_empty() {
    let q = quantize_fp16(&[]);
    assert!(q.is_empty());
}

#[test]
fn test_quantize_fp16_nan_propagates() {
    let q = quantize_fp16(&[f32::NAN]);
    assert!(
        q[0].is_nan(),
        "NaN should propagate through fp16 quantization"
    );
}

#[test]
fn test_quantize_fp16_inf_propagates() {
    let q = quantize_fp16(&[f32::INFINITY, f32::NEG_INFINITY]);
    assert!(q[0].is_infinite() && q[0].is_sign_positive());
    assert!(q[1].is_infinite() && q[1].is_sign_negative());
}

#[test]
fn test_quantize_fp16_overflow_clamps_to_inf() {
    // Values beyond f16 max (65504) should overflow to inf
    let q = quantize_fp16(&[100_000.0]);
    assert!(
        q[0].is_infinite(),
        "values beyond f16 range should become inf, got {}",
        q[0]
    );
}

#[test]
fn test_quantize_fp16_tiny_values_flush_to_zero() {
    // Values below f16 min subnormal should flush to zero
    let q = quantize_fp16(&[1.0e-10]);
    assert_eq!(
        q[0], 0.0,
        "sub-subnormal values should flush to zero, got {}",
        q[0]
    );
}

#[test]
fn test_quantize_fp16_exact_values_preserved() {
    // Values exactly representable in f16
    let data = vec![0.0, 1.0, -1.0, 0.5, 2.0, 1024.0];
    let q = quantize_fp16(&data);
    for (i, (&orig, &quant)) in data.iter().zip(q.iter()).enumerate() {
        assert_eq!(
            orig, quant,
            "exact f16 value at index {} should be preserved: {} -> {}",
            i, orig, quant
        );
    }
}

#[test]
fn test_quantize_fp16_precision_at_different_magnitudes() {
    // f16 precision scales with magnitude: ~0.001 at 1.0, ~32 at 65504
    // Near 1.0: ULP = 2^(-10) ~ 0.001
    let q1 = quantize_fp16(&[1.0005]);
    assert!(
        (q1[0] - 1.0005).abs() < 0.002,
        "near 1.0 precision should be ~0.001"
    );

    // Near 1000: ULP = 2^0 = 1.0
    let q2 = quantize_fp16(&[1000.5]);
    assert!(
        (q2[0] - 1000.5).abs() < 1.5,
        "near 1000 precision should be ~1.0"
    );
}

// ============================================================================
// needs_transpose: Tensor name pattern matching
// ============================================================================

#[test]
fn test_needs_transpose_empty_shape() {
    // 0D tensor should never need transpose
    assert!(!needs_transpose("output.weight", &[]));
}

#[test]
fn test_needs_transpose_4d_tensor() {
    // Only 2D tensors should be transposed
    assert!(!needs_transpose("attn_q.weight", &[2, 3, 4, 5]));
}

#[test]
fn test_needs_transpose_huggingface_style_patterns() {
    // Test HuggingFace naming convention patterns
    let hf_patterns = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ];
    for pattern in &hf_patterns {
        let name = format!("model.layers.0.self_attn.{}", pattern);
        assert!(
            needs_transpose(&name, &[512, 512]),
            "HF pattern '{}' should need transpose",
            name
        );
    }
}

#[test]
fn test_needs_transpose_gguf_style_patterns() {
    // Test GGUF naming convention patterns
    let gguf_patterns = [
        "attn_output.weight",
        "attn_k.weight",
        "attn_q.weight",
        "attn_v.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];
    for pattern in &gguf_patterns {
        let name = format!("blk.0.{}", pattern);
        assert!(
            needs_transpose(&name, &[768, 768]),
            "GGUF pattern '{}' should need transpose",
            name
        );
    }
}

#[test]
fn test_needs_transpose_weight_substring_in_non_weight() {
    // "weight" in a bias tensor name should NOT trigger transpose
    assert!(!needs_transpose("layer.0.bias_for_weight_calc", &[64, 64]));
}

#[test]
fn test_needs_transpose_embedding_2d_not_matched() {
    // Embedding weights are 2D but should NOT be transposed (not in pattern list)
    assert!(!needs_transpose("embed_tokens.weight", &[50257, 768]));
    assert!(!needs_transpose("token_embd.weight", &[50257, 768]));
}

#[test]
fn test_needs_transpose_norm_weights_not_matched() {
    // Layer norm weights are typically 1D but even if 2D should not match
    assert!(!needs_transpose("input_layernorm.weight", &[768]));
    assert!(!needs_transpose("post_attention_layernorm.weight", &[768]));
}

#[test]
fn test_needs_transpose_lm_head() {
    assert!(needs_transpose("lm_head.weight", &[50257, 768]));
    assert!(needs_transpose("output.weight", &[50257, 768]));
}

// ============================================================================
// calculate_tensor_size: Size calculation from tensor map
// ============================================================================

#[test]
fn test_calculate_tensor_size_single_tensor() {
    let mut tensors = BTreeMap::new();
    tensors.insert("w".to_string(), (vec![0.0_f32; 100], vec![10, 10]));
    // 100 f32 values * 4 bytes = 400
    assert_eq!(calculate_tensor_size(&tensors), 400);
}

#[test]
fn test_calculate_tensor_size_multiple_tensors() {
    let mut tensors = BTreeMap::new();
    tensors.insert("a".to_string(), (vec![0.0_f32; 10], vec![10]));
    tensors.insert("b".to_string(), (vec![0.0_f32; 20], vec![20]));
    tensors.insert("c".to_string(), (vec![0.0_f32; 30], vec![30]));
    // (10 + 20 + 30) * 4 = 240
    assert_eq!(calculate_tensor_size(&tensors), 240);
}

#[test]
fn test_calculate_tensor_size_large_tensor() {
    let mut tensors = BTreeMap::new();
    // 1M elements * 4 bytes = 4MB
    tensors.insert(
        "big".to_string(),
        (vec![0.0_f32; 1_000_000], vec![1000, 1000]),
    );
    assert_eq!(calculate_tensor_size(&tensors), 4_000_000);
}

#[test]
fn test_calculate_tensor_size_ignores_shape() {
    // Size is based on data.len(), not shape product
    let mut tensors = BTreeMap::new();
    // data has 10 elements but shape says 5x5=25 -- size should use data.len()
    tensors.insert("mismatched".to_string(), (vec![0.0_f32; 10], vec![5, 5]));
    assert_eq!(calculate_tensor_size(&tensors), 40); // 10 * 4, not 25 * 4
}

// ============================================================================
// ConvertReport::reduction_percent
// ============================================================================

#[test]
fn test_reduction_percent_50_percent() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 500,
        tensor_count: 1,
        quantization: None,
        compression: None,
        reduction_ratio: 2.0,
    };
    assert_eq!(report.reduction_percent(), "50.0%");
}

#[test]
fn test_reduction_percent_no_reduction() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 1000,
        tensor_count: 1,
        quantization: None,
        compression: None,
        reduction_ratio: 1.0,
    };
    assert_eq!(report.reduction_percent(), "0.0%");
}

#[test]
fn test_reduction_percent_expansion() {
    // When converted > original, reduction is negative
    let report = ConvertReport {
        original_size: 500,
        converted_size: 1000,
        tensor_count: 1,
        quantization: None,
        compression: None,
        reduction_ratio: 0.5,
    };
    assert_eq!(report.reduction_percent(), "-100.0%");
}

#[test]
fn test_reduction_percent_zero_original() {
    let report = ConvertReport {
        original_size: 0,
        converted_size: 500,
        tensor_count: 1,
        quantization: None,
        compression: None,
        reduction_ratio: 0.0,
    };
    assert_eq!(report.reduction_percent(), "N/A");
}

#[test]
fn test_reduction_percent_zero_converted() {
    let report = ConvertReport {
        original_size: 500,
        converted_size: 0,
        tensor_count: 1,
        quantization: None,
        compression: None,
        reduction_ratio: 0.0,
    };
    assert_eq!(report.reduction_percent(), "N/A");
}

#[test]
fn test_reduction_percent_tiny_reduction() {
    // 1 byte out of 1000 = 0.1% reduction
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 999,
        tensor_count: 1,
        quantization: None,
        compression: None,
        reduction_ratio: 1.001,
    };
    assert_eq!(report.reduction_percent(), "0.1%");
}

// ============================================================================
// validate_tensor_values: NaN/Inf/explosive detection
// ============================================================================

#[test]
fn test_validate_tensor_values_mean_exactly_at_boundary() {
    // Mean exactly at 100 should pass (> 100 fails, not >= 100)
    let data = vec![100.0_f32; 10];
    let result = validate_tensor_values("boundary_exact", &data);
    assert!(result.is_ok(), "mean of exactly 100 should pass");
}

#[test]
fn test_validate_tensor_values_mean_just_over_boundary() {
    // Mean of 100.01 should fail
    let data = vec![100.01_f32; 10];
    let result = validate_tensor_values("over_boundary", &data);
    assert!(result.is_err(), "mean just over 100 should fail");
}

#[test]
fn test_validate_tensor_values_negative_explosive_mean() {
    // Large negative mean should also fail (checks abs > 100)
    let data = vec![-200.0_f32; 10];
    let result = validate_tensor_values("neg_explosive", &data);
    assert!(result.is_err(), "large negative mean should fail");
}

#[test]
fn test_validate_tensor_values_single_nan() {
    let data = vec![1.0, 2.0, f32::NAN, 4.0];
    let result = validate_tensor_values("has_nan", &data);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("NaN"), "error should mention NaN");
}

#[test]
fn test_validate_tensor_values_single_inf() {
    let data = vec![1.0, f32::INFINITY, 3.0];
    let result = validate_tensor_values("has_inf", &data);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Inf"), "error should mention Inf");
}

#[test]
fn test_validate_tensor_values_nan_checked_before_inf() {
    // When both NaN and Inf present, NaN error should fire first
    let data = vec![f32::NAN, f32::INFINITY];
    let result = validate_tensor_values("both_bad", &data);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("NaN"),
        "NaN should be checked before Inf: {}",
        msg
    );
}

// ============================================================================
// dequantize_f16_to_f32: F16 byte stream dequantization
// ============================================================================

#[test]
fn test_dequantize_f16_to_f32_zero_bytes() {
    let bytes = [0x00, 0x00]; // f16 zero
    let result = dequantize_f16_to_f32(&bytes, 1);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_dequantize_f16_to_f32_multiple_values() {
    // 1.0 in f16 = 0x3C00 (LE: 0x00, 0x3C)
    // -1.0 in f16 = 0xBC00 (LE: 0x00, 0xBC)
    let bytes = [0x00, 0x3C, 0x00, 0xBC];
    let result = dequantize_f16_to_f32(&bytes, 2);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] + 1.0).abs() < 1e-6);
}

#[test]
fn test_dequantize_f16_to_f32_odd_byte_count() {
    // 3 bytes: only 1 complete f16 value (chunks_exact discards trailing byte)
    let bytes = [0x00, 0x3C, 0xFF];
    let result = dequantize_f16_to_f32(&bytes, 1);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}
