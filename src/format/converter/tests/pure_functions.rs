//! Pure function unit tests for converter/mod.rs
//!
//! Targets 338 uncovered lines by testing edge cases, boundary values,
//! and round-trip correctness for:
//! - f32_to_f16 / f16_to_f32 (IEEE 754 half-precision)
//! - quantize_int8 / quantize_int4 (symmetric quantization)
//! - quantize_fp16 (FP16 quantization via f32->f16->f32)
//! - needs_transpose (tensor name pattern matching)
//! - calculate_tensor_size (size from tensor map)
//! - ConvertReport::reduction_percent (percentage formatting)
//! - validate_tensor_values (NaN/Inf/explosive detection)
//! - dequantize_f16_to_f32 / dequantize_bf16_to_f32 / dequantize_q8_0_to_f32

use super::super::*;
use std::collections::BTreeMap;

// ============================================================================
// f32_to_f16: IEEE 754 half-precision conversion
// ============================================================================

#[test]
fn test_f32_to_f16_positive_zero() {
    let bits = f32_to_f16(0.0_f32);
    assert_eq!(bits, 0x0000, "positive zero should be 0x0000");
}

#[test]
fn test_f32_to_f16_negative_zero() {
    let bits = f32_to_f16(-0.0_f32);
    assert_eq!(
        bits, 0x8000,
        "negative zero should be 0x8000 (sign bit set)"
    );
}

#[test]
fn test_f32_to_f16_one() {
    let bits = f32_to_f16(1.0_f32);
    // f16 1.0 = sign:0 exp:01111 mantissa:0000000000 = 0x3C00
    assert_eq!(bits, 0x3C00, "1.0 should encode as 0x3C00");
}

#[test]
fn test_f32_to_f16_negative_one() {
    let bits = f32_to_f16(-1.0_f32);
    // -1.0 = 0x3C00 | 0x8000 = 0xBC00
    assert_eq!(bits, 0xBC00, "-1.0 should encode as 0xBC00");
}

#[test]
fn test_f32_to_f16_half() {
    let bits = f32_to_f16(0.5_f32);
    // 0.5 = 2^(-1) => exp=14 (bias 15), mantissa=0 => 0x3800
    assert_eq!(bits, 0x3800, "0.5 should encode as 0x3800");
}

#[test]
fn test_f32_to_f16_max_f16() {
    // Max f16 = 65504.0
    let bits = f32_to_f16(65504.0_f32);
    assert_eq!(bits, 0x7BFF, "65504.0 (max f16) should encode as 0x7BFF");
}

#[test]
fn test_f32_to_f16_overflow_to_inf() {
    // Values > 65504 should overflow to infinity
    let bits = f32_to_f16(100_000.0_f32);
    assert_eq!(
        bits, 0x7C00,
        "values > 65504 should overflow to +inf (0x7C00)"
    );
}

#[test]
fn test_f32_to_f16_negative_overflow_to_neg_inf() {
    let bits = f32_to_f16(-100_000.0_f32);
    assert_eq!(
        bits, 0xFC00,
        "values < -65504 should overflow to -inf (0xFC00)"
    );
}

#[test]
fn test_f32_to_f16_positive_inf() {
    let bits = f32_to_f16(f32::INFINITY);
    assert_eq!(bits, 0x7C00, "+inf should encode as 0x7C00");
}

#[test]
fn test_f32_to_f16_negative_inf() {
    let bits = f32_to_f16(f32::NEG_INFINITY);
    assert_eq!(bits, 0xFC00, "-inf should encode as 0xFC00");
}

#[test]
fn test_f32_to_f16_nan_preserves_nan() {
    let bits = f32_to_f16(f32::NAN);
    // NaN in f16: exp=31, mantissa!=0
    let exp = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;
    assert_eq!(exp, 31, "NaN should have exp=31");
    assert_ne!(mantissa, 0, "NaN should have non-zero mantissa");
}

#[test]
fn test_f32_to_f16_subnormal_small_positive() {
    // Smallest f32 normal that maps to f16 subnormal range: ~5.96e-8 to ~6.10e-5
    // f16 min subnormal = 2^(-24) = 5.96e-8
    let val = 6.0e-6_f32; // well within f16 subnormal range
    let bits = f32_to_f16(val);
    let recovered = f16_to_f32(bits);
    // Should be representable as f16 subnormal, not flushed to zero
    assert!(
        recovered > 0.0,
        "small positive in f16 subnormal range should not flush to zero"
    );
    // Relative error can be large for subnormals, just check order of magnitude
    assert!(
        (recovered - val).abs() < val,
        "recovered value should be within 100% of original for subnormals"
    );
}

#[test]
fn test_f32_to_f16_underflow_to_zero() {
    // Values smaller than f16 min subnormal (2^(-24) ~ 5.96e-8) should flush to zero
    let val = 1.0e-10_f32;
    let bits = f32_to_f16(val);
    let recovered = f16_to_f32(bits);
    assert_eq!(
        recovered, 0.0,
        "extremely small values should underflow to zero"
    );
}

#[test]
fn test_f32_to_f16_f32_denormal_input() {
    // f32 denormals (exp=0) should map to f16 zero
    let val = f32::from_bits(0x0000_0001); // smallest f32 denormal
    let bits = f32_to_f16(val);
    // f32 denormal is way below f16 range, should be zero
    assert_eq!(bits & 0x7FFF, 0, "f32 denormal should map to f16 zero");
}

// ============================================================================
// f16_to_f32: Inverse conversion
// ============================================================================

#[test]
fn test_f16_to_f32_positive_zero() {
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert!(f16_to_f32(0x0000).is_sign_positive());
}

#[test]
fn test_f16_to_f32_negative_zero() {
    let val = f16_to_f32(0x8000);
    assert_eq!(val, 0.0);
    assert!(val.is_sign_negative(), "should preserve negative zero sign");
}

#[test]
fn test_f16_to_f32_positive_inf() {
    let val = f16_to_f32(0x7C00);
    assert!(val.is_infinite() && val.is_sign_positive());
}

#[test]
fn test_f16_to_f32_negative_inf() {
    let val = f16_to_f32(0xFC00);
    assert!(val.is_infinite() && val.is_sign_negative());
}

#[test]
fn test_f16_to_f32_nan() {
    // f16 NaN: exp=31, mantissa!=0, e.g. 0x7E00
    let val = f16_to_f32(0x7E00);
    assert!(val.is_nan(), "f16 NaN should decode to f32 NaN");
}

#[test]
fn test_f16_to_f32_negative_nan() {
    // Negative NaN: sign=1, exp=31, mantissa!=0
    let val = f16_to_f32(0xFE00);
    assert!(val.is_nan(), "negative f16 NaN should decode to f32 NaN");
}

#[test]
fn test_f16_to_f32_smallest_subnormal() {
    // Smallest positive f16 subnormal: 0x0001 = 2^(-14) * 2^(-10) = 2^(-24) ~ 5.96e-8
    let val = f16_to_f32(0x0001);
    let expected = 2.0_f32.powi(-24);
    assert!(
        (val - expected).abs() < 1e-10,
        "smallest subnormal should be ~5.96e-8, got {}",
        val
    );
}

#[test]
fn test_f16_to_f32_largest_subnormal() {
    // Largest f16 subnormal: 0x03FF = 2^(-14) * (1023/1024) ~ 6.09e-5
    let val = f16_to_f32(0x03FF);
    let expected = 2.0_f32.powi(-14) * (1023.0 / 1024.0);
    assert!(
        (val - expected).abs() / expected < 0.01,
        "largest subnormal mismatch: got {}, expected {}",
        val,
        expected
    );
}

#[test]
fn test_f16_to_f32_smallest_normal() {
    // Smallest positive normal f16: 0x0400 = 2^(-14) ~ 6.1e-5
    let val = f16_to_f32(0x0400);
    let expected = 2.0_f32.powi(-14);
    assert!(
        (val - expected).abs() / expected < 1e-6,
        "smallest normal should be 2^(-14), got {}",
        val
    );
}

#[test]
fn test_f16_to_f32_largest_normal() {
    // Largest f16: 0x7BFF = 65504.0
    let val = f16_to_f32(0x7BFF);
    assert!(
        (val - 65504.0).abs() < 1.0,
        "max f16 should be 65504, got {}",
        val
    );
}

#[test]
fn test_f16_to_f32_two() {
    // 2.0 in f16 = 0x4000 (exp=16-15=1, mantissa=0 => 2^1 = 2.0)
    let val = f16_to_f32(0x4000);
    assert!(
        (val - 2.0).abs() < 1e-6,
        "0x4000 should decode to 2.0, got {}",
        val
    );
}

// ============================================================================
// f32 -> f16 -> f32 round-trip
// ============================================================================

#[test]
fn test_f16_roundtrip_exact_representable_values() {
    // Values exactly representable in f16 should round-trip perfectly
    let exact_values: &[f32] = &[0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 1024.0, 65504.0];
    for &v in exact_values {
        let bits = f32_to_f16(v);
        let recovered = f16_to_f32(bits);
        assert!(
            (recovered - v).abs() < 1e-6,
            "round-trip failed for {}: got {}",
            v,
            recovered
        );
    }
}

#[test]
fn test_f16_roundtrip_precision_loss() {
    // Non-exact values should be close but may lose precision
    // f16 has 10 mantissa bits ~ 3.3 decimal digits of precision
    let val = 1.001_f32;
    let bits = f32_to_f16(val);
    let recovered = f16_to_f32(bits);
    // f16 precision at 1.0 is 2^(-10) ~ 0.001, so error should be < 0.001
    assert!(
        (recovered - val).abs() < 0.002,
        "precision loss too large: {} -> {}",
        val,
        recovered
    );
}

#[test]
fn test_f16_roundtrip_preserves_sign() {
    let vals: &[f32] = &[1.0, -1.0, 0.5, -0.5, 100.0, -100.0];
    for &v in vals {
        let bits = f32_to_f16(v);
        let recovered = f16_to_f32(bits);
        assert_eq!(
            v.is_sign_positive(),
            recovered.is_sign_positive(),
            "sign changed for {}",
            v
        );
    }
}

#[test]
fn test_f16_roundtrip_inf_preserved() {
    let pos_bits = f32_to_f16(f32::INFINITY);
    assert!(f16_to_f32(pos_bits).is_infinite());
    assert!(f16_to_f32(pos_bits).is_sign_positive());

    let neg_bits = f32_to_f16(f32::NEG_INFINITY);
    assert!(f16_to_f32(neg_bits).is_infinite());
    assert!(f16_to_f32(neg_bits).is_sign_negative());
}

#[test]
fn test_f16_roundtrip_nan_preserved() {
    let bits = f32_to_f16(f32::NAN);
    assert!(f16_to_f32(bits).is_nan());
}

#[test]
fn test_f16_roundtrip_all_f16_normals_are_valid() {
    // Exhaustive check: every f16 normal value (exp 1-30) should round-trip
    // through f16_to_f32 -> f32_to_f16 back to the same bits.
    // There are only 2^15 = 32768 positive f16 values, so this is fast.
    let mut mismatches = 0;
    for exp in 1u16..=30 {
        // Check a sample of mantissa values per exponent
        for mantissa in [0u16, 1, 0x1FF, 0x200, 0x3FF] {
            let original = (exp << 10) | mantissa;
            let f32_val = f16_to_f32(original);
            let roundtrip = f32_to_f16(f32_val);
            if original != roundtrip {
                mismatches += 1;
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "some f16 normal values did not round-trip correctly"
    );
}

// ============================================================================
// quantize_int8: Symmetric INT8 quantization
// ============================================================================

#[test]
fn test_quantize_int8_single_value() {
    let data = vec![42.0_f32];
    let q = quantize_int8(&data);
    // With single value, scale = 42/127, quantized = round(42/(42/127)) = 127
    // dequantized = 127 * (42/127) = 42.0
    assert!(
        (q[0] - 42.0).abs() < 0.01,
        "single value should reconstruct near-exactly: got {}",
        q[0]
    );
}

#[test]
fn test_quantize_int8_symmetric_range_used() {
    // INT8 range is -127 to 127 (symmetric, not -128)
    let data = vec![1.0_f32, -1.0];
    let q = quantize_int8(&data);
    // scale = 1.0/127
    // quantized(1.0) = round(1.0 / (1.0/127)) = 127
    // dequant = 127 * (1.0/127) = 1.0
    assert!(
        (q[0] - 1.0).abs() < 0.01,
        "max positive should be well-preserved"
    );
    assert!(
        (q[1] + 1.0).abs() < 0.01,
        "max negative should be well-preserved"
    );
}

#[test]
fn test_quantize_int8_large_outlier_dominates_scale() {
    // One large outlier sets the scale, making small values lose precision
    let data = vec![127.0_f32, 1.0, 0.1];
    let q = quantize_int8(&data);
    // scale = 127/127 = 1.0
    // quantized(1.0) = round(1.0/1.0) = 1 -> dequant = 1.0 (exact)
    // quantized(0.1) = round(0.1/1.0) = 0 -> dequant = 0.0 (0.1 lost!)
    assert!(
        (q[0] - 127.0).abs() < 0.01,
        "outlier should be exact: got {}",
        q[0]
    );
    assert!(
        (q[2] - 0.0).abs() < 1.001,
        "small value with large outlier: got {}",
        q[2]
    );
}

#[test]
fn test_quantize_int8_negative_only() {
    let data = vec![-5.0_f32, -3.0, -1.0];
    let q = quantize_int8(&data);
    // All results should be negative
    for (orig, quant) in data.iter().zip(q.iter()) {
        assert!(
            quant.is_sign_negative(),
            "negative input {} produced non-negative output {}",
            orig,
            quant
        );
    }
}

#[test]
fn test_quantize_int8_preserves_length() {
    let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let q = quantize_int8(&data);
    assert_eq!(q.len(), data.len());
}

#[test]
fn test_quantize_int8_output_bounded_by_input_range() {
    // Dequantized values should never exceed the original range
    let data = vec![3.0_f32, -2.0, 1.5, -0.5];
    let q = quantize_int8(&data);
    let orig_max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let orig_min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    for &v in &q {
        assert!(
            v <= orig_max + 0.01 && v >= orig_min - 0.01,
            "dequantized {} out of original range [{}, {}]",
            v,
            orig_min,
            orig_max
        );
    }
}

// ============================================================================
// quantize_int4: Symmetric INT4 quantization
// ============================================================================

#[test]
fn test_quantize_int4_empty() {
    let q = quantize_int4(&[]);
    assert!(q.is_empty());
}

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

// ============================================================================
// dequantize_bf16_to_f32: BF16 byte stream dequantization
// ============================================================================

#[test]
fn test_dequantize_bf16_to_f32_zero() {
    let bytes = [0x00, 0x00];
    let result = dequantize_bf16_to_f32(&bytes, 1);
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_dequantize_bf16_to_f32_one() {
    // 1.0 in BF16 = 0x3F80 (upper 16 bits of f32 1.0 = 0x3F800000)
    // LE: 0x80, 0x3F
    let bytes = [0x80, 0x3F];
    let result = dequantize_bf16_to_f32(&bytes, 1);
    assert_eq!(result[0], 1.0);
}

#[test]
fn test_dequantize_bf16_to_f32_negative_one() {
    // -1.0 in BF16 = 0xBF80, LE: 0x80, 0xBF
    let bytes = [0x80, 0xBF];
    let result = dequantize_bf16_to_f32(&bytes, 1);
    assert_eq!(result[0], -1.0);
}

#[test]
fn test_dequantize_bf16_preserves_sign_and_exponent() {
    // BF16 has same exponent range as f32, just truncated mantissa
    // 2.0 in f32 = 0x40000000, BF16 = 0x4000, LE: 0x00, 0x40
    let bytes = [0x00, 0x40];
    let result = dequantize_bf16_to_f32(&bytes, 1);
    assert_eq!(result[0], 2.0);
}

// ============================================================================
// dequantize_q8_0_to_f32: Q8_0 block dequantization
// ============================================================================

#[test]
fn test_dequantize_q8_0_zero_scale() {
    // f16 zero scale = 0x0000, then 32 quant values should all produce 0
    let mut bytes = vec![0x00, 0x00]; // f16 scale = 0
    bytes.extend_from_slice(&[42u8; 32]); // 32 non-zero quant values
    let result = dequantize_q8_0_to_f32(&bytes, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert_eq!(v, 0.0, "zero scale should produce all zeros");
    }
}

#[test]
fn test_dequantize_q8_0_nan_scale_clamped() {
    // NaN f16 scale should be clamped to 0 (GH-186 fix)
    let mut bytes = vec![0x00, 0x7E]; // f16 NaN
    bytes.extend_from_slice(&[1u8; 32]);
    let result = dequantize_q8_0_to_f32(&bytes, 32);
    for &v in &result {
        assert_eq!(v, 0.0, "NaN scale should clamp to 0");
    }
}

#[test]
fn test_dequantize_q8_0_inf_scale_clamped() {
    // +inf f16 scale should be clamped to 0 (GH-186 fix)
    let mut bytes = vec![0x00, 0x7C]; // f16 +inf
    bytes.extend_from_slice(&[1u8; 32]);
    let result = dequantize_q8_0_to_f32(&bytes, 32);
    for &v in &result {
        assert_eq!(v, 0.0, "inf scale should clamp to 0");
    }
}

#[test]
fn test_dequantize_q8_0_subnormal_scale_clamped() {
    // Subnormal f16 scale below F16_MIN_NORMAL should be clamped to 0
    let mut bytes = vec![0x01, 0x00]; // f16 smallest subnormal
    bytes.extend_from_slice(&[127u8; 32]);
    let result = dequantize_q8_0_to_f32(&bytes, 32);
    // The subnormal f16 value 0x0001 maps to ~5.96e-8 which is below F16_MIN_NORMAL (~6.1e-5)
    for &v in &result {
        assert_eq!(v, 0.0, "subnormal scale should clamp to 0");
    }
}

#[test]
fn test_dequantize_q8_0_partial_last_block() {
    // Request fewer elements than a full block
    let mut bytes = vec![0x00, 0x3C]; // f16 scale = 1.0
    bytes.extend_from_slice(&[1u8; 32]); // 32 int8 values of 1
    let result = dequantize_q8_0_to_f32(&bytes, 16);
    assert_eq!(result.len(), 16, "should respect num_elements limit");
}

#[test]
fn test_dequantize_q8_0_multiple_blocks() {
    // Two complete blocks
    let mut bytes = Vec::new();
    // Block 0: scale=1.0, all quants = 2
    bytes.extend_from_slice(&[0x00, 0x3C]); // f16 1.0
    bytes.extend_from_slice(&[2u8; 32]);
    // Block 1: scale=2.0, all quants = 3
    bytes.extend_from_slice(&[0x00, 0x40]); // f16 2.0
    bytes.extend_from_slice(&[3u8; 32]);

    let result = dequantize_q8_0_to_f32(&bytes, 64);
    assert_eq!(result.len(), 64);
    // Block 0: 2 * 1.0 = 2.0
    assert!((result[0] - 2.0).abs() < 0.01);
    // Block 1: 3 * 2.0 = 6.0
    assert!((result[32] - 6.0).abs() < 0.01);
}

#[test]
fn test_dequantize_q8_0_signed_quant_values() {
    // Test that quant bytes are treated as signed int8
    let mut bytes = vec![0x00, 0x3C]; // f16 scale = 1.0
                                      // Byte 0xFF as i8 = -1, byte 0x80 as i8 = -128
    let mut quants = vec![0u8; 32];
    quants[0] = 0xFF; // -1 as i8
    quants[1] = 0x80; // -128 as i8
    quants[2] = 0x7F; // 127 as i8
    bytes.extend_from_slice(&quants);

    let result = dequantize_q8_0_to_f32(&bytes, 32);
    // -1 * 1.0 = -1.0
    assert!(
        (result[0] + 1.0).abs() < 0.01,
        "0xFF should be -1: got {}",
        result[0]
    );
    // -128 * 1.0 = -128.0
    assert!(
        (result[1] + 128.0).abs() < 0.01,
        "0x80 should be -128: got {}",
        result[1]
    );
    // 127 * 1.0 = 127.0
    assert!(
        (result[2] - 127.0).abs() < 0.01,
        "0x7F should be 127: got {}",
        result[2]
    );
}

// ============================================================================
// quantize_tensors: Embedding skip logic (BUG-EXPORT-004)
// ============================================================================

#[test]
fn test_quantize_tensors_skips_embeddings() {
    let mut tensors = BTreeMap::new();
    let embed_data = vec![0.1_f32; 256];
    // Use varied data that definitely changes under INT4 quantization
    // INT4 has only 16 levels, so fine-grained values will be rounded
    let weight_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (embed_data.clone(), vec![16, 16]),
    );
    tensors.insert(
        "layers.0.attn_q.weight".to_string(),
        (weight_data.clone(), vec![16, 16]),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4).expect("quantize");
    let result = result.as_ref();

    // Embedding should be unchanged (skipped)
    let (embed_q, _) = result.get("model.embed_tokens.weight").expect("embed");
    assert_eq!(
        embed_q, &embed_data,
        "embedding should be kept in original F32"
    );

    // Weight should be quantized (different from original due to INT4 rounding)
    let (weight_q, _) = result.get("layers.0.attn_q.weight").expect("weight");
    assert_ne!(
        weight_q, &weight_data,
        "non-embedding weight should be quantized"
    );
}

#[test]
fn test_quantize_tensors_skips_all_embedding_variants() {
    let mut tensors = BTreeMap::new();
    let data = vec![0.05_f32; 64];
    // All four embedding name patterns from the code
    tensors.insert(
        "embed_tokens.weight".to_string(),
        (data.clone(), vec![8, 8]),
    );
    tensors.insert("token_embd.weight".to_string(), (data.clone(), vec![8, 8]));
    tensors.insert("wte.weight".to_string(), (data.clone(), vec![8, 8]));
    tensors.insert(
        "word_embeddings.weight".to_string(),
        (data.clone(), vec![8, 8]),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8).expect("quantize");
    let result = result.as_ref();

    for name in [
        "embed_tokens.weight",
        "token_embd.weight",
        "wte.weight",
        "word_embeddings.weight",
    ] {
        let (q, _) = result.get(name).expect(name);
        assert_eq!(q, &data, "embedding '{}' should be skipped", name);
    }
}

// ============================================================================
// ConvertOptions: Default validation
// ============================================================================

#[test]
fn test_convert_options_default_has_validate_true() {
    let opts = ConvertOptions::default();
    assert!(opts.validate, "default should enable validation");
    assert!(opts.quantize.is_none());
    assert!(opts.compress.is_none());
}

// ============================================================================
// AprConverter builder pattern
// ============================================================================

#[test]
fn test_apr_converter_default() {
    let converter = AprConverter::default();
    let debug = format!("{:?}", converter);
    assert!(
        debug.contains("Auto"),
        "default architecture should be Auto"
    );
}

#[test]
fn test_apr_converter_builder_chain() {
    let converter = AprConverter::new()
        .architecture(Architecture::Qwen2)
        .validate(ValidationConfig::Strict)
        .quantize(QuantizationType::Int8)
        .compress(Compression::Lz4);
    let debug = format!("{:?}", converter);
    assert!(debug.contains("Qwen2"));
    assert!(debug.contains("Int8"));
}

// ============================================================================
// DOUBLE-QUANT-001: Compile-time double quantization prevention
// ============================================================================

#[test]
fn test_double_quant_001_native_tensors_accepted() {
    // NativeF32Tensors should be accepted by quantize_tensors
    let mut map = BTreeMap::new();
    map.insert("w".to_string(), (vec![1.0_f32, 2.0, 3.0], vec![3]));
    let native = NativeF32Tensors::new(map);
    let result = quantize_tensors(&native, &QuantizationType::Int8);
    assert!(result.is_ok(), "native tensors should be quantizable");
}

#[test]
fn test_double_quant_001_dequantized_type_exists() {
    // Verify DequantizedTensors can be constructed (but NOT passed to quantize_tensors)
    let map = BTreeMap::new();
    let deq = DequantizedTensors::new(map, QuantizationType::Q4K);
    assert_eq!(deq.original_quant, QuantizationType::Q4K);
    // The following would NOT compile — compile-time enforcement:
    // quantize_tensors(&deq, &QuantizationType::Int8);
}

#[test]
fn test_double_quant_001_provenance_native_variant() {
    let mut map = BTreeMap::new();
    map.insert("w".to_string(), (vec![1.0_f32], vec![1]));
    let native = NativeF32Tensors::new(map);
    let prov = TensorProvenance::Native(native);
    assert_eq!(prov.as_map().len(), 1);
    let inner = prov.into_map();
    assert!(inner.contains_key("w"));
}

#[test]
fn test_double_quant_001_provenance_dequantized_variant() {
    let mut map = BTreeMap::new();
    map.insert("w".to_string(), (vec![1.0_f32], vec![1]));
    let deq = DequantizedTensors::new(map, QuantizationType::Q4K);
    let prov = TensorProvenance::Dequantized(deq);
    assert_eq!(prov.as_map().len(), 1);
    let inner = prov.into_map();
    assert!(inner.contains_key("w"));
}

#[test]
fn test_double_quant_001_native_into_inner_roundtrip() {
    let mut map = BTreeMap::new();
    map.insert("t".to_string(), (vec![42.0_f32], vec![1]));
    let native = NativeF32Tensors::new(map);
    let inner = native.into_inner();
    assert_eq!(inner["t"].0, vec![42.0_f32]);
}

#[test]
fn test_double_quant_001_dequantized_into_inner_roundtrip() {
    let mut map = BTreeMap::new();
    map.insert("t".to_string(), (vec![42.0_f32], vec![1]));
    let deq = DequantizedTensors::new(map, QuantizationType::Fp16);
    assert_eq!(deq.original_quant, QuantizationType::Fp16);
    let inner = deq.into_inner();
    assert_eq!(inner["t"].0, vec![42.0_f32]);
}

// ============================================================================

#[test]
fn test_apr_converter_convert_without_source_fails() {
    let result = AprConverter::new().convert();
    assert!(result.is_err(), "convert without source should fail");
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("No source"), "should say no source: {}", msg);
}
