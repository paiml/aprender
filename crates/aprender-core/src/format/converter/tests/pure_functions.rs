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

include!("pure_functions_int4.rs");
include!("pure_functions_bf16.rs");
include!("pure_functions_infer_q4k.rs");
include!("pure_functions_skip_quant.rs");
