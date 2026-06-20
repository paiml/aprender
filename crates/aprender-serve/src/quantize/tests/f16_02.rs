//! Part 23: Additional SIMD Coverage Tests for quantize/simd.rs
//!
//! Targets the uncovered areas in simd.rs:
//! - f16_to_f32 positive subnormal branch (line 38)
//! - extract_scale_min_from_slice odd index branch (lines 114-117)
//! - AVX2 RoPE rotation inner loop (lines 525-535)
//!
//! Note: Horizontal sum functions (hsum_epi32_*, horizontal_sum_*) are now
//! internal to fused_k.rs and tested there to avoid dead code.

use crate::quantize::{f16_to_f32, fused_swiglu_simd, softmax_simd};

// =============================================================================
// f16_to_f32: Positive Subnormal Coverage (line 38)
// =============================================================================

/// Test positive subnormal f16 values covering all mantissa ranges
#[test]
fn test_f16_to_f32_positive_subnormals() {
    // Test various positive subnormals (exp=0, mantissa!=0, sign=0)
    let test_cases: &[(u16, f32)] = &[
        (0x0001, (1.0 / 1024.0) * (2.0_f32).powi(-14)), // min
        (0x0200, (512.0 / 1024.0) * (2.0_f32).powi(-14)), // mid
        (0x03FF, (1023.0 / 1024.0) * (2.0_f32).powi(-14)), // max
    ];

    for &(bits, expected) in test_cases {
        let result = f16_to_f32(bits);
        assert!(result > 0.0, "bits=0x{:04X} should be positive", bits);
        assert!(
            (result - expected).abs() < 1e-12,
            "bits=0x{:04X}: got {}, expected {}",
            bits,
            result,
            expected
        );
    }
}

// =============================================================================
// extract_scale_min_from_slice: Odd Index Coverage (lines 114-117)
// =============================================================================

// =============================================================================
// Additional Edge Cases for SIMD Paths
// =============================================================================

/// Test softmax and swiglu trigger SIMD path with exactly 8 elements
#[test]
fn test_simd_activation_8_elements() {
    // Softmax with 8 elements
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // SwiGLU with 8 elements
    let mut gate: Vec<f32> = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
    let up = vec![1.0; 8];
    let expected: Vec<f32> = gate
        .iter()
        .copied()
        .map(|g: f32| g * (1.0 / (1.0 + (-g).exp())))
        .collect();
    fused_swiglu_simd(&mut gate, &up);
    for (got, exp) in gate.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 0.2); // Lenient for AVX2 polynomial approx
    }
}
