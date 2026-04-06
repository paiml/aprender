
#[test]
fn test_fused_swiglu_simd_16_elements() {
    // 16 elements - 2 SIMD iterations
    let mut gate: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let up: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let expected = swiglu_reference(&gate, &up);

    fused_swiglu_simd(&mut gate, &up);

    for (i, (g, e)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 0.15, // Lenient for AVX2 polynomial approx
            "16 elements SwiGLU: mismatch at {}: got {}, expected {}",
            i,
            g,
            e
        );
    }
}

#[test]
fn test_fused_swiglu_simd_17_elements_remainder() {
    // 17 elements - tests remainder handling
    let mut gate: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 0.3).collect();
    let up: Vec<f32> = (0..17).map(|i| (i as f32 + 1.0) * 0.2).collect();
    let expected = swiglu_reference(&gate, &up);

    fused_swiglu_simd(&mut gate, &up);

    for (i, (g, e)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 0.15, // Lenient for AVX2 polynomial approx
            "17 elements SwiGLU: mismatch at {}: got {}, expected {}",
            i,
            g,
            e
        );
    }
}

#[test]
fn test_fused_swiglu_simd_7_elements_scalar() {
    // 7 elements - scalar fallback
    let mut gate = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let up = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let expected = swiglu_reference(&gate, &up);

    fused_swiglu_simd(&mut gate, &up);

    for (i, (g, e)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 0.15, // Lenient for AVX2 polynomial approx
            "7 elements SwiGLU: mismatch at {}: got {}, expected {}",
            i,
            g,
            e
        );
    }
}

#[test]
fn test_fused_swiglu_simd_with_negative_up() {
    // Test with negative up values
    let mut gate = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let up = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
    let expected = swiglu_reference(&gate, &up);

    fused_swiglu_simd(&mut gate, &up);

    for (i, (g, e)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 0.15, // Lenient for AVX2 polynomial approx
            "Negative up SwiGLU: mismatch at {}: got {}, expected {}",
            i,
            g,
            e
        );
    }
}

#[test]
fn test_fused_swiglu_simd_mixed_up_values() {
    // Test with mixed positive/negative up values
    let mut gate = vec![1.0, -1.0, 2.0, -2.0, 0.0, 3.0, -3.0, 0.5];
    let up = vec![1.0, -1.0, 2.0, -2.0, 0.0, 3.0, -3.0, 0.5];
    let expected = swiglu_reference(&gate, &up);

    fused_swiglu_simd(&mut gate, &up);

    for (i, (g, e)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 0.15, // Lenient for AVX2 polynomial approx
            "Mixed up SwiGLU: mismatch at {}: got {}, expected {}",
            i,
            g,
            e
        );
    }
}

/// Reference implementation of SwiGLU
fn swiglu_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let sigmoid = 1.0 / (1.0 + (-g).exp());
            g * sigmoid * u
        })
        .collect()
}

// =============================================================================
// Horizontal Sum Tests (x86_64 specific)
// =============================================================================

// Note: The hsum_* functions are unsafe and require AVX2.
// We test them indirectly through the public SIMD functions above.
// The following tests verify the behavior when SIMD paths are taken.

#[test]
fn test_simd_path_selection_softmax() {
    // Test various sizes to exercise different code paths
    let sizes = [
        1, 2, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ];

    for &size in &sizes {
        let mut x: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.1)
            .collect();
        let reference = softmax_reference(&x);
        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Size {}: sum should be 1.0, got {}",
            size,
            sum
        );

        for (i, (actual, expected)) in x.iter().zip(reference.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-4,
                "Size {}: mismatch at {}: got {}, expected {}",
                size,
                i,
                actual,
                expected
            );
        }
    }
}

#[test]
fn test_simd_path_selection_swiglu() {
    // Test various sizes to exercise different code paths
    let sizes = [1, 2, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65];

    for &size in &sizes {
        let mut gate: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.2)
            .collect();
        let up: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let expected = swiglu_reference(&gate, &up);

        fused_swiglu_simd(&mut gate, &up);

        for (i, (actual, exp)) in gate.iter().zip(expected.iter()).enumerate() {
            // Use 0.2 tolerance for AVX2 polynomial approximation
            assert!(
                (actual - exp).abs() < 0.2,
                "Size {}: mismatch at {}: got {}, expected {}",
                size,
                i,
                actual,
                exp
            );
        }
    }
}

// =============================================================================
// Numerical Precision Tests
// =============================================================================

#[test]
fn test_f16_to_f32_round_trip_accuracy() {
    // Test that f16_to_f32 produces accurate results for known values
    // by comparing against the half crate

    let test_values: Vec<u16> = vec![
        0x0000, // 0
        0x3C00, // 1
        0x4000, // 2
        0x3800, // 0.5
        0x3400, // 0.25
        0x4200, // 3
        0x4400, // 4
        0x4800, // 8
        0x5000, // 32
        0x5800, // 128
        0x7BFF, // max normal (65504)
    ];

    for bits in test_values {
        let our_result = f16_to_f32(bits);
        let half_result = half::f16::from_bits(bits).to_f32();

        if our_result.abs() < 1e-10 && half_result.abs() < 1e-10 {
            // Both near zero
            continue;
        }

        let tolerance = half_result.abs() * 1e-4 + 1e-10;
        assert!(
            (our_result - half_result).abs() <= tolerance,
            "Round trip for 0x{:04X}: our={}, half={}",
            bits,
            our_result,
            half_result
        );
    }
}

#[test]
fn test_softmax_numerical_stability_extreme() {
    // Test with extreme values that could cause overflow/underflow
    let mut x = vec![
        1e38_f32,  // Very large
        -1e38_f32, // Very negative
        0.0,
        1.0,
        -1.0,
        1e-38_f32,  // Very small positive
        -1e-38_f32, // Very small negative
        f32::MAX / 2.0,
    ];

    softmax_simd(&mut x);

    // Check no NaN or Inf
    for (i, v) in x.iter().enumerate() {
        assert!(!v.is_nan(), "Extreme values: NaN at index {}", i);
        assert!(!v.is_infinite(), "Extreme values: Inf at index {}", i);
        assert!(*v >= 0.0, "Extreme values: negative at index {}: {}", i, v);
    }

    // Check sum is 1.0
    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "Extreme values: sum should be 1.0, got {}",
        sum
    );
}

#[test]
fn test_swiglu_boundary_values() {
    // Test SwiGLU with boundary values
    let mut gate = vec![
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::EPSILON,
        -f32::EPSILON,
        0.0,
        -0.0,
        1.0,
        -1.0,
    ];
    let up = vec![1.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // Check no NaN
    for (i, v) in gate.iter().enumerate() {
        assert!(!v.is_nan(), "Boundary SwiGLU: NaN at index {}", i);
        assert!(!v.is_infinite(), "Boundary SwiGLU: Inf at index {}", i);
    }
}
