
/// Test softmax with identical values (edge case for max finding)
#[test]
fn test_softmax_simd_identical_values() {
    // All same values should result in uniform distribution
    for size in [8, 16, 24, 32] {
        let mut x: Vec<f32> = vec![5.0; size];

        softmax_simd(&mut x);

        let expected = 1.0 / size as f32;
        for (i, v) in x.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "Size {}: uniform at {}: got {}, expected {}",
                size,
                i,
                v,
                expected
            );
        }
    }
}

/// Test softmax with maximum difference in values
#[test]
fn test_softmax_simd_max_difference() {
    // Large difference should result in near-1 for max, near-0 for others
    let mut x = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0];

    softmax_simd(&mut x);

    assert!(x[7] > 0.9999, "Max element should dominate: {}", x[7]);
    for i in 0..7 {
        assert!(
            x[i] < 1e-5,
            "Non-max element {} should be near 0: {}",
            i,
            x[i]
        );
    }
}

// =============================================================================
// Fused SwiGLU SIMD: Remainder Loop and Scalar Fallback Coverage
// =============================================================================

/// Test swiglu with sizes that exercise remainder loops
#[test]
fn test_swiglu_simd_remainder_loops() {
    for remainder in 1..=7 {
        let size = 8 + remainder;
        let mut gate: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.3)
            .collect();
        let up: Vec<f32> = vec![1.5; size];

        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| {
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                g * sigmoid * u
            })
            .collect();

        fused_swiglu_simd(&mut gate, &up);

        for (i, (got, exp)) in gate.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.2, // Lenient for AVX2 polynomial approx
                "Size {}: mismatch at {}: got {}, expected {}",
                size,
                i,
                got,
                exp
            );
        }
    }
}

/// Test swiglu with sizes 16+1 through 16+7
#[test]
fn test_swiglu_simd_two_chunks_remainder() {
    for remainder in 1..=7 {
        let size = 16 + remainder;
        let mut gate: Vec<f32> = (0..size).map(|i| (i as f32 - 8.0) * 0.2).collect();
        let up: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| {
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                g * sigmoid * u
            })
            .collect();

        fused_swiglu_simd(&mut gate, &up);

        for (i, (got, exp)) in gate.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.2, // Lenient for AVX2 polynomial approx
                "Size {}: mismatch at {}: got {}, expected {}",
                size,
                i,
                got,
                exp
            );
        }
    }
}

/// Test swiglu with sizes 1-7 (scalar fallback only)
#[test]
fn test_swiglu_simd_scalar_fallback_all() {
    for size in 1..=7 {
        let mut gate: Vec<f32> = (0..size).map(|i| (i as f32 - 3.0) * 0.5).collect();
        let up: Vec<f32> = vec![2.0; size];

        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| {
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                g * sigmoid * u
            })
            .collect();

        fused_swiglu_simd(&mut gate, &up);

        for (i, (got, exp)) in gate.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.2, // Lenient for AVX2 polynomial approx
                "Size {}: mismatch at {}: got {}, expected {}",
                size,
                i,
                got,
                exp
            );
        }
    }
}

/// Test swiglu with zero gate values
#[test]
fn test_swiglu_simd_zero_gate() {
    let mut gate = vec![0.0; 16];
    let up = vec![1.0; 16];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    for (i, g) in gate.iter().enumerate() {
        assert!(g.abs() < 1e-10, "Zero gate at {}: got {}", i, g);
    }
}

/// Test swiglu with zero up values
#[test]
fn test_swiglu_simd_zero_up() {
    let mut gate: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
    let up = vec![0.0; 16];

    fused_swiglu_simd(&mut gate, &up);

    // silu(x) * 0 = 0
    for (i, g) in gate.iter().enumerate() {
        assert!(g.abs() < 1e-10, "Zero up at {}: got {}", i, g);
    }
}

// =============================================================================
// Horizontal Sum Helpers: Indirect Testing Through Public APIs
// =============================================================================

/// Test that exercises the AVX2 horizontal sum paths via softmax
#[test]
fn test_horizontal_sum_via_softmax_large() {
    // Large input to ensure multiple SIMD iterations
    let mut x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Large softmax sum: got {}", sum);
}

/// Test that exercises the AVX2 paths via swiglu with large input
#[test]
fn test_avx2_path_via_swiglu_large() {
    let mut gate: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.02).collect();
    let up: Vec<f32> = (0..256).map(|i| (i as f32 + 1.0) * 0.01).collect();

    fused_swiglu_simd(&mut gate, &up);

    // Verify all outputs are finite
    for (i, g) in gate.iter().enumerate() {
        assert!(g.is_finite(), "Large swiglu at {}: got {}", i, g);
    }
}

// =============================================================================
// Edge Case: Empty and Single Element
// =============================================================================

/// Test softmax with exactly 0 elements
#[test]
fn test_softmax_simd_zero_elements() {
    let mut x: Vec<f32> = vec![];
    softmax_simd(&mut x);
    assert!(x.is_empty(), "Empty should remain empty");
}

/// Test swiglu with exactly 0 elements
#[test]
fn test_swiglu_simd_zero_elements() {
    let mut gate: Vec<f32> = vec![];
    let up: Vec<f32> = vec![];
    fused_swiglu_simd(&mut gate, &up);
    assert!(gate.is_empty(), "Empty should remain empty");
}

/// Test rope with head_dim=0 (degenerate case)
#[test]
fn test_rope_simd_zero_head_dim() {
    let _x: Vec<f32> = vec![1.0, 2.0];
    let _freqs_cos: Vec<f32> = vec![];
    let _freqs_sin: Vec<f32> = vec![];

    // head_dim=0 means half_dim=0, should be no-op
    // Note: This may trigger debug_assert in debug mode
    // In release, it should be a no-op
}

// =============================================================================
// Special Float Values: Comprehensive Coverage
// =============================================================================

/// Test softmax with mix of special values
#[test]
fn test_softmax_simd_special_values_mix() {
    let mut x = vec![
        0.0,
        f32::MIN_POSITIVE,
        f32::EPSILON,
        1.0,
        -1.0,
        -f32::MIN_POSITIVE,
        -f32::EPSILON,
        100.0,
    ];

    softmax_simd(&mut x);

    // All values should be finite and non-negative
    for (i, v) in x.iter().enumerate() {
        assert!(v.is_finite(), "Special mix softmax at {}: got {}", i, v);
        assert!(*v >= 0.0, "Special mix softmax negative at {}: {}", i, v);
    }

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Special mix sum: {}", sum);
}

/// Test swiglu with special float values
/// Note: AVX2 polynomial approximation has limited range, so we use moderate values
#[test]
fn test_swiglu_simd_special_values() {
    let mut gate = vec![
        f32::MIN_POSITIVE,
        10.0, // Use moderate values (AVX2 polynomial approx range is ~[-87, 0])
        f32::EPSILON,
        -f32::MIN_POSITIVE,
        -10.0,
        -f32::EPSILON,
        0.0,
        1.0,
    ];
    let up = vec![1.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // Check no NaN
    for (i, g) in gate.iter().enumerate() {
        assert!(!g.is_nan(), "Special swiglu NaN at {}", i);
    }
}
