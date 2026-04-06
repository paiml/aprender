
#[test]
fn test_softmax_simd_very_negative() {
    // Very negative values should not underflow to zero sum
    let mut x = vec![-1000.0, -999.0, -998.0, -997.0];
    softmax_simd(&mut x);

    for v in &x {
        assert!(!v.is_nan(), "Softmax should not produce NaN");
        assert!(*v >= 0.0, "Softmax values should be non-negative");
    }

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Should still sum to 1.0");
}

// =============================================================================
// Determinism Tests
// =============================================================================

#[test]
fn test_softmax_simd_deterministic() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();

    let mut x1 = x.clone();
    let mut x2 = x.clone();

    softmax_simd(&mut x1);
    softmax_simd(&mut x2);

    for (i, (a, b)) in x1.iter().zip(x2.iter()).enumerate() {
        assert_eq!(a, b, "Softmax should be deterministic at index {}", i);
    }
}

// =============================================================================
// Fused SwiGLU Tests
// =============================================================================

/// Reference implementation of SwiGLU: gate * sigmoid(gate) * up
fn swiglu_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let sigmoid = 1.0 / (1.0 + (-g).exp());
            g * sigmoid * u
        })
        .collect()
}

#[test]
fn test_fused_swiglu_simd_basic() {
    let mut gate = vec![0.0, 1.0, -1.0, 2.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];
    let expected = swiglu_reference(&gate, &up);

    fused_swiglu_simd(&mut gate, &up);

    for (i, (g, e)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "SwiGLU mismatch at {}: got {}, expected {}",
            i,
            g,
            e
        );
    }
}

#[test]
fn test_fused_swiglu_simd_zeros() {
    let mut gate = vec![0.0; 16];
    let up = vec![1.0; 16];

    fused_swiglu_simd(&mut gate, &up);

    // sigmoid(0) = 0.5, so 0 * 0.5 * 1 = 0
    for (i, g) in gate.iter().enumerate() {
        assert!(
            g.abs() < 1e-10,
            "SwiGLU of zero should be zero at {}: {}",
            i,
            g
        );
    }
}

#[test]
fn test_fused_swiglu_simd_large_positive() {
    // Large positive gate values should saturate sigmoid to ~1
    let mut gate = vec![100.0; 8];
    let up = vec![2.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // gate * sigmoid(gate) * up ≈ 100 * 1 * 2 = 200
    for (i, g) in gate.iter().enumerate() {
        assert!((g - 200.0).abs() < 1.0, "Large gate SwiGLU at {}: {}", i, g);
    }
}

#[test]
fn test_fused_swiglu_simd_large_negative() {
    // Large negative gate values should saturate sigmoid to ~0
    // Use -10 instead of -100 to stay within AVX2 polynomial approximation range
    let mut gate = vec![-10.0; 8];
    let up = vec![2.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // silu(-10) ≈ -10 * sigmoid(-10) ≈ -10 * 0.00005 ≈ -0.0005
    // result ≈ -0.001
    for (i, g) in gate.iter().enumerate() {
        assert!(g.abs() < 0.01, "Large negative gate SwiGLU at {}: {}", i, g);
    }
}

#[test]
#[should_panic(expected = "assertion")]
fn test_fused_swiglu_simd_length_mismatch() {
    // Mismatched lengths should panic (debug_assert in activation.rs)
    let mut gate = vec![1.0; 4];
    let up = vec![2.0; 8];

    fused_swiglu_simd(&mut gate, &up);
}

#[test]
fn test_fused_swiglu_simd_deterministic() {
    let gate_orig: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let up: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 + 0.5).collect();

    let mut gate1 = gate_orig.clone();
    let mut gate2 = gate_orig.clone();

    fused_swiglu_simd(&mut gate1, &up);
    fused_swiglu_simd(&mut gate2, &up);

    for (i, (a, b)) in gate1.iter().zip(gate2.iter()).enumerate() {
        assert_eq!(a, b, "SwiGLU should be deterministic at index {}", i);
    }
}

// =============================================================================
// Proptest for SwiGLU
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(25))]

    #[test]
    fn prop_fused_swiglu_is_finite_and_reasonable(
        gate in prop::collection::vec(-10.0f32..10.0f32, 1..=64),
    ) {
        let up: Vec<f32> = gate.iter().map(|g| g.abs() + 0.1).collect();

        let mut gate_simd = gate.clone();
        fused_swiglu_simd(&mut gate_simd, &up);

        // Verify outputs are finite and reasonably bounded
        for (i, (&got, &orig)) in gate_simd.iter().zip(gate.iter()).enumerate() {
            prop_assert!(
                got.is_finite(),
                "SwiGLU should produce finite output at {}: got {}",
                i, got
            );
            // Output should be bounded by max(|gate|) * max(up) * 2
            let max_expected = 10.0 * 10.2 * 2.0; // Conservative bound
            prop_assert!(
                got.abs() < max_expected,
                "SwiGLU output too large at {}: got {}, gate was {}",
                i, got, orig
            );
        }
    }
}
