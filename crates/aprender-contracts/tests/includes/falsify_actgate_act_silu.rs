proptest! {
    /// FALSIFY-ACT-001: ReLU non-negative
    /// Contract: activation-kernel-v1.yaml
    /// Prediction: relu(x)_i >= 0 for all i
    /// If fails: ReLU produces negative output, violating its definition
    #[test]
    fn falsify_act_001_relu_non_negative(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        relu_scalar(&input, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y >= 0.0,
                "FALSIFY-ACT-001 failed: relu output[{i}] = {y} < 0"
            );
        }
    }
}

/// FALSIFY-ACT-002: GELU at zero
/// Contract: activation-kernel-v1.yaml
/// Prediction: gelu([0.0]) is approximately 0.0
/// If fails: GELU does not pass through origin
#[test]
fn falsify_act_002_gelu_at_zero() {
    let input = [0.0f32];
    let mut output = [0.0f32; 1];
    gelu_scalar(&input, &mut output);
    assert!(
        output[0].abs() < 1e-7,
        "FALSIFY-ACT-002 failed: gelu(0.0) = {}, expected ~0.0",
        output[0]
    );
}

proptest! {
    /// FALSIFY-ACT-003: GELU approx error
    /// Contract: activation-kernel-v1.yaml
    /// Prediction: |gelu(x)_i - x_i * 0.5 * (1 + tanh(sqrt(2/pi) * (x_i + 0.044715 * x_i^3)))| < 0.005
    /// If fails: GELU implementation deviates from the standard tanh approximation formula
    #[test]
    fn falsify_act_003_gelu_approx_error(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        gelu_scalar(&input, &mut output);
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        for (&x, &y) in input.iter().zip(output.iter()) {
            let expected = x * 0.5 * (1.0 + (sqrt_2_over_pi * (x + 0.044_715 * x * x * x)).tanh());
            let err = (y - expected).abs();
            prop_assert!(
                err < 0.005,
                "FALSIFY-ACT-003 failed: gelu({x}) = {y}, expected {expected}, error = {err}"
            );
        }
    }
}

/// FALSIFY-ACT-004: `SiLU` zero preservation
/// Contract: activation-kernel-v1.yaml
/// Prediction: silu([0.0]) = [0.0]
/// If fails: `SiLU` does not preserve zero
#[test]
fn falsify_act_004_silu_zero_preservation() {
    let input = [0.0f32];
    let mut output = [0.0f32; 1];
    silu_scalar(&input, &mut output);
    assert!(
        output[0].abs() < 1e-7,
        "FALSIFY-ACT-004 failed: silu(0.0) = {}, expected 0.0",
        output[0]
    );
}

proptest! {
    /// FALSIFY-ACT-005: ReLU monotonicity
    /// Contract: activation-kernel-v1.yaml
    /// Prediction: for sorted input, relu output is non-decreasing
    /// If fails: ReLU violates monotonicity
    #[test]
    fn falsify_act_005_relu_monotonicity(
        mut input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        input.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut output = vec![0.0f32; input.len()];
        relu_scalar(&input, &mut output);
        for i in 1..output.len() {
            prop_assert!(
                output[i] >= output[i - 1],
                "FALSIFY-ACT-005 failed: relu not monotone: output[{}] = {} < output[{}] = {}",
                i, output[i], i - 1, output[i - 1]
            );
        }
    }

    /// FALSIFY-ACT-006: SIMD equivalence
    /// Contract: activation-kernel-v1.yaml
    /// Prediction: avx2 relu vs scalar relu within 4 ULP
    /// If fails: AVX2 ReLU diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_act_006_simd_equivalence(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];
        relu_scalar(&input, &mut scalar_out);
        unsafe { relu_avx2(&input, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 4,
            "FALSIFY-ACT-006 failed: relu avx2 vs scalar ULP distance = {ulp} > 4"
        );
    }
}

// ============================================================================
// SiLU Standalone (FALSIFY-SI-001 through FALSIFY-SI-006)
// ============================================================================

/// FALSIFY-SI-001: Zero preservation
/// Contract: silu-kernel-v1.yaml
/// Prediction: `silu_standalone`([0.0]) = [0.0]
/// If fails: standalone `SiLU` does not preserve zero
#[test]
fn falsify_si_001_zero_preservation() {
    let input = [0.0f32];
    let mut output = [0.0f32; 1];
    silu_standalone_scalar(&input, &mut output);
    assert!(
        output[0].abs() < 1e-7,
        "FALSIFY-SI-001 failed: silu_standalone(0.0) = {}, expected 0.0",
        output[0]
    );
}

proptest! {
    /// FALSIFY-SI-002: Lower bound
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: silu(x)_i >= -0.279 (known minimum of silu approx -0.278 at x approx -1.278)
    /// If fails: SiLU violates known lower bound
    #[test]
    fn falsify_si_002_lower_bound(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y >= -0.279,
                "FALSIFY-SI-002 failed: silu_standalone output[{i}] = {y} < -0.279"
            );
        }
    }

    /// FALSIFY-SI-003: Monotonicity for large x
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: for x > 1, silu is increasing (sorted positive values give non-decreasing output)
    /// If fails: SiLU is not monotone for positive inputs
    #[test]
    fn falsify_si_003_monotonicity_large_x(
        mut input in proptest::collection::vec(1.0f32..10.0, 2..=32),
    ) {
        input.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut output = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut output);
        for i in 1..output.len() {
            prop_assert!(
                output[i] >= output[i - 1],
                "FALSIFY-SI-003 failed: silu not monotone for positive x: output[{}] = {} < output[{}] = {}",
                i, output[i], i - 1, output[i - 1]
            );
        }
    }

    /// FALSIFY-SI-004: Asymptotic
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: for large positive x (> 10), silu(x) approx x (within |x|*0.01)
    /// If fails: SiLU does not converge to identity for large positive x
    #[test]
    fn falsify_si_004_asymptotic(
        input in proptest::collection::vec(10.0f32..100.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut output);
        for (&x, &y) in input.iter().zip(output.iter()) {
            let ratio = y / x;
            prop_assert!(
                (ratio - 1.0).abs() < 0.01,
                "FALSIFY-SI-004 failed: silu({x}) / {x} = {ratio}, expected ~1.0"
            );
        }
    }

    /// FALSIFY-SI-005: SIMD equivalence
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: avx2 silu_standalone vs scalar within 4 ULP
    /// If fails: AVX2 SiLU standalone diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_si_005_simd_equivalence(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut scalar_out);
        unsafe { silu_standalone_avx2(&input, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 4,
            "FALSIFY-SI-005 failed: silu_standalone avx2 vs scalar ULP distance = {ulp} > 4"
        );
    }

    /// FALSIFY-SI-006: Sigmoid range
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: sigmoid output in [0, 1]
    /// If fails: sigmoid produces values outside the unit interval
    #[test]
    fn falsify_si_006_sigmoid_range(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        sigmoid_scalar(&input, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&y),
                "FALSIFY-SI-006 failed: sigmoid output[{i}] = {y} not in [0, 1]"
            );
        }
    }
}
