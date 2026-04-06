proptest! {
    /// FALSIFY-SM-001: Normalization
    /// Contract: softmax-kernel-v1.yaml
    /// Prediction: sum(softmax(x)) = 1.0 for any random input in [-1000, 1000]^n
    /// If fails: missing max-subtraction trick causes overflow/NaN
    #[test]
    fn falsify_sm_001_normalization(
        v in proptest::collection::vec(-1000.0_f32..1000.0, 4..=32)
    ) {
        let mut output = vec![0.0_f32; v.len()];
        softmax_scalar(&v, &mut output);
        common::assert_all_finite(&output);
        common::assert_probability_distribution(&output, 1e-5);
    }

    /// FALSIFY-SM-002: Positivity
    /// Contract: softmax-kernel-v1.yaml
    /// Prediction: all softmax(x)_i > 0 for any random input in moderate range
    /// If fails: underflow or wrong formula produces zero/negative outputs
    /// Note: uses [-20, 20] range to avoid legitimate f32 underflow at extreme gaps
    #[test]
    fn falsify_sm_002_positivity(
        v in proptest::collection::vec(-20.0_f32..20.0, 4..=32)
    ) {
        let mut output = vec![0.0_f32; v.len()];
        softmax_scalar(&v, &mut output);
        for (i, &o) in output.iter().enumerate() {
            prop_assert!(
                o > 0.0,
                "FALSIFY-SM-002 failed: softmax output[{i}] = {o} is not strictly positive"
            );
        }
    }

    /// FALSIFY-SM-003: Order preservation
    /// Contract: softmax-kernel-v1.yaml
    /// Prediction: x_i > x_j implies softmax(x)_i >= softmax(x)_j
    /// Note: strict inequality relaxed to >= because f32 precision causes
    /// equal outputs when inputs differ by less than the precision threshold
    /// at the tail of the distribution (e.g., exp(-99.9) vs exp(-100.0)).
    #[test]
    fn falsify_sm_003_order_preservation(
        v in proptest::collection::vec(-50.0_f32..50.0, 4..=32)
    ) {
        let mut output = vec![0.0_f32; v.len()];
        softmax_scalar(&v, &mut output);
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                if v[i] > v[j] {
                    prop_assert!(
                        output[i] >= output[j],
                        "FALSIFY-SM-003 failed: v[{i}]={} > v[{j}]={} but softmax[{i}]={} < softmax[{j}]={}",
                        v[i], v[j], output[i], output[j]
                    );
                }
            }
        }
    }

    /// FALSIFY-SM-004: SIMD equivalence
    /// Contract: softmax-kernel-v1.yaml
    /// Prediction: avx2 and scalar produce results within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_sm_004_simd_equivalence(
        v in proptest::collection::vec(-100.0_f32..100.0, 4..=32)
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let mut scalar_out = vec![0.0_f32; v.len()];
        let mut avx2_out = vec![0.0_f32; v.len()];
        softmax_scalar(&v, &mut scalar_out);
        unsafe { softmax_avx2(&v, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-SM-004 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-SM-005: Single element
/// Contract: softmax-kernel-v1.yaml
/// Prediction: softmax([x]) = [1.0] for any single element
/// If fails: edge case handling for length-1 input is broken
#[test]
fn falsify_sm_005_single_element() {
    for &x in &[-1000.0_f32, -1.0, 0.0, 1.0, 1000.0] {
        let input = [x];
        let mut output = [0.0_f32; 1];
        softmax_scalar(&input, &mut output);
        assert!(
            (output[0] - 1.0).abs() < 1e-7,
            "FALSIFY-SM-005 failed: softmax([{x}]) = [{}], expected [1.0]",
            output[0]
        );
    }
}

/// FALSIFY-SM-006: Identical inputs
/// Contract: softmax-kernel-v1.yaml
/// Prediction: softmax([c, c, ..., c]) = uniform [1/n, ..., 1/n]
/// If fails: normalization does not produce uniform output for constant input
#[test]
fn falsify_sm_006_identical_inputs() {
    for n in [2, 5, 8, 16, 31] {
        let c = 42.0_f32;
        let input = vec![c; n];
        let mut output = vec![0.0_f32; n];
        softmax_scalar(&input, &mut output);
        #[allow(clippy::cast_precision_loss)]
        let expected = 1.0 / n as f32;
        for (i, &o) in output.iter().enumerate() {
            assert!(
                (o - expected).abs() < 1e-6,
                "FALSIFY-SM-006 failed: n={n}, output[{i}] = {o}, expected {expected}"
            );
        }
    }
}

// ============================================================================
// RMSNorm (5 tests: FALSIFY-RN-001 through FALSIFY-RN-005)
// ============================================================================

proptest! {
    /// FALSIFY-RN-001: Finiteness
    /// Contract: rmsnorm-kernel-v1.yaml
    /// Prediction: all outputs finite for random inputs
    /// If fails: division by zero without eps guard
    #[test]
    fn falsify_rn_001_finiteness(
        v in proptest::collection::vec(-100.0_f32..100.0, 4..=32)
    ) {
        let gamma = vec![1.0_f32; v.len()];
        let mut output = vec![0.0_f32; v.len()];
        rmsnorm_scalar(&v, &gamma, 1e-5, &mut output);
        for (i, &o) in output.iter().enumerate() {
            prop_assert!(
                o.is_finite(),
                "FALSIFY-RN-001 failed: output[{i}] = {o} is not finite"
            );
        }
    }

    /// FALSIFY-RN-002: Scale invariance
    /// Contract: rmsnorm-kernel-v1.yaml
    /// Prediction: rmsnorm(alpha*x) approx rmsnorm(x) for alpha in [0.1, 10]
    /// If fails: not properly normalizing by RMS
    #[test]
    fn falsify_rn_002_scale_invariance(
        v in proptest::collection::vec(0.1_f32..10.0, 4..=32),
        alpha in 0.1_f32..10.0
    ) {
        let gamma = vec![1.0_f32; v.len()];
        let mut out_original = vec![0.0_f32; v.len()];
        rmsnorm_scalar(&v, &gamma, 1e-8, &mut out_original);

        let scaled: Vec<f32> = v.iter().map(|&x| x * alpha).collect();
        let mut out_scaled = vec![0.0_f32; v.len()];
        rmsnorm_scalar(&scaled, &gamma, 1e-8, &mut out_scaled);

        let dist = common::l2_distance(&out_original, &out_scaled);
        prop_assert!(
            dist < 1e-3,
            "FALSIFY-RN-002 failed: l2_distance = {dist} between rmsnorm(x) and rmsnorm({alpha}*x)"
        );
    }

    /// FALSIFY-RN-003: SIMD equivalence
    /// Contract: rmsnorm-kernel-v1.yaml
    /// Prediction: avx2 and scalar produce results within 8 ULP
    /// Note: threshold widened from 4 to 8 — rmsnorm involves sqrt+div
    /// which accumulates more rounding than softmax (exp-only).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_rn_003_simd_equivalence(
        v in proptest::collection::vec(-10.0_f32..10.0, 4..=32)
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let gamma = vec![1.0_f32; v.len()];
        let mut scalar_out = vec![0.0_f32; v.len()];
        let mut avx2_out = vec![0.0_f32; v.len()];
        rmsnorm_scalar(&v, &gamma, 1e-5, &mut scalar_out);
        unsafe { rmsnorm_avx2(&v, &gamma, 1e-5, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-RN-003 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }

    /// FALSIFY-RN-005: Unit gamma RMS
    /// Contract: rmsnorm-kernel-v1.yaml
    /// Prediction: with gamma=1, output RMS approx 1.0
    /// If fails: normalization factor is incorrect
    #[test]
    fn falsify_rn_005_unit_gamma_rms(
        v in proptest::collection::vec(0.1_f32..10.0, 4..=32)
    ) {
        let gamma = vec![1.0_f32; v.len()];
        let mut output = vec![0.0_f32; v.len()];
        rmsnorm_scalar(&v, &gamma, 1e-8, &mut output);

        let sum_sq: f32 = output.iter().map(|x| x * x).sum();
        #[allow(clippy::cast_precision_loss)]
        let rms_out = (sum_sq / output.len() as f32).sqrt();
        prop_assert!(
            (rms_out - 1.0).abs() < 1e-3,
            "FALSIFY-RN-005 failed: output RMS = {rms_out}, expected approx 1.0"
        );
    }
}

/// FALSIFY-RN-004: Zero vector
/// Contract: rmsnorm-kernel-v1.yaml
/// Prediction: rmsnorm([0,...,0]) is all finite (eps prevents div-by-zero)
/// If fails: eps is not used or division by zero occurs
#[test]
fn falsify_rn_004_zero_vector() {
    for n in [1, 4, 8, 16, 31] {
        let input = vec![0.0_f32; n];
        let gamma = vec![1.0_f32; n];
        let mut output = vec![0.0_f32; n];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut output);
        common::assert_all_finite(&output);
        for (i, &o) in output.iter().enumerate() {
            assert!(
                o.abs() < 1e-2,
                "FALSIFY-RN-004 failed: n={n}, output[{i}] = {o}, expected approx 0.0"
            );
        }
    }
}
