// ============================================================================
// SwiGLU (FALSIFY-SG-001 through FALSIFY-SG-006)
// ============================================================================

/// FALSIFY-SG-001: Zero gate
/// Contract: swiglu-kernel-v1.yaml
/// Prediction: swiglu(gate=[0,...], value=anything) = [0,...] since silu(0)=0
/// If fails: `SwiGLU` with zero gate does not produce zero output
#[test]
fn falsify_sg_001_zero_gate() {
    let gate = [0.0f32; 8];
    #[allow(clippy::approx_constant)]
    let value = [1.0f32, -2.5, 3.14, 100.0, -0.001, 42.0, 0.0, -99.9];
    let mut output = [0.0f32; 8];
    swiglu_scalar(&gate, &value, &mut output);
    for (i, &y) in output.iter().enumerate() {
        assert!(
            y == 0.0,
            "FALSIFY-SG-001 failed: swiglu(0, value)[{i}] = {y}, expected 0.0"
        );
    }
}

proptest! {
    /// FALSIFY-SG-002: Fused equals unfused
    /// Contract: swiglu-kernel-v1.yaml
    /// Prediction: swiglu(g, v) approx silu(g) * v element-wise (within tolerance)
    /// If fails: fused SwiGLU disagrees with manual silu(g) * v computation
    #[test]
    fn falsify_sg_002_fused_equals_unfused(
        gate in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        value in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let n = gate.len().min(value.len());
        let gate = &gate[..n];
        let value = &value[..n];
        let mut output = vec![0.0f32; n];
        swiglu_scalar(gate, value, &mut output);
        for ((&g, &v), &y) in gate.iter().zip(value.iter()).zip(output.iter()) {
            let sigmoid_g = 1.0 / (1.0 + (-g).exp());
            let expected = g * sigmoid_g * v;
            let err = (y - expected).abs();
            prop_assert!(
                err < 1e-4,
                "FALSIFY-SG-002 failed: swiglu({g}, {v}) = {y}, manual = {expected}, error = {err}"
            );
        }
    }

    /// FALSIFY-SG-003: Output bound
    /// Contract: swiglu-kernel-v1.yaml
    /// Prediction: swiglu output is finite for moderate inputs
    /// If fails: SwiGLU produces non-finite values for moderate inputs
    #[test]
    fn falsify_sg_003_output_bound(
        gate in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        value in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let n = gate.len().min(value.len());
        let gate = &gate[..n];
        let value = &value[..n];
        let mut output = vec![0.0f32; n];
        swiglu_scalar(gate, value, &mut output);
        common::assert_all_finite(&output);
    }

    /// FALSIFY-SG-004: SIMD equivalence
    /// Contract: swiglu-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: AVX2 SwiGLU diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_sg_004_simd_equivalence(
        gate in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        value in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = gate.len().min(value.len());
        let gate = &gate[..n];
        let value = &value[..n];
        let mut scalar_out = vec![0.0f32; n];
        let mut avx2_out = vec![0.0f32; n];
        swiglu_scalar(gate, value, &mut scalar_out);
        unsafe { swiglu_avx2(gate, value, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-SG-004 failed: swiglu avx2 vs scalar ULP distance = {ulp} > 8"
        );
    }
}

/// FALSIFY-SG-005: Symmetry check
/// Contract: swiglu-kernel-v1.yaml
/// Prediction: swiglu with value=[1,1,...] equals silu of gate
/// If fails: `SwiGLU` with unit value does not reduce to `SiLU`
#[test]
fn falsify_sg_005_symmetry_check() {
    let gate = [-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
    let value = [1.0f32; 8];
    let mut swiglu_out = [0.0f32; 8];
    let mut silu_out = [0.0f32; 8];
    swiglu_scalar(&gate, &value, &mut swiglu_out);
    silu_scalar(&gate, &mut silu_out);
    for (i, (&sg, &si)) in swiglu_out.iter().zip(silu_out.iter()).enumerate() {
        assert!(
            (sg - si).abs() < 1e-6,
            "FALSIFY-SG-005 failed: swiglu(gate, 1)[{i}] = {sg} != silu(gate)[{i}] = {si}"
        );
    }
}

proptest! {
    /// FALSIFY-SG-006: Finiteness
    /// Contract: swiglu-kernel-v1.yaml
    /// Prediction: output is finite for moderate inputs
    /// If fails: SwiGLU produces NaN or infinity for moderate inputs
    #[test]
    fn falsify_sg_006_finiteness(
        gate in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        value in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let n = gate.len().min(value.len());
        let gate = &gate[..n];
        let value = &value[..n];
        let mut output = vec![0.0f32; n];
        swiglu_scalar(gate, value, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y.is_finite(),
                "FALSIFY-SG-006 failed: swiglu output[{i}] = {y} is not finite"
            );
        }
    }
}

// ============================================================================
// Cross-Entropy (FALSIFY-CE-001 through FALSIFY-CE-006)
// ============================================================================

proptest! {
    /// FALSIFY-CE-001: Non-negative
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: cross_entropy(one_hot, logits) >= 0
    /// If fails: cross-entropy loss is negative, violating information-theoretic lower bound
    #[test]
    fn falsify_ce_001_non_negative(
        logits in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        class_idx_raw in 0usize..32,
    ) {
        let n = logits.len();
        let class_idx = class_idx_raw % n;
        let mut targets = vec![0.0f32; n];
        targets[class_idx] = 1.0;
        let loss = cross_entropy_scalar(&targets, &logits);
        prop_assert!(
            loss >= -1e-6,
            "FALSIFY-CE-001 failed: cross_entropy = {loss} < 0"
        );
    }

    /// FALSIFY-CE-002: Log-softmax <= 0
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: all elements of log_softmax(x) <= 0
    /// If fails: log-softmax produces positive values, violating log(probability) <= 0
    #[test]
    fn falsify_ce_002_log_softmax_nonpositive(
        logits in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; logits.len()];
        log_softmax_scalar(&logits, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y <= 1e-7,
                "FALSIFY-CE-002 failed: log_softmax output[{i}] = {y} > 0"
            );
        }
    }

    /// FALSIFY-CE-003: Finiteness
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: log_softmax output is all finite for moderate inputs [-100, 100]
    /// If fails: log-softmax produces NaN or infinity despite numerical stability measures
    #[test]
    fn falsify_ce_003_finiteness(
        logits in proptest::collection::vec(-100.0f32..100.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; logits.len()];
        log_softmax_scalar(&logits, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y.is_finite(),
                "FALSIFY-CE-003 failed: log_softmax output[{i}] = {y} is not finite"
            );
        }
    }

    /// FALSIFY-CE-004: NLL equivalence
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: cross_entropy with one-hot target selecting class k equals -log_softmax(x)[k]
    /// If fails: cross-entropy does not reduce to negative log-likelihood for one-hot targets
    #[test]
    fn falsify_ce_004_nll_equivalence(
        logits in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        class_idx_raw in 0usize..32,
    ) {
        let n = logits.len();
        let k = class_idx_raw % n;
        let mut targets = vec![0.0f32; n];
        targets[k] = 1.0;
        let loss = cross_entropy_scalar(&targets, &logits);
        let mut log_sm = vec![0.0f32; n];
        log_softmax_scalar(&logits, &mut log_sm);
        let expected = -log_sm[k];
        let err = (loss - expected).abs();
        prop_assert!(
            err < 1e-5,
            "FALSIFY-CE-004 failed: cross_entropy = {loss}, -log_softmax[{k}] = {expected}, error = {err}"
        );
    }

    /// FALSIFY-CE-005: SIMD equivalence
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: avx2 cross_entropy vs scalar within relative tolerance 1e-5
    /// If fails: AVX2 cross-entropy diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_ce_005_simd_equivalence(
        logits in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        class_idx_raw in 0usize..32,
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = logits.len();
        let k = class_idx_raw % n;
        let mut targets = vec![0.0f32; n];
        targets[k] = 1.0;
        let scalar_loss = cross_entropy_scalar(&targets, &logits);
        let avx2_loss = unsafe { cross_entropy_avx2(&targets, &logits) };
        let err = (scalar_loss - avx2_loss).abs();
        let denom = scalar_loss.abs().max(1e-8);
        let rel_err = err / denom;
        prop_assert!(
            rel_err < 1e-5,
            "FALSIFY-CE-005 failed: scalar = {scalar_loss}, avx2 = {avx2_loss}, rel_err = {rel_err}"
        );
    }

    /// FALSIFY-CE-006: Log-softmax normalization
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: exp(log_softmax(x)) sums to approximately 1.0
    /// If fails: softmax probabilities derived from log-softmax do not sum to 1
    #[test]
    fn falsify_ce_006_log_softmax_normalization(
        logits in proptest::collection::vec(-50.0f32..50.0, 2..=32),
    ) {
        let mut log_sm = vec![0.0f32; logits.len()];
        log_softmax_scalar(&logits, &mut log_sm);
        let sum: f32 = log_sm.iter().map(|&y| y.exp()).sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "FALSIFY-CE-006 failed: exp(log_softmax) sums to {sum}, expected ~1.0"
        );
    }
}

// ============================================================================
// RoPE (FALSIFY-RP-001 through FALSIFY-RP-004)
// ============================================================================

proptest! {
    /// FALSIFY-RP-001: Norm preservation
    /// Contract: rope-kernel-v1.yaml
    /// Prediction: ||rope(x)|| approx ||x|| (rotation preserves norm)
    /// If fails: RoPE rotation does not preserve vector norm
    #[test]
    fn falsify_rp_001_norm_preservation(
        half_len in 1usize..=16,
        raw_values in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        position in 0u32..1000,
    ) {
        let dim = half_len * 2;
        let x: Vec<f32> = raw_values.into_iter().take(dim).chain(std::iter::repeat(0.0)).take(dim).collect();
        let mut output = vec![0.0f32; dim];
        rope_scalar(&x, position, dim, 10000.0, &mut output);
        let input_norm = common::l2_norm(&x);
        let output_norm = common::l2_norm(&output);
        prop_assert!(
            (input_norm - output_norm).abs() < 1e-3,
            "FALSIFY-RP-001 failed: input norm = {input_norm}, output norm = {output_norm}"
        );
    }

    /// FALSIFY-RP-002: Relative position
    /// Contract: rope-kernel-v1.yaml
    /// Prediction: rope at position p1 != rope at position p2 for same non-zero input
    /// If fails: RoPE fails to encode different positions differently
    #[test]
    fn falsify_rp_002_relative_position(
        half_len in 1usize..=16,
        raw_values in proptest::collection::vec(1.0f32..10.0, 2..=32),
        p1 in 1u32..500,
        p2 in 500u32..1000,
    ) {
        let dim = half_len * 2;
        let x: Vec<f32> = raw_values.into_iter().take(dim).chain(std::iter::repeat(1.0)).take(dim).collect();
        let mut out1 = vec![0.0f32; dim];
        let mut out2 = vec![0.0f32; dim];
        rope_scalar(&x, p1, dim, 10000.0, &mut out1);
        rope_scalar(&x, p2, dim, 10000.0, &mut out2);
        let differs = out1.iter().zip(out2.iter()).any(|(&a, &b)| (a - b).abs() > 1e-7);
        prop_assert!(
            differs,
            "FALSIFY-RP-002 failed: rope at position {p1} and {p2} produced identical output"
        );
    }

    /// FALSIFY-RP-003: Output bounds
    /// Contract: rope-kernel-v1.yaml
    /// Prediction: all outputs finite and bounded
    /// If fails: RoPE produces non-finite output values
    #[test]
    fn falsify_rp_003_output_bounds(
        half_len in 1usize..=16,
        raw_values in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        position in 0u32..10000,
    ) {
        let dim = half_len * 2;
        let x: Vec<f32> = raw_values.into_iter().take(dim).chain(std::iter::repeat(0.0)).take(dim).collect();
        let mut output = vec![0.0f32; dim];
        rope_scalar(&x, position, dim, 10000.0, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y.is_finite(),
                "FALSIFY-RP-003 failed: rope output[{i}] = {y} is not finite"
            );
        }
    }

    /// FALSIFY-RP-004: SIMD equivalence
    /// Contract: rope-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 4 ULP
    /// If fails: AVX2 RoPE diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_rp_004_simd_equivalence(
        half_len in 1usize..=16,
        raw_values in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        position in 0u32..1000,
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let dim = half_len * 2;
        let x: Vec<f32> = raw_values.into_iter().take(dim).chain(std::iter::repeat(0.0)).take(dim).collect();
        let mut scalar_out = vec![0.0f32; dim];
        let mut avx2_out = vec![0.0f32; dim];
        rope_scalar(&x, position, dim, 10000.0, &mut scalar_out);
        unsafe { rope_avx2(&x, position, dim, 10000.0, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 4,
            "FALSIFY-RP-004 failed: rope avx2 vs scalar ULP distance = {ulp} > 4"
        );
    }
}
