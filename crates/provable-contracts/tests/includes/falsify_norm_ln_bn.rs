// ============================================================================
// LayerNorm (7 tests: FALSIFY-LN-001 through FALSIFY-LN-007)
// ============================================================================

proptest! {
    /// FALSIFY-LN-001: Centering
    /// Contract: layernorm-kernel-v1.yaml
    /// Prediction: mean(layernorm(x, gamma=1, beta=0)) approx 0
    /// If fails: wrong mean computation in the normalization
    #[test]
    fn falsify_ln_001_centering(
        v in proptest::collection::vec(-10.0_f32..10.0, 4..=32)
    ) {
        let gamma = vec![1.0_f32; v.len()];
        let beta = vec![0.0_f32; v.len()];
        let mut output = vec![0.0_f32; v.len()];
        layernorm_scalar(&v, &gamma, &beta, 1e-5, &mut output);
        let m = common::mean(&output);
        prop_assert!(
            m.abs() < 1e-4,
            "FALSIFY-LN-001 failed: mean of layernorm output = {m}, expected approx 0.0"
        );
    }

    /// FALSIFY-LN-002: Standardization
    /// Contract: layernorm-kernel-v1.yaml
    /// Prediction: variance(layernorm(x, gamma=1, beta=0)) approx 1 for len >= 4
    /// If fails: wrong variance computation in the normalization
    #[test]
    fn falsify_ln_002_standardization(
        v in proptest::collection::vec(-10.0_f32..10.0, 4..=32)
    ) {
        // Skip constant vectors (variance = 0 means output is all zeros)
        let vmin = v.iter().copied().fold(f32::INFINITY, f32::min);
        let vmax = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if (vmax - vmin).abs() < 1e-6 {
            return Ok(());
        }

        let gamma = vec![1.0_f32; v.len()];
        let beta = vec![0.0_f32; v.len()];
        let mut output = vec![0.0_f32; v.len()];
        layernorm_scalar(&v, &gamma, &beta, 1e-5, &mut output);
        let var = common::variance(&output);
        prop_assert!(
            (var - 1.0).abs() < 1e-3,
            "FALSIFY-LN-002 failed: variance of layernorm output = {var}, expected approx 1.0"
        );
    }

    /// FALSIFY-LN-004: SIMD equivalence
    /// Contract: layernorm-kernel-v1.yaml
    /// Prediction: avx2 and scalar produce results within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_ln_004_simd_equivalence(
        v in proptest::collection::vec(-10.0_f32..10.0, 4..=32)
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let gamma = vec![1.0_f32; v.len()];
        let beta = vec![0.0_f32; v.len()];
        let mut scalar_out = vec![0.0_f32; v.len()];
        let mut avx2_out = vec![0.0_f32; v.len()];
        layernorm_scalar(&v, &gamma, &beta, 1e-5, &mut scalar_out);
        unsafe { layernorm_avx2(&v, &gamma, &beta, 1e-5, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-LN-004 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }

    /// FALSIFY-LN-005: Idempotency
    /// Contract: layernorm-kernel-v1.yaml
    /// Prediction: layernorm(layernorm(x)) approx layernorm(x) (gamma=1, beta=0, len >= 4)
    /// If fails: normalization is not a stable fixed point
    #[test]
    fn falsify_ln_005_idempotency(
        v in proptest::collection::vec(-10.0_f32..10.0, 4..=32)
    ) {
        // Skip constant vectors
        let vmin = v.iter().copied().fold(f32::INFINITY, f32::min);
        let vmax = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if (vmax - vmin).abs() < 1e-6 {
            return Ok(());
        }

        let gamma = vec![1.0_f32; v.len()];
        let beta = vec![0.0_f32; v.len()];

        // First pass
        let mut first_pass = vec![0.0_f32; v.len()];
        layernorm_scalar(&v, &gamma, &beta, 1e-5, &mut first_pass);

        // Second pass
        let mut second_pass = vec![0.0_f32; v.len()];
        layernorm_scalar(&first_pass, &gamma, &beta, 1e-5, &mut second_pass);

        let dist = common::l2_distance(&first_pass, &second_pass);
        prop_assert!(
            dist < 1e-3,
            "FALSIFY-LN-005 failed: l2_distance between layernorm(x) and layernorm(layernorm(x)) = {dist}"
        );
    }

    /// FALSIFY-LN-006: Shift invariance
    /// Contract: layernorm-kernel-v1.yaml
    /// Prediction: layernorm(x + c) approx layernorm(x) (gamma=1, beta=0)
    /// If fails: mean subtraction is not working correctly
    #[test]
    fn falsify_ln_006_shift_invariance(
        v in proptest::collection::vec(-10.0_f32..10.0, 4..=32),
        c in -100.0_f32..100.0
    ) {
        let gamma = vec![1.0_f32; v.len()];
        let beta = vec![0.0_f32; v.len()];

        let mut out_original = vec![0.0_f32; v.len()];
        layernorm_scalar(&v, &gamma, &beta, 1e-5, &mut out_original);

        let shifted: Vec<f32> = v.iter().map(|&x| x + c).collect();
        let mut out_shifted = vec![0.0_f32; v.len()];
        layernorm_scalar(&shifted, &gamma, &beta, 1e-5, &mut out_shifted);

        for i in 0..v.len() {
            prop_assert!(
                (out_original[i] - out_shifted[i]).abs() < 1e-4,
                "FALSIFY-LN-006 failed: shift invariance violated at [{i}]: {} vs {}",
                out_original[i], out_shifted[i]
            );
        }
    }
}

/// FALSIFY-LN-003: Denominator safety
/// Contract: layernorm-kernel-v1.yaml
/// Prediction: layernorm on constant input does not produce NaN
/// If fails: division by zero when variance is zero (eps not applied)
#[test]
fn falsify_ln_003_denominator_safety() {
    for c in [-100.0_f32, 0.0, 42.0, 999.0] {
        for n in [1, 4, 8, 16] {
            let input = vec![c; n];
            let gamma = vec![1.0_f32; n];
            let beta = vec![0.0_f32; n];
            let mut output = vec![0.0_f32; n];
            layernorm_scalar(&input, &gamma, &beta, 1e-5, &mut output);
            common::assert_all_finite(&output);
        }
    }
}

/// FALSIFY-LN-007: Constant input
/// Contract: layernorm-kernel-v1.yaml
/// Prediction: layernorm on all-same input gives all-beta output
/// If fails: affine transform (beta shift) is not applied correctly
#[test]
fn falsify_ln_007_constant_input() {
    let n = 8;
    let c = 7.5_f32;
    let input = vec![c; n];
    let gamma = vec![1.0_f32; n];
    #[allow(clippy::approx_constant)]
    let beta_val = 3.14_f32;
    let beta = vec![beta_val; n];
    let mut output = vec![0.0_f32; n];
    layernorm_scalar(&input, &gamma, &beta, 1e-5, &mut output);
    for (i, &o) in output.iter().enumerate() {
        assert!(
            (o - beta_val).abs() < 1e-3,
            "FALSIFY-LN-007 failed: output[{i}] = {o}, expected approx {beta_val}"
        );
    }
}

// ============================================================================
// BatchNorm (6 tests: FALSIFY-BN-001 through FALSIFY-BN-006)
// ============================================================================

proptest! {
    /// FALSIFY-BN-001: Training standardization
    /// Contract: batchnorm-kernel-v1.yaml
    /// Prediction: per-channel mean approx 0 (with gamma=1, beta=0, training=true)
    /// If fails: batch mean computation or centering is incorrect
    #[test]
    fn falsify_bn_001_training_standardization(
        vals in proptest::collection::vec(-10.0_f32..10.0, 8..=8)
    ) {
        // n=4, c=2 => 8 elements
        let n = 4_usize;
        let c = 2_usize;
        let gamma = vec![1.0_f32; c];
        let beta = vec![0.0_f32; c];
        let mut running_mean = vec![0.0_f32; c];
        let mut running_var = vec![1.0_f32; c];
        let mut output = vec![0.0_f32; n * c];

        batchnorm_scalar(
            &vals, n, c, &gamma, &beta, 1e-5,
            &mut running_mean, &mut running_var,
            &mut output, 0.1, true,
        );

        // Check per-channel mean of the output is approx 0
        for ch in 0..c {
            let ch_sum: f32 = (0..n).map(|s| output[s * c + ch]).sum();
            #[allow(clippy::cast_precision_loss)]
            let ch_mean = ch_sum / n as f32;
            prop_assert!(
                ch_mean.abs() < 1e-4,
                "FALSIFY-BN-001 failed: channel {ch} output mean = {ch_mean}, expected approx 0.0"
            );
        }
    }

    /// FALSIFY-BN-004: SIMD equivalence
    /// Contract: batchnorm-kernel-v1.yaml
    /// Prediction: avx2 and scalar produce results within 8 ULP (training mode)
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_bn_004_simd_equivalence(
        vals in proptest::collection::vec(-10.0_f32..10.0, 8..=8)
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = 4_usize;
        let c = 2_usize;
        let gamma = vec![1.0_f32; c];
        let beta = vec![0.0_f32; c];

        let mut rm_scalar = vec![0.0_f32; c];
        let mut rv_scalar = vec![1.0_f32; c];
        let mut scalar_out = vec![0.0_f32; n * c];
        batchnorm_scalar(
            &vals, n, c, &gamma, &beta, 1e-5,
            &mut rm_scalar, &mut rv_scalar,
            &mut scalar_out, 0.1, true,
        );

        let mut rm_avx2 = vec![0.0_f32; c];
        let mut rv_avx2 = vec![1.0_f32; c];
        let mut avx2_out = vec![0.0_f32; n * c];
        unsafe {
            batchnorm_avx2(
                &vals, n, c, &gamma, &beta, 1e-5,
                &mut rm_avx2, &mut rv_avx2,
                &mut avx2_out, 0.1, true,
            );
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-BN-004 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-BN-002: Running variance update
/// Contract: batchnorm-kernel-v1.yaml
/// Prediction: after training step, `running_var` differs from initial
/// If fails: EMA update for running statistics is not implemented
#[test]
fn falsify_bn_002_running_variance_update() {
    let n = 4_usize;
    let c = 2_usize;
    let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0_f32; c];
    let beta = vec![0.0_f32; c];
    let initial_var = [1.0_f32, 1.0];
    let mut running_mean = vec![0.0_f32; c];
    let mut running_var = initial_var.to_vec();
    let mut output = vec![0.0_f32; n * c];

    batchnorm_scalar(
        &input,
        n,
        c,
        &gamma,
        &beta,
        1e-5,
        &mut running_mean,
        &mut running_var,
        &mut output,
        0.1,
        true,
    );

    let mut any_changed = false;
    for ch in 0..c {
        if (running_var[ch] - initial_var[ch]).abs() > 1e-10 {
            any_changed = true;
        }
    }
    assert!(
        any_changed,
        "FALSIFY-BN-002 failed: running_var was not updated after training step"
    );
}

/// FALSIFY-BN-003: Denominator safety
/// Contract: batchnorm-kernel-v1.yaml
/// Prediction: batchnorm on constant channel does not produce NaN
/// If fails: division by zero when per-channel variance is zero
#[test]
fn falsify_bn_003_denominator_safety() {
    let n = 4_usize;
    let c = 2_usize;
    // All values in channel 0 are 5.0, all values in channel 1 are 3.0
    let input = vec![5.0_f32, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0];
    let gamma = vec![1.0_f32; c];
    let beta = vec![0.0_f32; c];
    let mut running_mean = vec![0.0_f32; c];
    let mut running_var = vec![1.0_f32; c];
    let mut output = vec![0.0_f32; n * c];

    batchnorm_scalar(
        &input,
        n,
        c,
        &gamma,
        &beta,
        1e-5,
        &mut running_mean,
        &mut running_var,
        &mut output,
        0.1,
        true,
    );

    common::assert_all_finite(&output);
}

/// FALSIFY-BN-005: Eval vs train mode
/// Contract: batchnorm-kernel-v1.yaml
/// Prediction: eval mode uses running stats, produces different output than training on same input
/// If fails: mode flag is ignored or running stats are not used in eval mode
#[test]
fn falsify_bn_005_eval_vs_train_mode() {
    let n = 4_usize;
    let c = 2_usize;
    let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0_f32; c];
    let beta = vec![0.0_f32; c];

    // Training output
    let mut running_mean_train = vec![5.0_f32; c];
    let mut running_var_train = vec![2.0_f32; c];
    let mut train_out = vec![0.0_f32; n * c];
    batchnorm_scalar(
        &input,
        n,
        c,
        &gamma,
        &beta,
        1e-5,
        &mut running_mean_train,
        &mut running_var_train,
        &mut train_out,
        0.1,
        true,
    );

    // Eval output (with different running stats than batch stats)
    let mut running_mean_eval = vec![5.0_f32; c];
    let mut running_var_eval = vec![2.0_f32; c];
    let mut eval_out = vec![0.0_f32; n * c];
    batchnorm_scalar(
        &input,
        n,
        c,
        &gamma,
        &beta,
        1e-5,
        &mut running_mean_eval,
        &mut running_var_eval,
        &mut eval_out,
        0.1,
        false,
    );

    // They should differ since batch stats != running stats
    let mut any_differ = false;
    for i in 0..(n * c) {
        if (train_out[i] - eval_out[i]).abs() > 1e-6 {
            any_differ = true;
            break;
        }
    }
    assert!(
        any_differ,
        "FALSIFY-BN-005 failed: training and eval outputs are identical, \
         eval mode may not be using running stats"
    );
}

/// FALSIFY-BN-006: Running stats updated
/// Contract: batchnorm-kernel-v1.yaml
/// Prediction: after training, `running_mean` and `running_var` differ from zero initialization
/// If fails: EMA update is not being applied to running statistics
#[test]
fn falsify_bn_006_running_stats_updated() {
    let n = 4_usize;
    let c = 2_usize;
    let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0_f32; c];
    let beta = vec![0.0_f32; c];
    let mut running_mean = vec![0.0_f32; c];
    let mut running_var = vec![0.0_f32; c];
    let mut output = vec![0.0_f32; n * c];

    batchnorm_scalar(
        &input,
        n,
        c,
        &gamma,
        &beta,
        1e-5,
        &mut running_mean,
        &mut running_var,
        &mut output,
        0.1,
        true,
    );

    // After training, running_mean should be non-zero (input is non-zero)
    let mut mean_changed = false;
    for val in &running_mean[..c] {
        if val.abs() > 1e-10 {
            mean_changed = true;
            break;
        }
    }
    assert!(
        mean_changed,
        "FALSIFY-BN-006 failed: running_mean is still zero after training step"
    );

    // After training, running_var should be non-zero (input has variance)
    let mut var_changed = false;
    for val in &running_var[..c] {
        if val.abs() > 1e-10 {
            var_changed = true;
            break;
        }
    }
    assert!(
        var_changed,
        "FALSIFY-BN-006 failed: running_var is still zero after training step"
    );
}
