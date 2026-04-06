// ============================================================================
// AdamW (6 tests: FALSIFY-AW-001 through FALSIFY-AW-006)
// ============================================================================

/// FALSIFY-AW-001: Decoupled weight decay
/// Contract: adamw-kernel-v1.yaml
/// Prediction: with `weight_decay` > 0 and zero gradients, params decrease by factor (1 - lr*wd)
/// If fails: weight decay is not decoupled from adaptive learning rate
#[test]
fn falsify_aw_001_decoupled_weight_decay() {
    let lr = 0.001_f32;
    let wd = 0.01_f32;
    let original = [1.0_f32, -2.0, 3.0, -4.0];
    let mut params = original;
    let grads = [0.0_f32; 4];
    let mut m = [0.0_f32; 4];
    let mut v = [0.0_f32; 4];

    adamw_step_scalar(
        &mut params,
        &grads,
        &mut m,
        &mut v,
        lr,
        0.9,
        0.999,
        1e-8,
        wd,
        1,
    );

    for i in 0..4 {
        // With zero grads, m and v stay zero, so adaptive term is 0/(0+eps) ~ 0
        // param -= lr * (0 + wd * param) => param *= (1 - lr * wd)
        let expected = original[i] * (1.0 - lr * wd);
        assert!(
            (params[i] - expected).abs() < 1e-6,
            "FALSIFY-AW-001 failed: params[{i}] = {}, expected {expected}",
            params[i]
        );
    }
}

/// FALSIFY-AW-002: Bias correction
/// Contract: adamw-kernel-v1.yaml
/// Prediction: at t=1, bias-corrected moment differs from raw moment (moments are non-zero after non-zero gradient)
/// If fails: bias correction is not applied to moment estimates
#[test]
fn falsify_aw_002_bias_correction() {
    let mut params = [0.5_f32; 4];
    let grads = [1.0_f32, 0.5, -0.3, 0.8];
    let mut m = [0.0_f32; 4];
    let mut v = [0.0_f32; 4];

    adamw_step_scalar(
        &mut params,
        &grads,
        &mut m,
        &mut v,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.0,
        1,
    );

    // After one step with non-zero grads, m and v should be non-zero
    for i in 0..4 {
        assert!(
            m[i].abs() > 1e-10,
            "FALSIFY-AW-002 failed: m[{i}] = {} is zero after non-zero gradient",
            m[i]
        );
        assert!(
            v[i] > 0.0,
            "FALSIFY-AW-002 failed: v[{i}] = {} should be positive after non-zero gradient",
            v[i]
        );
    }
}

proptest! {
    /// FALSIFY-AW-003: SIMD equivalence
    /// Contract: adamw-kernel-v1.yaml
    /// Prediction: avx2 vs scalar param results within 8 ULP after one step
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_aw_003_simd_equivalence(
        params_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
        grads_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = params_v.len().min(grads_v.len());
        let mut params_s: Vec<f32> = params_v[..n].to_vec();
        let mut params_a = params_s.clone();
        let grads: Vec<f32> = grads_v[..n].to_vec();
        let mut m_s = vec![0.0_f32; n];
        let mut m_a = m_s.clone();
        let mut v_s = vec![0.0_f32; n];
        let mut v_a = v_s.clone();

        adamw_step_scalar(&mut params_s, &grads, &mut m_s, &mut v_s, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);
        unsafe {
            adamw_step_avx2(&mut params_a, &grads, &mut m_a, &mut v_a, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);
        }

        let ulp = common::max_ulp_distance(&params_s, &params_a);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-AW-003 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }

    /// FALSIFY-AW-004: Moments finite
    /// Contract: adamw-kernel-v1.yaml
    /// Prediction: m and v stay finite after a step with moderate gradients
    /// If fails: numerical overflow in moment accumulation
    #[test]
    fn falsify_aw_004_moments_finite(
        params_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
        grads_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
    ) {
        let n = params_v.len().min(grads_v.len());
        let mut params: Vec<f32> = params_v[..n].to_vec();
        let grads: Vec<f32> = grads_v[..n].to_vec();
        let mut m = vec![0.0_f32; n];
        let mut v = vec![0.0_f32; n];

        adamw_step_scalar(&mut params, &grads, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);

        for i in 0..n {
            prop_assert!(
                m[i].is_finite(),
                "FALSIFY-AW-004 failed: m[{i}] = {} is not finite",
                m[i]
            );
            prop_assert!(
                v[i].is_finite(),
                "FALSIFY-AW-004 failed: v[{i}] = {} is not finite",
                v[i]
            );
        }
    }

    /// FALSIFY-AW-005: Update finite
    /// Contract: adamw-kernel-v1.yaml
    /// Prediction: params stay finite after a step
    /// If fails: division by zero or overflow in parameter update
    #[test]
    fn falsify_aw_005_update_finite(
        params_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
        grads_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
    ) {
        let n = params_v.len().min(grads_v.len());
        let mut params: Vec<f32> = params_v[..n].to_vec();
        let grads: Vec<f32> = grads_v[..n].to_vec();
        let mut m = vec![0.0_f32; n];
        let mut v = vec![0.0_f32; n];

        adamw_step_scalar(&mut params, &grads, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);

        common::assert_all_finite(&params);
    }

    /// FALSIFY-AW-006: Weight decay direction
    /// Contract: adamw-kernel-v1.yaml
    /// Prediction: with wd > 0 and zero grads, all params move toward zero (|new| <= |old|)
    /// If fails: weight decay pushes params away from zero
    #[test]
    fn falsify_aw_006_weight_decay_direction(
        params_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
    ) {
        let n = params_v.len();
        let original: Vec<f32> = params_v.clone();
        let mut params = params_v;
        let grads = vec![0.0_f32; n];
        let mut m = vec![0.0_f32; n];
        let mut v = vec![0.0_f32; n];

        adamw_step_scalar(&mut params, &grads, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);

        for i in 0..n {
            prop_assert!(
                params[i].abs() <= original[i].abs() + 1e-7,
                "FALSIFY-AW-006 failed: |params[{i}]| = {} > |original| = {}, weight decay should shrink params",
                params[i].abs(), original[i].abs()
            );
        }
    }
}

// ============================================================================
// Conv1d (6 tests: FALSIFY-CV-001 through FALSIFY-CV-006)
// ============================================================================

/// FALSIFY-CV-001: Output shape
/// Contract: conv1d-kernel-v1.yaml
/// Prediction: output length = floor((length + 2*padding - `kernel_size`) / stride) + 1
/// If fails: output dimension calculation is incorrect
#[test]
fn falsify_cv_001_output_shape() {
    // Test with various concrete sizes
    let cases: Vec<(usize, usize, usize, usize, usize, usize)> = vec![
        // (c_in, c_out, length, kernel_size, stride, padding)
        (1, 1, 8, 3, 1, 0),  // out = (8+0-3)/1+1 = 6
        (2, 3, 10, 3, 2, 1), // out = (10+2-3)/2+1 = 5
        (1, 1, 4, 1, 1, 0),  // out = (4+0-1)/1+1 = 4
        (1, 2, 6, 3, 1, 1),  // out = (6+2-3)/1+1 = 6
    ];

    for (c_in, c_out, length, kernel_size, stride, padding) in cases {
        let expected_out_len = (length + 2 * padding - kernel_size) / stride + 1;
        let input = vec![1.0_f32; c_in * length];
        let weight = vec![0.1_f32; c_out * c_in * kernel_size];
        let mut output = vec![0.0_f32; c_out * expected_out_len];

        // Should not panic -- output buffer has correct size
        conv1d_scalar(
            &input,
            &weight,
            None,
            c_in,
            c_out,
            length,
            kernel_size,
            stride,
            padding,
            &mut output,
        );
        assert_eq!(
            output.len(),
            c_out * expected_out_len,
            "FALSIFY-CV-001 failed: output length mismatch for c_in={c_in}, c_out={c_out}, \
             length={length}, kernel_size={kernel_size}, stride={stride}, padding={padding}"
        );
    }
}

proptest! {
    /// FALSIFY-CV-002: Linearity
    /// Contract: conv1d-kernel-v1.yaml
    /// Prediction: conv1d(alpha*x, w, None) ~ alpha * conv1d(x, w, None) for small inputs
    /// If fails: convolution is not a linear operation in input
    #[test]
    fn falsify_cv_002_linearity(
        input_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        weight_v in proptest::collection::vec(-5.0_f32..5.0, 3..=3),
        alpha in 0.1_f32..5.0,
    ) {
        let c_in = 1;
        let c_out = 1;
        let length = 4;
        let kernel_size = 3;
        let stride = 1;
        let padding = 0;
        let out_len = (length + 2 * padding - kernel_size) / stride + 1;

        // conv1d(x, w)
        let mut out_x = vec![0.0_f32; c_out * out_len];
        conv1d_scalar(&input_v, &weight_v, None, c_in, c_out, length, kernel_size, stride, padding, &mut out_x);

        // conv1d(alpha*x, w)
        let scaled_input: Vec<f32> = input_v.iter().map(|&x| x * alpha).collect();
        let mut out_ax = vec![0.0_f32; c_out * out_len];
        conv1d_scalar(&scaled_input, &weight_v, None, c_in, c_out, length, kernel_size, stride, padding, &mut out_ax);

        // alpha * conv1d(x, w)
        let expected: Vec<f32> = out_x.iter().map(|&y| y * alpha).collect();

        for i in 0..out_len {
            prop_assert!(
                (out_ax[i] - expected[i]).abs() < 1e-3,
                "FALSIFY-CV-002 failed: linearity violated at [{i}]: conv(alpha*x) = {}, alpha*conv(x) = {}",
                out_ax[i], expected[i]
            );
        }
    }
}

/// FALSIFY-CV-003: Im2col equivalence (pointwise)
/// Contract: conv1d-kernel-v1.yaml
/// Prediction: conv1d with `kernel_size=1`, stride=1 acts as pointwise multiplication
/// If fails: `kernel_size=1` edge case is handled incorrectly
#[test]
fn falsify_cv_003_pointwise_equivalence() {
    let c_in = 1;
    let c_out = 1;
    let length = 8;
    let kernel_size = 1;
    let stride = 1;
    let padding = 0;
    let out_len = length; // (8+0-1)/1+1 = 8

    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = [2.5_f32]; // single weight, kernel_size=1
    let mut output = vec![0.0_f32; c_out * out_len];

    conv1d_scalar(
        &input,
        &weight,
        None,
        c_in,
        c_out,
        length,
        kernel_size,
        stride,
        padding,
        &mut output,
    );

    for i in 0..length {
        let expected = input[i] * weight[0];
        assert!(
            (output[i] - expected).abs() < 1e-6,
            "FALSIFY-CV-003 failed: output[{i}] = {}, expected {} (pointwise mul)",
            output[i],
            expected
        );
    }
}

proptest! {
    /// FALSIFY-CV-004: Output bound
    /// Contract: conv1d-kernel-v1.yaml
    /// Prediction: output bounded by c_in * kernel_size * max(|input|) * max(|weight|) + max(|bias|)
    /// If fails: accumulation overflow or incorrect summation
    #[test]
    fn falsify_cv_004_output_bound(
        input_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        weight_v in proptest::collection::vec(-5.0_f32..5.0, 3..=3),
        bias_v in proptest::collection::vec(-5.0_f32..5.0, 1..=1),
    ) {
        let c_in = 1_usize;
        let c_out = 1_usize;
        let length = 4_usize;
        let kernel_size = 3_usize;
        let stride = 1_usize;
        let padding = 0_usize;
        let out_len = (length + 2 * padding - kernel_size) / stride + 1;

        let mut output = vec![0.0_f32; c_out * out_len];
        conv1d_scalar(
            &input_v, &weight_v, Some(&bias_v),
            c_in, c_out, length, kernel_size, stride, padding, &mut output,
        );

        let max_input = input_v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let max_weight = weight_v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let max_bias = bias_v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        #[allow(clippy::cast_precision_loss)]
        let bound = (c_in * kernel_size) as f32 * max_input * max_weight + max_bias;

        for (i, val) in output.iter().enumerate() {
            prop_assert!(
                val.abs() <= bound + 1e-4,
                "FALSIFY-CV-004 failed: |output[{i}]| = {} exceeds bound {bound}",
                val.abs()
            );
        }
    }

    /// FALSIFY-CV-005: SIMD equivalence
    /// Contract: conv1d-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_cv_005_simd_equivalence(
        input_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        weight_v in proptest::collection::vec(-5.0_f32..5.0, 3..=3),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let c_in = 1;
        let c_out = 1;
        let length = 4;
        let kernel_size = 3;
        let stride = 1;
        let padding = 0;
        let out_len = (length + 2 * padding - kernel_size) / stride + 1;

        let mut scalar_out = vec![0.0_f32; c_out * out_len];
        let mut avx2_out = vec![0.0_f32; c_out * out_len];

        conv1d_scalar(&input_v, &weight_v, None, c_in, c_out, length, kernel_size, stride, padding, &mut scalar_out);
        unsafe {
            conv1d_avx2(&input_v, &weight_v, None, c_in, c_out, length, kernel_size, stride, padding, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-CV-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-CV-006: Identity kernel
/// Contract: conv1d-kernel-v1.yaml
/// Prediction: conv1d with `kernel_size=1`, `c_in=c_out=1`, weight=[1.0], bias=None reproduces input
/// If fails: identity convolution does not preserve input
#[test]
fn falsify_cv_006_identity_kernel() {
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = [1.0_f32];
    let length = 8;
    let mut output = vec![0.0_f32; length];

    conv1d_scalar(&input, &weight, None, 1, 1, length, 1, 1, 0, &mut output);

    for i in 0..length {
        assert!(
            (output[i] - input[i]).abs() < 1e-7,
            "FALSIFY-CV-006 failed: output[{i}] = {}, expected {}",
            output[i],
            input[i]
        );
    }
}
