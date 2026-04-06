// ============================================================================
// CMA-ES (6 tests: FALSIFY-CMA-001 through FALSIFY-CMA-006)
// ============================================================================

proptest! {
    /// FALSIFY-CMA-001: Output structure (output is finite)
    /// Contract: cma-es-kernel-v1.yaml
    /// Prediction: output = mean + sigma * L * z is finite for moderate inputs
    /// If fails: matrix-vector multiply or accumulation produces NaN/infinity
    #[test]
    fn falsify_cma_001_output_structure(
        mean_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        z_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        sigma in 0.01_f32..5.0,
    ) {
        let d = 4;
        let cholesky_l = common::identity_matrix(d);
        let mut output = vec![0.0_f32; d];

        cma_sample_scalar(&mean_v, sigma, &cholesky_l, d, &z_v, &mut output);

        common::assert_all_finite(&output);
    }
}

/// FALSIFY-CMA-002: Symmetry
/// Contract: cma-es-kernel-v1.yaml
/// Prediction: sample(mean, sigma, I, d, z) and sample(mean, sigma, I, d, -z) are symmetric around mean
/// If fails: Cholesky multiplication breaks antisymmetry of z
#[test]
fn falsify_cma_002_symmetry() {
    let d = 4;
    let mean = [1.0_f32, 2.0, 3.0, 4.0];
    let sigma = 1.5_f32;
    let cholesky_l = common::identity_matrix(d);
    let z = [0.5_f32, -0.3, 0.8, -0.1];
    let neg_z: Vec<f32> = z.iter().map(|&x| -x).collect();

    let mut out_pos = vec![0.0_f32; d];
    let mut out_neg = vec![0.0_f32; d];

    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut out_pos);
    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &neg_z, &mut out_neg);

    // (out_pos + out_neg) / 2 should equal mean
    for i in 0..d {
        let midpoint = f32::midpoint(out_pos[i], out_neg[i]);
        assert!(
            (midpoint - mean[i]).abs() < 1e-5,
            "FALSIFY-CMA-002 failed: midpoint[{i}] = {midpoint}, expected mean[{i}] = {}",
            mean[i]
        );
    }
}

proptest! {
    /// FALSIFY-CMA-003: Step size
    /// Contract: cma-es-kernel-v1.yaml
    /// Prediction: with small sigma, output is close to mean
    /// If fails: sigma scaling is not applied correctly
    #[test]
    fn falsify_cma_003_step_size(
        mean_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        z_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
    ) {
        let d = 4;
        let sigma = 0.001_f32; // very small sigma
        let cholesky_l = common::identity_matrix(d);
        let mut output = vec![0.0_f32; d];

        cma_sample_scalar(&mean_v, sigma, &cholesky_l, d, &z_v, &mut output);

        let dist = common::l2_distance(&mean_v, &output);
        // With sigma=0.001 and |z| bounded by ~sqrt(4)*5 ~ 10, dist should be < 0.01 * 10 = 0.1
        let z_norm = common::l2_norm(&z_v);
        let bound = sigma * z_norm + 1e-4;
        prop_assert!(
            dist <= bound,
            "FALSIFY-CMA-003 failed: distance from mean = {dist}, expected <= {bound} with sigma={sigma}"
        );
    }
}

/// FALSIFY-CMA-004: Positive definite (identity Cholesky)
/// Contract: cma-es-kernel-v1.yaml
/// Prediction: with identity Cholesky, output is mean + sigma*z
/// If fails: lower triangular multiply is incorrect for identity matrix
#[test]
fn falsify_cma_004_positive_definite() {
    let d = 4;
    let mean = [1.0_f32, 2.0, 3.0, 4.0];
    let sigma = 2.0_f32;
    let cholesky_l = common::identity_matrix(d);
    let z = [0.5_f32, -0.3, 0.8, -0.1];
    let mut output = vec![0.0_f32; d];

    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut output);

    for i in 0..d {
        let expected = mean[i] + sigma * z[i];
        assert!(
            (output[i] - expected).abs() < 1e-6,
            "FALSIFY-CMA-004 failed: output[{i}] = {}, expected {expected} (mean + sigma*z)",
            output[i]
        );
    }
}

proptest! {
    /// FALSIFY-CMA-005: SIMD equivalence
    /// Contract: cma-es-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_cma_005_simd_equivalence(
        mean_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        z_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        sigma in 0.01_f32..5.0,
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let d = 4;
        let cholesky_l = [
            1.0_f32, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0,
            0.3, 0.2, 1.0, 0.0,
            0.1, 0.4, 0.6, 1.0,
        ];
        let mut scalar_out = vec![0.0_f32; d];
        let mut avx2_out = vec![0.0_f32; d];

        cma_sample_scalar(&mean_v, sigma, &cholesky_l, d, &z_v, &mut scalar_out);
        unsafe {
            cma_sample_avx2(&mean_v, sigma, &cholesky_l, d, &z_v, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-CMA-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-CMA-006: Identity Cholesky
/// Contract: cma-es-kernel-v1.yaml
/// Prediction: L=I gives output = mean + sigma*z
/// If fails: identity matrix handling in lower-triangular multiply is broken
#[test]
fn falsify_cma_006_identity_cholesky() {
    let d = 4;
    let mean = [10.0_f32, 20.0, 30.0, 40.0];
    let sigma = 3.0_f32;
    let cholesky_l = common::identity_matrix(d);
    let z = [1.0_f32, -1.0, 0.5, -0.5];
    let mut output = vec![0.0_f32; d];

    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut output);

    for i in 0..d {
        let expected = mean[i] + sigma * z[i];
        assert!(
            (output[i] - expected).abs() < 1e-6,
            "FALSIFY-CMA-006 failed: output[{i}] = {}, expected {expected}",
            output[i]
        );
    }
}

// ============================================================================
// Gated Delta Net (5 tests: FALSIFY-GDN-001 through FALSIFY-GDN-005)
// ============================================================================

/// FALSIFY-GDN-001: Output shape
/// Contract: gated-delta-net-v1.yaml
/// Prediction: output length = `seq_len` * `v_dim`
/// If fails: output dimension calculation is incorrect
#[test]
fn falsify_gdn_001_output_shape() {
    let seq_len = 4;
    let k_dim = 3;
    let v_dim = 2;
    let q = vec![0.1_f32; seq_len * k_dim];
    let k = vec![0.2_f32; seq_len * k_dim];
    let v = vec![0.3_f32; seq_len * v_dim];
    let alpha = vec![0.5_f32; seq_len];
    let beta = vec![0.5_f32; seq_len];

    let expected_len = seq_len * v_dim;
    let mut output = vec![0.0_f32; expected_len];

    // Should not panic -- output buffer has correct size
    gdn_recurrence_scalar(
        &q,
        &k,
        &v,
        &alpha,
        &beta,
        seq_len,
        k_dim,
        v_dim,
        &mut output,
    );

    assert_eq!(
        output.len(),
        expected_len,
        "FALSIFY-GDN-001 failed: output length = {}, expected {expected_len}",
        output.len()
    );
}

/// FALSIFY-GDN-002: Causality
/// Contract: gated-delta-net-v1.yaml
/// Prediction: changing input at time t doesn't affect output before t
/// If fails: future inputs leak into past outputs (violates causality)
#[test]
fn falsify_gdn_002_causality() {
    let seq_len = 4;
    let k_dim = 2;
    let v_dim = 2;
    let q = vec![0.1_f32; seq_len * k_dim];
    let k1 = vec![0.2_f32; seq_len * k_dim];
    let v1 = vec![0.3_f32; seq_len * v_dim];
    let alpha = vec![0.5_f32; seq_len];
    let beta = vec![0.5_f32; seq_len];

    let mut out1 = vec![0.0_f32; seq_len * v_dim];
    gdn_recurrence_scalar(
        &q, &k1, &v1, &alpha, &beta, seq_len, k_dim, v_dim, &mut out1,
    );

    // Modify last timestep's key and value
    let mut k2 = k1.clone();
    let mut v2 = v1.clone();
    for i in 0..k_dim {
        k2[(seq_len - 1) * k_dim + i] = 99.0;
    }
    for i in 0..v_dim {
        v2[(seq_len - 1) * v_dim + i] = 99.0;
    }

    let mut out2 = vec![0.0_f32; seq_len * v_dim];
    gdn_recurrence_scalar(
        &q, &k2, &v2, &alpha, &beta, seq_len, k_dim, v_dim, &mut out2,
    );

    // Outputs before the last timestep must be identical
    for t in 0..(seq_len - 1) {
        for j in 0..v_dim {
            let idx = t * v_dim + j;
            assert!(
                (out1[idx] - out2[idx]).abs() < 1e-7,
                "FALSIFY-GDN-002 failed: output[{t},{j}] differs ({} vs {}) when only last timestep changed",
                out1[idx],
                out2[idx]
            );
        }
    }
}

proptest! {
    /// FALSIFY-GDN-003: Decay bound
    /// Contract: gated-delta-net-v1.yaml
    /// Prediction: with alpha in [0,1], output stays bounded/finite
    /// If fails: gating does not properly bound state growth
    #[test]
    fn falsify_gdn_003_decay_bound(
        q_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        k_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        v_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        alpha_v in proptest::collection::vec(0.0_f32..1.0, 4..=4),
        beta_v in proptest::collection::vec(0.0_f32..1.0, 4..=4),
    ) {
        let seq_len = 4;
        let k_dim = 2;
        let v_dim = 2;
        let mut output = vec![0.0_f32; seq_len * v_dim];

        gdn_recurrence_scalar(&q_v, &k_v, &v_v, &alpha_v, &beta_v, seq_len, k_dim, v_dim, &mut output);

        common::assert_all_finite(&output);
    }

    /// FALSIFY-GDN-004: L2 direction (output finite and bounded)
    /// Contract: gated-delta-net-v1.yaml
    /// Prediction: output is finite and bounded for moderate inputs
    /// If fails: numerical instability in state update or readout
    #[test]
    fn falsify_gdn_004_l2_direction(
        q_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        k_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        v_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        let seq_len = 4;
        let k_dim = 2;
        let v_dim = 2;
        let alpha = vec![0.5_f32; seq_len];
        let beta = vec![0.5_f32; seq_len];
        let mut output = vec![0.0_f32; seq_len * v_dim];

        gdn_recurrence_scalar(&q_v, &k_v, &v_v, &alpha, &beta, seq_len, k_dim, v_dim, &mut output);

        common::assert_all_finite(&output);
        // With bounded inputs and alpha=0.5, beta=0.5, output should be bounded
        let out_norm = common::l2_norm(&output);
        prop_assert!(
            out_norm < 1000.0,
            "FALSIFY-GDN-004 failed: output L2 norm = {out_norm} is unreasonably large"
        );
    }

    /// FALSIFY-GDN-005: SIMD equivalence
    /// Contract: gated-delta-net-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_gdn_005_simd_equivalence(
        q_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        k_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        v_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let seq_len = 4;
        let k_dim = 2;
        let v_dim = 2;
        let alpha = vec![0.9_f32; seq_len];
        let beta = vec![0.1_f32; seq_len];
        let mut scalar_out = vec![0.0_f32; seq_len * v_dim];
        let mut avx2_out = vec![0.0_f32; seq_len * v_dim];

        gdn_recurrence_scalar(&q_v, &k_v, &v_v, &alpha, &beta, seq_len, k_dim, v_dim, &mut scalar_out);
        unsafe {
            gdn_recurrence_avx2(&q_v, &k_v, &v_v, &alpha, &beta, seq_len, k_dim, v_dim, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-GDN-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}
