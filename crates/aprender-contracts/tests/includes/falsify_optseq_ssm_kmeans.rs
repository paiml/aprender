// ============================================================================
// SSM (6 tests: FALSIFY-SSM-001 through FALSIFY-SSM-006)
// ============================================================================

/// FALSIFY-SSM-001: Causality
/// Contract: ssm-kernel-v1.yaml
/// Prediction: changing x[t] doesn't affect output[0..t]
/// If fails: future inputs leak into past outputs (violates causality)
#[test]
fn falsify_ssm_001_causality() {
    let state_dim = 2;
    let seq_len = 4;
    let a_bar = [0.5_f32, 0.8];
    let b_bar = vec![1.0_f32; state_dim * seq_len];
    let c = [1.0_f32, 1.0];

    // Run with original input
    let x1 = [1.0_f32, 2.0, 3.0, 4.0];
    let mut out1 = vec![0.0_f32; seq_len];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x1, state_dim, seq_len, &mut out1);

    // Run with different last element
    let x2 = [1.0_f32, 2.0, 3.0, 99.0];
    let mut out2 = vec![0.0_f32; seq_len];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x2, state_dim, seq_len, &mut out2);

    // Outputs before the changed position must be identical
    for t in 0..3 {
        assert!(
            (out1[t] - out2[t]).abs() < 1e-7,
            "FALSIFY-SSM-001 failed: output[{t}] differs ({} vs {}) when only x[3] changed",
            out1[t],
            out2[t]
        );
    }
}

proptest! {
    /// FALSIFY-SSM-002: Scan linearity (with a_bar=0)
    /// Contract: ssm-kernel-v1.yaml
    /// Prediction: when a_bar=0 (no recurrence), scaling input scales output
    /// If fails: recurrence incorrectly mixes timesteps when state should not accumulate
    #[test]
    fn falsify_ssm_002_scan_linearity(
        x_v in proptest::collection::vec(0.1_f32..5.0, 4..=4),
        alpha in 0.1_f32..5.0,
    ) {
        let state_dim = 2;
        let seq_len = 4;
        let a_bar = [0.0_f32; 2]; // no recurrence
        let b_bar = vec![1.0_f32; state_dim * seq_len];
        let c = [1.0_f32, 1.0];

        // ssm(x)
        let mut out_x = vec![0.0_f32; seq_len];
        ssm_scan_scalar(&a_bar, &b_bar, &c, &x_v, state_dim, seq_len, &mut out_x);

        // ssm(alpha * x)
        let scaled: Vec<f32> = x_v.iter().map(|&x| x * alpha).collect();
        let mut out_ax = vec![0.0_f32; seq_len];
        ssm_scan_scalar(&a_bar, &b_bar, &c, &scaled, state_dim, seq_len, &mut out_ax);

        // alpha * ssm(x)
        for t in 0..seq_len {
            let expected = alpha * out_x[t];
            prop_assert!(
                (out_ax[t] - expected).abs() < 1e-3,
                "FALSIFY-SSM-002 failed: linearity violated at t={t}: ssm(alpha*x) = {}, alpha*ssm(x) = {}",
                out_ax[t], expected
            );
        }
    }

    /// FALSIFY-SSM-003: Output finiteness
    /// Contract: ssm-kernel-v1.yaml
    /// Prediction: output stays finite for moderate inputs
    /// If fails: numerical instability in scan computation
    #[test]
    fn falsify_ssm_003_output_finite(
        x_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        a_v in proptest::collection::vec(0.0_f32..0.99, 2..=2),
    ) {
        let state_dim = 2;
        let seq_len = 4;
        let b_bar = vec![1.0_f32; state_dim * seq_len];
        let c = [1.0_f32; 2];

        let mut output = vec![0.0_f32; seq_len];
        ssm_scan_scalar(&a_v, &b_bar, &c, &x_v, state_dim, seq_len, &mut output);

        common::assert_all_finite(&output);
    }
}

/// FALSIFY-SSM-004: Deterministic
/// Contract: ssm-kernel-v1.yaml
/// Prediction: same input produces same output (run twice, compare)
/// If fails: non-deterministic behavior in scan
#[test]
fn falsify_ssm_004_deterministic() {
    let state_dim = 2;
    let seq_len = 4;
    let a_bar = [0.5_f32, 0.8];
    let b_bar = vec![0.3_f32; state_dim * seq_len];
    let c = [1.0_f32, 0.5];
    let x = [1.0_f32, -0.5, 0.3, 2.0];

    let mut out1 = vec![0.0_f32; seq_len];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut out1);

    let mut out2 = vec![0.0_f32; seq_len];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut out2);

    for t in 0..seq_len {
        assert!(
            (out1[t] - out2[t]).abs() < 1e-10,
            "FALSIFY-SSM-004 failed: output[{t}] differs between runs: {} vs {}",
            out1[t],
            out2[t]
        );
    }
}

proptest! {
    /// FALSIFY-SSM-005: SIMD equivalence
    /// Contract: ssm-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_ssm_005_simd_equivalence(
        x_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let state_dim = 2;
        let seq_len = 4;
        let a_bar = [0.5_f32, 0.8];
        let b_bar = vec![1.0_f32; state_dim * seq_len];
        let c = [1.0_f32, 1.0];

        let mut scalar_out = vec![0.0_f32; seq_len];
        let mut avx2_out = vec![0.0_f32; seq_len];

        ssm_scan_scalar(&a_bar, &b_bar, &c, &x_v, state_dim, seq_len, &mut scalar_out);
        unsafe {
            ssm_scan_avx2(&a_bar, &b_bar, &c, &x_v, state_dim, seq_len, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-SSM-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-SSM-006: Zero input
/// Contract: ssm-kernel-v1.yaml
/// Prediction: ssm with x=all zeros produces all-zero output
/// If fails: scan produces non-zero output from zero input
#[test]
fn falsify_ssm_006_zero_input() {
    let state_dim = 2;
    let seq_len = 4;
    let a_bar = [0.9_f32, 0.8];
    let b_bar = vec![1.0_f32; state_dim * seq_len];
    let c = [1.0_f32, 1.0];
    let x = [0.0_f32; 4];
    let mut output = vec![0.0_f32; seq_len];

    ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut output);

    for (t, &val) in output.iter().enumerate().take(seq_len) {
        assert!(
            val.abs() < 1e-7,
            "FALSIFY-SSM-006 failed: output[{t}] = {val}, expected 0.0 for zero input"
        );
    }
}

// ============================================================================
// KMeans (6 tests: FALSIFY-KM-001 through FALSIFY-KM-006)
// ============================================================================

proptest! {
    /// FALSIFY-KM-001: Nearest assignment
    /// Contract: kmeans-kernel-v1.yaml
    /// Prediction: each point assigned to nearest centroid
    /// If fails: distance computation or argmin is incorrect
    #[test]
    fn falsify_km_001_nearest_assignment(
        points_v in proptest::collection::vec(-5.0_f32..5.0, 32..=32),
        centroids_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        let n = 8_usize;
        let k = 2_usize;
        let d = 4_usize;
        let mut assignments = vec![0_u32; n];

        kmeans_assign_scalar(&points_v, &centroids_v, n, k, d, &mut assignments);

        for p in 0..n {
            let assigned = assignments[p] as usize;
            let assigned_dist: f32 = (0..d)
                .map(|j| {
                    let diff = points_v[p * d + j] - centroids_v[assigned * d + j];
                    diff * diff
                })
                .sum();

            for c in 0..k {
                let dist: f32 = (0..d)
                    .map(|j| {
                        let diff = points_v[p * d + j] - centroids_v[c * d + j];
                        diff * diff
                    })
                    .sum();
                prop_assert!(
                    assigned_dist <= dist + 1e-6,
                    "FALSIFY-KM-001 failed: point {p} assigned to cluster {assigned} (dist={assigned_dist}) \
                     but cluster {c} is closer (dist={dist})"
                );
            }
        }
    }

    /// FALSIFY-KM-002: Valid indices
    /// Contract: kmeans-kernel-v1.yaml
    /// Prediction: all assignments in [0, k)
    /// If fails: assignment index out of bounds
    #[test]
    fn falsify_km_002_valid_indices(
        points_v in proptest::collection::vec(-5.0_f32..5.0, 32..=32),
        centroids_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        let n = 8_usize;
        let k = 2_usize;
        let d = 4_usize;
        let mut assignments = vec![0_u32; n];

        kmeans_assign_scalar(&points_v, &centroids_v, n, k, d, &mut assignments);

        for (p, &a) in assignments.iter().enumerate() {
            prop_assert!(
                (a as usize) < k,
                "FALSIFY-KM-002 failed: assignment[{p}] = {a}, must be < {k}"
            );
        }
    }
}

/// FALSIFY-KM-003: Objective decrease
/// Contract: kmeans-kernel-v1.yaml
/// Prediction: after one assign+update cycle, total distance doesn't increase
/// If fails: update step does not reduce or maintain objective
#[test]
fn falsify_km_003_objective_decrease() {
    let n = 8_usize;
    let k = 2_usize;
    let d = 2_usize;
    // Well-separated clusters
    let points = [
        1.0_f32, 1.0, 1.5, 1.5, 2.0, 2.0, 0.5, 0.5, 10.0, 10.0, 10.5, 10.5, 11.0, 11.0, 9.5, 9.5,
    ];
    // Start with suboptimal centroids
    let mut centroids = [0.0_f32, 0.0, 5.0, 5.0];
    let mut assignments = vec![0_u32; n];

    // Compute initial assignment and objective
    kmeans_assign_scalar(&points, &centroids, n, k, d, &mut assignments);
    let obj_before: f32 = (0..n)
        .map(|p| {
            let c = assignments[p] as usize;
            (0..d)
                .map(|j| {
                    let diff = points[p * d + j] - centroids[c * d + j];
                    diff * diff
                })
                .sum::<f32>()
        })
        .sum();

    // Update centroids
    kmeans_update_scalar(&points, &assignments, n, k, d, &mut centroids);

    // Re-assign after update
    kmeans_assign_scalar(&points, &centroids, n, k, d, &mut assignments);
    let obj_after: f32 = (0..n)
        .map(|p| {
            let c = assignments[p] as usize;
            (0..d)
                .map(|j| {
                    let diff = points[p * d + j] - centroids[c * d + j];
                    diff * diff
                })
                .sum::<f32>()
        })
        .sum();

    assert!(
        obj_after <= obj_before + 1e-4,
        "FALSIFY-KM-003 failed: objective increased from {obj_before} to {obj_after}"
    );
}

proptest! {
    /// FALSIFY-KM-004: Non-negative distances (implied by correct assignment)
    /// Contract: kmeans-kernel-v1.yaml
    /// Prediction: squared distances are non-negative
    /// If fails: distance computation produces negative values
    #[test]
    fn falsify_km_004_nonneg_distances(
        points_v in proptest::collection::vec(-5.0_f32..5.0, 16..=16),
        centroids_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        let n = 4_usize;
        let k = 2_usize;
        let d = 4_usize;
        let mut assignments = vec![0_u32; n];

        kmeans_assign_scalar(&points_v, &centroids_v, n, k, d, &mut assignments);

        // Verify that distances to assigned centroid are non-negative
        for p in 0..n {
            let c = assignments[p] as usize;
            let dist: f32 = (0..d)
                .map(|j| {
                    let diff = points_v[p * d + j] - centroids_v[c * d + j];
                    diff * diff
                })
                .sum();
            prop_assert!(
                dist >= 0.0,
                "FALSIFY-KM-004 failed: squared distance for point {p} is {dist}"
            );
        }
    }

    /// FALSIFY-KM-005: SIMD equivalence
    /// Contract: kmeans-kernel-v1.yaml
    /// Prediction: avx2 assign vs scalar produces same assignments
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_km_005_simd_equivalence(
        points_v in proptest::collection::vec(-5.0_f32..5.0, 16..=16),
        centroids_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = 4_usize;
        let k = 2_usize;
        let d = 4_usize;
        let mut asg_s = vec![0_u32; n];
        let mut asg_a = vec![0_u32; n];

        kmeans_assign_scalar(&points_v, &centroids_v, n, k, d, &mut asg_s);
        unsafe {
            kmeans_assign_avx2(&points_v, &centroids_v, n, k, d, &mut asg_a);
        }

        prop_assert_eq!(
            asg_s, asg_a,
            "FALSIFY-KM-005 failed: avx2 assignments differ from scalar"
        );
    }
}

/// FALSIFY-KM-006: Convergence
/// Contract: kmeans-kernel-v1.yaml
/// Prediction: after multiple assign+update cycles, centroids stabilize
/// If fails: algorithm does not converge on well-separated data
#[test]
fn falsify_km_006_convergence() {
    let n = 8_usize;
    let k = 2_usize;
    let d = 2_usize;
    let points = [
        0.0_f32, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 10.1, 9.9,
    ];
    let mut centroids = [1.0_f32, 1.0, 8.0, 8.0];
    let mut assignments = vec![0_u32; n];

    // Run 20 iterations
    for _ in 0..20 {
        kmeans_assign_scalar(&points, &centroids, n, k, d, &mut assignments);
        kmeans_update_scalar(&points, &assignments, n, k, d, &mut centroids);
    }

    let prev_centroids = centroids;
    kmeans_assign_scalar(&points, &centroids, n, k, d, &mut assignments);
    kmeans_update_scalar(&points, &assignments, n, k, d, &mut centroids);

    let change: f32 = centroids
        .iter()
        .zip(prev_centroids.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();

    assert!(
        change < 1e-4,
        "FALSIFY-KM-006 failed: centroids not converged after 20 iterations, change = {change}"
    );
}
