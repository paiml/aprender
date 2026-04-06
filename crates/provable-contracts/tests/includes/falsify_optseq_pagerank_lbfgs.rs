// ============================================================================
// PageRank (6 tests: FALSIFY-PR-001 through FALSIFY-PR-006)
// ============================================================================

/// Helper: generate a row-stochastic matrix of size n x n.
#[allow(clippy::cast_precision_loss)]
fn row_stochastic_matrix(n: usize) -> Vec<f32> {
    let mut mat = vec![0.0_f32; n * n];
    for i in 0..n {
        let mut sum = 0.0_f32;
        for j in 0..n {
            let val = ((i * 7 + j * 13 + 3) % 11) as f32 + 1.0;
            mat[i * n + j] = val;
            sum += val;
        }
        for j in 0..n {
            mat[i * n + j] /= sum;
        }
    }
    mat
}

proptest! {
    /// FALSIFY-PR-001: Output distribution
    /// Contract: pagerank-kernel-v1.yaml
    /// Prediction: output sums to ~1 and all >= 0 with row-stochastic transition and uniform rank
    /// If fails: pagerank iteration does not preserve distribution property
    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn falsify_pr_001_output_distribution(n in 2_usize..8) {
        let transition = row_stochastic_matrix(n);
        let rank = vec![1.0 / n as f32; n];
        let mut output = vec![0.0_f32; n];

        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

        let sum: f32 = output.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "FALSIFY-PR-001 failed: output sums to {sum}, expected ~1.0"
        );
        for (i, &o) in output.iter().enumerate() {
            prop_assert!(
                o >= 0.0,
                "FALSIFY-PR-001 failed: output[{i}] = {o} is negative"
            );
        }
    }
}

/// FALSIFY-PR-002: Input normalization
/// Contract: pagerank-kernel-v1.yaml
/// Prediction: with row-stochastic transition matrix and uniform rank, output is valid distribution
/// If fails: transition matrix handling is incorrect
#[test]
#[allow(clippy::cast_precision_loss)]
fn falsify_pr_002_input_normalization() {
    let n = 4;
    let transition = row_stochastic_matrix(n);
    let rank = vec![1.0 / n as f32; n];
    let mut output = vec![0.0_f32; n];

    pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

    common::assert_all_finite(&output);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "FALSIFY-PR-002 failed: output sums to {sum}, expected ~1.0"
    );
    for (i, &o) in output.iter().enumerate() {
        assert!(
            o >= 0.0,
            "FALSIFY-PR-002 failed: output[{i}] = {o} is negative"
        );
    }
}

/// FALSIFY-PR-003: Convergence
/// Contract: pagerank-kernel-v1.yaml
/// Prediction: repeated iteration converges (output changes decrease)
/// If fails: iteration does not converge to stationary distribution
#[test]
#[allow(clippy::cast_precision_loss)]
fn falsify_pr_003_convergence() {
    let n = 4;
    let transition = row_stochastic_matrix(n);
    let mut rank = vec![1.0 / n as f32; n];
    let mut output = vec![0.0_f32; n];

    let mut prev_change = f32::MAX;
    for _ in 0..50 {
        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);
        let change: f32 = rank
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        // Change should generally decrease (may not be strictly monotone, so check final)
        prev_change = change;
        rank.copy_from_slice(&output);
    }

    assert!(
        prev_change < 1e-4,
        "FALSIFY-PR-003 failed: not converged after 50 iterations, last change = {prev_change}"
    );
}

proptest! {
    /// FALSIFY-PR-004: Non-negative
    /// Contract: pagerank-kernel-v1.yaml
    /// Prediction: all output ranks >= 0
    /// If fails: teleport or damping term produces negative values
    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn falsify_pr_004_non_negative(n in 2_usize..8) {
        let transition = row_stochastic_matrix(n);
        let rank = vec![1.0 / n as f32; n];
        let mut output = vec![0.0_f32; n];

        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

        for (i, &o) in output.iter().enumerate() {
            prop_assert!(
                o >= 0.0,
                "FALSIFY-PR-004 failed: output[{i}] = {o} is negative"
            );
        }
    }

    /// FALSIFY-PR-005: SIMD equivalence
    /// Contract: pagerank-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[allow(clippy::cast_precision_loss)]
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_pr_005_simd_equivalence(n in 2_usize..8) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let transition = row_stochastic_matrix(n);
        let rank = vec![1.0 / n as f32; n];
        let mut scalar_out = vec![0.0_f32; n];
        let mut avx2_out = vec![0.0_f32; n];

        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut scalar_out);
        unsafe {
            pagerank_iterate_avx2(&transition, &rank, n, 0.85, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-PR-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-PR-006: Uniform stationary
/// Contract: pagerank-kernel-v1.yaml
/// Prediction: uniform transition matrix with uniform rank gives uniform output
/// If fails: symmetry is broken by implementation
#[test]
#[allow(clippy::cast_precision_loss)]
fn falsify_pr_006_uniform_stationary() {
    let n = 4;
    // Uniform row-stochastic matrix
    let val = 1.0 / n as f32;
    let transition = vec![val; n * n];
    let rank = vec![val; n];
    let mut output = vec![0.0_f32; n];

    pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

    let expected = 1.0 / n as f32;
    for (i, &o) in output.iter().enumerate() {
        assert!(
            (o - expected).abs() < 1e-5,
            "FALSIFY-PR-006 failed: output[{i}] = {o}, expected {expected} (uniform stationary)"
        );
    }
}

// ============================================================================
// LBFGS (6 tests: FALSIFY-LB-001 through FALSIFY-LB-006)
// ============================================================================

proptest! {
    /// FALSIFY-LB-001: Descent direction
    /// Contract: lbfgs-kernel-v1.yaml
    /// Prediction: dot(direction, gradient) < 0 (direction opposes gradient)
    /// If fails: L-BFGS direction is not a descent direction
    #[test]
    fn falsify_lb_001_descent_direction(
        gradient_v in proptest::collection::vec(0.1_f32..5.0, 4..=4),
    ) {
        let d = 4;
        // Create valid curvature pair: s and y with s.y > 0
        let s_history = [0.1_f32, 0.2, 0.3, 0.4];
        let y_history = [0.5_f32, 0.6, 0.7, 0.8];
        // s.y = 0.05 + 0.12 + 0.21 + 0.32 = 0.7 > 0
        let mut direction = vec![0.0_f32; d];

        lbfgs_direction_scalar(&gradient_v, &s_history, &y_history, 1, d, &mut direction);

        let dot: f32 = gradient_v
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();
        prop_assert!(
            dot < 0.0,
            "FALSIFY-LB-001 failed: dot(direction, gradient) = {dot}, must be < 0 for descent"
        );
    }
}

/// FALSIFY-LB-002: Curvature condition
/// Contract: lbfgs-kernel-v1.yaml
/// Prediction: with s and y satisfying s.y > 0, direction is well-defined and finite
/// If fails: two-loop recursion produces NaN or infinity with valid curvature
#[test]
fn falsify_lb_002_curvature_condition() {
    let d = 4;
    let gradient = [1.0_f32, 2.0, 3.0, 4.0];
    let s_history = [0.1_f32, 0.2, 0.3, 0.4];
    let y_history = [0.5_f32, 0.6, 0.7, 0.8];
    let mut direction = vec![0.0_f32; d];

    lbfgs_direction_scalar(&gradient, &s_history, &y_history, 1, d, &mut direction);

    common::assert_all_finite(&direction);
}

/// FALSIFY-LB-003: History bound (m=0 gives steepest descent)
/// Contract: lbfgs-kernel-v1.yaml
/// Prediction: m=0 (no history) gives direction = -gradient
/// If fails: steepest descent fallback is incorrect
#[test]
fn falsify_lb_003_history_bound() {
    let d = 4;
    let gradient = [1.0_f32, -2.0, 3.0, -4.0];
    let s_history: [f32; 0] = [];
    let y_history: [f32; 0] = [];
    let mut direction = vec![0.0_f32; d];

    lbfgs_direction_scalar(&gradient, &s_history, &y_history, 0, d, &mut direction);

    for i in 0..d {
        assert!(
            (direction[i] - (-gradient[i])).abs() < 1e-7,
            "FALSIFY-LB-003 failed: direction[{i}] = {}, expected {} (-gradient)",
            direction[i],
            -gradient[i]
        );
    }
}

proptest! {
    /// FALSIFY-LB-004: Objective decrease (direction is finite and descending)
    /// Contract: lbfgs-kernel-v1.yaml
    /// Prediction: direction is finite and dot with gradient < 0
    /// If fails: L-BFGS direction has numerical issues
    #[test]
    fn falsify_lb_004_objective_decrease(
        gradient_v in proptest::collection::vec(0.1_f32..5.0, 4..=4),
    ) {
        let d = 4;
        let s_history = [0.2_f32, 0.3, 0.1, 0.4];
        let y_history = [0.4_f32, 0.5, 0.3, 0.6];
        let mut direction = vec![0.0_f32; d];

        lbfgs_direction_scalar(&gradient_v, &s_history, &y_history, 1, d, &mut direction);

        common::assert_all_finite(&direction);
        let dot: f32 = gradient_v
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();
        prop_assert!(
            dot < 0.0,
            "FALSIFY-LB-004 failed: dot(direction, gradient) = {dot}, must be < 0"
        );
    }

    /// FALSIFY-LB-005: SIMD equivalence
    /// Contract: lbfgs-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_lb_005_simd_equivalence(
        gradient_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let d = 4;
        let s_history = [0.1_f32, 0.2, 0.3, 0.4];
        let y_history = [0.5_f32, 0.6, 0.7, 0.8];
        let mut scalar_dir = vec![0.0_f32; d];
        let mut avx2_dir = vec![0.0_f32; d];

        lbfgs_direction_scalar(&gradient_v, &s_history, &y_history, 1, d, &mut scalar_dir);
        unsafe {
            lbfgs_direction_avx2(&gradient_v, &s_history, &y_history, 1, d, &mut avx2_dir);
        }

        let ulp = common::max_ulp_distance(&scalar_dir, &avx2_dir);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-LB-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-LB-006: Gradient recovery
/// Contract: lbfgs-kernel-v1.yaml
/// Prediction: with m=0 history, direction = -gradient
/// If fails: L-BFGS does not reduce to steepest descent with no history
#[test]
fn falsify_lb_006_gradient_recovery() {
    let d = 4;
    #[allow(clippy::approx_constant)]
    let gradient = [3.14_f32, -2.71, 1.41, -0.57];
    let s_history: [f32; 0] = [];
    let y_history: [f32; 0] = [];
    let mut direction = vec![0.0_f32; d];

    lbfgs_direction_scalar(&gradient, &s_history, &y_history, 0, d, &mut direction);

    for i in 0..d {
        assert!(
            (direction[i] + gradient[i]).abs() < 1e-7,
            "FALSIFY-LB-006 failed: direction[{i}] = {}, expected {} (should be -gradient)",
            direction[i],
            -gradient[i]
        );
    }
}
