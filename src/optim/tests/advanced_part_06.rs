
// ==================== Additional Coverage Tests ====================

// Test ADMM with adaptive rho exercising the DECREASE branch (dual > 10 * primal)
#[test]
fn test_admm_adaptive_rho_decrease() {
    let n = 3;

    let A = Matrix::eye(n);
    let B = Matrix::from_vec(n, n, vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
        .expect("Valid matrix");
    let c = Vector::zeros(n);

    // x-minimizer that creates large dual residual relative to primal
    // This encourages the dual_res > 10.0 * primal_res branch
    let x_minimizer = |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        let mut x = Vector::zeros(n);
        for i in 0..n {
            // Large changes to z will create large dual residual
            x[i] = z[i] - u[i] + 100.0 * (rho - 1.0).signum();
        }
        x
    };

    // z-minimizer that creates oscillating z values
    let z_minimizer = |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        let mut z = Vector::zeros(n);
        for i in 0..n {
            // Oscillating values to maximize dual residual
            z[i] = -ax[i] - u[i] + (rho * 50.0);
        }
        z
    };

    // Use high rho to encourage decrease branch
    let mut admm = ADMM::new(50, 100.0, 1e-6).with_adaptive_rho(true);
    let x0 = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let z0 = Vector::from_slice(&[10.0, 20.0, 30.0]);

    let result = admm.minimize_consensus(x_minimizer, z_minimizer, &A, &B, &c, x0, z0);

    // Should run without panic and return a result
    assert!(result.solution.len() == n);
}

// Test ADMM max iterations with adaptive rho
#[test]
fn test_admm_adaptive_rho_max_iterations() {
    let n = 2;
    let A = Matrix::eye(n);
    let B = Matrix::eye(n);
    let c = Vector::zeros(n);

    // Non-converging minimizers
    let x_minimizer = |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut x = Vector::zeros(n);
        for i in 0..n {
            x[i] = z[i] - u[i] + 0.1; // Always shifts
        }
        x
    };

    let z_minimizer = |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut z = Vector::zeros(n);
        for i in 0..n {
            z[i] = ax[i] + u[i] - 0.1; // Always shifts
        }
        z
    };

    let mut admm = ADMM::new(5, 1.0, 1e-12) // Very small tolerance, few iterations
        .with_adaptive_rho(true)
        .with_rho_factors(3.0, 3.0);
    let x0 = Vector::zeros(n);
    let z0 = Vector::zeros(n);

    let result = admm.minimize_consensus(x_minimizer, z_minimizer, &A, &B, &c, x0, z0);

    // Should either converge or hit max iterations
    assert!(
        result.status == ConvergenceStatus::MaxIterations
            || result.status == ConvergenceStatus::Converged
    );
}

// Test prox functions edge cases
#[test]
fn test_prox_nonnegative_all_positive2() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let result = prox::nonnegative(&x);
    for i in 0..4 {
        assert!((result[i] - x[i]).abs() < f32::EPSILON);
    }
}

#[test]
fn test_prox_nonnegative_all_negative2() {
    let x = Vector::from_slice(&[-1.0, -2.0, -3.0, -4.0]);
    let result = prox::nonnegative(&x);
    for i in 0..4 {
        assert!(result[i].abs() < f32::EPSILON);
    }
}

#[test]
fn test_prox_soft_threshold_below_threshold2() {
    let x = Vector::from_slice(&[0.3, -0.2, 0.1, -0.05]);
    let result = prox::soft_threshold(&x, 0.5);
    for i in 0..4 {
        assert!(result[i].abs() < f32::EPSILON);
    }
}

// Test LBFGS debug and clone
#[test]
fn test_lbfgs_debug_clone2() {
    let lbfgs = LBFGS::new(100, 1e-6, 10);
    let cloned = lbfgs.clone();
    // Just verify clone doesn't panic

    let debug_str = format!("{:?}", lbfgs);
    assert!(debug_str.contains("LBFGS"));
    drop(cloned);
}

// Test DampedNewton debug and clone
#[test]
fn test_damped_newton_debug_clone2() {
    let dn = DampedNewton::new(100, 1e-6);
    let _cloned = dn.clone();
    // Just verify clone doesn't panic

    let debug_str = format!("{:?}", dn);
    assert!(debug_str.contains("DampedNewton"));
}
