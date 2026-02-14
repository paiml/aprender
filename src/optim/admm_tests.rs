use super::*;

#[test]
fn test_admm_new() {
    let admm = ADMM::new(100, 1.0, 1e-4);
    assert_eq!(admm.max_iter(), 100);
    assert!((admm.rho() - 1.0).abs() < 1e-10);
    assert!((admm.tol() - 1e-4).abs() < 1e-10);
    assert!(!admm.adaptive_rho());
}

#[test]
fn test_admm_clone_debug() {
    let admm = ADMM::new(100, 1.0, 1e-4);
    let cloned = admm.clone();
    assert_eq!(admm.max_iter(), cloned.max_iter());
    let debug_str = format!("{:?}", admm);
    assert!(debug_str.contains("ADMM"));
}

#[test]
fn test_admm_with_adaptive_rho() {
    let admm = ADMM::new(100, 1.0, 1e-4).with_adaptive_rho(true);
    assert!(admm.adaptive_rho());
}

#[test]
fn test_admm_with_rho_factors() {
    let admm = ADMM::new(100, 1.0, 1e-4).with_rho_factors(3.0, 4.0);
    assert!((admm.rho_increase() - 3.0).abs() < 1e-10);
    assert!((admm.rho_decrease() - 4.0).abs() < 1e-10);
}

#[test]
fn test_admm_getters() {
    let admm = ADMM::new(200, 2.5, 1e-3)
        .with_adaptive_rho(true)
        .with_rho_factors(5.0, 3.0);

    assert_eq!(admm.max_iter(), 200);
    assert!((admm.rho() - 2.5).abs() < 1e-10);
    assert!((admm.tol() - 1e-3).abs() < 1e-10);
    assert!(admm.adaptive_rho());
    assert!((admm.rho_increase() - 5.0).abs() < 1e-10);
    assert!((admm.rho_decrease() - 3.0).abs() < 1e-10);
}

#[test]
fn test_admm_consensus_converges() {
    let n = 3;
    let a = Matrix::eye(n);
    let b_mat = Matrix::eye(n);
    let c = Vector::zeros(n);

    // x-min: x = z - u (trivial least-squares for identity A)
    let x_minimizer = |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut x = Vector::zeros(z.len());
        for i in 0..z.len() {
            x[i] = z[i] - u[i];
        }
        x
    };

    // z-min: z = Ax + u = x + u (identity B)
    let z_minimizer = |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut z = Vector::zeros(ax.len());
        for i in 0..ax.len() {
            z[i] = ax[i] + u[i];
        }
        z
    };

    let mut admm = ADMM::new(100, 1.0, 1e-4);
    let x0 = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let z0 = Vector::zeros(n);

    let result = admm.minimize_consensus(x_minimizer, z_minimizer, &a, &b_mat, &c, x0, z0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_admm_max_iterations() {
    let n = 2;
    let a = Matrix::eye(n);
    let b_mat = Matrix::eye(n);
    let c = Vector::zeros(n);

    // Minimizers that don't converge quickly
    let x_minimizer = |z: &Vector<f32>, _u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut x = Vector::zeros(z.len());
        for i in 0..z.len() {
            x[i] = z[i] + 1.0; // Always shifts
        }
        x
    };
    let z_minimizer = |ax: &Vector<f32>, _u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut z = Vector::zeros(ax.len());
        for i in 0..ax.len() {
            z[i] = ax[i] - 1.0; // Always shifts other way
        }
        z
    };

    let mut admm = ADMM::new(3, 1.0, 1e-20);
    let x0 = Vector::zeros(n);
    let z0 = Vector::zeros(n);

    let result = admm.minimize_consensus(x_minimizer, z_minimizer, &a, &b_mat, &c, x0, z0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 3);
}

#[test]
fn test_admm_adaptive_rho_primal_large() {
    // Set up problem where primal residual >> dual residual
    // to trigger rho increase on iteration 0 (iter % 10 == 0)
    let n = 2;
    let a = Matrix::eye(n);
    let b_mat = Matrix::eye(n);
    let c = Vector::zeros(n);

    // x-min returns something far from z (large primal residual)
    let x_minimizer = |_z: &Vector<f32>, _u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        Vector::from_slice(&[100.0, 100.0])
    };
    // z-min returns same as z_old (zero dual residual)
    let z_minimizer =
        |_ax: &Vector<f32>, _u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| Vector::zeros(2);

    let mut admm = ADMM::new(20, 1.0, 1e-4).with_adaptive_rho(true);
    let x0 = Vector::zeros(n);
    let z0 = Vector::zeros(n);

    let _ = admm.minimize_consensus(x_minimizer, z_minimizer, &a, &b_mat, &c, x0, z0);

    // The adaptive rho path was exercised - no panic
}

#[test]
fn test_admm_adaptive_rho_dual_large() {
    // Set up scenario where dual residual >> primal residual
    // Dual residual = rho * ||B^T(z_{k+1} - z_k)||
    // To make dual large: z changes a lot between iterations
    // To make primal small: Ax + Bz ≈ c
    let n = 2;
    let a = Matrix::eye(n);
    let b_mat = Matrix::eye(n);
    let c = Vector::zeros(n);

    let call_count = std::cell::Cell::new(0);
    let x_minimizer = |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        // x = -z so Ax + Bz = 0 = c (small primal residual)
        let mut x = Vector::zeros(z.len());
        for i in 0..z.len() {
            x[i] = -(z[i] + u[i]);
        }
        x
    };

    let z_minimizer = {
        let call_count = &call_count;
        move |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
            let count = call_count.get();
            call_count.set(count + 1);
            // z changes a lot each iteration (large dual residual)
            let mut z = Vector::zeros(ax.len());
            for i in 0..ax.len() {
                z[i] = -(ax[i] + u[i]) + (count as f32) * 10.0;
            }
            z
        }
    };

    let mut admm = ADMM::new(15, 1.0, 1e-4)
        .with_adaptive_rho(true)
        .with_rho_factors(2.0, 2.0);
    let x0 = Vector::zeros(n);
    let z0 = Vector::zeros(n);

    let _ = admm.minimize_consensus(x_minimizer, z_minimizer, &a, &b_mat, &c, x0, z0);

    // Dual residual path was exercised
}

#[test]
fn test_admm_reset() {
    let mut admm = ADMM::new(100, 1.0, 1e-4);
    admm.reset(); // Stateless, should not panic
}

#[test]
#[should_panic(expected = "does not support stochastic updates")]
fn test_admm_step_panics() {
    let mut admm = ADMM::new(100, 1.0, 1e-4);
    let mut params = Vector::from_slice(&[1.0]);
    let grad = Vector::from_slice(&[0.1]);
    admm.step(&mut params, &grad);
}

#[test]
fn test_admm_result_fields() {
    let n = 2;
    let a = Matrix::eye(n);
    let b_mat = Matrix::eye(n);
    let c = Vector::zeros(n);

    let x_minimizer = |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut x = Vector::zeros(z.len());
        for i in 0..z.len() {
            x[i] = z[i] - u[i];
        }
        x
    };
    let z_minimizer = |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut z = Vector::zeros(ax.len());
        for i in 0..ax.len() {
            z[i] = ax[i] + u[i];
        }
        z
    };

    let mut admm = ADMM::new(100, 1.0, 1e-4);
    let result = admm.minimize_consensus(
        x_minimizer,
        z_minimizer,
        &a,
        &b_mat,
        &c,
        Vector::from_slice(&[1.0, 1.0]),
        Vector::zeros(n),
    );

    // Check all fields are populated
    assert!(result.solution.len() == 2);
    let _ = result.elapsed_time.as_nanos();
    assert!(result.objective_value.is_finite() || result.objective_value == 0.0);
}
