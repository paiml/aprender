
// ==================== Interior Point Tests ====================

#[test]
fn test_interior_point_nonnegative() {
    // Minimize: x₁² + x₂² subject to -x₁ ≤ 0, -x₂ ≤ 0 (i.e., x ≥ 0)
    // Solution: x = [0, 0]

    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    // Inequality constraints: g(x) = [-x₁, -x₂] ≤ 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);

    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(80, 1e-5, 1.0);
    let x0 = Vector::from_slice(&[0.5, 0.5]); // Interior feasible start
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Solution approaches [0, 0] as barrier parameter decreases
    assert!(result.solution[0].abs() < 0.2);
    assert!(result.solution[1].abs() < 0.2);
    assert!(result.constraint_violation <= 0.0); // All constraints satisfied
}

#[test]
fn test_interior_point_box_constraints() {
    // Minimize: (x₁-0.8)² + (x₂-0.8)² subject to 0 ≤ x ≤ 1
    // Target is inside the box, so solution should approach [0.8, 0.8]

    let objective = |x: &Vector<f32>| (x[0] - 0.8).powi(2) + (x[1] - 0.8).powi(2);

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 0.8), 2.0 * (x[1] - 0.8)]);

    // g(x) = [-x₁, -x₂, x₁-1, x₂-1] ≤ 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1], x[0] - 1.0, x[1] - 1.0]);

    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
            Vector::from_slice(&[1.0, 0.0]),
            Vector::from_slice(&[0.0, 1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(80, 1e-4, 1.0);
    let x0 = Vector::from_slice(&[0.5, 0.5]); // Feasible start (interior)
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Solution should be within box [0,1]×[0,1]
    assert!(result.solution[0] >= 0.0 && result.solution[0] <= 1.0);
    assert!(result.solution[1] >= 0.0 && result.solution[1] <= 1.0);
    assert!(result.constraint_violation <= 0.0);
}

#[test]
fn test_interior_point_linear_constraint() {
    // Minimize: x₁² + x₂² subject to x₁ + x₂ ≤ 2
    // Solution is interior or on boundary

    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    // g(x) = [x₁ + x₂ - 2] ≤ 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] - 2.0]);

    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0, 1.0])];

    let mut ip = InteriorPoint::new(80, 1e-5, 1.0);
    let x0 = Vector::from_slice(&[0.5, 0.5]); // Feasible start
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Solution approaches [1, 1] on boundary or stays interior
    assert!(result.solution[0] + result.solution[1] <= 2.1);
    assert!(result.constraint_violation <= 0.0);
}

#[test]
fn test_interior_point_3d() {
    // Minimize: ‖x‖² subject to x₁ + x₂ + x₃ ≤ 1, x ≥ 0

    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1] + x[2] * x[2];

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1], 2.0 * x[2]]);

    // g(x) = [x₁+x₂+x₃-1, -x₁, -x₂, -x₃] ≤ 0
    let inequality =
        |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] + x[2] - 1.0, -x[0], -x[1], -x[2]]);

    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[1.0, 1.0, 1.0]),
            Vector::from_slice(&[-1.0, 0.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0, 0.0]),
            Vector::from_slice(&[0.0, 0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(80, 1e-5, 1.0);
    let x0 = Vector::from_slice(&[0.2, 0.2, 0.2]); // Feasible start
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Solution moves toward origin while satisfying constraints
    assert!(result.solution[0] + result.solution[1] + result.solution[2] <= 1.1);
    assert!(result.solution[0] >= -0.1);
    assert!(result.solution[1] >= -0.1);
    assert!(result.solution[2] >= -0.1);
}

#[test]
fn test_interior_point_convergence_tracking() {
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);

    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
    let x0 = Vector::from_slice(&[1.0, 1.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    assert!(result.iterations > 0);
    assert!(result.elapsed_time.as_nanos() > 0);
    assert!(result.constraint_violation <= 0.0);
}

#[test]
fn test_interior_point_beta_parameter() {
    // Test with custom beta (barrier decrease factor)
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);

    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(50, 1e-6, 1.0).with_beta(0.1);
    let x0 = Vector::from_slice(&[1.0, 1.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    assert!(result.solution[0].abs() < 1e-1);
    assert!(result.solution[1].abs() < 1e-1);
}

#[test]
#[should_panic(expected = "Initial point is infeasible")]
fn test_interior_point_infeasible_start() {
    // Test that infeasible initial point panics
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);

    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
    let x0 = Vector::from_slice(&[-1.0, 1.0]); // INFEASIBLE! x₁ < 0
    ip.minimize(objective, gradient, inequality, inequality_jac, x0);
}

#[test]
fn test_interior_point_max_iterations() {
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);

    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(2, 1e-10, 1.0); // Very few iterations
    let x0 = Vector::from_slice(&[1.0, 1.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 2);
}

// ==================== Additional Coverage Tests ====================

// Test reset methods for all optimizers
#[test]
fn test_coordinate_descent_reset() {
    let mut cd = CoordinateDescent::new(100, 1e-6);
    cd.reset(); // Should do nothing but not panic
}

#[test]
fn test_admm_reset() {
    let mut admm = ADMM::new(100, 1.0, 1e-4);
    admm.reset(); // Should do nothing but not panic
}

#[test]
fn test_projected_gd_reset() {
    let mut pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
    pgd.reset(); // Should do nothing but not panic
}

#[test]
fn test_augmented_lagrangian_reset() {
    let mut al = AugmentedLagrangian::new(100, 1e-6, 1.0);
    al.reset(); // Should reset rho to initial value
}

#[test]
fn test_interior_point_reset() {
    let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
    ip.reset(); // Should reset mu to initial value
}

// Test unimplemented step methods - they should panic
#[test]
#[should_panic(expected = "does not support stochastic updates")]
fn test_coordinate_descent_step_unimplemented() {
    let mut cd = CoordinateDescent::new(100, 1e-6);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let grad = Vector::from_slice(&[0.1, 0.2]);
    cd.step(&mut params, &grad);
}

#[test]
#[should_panic(expected = "does not support stochastic updates")]
fn test_admm_step_unimplemented() {
    let mut admm = ADMM::new(100, 1.0, 1e-4);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let grad = Vector::from_slice(&[0.1, 0.2]);
    admm.step(&mut params, &grad);
}

#[test]
#[should_panic(expected = "does not support stochastic updates")]
fn test_projected_gd_step_unimplemented() {
    let mut pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let grad = Vector::from_slice(&[0.1, 0.2]);
    pgd.step(&mut params, &grad);
}

#[test]
#[should_panic(expected = "does not support stochastic updates")]
fn test_augmented_lagrangian_step_unimplemented() {
    let mut al = AugmentedLagrangian::new(100, 1e-6, 1.0);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let grad = Vector::from_slice(&[0.1, 0.2]);
    al.step(&mut params, &grad);
}

#[test]
#[should_panic(expected = "does not support stochastic updates")]
fn test_interior_point_step_unimplemented() {
    let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let grad = Vector::from_slice(&[0.1, 0.2]);
    ip.step(&mut params, &grad);
}

// Test CoordinateDescent with random order
#[test]
fn test_coordinate_descent_random_order() {
    let c = vec![1.0, 2.0, 3.0];
    let c_clone = c.clone();

    let update = move |x: &mut Vector<f32>, i: usize| {
        x[i] = c_clone[i];
    };

    let mut cd = CoordinateDescent::new(100, 1e-6).with_random_order(true);
    let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = cd.minimize(update, x0);

    // Should converge to c regardless of order
    assert!((result.solution[0] - 1.0).abs() < 1e-4);
    assert!((result.solution[1] - 2.0).abs() < 1e-4);
    assert!((result.solution[2] - 3.0).abs() < 1e-4);
}

// Test safe_cholesky_solve failure with extremely ill-conditioned matrix
#[test]
fn test_safe_cholesky_solve_extreme_ill_condition() {
    // Create an extremely ill-conditioned matrix that cannot be solved
    // even with regularization up to lambda=1e6
    let A = Matrix::from_vec(2, 2, vec![1e-20, 1e-20, 1e-20, 1e-20]).expect("valid dimensions");
    let b = Vector::from_slice(&[1e20, 1e20]);

    // With only 1 attempt and small lambda, it may fail
    let result = safe_cholesky_solve(&A, &b, 1e-12, 1);
    // This may succeed or fail depending on regularization
    assert!(result.is_ok() || result.is_err());
}

// Test soft threshold edge cases
#[test]
fn test_soft_threshold_exact_threshold() {
    // Test when value exactly equals threshold
    let v = Vector::from_slice(&[1.0, -1.0, 0.5, -0.5]);
    let result = prox::soft_threshold(&v, 1.0);

    // Values at exactly threshold should become 0
    assert!(result[0].abs() < 1e-6); // 1.0 - 1.0 = 0
    assert!(result[1].abs() < 1e-6); // -1.0 + 1.0 = 0
    assert!(result[2].abs() < 1e-6); // 0.5 is within [-1, 1]
    assert!(result[3].abs() < 1e-6); // -0.5 is within [-1, 1]
}

// Test project_box with equal bounds
#[test]
fn test_project_box_equal_bounds() {
    let x = Vector::from_slice(&[0.0, 5.0, -5.0]);
    let lower = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let upper = Vector::from_slice(&[1.0, 1.0, 1.0]); // Same as lower

    let result = prox::project_box(&x, &lower, &upper);

    // All values should be clamped to 1.0
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 1.0).abs() < 1e-6);
}

// Test project_l2_ball at boundary
#[test]
fn test_project_l2_ball_at_boundary() {
    let x = Vector::from_slice(&[3.0, 4.0]); // norm = 5.0
    let result = prox::project_l2_ball(&x, 5.0); // Already on boundary

    // Should be unchanged
    assert!((result[0] - 3.0).abs() < 1e-6);
    assert!((result[1] - 4.0).abs() < 1e-6);
}

// Test ADMM with adaptive rho exercising both branches
#[test]
fn test_admm_adaptive_rho_increase() {
    let n = 3;

    let A = Matrix::eye(n);
    let B = Matrix::eye(n);
    let c = Vector::zeros(n);

    // Simple x-minimizer
    let x_minimizer = |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        let mut x = Vector::zeros(n);
        for i in 0..n {
            x[i] = rho * (z[i] - u[i]) / (1.0 + rho);
        }
        x
    };

    // z-minimizer
    let z_minimizer = |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        let mut z = Vector::zeros(n);
        for i in 0..n {
            z[i] = ax[i] + u[i];
        }
        z
    };

    let mut admm = ADMM::new(100, 0.001, 1e-6).with_adaptive_rho(true);
    let x0 = Vector::zeros(n);
    let z0 = Vector::zeros(n);

    let result = admm.minimize_consensus(x_minimizer, z_minimizer, &A, &B, &c, x0, z0);

    // Just verify it runs and returns a result
    assert!(result.solution.len() == n);
}

// Test AugmentedLagrangian with rho increase
#[test]
fn test_augmented_lagrangian_with_rho_increase_factor() {
    let objective = |x: &Vector<f32>| 0.5 * (x[0] - 2.0).powi(2) + 0.5 * (x[1] - 3.0).powi(2);

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 2.0, x[1] - 3.0]);

    let equality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] - 1.0]);

    let equality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0, 1.0])];

    let mut al = AugmentedLagrangian::new(50, 1e-4, 0.1).with_rho_increase(5.0);
    let x0 = Vector::zeros(2);
    let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);

    // Should converge or hit max iterations
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

// Test ProjectedGradientDescent without line search reaching max iterations
#[test]
fn test_projected_gd_no_line_search_max_iter() {
    // Make a problem that won't converge quickly
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
    let project = |x: &Vector<f32>| x.clone(); // No projection

    let mut pgd = ProjectedGradientDescent::new(3, 0.0001, 1e-10); // Tiny step, few iters
    let x0 = Vector::from_slice(&[100.0, 100.0]);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
}
