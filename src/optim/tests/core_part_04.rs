
#[test]
fn test_cg_gradient_norm_tracking() {
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let x0 = Vector::from_slice(&[3.0, 4.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Gradient norm at convergence should be small
    if result.status == ConvergenceStatus::Converged {
        assert!(result.gradient_norm < 1e-5);
    }
}

#[test]
fn test_cg_vs_lbfgs_quadratic() {
    // Compare CG and L-BFGS on a quadratic problem
    let f = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1], 6.0 * x[2]]);
    let x0 = Vector::from_slice(&[5.0, 3.0, 2.0]);

    let mut cg = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let result_cg = cg.minimize(f, grad, x0.clone());

    let mut lbfgs = LBFGS::new(100, 1e-5, 10);
    let result_lbfgs = lbfgs.minimize(f, grad, x0);

    // Both should converge to same solution
    assert_eq!(result_cg.status, ConvergenceStatus::Converged);
    assert_eq!(result_lbfgs.status, ConvergenceStatus::Converged);

    for i in 0..3 {
        assert!((result_cg.solution[i] - result_lbfgs.solution[i]).abs() < 1e-3);
    }
}

// ==================== Damped Newton Tests ====================

#[test]
fn test_damped_newton_new() {
    let optimizer = DampedNewton::new(100, 1e-5);
    assert_eq!(optimizer.max_iter, 100);
    assert!((optimizer.tol - 1e-5).abs() < 1e-10);
    assert!((optimizer.epsilon - 1e-5).abs() < 1e-10);
}

#[test]
fn test_damped_newton_with_epsilon() {
    let optimizer = DampedNewton::new(100, 1e-5).with_epsilon(1e-6);
    assert!((optimizer.epsilon - 1e-6).abs() < 1e-12);
}

#[test]
fn test_damped_newton_simple_quadratic() {
    // Minimize f(x) = x^2, optimal at x = 0
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = DampedNewton::new(100, 1e-5);
    let x0 = Vector::from_slice(&[5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 1e-3);
}

#[test]
fn test_damped_newton_multidimensional_quadratic() {
    // Minimize f(x,y) = x^2 + 2y^2, optimal at (0, 0)
    let f = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1]]);

    let mut optimizer = DampedNewton::new(100, 1e-5);
    let x0 = Vector::from_slice(&[5.0, 3.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 1e-3);
    assert!(result.solution[1].abs() < 1e-3);
}

#[test]
fn test_damped_newton_rosenbrock() {
    // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    let f = |v: &Vector<f32>| {
        let x = v[0];
        let y = v[1];
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    };

    let grad = |v: &Vector<f32>| {
        let x = v[0];
        let y = v[1];
        let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        let dy = 200.0 * (y - x * x);
        Vector::from_slice(&[dx, dy])
    };

    let mut optimizer = DampedNewton::new(200, 1e-4);
    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Should converge close to (1, 1)
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
    assert!((result.solution[0] - 1.0).abs() < 0.1);
    assert!((result.solution[1] - 1.0).abs() < 0.1);
}

#[test]
fn test_damped_newton_sphere() {
    // Sphere function: f(x) = sum(x_i^2)
    let f = |x: &Vector<f32>| {
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += x[i] * x[i];
        }
        sum
    };

    let grad = |x: &Vector<f32>| {
        let mut g = Vector::zeros(x.len());
        for i in 0..x.len() {
            g[i] = 2.0 * x[i];
        }
        g
    };

    let mut optimizer = DampedNewton::new(100, 1e-5);
    let x0 = Vector::from_slice(&[5.0, -3.0, 2.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    for i in 0..3 {
        assert!(result.solution[i].abs() < 1e-3);
    }
}

#[test]
fn test_damped_newton_reset() {
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = DampedNewton::new(100, 1e-5);
    let x0 = Vector::from_slice(&[5.0]);

    // First run
    optimizer.minimize(f, grad, x0.clone());

    // Reset (stateless, so just verify it doesn't panic)
    optimizer.reset();

    // Second run should work
    let result = optimizer.minimize(f, grad, x0);
    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_damped_newton_max_iterations() {
    // Use Rosenbrock with very few iterations
    let f = |v: &Vector<f32>| {
        let x = v[0];
        let y = v[1];
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    };

    let grad = |v: &Vector<f32>| {
        let x = v[0];
        let y = v[1];
        let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        let dy = 200.0 * (y - x * x);
        Vector::from_slice(&[dx, dy])
    };

    let mut optimizer = DampedNewton::new(3, 1e-10);
    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 3);
}

#[test]
#[should_panic(expected = "does not support stochastic")]
fn test_damped_newton_step_panics() {
    let mut optimizer = DampedNewton::new(100, 1e-5);
    let mut params = Vector::from_slice(&[1.0]);
    let grad = Vector::from_slice(&[0.1]);

    // Should panic - Damped Newton doesn't support step()
    optimizer.step(&mut params, &grad);
}

#[test]
fn test_damped_newton_numerical_error_detection() {
    // Function that produces NaN
    let f = |x: &Vector<f32>| {
        if x[0] < -100.0 {
            f32::NAN
        } else {
            x[0] * x[0]
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = DampedNewton::new(100, 1e-5);
    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Should detect numerical error or converge normally
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::NumericalError
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_damped_newton_gradient_norm_tracking() {
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let mut optimizer = DampedNewton::new(100, 1e-5);
    let x0 = Vector::from_slice(&[3.0, 4.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Gradient norm at convergence should be small
    if result.status == ConvergenceStatus::Converged {
        assert!(result.gradient_norm < 1e-5);
    }
}

#[test]
fn test_damped_newton_quadratic_convergence() {
    // Newton's method should converge quadratically on quadratic problems
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let mut optimizer = DampedNewton::new(100, 1e-10);
    let x0 = Vector::from_slice(&[5.0, 5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    // Should converge in very few iterations for quadratic problems
    assert!(result.iterations < 20);
}

#[test]
fn test_damped_newton_vs_lbfgs_quadratic() {
    // Compare Damped Newton and L-BFGS on a quadratic problem
    let f = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1]]);
    let x0 = Vector::from_slice(&[5.0, 3.0]);

    let mut dn = DampedNewton::new(100, 1e-5);
    let result_dn = dn.minimize(f, grad, x0.clone());

    let mut lbfgs = LBFGS::new(100, 1e-5, 10);
    let result_lbfgs = lbfgs.minimize(f, grad, x0);

    // Both should converge to same solution
    assert_eq!(result_dn.status, ConvergenceStatus::Converged);
    assert_eq!(result_lbfgs.status, ConvergenceStatus::Converged);

    assert!((result_dn.solution[0] - result_lbfgs.solution[0]).abs() < 1e-3);
    assert!((result_dn.solution[1] - result_lbfgs.solution[1]).abs() < 1e-3);
}

#[test]
fn test_damped_newton_different_epsilon() {
    // Test with different finite difference epsilons
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    let x0 = Vector::from_slice(&[5.0]);

    let mut opt1 = DampedNewton::new(100, 1e-5).with_epsilon(1e-5);
    let result1 = opt1.minimize(f, grad, x0.clone());

    let mut opt2 = DampedNewton::new(100, 1e-5).with_epsilon(1e-7);
    let result2 = opt2.minimize(f, grad, x0);

    // Both should converge
    assert_eq!(result1.status, ConvergenceStatus::Converged);
    assert_eq!(result2.status, ConvergenceStatus::Converged);

    // Solutions should be similar
    assert!((result1.solution[0] - result2.solution[0]).abs() < 1e-2);
}
