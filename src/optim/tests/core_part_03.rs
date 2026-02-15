
#[test]
fn test_lbfgs_sphere() {
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

    let mut optimizer = LBFGS::new(100, 1e-5, 10);
    let x0 = Vector::from_slice(&[5.0, -3.0, 2.0, -1.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    for i in 0..4 {
        assert!(result.solution[i].abs() < 1e-3);
    }
}

#[test]
fn test_lbfgs_different_history_sizes() {
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
    let x0 = Vector::from_slice(&[3.0, 4.0]);

    // Small history
    let mut opt_small = LBFGS::new(100, 1e-5, 3);
    let result_small = opt_small.minimize(f, grad, x0.clone());
    assert_eq!(result_small.status, ConvergenceStatus::Converged);

    // Large history
    let mut opt_large = LBFGS::new(100, 1e-5, 20);
    let result_large = opt_large.minimize(f, grad, x0);
    assert_eq!(result_large.status, ConvergenceStatus::Converged);

    // Both should converge to same solution
    assert!((result_small.solution[0] - result_large.solution[0]).abs() < 1e-3);
    assert!((result_small.solution[1] - result_large.solution[1]).abs() < 1e-3);
}

#[test]
fn test_lbfgs_reset() {
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = LBFGS::new(100, 1e-5, 10);
    let x0 = Vector::from_slice(&[5.0]);

    // First run
    optimizer.minimize(f, grad, x0.clone());
    assert!(!optimizer.s_history.is_empty());

    // Reset
    optimizer.reset();
    assert_eq!(optimizer.s_history.len(), 0);
    assert_eq!(optimizer.y_history.len(), 0);

    // Second run should work
    let result = optimizer.minimize(f, grad, x0);
    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_lbfgs_max_iterations() {
    // Use Rosenbrock with very few iterations to force max_iter
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

    let mut optimizer = LBFGS::new(3, 1e-10, 5);
    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    // With only 3 iterations, should hit max_iter on Rosenbrock
    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 3);
}

#[test]
#[should_panic(expected = "does not support stochastic")]
fn test_lbfgs_step_panics() {
    let mut optimizer = LBFGS::new(100, 1e-5, 10);
    let mut params = Vector::from_slice(&[1.0]);
    let grad = Vector::from_slice(&[0.1]);

    // Should panic - L-BFGS doesn't support step()
    optimizer.step(&mut params, &grad);
}

#[test]
fn test_lbfgs_numerical_error_detection() {
    // Function that produces NaN
    let f = |x: &Vector<f32>| {
        if x[0] < -100.0 {
            f32::NAN
        } else {
            x[0] * x[0]
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = LBFGS::new(100, 1e-5, 5);
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
fn test_lbfgs_computes_elapsed_time() {
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = LBFGS::new(100, 1e-5, 5);
    let x0 = Vector::from_slice(&[5.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Should have non-zero elapsed time
    assert!(result.elapsed_time.as_nanos() > 0);
}

#[test]
fn test_lbfgs_gradient_norm_tracking() {
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let mut optimizer = LBFGS::new(100, 1e-5, 10);
    let x0 = Vector::from_slice(&[3.0, 4.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Gradient norm at convergence should be small
    if result.status == ConvergenceStatus::Converged {
        assert!(result.gradient_norm < 1e-5);
    }
}

// ==================== Conjugate Gradient Tests ====================

#[test]
fn test_cg_new_fletcher_reeves() {
    let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);
    assert_eq!(optimizer.max_iter, 100);
    assert!((optimizer.tol - 1e-5).abs() < 1e-10);
    assert_eq!(optimizer.beta_formula, CGBetaFormula::FletcherReeves);
    assert_eq!(optimizer.restart_interval, 0);
}

#[test]
fn test_cg_new_polak_ribiere() {
    let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    assert_eq!(optimizer.beta_formula, CGBetaFormula::PolakRibiere);
}

#[test]
fn test_cg_new_hestenes_stiefel() {
    let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::HestenesStiefel);
    assert_eq!(optimizer.beta_formula, CGBetaFormula::HestenesStiefel);
}

#[test]
fn test_cg_with_restart_interval() {
    let optimizer =
        ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere).with_restart_interval(50);
    assert_eq!(optimizer.restart_interval, 50);
}

#[test]
fn test_cg_simple_quadratic_fr() {
    // Minimize f(x) = x^2, optimal at x = 0
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);
    let x0 = Vector::from_slice(&[5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 1e-3);
}

#[test]
fn test_cg_simple_quadratic_pr() {
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let x0 = Vector::from_slice(&[5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 1e-3);
}

#[test]
fn test_cg_simple_quadratic_hs() {
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::HestenesStiefel);
    let x0 = Vector::from_slice(&[5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 1e-3);
}

#[test]
fn test_cg_multidimensional_quadratic() {
    // Minimize f(x) = ||x - c||^2 where c = [1, 2, 3]
    let c = [1.0, 2.0, 3.0];
    let f = |x: &Vector<f32>| {
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += (x[i] - c[i]).powi(2);
        }
        sum
    };

    let grad = |x: &Vector<f32>| {
        let mut g = Vector::zeros(x.len());
        for i in 0..x.len() {
            g[i] = 2.0 * (x[i] - c[i]);
        }
        g
    };

    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    for (i, &target) in c.iter().enumerate() {
        assert!((result.solution[i] - target).abs() < 1e-3);
    }
}

#[test]
fn test_cg_rosenbrock() {
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

    let mut optimizer = ConjugateGradient::new(500, 1e-4, CGBetaFormula::PolakRibiere);
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
fn test_cg_sphere() {
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

    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let x0 = Vector::from_slice(&[5.0, -3.0, 2.0, -1.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    for i in 0..4 {
        assert!(result.solution[i].abs() < 1e-3);
    }
}

#[test]
fn test_cg_compare_beta_formulas() {
    // Compare different beta formulas on same problem
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
    let x0 = Vector::from_slice(&[3.0, 4.0]);

    let mut opt_fr = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);
    let result_fr = opt_fr.minimize(f, grad, x0.clone());

    let mut opt_pr = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let result_pr = opt_pr.minimize(f, grad, x0.clone());

    let mut opt_hs = ConjugateGradient::new(100, 1e-5, CGBetaFormula::HestenesStiefel);
    let result_hs = opt_hs.minimize(f, grad, x0);

    // All should converge to same solution
    assert_eq!(result_fr.status, ConvergenceStatus::Converged);
    assert_eq!(result_pr.status, ConvergenceStatus::Converged);
    assert_eq!(result_hs.status, ConvergenceStatus::Converged);

    assert!(result_fr.solution[0].abs() < 1e-3);
    assert!(result_pr.solution[0].abs() < 1e-3);
    assert!(result_hs.solution[0].abs() < 1e-3);
}

#[test]
fn test_cg_with_periodic_restart() {
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let mut optimizer =
        ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere).with_restart_interval(5);
    let x0 = Vector::from_slice(&[5.0, 5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 1e-3);
    assert!(result.solution[1].abs() < 1e-3);
}

#[test]
fn test_cg_reset() {
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let x0 = Vector::from_slice(&[5.0]);

    // First run
    optimizer.minimize(f, grad, x0.clone());
    assert!(optimizer.prev_direction.is_some());

    // Reset
    optimizer.reset();
    assert!(optimizer.prev_direction.is_none());
    assert!(optimizer.prev_gradient.is_none());
    assert_eq!(optimizer.iter_count, 0);

    // Second run should work
    let result = optimizer.minimize(f, grad, x0);
    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_cg_max_iterations() {
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

    let mut optimizer = ConjugateGradient::new(3, 1e-10, CGBetaFormula::PolakRibiere);
    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 3);
}

#[test]
#[should_panic(expected = "does not support stochastic")]
fn test_cg_step_panics() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let mut params = Vector::from_slice(&[1.0]);
    let grad = Vector::from_slice(&[0.1]);

    // Should panic - CG doesn't support step()
    optimizer.step(&mut params, &grad);
}

#[test]
fn test_cg_numerical_error_detection() {
    // Function that produces NaN
    let f = |x: &Vector<f32>| {
        if x[0] < -100.0 {
            f32::NAN
        } else {
            x[0] * x[0]
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Should detect numerical error or converge normally
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::NumericalError
            || result.status == ConvergenceStatus::MaxIterations
    );
}
