use crate::optim::*;
use crate::primitives::Vector;

#[test]
#[should_panic(expected = "same length")]
fn test_adam_mismatched_lengths() {
    let mut optimizer = Adam::new(0.001);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let gradients = Vector::from_slice(&[1.0]);

    optimizer.step(&mut params, &gradients);
}

#[test]
fn test_adam_clone() {
    let optimizer = Adam::new(0.001).with_beta1(0.95).with_beta2(0.9999);
    let cloned = optimizer.clone();

    assert!((cloned.learning_rate() - optimizer.learning_rate()).abs() < 1e-9);
    assert!((cloned.beta1() - optimizer.beta1()).abs() < 1e-9);
    assert!((cloned.beta2() - optimizer.beta2()).abs() < 1e-9);
    assert!((cloned.epsilon() - optimizer.epsilon()).abs() < 1e-15);
}

#[test]
fn test_adam_adaptive_learning() {
    // Test that Adam adapts to gradient magnitudes
    let mut optimizer = Adam::new(0.01);
    let mut params = Vector::from_slice(&[1.0, 1.0]);

    // Large gradient on first param, small on second
    let gradients1 = Vector::from_slice(&[10.0, 0.1]);
    optimizer.step(&mut params, &gradients1);

    let step1_0 = 1.0 - params[0];
    let step1_1 = 1.0 - params[1];

    // Continue with same gradients
    optimizer.step(&mut params, &gradients1);

    // Adam should adapt - second param should take relatively larger steps
    // because it has more consistent small gradients
    assert!(step1_0 > 0.0);
    assert!(step1_1 > 0.0);
}

#[test]
fn test_adam_vs_sgd_behavior() {
    // Test that Adam and SGD behave differently (not necessarily one better)
    let mut adam = Adam::new(0.001);
    let mut sgd = SGD::new(0.1);

    let mut params_adam = Vector::from_slice(&[5.0]);
    let mut params_sgd = Vector::from_slice(&[5.0]);

    // Gradient pointing towards 0
    for _ in 0..10 {
        let gradients = Vector::from_slice(&[1.0]);
        adam.step(&mut params_adam, &gradients);
        sgd.step(&mut params_sgd, &gradients);
    }

    // Both should decrease but behave differently
    assert!(params_adam[0] < 5.0);
    assert!(params_sgd[0] < 5.0);
    // They should produce different results due to different mechanisms
    assert!((params_adam[0] - params_sgd[0]).abs() > 0.01);
}

#[test]
fn test_adam_moment_initialization() {
    let mut optimizer = Adam::new(0.001);

    // First with 2 params
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let gradients = Vector::from_slice(&[0.1, 0.2]);
    optimizer.step(&mut params, &gradients);

    // Now with 3 params - moments should reinitialize
    let mut params3 = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let gradients3 = Vector::from_slice(&[0.1, 0.2, 0.3]);
    optimizer.step(&mut params3, &gradients3);

    // Should work without error
    assert!(params3[0] < 1.0);
    assert!(params3[1] < 2.0);
    assert!(params3[2] < 3.0);
}

#[test]
fn test_adam_numerical_stability() {
    let mut optimizer = Adam::new(0.001);
    let mut params = Vector::from_slice(&[1.0]);

    // Very large gradients should be handled stably
    let gradients = Vector::from_slice(&[1000.0]);
    optimizer.step(&mut params, &gradients);

    // Should not produce NaN or extreme values
    assert!(!params[0].is_nan());
    assert!(params[0].is_finite());
}

// ==================== Line Search Tests ====================

#[test]
fn test_backtracking_line_search_new() {
    let ls = BacktrackingLineSearch::new(1e-4, 0.5, 50);
    assert!((ls.c1 - 1e-4).abs() < 1e-10);
    assert!((ls.rho - 0.5).abs() < 1e-10);
    assert_eq!(ls.max_iter, 50);
}

#[test]
fn test_backtracking_line_search_default() {
    let ls = BacktrackingLineSearch::default();
    assert!((ls.c1 - 1e-4).abs() < 1e-10);
    assert!((ls.rho - 0.5).abs() < 1e-10);
    assert_eq!(ls.max_iter, 50);
}

#[test]
fn test_backtracking_line_search_quadratic() {
    // Test on simple quadratic: f(x) = x^2
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let ls = BacktrackingLineSearch::default();
    let x = Vector::from_slice(&[2.0]);
    let d = Vector::from_slice(&[-4.0]); // Gradient direction at x=2

    let alpha = ls.search(&f, &grad, &x, &d);

    // Should find positive step size
    assert!(alpha > 0.0);
    assert!(alpha <= 1.0);

    // Verify Armijo condition is satisfied
    let x_new_data = x[0] + alpha * d[0];
    let x_new = Vector::from_slice(&[x_new_data]);
    let fx = f(&x);
    let fx_new = f(&x_new);
    let grad_x = grad(&x);
    let dir_deriv = grad_x[0] * d[0];

    assert!(fx_new <= fx + ls.c1 * alpha * dir_deriv);
}

#[test]
fn test_backtracking_line_search_rosenbrock() {
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

    let ls = BacktrackingLineSearch::default();
    let x = Vector::from_slice(&[0.0, 0.0]);
    let g = grad(&x);
    let d = Vector::from_slice(&[-g[0], -g[1]]); // Descent direction

    let alpha = ls.search(&f, &grad, &x, &d);

    assert!(alpha > 0.0);
}

#[test]
fn test_backtracking_line_search_multidimensional() {
    // f(x) = ||x||^2
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

    let ls = BacktrackingLineSearch::default();
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let g = grad(&x);
    let d = Vector::from_slice(&[-g[0], -g[1], -g[2]]);

    let alpha = ls.search(&f, &grad, &x, &d);

    assert!(alpha > 0.0);
    assert!(alpha <= 1.0);
}

#[test]
fn test_wolfe_line_search_new() {
    let ls = WolfeLineSearch::new(1e-4, 0.9, 50);
    assert!((ls.c1 - 1e-4).abs() < 1e-10);
    assert!((ls.c2 - 0.9).abs() < 1e-10);
    assert_eq!(ls.max_iter, 50);
}

#[test]
#[should_panic(expected = "Wolfe conditions require 0 < c1 < c2 < 1")]
fn test_wolfe_line_search_invalid_c1_c2() {
    // c1 >= c2 should panic
    let _ = WolfeLineSearch::new(0.9, 0.5, 50);
}

#[test]
#[should_panic(expected = "Wolfe conditions require 0 < c1 < c2 < 1")]
fn test_wolfe_line_search_c1_negative() {
    let _ = WolfeLineSearch::new(-0.1, 0.9, 50);
}

#[test]
#[should_panic(expected = "Wolfe conditions require 0 < c1 < c2 < 1")]
fn test_wolfe_line_search_c2_too_large() {
    let _ = WolfeLineSearch::new(0.1, 1.5, 50);
}

#[test]
fn test_wolfe_line_search_default() {
    let ls = WolfeLineSearch::default();
    assert!((ls.c1 - 1e-4).abs() < 1e-10);
    assert!((ls.c2 - 0.9).abs() < 1e-10);
    assert_eq!(ls.max_iter, 50);
}

#[test]
fn test_wolfe_line_search_quadratic() {
    // Test on simple quadratic: f(x) = x^2
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let ls = WolfeLineSearch::default();
    let x = Vector::from_slice(&[2.0]);
    let d = Vector::from_slice(&[-4.0]); // Gradient direction at x=2

    let alpha = ls.search(&f, &grad, &x, &d);

    // Should find positive step size
    assert!(alpha > 0.0);

    // Verify both Wolfe conditions
    let x_new_data = x[0] + alpha * d[0];
    let x_new = Vector::from_slice(&[x_new_data]);
    let fx = f(&x);
    let fx_new = f(&x_new);
    let grad_x = grad(&x);
    let grad_new = grad(&x_new);
    let dir_deriv = grad_x[0] * d[0];
    let dir_deriv_new = grad_new[0] * d[0];

    // Armijo condition
    assert!(fx_new <= fx + ls.c1 * alpha * dir_deriv + 1e-6);

    // Curvature condition
    assert!(dir_deriv_new.abs() <= ls.c2 * dir_deriv.abs() + 1e-6);
}

#[test]
fn test_wolfe_line_search_multidimensional() {
    // f(x) = ||x||^2
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

    let ls = WolfeLineSearch::default();
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let g = grad(&x);
    let d = Vector::from_slice(&[-g[0], -g[1], -g[2]]);

    let alpha = ls.search(&f, &grad, &x, &d);

    assert!(alpha > 0.0);
}

#[test]
fn test_backtracking_vs_wolfe() {
    // Compare backtracking and Wolfe on same problem
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let bt = BacktrackingLineSearch::default();
    let wolfe = WolfeLineSearch::default();

    let x = Vector::from_slice(&[1.0, 1.0]);
    let g = grad(&x);
    let d = Vector::from_slice(&[-g[0], -g[1]]);

    let alpha_bt = bt.search(&f, &grad, &x, &d);
    let alpha_wolfe = wolfe.search(&f, &grad, &x, &d);

    // Both should find valid step sizes
    assert!(alpha_bt > 0.0);
    assert!(alpha_wolfe > 0.0);

    // Wolfe often finds larger steps due to curvature condition
    // but not always, so just verify both are reasonable
    assert!(alpha_bt <= 1.0);
}

// ==================== OptimizationResult Tests ====================

#[test]
fn test_optimization_result_converged() {
    let solution = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = OptimizationResult::converged(solution.clone(), 42);

    assert_eq!(result.solution.len(), 3);
    assert_eq!(result.iterations, 42);
    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.objective_value - 0.0).abs() < 1e-10);
    assert!((result.gradient_norm - 0.0).abs() < 1e-10);
}

#[test]
fn test_optimization_result_max_iterations() {
    let solution = Vector::from_slice(&[5.0]);
    let result = OptimizationResult::max_iterations(solution);

    assert_eq!(result.iterations, 0);
    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
}

#[test]
fn test_convergence_status_equality() {
    assert_eq!(ConvergenceStatus::Converged, ConvergenceStatus::Converged);
    assert_ne!(
        ConvergenceStatus::Converged,
        ConvergenceStatus::MaxIterations
    );
    assert_ne!(
        ConvergenceStatus::Stalled,
        ConvergenceStatus::NumericalError
    );
}

// ==================== L-BFGS Tests ====================

#[test]
fn test_lbfgs_new() {
    let optimizer = LBFGS::new(100, 1e-5, 10);
    assert_eq!(optimizer.max_iter, 100);
    assert!((optimizer.tol - 1e-5).abs() < 1e-10);
    assert_eq!(optimizer.m, 10);
    assert_eq!(optimizer.s_history.len(), 0);
    assert_eq!(optimizer.y_history.len(), 0);
}

#[test]
fn test_lbfgs_simple_quadratic() {
    // Minimize f(x) = x^2, optimal at x = 0
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let mut optimizer = LBFGS::new(100, 1e-5, 5);
    let x0 = Vector::from_slice(&[5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 1e-3);
    assert!(result.iterations < 100);
    assert!(result.gradient_norm < 1e-5);
}

#[test]
fn test_lbfgs_multidimensional_quadratic() {
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

    let mut optimizer = LBFGS::new(100, 1e-5, 10);
    let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    for (i, &target) in c.iter().enumerate().take(3) {
        assert!((result.solution[i] - target).abs() < 1e-3);
    }
}

#[test]
fn test_lbfgs_rosenbrock() {
    // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // Global minimum at (1, 1)
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

    let mut optimizer = LBFGS::new(200, 1e-4, 10);
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
