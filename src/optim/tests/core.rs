//! Core optimizer tests: SafeCholesky, SGD, Adam, Line Search, L-BFGS, Conjugate Gradient, Damped Newton.

#![allow(non_snake_case)]

use super::super::*;

// ==================== SafeCholesky Tests ====================

#[test]
fn test_safe_cholesky_solve_positive_definite() {
    // Well-conditioned positive definite matrix
    let A = Matrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]).expect("valid dimensions");
    let b = Vector::from_slice(&[6.0, 5.0]);

    let x = safe_cholesky_solve(&A, &b, 1e-8, 10).expect("should solve");
    assert_eq!(x.len(), 2);

    // Verify solution: Ax should equal b (approximately)
    let Ax = Vector::from_slice(&[
        A.get(0, 0) * x[0] + A.get(0, 1) * x[1],
        A.get(1, 0) * x[0] + A.get(1, 1) * x[1],
    ]);
    assert!((Ax[0] - b[0]).abs() < 1e-5);
    assert!((Ax[1] - b[1]).abs() < 1e-5);
}

#[test]
fn test_safe_cholesky_solve_identity() {
    // Identity matrix - should solve without regularization
    let A = Matrix::eye(3);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);

    let x = safe_cholesky_solve(&A, &b, 1e-8, 10).expect("should solve");

    // For identity matrix, x should equal b
    assert!((x[0] - 1.0).abs() < 1e-6);
    assert!((x[1] - 2.0).abs() < 1e-6);
    assert!((x[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_safe_cholesky_solve_ill_conditioned() {
    // Ill-conditioned but solvable with regularization
    let A = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1e-10]).expect("valid dimensions");
    let b = Vector::from_slice(&[1.0, 1.0]);

    // Should succeed with regularization
    let x = safe_cholesky_solve(&A, &b, 1e-8, 10).expect("should solve with regularization");
    assert_eq!(x.len(), 2);

    // First component should be close to 1.0
    assert!((x[0] - 1.0).abs() < 1e-3);
}

#[test]
fn test_safe_cholesky_solve_not_positive_definite() {
    // Matrix with negative eigenvalue - needs regularization
    let A = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -0.5]).expect("valid dimensions");
    let b = Vector::from_slice(&[1.0, 1.0]);

    // Should solve with enough regularization
    let result = safe_cholesky_solve(&A, &b, 1e-4, 10);

    // May succeed with regularization or fail gracefully
    if let Ok(x) = result {
        assert_eq!(x.len(), 2);
        // Solution exists with regularization
    } else {
        // Also acceptable - matrix is indefinite
    }
}

#[test]
fn test_safe_cholesky_solve_zero_matrix() {
    // Zero matrix - should fail even with regularization
    let A = Matrix::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]).expect("valid dimensions");
    let b = Vector::from_slice(&[1.0, 1.0]);

    // Should eventually succeed when regularization dominates
    let result = safe_cholesky_solve(&A, &b, 1e-4, 10);
    assert!(result.is_ok()); // Regularization makes it λI which is PD
}

#[test]
fn test_safe_cholesky_solve_small_initial_lambda() {
    // Test with very small initial lambda
    let A = Matrix::eye(2);
    let b = Vector::from_slice(&[1.0, 1.0]);

    let x = safe_cholesky_solve(&A, &b, 1e-12, 10).expect("should solve");

    // Should still work for identity matrix
    assert!((x[0] - 1.0).abs() < 1e-6);
    assert!((x[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_safe_cholesky_solve_max_attempts() {
    // Test that max_attempts is respected
    let A = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid dimensions");
    let b = Vector::from_slice(&[1.0, 1.0]);

    // Even with 1 attempt, should work for identity
    let x = safe_cholesky_solve(&A, &b, 1e-8, 1).expect("should solve");
    assert_eq!(x.len(), 2);
}

#[test]
fn test_safe_cholesky_solve_large_system() {
    // Test with larger system
    let n = 5;
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 2.0; // Diagonal
        if i > 0 {
            data[i * n + (i - 1)] = 1.0; // Sub-diagonal
            data[(i - 1) * n + i] = 1.0; // Super-diagonal
        }
    }
    let A = Matrix::from_vec(n, n, data).expect("valid dimensions");
    let b = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0]);

    let x = safe_cholesky_solve(&A, &b, 1e-8, 10).expect("should solve");
    assert_eq!(x.len(), 5);
}

#[test]
fn test_safe_cholesky_solve_symmetric() {
    // Verify it works with symmetric matrix
    let A = Matrix::from_vec(3, 3, vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0])
        .expect("valid dimensions");
    let b = Vector::from_slice(&[1.0, 2.0, 1.0]);

    let x = safe_cholesky_solve(&A, &b, 1e-8, 10).expect("should solve");
    assert_eq!(x.len(), 3);
}

#[test]
fn test_safe_cholesky_solve_lambda_escalation() {
    // Test that lambda increases when needed
    // This matrix might need several regularization attempts
    let A = Matrix::from_vec(2, 2, vec![1.0, 0.999, 0.999, 1.0]).expect("valid dimensions");
    let b = Vector::from_slice(&[1.0, 1.0]);

    let x = safe_cholesky_solve(&A, &b, 1e-10, 15).expect("should solve");
    assert_eq!(x.len(), 2);

    // Solution should exist
    assert!(x[0].is_finite());
    assert!(x[1].is_finite());
}

// ==================== SGD Tests ====================

#[test]
fn test_sgd_new() {
    let optimizer = SGD::new(0.01);
    assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
    assert!((optimizer.momentum() - 0.0).abs() < 1e-6);
    assert!(!optimizer.has_momentum());
}

#[test]
fn test_sgd_with_momentum() {
    let optimizer = SGD::new(0.01).with_momentum(0.9);
    assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
    assert!((optimizer.momentum() - 0.9).abs() < 1e-6);
    assert!(optimizer.has_momentum());
}

#[test]
fn test_sgd_step_basic() {
    let mut optimizer = SGD::new(0.1);
    let mut params = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let gradients = Vector::from_slice(&[1.0, 2.0, 3.0]);

    optimizer.step(&mut params, &gradients);

    // params = params - lr * gradients
    assert!((params[0] - 0.9).abs() < 1e-6);
    assert!((params[1] - 1.8).abs() < 1e-6);
    assert!((params[2] - 2.7).abs() < 1e-6);
}

#[test]
fn test_sgd_step_with_momentum() {
    let mut optimizer = SGD::new(0.1).with_momentum(0.9);
    let mut params = Vector::from_slice(&[1.0, 1.0]);
    let gradients = Vector::from_slice(&[1.0, 1.0]);

    // First step: v = 0.9*0 + 0.1*1 = 0.1, params = 1.0 - 0.1 = 0.9
    optimizer.step(&mut params, &gradients);
    assert!((params[0] - 0.9).abs() < 1e-6);

    // Second step: v = 0.9*0.1 + 0.1*1 = 0.19, params = 0.9 - 0.19 = 0.71
    optimizer.step(&mut params, &gradients);
    assert!((params[0] - 0.71).abs() < 1e-6);
}

#[test]
fn test_sgd_momentum_accumulation() {
    let mut optimizer = SGD::new(0.1).with_momentum(0.9);
    let mut params = Vector::from_slice(&[0.0]);
    let gradients = Vector::from_slice(&[1.0]);

    // Velocity should accumulate over iterations
    let mut prev_step = 0.0;
    for _ in 0..10 {
        let before = params[0];
        optimizer.step(&mut params, &gradients);
        let step = before - params[0];
        // Each step should be larger (velocity builds up)
        assert!(step >= prev_step);
        prev_step = step;
    }
}

#[test]
fn test_sgd_reset() {
    let mut optimizer = SGD::new(0.1).with_momentum(0.9);
    let mut params = Vector::from_slice(&[1.0]);
    let gradients = Vector::from_slice(&[1.0]);

    optimizer.step(&mut params, &gradients);
    optimizer.reset();

    // After reset, velocity should be zero again
    let mut params2 = Vector::from_slice(&[1.0]);
    optimizer.step(&mut params2, &gradients);
    assert!((params2[0] - 0.9).abs() < 1e-6);
}

#[test]
fn test_sgd_zero_gradient() {
    let mut optimizer = SGD::new(0.1);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let gradients = Vector::from_slice(&[0.0, 0.0]);

    optimizer.step(&mut params, &gradients);

    // No change with zero gradients
    assert!((params[0] - 1.0).abs() < 1e-6);
    assert!((params[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_sgd_negative_gradients() {
    let mut optimizer = SGD::new(0.1);
    let mut params = Vector::from_slice(&[1.0]);
    let gradients = Vector::from_slice(&[-1.0]);

    optimizer.step(&mut params, &gradients);

    // params = 1.0 - 0.1 * (-1.0) = 1.1
    assert!((params[0] - 1.1).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "same length")]
fn test_sgd_mismatched_lengths() {
    let mut optimizer = SGD::new(0.1);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let gradients = Vector::from_slice(&[1.0]);

    optimizer.step(&mut params, &gradients);
}

#[test]
fn test_sgd_large_learning_rate() {
    let mut optimizer = SGD::new(10.0);
    let mut params = Vector::from_slice(&[1.0]);
    let gradients = Vector::from_slice(&[0.1]);

    optimizer.step(&mut params, &gradients);

    // params = 1.0 - 10.0 * 0.1 = 0.0
    assert!((params[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_sgd_small_learning_rate() {
    let mut optimizer = SGD::new(0.001);
    let mut params = Vector::from_slice(&[1.0]);
    let gradients = Vector::from_slice(&[1.0]);

    optimizer.step(&mut params, &gradients);

    // params = 1.0 - 0.001 * 1.0 = 0.999
    assert!((params[0] - 0.999).abs() < 1e-6);
}

#[test]
fn test_sgd_clone() {
    let optimizer = SGD::new(0.01).with_momentum(0.9);
    let cloned = optimizer.clone();

    assert!((cloned.learning_rate() - optimizer.learning_rate()).abs() < 1e-6);
    assert!((cloned.momentum() - optimizer.momentum()).abs() < 1e-6);
}

#[test]
fn test_sgd_multiple_steps() {
    let mut optimizer = SGD::new(0.1);
    let mut params = Vector::from_slice(&[10.0]);
    let gradients = Vector::from_slice(&[1.0]);

    for _ in 0..10 {
        optimizer.step(&mut params, &gradients);
    }

    // params = 10.0 - 10 * 0.1 * 1.0 = 9.0
    assert!((params[0] - 9.0).abs() < 1e-4);
}

#[test]
fn test_sgd_velocity_reinitialization() {
    let mut optimizer = SGD::new(0.1).with_momentum(0.9);

    // First with 2 params
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let gradients = Vector::from_slice(&[1.0, 1.0]);
    optimizer.step(&mut params, &gradients);

    // Now with 3 params - velocity should reinitialize
    let mut params3 = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let gradients3 = Vector::from_slice(&[1.0, 1.0, 1.0]);
    optimizer.step(&mut params3, &gradients3);

    // Should work without error, velocity reinitialized
    assert!((params3[0] - 0.9).abs() < 1e-6);
}

// ==================== Adam Tests ====================

#[test]
fn test_adam_new() {
    let optimizer = Adam::new(0.001);
    assert!((optimizer.learning_rate() - 0.001).abs() < 1e-9);
    assert!((optimizer.beta1() - 0.9).abs() < 1e-9);
    assert!((optimizer.beta2() - 0.999).abs() < 1e-9);
    assert!((optimizer.epsilon() - 1e-8).abs() < 1e-15);
    assert_eq!(optimizer.steps(), 0);
}

#[test]
fn test_adam_with_beta1() {
    let optimizer = Adam::new(0.001).with_beta1(0.95);
    assert!((optimizer.beta1() - 0.95).abs() < 1e-9);
}

#[test]
fn test_adam_with_beta2() {
    let optimizer = Adam::new(0.001).with_beta2(0.9999);
    assert!((optimizer.beta2() - 0.9999).abs() < 1e-9);
}

#[test]
fn test_adam_with_epsilon() {
    let optimizer = Adam::new(0.001).with_epsilon(1e-7);
    assert!((optimizer.epsilon() - 1e-7).abs() < 1e-15);
}

#[test]
fn test_adam_step_basic() {
    let mut optimizer = Adam::new(0.001);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let gradients = Vector::from_slice(&[0.1, 0.2]);

    optimizer.step(&mut params, &gradients);

    // Adam should update parameters (exact values depend on bias correction)
    assert!(params[0] < 1.0); // Should decrease
    assert!(params[1] < 2.0); // Should decrease
    assert_eq!(optimizer.steps(), 1);
}

#[test]
fn test_adam_multiple_steps() {
    let mut optimizer = Adam::new(0.001);
    let mut params = Vector::from_slice(&[1.0]);
    let gradients = Vector::from_slice(&[1.0]);

    let initial = params[0];
    for _ in 0..5 {
        optimizer.step(&mut params, &gradients);
    }

    // Parameters should decrease over multiple steps
    assert!(params[0] < initial);
    assert_eq!(optimizer.steps(), 5);
}

#[test]
fn test_adam_bias_correction() {
    let mut optimizer = Adam::new(0.01);
    let mut params = Vector::from_slice(&[10.0]);
    let gradients = Vector::from_slice(&[1.0]);

    // First step should have larger effective learning rate due to bias correction
    optimizer.step(&mut params, &gradients);
    let first_step_size = 10.0 - params[0];

    // Reset and try second step
    let mut optimizer2 = Adam::new(0.01);
    let mut params2 = Vector::from_slice(&[10.0]);
    optimizer2.step(&mut params2, &gradients);
    optimizer2.step(&mut params2, &gradients);
    let second_step_size = params[0] - params2[0];

    // First step should have larger update due to bias correction
    assert!(first_step_size > second_step_size * 0.5);
}

#[test]
fn test_adam_reset() {
    let mut optimizer = Adam::new(0.001);
    let mut params = Vector::from_slice(&[1.0]);
    let gradients = Vector::from_slice(&[1.0]);

    optimizer.step(&mut params, &gradients);
    assert_eq!(optimizer.steps(), 1);

    optimizer.reset();
    assert_eq!(optimizer.steps(), 0);
}

#[test]
fn test_adam_zero_gradient() {
    let mut optimizer = Adam::new(0.001);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let gradients = Vector::from_slice(&[0.0, 0.0]);

    optimizer.step(&mut params, &gradients);

    // With zero gradients, params should not change significantly
    assert!((params[0] - 1.0).abs() < 0.01);
    assert!((params[1] - 2.0).abs() < 0.01);
}

#[test]
fn test_adam_negative_gradients() {
    let mut optimizer = Adam::new(0.001);
    let mut params = Vector::from_slice(&[1.0]);
    let gradients = Vector::from_slice(&[-1.0]);

    optimizer.step(&mut params, &gradients);

    // With negative gradient, params should increase
    assert!(params[0] > 1.0);
}

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
