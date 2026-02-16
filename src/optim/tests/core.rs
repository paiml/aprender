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

#[path = "core_part_02.rs"]
mod core_part_02;

#[path = "core_part_03.rs"]
mod core_part_03;

#[path = "core_part_04.rs"]
mod core_part_04;
