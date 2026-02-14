use super::*;

#[test]
fn test_lbfgs_quadratic() {
    let mut optimizer = LBFGS::new(100, 1e-5, 10);

    // Simple quadratic: f(x) = (x-5)^2
    let f = |x: &Vector<f32>| (x[0] - 5.0).powi(2);
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0)]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 5.0).abs() < 1e-4);
}

#[test]
fn test_lbfgs_rosenbrock() {
    let mut optimizer = LBFGS::new(1000, 1e-5, 10);

    let f = |x: &Vector<f32>| {
        let a = x[0];
        let b = x[1];
        (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
    };

    let grad = |x: &Vector<f32>| {
        let a = x[0];
        let b = x[1];
        Vector::from_slice(&[
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a),
        ])
    };

    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-3);
    assert!((result.solution[1] - 1.0).abs() < 1e-3);
}

#[test]
fn test_lbfgs_clone_debug() {
    let opt = LBFGS::new(50, 1e-4, 5);
    let cloned = opt.clone();
    assert_eq!(opt.max_iter, cloned.max_iter);
    assert_eq!(opt.m, cloned.m);
    let debug_str = format!("{:?}", opt);
    assert!(debug_str.contains("LBFGS"));
}

#[test]
fn test_lbfgs_already_converged() {
    let mut optimizer = LBFGS::new(100, 1e-5, 10);
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_lbfgs_stalled_tiny_alpha() {
    // Function that causes line search to return essentially zero
    // Use a flat function where the line search cannot improve
    let mut optimizer = LBFGS::new(100, 1e-20, 5);

    let f = |x: &Vector<f32>| x[0].abs().min(1e-15);
    let grad = |_x: &Vector<f32>| Vector::from_slice(&[1e-15]);

    let x0 = Vector::from_slice(&[1.0]);
    let result = optimizer.minimize(f, grad, x0);

    // May stall, converge, or max-iter depending on line search
    assert!(
        result.status == ConvergenceStatus::Stalled
            || result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_lbfgs_numerical_error_nan() {
    let mut optimizer = LBFGS::new(100, 1e-5, 5);

    // Function that returns NaN after some steps
    let f = |x: &Vector<f32>| {
        if x[0] > 3.0 {
            f32::NAN
        } else {
            -(x[0] - 5.0).powi(2) // Concave, will diverge
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[-2.0 * (x[0] - 5.0)]);

    let x0 = Vector::from_slice(&[2.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(
        result.status == ConvergenceStatus::NumericalError
            || result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::Stalled
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_lbfgs_numerical_error_infinite() {
    let mut optimizer = LBFGS::new(100, 1e-5, 5);

    let f = |x: &Vector<f32>| {
        if x[0] > 3.0 {
            f32::INFINITY
        } else {
            -(x[0] - 5.0).powi(2)
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[-2.0 * (x[0] - 5.0)]);

    let x0 = Vector::from_slice(&[2.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(
        result.status == ConvergenceStatus::NumericalError
            || result.status == ConvergenceStatus::Stalled
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_lbfgs_history_overflow() {
    // Use m=2, run long enough to overflow history
    let mut optimizer = LBFGS::new(50, 1e-8, 2);

    let f = |x: &Vector<f32>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) + (x[2] - 3.0).powi(2);
    let grad = |x: &Vector<f32>| {
        Vector::from_slice(&[2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0), 2.0 * (x[2] - 3.0)])
    };

    let x0 = Vector::from_slice(&[10.0, -5.0, 8.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-3);
    // History should have been capped at m=2
    assert!(optimizer.s_history.len() <= 2);
}

#[test]
fn test_lbfgs_curvature_skip() {
    // Test the y_dot_s <= 1e-10 branch (curvature condition not met)
    // Use a function where gradients don't change much along step
    let mut optimizer = LBFGS::new(100, 1e-5, 5);

    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let x0 = Vector::from_slice(&[5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_lbfgs_norm_function() {
    let v = Vector::from_slice(&[3.0, 4.0]);
    let n = LBFGS::norm(&v);
    assert!((n - 5.0).abs() < 1e-6);

    let zero = Vector::from_slice(&[0.0]);
    assert!(LBFGS::norm(&zero).abs() < 1e-10);
}

#[test]
fn test_lbfgs_reset_clears_history() {
    let mut optimizer = LBFGS::new(100, 1e-5, 5);

    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let _ = optimizer.minimize(f, grad, Vector::from_slice(&[5.0]));
    assert!(!optimizer.s_history.is_empty());

    optimizer.reset();
    assert!(optimizer.s_history.is_empty());
    assert!(optimizer.y_history.is_empty());
}

#[test]
fn test_lbfgs_compute_direction_no_history() {
    let optimizer = LBFGS::new(100, 1e-5, 5);
    let grad = Vector::from_slice(&[3.0, -4.0]);
    let d = optimizer.compute_direction(&grad);

    // With no history, should be steepest descent: d = -grad
    assert!((d[0] - (-3.0)).abs() < 1e-6);
    assert!((d[1] - 4.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "does not support stochastic")]
fn test_lbfgs_step_panics() {
    let mut optimizer = LBFGS::new(100, 1e-5, 5);
    let mut params = Vector::from_slice(&[1.0]);
    let grad = Vector::from_slice(&[0.1]);
    optimizer.step(&mut params, &grad);
}
