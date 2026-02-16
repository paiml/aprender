pub(crate) use super::*;

#[test]
fn test_ip_new() {
    let ip = InteriorPoint::new(50, 1e-6, 1.0);
    let debug_str = format!("{:?}", ip);
    assert!(debug_str.contains("InteriorPoint"));
    assert!(debug_str.contains("50"));
}

#[test]
fn test_ip_clone_debug() {
    let ip = InteriorPoint::new(50, 1e-6, 1.0);
    let cloned = ip.clone();
    let d1 = format!("{:?}", ip);
    let d2 = format!("{:?}", cloned);
    assert_eq!(d1, d2);
}

#[test]
fn test_ip_with_beta() {
    let ip = InteriorPoint::new(50, 1e-6, 1.0).with_beta(0.1);
    let debug_str = format!("{:?}", ip);
    assert!(debug_str.contains("0.1"));
}

#[test]
fn test_ip_nonnegative_quadratic() {
    // min x1^2 + x2^2 s.t. x >= 0
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);
    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(50, 1e-3, 1.0);
    let x0 = Vector::from_slice(&[1.0, 1.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    assert!(result.solution[0] >= -1e-2);
    assert!(result.solution[1] >= -1e-2);
}

#[test]
fn test_ip_max_iterations() {
    let objective = |x: &Vector<f32>| (x[0] - 5.0).powi(2);
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0)]);
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

    let mut ip = InteriorPoint::new(2, 1e-20, 1.0);
    let x0 = Vector::from_slice(&[1.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 2);
    assert!(result.objective_value.is_finite());
    assert!(result.gradient_norm >= 0.0);
}

#[test]
fn test_ip_converged_result_fields() {
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);
    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(100, 1e-3, 1.0).with_beta(0.1);
    let x0 = Vector::from_slice(&[1.0, 1.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    assert!(result.constraint_violation >= 0.0);
    let _ = result.elapsed_time.as_nanos();
}

#[test]
#[should_panic(expected = "Initial point is infeasible")]
fn test_ip_infeasible_start() {
    let objective = |x: &Vector<f32>| x[0] * x[0];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    // Constraint: -x <= 0, i.e. x >= 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

    let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
    let x0 = Vector::from_slice(&[-1.0]); // Infeasible: g(x) = 1.0 >= 0
    let _ = ip.minimize(objective, gradient, inequality, inequality_jac, x0);
}

#[test]
fn test_ip_reset() {
    let mut ip = InteriorPoint::new(50, 1e-6, 5.0);
    // Run a minimization to change mu
    let objective = |x: &Vector<f32>| x[0] * x[0];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

    let _ = ip.minimize(
        objective,
        gradient,
        inequality,
        inequality_jac,
        Vector::from_slice(&[1.0]),
    );

    // Reset should restore mu to initial value
    ip.reset();
    // The next call should start with initial_mu again
    let debug_str = format!("{:?}", ip);
    assert!(debug_str.contains("mu: 5.0"));
}

#[test]
#[should_panic(expected = "does not support stochastic updates")]
fn test_ip_step_panics() {
    let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
    let mut params = Vector::from_slice(&[1.0]);
    let grad = Vector::from_slice(&[0.1]);
    ip.step(&mut params, &grad);
}

#[test]
fn test_ip_constraint_boundary_hit() {
    // Problem where solution is at constraint boundary (barrier subproblem
    // exercises the infeasible branch within the inner loop)
    let objective = |x: &Vector<f32>| (x[0] - 10.0).powi(2);
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 10.0)]);
    // Constraint: x <= 2 => g(x) = x - 2 <= 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 2.0]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0])];

    let mut ip = InteriorPoint::new(100, 1e-3, 1.0).with_beta(0.1);
    let x0 = Vector::from_slice(&[1.0]); // feasible: g(1) = -1 < 0
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Solution should move towards the boundary x = 2 (interior point
    // uses small gradient steps, so it may not reach exactly)
    assert!(
        result.solution[0] > 0.5,
        "Expected solution > 0.5, got {}",
        result.solution[0]
    );
}

#[test]
fn test_ip_aggressive_beta() {
    // Use very aggressive beta to exercise fast mu decrease
    let objective = |x: &Vector<f32>| x[0] * x[0];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

    let mut ip = InteriorPoint::new(100, 1e-3, 10.0).with_beta(0.01);
    let x0 = Vector::from_slice(&[1.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Should converge since mu decreases quickly
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_ip_constraint_boundary_stepback() {
    // Design a problem that triggers the step-back logic when we approach
    // constraint boundaries during gradient descent.
    // Minimize (x - 0.5)^2 subject to x >= 0 (constraint: -x <= 0)
    // The objective minimum at x=0.5 is feasible, so it should converge there.
    let objective = |x: &Vector<f32>| (x[0] - 0.5).powi(2);
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 0.5)]);
    // Constraint: x >= 0 => g(x) = -x <= 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

    // Start close to boundary to trigger step-back logic
    let mut ip = InteriorPoint::new(100, 1e-4, 0.5).with_beta(0.3);
    let x0 = Vector::from_slice(&[0.01]); // Very close to boundary x=0
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Solution should move towards x=0.5 while respecting constraint
    assert!(
        result.solution[0] >= -0.01,
        "Solution {} should respect constraint x >= 0",
        result.solution[0]
    );
    assert!(result.objective_value.is_finite());
}

#[test]
fn test_ip_multi_constraint_with_boundary_skip() {
    // Multiple constraints where one becomes inactive (g >= 0 continue branch)
    // Minimize x^2 + y^2 subject to x >= -10, y >= -10 (loose constraints)
    // g(x) = [-x - 10, -y - 10] <= 0
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0] - 10.0, -x[1] - 10.0]);
    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(100, 1e-4, 1.0).with_beta(0.1);
    let x0 = Vector::from_slice(&[5.0, 5.0]); // Far from constraints
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Solution should approach origin since constraints are loose
    assert!(
        result.solution[0].abs() < 1.0,
        "x should approach 0, got {}",
        result.solution[0]
    );
    assert!(
        result.solution[1].abs() < 1.0,
        "y should approach 0, got {}",
        result.solution[1]
    );
}

#[test]
fn test_ip_early_gradient_convergence() {
    // Test early exit when gradient norm is very small in sub-iteration
    // Use a simple quadratic with tight constraints at origin
    let objective = |x: &Vector<f32>| x[0] * x[0];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    // Constraint: x >= 0 => g(x) = -x <= 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

    // Start very close to optimum
    let mut ip = InteriorPoint::new(50, 1e-6, 0.1).with_beta(0.1);
    let x0 = Vector::from_slice(&[0.001]); // Very close to optimum
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Should converge quickly
    assert!(result.solution[0] >= -1e-3);
    assert!(result.solution[0] < 0.1);
}

#[test]
fn test_ip_positive_max_violation() {
    // Design a problem that may have positive constraint violation during iteration
    // This exercises the max_violation > 0 path
    let objective = |x: &Vector<f32>| (x[0] - 100.0).powi(2);
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 100.0)]);
    // Tight constraint: x <= 5
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 5.0]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0])];

    let mut ip = InteriorPoint::new(20, 1e-4, 1.0).with_beta(0.3);
    let x0 = Vector::from_slice(&[4.0]); // Feasible start
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Result should have constraint_violation tracked
    assert!(result.constraint_violation >= 0.0);
    // Solution should be at or near boundary
    assert!(
        result.solution[0] >= 3.0,
        "Solution should be pushed towards boundary"
    );
}

#[test]
fn test_ip_three_constraints() {
    // Problem with three constraints to test loop iteration
    // Minimize x^2 + y^2 subject to x >= 0, y >= 0, x + y <= 3
    let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
    // g1: -x <= 0, g2: -y <= 0, g3: x + y - 3 <= 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1], x[0] + x[1] - 3.0]);
    let inequality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[-1.0, 0.0]),
            Vector::from_slice(&[0.0, -1.0]),
            Vector::from_slice(&[1.0, 1.0]),
        ]
    };

    let mut ip = InteriorPoint::new(100, 1e-3, 1.0).with_beta(0.2);
    let x0 = Vector::from_slice(&[1.0, 1.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Solution should be close to origin
    assert!(result.solution[0] >= -0.1);
    assert!(result.solution[1] >= -0.1);
    assert!(result.solution[0] + result.solution[1] <= 3.1);
}

#[test]
fn test_ip_hit_constraint_boundary_exactly() {
    // Create scenario where g_val[j] can become exactly 0 or very close
    // to trigger the g_val[j] >= 0.0 continue branch
    let objective = |x: &Vector<f32>| (x[0] - 10.0).powi(2);
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 10.0)]);
    // Constraint: x <= 2 => g(x) = x - 2 <= 0
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 2.0]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0])];

    // Use very aggressive parameters to potentially hit boundary
    let mut ip = InteriorPoint::new(100, 1e-6, 0.01).with_beta(0.01);
    let x0 = Vector::from_slice(&[1.9]); // Start very close to boundary
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    // Should reach boundary
    assert!(result.solution[0] >= 1.5, "Should move toward boundary");
    assert!(result.objective_value.is_finite());
}

#[test]
fn test_ip_max_iter_with_violation_tracking() {
    // Test that max_violation is correctly computed in the max iterations path
    // Use impossible convergence tolerance
    let objective = |x: &Vector<f32>| x[0] * x[0];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
    let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

    // Impossibly tight tolerance forces max iterations
    let mut ip = InteriorPoint::new(3, 1e-30, 1.0);
    let x0 = Vector::from_slice(&[5.0]);
    let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 3);
    // max_violation path should be exercised
    assert!(result.constraint_violation >= 0.0);
    assert!(result.gradient_norm >= 0.0);
}
