pub(crate) use super::*;

#[test]
fn test_cg_quadratic() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    // Simple quadratic: f(x) = (x-3)^2
    let f = |x: &Vector<f32>| (x[0] - 3.0).powi(2);
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 3.0)]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 3.0).abs() < 1e-4);
}

#[test]
fn test_cg_rosenbrock() {
    // Rosenbrock is a challenging test function; CG may need many iterations
    let mut optimizer = ConjugateGradient::new(5000, 1e-4, CGBetaFormula::PolakRibiere);

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

    // Check that we made progress toward (1, 1) even if not fully converged
    // Rosenbrock is notoriously difficult for CG
    let dist_to_opt =
        ((result.solution[0] - 1.0).powi(2) + (result.solution[1] - 1.0).powi(2)).sqrt();
    assert!(
        dist_to_opt < 0.1 || result.status == ConvergenceStatus::Converged,
        "Expected solution near (1,1), got ({}, {}), dist={}",
        result.solution[0],
        result.solution[1],
        dist_to_opt
    );
}

#[test]
fn test_cg_beta_formulas() {
    for formula in [
        CGBetaFormula::FletcherReeves,
        CGBetaFormula::PolakRibiere,
        CGBetaFormula::HestenesStiefel,
    ] {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, formula);

        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let x0 = Vector::from_slice(&[5.0, 3.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-4);
        assert!(result.solution[1].abs() < 1e-4);
    }
}

#[test]
fn test_cg_restart_interval() {
    let optimizer =
        ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere).with_restart_interval(10);
    assert_eq!(optimizer.restart_interval, 10);

    // Run optimization with restart interval
    let mut opt = optimizer;
    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
    let x0 = Vector::from_slice(&[5.0, 3.0]);
    let result = opt.minimize(f, grad, x0);
    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_cg_reset() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    // Run an optimization first
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    let x0 = Vector::from_slice(&[5.0]);
    let _ = optimizer.minimize(f, grad, x0);

    // Check state was set
    assert!(optimizer.prev_direction.is_some());

    // Reset and check state is cleared
    optimizer.reset();
    assert!(optimizer.prev_direction.is_none());
    assert!(optimizer.prev_gradient.is_none());
    assert_eq!(optimizer.iter_count, 0);
}

#[test]
#[should_panic(expected = "does not support stochastic")]
fn test_cg_step_panics() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let mut params = Vector::from_slice(&[1.0, 2.0]);
    let grads = Vector::from_slice(&[0.1, 0.2]);
    optimizer.step(&mut params, &grads);
}

#[test]
fn test_cg_numerical_error() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    // Function that returns NaN
    let f = |x: &Vector<f32>| {
        if x[0] > 0.5 {
            f32::NAN
        } else {
            x[0] * x[0]
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let x0 = Vector::from_slice(&[0.4]);
    let result = optimizer.minimize(f, grad, x0);
    // May hit NaN or converge depending on line search
    assert!(
        result.status == ConvergenceStatus::NumericalError
            || result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::Stalled
    );
}

#[test]
fn test_cg_max_iterations() {
    // Very tight tolerance that won't be reached with only 2 iterations
    let mut optimizer = ConjugateGradient::new(2, 1e-20, CGBetaFormula::PolakRibiere);

    // Rosenbrock is hard to converge in 2 iterations
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

    let x0 = Vector::from_slice(&[-5.0, -5.0]);
    let result = optimizer.minimize(f, grad, x0);
    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
}

#[test]
fn test_cg_hestenes_stiefel() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::HestenesStiefel);

    // Multi-dimensional problem to exercise HS formula
    let f = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1], 6.0 * x[2]]);

    let x0 = Vector::from_slice(&[5.0, 3.0, 2.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 1e-4);
    assert!(result.solution[1].abs() < 1e-4);
    assert!(result.solution[2].abs() < 1e-4);
}

#[test]
fn test_cg_fletcher_reeves() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);

    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let x0 = Vector::from_slice(&[10.0, 10.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_cg_beta_formula_equality() {
    assert_eq!(CGBetaFormula::FletcherReeves, CGBetaFormula::FletcherReeves);
    assert_ne!(CGBetaFormula::FletcherReeves, CGBetaFormula::PolakRibiere);

    // Test Clone
    let formula = CGBetaFormula::HestenesStiefel;
    let cloned = formula;
    assert_eq!(formula, cloned);

    // Test Debug
    let debug_str = format!("{:?}", CGBetaFormula::PolakRibiere);
    assert!(debug_str.contains("PolakRibiere"));
}

#[test]
fn test_cg_clone() {
    let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let cloned = optimizer.clone();
    assert_eq!(cloned.max_iter, 100);
    assert_eq!(cloned.beta_formula, CGBetaFormula::PolakRibiere);
}

#[test]
fn test_cg_debug() {
    let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    let debug_str = format!("{:?}", optimizer);
    assert!(debug_str.contains("ConjugateGradient"));
    assert!(debug_str.contains("max_iter"));
}

#[test]
fn test_cg_stalled_zero_alpha() {
    // Test stalled status when line search returns very small alpha
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    // Function that's extremely flat - line search should return tiny alpha
    let f = |x: &Vector<f32>| x[0] * 1e-20;
    let grad = |_x: &Vector<f32>| Vector::from_slice(&[1e-20]);

    let x0 = Vector::from_slice(&[1.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Should either converge (gradient too small) or stall
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::Stalled
    );
}

#[test]
fn test_cg_restart_triggers() {
    // Use restart interval of 2 and run enough iterations to trigger restart
    let mut optimizer =
        ConjugateGradient::new(20, 1e-8, CGBetaFormula::PolakRibiere).with_restart_interval(2);

    // Function that needs many iterations
    let f = |x: &Vector<f32>| {
        let a = x[0];
        let b = x[1];
        (a - 5.0).powi(2) + (b - 3.0).powi(2)
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0), 2.0 * (x[1] - 3.0)]);

    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Should converge despite restarts
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_cg_non_descent_direction_restart() {
    // Test the case where beta calculation leads to non-descent direction
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);

    // Function where gradients change rapidly
    let f = |x: &Vector<f32>| x[0].powi(4) + x[1].powi(4);
    let grad = |x: &Vector<f32>| Vector::from_slice(&[4.0 * x[0].powi(3), 4.0 * x[1].powi(3)]);

    let x0 = Vector::from_slice(&[10.0, 10.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
    // Should get close to (0, 0)
    assert!(result.solution[0].abs() < 1.0);
}

#[test]
fn test_cg_objective_value_tracking() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    let f = |x: &Vector<f32>| (x[0] - 2.0).powi(2);
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 2.0)]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(result.objective_value < 1e-4);
    assert!(result.constraint_violation == 0.0);
}

#[test]
fn test_cg_gradient_norm_tracking() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let x0 = Vector::from_slice(&[5.0, 3.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Gradient norm should be small at convergence
    assert!(result.gradient_norm < 1e-4);
}

#[test]
fn test_cg_elapsed_time() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let x0 = Vector::from_slice(&[1.0]);
    let result = optimizer.minimize(f, grad, x0);

    // Elapsed time should be tracked
    let _ = result.elapsed_time.as_nanos();
}

#[test]
fn test_cg_already_at_optimum() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    // Start at optimum
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_cg_infinite_objective() {
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    // Function that returns infinity after a few steps
    let f = |x: &Vector<f32>| {
        if x[0] > 2.0 {
            f32::INFINITY
        } else {
            x[0] * x[0]
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    // Start somewhere that would move towards infinity
    let x0 = Vector::from_slice(&[1.5]);
    let result = optimizer.minimize(f, grad, x0);

    // Should handle gracefully
    assert!(
        result.status == ConvergenceStatus::NumericalError
            || result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::Stalled
    );
}

#[test]
fn test_cg_norm() {
    // Test the private norm function via minimize that uses it
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    // Large initial gradient to ensure norm is calculated
    let f = |x: &Vector<f32>| (x[0] - 100.0).powi(2) + (x[1] - 200.0).powi(2);
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 100.0), 2.0 * (x[1] - 200.0)]);

    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(result.status == ConvergenceStatus::Converged);
    assert!((result.solution[0] - 100.0).abs() < 1e-3);
    assert!((result.solution[1] - 200.0).abs() < 1e-3);
}

#[test]
fn test_cg_with_restart_interval_zero() {
    // Zero restart interval should disable periodic restarts
    let optimizer =
        ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere).with_restart_interval(0);
    assert_eq!(optimizer.restart_interval, 0);
}

#[test]
fn test_cg_restart_interval_1() {
    // Restart every single iteration forces steepest descent behavior
    let mut optimizer =
        ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves).with_restart_interval(1);

    let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

    let x0 = Vector::from_slice(&[5.0, 3.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_cg_norm_function() {
    // Exercise the norm utility indirectly by running minimize
    // with a known-norm initial gradient
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    let f = |x: &Vector<f32>| (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2);
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 3.0), 2.0 * (x[1] - 4.0)]);

    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.gradient_norm < 1e-4);
}

#[path = "conjugate_gradient_tests_part_02.rs"]
mod conjugate_gradient_tests_part_02;
