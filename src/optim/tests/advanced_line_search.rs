use crate::optim::*;
use crate::primitives::Vector;

// Test ProjectedGradientDescent line search with actual backtracking
#[test]
fn test_projected_gd_line_search_backtracking() {
    // Use a large initial step size that will require backtracking
    // Problem: minimize x² starting far from optimum with huge step
    let objective = |x: &Vector<f32>| x[0] * x[0];
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    let project = |x: &Vector<f32>| x.clone(); // No projection

    // Initial step=10.0 is too large for this problem (gradient = 2*10 = 20, step = 200!)
    // This will cause f(x_new) > f(x) triggering backtracking
    let mut pgd = ProjectedGradientDescent::new(100, 10.0, 1e-6).with_line_search(0.5);
    let x0 = Vector::from_slice(&[10.0]);
    let result = pgd.minimize(objective, gradient, project, x0);

    // Should still converge due to line search
    assert_eq!(result.status, ConvergenceStatus::Converged);
    // Solution should be near 0
    assert!(result.solution[0].abs() < 1e-3);
}

// Test ProjectedGradientDescent line search max backtracking iterations
#[test]
fn test_projected_gd_line_search_max_backtrack() {
    // Create a problem where backtracking won't help much
    // but the solver still needs to try multiple backtracking steps
    use std::cell::Cell;
    use std::rc::Rc;

    let call_count = Rc::new(Cell::new(0));
    let count_clone = Rc::clone(&call_count);

    // Objective that oscillates based on call count
    let objective = move |x: &Vector<f32>| {
        count_clone.set(count_clone.get() + 1);
        // Large value initially, then small
        x[0] * x[0] + if count_clone.get() < 5 { 1000.0 } else { 0.0 }
    };
    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
    let project = |x: &Vector<f32>| x.clone();

    // Large step will cause issues
    let mut pgd = ProjectedGradientDescent::new(100, 5.0, 1e-6).with_line_search(0.5);
    let x0 = Vector::from_slice(&[5.0]);
    let result = pgd.minimize(objective, gradient, project, x0);

    // Should eventually converge or hit max iterations
    assert!(result.iterations > 0);
    assert!(call_count.get() > 5); // Objective was called multiple times due to line search
}

// Test ConvergenceStatus variants
#[test]
fn test_convergence_status_all_variants() {
    // Test all variants exist and can be compared
    let statuses = [
        ConvergenceStatus::Converged,
        ConvergenceStatus::MaxIterations,
        ConvergenceStatus::Stalled,
        ConvergenceStatus::NumericalError,
        ConvergenceStatus::Running,
        ConvergenceStatus::UserTerminated,
    ];

    for status in &statuses {
        // Test Clone
        let cloned = *status;
        assert_eq!(*status, cloned);

        // Test Debug
        let debug_str = format!("{:?}", status);
        assert!(!debug_str.is_empty());
    }
}

// Test OptimizationResult fields
#[test]
fn test_optimization_result_fields() {
    let result = OptimizationResult::converged(Vector::from_slice(&[1.0, 2.0]), 10);
    assert_eq!(result.iterations, 10);
    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert_eq!(result.solution.len(), 2);
    assert!((result.objective_value - 0.0).abs() < 1e-6);
    assert!((result.gradient_norm - 0.0).abs() < 1e-6);
    assert!((result.constraint_violation - 0.0).abs() < 1e-6);
    assert_eq!(result.elapsed_time, std::time::Duration::ZERO);

    let result2 = OptimizationResult::max_iterations(Vector::from_slice(&[3.0]));
    assert_eq!(result2.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result2.iterations, 0);
}

// Test nonnegative projection with all negative
#[test]
fn test_nonnegative_all_negative() {
    let x = Vector::from_slice(&[-1.0, -2.0, -3.0, -0.1]);
    let result = prox::nonnegative(&x);

    for i in 0..result.len() {
        assert!(result[i].abs() < 1e-6);
    }
}

// Test nonnegative projection with all positive
#[test]
fn test_nonnegative_all_positive() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 0.1]);
    let result = prox::nonnegative(&x);

    for i in 0..result.len() {
        assert!((result[i] - x[i]).abs() < 1e-6);
    }
}

// Test project_l2_ball with zero vector
#[test]
fn test_project_l2_ball_zero_vector() {
    let x = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = prox::project_l2_ball(&x, 1.0);

    // Zero vector should stay zero
    for i in 0..result.len() {
        assert!(result[i].abs() < 1e-6);
    }
}

// ============================================================================
// Projected Gradient Descent Tests (Coverage for projected_gradient.rs)
// ============================================================================

#[test]
fn test_pgd_with_line_search_converges() {
    // Simple quadratic: minimize ½‖x - c‖² subject to x ≥ 0
    let c = Vector::from_slice(&[1.0, -2.0, 3.0, -1.0]);

    let objective = |x: &Vector<f32>| {
        let mut obj = 0.0;
        for i in 0..x.len() {
            let diff = x[i] - c[i];
            obj += 0.5 * diff * diff;
        }
        obj
    };

    let gradient = |x: &Vector<f32>| {
        let mut grad = Vector::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = x[i] - c[i];
        }
        grad
    };

    let project = |x: &Vector<f32>| prox::nonnegative(x);

    // Enable line search
    let mut pgd = ProjectedGradientDescent::new(100, 1.0, 1e-6).with_line_search(0.5);
    let x0 = Vector::zeros(4);
    let result = pgd.minimize(&objective, &gradient, &project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    // Solution should be max(c, 0) = [1.0, 0.0, 3.0, 0.0]
    assert!((result.solution[0] - 1.0).abs() < 1e-3);
    assert!(result.solution[1].abs() < 1e-3);
    assert!((result.solution[2] - 3.0).abs() < 1e-3);
    assert!(result.solution[3].abs() < 1e-3);
}

#[test]
fn test_pgd_line_search_triggers_backtracking() {
    // Quadratic with large initial step size to trigger backtracking
    let c = Vector::from_slice(&[2.0, 3.0]);

    let objective = |x: &Vector<f32>| {
        let mut obj = 0.0;
        for i in 0..x.len() {
            let diff = x[i] - c[i];
            obj += 0.5 * diff * diff;
        }
        obj
    };

    let gradient = |x: &Vector<f32>| {
        let mut grad = Vector::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = x[i] - c[i];
        }
        grad
    };

    let project = |x: &Vector<f32>| prox::nonnegative(x);

    // Large step size to trigger backtracking line search
    let mut pgd = ProjectedGradientDescent::new(200, 10.0, 1e-6).with_line_search(0.5);
    let x0 = Vector::zeros(2);
    let result = pgd.minimize(&objective, &gradient, &project, x0);

    // Should converge to [2.0, 3.0]
    assert!((result.solution[0] - 2.0).abs() < 1e-2);
    assert!((result.solution[1] - 3.0).abs() < 1e-2);
}

#[test]
fn test_pgd_max_iterations_reached() {
    // Poorly conditioned problem with tiny tolerance that won't converge in 5 iterations
    let c = Vector::from_slice(&[100.0, 100.0, 100.0, 100.0]);

    let objective = |x: &Vector<f32>| {
        let mut obj = 0.0;
        for i in 0..x.len() {
            let diff = x[i] - c[i];
            obj += 0.5 * diff * diff;
        }
        obj
    };

    let gradient = |x: &Vector<f32>| {
        let mut grad = Vector::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = x[i] - c[i];
        }
        grad
    };

    let project = |x: &Vector<f32>| prox::nonnegative(x);

    // Very few iterations with very small step size - won't converge
    let mut pgd = ProjectedGradientDescent::new(5, 0.01, 1e-10);
    let x0 = Vector::zeros(4);
    let result = pgd.minimize(&objective, &gradient, &project, x0);

    // Should reach max iterations
    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 5);
}

#[test]
fn test_pgd_reset() {
    let mut pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
    // Reset should not panic
    Optimizer::reset(&mut pgd);
}

#[test]
#[should_panic(expected = "Projected Gradient Descent does not support stochastic updates")]
fn test_pgd_step_not_implemented() {
    let mut pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
    let mut params = Vector::zeros(4);
    let grads = Vector::zeros(4);
    pgd.step(&mut params, &grads);
}

#[test]
fn test_pgd_struct_debug() {
    let pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
    let debug = format!("{:?}", pgd);
    assert!(debug.contains("ProjectedGradientDescent"));
}

#[test]
fn test_pgd_struct_clone() {
    let pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
    let cloned = pgd.clone();
    let debug1 = format!("{:?}", pgd);
    let debug2 = format!("{:?}", cloned);
    assert_eq!(debug1, debug2);
}

#[test]
fn test_pgd_with_line_search_builder() {
    let pgd = ProjectedGradientDescent::new(100, 1.0, 1e-6).with_line_search(0.3);
    let debug = format!("{:?}", pgd);
    assert!(debug.contains("use_line_search: true"));
    assert!(debug.contains("beta: 0.3"));
}

#[test]
fn test_pgd_without_line_search() {
    let c = Vector::from_slice(&[1.0, 2.0]);

    let objective = |x: &Vector<f32>| {
        let mut obj = 0.0;
        for i in 0..x.len() {
            let diff = x[i] - c[i];
            obj += 0.5 * diff * diff;
        }
        obj
    };

    let gradient = |x: &Vector<f32>| {
        let mut grad = Vector::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = x[i] - c[i];
        }
        grad
    };

    let project = |x: &Vector<f32>| x.clone();

    // Without line search (default)
    let mut pgd = ProjectedGradientDescent::new(1000, 0.5, 1e-6);
    let x0 = Vector::zeros(2);
    let result = pgd.minimize(&objective, &gradient, &project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-3);
    assert!((result.solution[1] - 2.0).abs() < 1e-3);
}

#[test]
fn test_pgd_line_search_max_backtracking() {
    // Create a scenario where line search might hit max iterations (20)
    // Using an objective where the gradient points in a direction that doesn't decrease
    // the objective when projected

    let objective = |x: &Vector<f32>| {
        // Objective that increases with any step
        let mut obj = 0.0;
        for i in 0..x.len() {
            obj += x[i] * x[i];
        }
        obj
    };

    let gradient = |x: &Vector<f32>| {
        let mut grad = Vector::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = 2.0 * x[i];
        }
        grad
    };

    // Project to L2 ball - ensures projection changes the point
    let project = |x: &Vector<f32>| prox::project_l2_ball(x, 0.5);

    let mut pgd = ProjectedGradientDescent::new(50, 1.0, 1e-6).with_line_search(0.9);
    let x0 = Vector::from_slice(&[0.5, 0.5]);
    let result = pgd.minimize(&objective, &gradient, &project, x0);

    // Result is valid regardless of convergence
    assert!(result.iterations > 0);
}

#[test]
fn test_pgd_gradient_norm_tracking() {
    let c = Vector::from_slice(&[1.0, 1.0]);

    let objective = |x: &Vector<f32>| {
        let mut obj = 0.0;
        for i in 0..x.len() {
            let diff = x[i] - c[i];
            obj += 0.5 * diff * diff;
        }
        obj
    };

    let gradient = |x: &Vector<f32>| {
        let mut grad = Vector::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = x[i] - c[i];
        }
        grad
    };

    let project = |x: &Vector<f32>| x.clone();

    let mut pgd = ProjectedGradientDescent::new(100, 0.5, 1e-6);
    let x0 = Vector::zeros(2);
    let result = pgd.minimize(&objective, &gradient, &project, x0);

    // At convergence, gradient norm should be small
    assert!(result.gradient_norm < 1e-3);
}

#[test]
fn test_pgd_elapsed_time_recorded() {
    let c = Vector::from_slice(&[1.0]);

    let objective = |x: &Vector<f32>| (x[0] - c[0]).powi(2);
    let gradient = |x: &Vector<f32>| {
        let mut grad = Vector::zeros(1);
        grad[0] = 2.0 * (x[0] - c[0]);
        grad
    };
    let project = |x: &Vector<f32>| x.clone();

    let mut pgd = ProjectedGradientDescent::new(100, 0.5, 1e-6);
    let x0 = Vector::zeros(1);
    let result = pgd.minimize(&objective, &gradient, &project, x0);

    // Elapsed time should be non-zero
    assert!(result.elapsed_time.as_nanos() > 0);
}

#[test]
fn test_pgd_constraint_violation_zero() {
    let c = Vector::from_slice(&[1.0]);

    let objective = |x: &Vector<f32>| (x[0] - c[0]).powi(2);
    let gradient = |x: &Vector<f32>| {
        let mut grad = Vector::zeros(1);
        grad[0] = 2.0 * (x[0] - c[0]);
        grad
    };
    let project = |x: &Vector<f32>| x.clone();

    let mut pgd = ProjectedGradientDescent::new(100, 0.5, 1e-6);
    let x0 = Vector::zeros(1);
    let result = pgd.minimize(&objective, &gradient, &project, x0);

    // Constraint violation should be zero (no constraints violated with identity projection)
    assert_eq!(result.constraint_violation, 0.0);
}
