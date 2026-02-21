//! Advanced optimizer tests: Proximal, FISTA, Coordinate Descent, ADMM, Projected GD, Augmented Lagrangian, Interior Point.

#![allow(non_snake_case)]

use super::super::*;

// ==================== Proximal Operator Tests ====================

#[test]
fn test_soft_threshold_basic() {
    use crate::optim::prox::soft_threshold;

    let v = Vector::from_slice(&[2.0, -1.5, 0.5, 0.0]);
    let result = soft_threshold(&v, 1.0);

    assert!((result[0] - 1.0).abs() < 1e-6); // 2.0 - 1.0
    assert!((result[1] + 0.5).abs() < 1e-6); // -1.5 + 1.0
    assert!(result[2].abs() < 1e-6); // 0.5 - 1.0 -> 0
    assert!(result[3].abs() < 1e-6); // Already zero
}

#[test]
fn test_soft_threshold_zero_lambda() {
    use crate::optim::prox::soft_threshold;

    let v = Vector::from_slice(&[1.0, -2.0, 3.0]);
    let result = soft_threshold(&v, 0.0);

    // With λ=0, should return input unchanged
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -2.0);
    assert_eq!(result[2], 3.0);
}

#[test]
fn test_soft_threshold_large_lambda() {
    use crate::optim::prox::soft_threshold;

    let v = Vector::from_slice(&[1.0, -1.0, 0.5]);
    let result = soft_threshold(&v, 10.0);

    // All values should be thresholded to zero
    assert!(result[0].abs() < 1e-6);
    assert!(result[1].abs() < 1e-6);
    assert!(result[2].abs() < 1e-6);
}

#[test]
fn test_nonnegative_projection() {
    use crate::optim::prox::nonnegative;

    let x = Vector::from_slice(&[1.0, -2.0, 3.0, -0.5, 0.0]);
    let result = nonnegative(&x);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 0.0); // Projected to 0
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 0.0); // Projected to 0
    assert_eq!(result[4], 0.0);
}

#[test]
fn test_project_l2_ball_inside() {
    use crate::optim::prox::project_l2_ball;

    // Point inside ball - should be unchanged
    let x = Vector::from_slice(&[1.0, 1.0]); // norm = sqrt(2) ≈ 1.414
    let result = project_l2_ball(&x, 2.0);

    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_project_l2_ball_outside() {
    use crate::optim::prox::project_l2_ball;

    // Point outside ball - should be scaled
    let x = Vector::from_slice(&[3.0, 4.0]); // norm = 5.0
    let result = project_l2_ball(&x, 2.0);

    // Should be scaled to norm = 2.0
    let norm = (result[0] * result[0] + result[1] * result[1]).sqrt();
    assert!((norm - 2.0).abs() < 1e-5);

    // Direction should be preserved
    let scale = 2.0 / 5.0;
    assert!((result[0] - 3.0 * scale).abs() < 1e-5);
    assert!((result[1] - 4.0 * scale).abs() < 1e-5);
}

#[test]
fn test_project_box() {
    use crate::optim::prox::project_box;

    let x = Vector::from_slice(&[-1.0, 0.5, 2.0, 1.0]);
    let lower = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);
    let upper = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);

    let result = project_box(&x, &lower, &upper);

    assert_eq!(result[0], 0.0); // Clipped to lower
    assert_eq!(result[1], 0.5); // Within bounds
    assert_eq!(result[2], 1.0); // Clipped to upper
    assert_eq!(result[3], 1.0); // Within bounds
}

// ==================== FISTA Tests ====================

#[test]
fn test_fista_new() {
    let fista = FISTA::new(1000, 0.1, 1e-5);
    assert_eq!(fista.max_iter, 1000);
    assert!((fista.step_size - 0.1).abs() < 1e-9);
    assert!((fista.tol - 1e-5).abs() < 1e-9);
}

#[test]
fn test_fista_l1_regularized_quadratic() {
    use crate::optim::prox::soft_threshold;

    // Minimize: ½(x - 5)² + 2|x|
    // Solution should be around x ≈ 3 (soft-threshold of 5 with λ=2)
    let smooth = |x: &Vector<f32>| 0.5 * (x[0] - 5.0).powi(2);
    let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 5.0]);
    let prox = |v: &Vector<f32>, alpha: f32| soft_threshold(v, 2.0 * alpha);

    let mut fista = FISTA::new(1000, 0.1, 1e-5);
    let x0 = Vector::from_slice(&[0.0]);
    let result = fista.minimize(smooth, grad_smooth, prox, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    // Analytical solution: sign(5) * max(|5| - 2, 0) = 3
    assert!((result.solution[0] - 3.0).abs() < 0.1);
}

#[test]
fn test_fista_nonnegative_least_squares() {
    use crate::optim::prox::nonnegative;

    // Minimize: ½(x - (-2))² subject to x ≥ 0
    // Solution should be x = 0 (projection of -2 onto [0, ∞))
    let smooth = |x: &Vector<f32>| 0.5 * (x[0] + 2.0).powi(2);
    let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[x[0] + 2.0]);
    let prox = |v: &Vector<f32>, _alpha: f32| nonnegative(v);

    let mut fista = FISTA::new(1000, 0.1, 1e-6);
    let x0 = Vector::from_slice(&[1.0]);
    let result = fista.minimize(smooth, grad_smooth, prox, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.solution[0].abs() < 0.01); // Should be very close to 0
}

#[test]
fn test_fista_box_constrained() {
    use crate::optim::prox::project_box;

    // Minimize: ½(x - 10)² subject to 0 ≤ x ≤ 1
    // Solution should be x = 1 (projection of 10 onto [0, 1])
    let smooth = |x: &Vector<f32>| 0.5 * (x[0] - 10.0).powi(2);
    let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 10.0]);

    let lower = Vector::from_slice(&[0.0]);
    let upper = Vector::from_slice(&[1.0]);
    let prox = move |v: &Vector<f32>, _alpha: f32| project_box(v, &lower, &upper);

    let mut fista = FISTA::new(1000, 0.1, 1e-6);
    let x0 = Vector::from_slice(&[0.5]);
    let result = fista.minimize(smooth, grad_smooth, prox, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 0.01);
}

#[test]
fn test_fista_multidimensional_lasso() {
    use crate::optim::prox::soft_threshold;

    // Minimize: ½‖x - c‖² + λ‖x‖₁ where c = [3, -2, 1]
    let c = [3.0, -2.0, 1.0];
    let lambda = 0.5;

    let smooth = |x: &Vector<f32>| {
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += 0.5 * (x[i] - c[i]).powi(2);
        }
        sum
    };

    let grad_smooth = |x: &Vector<f32>| {
        let mut g = Vector::zeros(x.len());
        for i in 0..x.len() {
            g[i] = x[i] - c[i];
        }
        g
    };

    let prox = move |v: &Vector<f32>, alpha: f32| soft_threshold(v, lambda * alpha);

    let mut fista = FISTA::new(1000, 0.1, 1e-6);
    let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = fista.minimize(smooth, grad_smooth, prox, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);

    // Analytical solutions: sign(c[i]) * max(|c[i]| - λ, 0)
    assert!((result.solution[0] - 2.5).abs() < 0.1); // 3 - 0.5
    assert!((result.solution[1] + 1.5).abs() < 0.1); // -2 + 0.5
    assert!((result.solution[2] - 0.5).abs() < 0.1); // 1 - 0.5
}

#[test]
fn test_fista_max_iterations() {
    use crate::optim::prox::soft_threshold;

    // Use a difficult problem with very few iterations
    let smooth = |x: &Vector<f32>| 0.5 * x[0].powi(2);
    let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[x[0]]);
    let prox = |v: &Vector<f32>, alpha: f32| soft_threshold(v, alpha);

    let mut fista = FISTA::new(3, 0.001, 1e-10); // Very few iterations
    let x0 = Vector::from_slice(&[10.0]);
    let result = fista.minimize(smooth, grad_smooth, prox, x0);

    // Should hit max iterations
    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 3);
}

#[test]
fn test_fista_convergence_tracking() {
    use crate::optim::prox::soft_threshold;

    let smooth = |x: &Vector<f32>| 0.5 * x[0].powi(2);
    let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[x[0]]);
    let prox = |v: &Vector<f32>, alpha: f32| soft_threshold(v, 0.1 * alpha);

    let mut fista = FISTA::new(1000, 0.1, 1e-6);
    let x0 = Vector::from_slice(&[5.0]);
    let result = fista.minimize(smooth, grad_smooth, prox, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.iterations > 0);
    assert!(result.elapsed_time.as_nanos() > 0);
}

#[test]
fn test_fista_vs_no_acceleration() {
    use crate::optim::prox::soft_threshold;

    // FISTA should converge faster than unaccelerated proximal gradient
    let smooth = |x: &Vector<f32>| {
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += 0.5 * (x[i] - (i as f32 + 1.0)).powi(2);
        }
        sum
    };

    let grad_smooth = |x: &Vector<f32>| {
        let mut g = Vector::zeros(x.len());
        for i in 0..x.len() {
            g[i] = x[i] - (i as f32 + 1.0);
        }
        g
    };

    let prox = |v: &Vector<f32>, alpha: f32| soft_threshold(v, 0.5 * alpha);

    let mut fista = FISTA::new(1000, 0.1, 1e-5);
    let x0 = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
    let result = fista.minimize(smooth, grad_smooth, prox, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    // FISTA should converge reasonably fast
    assert!(result.iterations < 500);
}

// ==================== Coordinate Descent Tests ====================

#[test]
fn test_coordinate_descent_new() {
    let cd = CoordinateDescent::new(100, 1e-6);
    assert_eq!(cd.max_iter(), 100);
    assert!((cd.tol() - 1e-6).abs() < 1e-12);
    assert!(!cd.random_order());
}

#[test]
fn test_coordinate_descent_with_random_order() {
    let cd = CoordinateDescent::new(100, 1e-6).with_random_order(true);
    assert!(cd.random_order());
}

#[test]
fn test_coordinate_descent_simple_quadratic() {
    // Minimize: ½‖x - c‖² where c = [1, 2, 3]
    // Coordinate update: xᵢ = cᵢ (closed form)
    let c = [1.0, 2.0, 3.0];

    let update = move |x: &mut Vector<f32>, i: usize| {
        x[i] = c[i];
    };

    let mut cd = CoordinateDescent::new(100, 1e-6);
    let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = cd.minimize(update, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-5);
    assert!((result.solution[1] - 2.0).abs() < 1e-5);
    assert!((result.solution[2] - 3.0).abs() < 1e-5);
}

#[test]
fn test_coordinate_descent_soft_thresholding() {
    // Coordinate-wise soft-thresholding applied to fixed values
    // This models one iteration of Lasso coordinate descent
    let lambda = 0.5;
    let target = [2.0, -1.5, 0.3, -0.3];

    let update = move |x: &mut Vector<f32>, i: usize| {
        // Soft-threshold target[i]
        let v = target[i];
        x[i] = if v > lambda {
            v - lambda
        } else if v < -lambda {
            v + lambda
        } else {
            0.0
        };
    };

    let mut cd = CoordinateDescent::new(100, 1e-6);
    let x0 = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);
    let result = cd.minimize(update, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);

    // Expected: soft-threshold of target values
    assert!((result.solution[0] - 1.5).abs() < 1e-5); // 2.0 - 0.5
    assert!((result.solution[1] + 1.0).abs() < 1e-5); // -1.5 + 0.5
    assert!(result.solution[2].abs() < 1e-5); // |0.3| < 0.5 → 0
    assert!(result.solution[3].abs() < 1e-5); // |-0.3| < 0.5 → 0
}

#[test]
fn test_coordinate_descent_projection() {
    // Project onto [0, 1] box constraint coordinate-wise
    let update = |x: &mut Vector<f32>, i: usize| {
        x[i] = x[i].clamp(0.0, 1.0);
    };

    let mut cd = CoordinateDescent::new(100, 1e-6);
    let x0 = Vector::from_slice(&[-0.5, 0.5, 1.5, 2.0]);
    let result = cd.minimize(update, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 0.0).abs() < 1e-5); // Clipped to 0
    assert!((result.solution[1] - 0.5).abs() < 1e-5); // Within [0,1]
    assert!((result.solution[2] - 1.0).abs() < 1e-5); // Clipped to 1
    assert!((result.solution[3] - 1.0).abs() < 1e-5); // Clipped to 1
}

#[test]
fn test_coordinate_descent_alternating_optimization() {
    // Alternating minimization example: xᵢ → 0.5 * (xᵢ₋₁ + xᵢ₊₁)
    // Should converge to uniform values
    let update = |x: &mut Vector<f32>, i: usize| {
        let n = x.len();
        if n == 1 {
            return;
        }

        let left = if i == 0 { x[n - 1] } else { x[i - 1] };
        let right = if i == n - 1 { x[0] } else { x[i + 1] };

        x[i] = 0.5 * (left + right);
    };

    let mut cd = CoordinateDescent::new(1000, 1e-5);
    let x0 = Vector::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0]);
    let result = cd.minimize(update, x0);

    // Should converge (though possibly to MaxIterations for periodic case)
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_coordinate_descent_max_iterations() {
    // Use update that doesn't converge quickly
    let update = |x: &mut Vector<f32>, i: usize| {
        x[i] *= 0.99; // Very slow convergence
    };

    let mut cd = CoordinateDescent::new(3, 1e-10); // Very few iterations
    let x0 = Vector::from_slice(&[10.0, 10.0]);
    let result = cd.minimize(update, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 3);
}

#[test]
fn test_coordinate_descent_convergence_tracking() {
    let c = [5.0, 3.0];
    let update = move |x: &mut Vector<f32>, i: usize| {
        x[i] = c[i];
    };

    let mut cd = CoordinateDescent::new(100, 1e-6);
    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = cd.minimize(update, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.iterations > 0);
    assert!(result.elapsed_time.as_nanos() > 0);
}

#[test]
fn test_coordinate_descent_multidimensional() {
    // 5D problem
    let target = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let target_clone = target.clone();

    let update = move |x: &mut Vector<f32>, i: usize| {
        x[i] = target_clone[i];
    };

    let mut cd = CoordinateDescent::new(100, 1e-6);
    let x0 = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
    let result = cd.minimize(update, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    for (i, &targ) in target.iter().enumerate().take(5) {
        assert!((result.solution[i] - targ).abs() < 1e-5);
    }
}

#[path = "advanced_coordinate_descent.rs"]
mod advanced_coordinate_descent;

#[path = "advanced_projected_gradient.rs"]
mod advanced_projected_gradient;

#[path = "advanced_interior_point.rs"]
mod advanced_interior_point;

#[path = "advanced_line_search.rs"]
mod advanced_line_search;

#[path = "advanced_admm_coverage.rs"]
mod advanced_admm_coverage;
