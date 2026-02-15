
// ==================== Projected Gradient Descent Tests ====================

#[test]
fn test_projected_gd_nonnegative_constraint() {
    // Minimize: ½‖x - c‖² subject to x ≥ 0
    // Analytical solution: max(c, 0)
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

    let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
    let x0 = Vector::zeros(4);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-4); // max(1.0, 0) = 1.0
    assert!(result.solution[1].abs() < 1e-4); // max(-2.0, 0) = 0.0
    assert!((result.solution[2] - 3.0).abs() < 1e-4); // max(3.0, 0) = 3.0
    assert!(result.solution[3].abs() < 1e-4); // max(-1.0, 0) = 0.0
}

#[test]
fn test_projected_gd_box_constraints() {
    // Minimize: ½‖x - c‖² subject to 0 ≤ x ≤ 2
    let c = Vector::from_slice(&[1.5, -1.0, 3.0, 0.5]);
    let lower = Vector::zeros(4);
    let upper = Vector::from_slice(&[2.0, 2.0, 2.0, 2.0]);

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

    let lower_clone = lower.clone();
    let upper_clone = upper.clone();
    let project = move |x: &Vector<f32>| prox::project_box(x, &lower_clone, &upper_clone);

    let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
    let x0 = Vector::ones(4);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.5).abs() < 1e-4); // clamp(1.5, 0, 2) = 1.5
    assert!(result.solution[1].abs() < 1e-4); // clamp(-1.0, 0, 2) = 0.0
    assert!((result.solution[2] - 2.0).abs() < 1e-4); // clamp(3.0, 0, 2) = 2.0
    assert!((result.solution[3] - 0.5).abs() < 1e-4); // clamp(0.5, 0, 2) = 0.5
}

#[test]
fn test_projected_gd_l2_ball() {
    // Minimize: ½‖x - c‖² subject to ‖x‖₂ ≤ 1
    let c = Vector::from_slice(&[2.0, 2.0]);
    let radius = 1.0;

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

    let project = move |x: &Vector<f32>| prox::project_l2_ball(x, radius);

    let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
    let x0 = Vector::zeros(2);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);

    // Solution should be c/‖c‖₂ * radius = [2,2]/√8 = [√2/2, √2/2]
    let norm =
        (result.solution[0] * result.solution[0] + result.solution[1] * result.solution[1]).sqrt();
    assert!((norm - radius).abs() < 1e-4); // On boundary
    assert!((result.solution[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3); // √2/2
    assert!((result.solution[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
}

#[test]
fn test_projected_gd_with_line_search() {
    // Same problem as nonnegative, but with line search
    let c = Vector::from_slice(&[1.0, -2.0, 3.0]);

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

    let mut pgd = ProjectedGradientDescent::new(1000, 1.0, 1e-6).with_line_search(0.5);
    let x0 = Vector::zeros(3);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-4);
    assert!(result.solution[1].abs() < 1e-4);
    assert!((result.solution[2] - 3.0).abs() < 1e-4);
}

#[test]
fn test_projected_gd_quadratic() {
    // Minimize: ½xᵀQx - bᵀx subject to x ≥ 0
    // Q = [[2, 0], [0, 2]] (identity scaled by 2)
    // b = [4, -2]
    // Unconstrained solution: x = Q⁻¹b = [2, -1]
    // Constrained solution: x = [2, 0]

    let objective =
        |x: &Vector<f32>| 0.5 * (2.0 * x[0] * x[0] + 2.0 * x[1] * x[1]) - (4.0 * x[0] - 2.0 * x[1]);

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0] - 4.0, 2.0 * x[1] + 2.0]);

    let project = |x: &Vector<f32>| prox::nonnegative(x);

    let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
    let x0 = Vector::zeros(2);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 2.0).abs() < 1e-3);
    assert!(result.solution[1].abs() < 1e-3);
}

#[test]
fn test_projected_gd_convergence_tracking() {
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

    let project = |x: &Vector<f32>| prox::nonnegative(x);

    let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
    let x0 = Vector::zeros(2);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.iterations > 0);
    assert!(result.elapsed_time.as_nanos() > 0);
    assert!(result.gradient_norm < 1.0); // Should have small gradient at solution
}

#[test]
fn test_projected_gd_max_iterations() {
    // Use very tight tolerance to force max iterations
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

    let project = |x: &Vector<f32>| prox::nonnegative(x);

    let mut pgd = ProjectedGradientDescent::new(3, 0.01, 1e-12); // Very few iterations
    let x0 = Vector::zeros(2);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 3);
}

#[test]
fn test_projected_gd_unconstrained_equivalent() {
    // When projection is identity, should behave like gradient descent
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

    let project = |x: &Vector<f32>| x.clone(); // Identity projection

    let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
    let x0 = Vector::zeros(2);
    let result = pgd.minimize(objective, gradient, project, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-4);
    assert!((result.solution[1] - 2.0).abs() < 1e-4);
}

// ==================== Augmented Lagrangian Tests ====================

#[test]
fn test_augmented_lagrangian_linear_equality() {
    // Minimize: ½(x₁-2)² + ½(x₂-3)² subject to x₁ + x₂ = 1
    // Analytical solution: x = [2, 3] - λ[1, 1] where x₁+x₂=1
    // Solving: 2-λ + 3-λ = 1 → λ = 2, so x = [0, 1]

    let objective = |x: &Vector<f32>| 0.5 * (x[0] - 2.0).powi(2) + 0.5 * (x[1] - 3.0).powi(2);

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 2.0, x[1] - 3.0]);

    let equality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] - 1.0]);

    let equality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0, 1.0])];

    let mut al = AugmentedLagrangian::new(100, 1e-4, 1.0);
    let x0 = Vector::zeros(2);
    let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);

    // Check constraint satisfaction
    assert!(result.constraint_violation < 1e-3);
    // Check that x₁ + x₂ ≈ 1
    assert!((result.solution[0] + result.solution[1] - 1.0).abs() < 1e-3);
}

#[test]
fn test_augmented_lagrangian_multiple_constraints() {
    // Minimize: ½‖x‖² subject to x₁ + x₂ = 1, x₁ - x₂ = 0
    // This means x₁ = x₂ and x₁ + x₂ = 1, so x = [0.5, 0.5]

    let objective = |x: &Vector<f32>| 0.5 * (x[0] * x[0] + x[1] * x[1]);

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[x[0], x[1]]);

    let equality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] - 1.0, x[0] - x[1]]);

    let equality_jac = |_x: &Vector<f32>| {
        vec![
            Vector::from_slice(&[1.0, 1.0]),
            Vector::from_slice(&[1.0, -1.0]),
        ]
    };

    let mut al = AugmentedLagrangian::new(200, 1e-4, 1.0);
    let x0 = Vector::zeros(2);
    let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);

    assert!(result.constraint_violation < 1e-3);
    assert!((result.solution[0] - 0.5).abs() < 1e-2);
    assert!((result.solution[1] - 0.5).abs() < 1e-2);
}

#[test]
fn test_augmented_lagrangian_3d() {
    // Minimize: ½‖x - c‖² subject to x₁ + x₂ + x₃ = 1
    let c = Vector::from_slice(&[1.0, 2.0, 3.0]);

    let objective = |x: &Vector<f32>| {
        0.5 * ((x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2))
    };

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[x[0] - c[0], x[1] - c[1], x[2] - c[2]]);

    let equality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] + x[2] - 1.0]);

    let equality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0, 1.0, 1.0])];

    let mut al = AugmentedLagrangian::new(100, 1e-4, 1.0);
    let x0 = Vector::zeros(3);
    let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);

    assert!(result.constraint_violation < 1e-3);
    assert!((result.solution[0] + result.solution[1] + result.solution[2] - 1.0).abs() < 1e-3);
}

#[test]
fn test_augmented_lagrangian_quadratic_with_constraint() {
    // Minimize: x₁² + 2x₂² subject to 2x₁ + x₂ = 1
    // Lagrangian: L = x₁² + 2x₂² - λ(2x₁ + x₂ - 1)
    // KKT: 2x₁ - 2λ = 0, 4x₂ - λ = 0, 2x₁ + x₂ = 1
    // Solution: x₁ = λ, x₂ = λ/4, 2λ + λ/4 = 1 → λ = 4/9
    // So x = [4/9, 1/9]

    let objective = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1];

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1]]);

    let equality = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0] + x[1] - 1.0]);

    let equality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[2.0, 1.0])];

    let mut al = AugmentedLagrangian::new(150, 1e-4, 1.0);
    let x0 = Vector::zeros(2);
    let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);

    assert!(result.constraint_violation < 1e-3);
    assert!((result.solution[0] - 4.0 / 9.0).abs() < 1e-2);
    assert!((result.solution[1] - 1.0 / 9.0).abs() < 1e-2);
}

#[test]
fn test_augmented_lagrangian_convergence_tracking() {
    let objective = |x: &Vector<f32>| 0.5 * (x[0] * x[0] + x[1] * x[1]);

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[x[0], x[1]]);

    let equality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] - 1.0]);

    let equality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0, 1.0])];

    let mut al = AugmentedLagrangian::new(100, 1e-4, 1.0);
    let x0 = Vector::zeros(2);
    let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!(result.iterations > 0);
    assert!(result.elapsed_time.as_nanos() > 0);
    assert!(result.constraint_violation < 1e-3);
}

#[test]
fn test_augmented_lagrangian_rho_adaptation() {
    // Test with custom rho increase factor
    let objective = |x: &Vector<f32>| 0.5 * (x[0] * x[0] + x[1] * x[1]);

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[x[0], x[1]]);

    let equality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] - 1.0]);

    let equality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0, 1.0])];

    let mut al = AugmentedLagrangian::new(200, 1e-4, 1.0).with_rho_increase(3.0);
    let x0 = Vector::zeros(2);
    let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);

    assert!(result.constraint_violation < 1e-2); // Relaxed tolerance for high rho_increase
}

#[test]
fn test_augmented_lagrangian_max_iterations() {
    // Use very few iterations to force max iterations status
    let objective = |x: &Vector<f32>| 0.5 * (x[0] * x[0] + x[1] * x[1]);

    let gradient = |x: &Vector<f32>| Vector::from_slice(&[x[0], x[1]]);

    let equality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] - 1.0]);

    let equality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0, 1.0])];

    let mut al = AugmentedLagrangian::new(2, 1e-10, 1.0); // Very few iterations
    let x0 = Vector::zeros(2);
    let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);

    assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    assert_eq!(result.iterations, 2);
}
