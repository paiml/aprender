#![allow(clippy::disallowed_methods)]
//! # Constrained Optimization Examples
//!
//! This example demonstrates Phase 3 constrained optimization methods:
//! - Projected Gradient Descent (PGD)
//! - Augmented Lagrangian Method
//! - Interior Point (Barrier) Method
//!
//! These methods handle various constraint types:
//! - **Projection constraints**: x ∈ C (convex sets)
//! - **Equality constraints**: h(x) = 0

#![allow(non_snake_case)] // Allow mathematical matrix notation (A, B, Q, etc.)
//! - **Inequality constraints**: g(x) ≤ 0
//!
//! ## Mathematical Background
//!
//! ### Projected Gradient Descent
//! Solves: minimize f(x) subject to x ∈ C
//!
//! Algorithm: x^{k+1} = P_C(x^k - α∇f(x^k))
//! where P_C is projection onto convex set C.
//!
//! Applications: Portfolio optimization, signal processing, compressed sensing
//!
//! ### Augmented Lagrangian
//! Solves: minimize f(x) subject to h(x) = 0
//!
//! Augmented Lagrangian: L_ρ(x, λ) = f(x) + λᵀh(x) + ½ρ‖h(x)‖²
//! Updates: λ^{k+1} = λ^k + ρh(x^{k+1})
//!
//! Applications: Equality-constrained least squares, manifold optimization, PDEs
//!
//! ### Interior Point Method
//! Solves: minimize f(x) subject to g(x) ≤ 0
//!
//! Log-barrier: B_μ(x) = f(x) - μ Σ log(-g_i(x))
//! As μ → 0, solution approaches constrained optimum.
//!
//! Applications: Linear programming, quadratic programming, convex optimization

use aprender::optim::{AugmentedLagrangian, InteriorPoint, ProjectedGradientDescent};
use aprender::primitives::{Matrix, Vector};

/// Example 1: Non-Negative Quadratic Minimization with Projected GD
///
/// Problem: Minimize ½‖x - target‖² subject to x ≥ 0
///
/// This is a simple but important problem that appears in:
/// - Portfolio optimization (long-only constraints)
/// - Non-negative matrix factorization
/// - Signal processing
fn nonnegative_quadratic_pgd() {
    println!("=== Example 1: Non-Negative Quadratic with Projected GD ===\n");

    let n = 8;

    // Target point (some coordinates negative, will be projected)
    let target = Vector::from_slice(&[1.5, -0.8, 2.3, -1.2, 0.5, 1.8, -0.3, 0.9]);

    println!("Quadratic Minimization Problem:");
    println!("minimize ½‖x - target‖²");
    println!("subject to: x ≥ 0 (non-negativity)");
    println!("Problem size: {n} variables\n");

    // Objective: minimize ½‖x - target‖²
    let objective = |x: &Vector<f32>| -> f32 {
        let diff = x - &target;
        0.5 * diff.dot(&diff)
    };

    // Gradient: ∇f(x) = x - target
    let gradient = |x: &Vector<f32>| -> Vector<f32> { x - &target };

    // Projection onto x ≥ 0
    let project = |x: &Vector<f32>| -> Vector<f32> {
        let mut x_proj = Vector::zeros(n);
        for i in 0..n {
            x_proj[i] = x[i].max(0.0);
        }
        x_proj
    };

    // Initialize: zeros (feasible)
    let x0 = Vector::zeros(n);

    let mut pgd = ProjectedGradientDescent::new(500, 0.1, 1e-6).with_line_search(0.5);

    println!("Running Projected Gradient Descent...\n");

    let result = pgd.minimize(objective, gradient, project, x0);

    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Final objective: {:.6}", result.objective_value);
    println!("Gradient norm: {:.6}", result.gradient_norm);
    println!("Elapsed time: {:?}", result.elapsed_time);

    // Analyze solution
    println!("\nOptimal solution:");
    print!("  x = [");
    for i in 0..n {
        print!("{:.3}", result.solution[i]);
        if i < n - 1 {
            print!(", ");
        }
    }
    println!("]");

    println!("\nExpected (projection of target onto x ≥ 0):");
    print!("  [");
    for i in 0..n {
        print!("{:.3}", target[i].max(0.0));
        if i < n - 1 {
            print!(", ");
        }
    }
    println!("]");

    // Check constraint satisfaction
    let mut min_val = f32::INFINITY;
    for i in 0..n {
        if result.solution[i] < min_val {
            min_val = result.solution[i];
        }
    }

    println!("\nConstraint satisfaction:");
    println!("  Minimum value: {min_val:.6} (should be ≥ 0.0)");

    println!();
}

/// Example 2: Equality-Constrained Least Squares with Augmented Lagrangian
///
/// Problem: minimize ½‖Ax - b‖² subject to Cx = d
///
/// This appears in:
/// - Constrained regression
/// - State estimation with constraints
/// - Optimization on manifolds
fn equality_constrained_least_squares() {
    println!("=== Example 2: Equality-Constrained Least Squares ===\n");

    let n = 10; // Number of variables
    let m = 20; // Number of observations
    let p = 3; // Number of equality constraints

    // Create least squares problem: minimize ½‖Ax - b‖²
    let mut a_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.2).sin();
        }
    }
    let A = Matrix::from_vec(m, n, a_data).expect("Valid matrix");

    // Target vector
    let mut b_data = vec![0.0; m];
    for (i, item) in b_data.iter_mut().enumerate().take(m) {
        *item = (i as f32 * 0.3).cos();
    }
    let b = Vector::from_slice(&b_data);

    // Equality constraints: Cx = d
    // Example: Sum constraints and linear relationships
    let mut c_data = vec![0.0; p * n];
    // Constraint 1: x_0 + x_1 + x_2 = 1.0
    c_data[0] = 1.0;
    c_data[1] = 1.0;
    c_data[2] = 1.0;
    // Constraint 2: x_3 + x_4 = 0.5
    c_data[n + 3] = 1.0;
    c_data[n + 4] = 1.0;
    // Constraint 3: x_5 - x_6 = 0.0
    c_data[2 * n + 5] = 1.0;
    c_data[2 * n + 6] = -1.0;
    let C = Matrix::from_vec(p, n, c_data).expect("Valid constraint matrix");

    let d = Vector::from_slice(&[1.0, 0.5, 0.0]);

    println!("Equality-Constrained Least Squares:");
    println!("Variables: {n}");
    println!("Observations: {m}");
    println!("Equality constraints: {p}");
    println!("Constraints:");
    println!("  x₀ + x₁ + x₂ = 1.0");
    println!("  x₃ + x₄ = 0.5");
    println!("  x₅ - x₆ = 0.0\n");

    // Objective: f(x) = ½‖Ax - b‖²
    let objective = |x: &Vector<f32>| -> f32 {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        0.5 * residual.dot(&residual)
    };

    // Gradient: ∇f(x) = Aᵀ(Ax - b)
    let gradient = |x: &Vector<f32>| -> Vector<f32> {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        A.transpose()
            .matvec(&residual)
            .expect("Matrix-vector multiplication")
    };

    // Equality constraints: h(x) = Cx - d
    let equality = |x: &Vector<f32>| -> Vector<f32> {
        let cx = C.matvec(x).expect("Matrix-vector multiplication");
        &cx - &d
    };

    // Jacobian of equality constraints: J_h(x) = C
    let equality_jac = |_x: &Vector<f32>| -> Vec<Vector<f32>> {
        let mut jac = Vec::with_capacity(p);
        for i in 0..p {
            let mut row = Vector::zeros(n);
            for j in 0..n {
                row[j] = C.get(i, j);
            }
            jac.push(row);
        }
        jac
    };

    let x0 = Vector::from_slice(&[0.3, 0.3, 0.4, 0.2, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0]);

    let mut auglag = AugmentedLagrangian::new(100, 1e-4, 0.1);

    println!("Running Augmented Lagrangian Method...\n");

    let result = auglag.minimize_equality(objective, gradient, equality, equality_jac, x0);

    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Final objective: {:.6}", result.objective_value);
    println!("Elapsed time: {:?}", result.elapsed_time);

    // Check constraint satisfaction
    let h_final = equality(&result.solution);
    println!("\nConstraint satisfaction (h(x) should be ≈ 0):");
    for i in 0..p {
        println!("  h[{}] = {:.6}", i, h_final[i]);
    }
    let constraint_violation = h_final.norm();
    println!("  ‖h(x)‖ = {constraint_violation:.6}");

    // Verify specific constraints
    println!("\nVerifying constraints:");
    let sum_012 = result.solution[0] + result.solution[1] + result.solution[2];
    println!("  x₀ + x₁ + x₂ = {sum_012:.6} (target: 1.0)");
    let sum_34 = result.solution[3] + result.solution[4];
    println!("  x₃ + x₄ = {sum_34:.6} (target: 0.5)");
    let diff_56 = result.solution[5] - result.solution[6];
    println!("  x₅ - x₆ = {diff_56:.6} (target: 0.0)");

    // Compute residual
    let y_pred = A
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let mut residual_norm = 0.0;
    for i in 0..m {
        let diff = y_pred[i] - b[i];
        residual_norm += diff * diff;
    }
    residual_norm = residual_norm.sqrt();
    println!("\nLeast squares residual ‖Ax - b‖: {residual_norm:.6}");

    println!();
}

/// Example 3: Linear Programming with Interior Point Method
///
/// Problem: minimize cᵀx subject to Ax ≤ b, x ≥ 0
///
/// Classic linear programming problem solved via log-barrier method.
fn linear_programming_interior_point() {
    println!("=== Example 3: Linear Programming with Interior Point ===\n");

    // Example LP:
    // minimize -2x₀ - 3x₁ (maximize profit)
    // subject to:
    //   x₀ + 2x₁ ≤ 8  (resource constraint 1)
    //   3x₀ + 2x₁ ≤ 12 (resource constraint 2)
    //   x₀ ≥ 0, x₁ ≥ 0 (non-negativity)

    println!("Linear Programming Problem:");
    println!("minimize -2x₀ - 3x₁");
    println!("subject to:");
    println!("  x₀ + 2x₁ ≤ 8");
    println!("  3x₀ + 2x₁ ≤ 12");
    println!("  x₀ ≥ 0, x₁ ≥ 0\n");

    // Objective: f(x) = -2x₀ - 3x₁
    let c = Vector::from_slice(&[-2.0, -3.0]);

    let objective = |x: &Vector<f32>| -> f32 { c.dot(x) };

    let gradient = |_x: &Vector<f32>| -> Vector<f32> { c.clone() };

    // Inequality constraints: g(x) ≤ 0
    // g₀(x) = x₀ + 2x₁ - 8 ≤ 0
    // g₁(x) = 3x₀ + 2x₁ - 12 ≤ 0
    // g₂(x) = -x₀ ≤ 0
    // g₃(x) = -x₁ ≤ 0
    let inequality = |x: &Vector<f32>| -> Vector<f32> {
        let g0 = x[0] + 2.0 * x[1] - 8.0;
        let g1 = 3.0 * x[0] + 2.0 * x[1] - 12.0;
        let g2 = -x[0];
        let g3 = -x[1];
        Vector::from_slice(&[g0, g1, g2, g3])
    };

    // Jacobian of inequality constraints
    let inequality_jac = |_x: &Vector<f32>| -> Vec<Vector<f32>> {
        vec![
            Vector::from_slice(&[1.0, 2.0]),  // ∇g₀
            Vector::from_slice(&[3.0, 2.0]),  // ∇g₁
            Vector::from_slice(&[-1.0, 0.0]), // ∇g₂
            Vector::from_slice(&[0.0, -1.0]), // ∇g₃
        ]
    };

    // Start from interior feasible point
    let x0 = Vector::from_slice(&[1.0, 2.0]);

    let mut interior_point = InteriorPoint::new(100, 1e-5, 0.1);

    println!("Running Interior Point Method...\n");

    let result = interior_point.minimize(objective, gradient, inequality, inequality_jac, x0);

    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Final objective: {:.6}", result.objective_value);
    println!(
        "Optimal solution: x₀ = {:.6}, x₁ = {:.6}",
        result.solution[0], result.solution[1]
    );
    println!("Elapsed time: {:?}", result.elapsed_time);

    // Check constraint satisfaction
    let g_final = inequality(&result.solution);
    println!("\nConstraint satisfaction (g(x) ≤ 0):");
    println!("  x₀ + 2x₁ - 8 = {:.6} (≤ 0)", g_final[0]);
    println!("  3x₀ + 2x₁ - 12 = {:.6} (≤ 0)", g_final[1]);
    println!("  -x₀ = {:.6} (≤ 0, i.e., x₀ ≥ 0)", g_final[2]);
    println!("  -x₁ = {:.6} (≤ 0, i.e., x₁ ≥ 0)", g_final[3]);

    // Identify active constraints
    println!("\nActive constraints (g(x) ≈ 0):");
    let constraint_names = ["x₀ + 2x₁ ≤ 8", "3x₀ + 2x₁ ≤ 12", "x₀ ≥ 0", "x₁ ≥ 0"];
    for i in 0..4 {
        if g_final[i].abs() < 0.1 {
            println!("  {} (active)", constraint_names[i]);
        }
    }

    println!();
}

/// Example 4: Quadratic Programming with Interior Point Method
///
/// Problem: minimize ½xᵀQx + cᵀx subject to Ax ≤ b
///
/// QP problems appear in:
/// - Model predictive control
/// - Portfolio optimization with risk constraints
/// - Support vector machines

include!("includes/constrained_optimization_include_01.rs");
