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
    println!("Problem size: {} variables\n", n);

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
    println!("  Minimum value: {:.6} (should be ≥ 0.0)", min_val);

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
    for i in 0..m {
        b_data[i] = (i as f32 * 0.3).cos();
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
    println!("Variables: {}", n);
    println!("Observations: {}", m);
    println!("Equality constraints: {}", p);
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
    println!("  ‖h(x)‖ = {:.6}", constraint_violation);

    // Verify specific constraints
    println!("\nVerifying constraints:");
    let sum_012 = result.solution[0] + result.solution[1] + result.solution[2];
    println!("  x₀ + x₁ + x₂ = {:.6} (target: 1.0)", sum_012);
    let sum_34 = result.solution[3] + result.solution[4];
    println!("  x₃ + x₄ = {:.6} (target: 0.5)", sum_34);
    let diff_56 = result.solution[5] - result.solution[6];
    println!("  x₅ - x₆ = {:.6} (target: 0.0)", diff_56);

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
    println!("\nLeast squares residual ‖Ax - b‖: {:.6}", residual_norm);

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
fn quadratic_programming_interior_point() {
    println!("=== Example 4: Quadratic Programming with Interior Point ===\n");

    let n = 5;

    // Create positive definite Q matrix
    let mut q_data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                q_data[i * n + j] = 2.0 + (i as f32) * 0.5;
            } else {
                let val = ((i + j) as f32 * 0.3).sin() * 0.2;
                q_data[i * n + j] = val;
            }
        }
    }
    let Q = Matrix::from_vec(n, n, q_data).expect("Valid Q matrix");

    // Linear term
    let c = Vector::from_slice(&[1.0, -0.5, 0.8, -0.3, 0.6]);

    println!("Quadratic Programming Problem:");
    println!("minimize ½xᵀQx + cᵀx");
    println!("subject to:");
    println!("  Σx_i ≤ 5 (budget constraint)");
    println!("  x_i ≥ 0 (non-negativity)\n");

    // Objective: f(x) = ½xᵀQx + cᵀx
    let objective = |x: &Vector<f32>| -> f32 {
        let qx = Q.matvec(x).expect("Matrix-vector multiplication");
        0.5 * x.dot(&qx) + c.dot(x)
    };

    // Gradient: ∇f(x) = Qx + c
    let gradient = |x: &Vector<f32>| -> Vector<f32> {
        let qx = Q.matvec(x).expect("Matrix-vector multiplication");
        &qx + &c
    };

    // Inequality constraints: g(x) ≤ 0
    // g₀(x) = Σx_i - 5 ≤ 0 (budget)
    // g_i(x) = -x_{i-1} ≤ 0, i=1..n (non-negativity)
    let inequality = |x: &Vector<f32>| -> Vector<f32> {
        let mut g = Vector::zeros(n + 1);

        // Budget constraint
        let mut sum = 0.0;
        for i in 0..n {
            sum += x[i];
        }
        g[0] = sum - 5.0;

        // Non-negativity constraints
        for i in 0..n {
            g[i + 1] = -x[i];
        }

        g
    };

    // Jacobian
    let inequality_jac = |_x: &Vector<f32>| -> Vec<Vector<f32>> {
        let mut jac = Vec::with_capacity(n + 1);

        // ∇g₀ = [1, 1, ..., 1]
        jac.push(Vector::ones(n));

        // ∇g_i = -e_i (standard basis vectors)
        for i in 0..n {
            let mut row = Vector::zeros(n);
            row[i] = -1.0;
            jac.push(row);
        }

        jac
    };

    // Start from interior feasible point (strictly feasible: sum < 5, all > 0)
    let x0 = Vector::from_slice(&[0.5, 0.6, 0.7, 0.8, 0.9]); // sum = 3.5 < 5

    let mut interior_point = InteriorPoint::new(100, 1e-5, 1.0);

    println!("Running Interior Point Method for QP...\n");

    let result = interior_point.minimize(objective, gradient, inequality, inequality_jac, x0);

    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Final objective: {:.6}", result.objective_value);
    println!("Elapsed time: {:?}", result.elapsed_time);

    println!("\nOptimal solution:");
    for i in 0..n {
        println!("  x[{}] = {:.6}", i, result.solution[i]);
    }

    // Check constraints
    let g_final = inequality(&result.solution);
    let mut sum_x = 0.0;
    for i in 0..n {
        sum_x += result.solution[i];
    }

    println!("\nConstraint satisfaction:");
    println!("  Σx_i = {:.6} (≤ 5.0)", sum_x);
    println!("  Budget slack: {:.6}", 5.0 - sum_x);

    let mut min_x = f32::INFINITY;
    for i in 0..n {
        if result.solution[i] < min_x {
            min_x = result.solution[i];
        }
    }
    println!("  min(x_i) = {:.6} (≥ 0)", min_x);

    // Count active constraints
    let active_budget = g_final[0].abs() < 0.1;
    let mut active_bounds = 0;
    for i in 1..=n {
        if g_final[i].abs() < 0.1 {
            active_bounds += 1;
        }
    }

    println!("\nActive constraints:");
    if active_budget {
        println!("  Budget constraint is active (solution on boundary)");
    } else {
        println!("  Budget constraint is inactive (interior solution)");
    }
    println!("  {} variables at lower bound (x_i = 0)", active_bounds);

    println!();
}

/// Example 5: Method Comparison - Box-Constrained Quadratic
///
/// Compare all three Phase 3 methods on the same problem:
/// minimize ½‖x - target‖² subject to 0 ≤ x ≤ 1
fn method_comparison() {
    println!("=== Example 5: Method Comparison - Box-Constrained Quadratic ===\n");

    let n = 6;
    let target = Vector::from_slice(&[1.5, -0.5, 0.3, 0.8, 1.2, -0.2]);

    println!("Problem: minimize ½‖x - target‖²");
    println!("subject to: 0 ≤ x ≤ 1 (box constraints)");
    println!("Problem size: {} variables\n", n);

    println!(
        "Target vector: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
        target[0], target[1], target[2], target[3], target[4], target[5]
    );
    println!("(Note: Some target values outside [0,1] box)\n");

    // Objective: f(x) = ½‖x - target‖²
    let objective = |x: &Vector<f32>| -> f32 {
        let diff = x - &target;
        0.5 * diff.dot(&diff)
    };

    // Gradient: ∇f(x) = x - target
    let gradient = |x: &Vector<f32>| -> Vector<f32> { x - &target };

    // ===== Method 1: Projected Gradient Descent =====
    println!("--- Method 1: Projected Gradient Descent ---");

    let project_box = |x: &Vector<f32>| -> Vector<f32> {
        let mut x_proj = Vector::zeros(n);
        for i in 0..n {
            x_proj[i] = x[i].max(0.0).min(1.0);
        }
        x_proj
    };

    let x0_pgd = Vector::from_slice(&[0.5; 6]);
    let mut pgd = ProjectedGradientDescent::new(200, 0.1, 1e-6);

    let result_pgd = pgd.minimize(|x| objective(x), |x| gradient(x), project_box, x0_pgd);

    println!("Status: {:?}", result_pgd.status);
    println!("Iterations: {}", result_pgd.iterations);
    println!("Objective: {:.6}", result_pgd.objective_value);
    println!("Time: {:?}", result_pgd.elapsed_time);
    println!(
        "Solution: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        result_pgd.solution[0],
        result_pgd.solution[1],
        result_pgd.solution[2],
        result_pgd.solution[3],
        result_pgd.solution[4],
        result_pgd.solution[5]
    );

    // ===== Method 2: Augmented Lagrangian (via equality reformulation) =====
    println!("\n--- Method 2: Augmented Lagrangian ---");
    println!("(Equality reformulation: minimize f(x) s.t. x² - x + s = 0, 0 ≤ s ≤ 0.25)");

    // For box constraints via AL, we use slack variables: 0 ≤ x ≤ 1
    // Reformulate as: x_i(1-x_i) ≥ 0 for all i
    // Equality: h_i(x) = x_i(1-x_i) - s_i = 0 with s_i ≥ 0

    let equality_box = |x: &Vector<f32>| -> Vector<f32> {
        // Barrier-like reformulation: push solution away from boundaries
        // h_i(x) = min(x_i, 0) + max(x_i - 1, 0)
        let mut h = Vector::zeros(n);
        for i in 0..n {
            if x[i] < 0.0 {
                h[i] = x[i]; // Violation below 0
            } else if x[i] > 1.0 {
                h[i] = x[i] - 1.0; // Violation above 1
            }
            // else h[i] = 0 (feasible)
        }
        h
    };

    let equality_jac_box = |_x: &Vector<f32>| -> Vec<Vector<f32>> {
        let mut jac = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vector::zeros(n);
            row[i] = 1.0;
            jac.push(row);
        }
        jac
    };

    let x0_al = Vector::from_slice(&[0.5; 6]);
    let mut auglag = AugmentedLagrangian::new(50, 1e-5, 10.0);

    let result_al = auglag.minimize_equality(
        |x| objective(x),
        |x| gradient(x),
        equality_box,
        equality_jac_box,
        x0_al,
    );

    println!("Status: {:?}", result_al.status);
    println!("Iterations: {}", result_al.iterations);
    println!("Objective: {:.6}", result_al.objective_value);
    println!("Time: {:?}", result_al.elapsed_time);
    println!(
        "Solution: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        result_al.solution[0],
        result_al.solution[1],
        result_al.solution[2],
        result_al.solution[3],
        result_al.solution[4],
        result_al.solution[5]
    );

    // ===== Method 3: Interior Point =====
    println!("\n--- Method 3: Interior Point Method ---");

    // Inequality constraints: g(x) ≤ 0
    // -x_i ≤ 0 (x_i ≥ 0) and x_i - 1 ≤ 0 (x_i ≤ 1)
    let inequality_box = |x: &Vector<f32>| -> Vector<f32> {
        let mut g = Vector::zeros(2 * n);
        for i in 0..n {
            g[i] = -x[i]; // -x_i ≤ 0
            g[n + i] = x[i] - 1.0; // x_i - 1 ≤ 0
        }
        g
    };

    let inequality_jac_box = |_x: &Vector<f32>| -> Vec<Vector<f32>> {
        let mut jac = Vec::with_capacity(2 * n);
        for i in 0..n {
            let mut row_lower = Vector::zeros(n);
            row_lower[i] = -1.0;
            jac.push(row_lower);
        }
        for i in 0..n {
            let mut row_upper = Vector::zeros(n);
            row_upper[i] = 1.0;
            jac.push(row_upper);
        }
        jac
    };

    let x0_ip = Vector::from_slice(&[0.5; 6]);
    let mut interior_point = InteriorPoint::new(100, 1e-5, 0.1);

    let result_ip = interior_point.minimize(
        |x| objective(x),
        |x| gradient(x),
        inequality_box,
        inequality_jac_box,
        x0_ip,
    );

    println!("Status: {:?}", result_ip.status);
    println!("Iterations: {}", result_ip.iterations);
    println!("Objective: {:.6}", result_ip.objective_value);
    println!("Time: {:?}", result_ip.elapsed_time);
    println!(
        "Solution: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        result_ip.solution[0],
        result_ip.solution[1],
        result_ip.solution[2],
        result_ip.solution[3],
        result_ip.solution[4],
        result_ip.solution[5]
    );

    // ===== Comparison =====
    println!("\n--- Comparison Summary ---");
    println!(
        "\n{:<25} {:>12} {:>12} {:>12}",
        "Method", "Iterations", "Objective", "Time (μs)"
    );
    println!("{}", "─".repeat(65));
    println!(
        "{:<25} {:>12} {:>12.6} {:>12.0}",
        "Projected GD",
        result_pgd.iterations,
        result_pgd.objective_value,
        result_pgd.elapsed_time.as_micros()
    );
    println!(
        "{:<25} {:>12} {:>12.6} {:>12.0}",
        "Augmented Lagrangian",
        result_al.iterations,
        result_al.objective_value,
        result_al.elapsed_time.as_micros()
    );
    println!(
        "{:<25} {:>12} {:>12.6} {:>12.0}",
        "Interior Point",
        result_ip.iterations,
        result_ip.objective_value,
        result_ip.elapsed_time.as_micros()
    );

    println!("\nKey Insights:");
    println!("• Projected GD: Simple projection, fast iterations, O(1/k) convergence");
    println!(
        "• Augmented Lagrangian: Handles general equality constraints, penalty parameter tuning"
    );
    println!("• Interior Point: Natural for inequalities, log-barrier guarantees feasibility");

    println!("\nWhen to use each method:");
    println!("• Projected GD: Simple convex constraints (box, simplex, ball), fast projection");
    println!("• Augmented Lagrangian: Equality constraints, nonlinear constraints");
    println!("• Interior Point: Inequality constraints, LP/QP, strict interior feasibility");

    println!();
}

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Constrained Optimization Examples (Phase 3)                ║");
    println!("║   Projected GD + Augmented Lagrangian + Interior Point       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    nonnegative_quadratic_pgd();
    println!("{}", "═".repeat(70));
    println!();

    equality_constrained_least_squares();
    println!("{}", "═".repeat(70));
    println!();

    linear_programming_interior_point();
    println!("{}", "═".repeat(70));
    println!();

    quadratic_programming_interior_point();
    println!("{}", "═".repeat(70));
    println!();

    method_comparison();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   All constrained optimization examples completed!           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
}
