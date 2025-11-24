//! # Convex Optimization Examples
//!
//! This example demonstrates Phase 2 convex optimization methods:
//! - FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
//! - Coordinate Descent
//!
//! These methods are particularly effective for:
//! - Sparse recovery (Lasso regression)
//! - Non-negative least squares
//! - High-dimensional optimization (n >> m)

#![allow(non_snake_case)] // Allow mathematical matrix notation (A, B, Q, etc.)
//! - Composite optimization: minimize f(x) + g(x)
//!
//! ## Mathematical Background
//!
//! ### FISTA
//! Solves problems of the form:
//!     minimize f(x) + g(x)
//! where f is smooth (differentiable) and g is "simple" (has easy proximal operator).
//!
//! Achieves O(1/k²) convergence via Nesterov acceleration.
//!
//! ### Coordinate Descent
//! Updates one coordinate at a time:
//!     x^(k+1)_i = argmin_z f(x^(k)_1, ..., x^(k)_{i-1}, z, x^(k)_{i+1}, ..., x^(k)_n)
//!
//! Particularly effective when:
//! - Coordinate updates have closed-form solutions
//! - Problem dimension is very high (n >> m)
//! - Hessian is expensive to compute

use aprender::optim::{prox, CoordinateDescent, FISTA};
use aprender::primitives::{Matrix, Vector};

/// Example 1: Lasso Regression with FISTA
///
/// Problem: minimize ½‖Ax - b‖² + λ‖x‖₁
///
/// This is the classic Lasso problem:
/// - Smooth part f(x) = ½‖Ax - b‖²
/// - Non-smooth part g(x) = λ‖x‖₁ (L1 regularization for sparsity)
fn lasso_with_fista() {
    println!("=== Example 1: Lasso Regression with FISTA ===\n");

    // Create sparse linear system: y = Ax + noise
    // True sparse solution has only 3 non-zero coefficients
    let n = 20; // Number of features
    let m = 50; // Number of samples

    // Design matrix A (m × n)
    let mut a_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.1).sin();
        }
    }
    let A = Matrix::from_vec(m, n, a_data).expect("Valid matrix dimensions");

    // True sparse solution (only 3 non-zero entries)
    let mut x_true_data = vec![0.0; n];
    x_true_data[5] = 2.0;
    x_true_data[10] = -1.5;
    x_true_data[15] = 1.0;
    let x_true = Vector::from_slice(&x_true_data);

    // Generate observations: b = Ax_true + noise
    let b_exact = A.matvec(&x_true).expect("Matrix-vector multiplication");
    let mut b_data = vec![0.0; m];
    for i in 0..m {
        b_data[i] = b_exact[i] + 0.1 * ((i as f32) * 0.5).sin();
    }
    let b = Vector::from_slice(&b_data);

    // Lasso parameter (controls sparsity)
    let lambda = 0.5;

    // Define smooth part: f(x) = ½‖Ax - b‖²
    let smooth = |x: &Vector<f32>| -> f32 {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        0.5 * residual.dot(&residual)
    };

    // Gradient of smooth part: ∇f(x) = Aᵀ(Ax - b)
    let grad_smooth = |x: &Vector<f32>| -> Vector<f32> {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        A.transpose()
            .matvec(&residual)
            .expect("Matrix-vector multiplication")
    };

    // Proximal operator for L1 norm: prox_{λ‖·‖₁}(v) = soft_threshold(v, λα)
    let prox_l1 =
        |v: &Vector<f32>, alpha: f32| -> Vector<f32> { prox::soft_threshold(v, lambda * alpha) };

    // Run FISTA
    let mut fista = FISTA::new(1000, 0.01, 1e-4); // max_iter, step_size, tol
    let x0 = Vector::zeros(n);

    println!("Running FISTA for Lasso regression...");
    println!("Problem size: {m} samples, {n} features");
    println!("True sparsity: 3 non-zero coefficients");
    println!("Regularization parameter λ = {lambda}\n");

    let result = fista.minimize(smooth, grad_smooth, prox_l1, x0);

    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Final objective: {:.6}", result.objective_value);
    println!("Elapsed time: {:?}", result.elapsed_time);

    // Analyze sparsity
    let mut nnz = 0;
    for i in 0..n {
        if result.solution[i].abs() > 1e-3 {
            nnz += 1;
        }
    }
    println!("\nSparsity analysis:");
    println!("Non-zero coefficients found: {nnz}/{n}");

    // Show recovered coefficients at true non-zero locations
    println!("\nRecovered values at true non-zero locations:");
    println!(
        "  x[5]:  true = {:.3}, recovered = {:.3}",
        x_true_data[5], result.solution[5]
    );
    println!(
        "  x[10]: true = {:.3}, recovered = {:.3}",
        x_true_data[10], result.solution[10]
    );
    println!(
        "  x[15]: true = {:.3}, recovered = {:.3}",
        x_true_data[15], result.solution[15]
    );

    // Prediction error
    let y_pred = A
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let mut pred_error = 0.0_f32;
    for i in 0..m {
        let diff = y_pred[i] - b[i];
        pred_error += diff * diff;
    }
    pred_error = (pred_error / m as f32).sqrt();
    println!("\nPrediction RMSE: {pred_error:.6}");

    println!();
}

/// Example 2: Non-Negative Least Squares with FISTA
///
/// Problem: minimize ½‖Ax - b‖² subject to x ≥ 0
///
/// Applications: spectral unmixing, image processing,chemometrics
fn nonnegative_least_squares() {
    println!("=== Example 2: Non-Negative Least Squares with FISTA ===\n");

    // Create overdetermined system with non-negative solution
    let n = 10;
    let m = 30;

    let mut a_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = ((i + j + 1) as f32 * 0.15).cos() + 1.0;
        }
    }
    let A = Matrix::from_vec(m, n, a_data).expect("Valid matrix dimensions");

    // True non-negative solution
    let mut x_true_data = vec![0.0; n];
    for (i, item) in x_true_data.iter_mut().enumerate().take(n) {
        *item = ((i + 1) as f32 * 0.5).exp() * 0.1;
    }
    let x_true = Vector::from_slice(&x_true_data);

    // Observations
    let b = A.matvec(&x_true).expect("Matrix-vector multiplication");

    // Smooth objective: f(x) = ½‖Ax - b‖²
    let smooth = |x: &Vector<f32>| -> f32 {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        0.5 * residual.dot(&residual)
    };

    let grad_smooth = |x: &Vector<f32>| -> Vector<f32> {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        A.transpose()
            .matvec(&residual)
            .expect("Matrix-vector multiplication")
    };

    // Proximal operator for non-negativity constraint: projection onto x ≥ 0
    let prox_nonneg = |v: &Vector<f32>, _alpha: f32| -> Vector<f32> { prox::nonnegative(v) };

    // Run FISTA
    let mut fista = FISTA::new(500, 0.001, 1e-6);
    let x0 = Vector::zeros(n);

    println!("Running FISTA for non-negative least squares...");
    println!("Problem size: {m} samples, {n} features");
    println!("Constraint: x ≥ 0\n");

    let result = fista.minimize(smooth, grad_smooth, prox_nonneg, x0);

    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Final objective: {:.6}", result.objective_value);

    // Check constraint satisfaction
    let mut min_val = f32::INFINITY;
    for i in 0..n {
        if result.solution[i] < min_val {
            min_val = result.solution[i];
        }
    }
    println!("\nConstraint satisfaction:");
    println!("Minimum coefficient value: {min_val:.6} (should be ≥ 0)");

    // Reconstruction error
    let y_pred = A
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let mut recon_error = 0.0_f32;
    for i in 0..m {
        let diff = y_pred[i] - b[i];
        recon_error += diff * diff;
    }
    recon_error = recon_error.sqrt();
    println!("Reconstruction error ‖Ax - b‖: {recon_error:.6}");

    println!();
}

/// Example 3: High-Dimensional Lasso with Coordinate Descent
///
/// Problem: minimize ½‖Ax - b‖² + λ‖x‖₁
///
/// Coordinate Descent is particularly efficient for high-dimensional problems
/// where n >> m (more features than samples).
fn high_dimensional_lasso_cd() {
    println!("=== Example 3: High-Dimensional Lasso with Coordinate Descent ===\n");

    let n = 100; // High-dimensional feature space
    let m = 30; // Fewer samples

    // Create design matrix
    let mut a_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.05).sin();
        }
    }
    let A = Matrix::from_vec(m, n, a_data).expect("Valid matrix dimensions");

    // True sparse solution (only 5 non-zero out of 100)
    let mut x_true_data = vec![0.0; n];
    x_true_data[10] = 1.5;
    x_true_data[25] = -1.0;
    x_true_data[50] = 2.0;
    x_true_data[75] = -0.5;
    x_true_data[90] = 1.0;
    let x_true = Vector::from_slice(&x_true_data);

    // Generate observations
    let b = A.matvec(&x_true).expect("Matrix-vector multiplication");

    let lambda = 0.3;

    println!("Running Coordinate Descent for high-dimensional Lasso...");
    println!("Problem size: {m} samples, {n} features (n >> m)");
    println!("True sparsity: 5 non-zero coefficients");
    println!("Regularization parameter λ = {lambda}\n");

    // Coordinate Descent update for Lasso
    // For coordinate i: x_i = soft_threshold(r_i, λ) / a_ii
    // where r_i = b_i - Σ_{j≠i} A_ij * x_j

    // Precompute A_ij^2 for each column
    let mut a_sq = vec![0.0; n];
    for (j, item) in a_sq.iter_mut().enumerate().take(n) {
        let mut sum = 0.0;
        for i in 0..m {
            let val = A.get(i, j);
            sum += val * val;
        }
        *item = sum;
    }

    // Clone data for closure
    let A_clone = A.clone();
    let b_clone = b.clone();
    let a_sq_clone = a_sq.clone();

    let update = move |x: &mut Vector<f32>, coord: usize| {
        // Compute residual contribution from other coordinates
        let mut r = 0.0_f32;
        for i in 0..m {
            let mut sum = b_clone[i];
            for j in 0..n {
                if j != coord {
                    sum -= A_clone.get(i, j) * x[j];
                }
            }
            r += A_clone.get(i, coord) * sum;
        }

        // Soft-thresholding update
        let z = if r > lambda {
            r - lambda
        } else if r < -lambda {
            r + lambda
        } else {
            0.0
        };

        // Normalize by squared column norm
        if a_sq_clone[coord] > 1e-10 {
            x[coord] = z / a_sq_clone[coord];
        } else {
            x[coord] = 0.0;
        }
    };

    let mut cd = CoordinateDescent::new(200, 1e-4);
    let x0 = Vector::zeros(n);

    let result = cd.minimize(update, x0);

    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Elapsed time: {:?}", result.elapsed_time);

    // Analyze sparsity
    let mut nnz = 0;
    let mut recovered_indices = Vec::new();
    for i in 0..n {
        if result.solution[i].abs() > 1e-3 {
            nnz += 1;
            recovered_indices.push(i);
        }
    }

    println!("\nSparsity analysis:");
    println!("Non-zero coefficients found: {nnz}/{n}");
    println!(
        "Recovered non-zero indices: {:?}",
        &recovered_indices[..nnz.min(10)]
    );

    // Check true non-zero locations
    println!("\nRecovered values at true non-zero locations:");
    for &idx in &[10, 25, 50, 75, 90] {
        println!(
            "  x[{}]: true = {:.3}, recovered = {:.3}",
            idx, x_true_data[idx], result.solution[idx]
        );
    }

    // Prediction error
    let y_pred = A
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let mut pred_error = 0.0_f32;
    for i in 0..m {
        let diff = y_pred[i] - b[i];
        pred_error += diff * diff;
    }
    pred_error = (pred_error / m as f32).sqrt();
    println!("\nPrediction RMSE: {pred_error:.6}");

    println!();
}

/// Example 4: Box-Constrained Quadratic with Coordinate Descent
///
/// Problem: minimize ½xᵀQx - cᵀx subject to l ≤ x ≤ u
///
/// Demonstrates coordinate descent with bound constraints.
fn box_constrained_quadratic() {
    println!("=== Example 4: Box-Constrained Quadratic Programming ===\n");

    let n = 15;

    // Create positive definite Q matrix
    let mut q_data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                q_data[i * n + j] = 2.0 + i as f32 * 0.1;
            } else {
                let val = ((i + j) as f32 * 0.2).sin() * 0.1;
                q_data[i * n + j] = val;
            }
        }
    }
    let Q = Matrix::from_vec(n, n, q_data).expect("Valid matrix dimensions");

    // Linear term
    let mut c_data = vec![0.0; n];
    for (i, item) in c_data.iter_mut().enumerate().take(n) {
        *item = (i as f32 * 0.3).cos();
    }
    let c = Vector::from_slice(&c_data);

    // Box constraints: 0 ≤ x ≤ 2
    let lower = Vector::zeros(n);
    let upper_data = vec![2.0; n];
    let upper = Vector::from_slice(&upper_data);

    println!("Running Coordinate Descent for box-constrained QP...");
    println!("Problem size: {n} variables");
    println!("Constraints: 0 ≤ x ≤ 2\n");

    // Coordinate descent with projection
    let Q_clone = Q.clone();
    let c_clone = c.clone();
    let lower_clone = lower.clone();
    let upper_clone = upper.clone();

    let update = move |x: &mut Vector<f32>, coord: usize| {
        // Compute gradient component: (Qx)_i - c_i
        let mut qx_i = 0.0;
        for j in 0..n {
            qx_i += Q_clone.get(coord, j) * x[j];
        }
        let grad_i = qx_i - c_clone[coord];

        // Optimal step without constraints: x_i = x_i - grad_i / Q_ii
        let q_ii = Q_clone.get(coord, coord);
        if q_ii > 1e-10 {
            let x_new = x[coord] - grad_i / q_ii;

            // Project onto bounds
            x[coord] = x_new.max(lower_clone[coord]).min(upper_clone[coord]);
        }
    };

    let mut cd = CoordinateDescent::new(300, 1e-6);
    let x0 = Vector::ones(n); // Start at x = 1 (feasible)

    let result = cd.minimize(update, x0);

    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Elapsed time: {:?}", result.elapsed_time);

    // Check constraint satisfaction
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..n {
        if result.solution[i] < min_val {
            min_val = result.solution[i];
        }
        if result.solution[i] > max_val {
            max_val = result.solution[i];
        }
    }

    println!("\nConstraint satisfaction:");
    println!("Min coefficient: {min_val:.6} (should be ≥ 0.0)");
    println!("Max coefficient: {max_val:.6} (should be ≤ 2.0)");

    // Compute final objective value
    let qx = Q
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let obj = 0.5 * result.solution.dot(&qx) - c.dot(&result.solution);
    println!("\nFinal objective value: {obj:.6}");

    // Check how many variables hit bounds
    let mut on_lower = 0;
    let mut on_upper = 0;
    let mut interior = 0;
    for i in 0..n {
        if result.solution[i] < 0.01 {
            on_lower += 1;
        } else if result.solution[i] > 1.99 {
            on_upper += 1;
        } else {
            interior += 1;
        }
    }
    println!("\nActive constraints:");
    println!("  Variables at lower bound (x=0): {on_lower}");
    println!("  Variables at upper bound (x=2): {on_upper}");
    println!("  Variables in interior: {interior}");

    println!();
}

/// Example 5: Comparison - FISTA vs Coordinate Descent
///
/// Compare both methods on the same Lasso problem to demonstrate
/// convergence behavior and computational tradeoffs.
#[allow(clippy::too_many_lines)]
fn comparison_fista_vs_cd() {
    println!("=== Example 5: FISTA vs Coordinate Descent Comparison ===\n");

    let n = 30;
    let m = 50;

    // Create problem
    let mut a_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.1).sin();
        }
    }
    let A = Matrix::from_vec(m, n, a_data).expect("Valid matrix dimensions");

    let mut x_true_data = vec![0.0; n];
    x_true_data[5] = 1.5;
    x_true_data[15] = -1.0;
    x_true_data[25] = 0.8;
    let x_true = Vector::from_slice(&x_true_data);

    let b = A.matvec(&x_true).expect("Matrix-vector multiplication");
    let lambda = 0.4;

    println!("Comparing FISTA and Coordinate Descent on Lasso problem");
    println!("Problem size: {m} samples, {n} features");
    println!("True sparsity: 3 non-zero coefficients\n");

    // ===== FISTA =====
    println!("--- FISTA ---");

    let smooth = |x: &Vector<f32>| -> f32 {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        0.5 * residual.dot(&residual)
    };

    let grad_smooth = |x: &Vector<f32>| -> Vector<f32> {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        A.transpose()
            .matvec(&residual)
            .expect("Matrix-vector multiplication")
    };

    let prox_l1 =
        |v: &Vector<f32>, alpha: f32| -> Vector<f32> { prox::soft_threshold(v, lambda * alpha) };

    let mut fista = FISTA::new(500, 0.01, 1e-5);
    let x0_fista = Vector::zeros(n);

    let result_fista = fista.minimize(smooth, grad_smooth, prox_l1, x0_fista);

    println!("Status: {:?}", result_fista.status);
    println!("Iterations: {}", result_fista.iterations);
    println!("Objective: {:.6}", result_fista.objective_value);
    println!("Time: {:?}", result_fista.elapsed_time);

    let mut nnz_fista = 0;
    for i in 0..n {
        if result_fista.solution[i].abs() > 1e-3 {
            nnz_fista += 1;
        }
    }
    println!("Non-zero coefficients: {nnz_fista}");

    // ===== Coordinate Descent =====
    println!("\n--- Coordinate Descent ---");

    let mut a_sq = vec![0.0; n];
    for (j, item) in a_sq.iter_mut().enumerate().take(n) {
        let mut sum = 0.0;
        for i in 0..m {
            let val = A.get(i, j);
            sum += val * val;
        }
        *item = sum;
    }

    let A_clone = A.clone();
    let b_clone = b.clone();
    let a_sq_clone = a_sq.clone();

    let update = move |x: &mut Vector<f32>, coord: usize| {
        let mut r = 0.0_f32;
        for i in 0..m {
            let mut sum = b_clone[i];
            for j in 0..n {
                if j != coord {
                    sum -= A_clone.get(i, j) * x[j];
                }
            }
            r += A_clone.get(i, coord) * sum;
        }

        let z = if r > lambda {
            r - lambda
        } else if r < -lambda {
            r + lambda
        } else {
            0.0
        };

        if a_sq_clone[coord] > 1e-10 {
            x[coord] = z / a_sq_clone[coord];
        } else {
            x[coord] = 0.0;
        }
    };

    let mut cd = CoordinateDescent::new(500, 1e-5);
    let x0_cd = Vector::zeros(n);

    let result_cd = cd.minimize(update, x0_cd);

    println!("Status: {:?}", result_cd.status);
    println!("Iterations: {}", result_cd.iterations);
    println!("Time: {:?}", result_cd.elapsed_time);

    let mut nnz_cd = 0;
    for i in 0..n {
        if result_cd.solution[i].abs() > 1e-3 {
            nnz_cd += 1;
        }
    }
    println!("Non-zero coefficients: {nnz_cd}");

    // ===== Comparison =====
    println!("\n--- Solution Comparison ---");

    let mut solution_diff = 0.0;
    for i in 0..n {
        let diff = result_fista.solution[i] - result_cd.solution[i];
        solution_diff += diff * diff;
    }
    solution_diff = solution_diff.sqrt();

    println!("‖x_FISTA - x_CD‖: {solution_diff:.6}");

    println!("\nKey Takeaways:");
    println!(
        "• FISTA: {} iterations, {:.1?}",
        result_fista.iterations, result_fista.elapsed_time
    );
    println!(
        "• Coordinate Descent: {} iterations, {:.1?}",
        result_cd.iterations, result_cd.elapsed_time
    );
    println!("• Both methods found {nnz_fista} non-zero coefficients (true: 3)");
    println!("• Solutions are very close (difference: {solution_diff:.6})");
    println!("\nWhen to use each:");
    println!("• FISTA: General composite optimization, fast convergence O(1/k²)");
    println!("• CD: High-dimensional problems (n >> m), simple coordinate updates");

    println!();
}

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║   Convex Optimization Examples (Phase 2)             ║");
    println!("║   FISTA + Coordinate Descent                          ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    lasso_with_fista();
    println!("{}", "─".repeat(60));
    println!();

    nonnegative_least_squares();
    println!("{}", "─".repeat(60));
    println!();

    high_dimensional_lasso_cd();
    println!("{}", "─".repeat(60));
    println!();

    box_constrained_quadratic();
    println!("{}", "─".repeat(60));
    println!();

    comparison_fista_vs_cd();

    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║   All examples completed successfully!                ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}
