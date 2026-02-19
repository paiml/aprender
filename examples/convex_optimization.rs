#![allow(clippy::disallowed_methods)]
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

    let n = 20;
    let m = 50;
    let lambda = 0.5;

    let (A, x_true_data, b) = create_lasso_problem(n, m);
    let result = run_lasso_fista(&A, &b, n, m, lambda);
    print_lasso_results(&A, &b, &result, &x_true_data, n, m);
    println!();
}

fn create_lasso_problem(n: usize, m: usize) -> (Matrix<f32>, Vec<f32>, Vector<f32>) {
    let mut a_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.1).sin();
        }
    }
    let A = Matrix::from_vec(m, n, a_data).expect("Valid matrix dimensions");

    let mut x_true_data = vec![0.0; n];
    x_true_data[5] = 2.0;
    x_true_data[10] = -1.5;
    x_true_data[15] = 1.0;
    let x_true = Vector::from_slice(&x_true_data);

    let b_exact = A.matvec(&x_true).expect("Matrix-vector multiplication");
    let mut b_data = vec![0.0; m];
    for i in 0..m {
        b_data[i] = b_exact[i] + 0.1 * ((i as f32) * 0.5).sin();
    }
    let b = Vector::from_slice(&b_data);

    (A, x_true_data, b)
}

fn run_lasso_fista(
    A: &Matrix<f32>,
    b: &Vector<f32>,
    n: usize,
    m: usize,
    lambda: f32,
) -> aprender::optim::OptimizationResult {
    let smooth = |x: &Vector<f32>| -> f32 {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - b;
        0.5 * residual.dot(&residual)
    };

    let grad_smooth = |x: &Vector<f32>| -> Vector<f32> {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - b;
        A.transpose()
            .matvec(&residual)
            .expect("Matrix-vector multiplication")
    };

    let prox_l1 =
        |v: &Vector<f32>, alpha: f32| -> Vector<f32> { prox::soft_threshold(v, lambda * alpha) };

    println!("Running FISTA for Lasso regression...");
    println!("Problem size: {m} samples, {n} features");
    println!("True sparsity: 3 non-zero coefficients");
    println!("Regularization parameter λ = {lambda}\n");

    let mut fista = FISTA::new(1000, 0.01, 1e-4);
    fista.minimize(smooth, grad_smooth, prox_l1, Vector::zeros(n))
}

fn print_lasso_results(
    A: &Matrix<f32>,
    b: &Vector<f32>,
    result: &aprender::optim::OptimizationResult,
    x_true_data: &[f32],
    n: usize,
    m: usize,
) {
    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Final objective: {:.6}", result.objective_value);
    println!("Elapsed time: {:?}", result.elapsed_time);

    let nnz = (0..n).filter(|&i| result.solution[i].abs() > 1e-3).count();
    println!("\nSparsity analysis:");
    println!("Non-zero coefficients found: {nnz}/{n}");

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

    let y_pred = A
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let pred_error: f32 = (0..m).map(|i| (y_pred[i] - b[i]).powi(2)).sum::<f32>() / m as f32;
    println!("\nPrediction RMSE: {:.6}", pred_error.sqrt());
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

    let n = 100;
    let m = 30;
    let lambda = 0.3;

    let (A, x_true_data, b) = create_high_dim_problem(n, m);
    let result = run_cd_lasso(&A, &b, n, m, lambda);
    print_cd_results(&A, &b, &result, &x_true_data, n, m);
    println!();
}

fn create_high_dim_problem(n: usize, m: usize) -> (Matrix<f32>, Vec<f32>, Vector<f32>) {
    let mut a_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.05).sin();
        }
    }
    let A = Matrix::from_vec(m, n, a_data).expect("Valid matrix dimensions");

    let mut x_true_data = vec![0.0; n];
    x_true_data[10] = 1.5;
    x_true_data[25] = -1.0;
    x_true_data[50] = 2.0;
    x_true_data[75] = -0.5;
    x_true_data[90] = 1.0;
    let x_true = Vector::from_slice(&x_true_data);

    let b = A.matvec(&x_true).expect("Matrix-vector multiplication");
    (A, x_true_data, b)
}

fn compute_column_norms(A: &Matrix<f32>, n: usize, m: usize) -> Vec<f32> {
    let mut a_sq = vec![0.0; n];
    for (j, item) in a_sq.iter_mut().enumerate().take(n) {
        let mut sum = 0.0;
        for i in 0..m {
            let val = A.get(i, j);
            sum += val * val;
        }
        *item = sum;
    }
    a_sq
}

fn run_cd_lasso(
    A: &Matrix<f32>,
    b: &Vector<f32>,
    n: usize,
    m: usize,
    lambda: f32,
) -> aprender::optim::OptimizationResult {
    println!("Running Coordinate Descent for high-dimensional Lasso...");
    println!("Problem size: {m} samples, {n} features (n >> m)");
    println!("True sparsity: 5 non-zero coefficients");
    println!("Regularization parameter λ = {lambda}\n");

    let a_sq = compute_column_norms(A, n, m);
    let A_clone = A.clone();
    let b_clone = b.clone();

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

        let z = soft_threshold_scalar(r, lambda);
        x[coord] = if a_sq[coord] > 1e-10 {
            z / a_sq[coord]
        } else {
            0.0
        };
    };

    let mut cd = CoordinateDescent::new(200, 1e-4);
    cd.minimize(update, Vector::zeros(n))
}

fn soft_threshold_scalar(r: f32, lambda: f32) -> f32 {
    if r > lambda {
        r - lambda
    } else if r < -lambda {
        r + lambda
    } else {
        0.0
    }
}

fn print_cd_results(
    A: &Matrix<f32>,
    b: &Vector<f32>,
    result: &aprender::optim::OptimizationResult,
    x_true_data: &[f32],
    n: usize,
    m: usize,
) {
    println!("Convergence status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Elapsed time: {:?}", result.elapsed_time);

    let recovered_indices: Vec<usize> = (0..n)
        .filter(|&i| result.solution[i].abs() > 1e-3)
        .collect();

    println!("\nSparsity analysis:");
    println!(
        "Non-zero coefficients found: {}/{n}",
        recovered_indices.len()
    );
    println!(
        "Recovered non-zero indices: {:?}",
        &recovered_indices[..recovered_indices.len().min(10)]
    );

    println!("\nRecovered values at true non-zero locations:");
    for &idx in &[10, 25, 50, 75, 90] {
        println!(
            "  x[{}]: true = {:.3}, recovered = {:.3}",
            idx, x_true_data[idx], result.solution[idx]
        );
    }

    let y_pred = A
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let pred_error: f32 = (0..m).map(|i| (y_pred[i] - b[i]).powi(2)).sum::<f32>() / m as f32;
    println!("\nPrediction RMSE: {:.6}", pred_error.sqrt());
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
fn comparison_fista_vs_cd() {
    println!("=== Example 5: FISTA vs Coordinate Descent Comparison ===\n");

    let n = 30;
    let m = 50;
    let lambda = 0.4;

    let (A, b) = create_comparison_problem(n, m);

    println!("Comparing FISTA and Coordinate Descent on Lasso problem");
    println!("Problem size: {m} samples, {n} features");
    println!("True sparsity: 3 non-zero coefficients\n");

    let result_fista = run_comparison_fista(&A, &b, n, lambda);
    let result_cd = run_comparison_cd(&A, &b, n, m, lambda);
    print_comparison_results(&result_fista, &result_cd, n);

    println!();
}

fn create_comparison_problem(n: usize, m: usize) -> (Matrix<f32>, Vector<f32>) {
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
    (A, b)
}

fn run_comparison_fista(
    A: &Matrix<f32>,
    b: &Vector<f32>,
    n: usize,
    lambda: f32,
) -> aprender::optim::OptimizationResult {
    println!("--- FISTA ---");

    let smooth = |x: &Vector<f32>| -> f32 {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - b;
        0.5 * residual.dot(&residual)
    };

    let grad_smooth = |x: &Vector<f32>| -> Vector<f32> {
        let ax = A.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - b;
        A.transpose()
            .matvec(&residual)
            .expect("Matrix-vector multiplication")
    };

    let prox_l1 =
        |v: &Vector<f32>, alpha: f32| -> Vector<f32> { prox::soft_threshold(v, lambda * alpha) };

    let mut fista = FISTA::new(500, 0.01, 1e-5);
    let result = fista.minimize(smooth, grad_smooth, prox_l1, Vector::zeros(n));

    let nnz = (0..n).filter(|&i| result.solution[i].abs() > 1e-3).count();
    println!("Status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Objective: {:.6}", result.objective_value);
    println!("Time: {:?}", result.elapsed_time);
    println!("Non-zero coefficients: {nnz}");

    result
}

fn run_comparison_cd(
    A: &Matrix<f32>,
    b: &Vector<f32>,
    n: usize,
    m: usize,
    lambda: f32,
) -> aprender::optim::OptimizationResult {
    println!("\n--- Coordinate Descent ---");

    let a_sq = compute_column_norms(A, n, m);
    let A_clone = A.clone();
    let b_clone = b.clone();

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

        let z = soft_threshold_scalar(r, lambda);
        x[coord] = if a_sq[coord] > 1e-10 {
            z / a_sq[coord]
        } else {
            0.0
        };
    };

    let mut cd = CoordinateDescent::new(500, 1e-5);
    let result = cd.minimize(update, Vector::zeros(n));

    let nnz = (0..n).filter(|&i| result.solution[i].abs() > 1e-3).count();
    println!("Status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Time: {:?}", result.elapsed_time);
    println!("Non-zero coefficients: {nnz}");

    result
}

fn print_comparison_results(
    result_fista: &aprender::optim::OptimizationResult,
    result_cd: &aprender::optim::OptimizationResult,
    n: usize,
) {
    println!("\n--- Solution Comparison ---");

    let solution_diff: f32 = (0..n)
        .map(|i| (result_fista.solution[i] - result_cd.solution[i]).powi(2))
        .sum::<f32>()
        .sqrt();
    let nnz = (0..n)
        .filter(|&i| result_fista.solution[i].abs() > 1e-3)
        .count();

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
    println!("• Both methods found {nnz} non-zero coefficients (true: 3)");
    println!("• Solutions are very close (difference: {solution_diff:.6})");
    println!("\nWhen to use each:");
    println!("• FISTA: General composite optimization, fast convergence O(1/k²)");
    println!("• CD: High-dimensional problems (n >> m), simple coordinate updates");
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
