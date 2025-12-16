//! # ADMM (Alternating Direction Method of Multipliers) Examples
//!
//! Demonstrates the use of ADMM for distributed and constrained optimization:
//! - Distributed Lasso regression
//! - Consensus optimization (federated learning)
//! - Comparison with FISTA
//!
//! ADMM is particularly powerful for:
//! - **Distributed ML**: Split data across workers
//! - **Federated learning**: Train models across devices

#![allow(non_snake_case)] // Allow mathematical matrix notation (A, B, Q, etc.)
//! - **Constrained problems**: Equality constraints via consensus
//!
//! ## Mathematical Background
//!
//! ADMM solves problems of the form:
//! ```text
//! minimize  f(x) + g(z)
//! subject to Ax + Bz = c
//! ```
//!
//! The algorithm alternates between:
//! 1. **x-update**: minimize f(x) + (ρ/2)‖Ax + Bz - c + u‖²
//! 2. **z-update**: minimize g(z) + (ρ/2)‖Ax + Bz - c + u‖²
//! 3. **u-update**: u ← u + (Ax + Bz - c)
//!
//! **Consensus form** (x = z): A = I, B = -I, c = 0
//!
//! ## Reference
//!
//! Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
//! "Distributed Optimization and Statistical Learning via ADMM"
//! Foundations and Trends in Machine Learning, 3(1), 1-122.

use aprender::optim::{prox, OptimizationResult, ADMM};
use aprender::primitives::{Matrix, Vector};

/// Example 1: Distributed Lasso Regression
///
/// Problem: minimize ½‖Dx - b‖² + λ‖x‖₁
///
/// ADMM consensus form: minimize ½‖Dx - b‖² + λ‖z‖₁ subject to x = z
///
/// This separates the smooth (least squares) and non-smooth (L1) parts,
/// allowing each to be solved efficiently.
#[allow(clippy::too_many_lines)]
fn distributed_lasso_admm() {
    println!("=== Example 1: Distributed Lasso with ADMM ===\n");

    let n = 10; // Number of features
    let m = 30; // Number of samples

    // Create sparse linear regression problem
    let mut d_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            d_data[i * n + j] = ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.15).sin();
        }
    }
    let D = Matrix::from_vec(m, n, d_data).expect("Valid matrix");

    // True sparse solution (only 3 non-zero coefficients)
    let mut x_true_data = vec![0.0; n];
    x_true_data[2] = 1.5;
    x_true_data[5] = -1.0;
    x_true_data[8] = 0.8;
    let x_true = Vector::from_slice(&x_true_data);

    // Generate observations: b = Dx_true + noise
    let b_exact = D.matvec(&x_true).expect("Matrix-vector multiplication");
    let mut b_data = vec![0.0; m];
    for i in 0..m {
        b_data[i] = b_exact[i] + 0.05 * ((i as f32) * 0.7).sin();
    }
    let b = Vector::from_slice(&b_data);

    let lambda = 0.3;

    println!("Distributed Lasso Problem:");
    println!("Samples: {m}, Features: {n}");
    println!("True sparsity: 3 non-zero coefficients");
    println!("Regularization: λ = {lambda}\n");

    // Consensus form: x = z (A = I, B = -I, c = 0)
    let A = Matrix::eye(n);
    let mut B = Matrix::from_vec(n, n, vec![0.0; n * n]).expect("Valid matrix");
    for i in 0..n {
        B.set(i, i, -1.0);
    }
    let c = Vector::zeros(n);

    // x-minimizer: Solve (DᵀD + ρI)x = Dᵀb + ρ(z - u)
    // This is the least squares part with consensus penalty
    let dt = D.transpose();
    let dtd = dt.matmul(&D).expect("Matrix multiplication");
    let dtb = dt.matvec(&b).expect("Matrix-vector multiplication");

    let x_minimizer = move |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        // Build system: (DᵀD + ρI)x = Dᵀb + ρ(z - u)
        let mut lhs_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let val = dtd.get(i, j);
                lhs_data[i * n + j] = if i == j { val + rho } else { val };
            }
        }
        let lhs = Matrix::from_vec(n, n, lhs_data).expect("Valid matrix");

        let mut rhs = Vector::zeros(n);
        for i in 0..n {
            rhs[i] = dtb[i] + rho * (z[i] - u[i]);
        }

        // Solve using Cholesky
        if let Ok(x) = lhs.cholesky_solve(&rhs) {
            x
        } else {
            // Fallback: use gradient step if Cholesky fails
            let mut x_new = Vector::zeros(n);
            for i in 0..n {
                x_new[i] = (dtb[i] + rho * (z[i] - u[i])) / (dtd.get(i, i) + rho);
            }
            x_new
        }
    };

    // z-minimizer: Soft-thresholding (proximal operator for L1 norm)
    // z = soft_threshold(-(Ax + u), λ/ρ)
    let z_minimizer = move |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        let threshold = lambda / rho;
        prox::soft_threshold(&(ax + u).mul_scalar(-1.0), threshold)
    };

    // Run ADMM with adaptive rho
    let mut admm = ADMM::new(500, 1.0, 1e-4).with_adaptive_rho(true);
    let x0 = Vector::zeros(n);
    let z0 = Vector::zeros(n);

    println!("Running ADMM with adaptive ρ...\n");

    let result = admm.minimize_consensus(x_minimizer, z_minimizer, &A, &B, &c, x0, z0);

    println!("Convergence: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Constraint violation: {:.6}", result.constraint_violation);
    println!("Elapsed time: {:?}", result.elapsed_time);

    // Analyze sparsity
    let mut nnz = 0;
    let mut recovered_indices = Vec::new();
    for i in 0..n {
        if result.solution[i].abs() > 0.1 {
            nnz += 1;
            recovered_indices.push(i);
        }
    }

    println!("\nSparsity Analysis:");
    println!("Non-zero coefficients found: {nnz}/{n}");
    println!("Recovered indices: {recovered_indices:?}");

    println!("\nRecovered vs True coefficients:");
    println!(
        "  x[2]: true = {:.3}, recovered = {:.3}",
        x_true_data[2], result.solution[2]
    );
    println!(
        "  x[5]: true = {:.3}, recovered = {:.3}",
        x_true_data[5], result.solution[5]
    );
    println!(
        "  x[8]: true = {:.3}, recovered = {:.3}",
        x_true_data[8], result.solution[8]
    );

    // Prediction error
    let y_pred = D
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let mut pred_error = 0.0;
    for i in 0..m {
        let diff = y_pred[i] - b[i];
        pred_error += diff * diff;
    }
    pred_error = (pred_error / (m as f32)).sqrt();
    println!("\nPrediction RMSE: {pred_error:.6}");

    println!();
}

/// Example 2: Consensus Optimization (Federated Learning Simulation)
///
/// Problem: Average solutions from N distributed workers
///
/// Each worker has local data and computes a local solution.
/// ADMM enforces consensus: all workers converge to the same global solution.
fn consensus_optimization_federated() {
    println!("=== Example 2: Consensus Optimization (Federated Learning) ===\n");

    let n = 5; // Dimension of solution
    let num_workers = 3; // Number of distributed workers

    // Each worker has a different objective (simulating different local data)
    let worker_targets = vec![
        Vector::from_slice(&[1.0, 2.0, 0.5, 1.5, 0.8]), // Worker 1
        Vector::from_slice(&[1.2, 1.8, 0.6, 1.3, 0.9]), // Worker 2
        Vector::from_slice(&[0.9, 2.1, 0.4, 1.6, 0.7]), // Worker 3
    ];

    println!("Federated Learning Simulation:");
    println!("Workers: {num_workers}");
    println!("Solution dimension: {n}");
    println!("Goal: Reach consensus across all workers\n");

    // Consensus constraint: all worker solutions must agree
    // We'll use a simplified version: enforce x = z for global consensus
    let A = Matrix::eye(n);
    let mut B = Matrix::from_vec(n, n, vec![0.0; n * n]).expect("Valid matrix");
    for i in 0..n {
        B.set(i, i, -1.0);
    }
    let c = Vector::zeros(n);

    // x-minimizer: Average of worker objectives
    // In real federated learning, each worker would solve locally
    let targets_clone = worker_targets.clone();
    let x_minimizer = move |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        // Each worker minimizes: ½‖x - target_i‖² + (ρ/2)‖x - z + u‖²
        // Combined solution: average of all worker contributions
        let mut x_sum = Vector::zeros(n);
        for target in &targets_clone {
            for i in 0..n {
                let numerator = target[i] + rho * (z[i] - u[i]);
                let denominator = 1.0 + rho;
                x_sum[i] += numerator / denominator;
            }
        }
        // Average across workers
        x_sum.mul_scalar(1.0 / (num_workers as f32))
    };

    // z-minimizer: Consensus variable (just pass through)
    let z_minimizer = |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        // z = -(Ax + u) for consensus
        (ax + u).mul_scalar(-1.0)
    };

    let mut admm = ADMM::new(200, 1.0, 1e-5).with_adaptive_rho(true);
    let x0 = Vector::zeros(n);
    let z0 = Vector::zeros(n);

    println!("Running consensus ADMM...\n");

    let result = admm.minimize_consensus(x_minimizer, z_minimizer, &A, &B, &c, x0, z0);

    println!("Convergence: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Constraint violation: {:.6}", result.constraint_violation);
    println!("Elapsed time: {:?}", result.elapsed_time);

    println!("\nGlobal consensus solution:");
    print!("  x = [");
    for i in 0..n {
        print!("{:.3}", result.solution[i]);
        if i < n - 1 {
            print!(", ");
        }
    }
    println!("]");

    // Compute true average across all worker targets
    let mut true_avg = Vector::zeros(n);
    for target in &worker_targets {
        for i in 0..n {
            true_avg[i] += target[i];
        }
    }
    true_avg = true_avg.mul_scalar(1.0 / (num_workers as f32));

    println!("\nExpected (average of worker targets):");
    print!("  [");
    for i in 0..n {
        print!("{:.3}", true_avg[i]);
        if i < n - 1 {
            print!(", ");
        }
    }
    println!("]");

    // Measure consensus quality
    let mut consensus_error = 0.0;
    for i in 0..n {
        let diff = result.solution[i] - true_avg[i];
        consensus_error += diff * diff;
    }
    consensus_error = consensus_error.sqrt();
    println!("\nConsensus error: {consensus_error:.6}");

    println!();
}

/// Example 3: Quadratic Programming with ADMM
///
/// Problem: minimize ½xᵀQx + cᵀx subject to x ≥ 0
///
/// Using consensus form to separate the quadratic objective from constraints.
fn quadratic_programming_admm() {
    println!("=== Example 3: Quadratic Programming with ADMM ===\n");

    let n = 6;
    let (Q, c_vec) = create_qp_problem(n);

    println!("Quadratic Programming:");
    println!("minimize ½xᵀQx + cᵀx");
    println!("subject to: x ≥ 0");
    println!("Variables: {n}\n");

    let (A, B, c_constraint) = create_consensus_constraints(n);
    let (x_minimizer, z_minimizer) = create_qp_minimizers(n, Q.clone(), c_vec.clone());

    let mut admm = ADMM::new(300, 1.0, 1e-5).with_adaptive_rho(true);
    println!("Running ADMM for QP...\n");

    let result = admm.minimize_consensus(
        x_minimizer,
        z_minimizer,
        &A,
        &B,
        &c_constraint,
        Vector::ones(n),
        Vector::ones(n),
    );

    print_qp_results(&result, &Q, &c_vec, n);
    println!();
}

/// Create the QP problem matrices (Q and c).
fn create_qp_problem(n: usize) -> (Matrix<f32>, Vector<f32>) {
    let q_data: Vec<f32> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                if i == j {
                    2.0 + (i as f32) * 0.3
                } else {
                    ((i + j) as f32 * 0.25).sin() * 0.15
                }
            })
        })
        .collect();
    let Q = Matrix::from_vec(n, n, q_data).expect("Valid Q matrix");
    let c_vec = Vector::from_slice(&[1.0, -0.5, 0.8, -0.3, 0.6, -0.2]);
    (Q, c_vec)
}

/// Create consensus constraints (A, B, c) for x = z form.
fn create_consensus_constraints(n: usize) -> (Matrix<f32>, Matrix<f32>, Vector<f32>) {
    let A = Matrix::eye(n);
    let mut B = Matrix::from_vec(n, n, vec![0.0; n * n]).expect("Valid matrix");
    for i in 0..n {
        B.set(i, i, -1.0);
    }
    (A, B, Vector::zeros(n))
}

/// Create x-minimizer and z-minimizer closures for QP.
fn create_qp_minimizers(
    n: usize,
    Q: Matrix<f32>,
    c_vec: Vector<f32>,
) -> (
    impl Fn(&Vector<f32>, &Vector<f32>, &Vector<f32>, f32) -> Vector<f32>,
    impl Fn(&Vector<f32>, &Vector<f32>, &Vector<f32>, f32) -> Vector<f32>,
) {
    let x_minimizer = move |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        solve_qp_x_subproblem(&Q, &c_vec, z, u, rho, n)
    };

    let z_minimizer = move |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, _rho: f32| {
        project_nonnegative(ax, u, n)
    };

    (x_minimizer, z_minimizer)
}

/// Solve x-subproblem: (Q + ρI)x = -c + ρ(z - u).
fn solve_qp_x_subproblem(
    Q: &Matrix<f32>,
    c: &Vector<f32>,
    z: &Vector<f32>,
    u: &Vector<f32>,
    rho: f32,
    n: usize,
) -> Vector<f32> {
    let lhs_data: Vec<f32> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                let val = Q.get(i, j);
                if i == j {
                    val + rho
                } else {
                    val
                }
            })
        })
        .collect();
    let lhs = Matrix::from_vec(n, n, lhs_data).expect("Valid matrix");

    let rhs_data: Vec<f32> = (0..n).map(|i| -c[i] + rho * (z[i] - u[i])).collect();
    let rhs = Vector::from_slice(&rhs_data);

    lhs.cholesky_solve(&rhs).unwrap_or_else(|_| {
        // Fallback: diagonal approximation
        Vector::from_slice(
            &(0..n)
                .map(|i| (-c[i] + rho * (z[i] - u[i])) / (Q.get(i, i) + rho))
                .collect::<Vec<_>>(),
        )
    })
}

/// Project onto non-negative orthant: z = max(-(Ax + u), 0).
fn project_nonnegative(ax: &Vector<f32>, u: &Vector<f32>, n: usize) -> Vector<f32> {
    Vector::from_slice(
        &(0..n)
            .map(|i| (-(ax[i] + u[i])).max(0.0))
            .collect::<Vec<_>>(),
    )
}

/// Print QP optimization results.
fn print_qp_results(result: &OptimizationResult, Q: &Matrix<f32>, c_vec: &Vector<f32>, n: usize) {
    println!("Convergence: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Constraint violation: {:.6}", result.constraint_violation);
    println!("Elapsed time: {:?}", result.elapsed_time);

    let solution_str: Vec<String> = (0..n)
        .map(|i| format!("{:.3}", result.solution[i]))
        .collect();
    println!("\nOptimal solution:");
    println!("  x = [{}]", solution_str.join(", "));

    let qx = Q
        .matvec(&result.solution)
        .expect("Matrix-vector multiplication");
    let obj = 0.5 * result.solution.dot(&qx) + c_vec.dot(&result.solution);
    println!("\nObjective value: {obj:.6}");

    let min_val = (0..n)
        .map(|i| result.solution[i])
        .fold(f32::INFINITY, f32::min);
    println!("Minimum coefficient: {min_val:.6} (should be ≥ 0)");
}

/// Example 4: ADMM vs FISTA Comparison
///
/// Compare ADMM and FISTA on the same Lasso problem to demonstrate
/// convergence behavior and computational tradeoffs.
#[allow(clippy::too_many_lines)]
fn admm_vs_fista_comparison() {
    println!("=== Example 4: ADMM vs FISTA Comparison ===\n");

    let n = 15;
    let m = 40;

    // Create problem
    let mut a_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.12).sin();
        }
    }
    let A_mat = Matrix::from_vec(m, n, a_data).expect("Valid matrix");

    let mut x_true_data = vec![0.0; n];
    x_true_data[3] = 1.2;
    x_true_data[8] = -0.9;
    x_true_data[12] = 0.7;
    let x_true = Vector::from_slice(&x_true_data);

    let b = A_mat.matvec(&x_true).expect("Matrix-vector multiplication");
    let lambda = 0.4;

    println!("Lasso Comparison: ADMM vs FISTA");
    println!("Samples: {m}, Features: {n}");
    println!("True sparsity: 3 non-zero coefficients");
    println!("Regularization: λ = {lambda}\n");

    // ===== ADMM =====
    println!("--- ADMM ---");

    let A_eye = Matrix::eye(n);
    let mut B_neg = Matrix::from_vec(n, n, vec![0.0; n * n]).expect("Valid matrix");
    for i in 0..n {
        B_neg.set(i, i, -1.0);
    }
    let c_zero = Vector::zeros(n);

    let at = A_mat.transpose();
    let ata = at.matmul(&A_mat).expect("Matrix multiplication");
    let atb = at.matvec(&b).expect("Matrix-vector multiplication");

    let x_min_admm = move |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        let mut lhs_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let val = ata.get(i, j);
                lhs_data[i * n + j] = if i == j { val + rho } else { val };
            }
        }
        let lhs = Matrix::from_vec(n, n, lhs_data).expect("Valid matrix");

        let mut rhs = Vector::zeros(n);
        for i in 0..n {
            rhs[i] = atb[i] + rho * (z[i] - u[i]);
        }

        lhs.cholesky_solve(&rhs)
            .unwrap_or_else(|_| Vector::zeros(n))
    };

    let z_min_admm = move |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
        let threshold = lambda / rho;
        prox::soft_threshold(&(ax + u).mul_scalar(-1.0), threshold)
    };

    let mut admm = ADMM::new(300, 1.0, 1e-5).with_adaptive_rho(true);
    let x0_admm = Vector::zeros(n);
    let z0_admm = Vector::zeros(n);

    let result_admm = admm.minimize_consensus(
        x_min_admm, z_min_admm, &A_eye, &B_neg, &c_zero, x0_admm, z0_admm,
    );

    println!("Status: {:?}", result_admm.status);
    println!("Iterations: {}", result_admm.iterations);
    println!("Time: {:?}", result_admm.elapsed_time);

    let mut nnz_admm = 0;
    for i in 0..n {
        if result_admm.solution[i].abs() > 0.1 {
            nnz_admm += 1;
        }
    }
    println!("Non-zero coefficients: {nnz_admm}");

    // ===== FISTA =====
    println!("\n--- FISTA ---");

    use aprender::optim::FISTA;

    let smooth = |x: &Vector<f32>| -> f32 {
        let ax = A_mat.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        0.5 * residual.dot(&residual)
    };

    let grad_smooth = |x: &Vector<f32>| -> Vector<f32> {
        let ax = A_mat.matvec(x).expect("Matrix-vector multiplication");
        let residual = &ax - &b;
        A_mat
            .transpose()
            .matvec(&residual)
            .expect("Matrix-vector multiplication")
    };

    let prox_l1 =
        |v: &Vector<f32>, alpha: f32| -> Vector<f32> { prox::soft_threshold(v, lambda * alpha) };

    let mut fista = FISTA::new(300, 0.01, 1e-5);
    let x0_fista = Vector::zeros(n);

    let result_fista = fista.minimize(smooth, grad_smooth, prox_l1, x0_fista);

    println!("Status: {:?}", result_fista.status);
    println!("Iterations: {}", result_fista.iterations);
    println!("Time: {:?}", result_fista.elapsed_time);

    let mut nnz_fista = 0;
    for i in 0..n {
        if result_fista.solution[i].abs() > 0.1 {
            nnz_fista += 1;
        }
    }
    println!("Non-zero coefficients: {nnz_fista}");

    // ===== Comparison =====
    println!("\n--- Comparison Summary ---");
    println!(
        "\n{:<15} {:>12} {:>12}",
        "Method", "Iterations", "Time (μs)"
    );
    println!("{}", "─".repeat(42));
    println!(
        "{:<15} {:>12} {:>12.0}",
        "ADMM",
        result_admm.iterations,
        result_admm.elapsed_time.as_micros()
    );
    println!(
        "{:<15} {:>12} {:>12.0}",
        "FISTA",
        result_fista.iterations,
        result_fista.elapsed_time.as_micros()
    );

    println!("\nKey Insights:");
    println!("• ADMM: Consensus form, adaptive ρ, distributable");
    println!("• FISTA: Proximal gradient, O(1/k²) convergence, simpler for small problems");
    println!("\nWhen to use each:");
    println!("• ADMM: Distributed data, federated learning, complex constraints");
    println!("• FISTA: Centralized data, simple composite problems, faster for small n");

    println!();
}

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║   ADMM Optimization Examples                                  ║");
    println!("║   Distributed ML + Federated Learning                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    distributed_lasso_admm();
    println!("{}", "═".repeat(70));
    println!();

    consensus_optimization_federated();
    println!("{}", "═".repeat(70));
    println!();

    quadratic_programming_admm();
    println!("{}", "═".repeat(70));
    println!();

    admm_vs_fista_comparison();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   All ADMM examples completed successfully!                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
}
