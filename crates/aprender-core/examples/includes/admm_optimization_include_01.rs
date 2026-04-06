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
