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
    println!("  Σx_i = {sum_x:.6} (≤ 5.0)");
    println!("  Budget slack: {:.6}", 5.0 - sum_x);

    let mut min_x = f32::INFINITY;
    for i in 0..n {
        if result.solution[i] < min_x {
            min_x = result.solution[i];
        }
    }
    println!("  min(x_i) = {min_x:.6} (≥ 0)");

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
    println!("  {active_bounds} variables at lower bound (x_i = 0)");

    println!();
}

/// Example 5: Method Comparison - Box-Constrained Quadratic
///
/// Compare all three Phase 3 methods on the same problem:
/// minimize ½‖x - target‖² subject to 0 ≤ x ≤ 1
#[allow(clippy::too_many_lines)]
fn method_comparison() {
    println!("=== Example 5: Method Comparison - Box-Constrained Quadratic ===\n");

    let n = 6;
    let target = Vector::from_slice(&[1.5, -0.5, 0.3, 0.8, 1.2, -0.2]);

    println!("Problem: minimize ½‖x - target‖²");
    println!("subject to: 0 ≤ x ≤ 1 (box constraints)");
    println!("Problem size: {n} variables\n");

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
            x_proj[i] = x[i].clamp(0.0, 1.0);
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
