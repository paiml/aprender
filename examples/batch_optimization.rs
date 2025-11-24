//! Batch Optimization Examples
//!
//! Demonstrates the use of batch optimization algorithms (L-BFGS, Conjugate Gradient,
//! and Damped Newton) on various test functions.
//!
//! Run with: cargo run --example batch_optimization

use aprender::optim::{CGBetaFormula, ConjugateGradient, DampedNewton, Optimizer, LBFGS};
use aprender::primitives::Vector;

/// Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
///
/// Classic non-convex optimization test function with global minimum at (1, 1).
/// Known for having a narrow, curved valley making it challenging for optimizers.
fn rosenbrock(x: &Vector<f32>) -> f32 {
    let a = x[0];
    let b = x[1];
    (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
}

fn rosenbrock_grad(x: &Vector<f32>) -> Vector<f32> {
    let a = x[0];
    let b = x[1];
    let dx = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
    let dy = 200.0 * (b - a * a);
    Vector::from_slice(&[dx, dy])
}

/// Sphere function: f(x) = sum(x_i^2)
///
/// Convex quadratic with global minimum at origin.
/// Easy test function, all optimizers should converge quickly.
fn sphere(x: &Vector<f32>) -> f32 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i] * x[i];
    }
    sum
}

fn sphere_grad(x: &Vector<f32>) -> Vector<f32> {
    let mut g = Vector::zeros(x.len());
    for i in 0..x.len() {
        g[i] = 2.0 * x[i];
    }
    g
}

/// Booth function: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
///
/// Global minimum at (1, 3) with f(1, 3) = 0.
fn booth(x: &Vector<f32>) -> f32 {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    a * a + b * b
}

fn booth_grad(x: &Vector<f32>) -> Vector<f32> {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    let dx = 2.0 * a + 4.0 * b;
    let dy = 4.0 * a + 2.0 * b;
    Vector::from_slice(&[dx, dy])
}

fn print_result(name: &str, result: &aprender::optim::OptimizationResult) {
    println!("\n{name} Results:");
    println!("  Status: {:?}", result.status);
    println!("  Iterations: {}", result.iterations);
    println!(
        "  Solution: [{:.6}, {:.6}]",
        result.solution[0], result.solution[1]
    );
    println!("  Objective value: {:.6}", result.objective_value);
    println!("  Gradient norm: {:.6}", result.gradient_norm);
    println!("  Time: {:?}", result.elapsed_time);
}

#[allow(clippy::too_many_lines)]
fn main() {
    println!("=== Batch Optimization Examples ===\n");

    // Example 1: Rosenbrock function with different optimizers
    println!("--- Example 1: Rosenbrock Function ---");
    println!("Minimizing f(x,y) = (1-x)^2 + 100(y-x^2)^2");
    println!("Global minimum at (1, 1)\n");

    let x0 = Vector::from_slice(&[0.0, 0.0]);

    // L-BFGS
    let mut lbfgs = LBFGS::new(1000, 1e-6, 10);
    let result_lbfgs = lbfgs.minimize(rosenbrock, rosenbrock_grad, x0.clone());
    print_result("L-BFGS", &result_lbfgs);

    // Conjugate Gradient (Polak-Ribière)
    let mut cg_pr = ConjugateGradient::new(1000, 1e-6, CGBetaFormula::PolakRibiere);
    let result_cg_pr = cg_pr.minimize(rosenbrock, rosenbrock_grad, x0.clone());
    print_result("CG (Polak-Ribière)", &result_cg_pr);

    // Conjugate Gradient (Fletcher-Reeves)
    let mut cg_fr = ConjugateGradient::new(1000, 1e-6, CGBetaFormula::FletcherReeves);
    let result_cg_fr = cg_fr.minimize(rosenbrock, rosenbrock_grad, x0.clone());
    print_result("CG (Fletcher-Reeves)", &result_cg_fr);

    // Damped Newton
    let mut dn = DampedNewton::new(1000, 1e-6);
    let result_dn = dn.minimize(rosenbrock, rosenbrock_grad, x0);
    print_result("Damped Newton", &result_dn);

    // Example 2: Sphere function (convex quadratic)
    println!("\n\n--- Example 2: Sphere Function (5D) ---");
    println!("Minimizing f(x) = sum(x_i^2)");
    println!("Global minimum at origin\n");

    let x0_sphere = Vector::from_slice(&[5.0, -3.0, 2.0, -4.0, 1.0]);

    let mut lbfgs = LBFGS::new(100, 1e-8, 10);
    let result = lbfgs.minimize(sphere, sphere_grad, x0_sphere.clone());
    println!(
        "L-BFGS: {} iterations, gradient norm: {:.2e}",
        result.iterations, result.gradient_norm
    );

    let mut cg = ConjugateGradient::new(100, 1e-8, CGBetaFormula::PolakRibiere);
    let result = cg.minimize(sphere, sphere_grad, x0_sphere.clone());
    println!(
        "CG: {} iterations, gradient norm: {:.2e}",
        result.iterations, result.gradient_norm
    );

    let mut dn = DampedNewton::new(100, 1e-8);
    let result = dn.minimize(sphere, sphere_grad, x0_sphere);
    println!(
        "Damped Newton: {} iterations, gradient norm: {:.2e}",
        result.iterations, result.gradient_norm
    );

    // Example 3: Booth function
    println!("\n\n--- Example 3: Booth Function ---");
    println!("Minimizing f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2");
    println!("Global minimum at (1, 3)\n");

    let x0_booth = Vector::from_slice(&[0.0, 0.0]);

    let mut lbfgs = LBFGS::new(100, 1e-6, 5);
    let result = lbfgs.minimize(booth, booth_grad, x0_booth.clone());
    print_result("L-BFGS", &result);

    let mut cg = ConjugateGradient::new(100, 1e-6, CGBetaFormula::HestenesStiefel);
    let result = cg.minimize(booth, booth_grad, x0_booth.clone());
    print_result("CG (Hestenes-Stiefel)", &result);

    let mut dn = DampedNewton::new(100, 1e-6);
    let result = dn.minimize(booth, booth_grad, x0_booth);
    print_result("Damped Newton", &result);

    // Example 4: Comparing convergence behavior
    println!("\n\n--- Example 4: Convergence Comparison ---");
    println!("Running optimizers with different initial points\n");

    let initial_points = [
        Vector::from_slice(&[-2.0, 2.0]),
        Vector::from_slice(&[5.0, -5.0]),
        Vector::from_slice(&[0.5, 0.5]),
    ];

    for (i, x0) in initial_points.iter().enumerate() {
        println!("Initial point {}: [{:.1}, {:.1}]", i + 1, x0[0], x0[1]);

        let mut lbfgs = LBFGS::new(500, 1e-5, 10);
        let result = lbfgs.minimize(rosenbrock, rosenbrock_grad, x0.clone());
        println!(
            "  L-BFGS: {} iters, final objective: {:.6}",
            result.iterations, result.objective_value
        );

        let mut cg = ConjugateGradient::new(500, 1e-5, CGBetaFormula::PolakRibiere);
        let result = cg.minimize(rosenbrock, rosenbrock_grad, x0.clone());
        println!(
            "  CG-PR:  {} iters, final objective: {:.6}",
            result.iterations, result.objective_value
        );

        let mut dn = DampedNewton::new(500, 1e-5);
        let result = dn.minimize(rosenbrock, rosenbrock_grad, x0.clone());
        println!(
            "  DampedN: {} iters, final objective: {:.6}",
            result.iterations, result.objective_value
        );
        println!();
    }

    // Example 5: Optimizer configuration
    println!("\n--- Example 5: Optimizer Configuration ---\n");

    // L-BFGS with different history sizes
    println!("L-BFGS with different history sizes (m):");
    for m in [3, 10, 20] {
        let mut opt = LBFGS::new(100, 1e-6, m);
        let result = opt.minimize(rosenbrock, rosenbrock_grad, Vector::from_slice(&[0.0, 0.0]));
        println!("  m={}: {} iterations", m, result.iterations);
    }

    // CG with periodic restart
    println!("\nConjugate Gradient with restart:");
    let mut cg_no_restart = ConjugateGradient::new(100, 1e-6, CGBetaFormula::PolakRibiere);
    let result =
        cg_no_restart.minimize(rosenbrock, rosenbrock_grad, Vector::from_slice(&[0.0, 0.0]));
    println!("  No restart: {} iterations", result.iterations);

    let mut cg_restart =
        ConjugateGradient::new(100, 1e-6, CGBetaFormula::PolakRibiere).with_restart_interval(10);
    let result = cg_restart.minimize(rosenbrock, rosenbrock_grad, Vector::from_slice(&[0.0, 0.0]));
    println!("  Restart every 10: {} iterations", result.iterations);

    // Damped Newton with different epsilon
    println!("\nDamped Newton with different finite difference epsilon:");
    for &eps in &[1e-4, 1e-5, 1e-6] {
        let mut opt = DampedNewton::new(100, 1e-6).with_epsilon(eps);
        let result = opt.minimize(booth, booth_grad, Vector::from_slice(&[0.0, 0.0]));
        println!("  epsilon={:.0e}: {} iterations", eps, result.iterations);
    }

    println!("\n=== Examples Complete ===");
}
