//! Optimization Example
//!
//! Demonstrates simular's Bayesian optimization capabilities:
//! - Gaussian Process surrogate modeling
//! - Acquisition functions (EI, UCB, PoI)
//! - Reproducible optimization with deterministic RNG
//!
//! # Running
//! ```bash
//! cargo run --example optimization
//! ```

use simular::domains::optimization::{AcquisitionFunction, BayesianOptimizer, OptimizerConfig};

fn main() {
    println!("=== Simular Bayesian Optimization ===\n");

    // Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
    // Global minimum at (1, 1) with f(1,1) = 0
    let rosenbrock = |x: &[f64]| -> f64 {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };

    // 1. Bayesian Optimization with Expected Improvement
    println!("1. Bayesian Optimization on Rosenbrock Function:");
    println!("   Target minimum: (1.0, 1.0) with f(x) = 0");
    println!("   Using Expected Improvement acquisition\n");

    let config = OptimizerConfig {
        bounds: vec![(-2.0, 2.0), (-2.0, 2.0)], // 2D bounds
        acquisition: AcquisitionFunction::ExpectedImprovement,
        length_scale: 0.5,
        signal_variance: 1.0,
        noise_variance: 1e-6,
        n_candidates: 100,
        seed: 42, // Deterministic!
    };

    let mut optimizer = BayesianOptimizer::new(config);
    let n_iterations = 20;

    println!("   Iteration | Point (x, y)          | f(x, y)");
    println!("   ----------|----------------------|----------");

    for i in 0..n_iterations {
        let x = optimizer.suggest();
        let y = rosenbrock(&x);
        let _ = optimizer.observe(x.clone(), y);

        if i < 5 || i >= n_iterations - 3 {
            println!(
                "   {:>9} | ({:>6.3}, {:>6.3})     | {:.6}",
                i + 1,
                x[0],
                x[1],
                y
            );
        } else if i == 5 {
            println!("        ... | ...                  | ...");
        }
    }

    let (best_x, best_y) = optimizer.best().unwrap_or((&[0.0, 0.0], f64::INFINITY));
    println!(
        "\n   Best found: ({:.4}, {:.4}) with f(x) = {:.6}",
        best_x[0], best_x[1], best_y
    );
    println!("   True optimum: (1.0, 1.0) with f(x) = 0");

    // 2. Compare acquisition functions
    println!("\n2. Acquisition Function Comparison:");

    let acquisition_fns = [
        (
            "Expected Improvement",
            AcquisitionFunction::ExpectedImprovement,
        ),
        ("UCB (kappa=2.0)", AcquisitionFunction::UCB { kappa: 2.0 }),
        (
            "Probability of Improvement",
            AcquisitionFunction::ProbabilityOfImprovement,
        ),
    ];

    for (name, acq) in &acquisition_fns {
        let config = OptimizerConfig {
            bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
            acquisition: acq.clone(),
            seed: 42, // Same seed for fair comparison
            ..Default::default()
        };

        let mut optimizer = BayesianOptimizer::new(config);

        for _ in 0..15 {
            let x = optimizer.suggest();
            let y = rosenbrock(&x);
            let _ = optimizer.observe(x, y);
        }

        let (_, best_y) = optimizer.best().unwrap_or((&[], f64::INFINITY));
        println!("   {}: best f(x) = {:.6}", name, best_y);
    }

    // 3. Reproducibility demonstration
    println!("\n3. Reproducibility Verification:");
    println!("   Same seed produces identical optimization trajectory\n");

    let seed = 12345u64;

    fn optimize_with_seed(seed: u64) -> (f64, f64, f64) {
        let config = OptimizerConfig {
            bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
            acquisition: AcquisitionFunction::ExpectedImprovement,
            seed,
            ..Default::default()
        };

        let mut optimizer = BayesianOptimizer::new(config);
        let rosenbrock =
            |x: &[f64]| -> f64 { (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2) };

        for _ in 0..10 {
            let x = optimizer.suggest();
            let y = rosenbrock(&x);
            let _ = optimizer.observe(x, y);
        }

        let (best_x, best_y) = optimizer.best().unwrap_or((&[0.0, 0.0], f64::INFINITY));
        (best_x[0], best_x[1], best_y)
    }

    let (x1_0, x1_1, y1) = optimize_with_seed(seed);
    let (x2_0, x2_1, y2) = optimize_with_seed(seed);

    println!("   Run 1: ({:.10}, {:.10}), f = {:.10}", x1_0, x1_1, y1);
    println!("   Run 2: ({:.10}, {:.10}), f = {:.10}", x2_0, x2_1, y2);
    println!(
        "   Bitwise identical: {}",
        x1_0 == x2_0 && x1_1 == x2_1 && y1 == y2
    );

    // 4. 1D optimization example
    println!("\n4. 1D Optimization Example:");
    println!("   f(x) = sin(x) + 0.1*x², searching for minimum\n");

    let f = |x: &[f64]| -> f64 { x[0].sin() + 0.1 * x[0] * x[0] };

    let config = OptimizerConfig {
        bounds: vec![(-3.0, 3.0)],
        acquisition: AcquisitionFunction::ExpectedImprovement,
        seed: 42,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(config);

    for _ in 0..10 {
        let x = optimizer.suggest();
        let y = f(&x);
        let _ = optimizer.observe(x, y);
    }

    let (best_x, best_y) = optimizer.best().unwrap_or((&[0.0], f64::INFINITY));
    println!("   Best found: x = {:.4}, f(x) = {:.6}", best_x[0], best_y);

    // Analytical minimum is around x ≈ -0.88
    let analytical_x = -0.88;
    let analytical_y = f(&[analytical_x]);
    println!(
        "   Analytical minimum: x ≈ {:.2}, f(x) ≈ {:.6}",
        analytical_x, analytical_y
    );

    println!("\n=== Optimization Complete ===");
}
