//! Monte Carlo Pi Estimation Example
//!
//! Demonstrates the core Monte Carlo capabilities of simular:
//! - Deterministic random number generation
//! - Variance reduction techniques
//! - Reproducible results across runs
//!
//! # Running
//! ```bash
//! cargo run --example monte_carlo
//! ```

use simular::domains::monte_carlo::{MonteCarloEngine, VarianceReduction};
use simular::engine::rng::SimRng;

fn main() {
    println!("=== Simular Monte Carlo Pi Estimation ===\n");

    // Seed for reproducibility - same seed = same results every time
    let seed = 42u64;
    println!("Master seed: {seed} (deterministic results guaranteed)\n");

    // Number of samples
    let n_samples = 1_000_000;

    // Standard Monte Carlo estimation
    println!("1. Standard Monte Carlo (no variance reduction):");
    let mut rng = SimRng::new(seed);
    let engine = MonteCarloEngine::with_samples(n_samples);

    // Use run_nd for 2D sampling
    let result = engine.run_nd(
        2,
        |x| {
            let (u, v) = (x[0], x[1]);
            if u * u + v * v <= 1.0 {
                1.0
            } else {
                0.0
            }
        },
        &mut rng,
    );

    let pi_estimate = result.estimate * 4.0;
    let std_error = result.std_error * 4.0;
    println!("   Pi estimate: {pi_estimate:.6}");
    println!("   Std error:   {std_error:.6}");
    println!("   True pi:     {:.6}", std::f64::consts::PI);
    println!(
        "   Error:       {:.6}\n",
        (pi_estimate - std::f64::consts::PI).abs()
    );

    // Antithetic variates for variance reduction
    println!("2. Antithetic Variates (variance reduction):");
    let mut rng = SimRng::new(seed);
    let engine = MonteCarloEngine::new(n_samples, VarianceReduction::Antithetic);

    // 1D antithetic with transformed variable
    let result = engine.run(
        |u| {
            // Transform [0,1] to circle check via clever mapping
            let x = 2.0 * u - 1.0;
            let y_max = (1.0 - x * x).sqrt();
            2.0 * y_max // Area slice
        },
        &mut rng,
    );

    let pi_estimate = result.estimate * 2.0;
    let std_error = result.std_error * 2.0;
    println!("   Pi estimate: {pi_estimate:.6}");
    println!("   Std error:   {std_error:.6}");
    println!("   True pi:     {:.6}", std::f64::consts::PI);
    println!(
        "   Error:       {:.6}\n",
        (pi_estimate - std::f64::consts::PI).abs()
    );

    // Demonstrate reproducibility
    println!("3. Reproducibility Verification:");
    let engine = MonteCarloEngine::with_samples(10_000);

    let mut rng1 = SimRng::new(seed);
    let mut rng2 = SimRng::new(seed);

    let result1 = engine.run_nd(
        2,
        |x| {
            if x[0] * x[0] + x[1] * x[1] <= 1.0 {
                1.0
            } else {
                0.0
            }
        },
        &mut rng1,
    );
    let result2 = engine.run_nd(
        2,
        |x| {
            if x[0] * x[0] + x[1] * x[1] <= 1.0 {
                1.0
            } else {
                0.0
            }
        },
        &mut rng2,
    );

    println!("   Run 1 mean: {:.10}", result1.estimate);
    println!("   Run 2 mean: {:.10}", result2.estimate);
    println!(
        "   Bitwise identical: {}\n",
        result1.estimate == result2.estimate
    );

    // Convergence demonstration
    println!("4. Convergence Rate (error ~ 1/sqrt(n)):");
    let sample_sizes = [100, 1_000, 10_000, 100_000, 1_000_000];
    for &n in &sample_sizes {
        let mut rng = SimRng::new(seed);
        let engine = MonteCarloEngine::with_samples(n);
        let result = engine.run_nd(
            2,
            |x| {
                if x[0] * x[0] + x[1] * x[1] <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            },
            &mut rng,
        );
        let error = (result.estimate * 4.0 - std::f64::consts::PI).abs();
        let theoretical_error = 1.0 / (n as f64).sqrt();
        println!(
            "   n={:>7}: error={:.6}, theoretical~{:.6}",
            n, error, theoretical_error
        );
    }

    // Integral estimation example
    println!("\n5. Integral Estimation: ∫x² dx from 0 to 1:");
    let mut rng = SimRng::new(seed);
    let engine = MonteCarloEngine::with_samples(100_000);
    let result = engine.run(|x| x * x, &mut rng);
    println!("   Monte Carlo estimate: {:.6}", result.estimate);
    println!("   True value (1/3):     {:.6}", 1.0 / 3.0);
    println!(
        "   Error:                {:.6}",
        (result.estimate - 1.0 / 3.0).abs()
    );

    println!("\n=== Simulation Complete ===");
}
