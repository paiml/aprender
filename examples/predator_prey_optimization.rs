//! Predator-Prey Ecosystem Parameter Optimization
//!
//! This example uses metaheuristics to optimize parameters of a
//! Lotka-Volterra predator-prey model to match observed population data.
//!
//! The Lotka-Volterra equations model predator-prey dynamics:
//!   dx/dt = αx - βxy    (prey growth minus predation)
//!   dy/dt = δxy - γy    (predator growth from predation minus death)
//!
//! Run with: `cargo run --example predator_prey_optimization`

use aprender::metaheuristics::{
    Budget, DifferentialEvolution, PerturbativeMetaheuristic, SearchSpace,
};

/// Lotka-Volterra model parameters
#[derive(Debug, Clone, Copy)]
struct LotkaVolterraParams {
    alpha: f64, // Prey birth rate
    beta: f64,  // Predation rate
    delta: f64, // Predator reproduction rate
    gamma: f64, // Predator death rate
}

/// Simulate Lotka-Volterra dynamics using Euler method
fn simulate_lotka_volterra(
    params: &LotkaVolterraParams,
    x0: f64,      // Initial prey population
    y0: f64,      // Initial predator population
    dt: f64,      // Time step
    steps: usize, // Number of steps
) -> Vec<(f64, f64)> {
    let mut trajectory = Vec::with_capacity(steps);
    let mut x = x0;
    let mut y = y0;

    for _ in 0..steps {
        trajectory.push((x, y));

        // Lotka-Volterra equations
        let dx = params.alpha * x - params.beta * x * y;
        let dy = params.delta * x * y - params.gamma * y;

        // Euler step
        x += dx * dt;
        y += dy * dt;

        // Prevent negative populations
        x = x.max(0.0);
        y = y.max(0.0);
    }

    trajectory
}

/// Generate synthetic "observed" data with known parameters
fn generate_observed_data() -> (Vec<(f64, f64)>, LotkaVolterraParams) {
    let true_params = LotkaVolterraParams {
        alpha: 1.1, // Prey birth rate
        beta: 0.4,  // Predation rate
        delta: 0.1, // Predator growth from predation
        gamma: 0.4, // Predator death rate
    };

    let data = simulate_lotka_volterra(&true_params, 10.0, 5.0, 0.1, 100);
    (data, true_params)
}

fn main() {
    println!("=== Predator-Prey Ecosystem Parameter Optimization ===\n");

    // Generate synthetic observed data
    let (observed, true_params) = generate_observed_data();

    println!("True parameters (to be recovered):");
    println!("  α (prey birth rate):     {:.3}", true_params.alpha);
    println!("  β (predation rate):      {:.3}", true_params.beta);
    println!("  δ (predator growth):     {:.3}", true_params.delta);
    println!("  γ (predator death rate): {:.3}", true_params.gamma);
    println!();

    // Search space: [alpha, beta, delta, gamma]
    let space = SearchSpace::Continuous {
        dim: 4,
        lower: vec![0.1, 0.01, 0.01, 0.1],
        upper: vec![2.0, 1.0, 0.5, 1.0],
    };

    // Objective: Mean squared error between simulated and observed data
    let objective = |params_vec: &[f64]| -> f64 {
        let params = LotkaVolterraParams {
            alpha: params_vec[0],
            beta: params_vec[1],
            delta: params_vec[2],
            gamma: params_vec[3],
        };

        let simulated = simulate_lotka_volterra(&params, 10.0, 5.0, 0.1, 100);

        // Calculate MSE
        let mse: f64 = observed
            .iter()
            .zip(simulated.iter())
            .map(|((ox, oy), (sx, sy))| (ox - sx).powi(2) + (oy - sy).powi(2))
            .sum::<f64>()
            / (observed.len() as f64);

        mse
    };

    println!("Optimization objective: Minimize MSE between model and observed data");
    println!("Search space: 4D continuous (α, β, δ, γ)\n");

    // ==========================================
    // Method 1: Differential Evolution
    // ==========================================
    println!("=== Method 1: Differential Evolution ===");
    let mut de = DifferentialEvolution::default().with_seed(42);
    let de_result = de.optimize(&objective, &space, Budget::Evaluations(5000));

    println!("DE Result:");
    println!(
        "  α = {:.4} (true: {:.4})",
        de_result.solution[0], true_params.alpha
    );
    println!(
        "  β = {:.4} (true: {:.4})",
        de_result.solution[1], true_params.beta
    );
    println!(
        "  δ = {:.4} (true: {:.4})",
        de_result.solution[2], true_params.delta
    );
    println!(
        "  γ = {:.4} (true: {:.4})",
        de_result.solution[3], true_params.gamma
    );
    println!("  MSE: {:.6}", de_result.objective_value);
    println!();

    // ==========================================
    // Method 2: Compare with Discretized ACO
    // ==========================================
    println!("=== Method 2: Discretized Parameter Search (ACO-style) ===");

    // Discretize parameter space for combinatorial approach
    // This demonstrates using ACO for continuous optimization via discretization
    let alpha_values = [0.5, 0.8, 1.0, 1.1, 1.2, 1.5];
    let beta_values = [0.2, 0.3, 0.4, 0.5, 0.6];
    let delta_values = [0.05, 0.1, 0.15, 0.2];
    let gamma_values = [0.2, 0.3, 0.4, 0.5, 0.6];

    // Grid search for comparison
    let mut best_mse = f64::INFINITY;
    let mut best_discrete = (0.0, 0.0, 0.0, 0.0);

    for &a in &alpha_values {
        for &b in &beta_values {
            for &d in &delta_values {
                for &g in &gamma_values {
                    let mse = objective(&[a, b, d, g]);
                    if mse < best_mse {
                        best_mse = mse;
                        best_discrete = (a, b, d, g);
                    }
                }
            }
        }
    }

    println!("Grid Search Result (discretized):");
    println!(
        "  α = {:.4} (true: {:.4})",
        best_discrete.0, true_params.alpha
    );
    println!(
        "  β = {:.4} (true: {:.4})",
        best_discrete.1, true_params.beta
    );
    println!(
        "  δ = {:.4} (true: {:.4})",
        best_discrete.2, true_params.delta
    );
    println!(
        "  γ = {:.4} (true: {:.4})",
        best_discrete.3, true_params.gamma
    );
    println!("  MSE: {:.6}", best_mse);
    println!();

    // ==========================================
    // Results Summary
    // ==========================================
    println!("=== Summary ===\n");
    println!(
        "DE achieved {:.2}x better fit than grid search",
        best_mse / de_result.objective_value.max(1e-10)
    );

    // Calculate parameter recovery accuracy
    let de_error = ((de_result.solution[0] - true_params.alpha).powi(2)
        + (de_result.solution[1] - true_params.beta).powi(2)
        + (de_result.solution[2] - true_params.delta).powi(2)
        + (de_result.solution[3] - true_params.gamma).powi(2))
    .sqrt();

    let grid_error = ((best_discrete.0 - true_params.alpha).powi(2)
        + (best_discrete.1 - true_params.beta).powi(2)
        + (best_discrete.2 - true_params.delta).powi(2)
        + (best_discrete.3 - true_params.gamma).powi(2))
    .sqrt();

    println!("\nParameter Recovery Error (Euclidean distance from true):");
    println!("  DE:   {:.4}", de_error);
    println!("  Grid: {:.4}", grid_error);

    // Show population dynamics with recovered parameters
    println!("\n=== Population Dynamics with Recovered Parameters ===");
    let recovered_params = LotkaVolterraParams {
        alpha: de_result.solution[0],
        beta: de_result.solution[1],
        delta: de_result.solution[2],
        gamma: de_result.solution[3],
    };

    let recovered_sim = simulate_lotka_volterra(&recovered_params, 10.0, 5.0, 0.1, 100);

    println!("\nTime  Prey(Obs) Prey(Sim)  Pred(Obs) Pred(Sim)");
    println!("----  --------- ---------  --------- ---------");
    for i in (0..100).step_by(10) {
        println!(
            "{:4}   {:7.2}   {:7.2}    {:7.2}   {:7.2}",
            i, observed[i].0, recovered_sim[i].0, observed[i].1, recovered_sim[i].1
        );
    }
}
