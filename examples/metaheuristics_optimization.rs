//! Metaheuristics Optimization Example
//!
//! Demonstrates derivative-free global optimization using various
//! metaheuristic algorithms from the aprender library.
//!
//! Run with: `cargo run --example metaheuristics_optimization`

use aprender::metaheuristics::{
    benchmarks, BinaryGA, Budget, CmaEs, DifferentialEvolution, GeneticAlgorithm, HarmonySearch,
    ParticleSwarm, PerturbativeMetaheuristic, SearchSpace, SimulatedAnnealing,
};

fn main() {
    println!("=== Metaheuristics Optimization Demo ===\n");

    // Define the Sphere benchmark function
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();

    // Define the Rosenbrock benchmark function
    let rosenbrock = |x: &[f64]| -> f64 {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    };

    let dim = 5;
    let space = SearchSpace::continuous(dim, -5.0, 5.0);
    let budget = Budget::Evaluations(5000);

    // ========== Differential Evolution ==========
    println!("1. Differential Evolution (DE/rand/1/bin)");
    let mut de = DifferentialEvolution::default().with_seed(42);
    let result = de.optimize(&sphere, &space, budget.clone());
    println!("   Sphere f(x*) = {:.6}", result.objective_value);
    println!(
        "   Solution: [{:.4}, {:.4}, ...]",
        result.solution[0], result.solution[1]
    );
    println!("   Evaluations: {}\n", result.evaluations);

    // ========== Particle Swarm Optimization ==========
    println!("2. Particle Swarm Optimization (PSO)");
    let mut pso = ParticleSwarm::default().with_seed(42);
    let result = pso.optimize(&sphere, &space, budget.clone());
    println!("   Sphere f(x*) = {:.6}", result.objective_value);
    println!("   Evaluations: {}\n", result.evaluations);

    // ========== Simulated Annealing ==========
    println!("3. Simulated Annealing (SA)");
    let mut sa = SimulatedAnnealing::default().with_seed(42);
    let result = sa.optimize(&sphere, &space, budget.clone());
    println!("   Sphere f(x*) = {:.6}", result.objective_value);
    println!("   Evaluations: {}\n", result.evaluations);

    // ========== Genetic Algorithm ==========
    println!("4. Genetic Algorithm (SBX + Polynomial Mutation)");
    let mut ga = GeneticAlgorithm::default().with_seed(42);
    let result = ga.optimize(&sphere, &space, budget.clone());
    println!("   Sphere f(x*) = {:.6}", result.objective_value);
    println!("   Evaluations: {}\n", result.evaluations);

    // ========== Harmony Search ==========
    println!("5. Harmony Search (HS)");
    let mut hs = HarmonySearch::default().with_seed(42);
    let result = hs.optimize(&sphere, &space, budget.clone());
    println!("   Sphere f(x*) = {:.6}", result.objective_value);
    println!("   Evaluations: {}\n", result.evaluations);

    // ========== CMA-ES ==========
    println!("6. CMA-ES (Covariance Matrix Adaptation)");
    let mut cma = CmaEs::new(dim).with_seed(42);
    let result = cma.optimize(&sphere, &space, budget.clone());
    println!("   Sphere f(x*) = {:.6}", result.objective_value);
    println!("   Evaluations: {}\n", result.evaluations);

    // ========== Rosenbrock Comparison ==========
    println!("=== Rosenbrock Function Comparison ===\n");
    let rosenbrock_budget = Budget::Evaluations(10000);

    let mut de = DifferentialEvolution::default().with_seed(123);
    let de_result = de.optimize(&rosenbrock, &space, rosenbrock_budget.clone());

    let mut cma = CmaEs::new(dim).with_seed(123);
    let cma_result = cma.optimize(&rosenbrock, &space, rosenbrock_budget.clone());

    println!("   DE:     f(x*) = {:.6}", de_result.objective_value);
    println!("   CMA-ES: f(x*) = {:.6}", cma_result.objective_value);
    println!("   (Global minimum is 0.0 at x = [1, 1, ..., 1])\n");

    // ========== Binary GA for Feature Selection ==========
    println!("=== Feature Selection with Binary GA ===\n");

    // Simulate feature selection: minimize features while maintaining "accuracy"
    let feature_objective = |bits: &[f64]| {
        let selected: usize = bits.iter().filter(|&&b| b > 0.5).count();
        if selected == 0 {
            100.0 // Penalty for no features
        } else {
            // Fewer features = lower cost, but need at least 3
            let accuracy_loss = if selected >= 3 { 0.0 } else { 10.0 };
            selected as f64 * 0.5 + accuracy_loss
        }
    };

    let feature_space = SearchSpace::binary(10); // 10 features
    let mut binary_ga = BinaryGA::default().with_seed(42);
    let result = binary_ga.optimize(
        &feature_objective,
        &feature_space,
        Budget::Evaluations(2000),
    );

    let selected = BinaryGA::selected_features(&result.solution);
    println!("   Selected features: {:?}", selected);
    println!("   Objective: {:.2}\n", result.objective_value);

    // ========== Benchmark Functions ==========
    println!("=== Available Benchmark Functions ===\n");
    for info in benchmarks::all_benchmarks() {
        println!(
            "   f{}: {} ({}, {})",
            info.id,
            info.name,
            if info.multimodal {
                "multimodal"
            } else {
                "unimodal"
            },
            if info.separable {
                "separable"
            } else {
                "non-separable"
            }
        );
    }

    println!("\n=== Demo Complete ===");
}
