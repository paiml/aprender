//! Derivative-free global optimization (metaheuristics).
//!
//! This module provides metaheuristic algorithms for black-box optimization
//! where gradients are unavailable or the landscape is highly multimodal.
//!
//! # Algorithm Categories
//!
//! ## Perturbative Metaheuristics
//! Modify complete solutions through perturbation operators:
//! - [`DifferentialEvolution`] - Population-based, excellent for continuous HPO
//! - [`ParticleSwarm`] - Swarm intelligence with velocity updates
//! - [`SimulatedAnnealing`] - Single-point, Metropolis acceptance
//! - [`GeneticAlgorithm`] - Selection, crossover, mutation
//! - [`HarmonySearch`] - Music-inspired memory-based optimization
//!
//! ## Benchmark Functions
//! Standard test functions for algorithm evaluation:
//! - [`benchmarks`] - CEC 2013 benchmark suite (Sphere, Rosenbrock, Rastrigin, etc.)
//!
//! ## Constructive Metaheuristics (Phase 3)
//! Build solutions incrementally:
//! - `AntColony` - Pheromone-guided construction
//! - `TabuSearch` - Memory-based local search
//!
//! # Search Space Abstraction
//!
//! Unlike gradient-based optimizers that assume continuous spaces,
//! metaheuristics support diverse problem representations:
//!
//! ```
//! use aprender::metaheuristics::SearchSpace;
//!
//! // Continuous optimization (HPO)
//! let hpo_space = SearchSpace::continuous(5, -10.0, 10.0);
//!
//! // Binary feature selection
//! let feature_space = SearchSpace::binary(100);
//!
//! // Permutation (TSP)
//! let tsp_space = SearchSpace::permutation(50);
//! ```
//!
//! # Example: Hyperparameter Optimization
//!
//! ```
//! use aprender::metaheuristics::{DifferentialEvolution, SearchSpace, Budget, PerturbativeMetaheuristic};
//!
//! // Define search space for learning rate and regularization
//! let space = SearchSpace::Continuous {
//!     dim: 2,
//!     lower: vec![1e-5, 1e-6],
//!     upper: vec![1e-1, 1e-2],
//! };
//!
//! // Objective: minimize validation loss (simulated)
//! let objective = |params: &[f64]| {
//!     let lr = params[0];
//!     let reg = params[1];
//!     // Simulated loss landscape
//!     (lr - 0.01).powi(2) + (reg - 0.001).powi(2) + 0.1 * (lr * 100.0).sin()
//! };
//!
//! let mut de = DifferentialEvolution::default();
//! let result = de.optimize(&objective, &space, Budget::Evaluations(5000));
//!
//! assert!(result.objective_value < 0.1);  // Reasonable tolerance for small budget
//! ```
//!
//! # References
//!
//! - Storn & Price (1997): Differential Evolution
//! - Kennedy & Eberhart (1995): Particle Swarm Optimization
//! - Kirkpatrick et al. (1983): Simulated Annealing
//! - Hansen (2016): CMA-ES Tutorial

pub mod benchmarks;
mod binary_ga;
mod budget;
mod cmaes;
mod de;
mod ga;
mod hs;
mod pso;
mod sa;
mod search_space;
mod traits;

pub use binary_ga::BinaryGA;
pub use budget::{Budget, ConvergenceTracker};
pub use cmaes::CmaEs;
pub use de::{AdaptationStrategy, DEStrategy, DifferentialEvolution};
pub use ga::GeneticAlgorithm;
pub use hs::HarmonySearch;
pub use pso::ParticleSwarm;
pub use sa::SimulatedAnnealing;
pub use search_space::SearchSpace;
pub use traits::{OptimizationResult, PerturbativeMetaheuristic, TerminationReason};

#[cfg(test)]
mod tests;
