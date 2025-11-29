//! aprender-tsp: Local TSP optimization with personalized .apr models.
//!
//! This crate provides command-line tools for training, optimizing, and deploying
//! Traveling Salesman Problem (TSP) solvers using local `.apr` model files.
//!
//! # Features
//!
//! - **Train personalized TSP models** from your own problem instances
//! - **Optimize routes** using state-of-the-art metaheuristics (ACO, Tabu Search, GA)
//! - **Export solutions** in standard formats (JSON, CSV)
//! - **Deploy offline** with zero cloud dependency
//!
//! # Example
//!
//! ```rust
//! use aprender_tsp::{TspInstance, AcoSolver, TspSolver, Budget};
//!
//! // Create a simple instance
//! let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
//! let instance = TspInstance::from_coords("square", coords).unwrap();
//!
//! // Solve with ACO
//! let mut solver = AcoSolver::new().with_seed(42);
//! let solution = solver.solve(&instance, Budget::Iterations(100)).unwrap();
//!
//! println!("Tour length: {:.2}", solution.length);
//! ```
//!
//! # Toyota Way Principles
//!
//! - **Genchi Genbutsu**: Users understand their logistics problems best
//! - **Kaizen**: Continuous model improvement through incremental updates
//! - **Jidoka**: Build quality in through checksums and validation

pub mod error;
pub mod instance;
pub mod model;
pub mod solver;

// Re-exports for convenience
pub use error::{TspError, TspResult};
pub use instance::TspInstance;
pub use model::TspModel;
pub use solver::{
    AcoSolver, Budget, GaSolver, HybridSolver, SolutionTier, TabuSolver, TspAlgorithm, TspSolution,
    TspSolver,
};
