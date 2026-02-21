//! Differential Evolution (DE) optimizer.
//!
//! A population-based metaheuristic for continuous optimization.
//!
//! # Algorithm
//!
//! DE evolves a population of candidate solutions through mutation based on
//! difference vectors between population members:
//!
//! ```text
//! For each target vector xᵢ:
//!   1. Select 3 distinct random vectors xₐ, xᵦ, xᵧ
//!   2. Mutant: v = xₐ + F·(xᵦ - xᵧ)
//!   3. Crossover: uⱼ = vⱼ if rand() < CR else xᵢⱼ
//!   4. Selection: xᵢ' = u if f(u) < f(xᵢ) else xᵢ
//! ```
//!
//! # References
//!
//! - Storn & Price (1997): "Differential Evolution - A Simple and Efficient
//!   Heuristic for Global Optimization over Continuous Spaces"
//! - Zhang & Sanderson (2009): "JADE: Adaptive Differential Evolution"
//! - Tanabe & Fukunaga (2013): "SHADE: Success-History Based Adaptation"

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    budget::ConvergenceTracker, Budget, OptimizationResult, PerturbativeMetaheuristic, SearchSpace,
};
use crate::metaheuristics::traits::TerminationReason;

/// Differential Evolution optimizer.
///
/// # Example
///
/// ```
/// use aprender::metaheuristics::{DifferentialEvolution, SearchSpace, Budget, PerturbativeMetaheuristic};
///
/// // Sphere function: f(x) = Σxᵢ²
/// let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
///
/// let mut de = DifferentialEvolution::default();
/// let space = SearchSpace::continuous(5, -5.0, 5.0);  // 5D is easier
/// let result = de.optimize(&objective, &space, Budget::Evaluations(10_000));
///
/// assert!(result.objective_value < 1.0);  // Converges to near-zero
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialEvolution {
    /// Population size (default: 10 * dimension)
    pub population_size: usize,

    /// Mutation factor F ∈ [0, 2] (default: 0.8)
    pub mutation_factor: f64,

    /// Crossover rate CR ∈ [0, 1] (default: 0.9)
    pub crossover_rate: f64,

    /// Mutation strategy
    pub strategy: DEStrategy,

    /// Parameter adaptation strategy
    pub adaptation: AdaptationStrategy,

    /// Random seed for reproducibility
    #[serde(default)]
    seed: Option<u64>,

    // Internal state (not serialized)
    #[serde(skip)]
    population: Vec<Vec<f64>>,
    #[serde(skip)]
    fitness: Vec<f64>,
    #[serde(skip)]
    best_idx: usize,
    #[serde(skip)]
    history: Vec<f64>,
}

/// DE mutation strategy.
///
/// Different strategies offer trade-offs between exploration and exploitation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DEStrategy {
    /// DE/rand/1/bin: v = xₐ + F·(xᵦ - xᵧ)
    ///
    /// Good exploration, slower convergence. Default choice.
    #[default]
    Rand1Bin,

    /// DE/best/1/bin: v = `x_best` + F·(xₐ - xᵦ)
    ///
    /// Fast convergence, may premature converge.
    Best1Bin,

    /// DE/rand/2/bin: v = xₐ + F·(xᵦ - xᵧ) + F·(xδ - xε)
    ///
    /// More exploration, uses 5 random vectors.
    Rand2Bin,

    /// DE/current-to-best/1/bin: v = xᵢ + `F·(x_best` - xᵢ) + F·(xₐ - xᵦ)
    ///
    /// Balance between rand and best.
    CurrentToBest1Bin,
}

/// Adaptation strategy for self-adaptive DE variants.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AdaptationStrategy {
    /// Fixed F and CR (original DE)
    #[default]
    None,

    /// JADE: Adaptive F and CR with external archive
    ///
    /// Reference: Zhang & Sanderson (2009)
    JADE {
        /// External archive of inferior solutions
        archive: Vec<Vec<f64>>,
        /// Maximum archive size (typically = `population_size`)
        archive_size: usize,
        /// Location parameter for F (Cauchy distribution)
        mu_f: f64,
        /// Location parameter for CR (Normal distribution)
        mu_cr: f64,
        /// Learning rate for adaptation (typically 0.1)
        c: f64,
    },

    /// SHADE: Success-History Based Adaptation
    ///
    /// Reference: Tanabe & Fukunaga (2013)
    SHADE {
        /// Historical successful F values
        memory_f: Vec<f64>,
        /// Historical successful CR values
        memory_cr: Vec<f64>,
        /// Memory size H (typically 5-10)
        memory_size: usize,
        /// Current position in circular buffer
        memory_index: usize,
    },
}

include!("de_impl.rs");
include!("de_optimize.rs");
