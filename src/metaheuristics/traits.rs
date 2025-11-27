//! Core traits for metaheuristic optimization.
//!
//! Defines the interface contract for perturbative and constructive metaheuristics.

use serde::{Deserialize, Serialize};

use super::{Budget, SearchSpace};

/// Result of a metaheuristic optimization run.
///
/// Contains the best solution found, its objective value, and diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult<S> {
    /// Best solution found
    pub solution: S,

    /// Objective value of the best solution
    pub objective_value: f64,

    /// Total number of objective function evaluations
    pub evaluations: usize,

    /// Number of iterations/generations completed
    pub iterations: usize,

    /// Convergence history (best value at each iteration)
    pub history: Vec<f64>,

    /// Termination reason
    pub termination: TerminationReason,
}

impl<S> OptimizationResult<S> {
    /// Create a new optimization result.
    #[must_use]
    pub fn new(
        solution: S,
        objective_value: f64,
        evaluations: usize,
        iterations: usize,
        history: Vec<f64>,
        termination: TerminationReason,
    ) -> Self {
        Self {
            solution,
            objective_value,
            evaluations,
            iterations,
            history,
            termination,
        }
    }

    /// Check if the optimization converged (vs budget exhausted).
    #[must_use]
    pub const fn converged(&self) -> bool {
        matches!(self.termination, TerminationReason::Converged)
    }
}

/// Reason for optimization termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationReason {
    /// Convergence criteria met (no improvement for patience iterations)
    Converged,

    /// Maximum evaluations reached
    BudgetExhausted,

    /// Maximum iterations reached
    MaxIterations,

    /// User-requested termination (via callback)
    UserTerminated,

    /// Numerical error (NaN, Inf detected)
    NumericalError,
}

/// Trait for perturbative metaheuristics.
///
/// Perturbative metaheuristics modify complete solutions through perturbation operators.
/// They are suitable for continuous, discrete, and mixed-variable optimization.
///
/// # Flow
/// ```text
/// Initialize population → Evaluate → Perturb → Select → Repeat
/// ```
///
/// # Implementors
/// - [`DifferentialEvolution`] - Population-based, continuous
/// - [`ParticleSwarm`] - Swarm intelligence
/// - [`SimulatedAnnealing`] - Single-point trajectory
/// - [`GeneticAlgorithm`] - Selection/crossover/mutation
///
/// # Example
/// ```ignore
/// use aprender::metaheuristics::{PerturbativeMetaheuristic, SearchSpace, Budget};
///
/// let mut optimizer = DifferentialEvolution::default();
/// let space = SearchSpace::continuous(10, -5.0, 5.0);
/// let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
///
/// let result = optimizer.optimize(&objective, &space, Budget::Evaluations(10_000));
/// println!("Best: {:?}, Value: {}", result.solution, result.objective_value);
/// ```
pub trait PerturbativeMetaheuristic {
    /// Solution type (typically `Vec<f64>` for continuous, `BitVec` for binary)
    type Solution: Clone;

    /// Run optimization.
    ///
    /// # Arguments
    /// * `objective` - Function to minimize
    /// * `space` - Search space definition
    /// * `budget` - Termination criteria
    ///
    /// # Returns
    /// Optimization result with best solution and diagnostics
    fn optimize<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        budget: Budget,
    ) -> OptimizationResult<Self::Solution>
    where
        F: Fn(&[f64]) -> f64;

    /// Get the current best solution (if available).
    fn best(&self) -> Option<&Self::Solution>;

    /// Get the convergence history.
    fn history(&self) -> &[f64];

    /// Reset the optimizer state for a new run.
    fn reset(&mut self);
}

/// Trait for constructive metaheuristics.
///
/// Constructive metaheuristics build solutions incrementally rather than
/// modifying complete solutions. They are primarily used for combinatorial
/// optimization on graphs.
///
/// # Flow
/// ```text
/// Start empty → Add component → Update state → Repeat until complete
/// ```
///
/// # Implementors (Phase 3)
/// - `AntColony` - Pheromone-guided construction
/// - `TabuSearch` - Memory-based local search (hybrid constructive/perturbative)
///
/// # Key Difference from Perturbative
/// - Solutions are built step-by-step, not modified in-place
/// - Requires problem-specific construction rules
/// - Internal state (pheromones, memory) updated during construction
///
/// # Note
/// This trait is defined for Phase 3 implementation. Currently unused.
#[allow(dead_code, unreachable_pub)]
pub trait ConstructiveMetaheuristic {
    /// Solution type (e.g., `Vec<usize>` for permutations)
    type Solution: Clone;

    /// Component type (building block, e.g., edge for TSP)
    type Component;

    /// Construct a single solution.
    ///
    /// # Arguments
    /// * `space` - Search space (must be Graph or Permutation)
    /// * `rng` - Random number generator
    fn construct_solution(&self, space: &SearchSpace, rng: &mut impl rand::Rng) -> Self::Solution;

    /// Update internal state based on solutions found.
    ///
    /// For ACO: update pheromone matrix
    /// For Tabu: update tabu list and memory structures
    fn update_state(&mut self, solutions: &[(Self::Solution, f64)]);

    /// Run optimization.
    fn optimize<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        budget: Budget,
    ) -> OptimizationResult<Self::Solution>
    where
        F: Fn(&Self::Solution) -> f64;

    /// Reset the optimizer state.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_result_converged() {
        let result: OptimizationResult<Vec<f64>> = OptimizationResult::new(
            vec![1.0, 2.0],
            0.5,
            1000,
            100,
            vec![10.0, 5.0, 1.0, 0.5],
            TerminationReason::Converged,
        );

        assert!(result.converged());
        assert_eq!(result.evaluations, 1000);
        assert_eq!(result.iterations, 100);
    }

    #[test]
    fn test_optimization_result_budget_exhausted() {
        let result: OptimizationResult<Vec<f64>> = OptimizationResult::new(
            vec![1.0, 2.0],
            0.5,
            10000,
            100,
            vec![10.0, 5.0, 1.0, 0.5],
            TerminationReason::BudgetExhausted,
        );

        assert!(!result.converged());
    }

    #[test]
    fn test_termination_reason_equality() {
        assert_eq!(TerminationReason::Converged, TerminationReason::Converged);
        assert_ne!(
            TerminationReason::Converged,
            TerminationReason::BudgetExhausted
        );
    }
}
