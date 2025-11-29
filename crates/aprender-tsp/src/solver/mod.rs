//! TSP solver implementations.
//!
//! Provides multiple metaheuristic backends:
//! - ACO (Ant Colony Optimization)
//! - Tabu Search
//! - Genetic Algorithm
//! - Hybrid

mod aco;
mod ga;
mod hybrid;
mod tabu;

pub use aco::AcoSolver;
pub use ga::GaSolver;
pub use hybrid::HybridSolver;
pub use tabu::TabuSolver;

use crate::error::TspResult;
use crate::instance::TspInstance;

/// Budget for optimization
#[derive(Debug, Clone, Copy)]
pub enum Budget {
    /// Maximum number of iterations
    Iterations(usize),
    /// Maximum number of tour evaluations
    Evaluations(usize),
}

impl Budget {
    /// Get the limit value
    pub fn limit(&self) -> usize {
        match self {
            Self::Iterations(n) | Self::Evaluations(n) => *n,
        }
    }
}

/// TSP solution
#[derive(Debug, Clone)]
pub struct TspSolution {
    /// Tour as city indices
    pub tour: Vec<usize>,
    /// Total tour length
    pub length: f64,
    /// Number of evaluations used
    pub evaluations: usize,
    /// Convergence history (best length per iteration)
    pub history: Vec<f64>,
}

impl TspSolution {
    /// Create a new solution
    pub fn new(tour: Vec<usize>, length: f64) -> Self {
        Self {
            tour,
            length,
            evaluations: 0,
            history: vec![length],
        }
    }

    /// Check if this solution is better than another
    pub fn is_better_than(&self, other: &Self) -> bool {
        self.length < other.length
    }
}

/// Trait for TSP solvers
pub trait TspSolver: Send + Sync {
    /// Solve a TSP instance
    fn solve(&mut self, instance: &TspInstance, budget: Budget) -> TspResult<TspSolution>;

    /// Get algorithm name
    fn name(&self) -> &'static str;

    /// Reset solver state
    fn reset(&mut self);
}

/// Algorithm type for model persistence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TspAlgorithm {
    /// Ant Colony Optimization
    Aco,
    /// Tabu Search
    Tabu,
    /// Genetic Algorithm
    Ga,
    /// Hybrid (GA + Tabu + ACO)
    Hybrid,
}

impl TspAlgorithm {
    /// Get string name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Aco => "aco",
            Self::Tabu => "tabu",
            Self::Ga => "ga",
            Self::Hybrid => "hybrid",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "aco" | "ant" | "antcolony" => Some(Self::Aco),
            "tabu" | "tabusearch" => Some(Self::Tabu),
            "ga" | "genetic" => Some(Self::Ga),
            "hybrid" | "auto" => Some(Self::Hybrid),
            _ => None,
        }
    }
}

/// Solution quality tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionTier {
    /// Within 0.1% of optimal
    Optimal,
    /// Within 1% of optimal
    Excellent,
    /// Within 2% of optimal
    Good,
    /// Within 5% of optimal
    Acceptable,
    /// More than 5% from optimal
    Poor,
}

impl SolutionTier {
    /// Classify solution quality based on gap percentage
    pub fn from_gap(gap_percent: f64) -> Self {
        if gap_percent < 0.1 {
            Self::Optimal
        } else if gap_percent < 1.0 {
            Self::Excellent
        } else if gap_percent < 2.0 {
            Self::Good
        } else if gap_percent < 5.0 {
            Self::Acceptable
        } else {
            Self::Poor
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Optimal => "Optimal",
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Acceptable => "Acceptable",
            Self::Poor => "Poor",
        }
    }
}

/// Calculate gap from optimal (or best known)
pub fn optimality_gap(solution_length: f64, optimal_length: f64) -> f64 {
    ((solution_length - optimal_length) / optimal_length) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution_new() {
        let solution = TspSolution::new(vec![0, 1, 2], 10.5);
        assert_eq!(solution.tour, vec![0, 1, 2]);
        assert!((solution.length - 10.5).abs() < 1e-10);
        assert_eq!(solution.evaluations, 0);
        assert_eq!(solution.history.len(), 1);
    }

    #[test]
    fn test_solution_comparison() {
        let better = TspSolution::new(vec![0, 1, 2], 10.0);
        let worse = TspSolution::new(vec![0, 2, 1], 15.0);

        assert!(better.is_better_than(&worse));
        assert!(!worse.is_better_than(&better));
    }

    #[test]
    fn test_algorithm_parsing() {
        assert_eq!(TspAlgorithm::parse("aco"), Some(TspAlgorithm::Aco));
        assert_eq!(TspAlgorithm::parse("ACO"), Some(TspAlgorithm::Aco));
        assert_eq!(TspAlgorithm::parse("tabu"), Some(TspAlgorithm::Tabu));
        assert_eq!(TspAlgorithm::parse("ga"), Some(TspAlgorithm::Ga));
        assert_eq!(TspAlgorithm::parse("hybrid"), Some(TspAlgorithm::Hybrid));
        assert_eq!(TspAlgorithm::parse("auto"), Some(TspAlgorithm::Hybrid));
        assert_eq!(TspAlgorithm::parse("unknown"), None);
    }

    #[test]
    fn test_algorithm_as_str() {
        assert_eq!(TspAlgorithm::Aco.as_str(), "aco");
        assert_eq!(TspAlgorithm::Tabu.as_str(), "tabu");
        assert_eq!(TspAlgorithm::Ga.as_str(), "ga");
        assert_eq!(TspAlgorithm::Hybrid.as_str(), "hybrid");
    }

    #[test]
    fn test_solution_tier_from_gap() {
        assert_eq!(SolutionTier::from_gap(0.05), SolutionTier::Optimal);
        assert_eq!(SolutionTier::from_gap(0.5), SolutionTier::Excellent);
        assert_eq!(SolutionTier::from_gap(1.5), SolutionTier::Good);
        assert_eq!(SolutionTier::from_gap(3.0), SolutionTier::Acceptable);
        assert_eq!(SolutionTier::from_gap(10.0), SolutionTier::Poor);
    }

    #[test]
    fn test_optimality_gap() {
        let gap = optimality_gap(102.0, 100.0);
        assert!((gap - 2.0).abs() < 1e-10);

        let gap = optimality_gap(100.0, 100.0);
        assert!((gap - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_budget_limit() {
        assert_eq!(Budget::Iterations(100).limit(), 100);
        assert_eq!(Budget::Evaluations(500).limit(), 500);
    }
}
