//! Budget specification for metaheuristic optimization.
//!
//! Defines termination conditions for optimization runs.

use serde::{Deserialize, Serialize};

/// Budget specification for optimization runs.
///
/// Determines when to stop the optimization process.
///
/// # Jidoka Note (Toyota Way)
///
/// Avoid `Time` variant in CI/TDD contexts as it introduces non-determinism.
/// Use `Evaluations` or `Iterations` for reproducible "Standard Work".
///
/// # Example
/// ```
/// use aprender::metaheuristics::Budget;
///
/// // Stop after 10,000 function evaluations
/// let budget = Budget::Evaluations(10_000);
///
/// // Stop after 100 generations
/// let budget = Budget::Iterations(100);
///
/// // Early stopping with patience
/// let budget = Budget::Convergence {
///     patience: 50,
///     min_delta: 1e-6,
///     max_evaluations: 100_000,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Budget {
    /// Maximum number of objective function evaluations.
    ///
    /// This is the most precise budget control and recommended for
    /// fair algorithm comparison.
    Evaluations(usize),

    /// Maximum number of iterations/generations.
    ///
    /// For population-based methods, one iteration typically means
    /// one generation (population_size evaluations).
    Iterations(usize),

    /// Early stopping based on convergence detection.
    ///
    /// Stops when the best objective value hasn't improved by at least
    /// `min_delta` for `patience` consecutive iterations.
    ///
    /// `max_evaluations` provides a safety bound.
    Convergence {
        /// Number of iterations without improvement before stopping
        patience: usize,
        /// Minimum improvement to reset patience counter
        min_delta: f64,
        /// Maximum evaluations (safety bound)
        max_evaluations: usize,
    },
}

impl Budget {
    /// Create an evaluation-based budget.
    #[must_use]
    pub const fn evaluations(n: usize) -> Self {
        Self::Evaluations(n)
    }

    /// Create an iteration-based budget.
    #[must_use]
    pub const fn iterations(n: usize) -> Self {
        Self::Iterations(n)
    }

    /// Create a convergence-based budget with default patience.
    ///
    /// Uses patience=50, min_delta=1e-8, max_evaluations=1_000_000.
    #[must_use]
    pub const fn convergence() -> Self {
        Self::Convergence {
            patience: 50,
            min_delta: 1e-8,
            max_evaluations: 1_000_000,
        }
    }

    /// Create a convergence-based budget with custom parameters.
    #[must_use]
    pub const fn convergence_with(patience: usize, min_delta: f64, max_evaluations: usize) -> Self {
        Self::Convergence {
            patience,
            min_delta,
            max_evaluations,
        }
    }

    /// Get the maximum number of iterations for planning purposes.
    ///
    /// For population-based methods, multiply by population size to get evaluations.
    #[must_use]
    pub fn max_iterations(&self, population_size: usize) -> usize {
        let pop = if population_size == 0 {
            1
        } else {
            population_size
        };
        match self {
            Self::Evaluations(n) => *n / pop,
            Self::Iterations(n) => *n,
            Self::Convergence {
                max_evaluations, ..
            } => *max_evaluations / pop,
        }
    }

    /// Get the maximum number of evaluations.
    #[must_use]
    pub fn max_evaluations(&self, population_size: usize) -> usize {
        match self {
            Self::Evaluations(n) => *n,
            Self::Iterations(n) => *n * population_size,
            Self::Convergence {
                max_evaluations, ..
            } => *max_evaluations,
        }
    }
}

impl Default for Budget {
    /// Default budget: 10,000 evaluations.
    fn default() -> Self {
        Self::Evaluations(10_000)
    }
}

/// Convergence tracker for early stopping.
///
/// Tracks optimization progress and determines when to stop.
#[derive(Debug, Clone)]
pub struct ConvergenceTracker {
    best_value: f64,
    no_improvement_count: usize,
    patience: usize,
    min_delta: f64,
    max_evaluations: usize,
    total_evaluations: usize,
}

impl ConvergenceTracker {
    /// Create a new convergence tracker from a budget.
    #[must_use]
    pub fn from_budget(budget: &Budget) -> Self {
        match budget {
            Budget::Convergence {
                patience,
                min_delta,
                max_evaluations,
            } => Self {
                best_value: f64::INFINITY,
                no_improvement_count: 0,
                patience: *patience,
                min_delta: *min_delta,
                max_evaluations: *max_evaluations,
                total_evaluations: 0,
            },
            Budget::Evaluations(n) => Self {
                best_value: f64::INFINITY,
                no_improvement_count: 0,
                patience: usize::MAX,
                min_delta: 0.0,
                max_evaluations: *n,
                total_evaluations: 0,
            },
            Budget::Iterations(n) => Self {
                best_value: f64::INFINITY,
                no_improvement_count: 0,
                patience: usize::MAX,
                min_delta: 0.0,
                max_evaluations: *n * 1000, // Assume large population
                total_evaluations: 0,
            },
        }
    }

    /// Update the tracker with a new best value.
    ///
    /// Returns `true` if optimization should continue.
    pub fn update(&mut self, value: f64, evaluations: usize) -> bool {
        self.total_evaluations += evaluations;

        if self.best_value - value > self.min_delta {
            // Improvement found
            self.best_value = value;
            self.no_improvement_count = 0;
        } else if value < self.best_value {
            // Small improvement (less than min_delta), still update best
            self.best_value = value;
            self.no_improvement_count += 1;
        } else {
            self.no_improvement_count += 1;
        }

        // Continue if within budget and not converged
        self.total_evaluations < self.max_evaluations && self.no_improvement_count < self.patience
    }

    /// Check if the budget is exhausted.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.total_evaluations >= self.max_evaluations
    }

    /// Check if convergence criteria is met.
    #[must_use]
    pub const fn is_converged(&self) -> bool {
        self.no_improvement_count >= self.patience
    }

    /// Get total evaluations so far.
    #[must_use]
    pub const fn evaluations(&self) -> usize {
        self.total_evaluations
    }

    /// Get current best value.
    #[must_use]
    pub const fn best(&self) -> f64 {
        self.best_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_evaluations() {
        let budget = Budget::evaluations(10_000);
        assert_eq!(budget.max_evaluations(100), 10_000);
        assert_eq!(budget.max_iterations(100), 100);
    }

    #[test]
    fn test_budget_iterations() {
        let budget = Budget::iterations(100);
        assert_eq!(budget.max_evaluations(50), 5_000);
        assert_eq!(budget.max_iterations(50), 100);
    }

    #[test]
    fn test_budget_convergence() {
        let budget = Budget::convergence_with(50, 1e-6, 100_000);
        assert_eq!(budget.max_evaluations(100), 100_000);
    }

    #[test]
    fn test_budget_default() {
        let budget = Budget::default();
        assert_eq!(budget.max_evaluations(1), 10_000);
    }

    #[test]
    fn test_convergence_tracker_improvement() {
        let budget = Budget::convergence_with(3, 0.1, 10_000);
        let mut tracker = ConvergenceTracker::from_budget(&budget);

        // Initial value
        assert!(tracker.update(10.0, 100));
        assert!((tracker.best() - 10.0).abs() < 1e-10);

        // Improvement > min_delta
        assert!(tracker.update(9.5, 100));
        assert!((tracker.best() - 9.5).abs() < 1e-10);
        assert_eq!(tracker.no_improvement_count, 0);

        // No improvement
        assert!(tracker.update(9.6, 100));
        assert_eq!(tracker.no_improvement_count, 1);

        // No improvement
        assert!(tracker.update(9.6, 100));
        assert_eq!(tracker.no_improvement_count, 2);

        // No improvement - should stop (patience=3 reached)
        assert!(!tracker.update(9.6, 100));
        assert!(tracker.is_converged());
    }

    #[test]
    fn test_convergence_tracker_budget_exhausted() {
        let budget = Budget::evaluations(500);
        let mut tracker = ConvergenceTracker::from_budget(&budget);

        // Use up budget
        for i in 0..5 {
            let should_continue = tracker.update(10.0 - f64::from(i), 100);
            if i < 4 {
                assert!(should_continue);
            } else {
                assert!(!should_continue);
            }
        }

        assert!(tracker.is_exhausted());
    }
}
