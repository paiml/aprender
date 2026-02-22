//! Harmony Search (HS) optimizer.
//!
//! A music-inspired metaheuristic based on musician improvisation.
//!
//! # Algorithm
//!
//! ```text
//! 1. Initialize harmony memory (HM)
//! 2. Improvise new harmony:
//!    - Memory consideration (HMCR)
//!    - Pitch adjustment (PAR)
//!    - Random selection
//! 3. Update HM if better
//! 4. Repeat
//! ```
//!
//! # References
//!
//! - Geem et al. (2001): "A New Heuristic Optimization Algorithm: Harmony Search"

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Budget, OptimizationResult, PerturbativeMetaheuristic, SearchSpace};
use crate::metaheuristics::budget::ConvergenceTracker;
use crate::metaheuristics::traits::TerminationReason;

/// Harmony Search optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonySearch {
    /// Harmony memory size (default: 30)
    pub memory_size: usize,
    /// Harmony memory considering rate (default: 0.95)
    pub hmcr: f64,
    /// Pitch adjusting rate (default: 0.3)
    pub par: f64,
    /// Bandwidth for pitch adjustment (default: 0.01)
    pub bandwidth: f64,
    /// Random seed
    #[serde(default)]
    seed: Option<u64>,

    #[serde(skip)]
    memory: Vec<Vec<f64>>,
    #[serde(skip)]
    fitness: Vec<f64>,
    #[serde(skip)]
    best_idx: usize,
    #[serde(skip)]
    history: Vec<f64>,
}

impl Default for HarmonySearch {
    fn default() -> Self {
        Self {
            memory_size: 30,
            hmcr: 0.95,
            par: 0.3,
            bandwidth: 0.01,
            seed: None,
            memory: Vec::new(),
            fitness: Vec::new(),
            best_idx: 0,
            history: Vec::new(),
        }
    }
}

impl HarmonySearch {
    /// Set memory size.
    #[must_use]
    pub fn with_memory_size(mut self, size: usize) -> Self {
        self.memory_size = size;
        self
    }

    /// Set HMCR.
    #[must_use]
    pub fn with_hmcr(mut self, hmcr: f64) -> Self {
        self.hmcr = hmcr;
        self
    }

    /// Set random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn improvise(&self, lower: &[f64], upper: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        let dim = lower.len();
        let mut harmony = Vec::with_capacity(dim);

        for i in 0..dim {
            let val = if rng.random::<f64>() < self.hmcr {
                // Memory consideration: pick from harmony memory
                let idx = rng.random_range(0..self.memory_size);
                let mut v = self.memory[idx][i];

                // Pitch adjustment
                if rng.random::<f64>() < self.par {
                    let range = upper[i] - lower[i];
                    v += rng.random_range(-1.0..=1.0) * range * self.bandwidth;
                }
                v.clamp(lower[i], upper[i])
            } else {
                // Random selection
                rng.random_range(lower[i]..=upper[i])
            };
            harmony.push(val);
        }
        harmony
    }
}

impl PerturbativeMetaheuristic for HarmonySearch {
    type Solution = Vec<f64>;

    fn optimize<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        budget: Budget,
    ) -> OptimizationResult<Self::Solution>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        let (lower, upper, dim) = match space {
            SearchSpace::Continuous { dim, lower, upper } => (lower.clone(), upper.clone(), *dim),
            _ => panic!("HS requires continuous search space"),
        };

        // Initialize harmony memory
        self.memory = (0..self.memory_size)
            .map(|_| {
                (0..dim)
                    .map(|j| rng.random_range(lower[j]..=upper[j]))
                    .collect()
            })
            .collect();
        self.fitness = self.memory.iter().map(|h| objective(h)).collect();
        self.best_idx = self
            .fitness
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        self.history.clear();
        self.history.push(self.fitness[self.best_idx]);

        let mut tracker = ConvergenceTracker::from_budget(&budget);
        tracker.update(self.fitness[self.best_idx], self.memory_size);

        let max_iter = budget.max_iterations(1);

        for _ in 0..max_iter {
            // Improvise new harmony
            let new_harmony = self.improvise(&lower, &upper, &mut rng);
            let new_fitness = objective(&new_harmony);

            // Find worst in memory
            let worst_idx = self
                .fitness
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);

            // Replace if better
            if new_fitness < self.fitness[worst_idx] {
                self.memory[worst_idx] = new_harmony;
                self.fitness[worst_idx] = new_fitness;

                if new_fitness < self.fitness[self.best_idx] {
                    self.best_idx = worst_idx;
                }
            }

            self.history.push(self.fitness[self.best_idx]);

            if !tracker.update(self.fitness[self.best_idx], 1) {
                break;
            }
        }

        let termination = if tracker.is_converged() {
            TerminationReason::Converged
        } else if tracker.is_exhausted() {
            TerminationReason::BudgetExhausted
        } else {
            TerminationReason::MaxIterations
        };

        OptimizationResult::new(
            self.memory[self.best_idx].clone(),
            self.fitness[self.best_idx],
            tracker.evaluations(),
            self.history.len(),
            self.history.clone(),
            termination,
        )
    }

    fn best(&self) -> Option<&Self::Solution> {
        if self.memory.is_empty() {
            None
        } else {
            Some(&self.memory[self.best_idx])
        }
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn reset(&mut self) {
        self.memory.clear();
        self.fitness.clear();
        self.best_idx = 0;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hs_sphere() {
        let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
        let mut hs = HarmonySearch::default().with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = hs.optimize(&objective, &space, Budget::Evaluations(5000));
        assert!(result.objective_value < 1.0);
    }

    #[test]
    fn test_hs_builder() {
        let hs = HarmonySearch::default()
            .with_memory_size(50)
            .with_hmcr(0.9)
            .with_seed(123);
        assert_eq!(hs.memory_size, 50);
        assert!((hs.hmcr - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_hs_reset() {
        let objective = |x: &[f64]| x.iter().sum::<f64>();
        let mut hs = HarmonySearch::default().with_seed(42);
        let space = SearchSpace::continuous(2, -1.0, 1.0);
        let _ = hs.optimize(&objective, &space, Budget::Evaluations(100));
        assert!(hs.best().is_some());
        hs.reset();
        assert!(hs.best().is_none());
    }
}
