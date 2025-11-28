//! Simulated Annealing (SA) optimizer.
//!
//! A probabilistic single-point metaheuristic based on metallurgical annealing.
//!
//! # Algorithm
//!
//! ```text
//! x' = perturb(x)
//! ΔE = f(x') - f(x)
//! Accept if ΔE < 0 or rand() < exp(-ΔE/T)
//! T = α·T (cooling)
//! ```
//!
//! # References
//!
//! - Kirkpatrick et al. (1983): "Optimization by Simulated Annealing"

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Budget, OptimizationResult, PerturbativeMetaheuristic, SearchSpace};
use crate::metaheuristics::budget::ConvergenceTracker;
use crate::metaheuristics::traits::TerminationReason;

/// Simulated Annealing optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedAnnealing {
    /// Initial temperature (default: 100.0)
    pub initial_temp: f64,
    /// Final temperature (default: 1e-8)
    pub final_temp: f64,
    /// Cooling rate α (default: 0.95)
    pub cooling_rate: f64,
    /// Perturbation scale (fraction of range, default: 0.1)
    pub step_scale: f64,
    /// Random seed
    #[serde(default)]
    seed: Option<u64>,

    // Internal state
    #[serde(skip)]
    current: Vec<f64>,
    #[serde(skip)]
    current_val: f64,
    #[serde(skip)]
    best: Vec<f64>,
    #[serde(skip)]
    best_val: f64,
    #[serde(skip)]
    history: Vec<f64>,
}

impl Default for SimulatedAnnealing {
    fn default() -> Self {
        Self {
            initial_temp: 100.0,
            final_temp: 1e-8,
            cooling_rate: 0.95,
            step_scale: 0.1,
            seed: None,
            current: Vec::new(),
            current_val: f64::INFINITY,
            best: Vec::new(),
            best_val: f64::INFINITY,
            history: Vec::new(),
        }
    }
}

impl SimulatedAnnealing {
    /// Set initial temperature.
    #[must_use]
    pub fn with_initial_temp(mut self, t: f64) -> Self {
        self.initial_temp = t;
        self
    }

    /// Set cooling rate.
    #[must_use]
    pub fn with_cooling_rate(mut self, alpha: f64) -> Self {
        self.cooling_rate = alpha;
        self
    }

    /// Set random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn perturb(&self, x: &[f64], lower: &[f64], upper: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let range = upper[i] - lower[i];
                let delta = rng.gen_range(-1.0..=1.0) * range * self.step_scale;
                (xi + delta).clamp(lower[i], upper[i])
            })
            .collect()
    }
}

impl PerturbativeMetaheuristic for SimulatedAnnealing {
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
            None => Box::new(thread_rng()),
        };

        let (lower, upper, dim) = match space {
            SearchSpace::Continuous { dim, lower, upper } => (lower.clone(), upper.clone(), *dim),
            _ => panic!("SA requires continuous search space"),
        };

        // Initialize with random solution
        self.current = (0..dim)
            .map(|j| rng.gen_range(lower[j]..=upper[j]))
            .collect();
        self.current_val = objective(&self.current);
        self.best = self.current.clone();
        self.best_val = self.current_val;
        self.history.clear();
        self.history.push(self.best_val);

        let mut tracker = ConvergenceTracker::from_budget(&budget);
        tracker.update(self.best_val, 1);

        let mut temp = self.initial_temp;
        let max_iter = budget.max_evaluations(1);

        for _ in 0..max_iter {
            if temp < self.final_temp {
                break;
            }

            // Perturb
            let candidate = self.perturb(&self.current, &lower, &upper, &mut rng);
            let candidate_val = objective(&candidate);

            // Metropolis acceptance
            let delta = candidate_val - self.current_val;
            let accept = delta < 0.0 || rng.gen::<f64>() < (-delta / temp).exp();

            if accept {
                self.current = candidate;
                self.current_val = candidate_val;

                if self.current_val < self.best_val {
                    self.best = self.current.clone();
                    self.best_val = self.current_val;
                }
            }

            self.history.push(self.best_val);

            // Cooling
            temp *= self.cooling_rate;

            if !tracker.update(self.best_val, 1) {
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
            self.best.clone(),
            self.best_val,
            tracker.evaluations(),
            self.history.len(),
            self.history.clone(),
            termination,
        )
    }

    fn best(&self) -> Option<&Self::Solution> {
        if self.best.is_empty() {
            None
        } else {
            Some(&self.best)
        }
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn reset(&mut self) {
        self.current.clear();
        self.current_val = f64::INFINITY;
        self.best.clear();
        self.best_val = f64::INFINITY;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sa_sphere() {
        let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
        let mut sa = SimulatedAnnealing::default().with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = sa.optimize(&objective, &space, Budget::Evaluations(10000));
        assert!(result.objective_value < 1.0);
    }

    #[test]
    fn test_sa_builder() {
        let sa = SimulatedAnnealing::default()
            .with_initial_temp(500.0)
            .with_cooling_rate(0.99)
            .with_seed(123);
        assert!((sa.initial_temp - 500.0).abs() < 1e-10);
        assert!((sa.cooling_rate - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_sa_reset() {
        let objective = |x: &[f64]| x.iter().sum::<f64>();
        let mut sa = SimulatedAnnealing::default().with_seed(42);
        let space = SearchSpace::continuous(2, -1.0, 1.0);
        let _ = sa.optimize(&objective, &space, Budget::Evaluations(100));
        assert!(sa.best().is_some());
        sa.reset();
        assert!(sa.best().is_none());
    }
}
