//! Genetic Algorithm (GA) optimizer.
//!
//! An evolutionary algorithm using selection, crossover, and mutation.
//!
//! # Algorithm
//!
//! ```text
//! 1. Initialize population
//! 2. Evaluate fitness
//! 3. Select parents (tournament)
//! 4. Crossover (SBX)
//! 5. Mutate (polynomial)
//! 6. Replace and repeat
//! ```
//!
//! # References
//!
//! - Deb & Agrawal (1995): "Simulated Binary Crossover"
//! - Deb & Deb (2014): "Analyzing mutation schemes for real-parameter GA"

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Budget, OptimizationResult, PerturbativeMetaheuristic, SearchSpace};
use crate::metaheuristics::budget::ConvergenceTracker;
use crate::metaheuristics::traits::TerminationReason;

/// Genetic Algorithm optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticAlgorithm {
    /// Population size (default: 100)
    pub population_size: usize,
    /// Crossover probability (default: 0.9)
    pub crossover_prob: f64,
    /// Mutation probability (default: 1/dim)
    pub mutation_prob: Option<f64>,
    /// SBX distribution index (default: 20)
    pub sbx_eta: f64,
    /// Polynomial mutation index (default: 20)
    pub mutation_eta: f64,
    /// Tournament size (default: 2)
    pub tournament_size: usize,
    /// Random seed
    #[serde(default)]
    seed: Option<u64>,

    #[serde(skip)]
    population: Vec<Vec<f64>>,
    #[serde(skip)]
    fitness: Vec<f64>,
    #[serde(skip)]
    best_idx: usize,
    #[serde(skip)]
    history: Vec<f64>,
}

impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self {
            population_size: 100,
            crossover_prob: 0.9,
            mutation_prob: None,
            sbx_eta: 20.0,
            mutation_eta: 20.0,
            tournament_size: 2,
            seed: None,
            population: Vec::new(),
            fitness: Vec::new(),
            best_idx: 0,
            history: Vec::new(),
        }
    }
}

impl GeneticAlgorithm {
    /// Set population size.
    #[must_use]
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn tournament_select(&self, rng: &mut impl Rng) -> usize {
        let mut best = rng.random_range(0..self.population_size);
        for _ in 1..self.tournament_size {
            let candidate = rng.random_range(0..self.population_size);
            if self.fitness[candidate] < self.fitness[best] {
                best = candidate;
            }
        }
        best
    }

    fn sbx_crossover(
        &self,
        p1: &[f64],
        p2: &[f64],
        lower: &[f64],
        upper: &[f64],
        rng: &mut impl Rng,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = p1.len();
        let mut c1 = p1.to_vec();
        let mut c2 = p2.to_vec();

        if rng.random::<f64>() < self.crossover_prob {
            for i in 0..dim {
                if rng.random::<f64>() < 0.5 {
                    let y1 = p1[i].min(p2[i]);
                    let y2 = p1[i].max(p2[i]);
                    if (y2 - y1).abs() > 1e-14 {
                        let beta = 1.0 + (2.0 * (y1 - lower[i]) / (y2 - y1));
                        let alpha = 2.0 - beta.powf(-(self.sbx_eta + 1.0));
                        let u: f64 = rng.random();
                        let betaq = if u <= 1.0 / alpha {
                            (u * alpha).powf(1.0 / (self.sbx_eta + 1.0))
                        } else {
                            (1.0 / (2.0 - u * alpha)).powf(1.0 / (self.sbx_eta + 1.0))
                        };
                        c1[i] = (0.5 * ((y1 + y2) - betaq * (y2 - y1))).clamp(lower[i], upper[i]);
                        c2[i] = (0.5 * ((y1 + y2) + betaq * (y2 - y1))).clamp(lower[i], upper[i]);
                    }
                }
            }
        }
        (c1, c2)
    }

    fn polynomial_mutate(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        mut_prob: f64,
        rng: &mut impl Rng,
    ) {
        for i in 0..x.len() {
            if rng.random::<f64>() < mut_prob {
                let delta_max = upper[i] - lower[i];
                let u: f64 = rng.random();
                let delta = if u < 0.5 {
                    (2.0 * u).powf(1.0 / (self.mutation_eta + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).powf(1.0 / (self.mutation_eta + 1.0))
                };
                x[i] = (x[i] + delta * delta_max).clamp(lower[i], upper[i]);
            }
        }
    }
}

impl PerturbativeMetaheuristic for GeneticAlgorithm {
    type Solution = Vec<f64>;

    // Contract: metaheuristics-v1, equation = "ga_crossover"
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
            _ => panic!("GA requires continuous search space"),
        };

        let mut_prob = self.mutation_prob.unwrap_or(1.0 / dim as f64);

        // Initialize
        self.population = (0..self.population_size)
            .map(|_| {
                (0..dim)
                    .map(|j| rng.random_range(lower[j]..=upper[j]))
                    .collect()
            })
            .collect();
        self.fitness = self.population.iter().map(|x| objective(x)).collect();
        self.best_idx = self
            .fitness
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        self.history.clear();
        self.history.push(self.fitness[self.best_idx]);

        let mut tracker = ConvergenceTracker::from_budget(&budget);
        tracker.update(self.fitness[self.best_idx], self.population_size);

        let max_iter = budget.max_iterations(self.population_size);

        for _ in 0..max_iter {
            let mut offspring = Vec::with_capacity(self.population_size);
            let mut offspring_fit = Vec::with_capacity(self.population_size);

            while offspring.len() < self.population_size {
                let p1 = self.tournament_select(&mut rng);
                let p2 = self.tournament_select(&mut rng);
                let (mut c1, mut c2) = self.sbx_crossover(
                    &self.population[p1],
                    &self.population[p2],
                    &lower,
                    &upper,
                    &mut rng,
                );
                self.polynomial_mutate(&mut c1, &lower, &upper, mut_prob, &mut rng);
                self.polynomial_mutate(&mut c2, &lower, &upper, mut_prob, &mut rng);
                offspring_fit.push(objective(&c1));
                offspring_fit.push(objective(&c2));
                offspring.push(c1);
                offspring.push(c2);
            }

            offspring.truncate(self.population_size);
            offspring_fit.truncate(self.population_size);

            self.population = offspring;
            self.fitness = offspring_fit;
            self.best_idx = self
                .fitness
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);
            self.history.push(self.fitness[self.best_idx]);

            if !tracker.update(self.fitness[self.best_idx], self.population_size) {
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
            self.population[self.best_idx].clone(),
            self.fitness[self.best_idx],
            tracker.evaluations(),
            self.history.len(),
            self.history.clone(),
            termination,
        )
    }

    fn best(&self) -> Option<&Self::Solution> {
        if self.population.is_empty() {
            None
        } else {
            Some(&self.population[self.best_idx])
        }
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn reset(&mut self) {
        self.population.clear();
        self.fitness.clear();
        self.best_idx = 0;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ga_sphere() {
        let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
        let mut ga = GeneticAlgorithm::default()
            .with_seed(42)
            .with_population_size(50);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = ga.optimize(&objective, &space, Budget::Evaluations(5000));
        assert!(result.objective_value < 5.0);
    }

    #[test]
    fn test_ga_builder() {
        let ga = GeneticAlgorithm::default()
            .with_population_size(200)
            .with_seed(999);
        assert_eq!(ga.population_size, 200);
    }

    #[test]
    fn test_ga_reset() {
        let objective = |x: &[f64]| x.iter().sum::<f64>();
        let mut ga = GeneticAlgorithm::default().with_seed(42);
        let space = SearchSpace::continuous(2, -1.0, 1.0);
        let _ = ga.optimize(&objective, &space, Budget::Evaluations(200));
        assert!(ga.best().is_some());
        ga.reset();
        assert!(ga.best().is_none());
    }
}
