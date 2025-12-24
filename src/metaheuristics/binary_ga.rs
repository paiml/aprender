//! Binary Genetic Algorithm for Feature Selection
//!
//! Implements a genetic algorithm operating on binary vectors,
//! commonly used for feature selection and subset problems.
//!
//! # Example
//!
//! ```
//! use aprender::metaheuristics::{BinaryGA, SearchSpace, Budget, PerturbativeMetaheuristic};
//!
//! // Feature selection: minimize number of features while maximizing accuracy
//! let objective = |bits: &[f64]| {
//!     let selected: usize = bits.iter().filter(|&&b| b > 0.5).count();
//!     let penalty = if selected == 0 { 100.0 } else { 0.0 };
//!     // Simulate: fewer features = better, but need at least some
//!     selected as f64 * 0.1 + penalty
//! };
//!
//! let mut ga = BinaryGA::default().with_seed(42);
//! let space = SearchSpace::binary(10);  // 10 features
//! let result = ga.optimize(&objective, &space, Budget::Evaluations(1000));
//!
//! // Interpret: bits > 0.5 are selected features
//! let selected: Vec<usize> = result.solution.iter()
//!     .enumerate()
//!     .filter(|(_, &b)| b > 0.5)
//!     .map(|(i, _)| i)
//!     .collect();
//! ```

use super::{Budget, ConvergenceTracker, OptimizationResult, SearchSpace, TerminationReason};
use crate::metaheuristics::traits::PerturbativeMetaheuristic;
use rand::prelude::*;

/// Binary Genetic Algorithm
///
/// Uses tournament selection, uniform crossover, and bit-flip mutation.
/// Solutions are represented as f64 vectors where values > 0.5 indicate
/// selected features (1) and values <= 0.5 indicate unselected (0).
#[derive(Debug, Clone)]
pub struct BinaryGA {
    /// Population size
    pub population_size: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Mutation probability per bit
    pub mutation_prob: f64,
    /// Elitism: number of best individuals to preserve
    pub elitism: usize,
    /// Random seed for reproducibility
    seed: Option<u64>,
    /// Current population (as f64 for trait compatibility)
    population: Vec<Vec<f64>>,
    /// Fitness values
    fitness: Vec<f64>,
    /// Best individual index
    best_idx: usize,
    /// Convergence history
    history: Vec<f64>,
}

impl Default for BinaryGA {
    fn default() -> Self {
        Self {
            population_size: 100,
            tournament_size: 3,
            crossover_prob: 0.9,
            mutation_prob: 0.01, // Per-bit mutation rate
            elitism: 2,
            seed: None,
            population: Vec::new(),
            fitness: Vec::new(),
            best_idx: 0,
            history: Vec::new(),
        }
    }
}

impl BinaryGA {
    /// Set population size
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size.max(4);
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set mutation probability per bit
    pub fn with_mutation_prob(mut self, prob: f64) -> Self {
        self.mutation_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set crossover probability
    pub fn with_crossover_prob(mut self, prob: f64) -> Self {
        self.crossover_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Tournament selection
    fn tournament_select(&self, rng: &mut impl Rng) -> usize {
        let mut best = rng.gen_range(0..self.population_size);
        for _ in 1..self.tournament_size {
            let candidate = rng.gen_range(0..self.population_size);
            if self.fitness[candidate] < self.fitness[best] {
                best = candidate;
            }
        }
        best
    }

    /// Uniform crossover for binary vectors
    fn uniform_crossover(p1: &[f64], p2: &[f64], rng: &mut impl Rng) -> (Vec<f64>, Vec<f64>) {
        let dim = p1.len();
        let mut c1 = Vec::with_capacity(dim);
        let mut c2 = Vec::with_capacity(dim);

        for i in 0..dim {
            if rng.gen::<f64>() < 0.5 {
                c1.push(p1[i]);
                c2.push(p2[i]);
            } else {
                c1.push(p2[i]);
                c2.push(p1[i]);
            }
        }

        (c1, c2)
    }

    /// Bit-flip mutation
    fn mutate(&self, x: &mut [f64], rng: &mut impl Rng) {
        for bit in x.iter_mut() {
            if rng.gen::<f64>() < self.mutation_prob {
                // Flip: 0 -> 1 or 1 -> 0
                *bit = if *bit > 0.5 { 0.0 } else { 1.0 };
            }
        }
    }

    /// Get selected feature indices from solution
    pub fn selected_features(solution: &[f64]) -> Vec<usize> {
        solution
            .iter()
            .enumerate()
            .filter(|(_, &b)| b > 0.5)
            .map(|(i, _)| i)
            .collect()
    }
}

impl PerturbativeMetaheuristic for BinaryGA {
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

        let dim = match space {
            SearchSpace::Binary { dim } | SearchSpace::Continuous { dim, .. } => *dim,
            _ => panic!("BinaryGA requires Binary or Continuous search space"),
        };

        // Initialize population with random binary vectors
        self.population = (0..self.population_size)
            .map(|_| {
                (0..dim)
                    .map(|_| if rng.gen::<f64>() < 0.5 { 0.0 } else { 1.0 })
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

        for _gen in 0..max_iter {
            // Sort population by fitness for elitism
            let mut indices: Vec<usize> = (0..self.population_size).collect();
            indices.sort_by(|&a, &b| {
                self.fitness[a]
                    .partial_cmp(&self.fitness[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Create offspring
            let mut offspring = Vec::with_capacity(self.population_size);
            let mut offspring_fit = Vec::with_capacity(self.population_size);

            // Elitism: preserve best individuals
            for &i in indices.iter().take(self.elitism) {
                offspring.push(self.population[i].clone());
                offspring_fit.push(self.fitness[i]);
            }

            // Generate remaining offspring via crossover and mutation
            while offspring.len() < self.population_size {
                let p1 = self.tournament_select(&mut rng);
                let p2 = self.tournament_select(&mut rng);

                let (mut c1, mut c2) = if rng.gen::<f64>() < self.crossover_prob {
                    Self::uniform_crossover(&self.population[p1], &self.population[p2], &mut rng)
                } else {
                    (self.population[p1].clone(), self.population[p2].clone())
                };

                self.mutate(&mut c1, &mut rng);
                self.mutate(&mut c2, &mut rng);

                let f1 = objective(&c1);
                let f2 = objective(&c2);

                offspring.push(c1);
                offspring_fit.push(f1);

                if offspring.len() < self.population_size {
                    offspring.push(c2);
                    offspring_fit.push(f2);
                }
            }

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

        OptimizationResult {
            solution: self.population[self.best_idx].clone(),
            objective_value: self.fitness[self.best_idx],
            evaluations: tracker.evaluations(),
            iterations: self.history.len(),
            history: self.history.clone(),
            termination,
        }
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
    fn test_binary_ga_feature_selection() {
        // Objective: minimize selected features while keeping at least one
        let objective = |bits: &[f64]| {
            let count = bits.iter().filter(|&&b| b > 0.5).count();
            if count == 0 {
                100.0 // Penalty for no features
            } else {
                count as f64
            }
        };

        let mut ga = BinaryGA::default().with_seed(42).with_population_size(50);
        let space = SearchSpace::binary(10);
        let result = ga.optimize(&objective, &space, Budget::Evaluations(2000));

        // Should select minimal features (ideally 1)
        let selected = BinaryGA::selected_features(&result.solution);
        assert!(!selected.is_empty(), "Should select at least one feature");
        assert!(
            result.objective_value < 5.0,
            "Should minimize features, got: {}",
            result.objective_value
        );
    }

    #[test]
    fn test_binary_ga_onemax() {
        // OneMax: maximize number of 1s (minimize negative count)
        let objective = |bits: &[f64]| {
            let ones = bits.iter().filter(|&&b| b > 0.5).count();
            -(ones as f64) // Negate for minimization
        };

        let mut ga = BinaryGA::default().with_seed(123).with_population_size(100);
        let space = SearchSpace::binary(20);
        let result = ga.optimize(&objective, &space, Budget::Evaluations(5000));

        // Should find all 1s (objective = -20)
        assert!(
            result.objective_value < -15.0,
            "OneMax should find mostly 1s, got: {}",
            result.objective_value
        );
    }

    #[test]
    fn test_binary_ga_builder() {
        let ga = BinaryGA::default()
            .with_population_size(200)
            .with_mutation_prob(0.05)
            .with_seed(999);
        assert_eq!(ga.population_size, 200);
        assert!((ga.mutation_prob - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_selected_features_helper() {
        let solution = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let selected = BinaryGA::selected_features(&solution);
        assert_eq!(selected, vec![1, 3, 4]);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_binary_ga_with_crossover_prob() {
        let ga = BinaryGA::default().with_crossover_prob(0.7);
        assert!((ga.crossover_prob - 0.7).abs() < 1e-10);

        // Test clamping
        let ga2 = BinaryGA::default().with_crossover_prob(1.5);
        assert!((ga2.crossover_prob - 1.0).abs() < 1e-10);

        let ga3 = BinaryGA::default().with_crossover_prob(-0.5);
        assert!((ga3.crossover_prob - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_ga_mutation_prob_clamping() {
        let ga = BinaryGA::default().with_mutation_prob(2.0);
        assert!((ga.mutation_prob - 1.0).abs() < 1e-10);

        let ga2 = BinaryGA::default().with_mutation_prob(-1.0);
        assert!((ga2.mutation_prob - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_ga_population_size_min() {
        let ga = BinaryGA::default().with_population_size(1);
        assert_eq!(ga.population_size, 4); // Minimum is 4
    }

    #[test]
    fn test_binary_ga_best_empty() {
        let ga = BinaryGA::default();
        assert!(ga.best().is_none());
    }

    #[test]
    fn test_binary_ga_best_after_optimize() {
        let objective = |bits: &[f64]| bits.iter().sum::<f64>();

        let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
        let space = SearchSpace::binary(5);
        ga.optimize(&objective, &space, Budget::Evaluations(100));

        assert!(ga.best().is_some());
    }

    #[test]
    fn test_binary_ga_history_after_optimize() {
        let objective = |bits: &[f64]| bits.iter().sum::<f64>();

        let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
        let space = SearchSpace::binary(5);
        ga.optimize(&objective, &space, Budget::Evaluations(100));

        let history = ga.history();
        assert!(!history.is_empty());
        // History should be monotonically non-increasing (we're minimizing)
        for w in history.windows(2) {
            assert!(w[1] <= w[0] + 0.001, "History should not increase: {} > {}", w[1], w[0]);
        }
    }

    #[test]
    fn test_binary_ga_reset() {
        let objective = |bits: &[f64]| bits.iter().sum::<f64>();

        let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
        let space = SearchSpace::binary(5);
        ga.optimize(&objective, &space, Budget::Evaluations(100));

        assert!(ga.best().is_some());
        assert!(!ga.history().is_empty());

        ga.reset();

        assert!(ga.best().is_none());
        assert!(ga.history().is_empty());
    }

    #[test]
    fn test_binary_ga_debug() {
        let ga = BinaryGA::default();
        let debug_str = format!("{:?}", ga);
        assert!(debug_str.contains("BinaryGA"));
    }

    #[test]
    fn test_binary_ga_clone() {
        let ga = BinaryGA::default().with_seed(42).with_population_size(50);
        let cloned = ga.clone();
        assert_eq!(cloned.population_size, 50);
    }

    #[test]
    fn test_binary_ga_termination_converged() {
        // Use a simple objective that converges quickly
        let objective = |_bits: &[f64]| 0.0; // Always returns 0

        let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
        let space = SearchSpace::binary(5);
        let result = ga.optimize(&objective, &space, Budget::Evaluations(1000));

        // With constant objective, should converge
        assert!(
            result.termination == TerminationReason::Converged
                || result.termination == TerminationReason::BudgetExhausted
        );
    }

    #[test]
    fn test_binary_ga_no_seed() {
        // Test without seed (uses entropy)
        let objective = |bits: &[f64]| bits.iter().sum::<f64>();

        let mut ga = BinaryGA::default().with_population_size(20);
        let space = SearchSpace::binary(5);
        let result = ga.optimize(&objective, &space, Budget::Evaluations(100));

        assert!(result.objective_value.is_finite());
    }

    #[test]
    fn test_binary_ga_continuous_space_compatibility() {
        // BinaryGA can also work with continuous space (treated as binary)
        let objective = |bits: &[f64]| bits.iter().sum::<f64>();

        let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
        let space = SearchSpace::continuous(5, 0.0, 1.0);
        let result = ga.optimize(&objective, &space, Budget::Evaluations(100));

        assert!(result.objective_value.is_finite());
    }

    #[test]
    fn test_selected_features_all_zeros() {
        let solution = vec![0.0, 0.0, 0.0, 0.0];
        let selected = BinaryGA::selected_features(&solution);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_selected_features_all_ones() {
        let solution = vec![1.0, 1.0, 1.0];
        let selected = BinaryGA::selected_features(&solution);
        assert_eq!(selected, vec![0, 1, 2]);
    }
}
