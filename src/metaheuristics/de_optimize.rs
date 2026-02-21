
impl PerturbativeMetaheuristic for DifferentialEvolution {
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
        // Reset state
        self.reset();

        // Create RNG for this optimization run
        let mut rng = self.make_rng();

        // Initialize population
        self.initialize_population(space, &mut rng);

        // Initial evaluation
        self.evaluate_population(objective);
        let mut tracker = ConvergenceTracker::from_budget(&budget);
        tracker.update(self.fitness[self.best_idx], self.population_size);
        self.history.push(self.fitness[self.best_idx]);

        let max_iter = budget.max_iterations(self.population_size);

        // Main loop
        for _iter in 0..max_iter {
            let evals = self.evolve_generation(objective, space, &mut rng);
            self.history.push(self.fitness[self.best_idx]);

            if !tracker.update(self.fitness[self.best_idx], evals) {
                break;
            }
        }

        // Determine termination reason
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

        // Reset adaptation state
        match &mut self.adaptation {
            AdaptationStrategy::None => {}
            AdaptationStrategy::JADE {
                archive,
                mu_f,
                mu_cr,
                ..
            } => {
                archive.clear();
                *mu_f = 0.5;
                *mu_cr = 0.5;
            }
            AdaptationStrategy::SHADE {
                memory_f,
                memory_cr,
                memory_index,
                memory_size,
            } => {
                *memory_f = vec![0.5; *memory_size];
                *memory_cr = vec![0.5; *memory_size];
                *memory_index = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[allow(dead_code)]
    fn rastrigin(x: &[f64]) -> f64 {
        let n = f64::from(x.len() as i32);
        10.0 * n
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    #[test]
    fn test_de_sphere_convergence() {
        let mut de = DifferentialEvolution::new().with_seed(42);
        let space = SearchSpace::continuous(10, -5.0, 5.0);
        let result = de.optimize(&sphere, &space, Budget::Evaluations(50_000));

        // Allow reasonable tolerance for 10D sphere
        assert!(
            result.objective_value < 1e-2,
            "DE should solve sphere, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_de_sphere_5d() {
        let mut de = DifferentialEvolution::with_params(50, 0.8, 0.9).with_seed(42);
        let space = SearchSpace::continuous(5, -10.0, 10.0);
        let result = de.optimize(&sphere, &space, Budget::Evaluations(10_000));

        assert!(
            result.objective_value < 1e-6,
            "5D sphere should converge to near-zero, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_de_rastrigin_finds_basin() {
        let mut de = DifferentialEvolution::new().with_seed(42);
        let space = SearchSpace::continuous(5, -5.12, 5.12);
        let result = de.optimize(&rastrigin, &space, Budget::Evaluations(30_000));

        // Rastrigin is harder; just check we find a reasonable solution
        assert!(
            result.objective_value < 10.0,
            "Should find good basin, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_de_jade_adaptation() {
        let mut de = DifferentialEvolution::new().with_jade().with_seed(42);
        let space = SearchSpace::continuous(10, -5.0, 5.0);
        let result = de.optimize(&sphere, &space, Budget::Evaluations(20_000));

        assert!(
            result.objective_value < 1e-4,
            "JADE should solve sphere, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_de_best_strategy() {
        let mut de = DifferentialEvolution::new()
            .with_strategy(DEStrategy::Best1Bin)
            .with_seed(42);
        let space = SearchSpace::continuous(10, -5.0, 5.0);
        let result = de.optimize(&sphere, &space, Budget::Evaluations(15_000));

        assert!(
            result.objective_value < 1e-4,
            "Best1Bin should solve sphere, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_de_convergence_history() {
        let mut de = DifferentialEvolution::new().with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = de.optimize(&sphere, &space, Budget::Iterations(50));

        // History should be monotonically non-increasing (or nearly so)
        assert!(!result.history.is_empty());

        // Check general trend is decreasing
        assert!(
            result.history.last().expect("history not empty")
                <= result.history.first().expect("history not empty")
        );
    }

    #[test]
    fn test_de_reset() {
        let mut de = DifferentialEvolution::new().with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);

        // First run
        let result1 = de.optimize(&sphere, &space, Budget::Evaluations(5000));

        // Reset and run again with same seed
        de = de.with_seed(42);
        let result2 = de.optimize(&sphere, &space, Budget::Evaluations(5000));

        // Should get same result (deterministic with same seed)
        assert!((result1.objective_value - result2.objective_value).abs() < 1e-10);
    }

    #[test]
    fn test_de_empty_before_optimize() {
        let de = DifferentialEvolution::new();
        assert!(de.best().is_none());
        assert!(de.history().is_empty());
    }

    #[test]
    fn test_de_shade_adaptation() {
        let mut de = DifferentialEvolution::new().with_shade(10).with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = de.optimize(&sphere, &space, Budget::Evaluations(10_000));

        assert!(
            result.objective_value < 1e-3,
            "SHADE should solve sphere, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_de_rand2bin_strategy() {
        let mut de = DifferentialEvolution::new()
            .with_strategy(DEStrategy::Rand2Bin)
            .with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = de.optimize(&sphere, &space, Budget::Evaluations(15_000));

        assert!(
            result.objective_value < 1e-3,
            "Rand2Bin should solve sphere, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_de_current_to_best_strategy() {
        let mut de = DifferentialEvolution::new()
            .with_strategy(DEStrategy::CurrentToBest1Bin)
            .with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = de.optimize(&sphere, &space, Budget::Evaluations(10_000));

        assert!(
            result.objective_value < 1e-4,
            "CurrentToBest1Bin should solve sphere, got {}",
            result.objective_value
        );
    }
}
