impl PerturbativeMetaheuristic for CmaEs {
    type Solution = Vec<f64>;

    #[provable_contracts_macros::contract("cma-es-kernel-v1", equation = "mean_update")]
    #[allow(clippy::too_many_lines, clippy::needless_range_loop)]
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

        let (lower, upper) = match space {
            SearchSpace::Continuous { lower, upper, .. } => (lower.clone(), upper.clone()),
            _ => panic!("CMA-ES requires continuous search space"),
        };

        // Initialize mean at center of search space
        self.mean = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| (l + u) / 2.0)
            .collect();

        // Initialize sigma based on search range
        let range: f64 = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| u - l)
            .sum::<f64>()
            / self.dim as f64;
        self.sigma = range / 3.0;

        self.history.clear();
        self.best_value = f64::INFINITY;
        self.restart_count = 0;

        let mut tracker = ConvergenceTracker::from_budget(&budget);
        let max_iter = budget.max_iterations(self.initial_lambda);
        let chi_n = (self.dim as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * self.dim as f64) + 1.0 / (21.0 * (self.dim as f64).powi(2)));

        let mut gens_without_improvement = 0usize;
        let mut last_best = f64::INFINITY;

        for gen in 0..max_iter {
            // Sample population
            let population: Vec<Vec<f64>> = (0..self.lambda)
                .map(|_| {
                    let mut x = self.sample(&mut rng);
                    Self::clamp(&mut x, &lower, &upper);
                    x
                })
                .collect();

            // Evaluate
            let mut fitness: Vec<(usize, f64)> = population
                .iter()
                .enumerate()
                .map(|(i, x)| (i, objective(x)))
                .collect();

            // Sort by fitness (minimization)
            fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Update best
            if fitness[0].1 < self.best_value {
                self.best_value = fitness[0].1;
                self.best.clone_from(&population[fitness[0].0]);
            }

            // Track stagnation for IPOP
            if self.best_value < last_best - 1e-10 {
                gens_without_improvement = 0;
                last_best = self.best_value;
            } else {
                gens_without_improvement += 1;
            }

            self.history.push(self.best_value);

            // Check for IPOP restart
            if self.should_restart(gens_without_improvement) {
                self.perform_restart(&lower, &upper, &mut rng);
                gens_without_improvement = 0;
                continue; // Skip the rest of this iteration
            }

            // Update mean
            let old_mean = self.mean.clone();
            self.mean = vec![0.0; self.dim];
            for (rank, &(idx, _)) in fitness.iter().take(self.mu).enumerate() {
                for i in 0..self.dim {
                    self.mean[i] += self.weights[rank] * population[idx][i];
                }
            }

            // Update evolution paths and covariance
            let mean_diff: Vec<f64> = self
                .mean
                .iter()
                .zip(old_mean.iter())
                .map(|(m, o)| (m - o) / self.sigma)
                .collect();

            let p_sigma_norm = self.update_paths(&mean_diff, gen, chi_n);
            self.update_covariance(&population, &old_mean, &fitness, p_sigma_norm, chi_n);

            if !tracker.update(self.best_value, self.lambda) {
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
            solution: self.best.clone(),
            objective_value: self.best_value,
            evaluations: tracker.evaluations(),
            iterations: self.history.len(),
            history: self.history.clone(),
            termination,
        }
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
        self.p_sigma = vec![0.0; self.dim];
        self.p_c = vec![0.0; self.dim];
        self.c_diag = vec![1.0; self.dim];
        self.mean = vec![0.0; self.dim];
        self.history.clear();
        self.best.clear();
        self.best_value = f64::INFINITY;
        self.restart_count = 0;
        self.lambda = self.initial_lambda;
    }
}
