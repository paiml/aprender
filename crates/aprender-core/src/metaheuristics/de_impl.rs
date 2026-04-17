
/// Accumulator for one DE generation's side-effects.
#[derive(Default)]
struct GenerationOutcomes {
    evaluations: usize,
    successful_f: Vec<f64>,
    successful_cr: Vec<f64>,
    improvements: Vec<f64>,
    replacements: Vec<(usize, Vec<f64>, f64)>,
    archive_additions: Vec<Vec<f64>>,
}

impl DifferentialEvolution {
    /// Create a new DE optimizer with default parameters.
    ///
    /// Default: F=0.8, CR=0.9, strategy=Rand1Bin, no adaptation
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create DE with custom parameters.
    ///
    /// # Arguments
    /// * `population_size` - Number of individuals (0 = auto-select based on dimension)
    /// * `mutation_factor` - F parameter (typically 0.4-1.0)
    /// * `crossover_rate` - CR parameter (typically 0.1-0.9)
    #[must_use]
    pub fn with_params(population_size: usize, mutation_factor: f64, crossover_rate: f64) -> Self {
        Self {
            population_size,
            mutation_factor,
            crossover_rate,
            ..Default::default()
        }
    }

    /// Set the mutation strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: DEStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable JADE adaptation.
    #[must_use]
    pub fn with_jade(mut self) -> Self {
        self.adaptation = AdaptationStrategy::JADE {
            archive: Vec::new(),
            archive_size: 0, // Set during initialization
            mu_f: 0.5,
            mu_cr: 0.5,
            c: 0.1,
        };
        self
    }

    /// Enable SHADE adaptation.
    #[must_use]
    pub fn with_shade(mut self, memory_size: usize) -> Self {
        self.adaptation = AdaptationStrategy::SHADE {
            memory_f: vec![0.5; memory_size],
            memory_cr: vec![0.5; memory_size],
            memory_size,
            memory_index: 0,
        };
        self
    }

    /// Set random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Create RNG from seed or entropy.
    fn make_rng(&self) -> StdRng {
        match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        }
    }

    /// Initialize population within bounds.
    fn initialize_population(&mut self, space: &SearchSpace, rng: &mut StdRng) {
        let dim = space.dimension();
        let pop_size = if self.population_size == 0 {
            (10 * dim).clamp(20, 200)
        } else {
            self.population_size
        };

        self.population_size = pop_size;
        self.population = Vec::with_capacity(pop_size);
        self.fitness = vec![f64::INFINITY; pop_size];

        if let SearchSpace::Continuous { lower, upper, .. }
        | SearchSpace::Mixed { lower, upper, .. } = space
        {
            for _ in 0..pop_size {
                let individual: Vec<f64> = lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(&lo, &hi)| rng.random_range(lo..=hi))
                    .collect();
                self.population.push(individual);
            }
        }

        // Initialize JADE archive size
        if let AdaptationStrategy::JADE { archive_size, .. } = &mut self.adaptation {
            *archive_size = pop_size;
        }
    }

    /// Evaluate the entire population.
    fn evaluate_population<F>(&mut self, objective: &F)
    where
        F: Fn(&[f64]) -> f64,
    {
        for (i, individual) in self.population.iter().enumerate() {
            self.fitness[i] = objective(individual);
        }
        self.update_best();
    }

    /// Update best individual index.
    fn update_best(&mut self) {
        self.best_idx = self
            .fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
    }

    /// Generate mutant vector based on strategy.
    fn mutate(&self, target_idx: usize, f: f64, rng: &mut StdRng) -> Vec<f64> {
        let dim = self.population[0].len();

        match self.strategy {
            DEStrategy::Rand1Bin => {
                let (a, b, c) = self.select_random_triple(target_idx, rng);
                (0..dim)
                    .map(|j| {
                        self.population[a][j] + f * (self.population[b][j] - self.population[c][j])
                    })
                    .collect()
            }
            DEStrategy::Best1Bin => {
                let (a, b, _) = self.select_random_triple(target_idx, rng);
                (0..dim)
                    .map(|j| {
                        self.population[self.best_idx][j]
                            + f * (self.population[a][j] - self.population[b][j])
                    })
                    .collect()
            }
            DEStrategy::Rand2Bin => {
                let indices = self.select_random_five(target_idx, rng);
                (0..dim)
                    .map(|j| {
                        self.population[indices[0]][j]
                            + f * (self.population[indices[1]][j] - self.population[indices[2]][j])
                            + f * (self.population[indices[3]][j] - self.population[indices[4]][j])
                    })
                    .collect()
            }
            DEStrategy::CurrentToBest1Bin => {
                let (a, b, _) = self.select_random_triple(target_idx, rng);
                (0..dim)
                    .map(|j| {
                        self.population[target_idx][j]
                            + f * (self.population[self.best_idx][j]
                                - self.population[target_idx][j])
                            + f * (self.population[a][j] - self.population[b][j])
                    })
                    .collect()
            }
        }
    }

    /// Select 3 distinct random indices (excluding target).
    fn select_random_triple(&self, exclude: usize, rng: &mut StdRng) -> (usize, usize, usize) {
        let n = self.population.len();
        let mut indices = Vec::with_capacity(3);

        // Use a simple rejection sampling approach
        while indices.len() < 3 {
            let idx = rng.random_range(0..n);
            if idx != exclude && !indices.contains(&idx) {
                indices.push(idx);
            }
        }

        (indices[0], indices[1], indices[2])
    }

    /// Select 5 distinct random indices (excluding target).
    fn select_random_five(&self, exclude: usize, rng: &mut StdRng) -> [usize; 5] {
        let n = self.population.len();
        let mut indices = Vec::with_capacity(5);

        while indices.len() < 5 {
            let idx = rng.random_range(0..n);
            if idx != exclude && !indices.contains(&idx) {
                indices.push(idx);
            }
        }

        [indices[0], indices[1], indices[2], indices[3], indices[4]]
    }

    /// Binomial crossover (static helper).
    fn crossover(target: &[f64], mutant: &[f64], cr: f64, rng: &mut StdRng) -> Vec<f64> {
        let dim = target.len();

        // Ensure at least one dimension comes from mutant
        let j_rand = rng.random_range(0..dim);

        (0..dim)
            .map(|j| {
                if j == j_rand || rng.random::<f64>() < cr {
                    mutant[j]
                } else {
                    target[j]
                }
            })
            .collect()
    }

    /// Clip trial vector to bounds (static helper).
    fn clip_to_bounds(trial: &mut [f64], space: &SearchSpace) {
        if let Some(clipped) = space.clip(trial) {
            trial.copy_from_slice(&clipped);
        }
    }

    /// Get adaptive F and CR values (for JADE/SHADE).
    ///
    /// Uses simplified normal perturbation instead of Cauchy for F
    /// (avoids `rand_distr` dependency).
    fn get_adaptive_params(&self, rng: &mut StdRng) -> (f64, f64) {
        match &self.adaptation {
            AdaptationStrategy::None => (self.mutation_factor, self.crossover_rate),
            AdaptationStrategy::JADE { mu_f, mu_cr, .. } => {
                // Simplified: use uniform perturbation around mu
                // Original JADE uses Cauchy for F, Normal for CR
                let f = (*mu_f + rng.random_range(-0.2..0.2)).clamp(0.1, 1.0);
                let cr = (*mu_cr + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0);
                (f, cr)
            }
            AdaptationStrategy::SHADE {
                memory_f,
                memory_cr,
                ..
            } => {
                let idx = rng.random_range(0..memory_f.len());
                let f = (memory_f[idx] + rng.random_range(-0.2..0.2)).clamp(0.1, 1.0);
                let cr = (memory_cr[idx] + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0);
                (f, cr)
            }
        }
    }

    /// Perform one generation of DE.
    fn evolve_generation<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        rng: &mut StdRng,
    ) -> usize
    where
        F: Fn(&[f64]) -> f64,
    {
        let pop_size = self.population.len();
        let mut outcomes = GenerationOutcomes::default();

        for i in 0..pop_size {
            self.evaluate_trial(i, objective, space, rng, &mut outcomes);
        }

        for (i, trial, trial_fitness) in outcomes.replacements {
            self.population[i] = trial;
            self.fitness[i] = trial_fitness;
        }

        self.update_jade_archive(outcomes.archive_additions, rng);
        self.update_best();
        self.update_adaptation(
            &outcomes.successful_f,
            &outcomes.successful_cr,
            &outcomes.improvements,
        );

        outcomes.evaluations
    }

    /// Evaluate one trial (mutate + crossover + select) and record outcomes.
    fn evaluate_trial<F>(
        &self,
        i: usize,
        objective: &F,
        space: &SearchSpace,
        rng: &mut StdRng,
        outcomes: &mut GenerationOutcomes,
    ) where
        F: Fn(&[f64]) -> f64,
    {
        let (f, cr) = self.get_adaptive_params(rng);
        let mutant = self.mutate(i, f, rng);
        let mut trial = Self::crossover(&self.population[i], &mutant, cr, rng);
        Self::clip_to_bounds(&mut trial, space);
        let trial_fitness = objective(&trial);
        outcomes.evaluations += 1;

        if trial_fitness > self.fitness[i] {
            return;
        }
        if trial_fitness < self.fitness[i] {
            outcomes.successful_f.push(f);
            outcomes.successful_cr.push(cr);
            outcomes.improvements.push(self.fitness[i] - trial_fitness);
            if matches!(self.adaptation, AdaptationStrategy::JADE { .. }) {
                outcomes.archive_additions.push(self.population[i].clone());
            }
        }
        outcomes.replacements.push((i, trial, trial_fitness));
    }

    /// JADE: merge old individuals into the archive, replacing randomly when full.
    fn update_jade_archive(&mut self, additions: Vec<Vec<f64>>, rng: &mut StdRng) {
        let AdaptationStrategy::JADE {
            archive,
            archive_size,
            ..
        } = &mut self.adaptation
        else {
            return;
        };
        for individual in additions {
            if archive.len() < *archive_size {
                archive.push(individual);
            } else if !archive.is_empty() {
                let idx = rng.random_range(0..archive.len());
                archive[idx] = individual;
            }
        }
    }

    /// Update adaptive parameters based on successful mutations.
    fn update_adaptation(
        &mut self,
        successful_f: &[f64],
        successful_cr: &[f64],
        improvements: &[f64],
    ) {
        if successful_f.is_empty() {
            return;
        }

        match &mut self.adaptation {
            AdaptationStrategy::None => {}
            AdaptationStrategy::JADE { mu_f, mu_cr, c, .. } => {
                Self::update_jade(mu_f, mu_cr, *c, successful_f, successful_cr, improvements);
            }
            AdaptationStrategy::SHADE {
                memory_f,
                memory_cr,
                memory_index,
                memory_size,
            } => {
                Self::update_shade(
                    memory_f,
                    memory_cr,
                    memory_index,
                    *memory_size,
                    successful_f,
                    successful_cr,
                    improvements,
                );
            }
        }
    }

    fn update_jade(
        mu_f: &mut f64,
        mu_cr: &mut f64,
        c: f64,
        successful_f: &[f64],
        successful_cr: &[f64],
        improvements: &[f64],
    ) {
        let f_sum: f64 = successful_f.iter().sum();
        let f_sq_sum: f64 = successful_f.iter().map(|f| f * f).sum();
        if f_sum > 0.0 {
            let mean_f = f_sq_sum / f_sum;
            *mu_f = (1.0 - c) * (*mu_f) + c * mean_f;
        }

        let total_improvement: f64 = improvements.iter().sum();
        if total_improvement > 0.0 {
            let mean_cr: f64 = successful_cr
                .iter()
                .zip(improvements.iter())
                .map(|(cr, imp)| cr * imp)
                .sum::<f64>()
                / total_improvement;
            *mu_cr = (1.0 - c) * (*mu_cr) + c * mean_cr;
        }
    }

    fn update_shade(
        memory_f: &mut [f64],
        memory_cr: &mut [f64],
        memory_index: &mut usize,
        memory_size: usize,
        successful_f: &[f64],
        successful_cr: &[f64],
        improvements: &[f64],
    ) {
        let f_sum: f64 = successful_f.iter().sum();
        let f_sq_sum: f64 = successful_f.iter().map(|f| f * f).sum();
        if f_sum > 0.0 {
            memory_f[*memory_index] = f_sq_sum / f_sum;
        }

        let total_improvement: f64 = improvements.iter().sum();
        if total_improvement > 0.0 {
            memory_cr[*memory_index] = successful_cr
                .iter()
                .zip(improvements.iter())
                .map(|(cr, imp)| cr * imp)
                .sum::<f64>()
                / total_improvement;
        }

        *memory_index = (*memory_index + 1) % memory_size;
    }
}

impl Default for DifferentialEvolution {
    fn default() -> Self {
        Self {
            population_size: 0, // Auto-select
            mutation_factor: 0.8,
            crossover_rate: 0.9,
            strategy: DEStrategy::default(),
            adaptation: AdaptationStrategy::default(),
            seed: None,
            population: Vec::new(),
            fitness: Vec::new(),
            best_idx: 0,
            history: Vec::new(),
        }
    }
}
