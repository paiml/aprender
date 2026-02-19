
impl DESearch {
    /// Create DE search with n iterations.
    #[must_use]
    pub fn new(n_iter: usize) -> Self {
        Self {
            n_iter,
            population_size: 0, // Auto
            seed: 42,
            strategy: DEStrategy::Rand1Bin,
            use_jade: false,
            population: Vec::new(),
            fitness: Vec::new(),
            best_idx: 0,
            param_order: Vec::new(),
            param_bounds: Vec::new(),
            trials_generated: 0,
            initialized: false,
            mutation_factor: 0.8,
            crossover_rate: 0.9,
        }
    }

    /// Set population size (0 = auto-select based on dimension).
    #[must_use]
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set mutation strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: DEStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable JADE adaptive parameters.
    #[must_use]
    pub fn with_jade(mut self) -> Self {
        self.use_jade = true;
        self
    }

    /// Set mutation factor F (default: 0.8).
    #[must_use]
    pub fn with_mutation_factor(mut self, f: f64) -> Self {
        self.mutation_factor = f;
        self
    }

    /// Set crossover rate CR (default: 0.9).
    #[must_use]
    pub fn with_crossover_rate(mut self, cr: f64) -> Self {
        self.crossover_rate = cr;
        self
    }

    /// Remaining trials to generate.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.n_iter.saturating_sub(self.trials_generated)
    }

    /// Initialize population from search space.
    fn initialize<P: ParamKey>(&mut self, space: &SearchSpace<P>) {
        // Extract parameter info in deterministic order
        self.param_order.clear();
        self.param_bounds.clear();

        let mut params: Vec<_> = space.params.iter().collect();
        params.sort_by(|a, b| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)));

        for (key, hyper) in params {
            let key_str = format!("{key:?}");
            let bounds = match hyper {
                HyperParam::Continuous { low, high, log_scale } => (*low, *high, false, *log_scale),
                HyperParam::Integer { low, high } => (*low as f64, *high as f64, true, false),
                HyperParam::Categorical { choices } => (0.0, (choices.len() - 1) as f64, true, false),
            };
            self.param_order.push(key_str);
            self.param_bounds.push(bounds);
        }

        let dim = self.param_bounds.len();
        let pop_size = if self.population_size == 0 {
            (10 * dim).clamp(20, 100)
        } else {
            self.population_size
        };

        // Initialize population with random values
        let mut rng = XorShift64::new(self.seed);
        self.population = (0..pop_size)
            .map(|_| {
                self.param_bounds
                    .iter()
                    .map(|(low, high, is_int, is_log)| {
                        let val = if *is_log {
                            let log_low = low.ln();
                            let log_high = high.ln();
                            (log_low + rng.gen_f64() * (log_high - log_low)).exp()
                        } else {
                            *low + rng.gen_f64() * (*high - *low)
                        };
                        if *is_int {
                            val.round()
                        } else {
                            val
                        }
                    })
                    .collect()
            })
            .collect();

        self.fitness = vec![f64::INFINITY; pop_size];
        self.initialized = true;
    }

    /// Convert parameter vector to Trial.
    fn vector_to_trial<P: ParamKey>(vec: &[f64], space: &SearchSpace<P>) -> Trial<P> {
        let mut values = HashMap::new();
        let mut params: Vec<_> = space.params.iter().collect();
        params.sort_by(|a, b| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)));

        for (i, (key, hyper)) in params.iter().enumerate() {
            let val = vec[i];
            let param_value = match hyper {
                HyperParam::Continuous { .. } => ParamValue::Float(val),
                HyperParam::Integer { .. } => ParamValue::Int(val.round() as i64),
                HyperParam::Categorical { choices } => {
                    let idx = (val.round() as usize).min(choices.len() - 1);
                    choices[idx].clone()
                }
            };
            values.insert(**key, param_value);
        }

        Trial { values }
    }

    /// Clip value to bounds.
    fn clip(&self, val: f64, idx: usize) -> f64 {
        let (low, high, is_int, _) = self.param_bounds[idx];
        let clipped = val.clamp(low, high);
        if is_int {
            clipped.round()
        } else {
            clipped
        }
    }

    /// Generate a mutant vector by applying a per-dimension formula.
    fn mutate_vector(&self, dim: usize, formula: impl Fn(usize) -> f64) -> Vec<f64> {
        (0..dim).map(formula).collect()
    }

    /// Select `count` distinct random indices, all different from `exclude`.
    fn select_distinct_indices(rng: &mut XorShift64, pop_size: usize, exclude: usize, count: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(count);
        while indices.len() < count {
            let idx = rng.gen_usize(pop_size);
            if idx != exclude && !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        indices
    }
}

impl<P: ParamKey> SearchStrategy<P> for DESearch {
    fn suggest(&mut self, space: &SearchSpace<P>, n: usize) -> Vec<Trial<P>> {
        if !self.initialized {
            self.initialize(space);
        }

        let n = n.min(self.remaining()).min(self.population.len());

        // Return current population members as trials
        let trials: Vec<Trial<P>> = self.population[..n]
            .iter()
            .map(|vec| Self::vector_to_trial(vec, space))
            .collect();

        self.trials_generated += trials.len();
        trials
    }

    fn update(&mut self, results: &[TrialResult<P>]) {
        if results.is_empty() || !self.initialized {
            return;
        }

        // Update fitness for evaluated individuals
        // Note: AutoML uses higher=better, DE uses lower=better
        for (i, result) in results.iter().enumerate() {
            if i < self.fitness.len() {
                // Negate score since DE minimizes
                self.fitness[i] = -result.score;
            }
        }

        // Update best
        self.best_idx = self
            .fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        // Evolve population for next generation
        let pop_size = self.population.len();
        let dim = self.param_bounds.len();
        let mut rng = XorShift64::new(self.seed.wrapping_add(self.trials_generated as u64));

        let mut new_population = self.population.clone();

        #[allow(clippy::needless_range_loop)]
        for i in 0..pop_size {
            let indices = Self::select_distinct_indices(&mut rng, pop_size, i, 3);
            let (a, b, c) = (indices[0], indices[1], indices[2]);
            let f = self.mutation_factor;
            let pop = &self.population;
            let best = self.best_idx;

            // Mutation based on strategy
            let mutant: Vec<f64> = match self.strategy {
                DEStrategy::Rand1Bin => self.mutate_vector(dim, |j| {
                    pop[a][j] + f * (pop[b][j] - pop[c][j])
                }),
                DEStrategy::Best1Bin => self.mutate_vector(dim, |j| {
                    pop[best][j] + f * (pop[a][j] - pop[b][j])
                }),
                DEStrategy::CurrentToBest1Bin => self.mutate_vector(dim, |j| {
                    pop[i][j] + f * (pop[best][j] - pop[i][j]) + f * (pop[a][j] - pop[b][j])
                }),
                DEStrategy::Rand2Bin => {
                    let more = Self::select_distinct_indices(&mut rng, pop_size, i, 5);
                    let (ra, rb, rc, d, e) = (more[0], more[1], more[2], more[3], more[4]);
                    self.mutate_vector(dim, |j| {
                        pop[ra][j] + f * (pop[rb][j] - pop[rc][j]) + f * (pop[d][j] - pop[e][j])
                    })
                }
            };

            // Crossover
            let j_rand = rng.gen_usize(dim);
            let trial: Vec<f64> = (0..dim)
                .map(|j| {
                    let use_mutant = j == j_rand || rng.gen_f64() < self.crossover_rate;
                    let val = if use_mutant {
                        mutant[j]
                    } else {
                        self.population[i][j]
                    };
                    self.clip(val, j)
                })
                .collect();

            // Selection will happen on next update
            // For now, just replace with trial (we'll get actual fitness next round)
            if self.fitness[i] == f64::INFINITY {
                // Not yet evaluated, keep trial
                new_population[i] = trial;
            }
            // Otherwise keep current (greedy selection happens implicitly via fitness)
        }

        self.population = new_population;
    }
}

/// Active Learning search optimizer.
///
/// Wraps any base search strategy and adds uncertainty-based stopping.
/// Implements the "Pull System" from Lean manufacturing - only generates
/// samples while uncertainty is high (Settles, 2009).
///
/// # Muda Elimination (Waste Reduction)
///
/// Traditional batch generation ("Push System") produces many redundant samples.
/// Active Learning stops when confidence saturates, eliminating overproduction.
///
/// # Example
///
/// ```
/// use aprender::automl::{ActiveLearningSearch, RandomSearch, SearchSpace, SearchStrategy};
/// use aprender::automl::params::RandomForestParam as RF;
///
/// let space = SearchSpace::new()
///     .add_continuous(RF::NEstimators, 10.0, 500.0);
///
/// let base = RandomSearch::new(1000).with_seed(42);
/// let mut search = ActiveLearningSearch::new(base)
///     .with_uncertainty_threshold(0.1)  // Stop when uncertainty < 0.1
///     .with_min_samples(10);            // Need at least 10 samples
///
/// // Pull system: generate until confident
/// while !search.should_stop() {
///     let trials = search.suggest(&space, 5);
///     if trials.is_empty() { break; }
///     // ... evaluate trials ...
///     // search.update(&results);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ActiveLearningSearch<S> {
    /// Base search strategy.
    base: S,
    /// Uncertainty threshold for stopping (default: 0.1).
    uncertainty_threshold: f64,
    /// Minimum samples before stopping is allowed.
    min_samples: usize,
    /// Collected scores for uncertainty estimation.
    scores: Vec<f64>,
    /// Current uncertainty estimate.
    current_uncertainty: f64,
}

impl<S> ActiveLearningSearch<S> {
    /// Create active learning wrapper around base strategy.
    #[must_use]
    pub fn new(base: S) -> Self {
        Self {
            base,
            uncertainty_threshold: 0.1,
            min_samples: 10,
            scores: Vec::new(),
            current_uncertainty: f64::INFINITY,
        }
    }

    /// Set uncertainty threshold for stopping.
    ///
    /// When estimated uncertainty drops below this threshold, `should_stop()` returns true.
    #[must_use]
    pub fn with_uncertainty_threshold(mut self, threshold: f64) -> Self {
        self.uncertainty_threshold = threshold;
        self
    }

    /// Set minimum samples before stopping is considered.
    #[must_use]
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples = min;
        self
    }

    /// Check if optimization should stop due to low uncertainty.
    ///
    /// Returns true when:
    /// 1. At least `min_samples` have been evaluated
    /// 2. Uncertainty is below `uncertainty_threshold`
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.scores.len() >= self.min_samples
            && self.current_uncertainty < self.uncertainty_threshold
    }

    /// Get current uncertainty estimate.
    ///
    /// Uses coefficient of variation (`std_dev` / mean) as uncertainty metric.
    /// Returns infinity if not enough samples.
    #[must_use]
    pub fn uncertainty(&self) -> f64 {
        self.current_uncertainty
    }

    /// Compute uncertainty from collected scores.
    ///
    /// Uses coefficient of variation: σ / μ
    /// - Low CV = consistent scores = low uncertainty
    /// - High CV = variable scores = high uncertainty
    fn compute_uncertainty(&mut self) {
        if self.scores.len() < 2 {
            self.current_uncertainty = f64::INFINITY;
            return;
        }

        let n = self.scores.len() as f64;
        let mean = self.scores.iter().sum::<f64>() / n;

        if mean.abs() < 1e-10 {
            // Avoid division by zero - if mean is ~0, use std dev directly
            let variance = self.scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            self.current_uncertainty = variance.sqrt();
        } else {
            let variance = self.scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std_dev = variance.sqrt();
            // Coefficient of variation
            self.current_uncertainty = std_dev / mean.abs();
        }
    }

    /// Get number of samples collected.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.scores.len()
    }
}
