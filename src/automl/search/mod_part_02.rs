
/// A hyperparameter configuration to evaluate.
#[derive(Debug, Clone)]
pub struct Trial<P: ParamKey = GenericParam> {
    /// Parameter values for this trial.
    pub values: HashMap<P, ParamValue>,
}

impl<P: ParamKey> Trial<P> {
    /// Get a parameter value.
    #[must_use]
    pub fn get(&self, key: &P) -> Option<&ParamValue> {
        self.values.get(key)
    }

    /// Get parameter as f64.
    #[must_use]
    pub fn get_f64(&self, key: &P) -> Option<f64> {
        self.values.get(key).and_then(ParamValue::as_f64)
    }

    /// Get parameter as i64.
    #[must_use]
    pub fn get_i64(&self, key: &P) -> Option<i64> {
        self.values.get(key).and_then(ParamValue::as_i64)
    }

    /// Get parameter as usize.
    #[must_use]
    pub fn get_usize(&self, key: &P) -> Option<usize> {
        self.values
            .get(key)
            .and_then(ParamValue::as_i64)
            .map(|v| v as usize)
    }

    /// Get parameter as bool.
    #[must_use]
    pub fn get_bool(&self, key: &P) -> Option<bool> {
        self.values.get(key).and_then(ParamValue::as_bool)
    }
}

impl<P: ParamKey> std::fmt::Display for Trial<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<String> = self
            .values
            .iter()
            .map(|(k, v)| format!("{}={}", k.name(), v))
            .collect();
        write!(f, "{{{}}}", params.join(", "))
    }
}

/// Result of evaluating a trial.
#[derive(Debug, Clone)]
pub struct TrialResult<P: ParamKey = GenericParam> {
    /// The trial configuration.
    pub trial: Trial<P>,
    /// Objective score (higher is better by default).
    pub score: f64,
    /// Additional metrics.
    pub metrics: HashMap<String, f64>,
}

/// Search strategy trait for hyperparameter optimization.
pub trait SearchStrategy<P: ParamKey> {
    /// Generate candidate configurations to evaluate.
    fn suggest(&mut self, space: &SearchSpace<P>, n: usize) -> Vec<Trial<P>>;

    /// Update strategy with evaluation results (for adaptive methods).
    fn update(&mut self, _results: &[TrialResult<P>]) {}
}

/// Simple random number generator trait.
pub trait Rng {
    /// Generate uniform random in [0, 1).
    fn gen_f64(&mut self) -> f64;

    /// Generate random f64 in range [low, high).
    fn gen_f64_range(&mut self, low: f64, high: f64) -> f64 {
        low + self.gen_f64() * (high - low)
    }

    /// Generate random i64 in range [low, high].
    fn gen_i64_range(&mut self, low: i64, high: i64) -> i64;

    /// Generate random usize in range [0, len).
    fn gen_usize(&mut self, len: usize) -> usize;
}

/// Simple xorshift64 RNG for deterministic reproducibility.
#[derive(Debug, Clone)]
pub(crate) struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create RNG with seed.
    #[must_use]
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate next u64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

impl Rng for XorShift64 {
    fn gen_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    fn gen_i64_range(&mut self, low: i64, high: i64) -> i64 {
        if low >= high {
            return low;
        }
        let range = (high - low + 1) as u64;
        low + (self.next_u64() % range) as i64
    }

    fn gen_usize(&mut self, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        (self.next_u64() as usize) % len
    }
}

/// Random search optimizer.
///
/// Random search is surprisingly effective, achieving equivalent results to
/// grid search with 60x fewer trials in many cases (see reference below).
///
/// # References
///
/// Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization. JMLR.
///
/// # Example
///
/// ```
/// use aprender::automl::{RandomSearch, SearchSpace, SearchStrategy};
/// use aprender::automl::params::RandomForestParam as RF;
///
/// let space = SearchSpace::new()
///     .add(RF::NEstimators, 10..500)
///     .add(RF::MaxDepth, 2..20);
///
/// let mut search = RandomSearch::new(50).with_seed(42);
/// let trials = search.suggest(&space, 10);
///
/// assert_eq!(trials.len(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct RandomSearch {
    /// Total number of trials to run.
    pub n_iter: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Internal RNG state.
    rng: XorShift64,
    /// Trials generated so far.
    trials_generated: usize,
}

impl RandomSearch {
    /// Create random search with n iterations.
    #[must_use]
    pub fn new(n_iter: usize) -> Self {
        Self {
            n_iter,
            seed: 42,
            rng: XorShift64::new(42),
            trials_generated: 0,
        }
    }

    /// Set random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = XorShift64::new(seed);
        self
    }

    /// Remaining trials to generate.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.n_iter.saturating_sub(self.trials_generated)
    }
}

impl<P: ParamKey> SearchStrategy<P> for RandomSearch {
    fn suggest(&mut self, space: &SearchSpace<P>, n: usize) -> Vec<Trial<P>> {
        let n = n.min(self.remaining());
        let trials: Vec<Trial<P>> = (0..n).map(|_| space.sample(&mut self.rng)).collect();
        self.trials_generated += trials.len();
        trials
    }
}

/// Grid search optimizer.
///
/// Exhaustively searches all combinations of parameter values.
/// Best for small, discrete search spaces.
///
/// # Example
///
/// ```
/// use aprender::automl::{GridSearch, SearchSpace, SearchStrategy};
/// use aprender::automl::params::RandomForestParam as RF;
///
/// let space = SearchSpace::new()
///     .add(RF::NEstimators, 10..100)
///     .add(RF::MaxDepth, 2..10);
///
/// let mut search = GridSearch::new(5); // 5 points per continuous param
/// let trials = search.suggest(&space, 100);
///
/// // Grid of 5x5 = 25 configurations
/// assert!(trials.len() <= 25);
/// ```
#[derive(Debug, Clone)]
pub struct GridSearch {
    /// Number of grid points per continuous parameter.
    pub points_per_param: usize,
    /// Current position in grid.
    position: usize,
}

impl GridSearch {
    /// Create grid search with specified granularity.
    #[must_use]
    pub fn new(points_per_param: usize) -> Self {
        Self {
            points_per_param: points_per_param.max(2),
            position: 0,
        }
    }
}

impl<P: ParamKey> SearchStrategy<P> for GridSearch {
    fn suggest(&mut self, space: &SearchSpace<P>, n: usize) -> Vec<Trial<P>> {
        // Generate full grid on first call
        let grid = space.grid(self.points_per_param);
        let remaining = grid.len().saturating_sub(self.position);
        let n = n.min(remaining);

        let trials = grid[self.position..self.position + n].to_vec();
        self.position += n;
        trials
    }
}

/// Differential Evolution search optimizer.
///
/// Population-based evolutionary algorithm that adapts mutation and crossover
/// parameters during optimization. Excellent for continuous hyperparameter spaces.
///
/// # Advantages over Random Search
///
/// - **Adaptive**: Learns from successful mutations
/// - **Population-based**: Maintains diverse candidates
/// - **Exploitation**: Focuses on promising regions
///
/// # Example
///
/// ```
/// use aprender::automl::{DESearch, SearchSpace, SearchStrategy};
/// use aprender::automl::params::RandomForestParam as RF;
///
/// let space = SearchSpace::new()
///     .add_continuous(RF::NEstimators, 10.0, 500.0)
///     .add_continuous(RF::MaxDepth, 2.0, 20.0);
///
/// let mut search = DESearch::new(50).with_seed(42);
/// let trials = search.suggest(&space, 20);
///
/// assert_eq!(trials.len(), 20);
/// ```
#[derive(Debug, Clone)]
pub struct DESearch {
    /// Total number of trials to run.
    pub n_iter: usize,
    /// Population size (0 = auto).
    pub population_size: usize,
    /// Random seed.
    pub seed: u64,
    /// DE strategy.
    pub strategy: DEStrategy,
    /// Use JADE adaptation.
    pub use_jade: bool,
    /// Internal population (flattened parameter vectors).
    population: Vec<Vec<f64>>,
    /// Fitness values for population.
    fitness: Vec<f64>,
    /// Best individual index.
    best_idx: usize,
    /// Parameter keys in order (for conversion).
    param_order: Vec<String>,
    /// Parameter bounds (lower, upper, `is_integer`, `is_log`).
    param_bounds: Vec<(f64, f64, bool, bool)>,
    /// Trials generated so far.
    trials_generated: usize,
    /// Whether population is initialized.
    initialized: bool,
    /// Mutation factor F.
    mutation_factor: f64,
    /// Crossover rate CR.
    crossover_rate: f64,
}
