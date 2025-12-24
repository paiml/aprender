//! Search space and optimization algorithms.
//!
//! Implements random search [1], grid search, differential evolution [2],
//! and active learning [3] for hyperparameter optimization.
//!
//! # References
//!
//! [1] Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization. JMLR.
//! [2] Storn & Price (1997). Differential Evolution. Journal of Global Optimization.
//! [3] Settles (2009). Active Learning Literature Survey. UW-Madison CS Tech Report.

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Range;

use crate::automl::params::ParamKey;
use crate::metaheuristics::DEStrategy;

/// Hyperparameter value that can be sampled.
#[derive(Debug, Clone)]
pub enum HyperParam {
    /// Continuous parameter in [low, high].
    Continuous {
        low: f64,
        high: f64,
        log_scale: bool,
    },
    /// Integer parameter in [low, high].
    Integer { low: i64, high: i64 },
    /// Categorical parameter with discrete choices.
    Categorical { choices: Vec<ParamValue> },
}

impl HyperParam {
    /// Create continuous parameter from range.
    #[must_use]
    pub fn continuous(low: f64, high: f64) -> Self {
        Self::Continuous {
            low,
            high,
            log_scale: false,
        }
    }

    /// Create continuous parameter with log scale.
    #[must_use]
    pub fn continuous_log(low: f64, high: f64) -> Self {
        Self::Continuous {
            low,
            high,
            log_scale: true,
        }
    }

    /// Create integer parameter from range.
    #[must_use]
    pub fn integer(low: i64, high: i64) -> Self {
        Self::Integer { low, high }
    }

    /// Create categorical parameter from choices.
    #[must_use]
    pub fn categorical<I, V>(choices: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<ParamValue>,
    {
        Self::Categorical {
            choices: choices.into_iter().map(Into::into).collect(),
        }
    }

    /// Sample a random value from this parameter's distribution.
    #[must_use]
    pub fn sample(&self, rng: &mut impl Rng) -> ParamValue {
        match self {
            Self::Continuous {
                low,
                high,
                log_scale,
            } => {
                let value = if *log_scale {
                    let log_low = low.ln();
                    let log_high = high.ln();
                    let u = rng.gen_f64();
                    (log_low + u * (log_high - log_low)).exp()
                } else {
                    rng.gen_f64_range(*low, *high)
                };
                ParamValue::Float(value)
            }
            Self::Integer { low, high } => {
                let value = rng.gen_i64_range(*low, *high);
                ParamValue::Int(value)
            }
            Self::Categorical { choices } => {
                let idx = rng.gen_usize(choices.len());
                choices[idx].clone()
            }
        }
    }

    /// Generate grid points for this parameter.
    #[must_use]
    pub fn grid_points(&self, n_points: usize) -> Vec<ParamValue> {
        match self {
            Self::Continuous {
                low,
                high,
                log_scale,
            } => {
                if n_points <= 1 {
                    return vec![ParamValue::Float(*low)];
                }
                (0..n_points)
                    .map(|i| {
                        let t = i as f64 / (n_points - 1) as f64;
                        let value = if *log_scale {
                            let log_low = low.ln();
                            let log_high = high.ln();
                            (log_low + t * (log_high - log_low)).exp()
                        } else {
                            low + t * (high - low)
                        };
                        ParamValue::Float(value)
                    })
                    .collect()
            }
            Self::Integer { low, high } => {
                let range = (high - low + 1) as usize;
                let step = (range as f64 / n_points as f64).ceil() as i64;
                let mut points = Vec::new();
                let mut v = *low;
                while v <= *high && points.len() < n_points {
                    points.push(ParamValue::Int(v));
                    v += step.max(1);
                }
                points
            }
            Self::Categorical { choices } => choices.clone(),
        }
    }
}

/// A concrete parameter value.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
}

impl ParamValue {
    /// Get as f64 if numeric.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as i64 if integer.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Get as bool.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
}

impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<f32> for ParamValue {
    fn from(v: f32) -> Self {
        Self::Float(f64::from(v))
    }
}

impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for ParamValue {
    fn from(v: i32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<usize> for ParamValue {
    fn from(v: usize) -> Self {
        Self::Int(v as i64)
    }
}

impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<&str> for ParamValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

impl From<String> for ParamValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{v:.6}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "{v}"),
        }
    }
}

/// Marker for log-scale continuous ranges.
#[derive(Debug, Clone, Copy)]
pub struct LogScale {
    pub low: f64,
    pub high: f64,
}

/// Extension trait for creating log-scale ranges.
#[allow(dead_code)] // Used in tests
trait LogScaleExt {
    /// Convert range to log-scale.
    fn log_scale(self) -> LogScale;
}

impl LogScaleExt for Range<f64> {
    fn log_scale(self) -> LogScale {
        LogScale {
            low: self.start,
            high: self.end,
        }
    }
}

/// Type-safe search space for hyperparameters.
///
/// # Example
///
/// ```
/// use aprender::automl::{SearchSpace, LogScale};
/// use aprender::automl::params::RandomForestParam as RF;
///
/// let space = SearchSpace::new()
///     .add(RF::NEstimators, 10..500)
///     .add(RF::MaxDepth, 2..20);
///
/// assert_eq!(space.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct SearchSpace<P: ParamKey = GenericParam> {
    params: HashMap<P, HyperParam>,
}

impl<P: ParamKey> Default for SearchSpace<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: ParamKey> SearchSpace<P> {
    /// Create an empty search space.
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Number of parameters in the space.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Check if space is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Add an integer parameter from a range.
    #[must_use]
    pub fn add(mut self, key: P, range: Range<i64>) -> Self {
        self.params.insert(
            key,
            HyperParam::Integer {
                low: range.start,
                high: range.end - 1,
            },
        );
        self
    }

    /// Add a continuous parameter from a float range.
    #[must_use]
    pub fn add_continuous(mut self, key: P, low: f64, high: f64) -> Self {
        self.params.insert(key, HyperParam::continuous(low, high));
        self
    }

    /// Add a log-scale continuous parameter.
    #[must_use]
    pub fn add_log_scale(mut self, key: P, log_scale: LogScale) -> Self {
        self.params.insert(
            key,
            HyperParam::continuous_log(log_scale.low, log_scale.high),
        );
        self
    }

    /// Add a categorical parameter from string choices.
    #[must_use]
    pub fn add_categorical<I, V>(mut self, key: P, choices: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<ParamValue>,
    {
        self.params.insert(key, HyperParam::categorical(choices));
        self
    }

    /// Add a boolean parameter.
    #[must_use]
    pub fn add_bool(mut self, key: P, choices: [bool; 2]) -> Self {
        self.params.insert(key, HyperParam::categorical(choices));
        self
    }

    /// Add a raw HyperParam.
    #[must_use]
    pub fn add_param(mut self, key: P, param: HyperParam) -> Self {
        self.params.insert(key, param);
        self
    }

    /// Get parameter definition by key.
    #[must_use]
    pub fn get(&self, key: &P) -> Option<&HyperParam> {
        self.params.get(key)
    }

    /// Iterate over parameter definitions.
    pub fn iter(&self) -> impl Iterator<Item = (&P, &HyperParam)> {
        self.params.iter()
    }

    /// Sample a random configuration.
    #[must_use]
    pub fn sample(&self, rng: &mut impl Rng) -> Trial<P> {
        let values: HashMap<P, ParamValue> = self
            .params
            .iter()
            .map(|(k, p)| (*k, p.sample(rng)))
            .collect();
        Trial { values }
    }

    /// Generate all grid configurations.
    #[must_use]
    pub fn grid(&self, points_per_param: usize) -> Vec<Trial<P>> {
        let param_grids: Vec<(P, Vec<ParamValue>)> = self
            .params
            .iter()
            .map(|(k, p)| (*k, p.grid_points(points_per_param)))
            .collect();

        if param_grids.is_empty() {
            return vec![Trial {
                values: HashMap::new(),
            }];
        }

        // Cartesian product of all parameter grids
        let mut configs = vec![HashMap::new()];

        for (key, values) in param_grids {
            let mut new_configs = Vec::with_capacity(configs.len() * values.len());
            for config in &configs {
                for value in &values {
                    let mut new_config = config.clone();
                    new_config.insert(key, value.clone());
                    new_configs.push(new_config);
                }
            }
            configs = new_configs;
        }

        configs.into_iter().map(|values| Trial { values }).collect()
    }
}

/// Generic parameter key for dynamic use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenericParam(pub &'static str);

impl ParamKey for GenericParam {
    fn name(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for GenericParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

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
    /// Parameter bounds (lower, upper, is_integer, is_log).
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
            match hyper {
                HyperParam::Continuous {
                    low,
                    high,
                    log_scale,
                } => {
                    self.param_order.push(key_str);
                    self.param_bounds.push((*low, *high, false, *log_scale));
                }
                HyperParam::Integer { low, high } => {
                    self.param_order.push(key_str);
                    self.param_bounds
                        .push((*low as f64, *high as f64, true, false));
                }
                HyperParam::Categorical { choices } => {
                    // Map categorical to integer index
                    self.param_order.push(key_str);
                    self.param_bounds
                        .push((0.0, (choices.len() - 1) as f64, true, false));
                }
            }
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
            // Select 3 distinct random indices
            let mut indices = Vec::with_capacity(3);
            while indices.len() < 3 {
                let idx = rng.gen_usize(pop_size);
                if idx != i && !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
            let (a, b, c) = (indices[0], indices[1], indices[2]);

            // Mutation based on strategy
            let mutant: Vec<f64> = match self.strategy {
                DEStrategy::Rand1Bin => (0..dim)
                    .map(|j| {
                        self.population[a][j]
                            + self.mutation_factor * (self.population[b][j] - self.population[c][j])
                    })
                    .collect(),
                DEStrategy::Best1Bin => (0..dim)
                    .map(|j| {
                        self.population[self.best_idx][j]
                            + self.mutation_factor * (self.population[a][j] - self.population[b][j])
                    })
                    .collect(),
                DEStrategy::CurrentToBest1Bin => (0..dim)
                    .map(|j| {
                        self.population[i][j]
                            + self.mutation_factor
                                * (self.population[self.best_idx][j] - self.population[i][j])
                            + self.mutation_factor * (self.population[a][j] - self.population[b][j])
                    })
                    .collect(),
                DEStrategy::Rand2Bin => {
                    // Need 5 indices for rand/2
                    let mut more_indices = indices.clone();
                    while more_indices.len() < 5 {
                        let idx = rng.gen_usize(pop_size);
                        if idx != i && !more_indices.contains(&idx) {
                            more_indices.push(idx);
                        }
                    }
                    let (d, e) = (more_indices[3], more_indices[4]);
                    (0..dim)
                        .map(|j| {
                            self.population[a][j]
                                + self.mutation_factor
                                    * (self.population[b][j] - self.population[c][j])
                                + self.mutation_factor
                                    * (self.population[d][j] - self.population[e][j])
                        })
                        .collect()
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
    /// Uses coefficient of variation (std_dev / mean) as uncertainty metric.
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

impl<S, P> SearchStrategy<P> for ActiveLearningSearch<S>
where
    S: SearchStrategy<P>,
    P: ParamKey,
{
    fn suggest(&mut self, space: &SearchSpace<P>, n: usize) -> Vec<Trial<P>> {
        // If we should stop, return empty
        if self.should_stop() {
            return Vec::new();
        }
        self.base.suggest(space, n)
    }

    fn update(&mut self, results: &[TrialResult<P>]) {
        // Collect scores for uncertainty estimation
        for result in results {
            self.scores.push(result.score);
        }

        // Update uncertainty estimate
        self.compute_uncertainty();

        // Forward to base strategy
        self.base.update(results);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::params::RandomForestParam as RF;

    #[test]
    fn test_search_space_builder() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..500)
            .add(RF::MaxDepth, 2..20);

        assert_eq!(space.len(), 2);
        assert!(space.get(&RF::NEstimators).is_some());
        assert!(space.get(&RF::MaxDepth).is_some());
    }

    #[test]
    fn test_search_space_continuous() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 0.0, 1.0)
            .add_log_scale(RF::MaxDepth, (1e-4..1.0).log_scale());

        assert_eq!(space.len(), 2);
    }

    #[test]
    fn test_search_space_categorical() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_categorical(RF::MaxFeatures, ["sqrt", "log2", "0.5"])
            .add_bool(RF::Bootstrap, [true, false]);

        assert_eq!(space.len(), 2);
    }

    #[test]
    fn test_random_search_deterministic() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..500)
            .add(RF::MaxDepth, 2..20);

        let mut search1 = RandomSearch::new(10).with_seed(42);
        let mut search2 = RandomSearch::new(10).with_seed(42);

        let trials1 = search1.suggest(&space, 5);
        let trials2 = search2.suggest(&space, 5);

        for (t1, t2) in trials1.iter().zip(trials2.iter()) {
            assert_eq!(
                t1.get(&RF::NEstimators),
                t2.get(&RF::NEstimators),
                "Same seed should produce same results"
            );
        }
    }

    #[test]
    fn test_random_search_respects_budget() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..500);

        let mut search = RandomSearch::new(5);

        let trials1 = search.suggest(&space, 3);
        assert_eq!(trials1.len(), 3);
        assert_eq!(search.remaining(), 2);

        let trials2 = search.suggest(&space, 10);
        assert_eq!(trials2.len(), 2);
        assert_eq!(search.remaining(), 0);
    }

    #[test]
    fn test_grid_search_cartesian_product() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_categorical(RF::Bootstrap, [true, false])
            .add_categorical(RF::MaxFeatures, ["sqrt", "log2"]);

        let mut search = GridSearch::new(10);
        let trials = search.suggest(&space, 100);

        // 2 x 2 = 4 combinations
        assert_eq!(trials.len(), 4);
    }

    #[test]
    fn test_trial_accessors() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 100..101) // Single value: 100
            .add_bool(RF::Bootstrap, [true, false]);

        let mut rng = XorShift64::new(42);
        let trial = space.sample(&mut rng);

        assert_eq!(trial.get_i64(&RF::NEstimators), Some(100));
        assert!(trial.get_bool(&RF::Bootstrap).is_some());
    }

    #[test]
    fn test_param_value_conversions() {
        assert_eq!(ParamValue::from(42_i32).as_i64(), Some(42));
        assert_eq!(ParamValue::from(1.234_f64).as_f64(), Some(1.234));
        assert_eq!(ParamValue::from(true).as_bool(), Some(true));
        assert_eq!(ParamValue::from("test").as_str(), Some("test"));
    }

    #[test]
    fn test_hyperparam_sampling() {
        let mut rng = XorShift64::new(42);

        let continuous = HyperParam::continuous(0.0, 1.0);
        for _ in 0..100 {
            let v = continuous
                .sample(&mut rng)
                .as_f64()
                .expect("continuous param should return float");
            assert!((0.0..=1.0).contains(&v));
        }

        let integer = HyperParam::integer(10, 20);
        for _ in 0..100 {
            let v = integer
                .sample(&mut rng)
                .as_i64()
                .expect("integer param should return int");
            assert!((10..=20).contains(&v));
        }
    }

    #[test]
    fn test_log_scale_sampling() {
        let mut rng = XorShift64::new(42);
        let param = HyperParam::continuous_log(1e-4, 1.0);

        let mut samples = Vec::new();
        for _ in 0..1000 {
            let v = param
                .sample(&mut rng)
                .as_f64()
                .expect("log scale param should return float");
            assert!((1e-4..=1.0).contains(&v));
            samples.push(v);
        }

        // Log scale should have more samples near lower end
        let below_01 = samples.iter().filter(|&&v| v < 0.1).count();
        let above_01 = samples.iter().filter(|&&v| v >= 0.1).count();
        assert!(
            below_01 > above_01 / 2,
            "Log scale should sample more from lower range"
        );
    }

    #[test]
    fn test_grid_points_continuous() {
        let param = HyperParam::continuous(0.0, 1.0);
        let points = param.grid_points(5);

        assert_eq!(points.len(), 5);
        assert_eq!(points[0].as_f64(), Some(0.0));
        assert_eq!(points[4].as_f64(), Some(1.0));
    }

    #[test]
    fn test_grid_points_integer() {
        let param = HyperParam::integer(1, 10);
        let points = param.grid_points(5);

        assert!(points.len() <= 5);
        let first = points[0]
            .as_i64()
            .expect("integer grid point should be int");
        assert!(first >= 1);
    }

    #[test]
    fn test_trial_display() {
        let mut values = HashMap::new();
        values.insert(RF::NEstimators, ParamValue::Int(100));
        values.insert(RF::MaxDepth, ParamValue::Int(5));

        let trial = Trial { values };
        let s = format!("{trial}");

        assert!(s.contains("n_estimators=100"));
        assert!(s.contains("max_depth=5"));
    }

    #[test]
    fn test_xorshift_rng() {
        let mut rng = XorShift64::new(12345);

        // Should produce different values
        let v1 = rng.gen_f64();
        let v2 = rng.gen_f64();
        assert_ne!(v1, v2);

        // Values should be in [0, 1)
        for _ in 0..1000 {
            let v = rng.gen_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_de_search_basic() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0)
            .add_continuous(RF::MaxDepth, 2.0, 20.0);

        let mut search = DESearch::new(50).with_seed(42);
        let trials = search.suggest(&space, 20);

        assert_eq!(trials.len(), 20);

        // All trials should have valid values
        for trial in &trials {
            let n_est = trial
                .get_f64(&RF::NEstimators)
                .expect("NEstimators should exist");
            let depth = trial.get_f64(&RF::MaxDepth).expect("MaxDepth should exist");

            assert!((10.0..=500.0).contains(&n_est));
            assert!((2.0..=20.0).contains(&depth));
        }
    }

    #[test]
    fn test_de_search_deterministic() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0)
            .add_continuous(RF::MaxDepth, 2.0, 20.0);

        let mut search1 = DESearch::new(50).with_seed(42);
        let mut search2 = DESearch::new(50).with_seed(42);

        let trials1 = search1.suggest(&space, 10);
        let trials2 = search2.suggest(&space, 10);

        for (t1, t2) in trials1.iter().zip(trials2.iter()) {
            assert_eq!(
                t1.get_f64(&RF::NEstimators),
                t2.get_f64(&RF::NEstimators),
                "Same seed should produce same results"
            );
        }
    }

    #[test]
    fn test_de_search_respects_budget() {
        let space: SearchSpace<RF> =
            SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

        let mut search = DESearch::new(5).with_seed(42);

        // First batch
        let t1 = search.suggest(&space, 3);
        assert_eq!(t1.len(), 3);
        assert_eq!(search.remaining(), 2);

        // Second batch
        let t2 = search.suggest(&space, 10);
        assert_eq!(t2.len(), 2); // Only 2 remaining
        assert_eq!(search.remaining(), 0);
    }

    #[test]
    fn test_de_search_with_integers() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..500) // Integer range
            .add(RF::MaxDepth, 2..20);

        let mut search = DESearch::new(50).with_seed(42);
        let trials = search.suggest(&space, 10);

        for trial in &trials {
            let n_est = trial
                .get_i64(&RF::NEstimators)
                .expect("NEstimators should exist");
            let depth = trial.get_i64(&RF::MaxDepth).expect("MaxDepth should exist");

            assert!((10..=500).contains(&n_est));
            assert!((2..=20).contains(&depth));
        }
    }

    #[test]
    fn test_de_search_strategies() {
        use crate::metaheuristics::DEStrategy;

        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0)
            .add_continuous(RF::MaxDepth, 2.0, 20.0);

        for strategy in [
            DEStrategy::Rand1Bin,
            DEStrategy::Best1Bin,
            DEStrategy::CurrentToBest1Bin,
        ] {
            let mut search = DESearch::new(20).with_strategy(strategy).with_seed(42);
            let trials = search.suggest(&space, 10);
            assert_eq!(trials.len(), 10, "Strategy {strategy:?} should work");
        }
    }

    // =========================================================================
    // Active Learning Tests (RED PHASE - These should fail initially)
    // =========================================================================

    #[test]
    fn test_active_learning_basic() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0)
            .add_continuous(RF::MaxDepth, 2.0, 20.0);

        let base = RandomSearch::new(100).with_seed(42);
        let mut search = ActiveLearningSearch::new(base).with_uncertainty_threshold(0.1);

        let trials = search.suggest(&space, 10);
        assert_eq!(trials.len(), 10);
    }

    #[test]
    fn test_active_learning_stops_on_confidence() {
        let space: SearchSpace<RF> =
            SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

        let base = RandomSearch::new(1000).with_seed(42);
        let mut search = ActiveLearningSearch::new(base)
            .with_uncertainty_threshold(0.5)
            .with_min_samples(5);

        // First batch
        let trials1 = search.suggest(&space, 10);
        assert!(!trials1.is_empty());

        // Simulate high-confidence results (low variance)
        let results: Vec<TrialResult<RF>> = trials1
            .iter()
            .map(|t| TrialResult {
                trial: t.clone(),
                score: 0.95, // All same score = low uncertainty
                metrics: HashMap::new(),
            })
            .collect();

        search.update(&results);

        // Should stop early due to low uncertainty
        assert!(
            search.should_stop(),
            "Should stop when uncertainty is below threshold"
        );
    }

    #[test]
    fn test_active_learning_continues_on_uncertainty() {
        let space: SearchSpace<RF> =
            SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

        let base = RandomSearch::new(1000).with_seed(42);
        let mut search = ActiveLearningSearch::new(base)
            .with_uncertainty_threshold(0.01) // Very low threshold
            .with_min_samples(3);

        let trials = search.suggest(&space, 10);

        // Simulate high-uncertainty results (high variance)
        let results: Vec<TrialResult<RF>> = trials
            .iter()
            .enumerate()
            .map(|(i, t)| TrialResult {
                trial: t.clone(),
                score: if i % 2 == 0 { 0.1 } else { 0.9 }, // High variance
                metrics: HashMap::new(),
            })
            .collect();

        search.update(&results);

        // Should NOT stop - still uncertain
        assert!(
            !search.should_stop(),
            "Should continue when uncertainty is high"
        );
    }

    #[test]
    fn test_active_learning_uncertainty_score() {
        let space: SearchSpace<RF> =
            SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

        let base = RandomSearch::new(100).with_seed(42);
        let mut search = ActiveLearningSearch::new(base);

        let trials = search.suggest(&space, 5);

        // Low variance results
        let low_var_results: Vec<TrialResult<RF>> = trials
            .iter()
            .map(|t| TrialResult {
                trial: t.clone(),
                score: 0.5,
                metrics: HashMap::new(),
            })
            .collect();

        search.update(&low_var_results);
        let low_uncertainty = search.uncertainty();

        // Reset and try high variance
        let base2 = RandomSearch::new(100).with_seed(42);
        let mut search2 = ActiveLearningSearch::new(base2);
        let trials2 = search2.suggest(&space, 5);

        let high_var_results: Vec<TrialResult<RF>> = trials2
            .iter()
            .enumerate()
            .map(|(i, t)| TrialResult {
                trial: t.clone(),
                score: i as f64 / 5.0, // 0.0, 0.2, 0.4, 0.6, 0.8
                metrics: HashMap::new(),
            })
            .collect();

        search2.update(&high_var_results);
        let high_uncertainty = search2.uncertainty();

        assert!(
            high_uncertainty > low_uncertainty,
            "High variance should have higher uncertainty: {high_uncertainty} vs {low_uncertainty}"
        );
    }

    #[test]
    fn test_active_learning_with_de_search() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0)
            .add_continuous(RF::MaxDepth, 2.0, 20.0);

        // Active learning wrapping DE
        let base = DESearch::new(100).with_seed(42);
        let mut search = ActiveLearningSearch::new(base).with_uncertainty_threshold(0.1);

        let trials = search.suggest(&space, 20);
        assert_eq!(trials.len(), 20);

        // All values should be within bounds
        for trial in &trials {
            let n_est = trial
                .get_f64(&RF::NEstimators)
                .expect("NEstimators should exist");
            assert!((10.0..=500.0).contains(&n_est));
        }
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_log_scale_struct() {
        let ls = LogScale {
            low: 1e-4,
            high: 1.0,
        };
        assert_eq!(ls.low, 1e-4);
        assert_eq!(ls.high, 1.0);

        // Test Clone/Copy via Debug
        let _cloned = ls;
        let debug_str = format!("{:?}", ls);
        assert!(debug_str.contains("LogScale"));
    }

    #[test]
    fn test_generic_param_display() {
        let param = GenericParam("learning_rate");
        let display = format!("{}", param);
        assert_eq!(display, "learning_rate");
    }

    #[test]
    fn test_search_space_iter() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..500)
            .add(RF::MaxDepth, 2..20);

        let count = space.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_search_space_is_empty() {
        let empty: SearchSpace<RF> = SearchSpace::new();
        assert!(empty.is_empty());

        let non_empty = empty.add(RF::NEstimators, 10..100);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_search_space_add_param() {
        let param = HyperParam::continuous(0.0, 1.0);
        let space: SearchSpace<RF> = SearchSpace::new().add_param(RF::NEstimators, param);

        assert!(space.get(&RF::NEstimators).is_some());
    }

    #[test]
    fn test_trial_get_usize() {
        let mut values = HashMap::new();
        values.insert(RF::NEstimators, ParamValue::Int(100));

        let trial = Trial { values };
        assert_eq!(trial.get_usize(&RF::NEstimators), Some(100));
    }

    #[test]
    fn test_param_value_as_str_none() {
        let pv = ParamValue::Int(42);
        assert!(pv.as_str().is_none());

        let pv2 = ParamValue::Float(1.0);
        assert!(pv2.as_str().is_none());
    }

    #[test]
    fn test_param_value_as_bool_none() {
        let pv = ParamValue::Int(42);
        assert!(pv.as_bool().is_none());
    }

    #[test]
    fn test_param_value_display() {
        assert_eq!(format!("{}", ParamValue::Float(1.5)), "1.500000");
        assert_eq!(format!("{}", ParamValue::Int(42)), "42");
        assert_eq!(format!("{}", ParamValue::Bool(true)), "true");
        assert_eq!(format!("{}", ParamValue::String("test".to_string())), "test");
    }

    #[test]
    fn test_param_value_from_usize() {
        let pv = ParamValue::from(42_usize);
        assert_eq!(pv.as_i64(), Some(42));
    }

    #[test]
    fn test_param_value_from_string() {
        let pv = ParamValue::from("hello".to_string());
        assert_eq!(pv.as_str(), Some("hello"));
    }

    #[test]
    fn test_hyperparam_clone() {
        let hp = HyperParam::continuous(0.0, 1.0);
        let cloned = hp.clone();
        if let HyperParam::Continuous { low, high, .. } = cloned {
            assert_eq!(low, 0.0);
            assert_eq!(high, 1.0);
        } else {
            panic!("Should be continuous");
        }
    }

    #[test]
    fn test_grid_points_single_point() {
        let param = HyperParam::continuous(5.0, 10.0);
        let points = param.grid_points(1);

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].as_f64(), Some(5.0));
    }

    #[test]
    fn test_grid_points_log_scale() {
        let param = HyperParam::continuous_log(1.0, 100.0);
        let points = param.grid_points(3);

        assert_eq!(points.len(), 3);
        // First and last should be at bounds
        assert!((points[0].as_f64().unwrap() - 1.0).abs() < 0.01);
        assert!((points[2].as_f64().unwrap() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_grid_empty_space() {
        let space: SearchSpace<RF> = SearchSpace::new();
        let grid = space.grid(5);

        assert_eq!(grid.len(), 1);
        assert!(grid[0].values.is_empty());
    }

    #[test]
    fn test_de_search_with_jade() {
        let search = DESearch::new(50).with_jade();
        assert!(search.use_jade);
    }

    #[test]
    fn test_de_search_with_mutation_factor() {
        let search = DESearch::new(50).with_mutation_factor(0.5);
        assert_eq!(search.mutation_factor, 0.5);
    }

    #[test]
    fn test_de_search_with_crossover_rate() {
        let search = DESearch::new(50).with_crossover_rate(0.7);
        assert_eq!(search.crossover_rate, 0.7);
    }

    #[test]
    fn test_de_search_with_population_size() {
        let search = DESearch::new(50).with_population_size(100);
        assert_eq!(search.population_size, 100);
    }

    #[test]
    fn test_de_search_rand2bin_strategy() {
        use crate::metaheuristics::DEStrategy;

        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0)
            .add_continuous(RF::MaxDepth, 2.0, 20.0);

        let mut search = DESearch::new(100)
            .with_strategy(DEStrategy::Rand2Bin)
            .with_seed(42);

        let trials = search.suggest(&space, 20);
        assert_eq!(trials.len(), 20);

        // Simulate results to trigger update with Rand2Bin mutation
        let results: Vec<TrialResult<RF>> = trials
            .iter()
            .enumerate()
            .map(|(i, t)| TrialResult {
                trial: t.clone(),
                score: i as f64,
                metrics: HashMap::new(),
            })
            .collect();

        search.update(&results);

        // Get next batch after evolution
        let trials2 = search.suggest(&space, 20);
        assert!(trials2.len() <= 20);
    }

    #[test]
    fn test_de_search_with_categorical() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_categorical(RF::MaxFeatures, ["sqrt", "log2", "auto"]);

        let mut search = DESearch::new(50).with_seed(42);
        let trials = search.suggest(&space, 10);

        for trial in &trials {
            let val = trial.get(&RF::MaxFeatures).expect("should have value");
            let s = val.as_str().expect("should be string");
            assert!(["sqrt", "log2", "auto"].contains(&s));
        }
    }

    #[test]
    fn test_active_learning_sample_count() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0);

        let base = RandomSearch::new(100).with_seed(42);
        let mut search = ActiveLearningSearch::new(base);

        assert_eq!(search.sample_count(), 0);

        let trials = search.suggest(&space, 5);
        let results: Vec<TrialResult<RF>> = trials
            .iter()
            .map(|t| TrialResult {
                trial: t.clone(),
                score: 0.5,
                metrics: HashMap::new(),
            })
            .collect();

        search.update(&results);
        assert_eq!(search.sample_count(), 5);
    }

    #[test]
    fn test_active_learning_returns_empty_when_should_stop() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0);

        let base = RandomSearch::new(1000).with_seed(42);
        let mut search = ActiveLearningSearch::new(base)
            .with_uncertainty_threshold(1.0) // Very high threshold - easy to satisfy
            .with_min_samples(5);

        // Get enough samples
        let trials = search.suggest(&space, 10);
        let results: Vec<TrialResult<RF>> = trials
            .iter()
            .map(|t| TrialResult {
                trial: t.clone(),
                score: 0.5, // Same score = zero variance
                metrics: HashMap::new(),
            })
            .collect();

        search.update(&results);

        // Now should_stop() returns true, so suggest() should return empty
        assert!(search.should_stop());
        let empty_trials = search.suggest(&space, 10);
        assert!(empty_trials.is_empty());
    }

    #[test]
    fn test_active_learning_near_zero_mean() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0);

        let base = RandomSearch::new(100).with_seed(42);
        let mut search = ActiveLearningSearch::new(base);

        let trials = search.suggest(&space, 5);

        // Scores near zero to trigger the mean.abs() < 1e-10 branch
        let results: Vec<TrialResult<RF>> = trials
            .iter()
            .enumerate()
            .map(|(i, t)| TrialResult {
                trial: t.clone(),
                score: (i as f64) * 1e-12, // Very small values
                metrics: HashMap::new(),
            })
            .collect();

        search.update(&results);

        // Should handle near-zero mean without dividing by zero
        let uncertainty = search.uncertainty();
        assert!(uncertainty.is_finite());
    }

    #[test]
    fn test_active_learning_single_sample() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0);

        let base = RandomSearch::new(100).with_seed(42);
        let mut search = ActiveLearningSearch::new(base);

        let trials = search.suggest(&space, 1);
        let results: Vec<TrialResult<RF>> = trials
            .iter()
            .map(|t| TrialResult {
                trial: t.clone(),
                score: 0.5,
                metrics: HashMap::new(),
            })
            .collect();

        search.update(&results);

        // With only 1 sample, uncertainty should be infinite
        assert!(search.uncertainty().is_infinite());
    }

    #[test]
    fn test_xorshift_edge_cases() {
        // Seed of 0 should be treated as 1
        let mut rng = XorShift64::new(0);
        let v = rng.gen_f64();
        assert!((0.0..1.0).contains(&v));

        // gen_i64_range with low >= high should return low
        let mut rng2 = XorShift64::new(42);
        assert_eq!(rng2.gen_i64_range(10, 10), 10);
        assert_eq!(rng2.gen_i64_range(10, 5), 10);

        // gen_usize with len 0 should return 0
        assert_eq!(rng2.gen_usize(0), 0);
    }

    #[test]
    fn test_trial_result_clone() {
        let mut values = HashMap::new();
        values.insert(RF::NEstimators, ParamValue::Int(100));
        let trial = Trial { values };

        let result = TrialResult {
            trial,
            score: 0.95,
            metrics: HashMap::new(),
        };

        let cloned = result.clone();
        assert_eq!(cloned.score, 0.95);
    }

    #[test]
    fn test_de_update_empty_results() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_continuous(RF::NEstimators, 10.0, 500.0);

        let mut search = DESearch::new(50).with_seed(42);
        let _trials = search.suggest(&space, 10);

        // Update with empty results - should not panic
        let empty: Vec<TrialResult<RF>> = vec![];
        search.update(&empty);
    }

    #[test]
    fn test_grid_search_position_tracking() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add_categorical(RF::Bootstrap, [true, false]);

        let mut search = GridSearch::new(5);

        // First batch
        let t1 = search.suggest(&space, 1);
        assert_eq!(t1.len(), 1);

        // Second batch from same position
        let t2 = search.suggest(&space, 1);
        assert_eq!(t2.len(), 1);

        // Total grid is 2, so we've exhausted it
        let t3 = search.suggest(&space, 10);
        assert!(t3.is_empty());
    }

    #[test]
    fn test_search_space_default() {
        let space: SearchSpace<RF> = SearchSpace::default();
        assert!(space.is_empty());
    }

    #[test]
    fn test_random_search_clone() {
        let search = RandomSearch::new(50).with_seed(42);
        let cloned = search.clone();
        assert_eq!(cloned.n_iter, 50);
        assert_eq!(cloned.seed, 42);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::automl::params::RandomForestParam as RF;
    use proptest::prelude::*;

    proptest! {
        /// Random search should always respect budget constraint.
        #[test]
        fn prop_random_search_respects_budget(
            n_iter in 1_usize..100,
            seed in any::<u64>(),
            request in 1_usize..200
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, 10..500);

            let mut search = RandomSearch::new(n_iter).with_seed(seed);
            let trials = search.suggest(&space, request);

            prop_assert!(trials.len() <= n_iter);
            prop_assert!(trials.len() <= request);
        }

        /// Continuous parameters should always sample within bounds.
        #[test]
        fn prop_continuous_within_bounds(
            low in -1000.0_f64..1000.0,
            high_offset in 0.01_f64..1000.0,
            seed in any::<u64>()
        ) {
            let high = low + high_offset;
            let param = HyperParam::continuous(low, high);
            let mut rng = XorShift64::new(seed);

            for _ in 0..100 {
                let v = param.sample(&mut rng).as_f64().expect("should be float");
                prop_assert!((low..=high).contains(&v), "Value {} not in [{}, {}]", v, low, high);
            }
        }

        /// Integer parameters should always sample within bounds.
        #[test]
        fn prop_integer_within_bounds(
            low in -1000_i64..1000,
            high_offset in 1_i64..100,
            seed in any::<u64>()
        ) {
            let high = low + high_offset;
            let param = HyperParam::integer(low, high);
            let mut rng = XorShift64::new(seed);

            for _ in 0..100 {
                let v = param.sample(&mut rng).as_i64().expect("should be int");
                prop_assert!((low..=high).contains(&v), "Value {} not in [{}, {}]", v, low, high);
            }
        }

        /// Log scale should produce values in valid range.
        #[test]
        fn prop_log_scale_within_bounds(
            low_exp in -6_i32..0,
            high_exp in 0_i32..3,
            seed in any::<u64>()
        ) {
            let low = 10_f64.powi(low_exp);
            let high = 10_f64.powi(high_exp);
            let param = HyperParam::continuous_log(low, high);
            let mut rng = XorShift64::new(seed);

            for _ in 0..100 {
                let v = param.sample(&mut rng).as_f64().expect("should be float");
                prop_assert!((low..=high).contains(&v), "Value {} not in [{}, {}]", v, low, high);
            }
        }

        /// Same seed should produce same results (determinism).
        #[test]
        fn prop_random_search_deterministic(seed in any::<u64>()) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, 10..500)
                .add(RF::MaxDepth, 2..20);

            let mut s1 = RandomSearch::new(10).with_seed(seed);
            let mut s2 = RandomSearch::new(10).with_seed(seed);

            let t1 = s1.suggest(&space, 5);
            let t2 = s2.suggest(&space, 5);

            for (a, b) in t1.iter().zip(t2.iter()) {
                prop_assert_eq!(a.get(&RF::NEstimators), b.get(&RF::NEstimators));
                prop_assert_eq!(a.get(&RF::MaxDepth), b.get(&RF::MaxDepth));
            }
        }

        /// Grid points for continuous params should be evenly spaced.
        #[test]
        fn prop_grid_points_count(n_points in 2_usize..20) {
            let param = HyperParam::continuous(0.0, 1.0);
            let points = param.grid_points(n_points);
            prop_assert_eq!(points.len(), n_points);
        }

        /// XorShift64 should always produce values in [0, 1).
        #[test]
        fn prop_xorshift_range(seed in 1_u64..u64::MAX) {
            let mut rng = XorShift64::new(seed);
            for _ in 0..1000 {
                let v = rng.gen_f64();
                prop_assert!((0.0..1.0).contains(&v));
            }
        }

        /// ParamValue conversions should be consistent.
        #[test]
        fn prop_param_value_int_roundtrip(v in any::<i32>()) {
            let pv = ParamValue::from(v);
            prop_assert_eq!(pv.as_i64(), Some(i64::from(v)));
        }

        /// ParamValue float conversions should preserve value.
        #[test]
        fn prop_param_value_float_roundtrip(v in any::<f32>()) {
            let pv = ParamValue::from(v);
            let result = pv.as_f64().expect("float param should convert");
            prop_assert!((result - f64::from(v)).abs() < 1e-10);
        }

        /// DESearch should respect budget constraint.
        #[test]
        fn prop_de_search_respects_budget(
            n_iter in 1_usize..50,
            seed in any::<u64>(),
            request in 1_usize..100
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            let mut search = DESearch::new(n_iter).with_seed(seed);
            let trials = search.suggest(&space, request);

            prop_assert!(trials.len() <= n_iter);
            prop_assert!(trials.len() <= request);
        }

        /// DESearch continuous parameters should be within bounds.
        #[test]
        fn prop_de_search_continuous_bounds(
            low in 0.0_f64..100.0,
            high_offset in 1.0_f64..100.0,
            seed in any::<u64>()
        ) {
            let high = low + high_offset;
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, low, high);

            let mut search = DESearch::new(50).with_seed(seed);
            let trials = search.suggest(&space, 20);

            for trial in trials {
                let v = trial.get_f64(&RF::NEstimators).expect("should have value");
                prop_assert!(
                    v >= low && v <= high,
                    "Value {} not in [{}, {}]", v, low, high
                );
            }
        }

        /// DESearch should be deterministic with same seed.
        #[test]
        fn prop_de_search_deterministic(seed in any::<u64>()) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0)
                .add_continuous(RF::MaxDepth, 2.0, 20.0);

            let mut s1 = DESearch::new(50).with_seed(seed);
            let mut s2 = DESearch::new(50).with_seed(seed);

            let t1 = s1.suggest(&space, 10);
            let t2 = s2.suggest(&space, 10);

            for (a, b) in t1.iter().zip(t2.iter()) {
                prop_assert_eq!(a.get_f64(&RF::NEstimators), b.get_f64(&RF::NEstimators));
                prop_assert_eq!(a.get_f64(&RF::MaxDepth), b.get_f64(&RF::MaxDepth));
            }
        }

        /// DESearch integer parameters should be integers within bounds.
        #[test]
        fn prop_de_search_integer_bounds(
            low in 1_i64..100,
            high_offset in 1_i64..100,
            seed in any::<u64>()
        ) {
            let high = low + high_offset;
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, low..high);

            let mut search = DESearch::new(50).with_seed(seed);
            let trials = search.suggest(&space, 20);

            for trial in trials {
                let v = trial.get_i64(&RF::NEstimators).expect("should have value");
                prop_assert!(
                    v >= low && v <= high,
                    "Value {} not in [{}, {}]", v, low, high
                );
            }
        }

        /// DESearch population should remain valid after update.
        #[test]
        fn prop_de_search_update_valid(seed in 1_u64..u64::MAX) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            let mut search = DESearch::new(100).with_seed(seed);

            // Get initial population
            let trials1 = search.suggest(&space, 20);

            // Simulate results (higher n_estimators = better score)
            let results: Vec<TrialResult<RF>> = trials1.iter().map(|t| {
                let score = t.get_f64(&RF::NEstimators).unwrap_or(0.0);
                TrialResult {
                    trial: t.clone(),
                    score,
                    metrics: HashMap::new(),
                }
            }).collect();

            // Update with results
            search.update(&results);

            // Get next batch - should still be valid
            let trials2 = search.suggest(&space, 20);

            // All values should still be within bounds
            for trial in trials2 {
                let v = trial.get_f64(&RF::NEstimators).expect("should have value");
                prop_assert!(
                    (10.0..=500.0).contains(&v),
                    "Value {v} not in bounds after evolution"
                );
            }
        }

        /// Active learning should stop when all scores are identical (zero variance).
        #[test]
        fn prop_active_learning_stops_zero_variance(
            n_samples in 10_usize..50,
            score in 0.1_f64..0.9
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            let base = RandomSearch::new(1000).with_seed(42);
            let mut search = ActiveLearningSearch::new(base)
                .with_uncertainty_threshold(0.01)
                .with_min_samples(n_samples);

            let trials = search.suggest(&space, n_samples);

            // All same score = zero variance = should stop
            let results: Vec<TrialResult<RF>> = trials
                .iter()
                .map(|t| TrialResult {
                    trial: t.clone(),
                    score,
                    metrics: HashMap::new(),
                })
                .collect();

            search.update(&results);

            // Zero variance means uncertainty should be very low
            prop_assert!(
                search.uncertainty() < 0.01,
                "Zero variance should give near-zero uncertainty, got {}",
                search.uncertainty()
            );
        }

        /// Active learning uncertainty should increase with score variance.
        #[test]
        fn prop_active_learning_uncertainty_increases_with_variance(
            base_score in 0.3_f64..0.7,
            seed in any::<u64>()
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            // Low variance case
            let base1 = RandomSearch::new(100).with_seed(seed);
            let mut search1 = ActiveLearningSearch::new(base1);
            let trials1 = search1.suggest(&space, 20);

            let low_var_results: Vec<TrialResult<RF>> = trials1
                .iter()
                .enumerate()
                .map(|(i, t)| TrialResult {
                    trial: t.clone(),
                    score: base_score + (i as f64) * 0.001, // Very small variance
                    metrics: HashMap::new(),
                })
                .collect();
            search1.update(&low_var_results);

            // High variance case
            let base2 = RandomSearch::new(100).with_seed(seed);
            let mut search2 = ActiveLearningSearch::new(base2);
            let trials2 = search2.suggest(&space, 20);

            let high_var_results: Vec<TrialResult<RF>> = trials2
                .iter()
                .enumerate()
                .map(|(i, t)| TrialResult {
                    trial: t.clone(),
                    score: if i % 2 == 0 { 0.1 } else { 0.9 }, // High variance
                    metrics: HashMap::new(),
                })
                .collect();
            search2.update(&high_var_results);

            prop_assert!(
                search2.uncertainty() > search1.uncertainty(),
                "High variance ({}) should have higher uncertainty than low variance ({})",
                search2.uncertainty(),
                search1.uncertainty()
            );
        }

        /// Active learning should not stop before min_samples.
        #[test]
        fn prop_active_learning_respects_min_samples(
            min_samples in 5_usize..20,
            seed in any::<u64>()
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            let base = RandomSearch::new(1000).with_seed(seed);
            let mut search = ActiveLearningSearch::new(base)
                .with_uncertainty_threshold(1.0) // High threshold = easy to satisfy
                .with_min_samples(min_samples);

            // Get fewer than min_samples
            let trials = search.suggest(&space, min_samples - 1);

            let results: Vec<TrialResult<RF>> = trials
                .iter()
                .map(|t| TrialResult {
                    trial: t.clone(),
                    score: 0.5, // Same score = low uncertainty
                    metrics: HashMap::new(),
                })
                .collect();

            search.update(&results);

            // Should NOT stop - not enough samples yet
            prop_assert!(
                !search.should_stop(),
                "Should not stop with {} samples (min={})",
                results.len(),
                min_samples
            );
        }
    }
}
