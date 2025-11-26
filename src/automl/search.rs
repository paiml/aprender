//! Search space and optimization algorithms.
//!
//! Implements random search [1] and grid search for hyperparameter optimization.
//!
//! # References
//!
//! [1] Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization. JMLR.

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Range;

use crate::automl::params::ParamKey;

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
/// grid search with 60x fewer trials in many cases [1].
///
/// # References
///
/// [1] Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization. JMLR.
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
    }
}
