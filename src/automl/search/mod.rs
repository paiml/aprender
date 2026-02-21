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

    /// Add a raw `HyperParam`.
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

include!("trial.rs");
include!("active_learning_search.rs");
include!("active_learning.rs");
