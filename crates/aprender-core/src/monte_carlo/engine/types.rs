//! Core types for Monte Carlo simulations
//!
//! Provides fundamental data structures for simulation paths,
//! time horizons, and statistical computations.

/// A single simulation path with time points and values
#[derive(Debug, Clone)]
pub struct SimulationPath {
    /// Time points
    pub time: Vec<f64>,
    /// Values at each time point
    pub values: Vec<f64>,
    /// Metadata about this path
    pub metadata: PathMetadata,
}

impl SimulationPath {
    /// Create a new simulation path
    #[must_use]
    pub fn new(time: Vec<f64>, values: Vec<f64>, metadata: PathMetadata) -> Self {
        Self {
            time,
            values,
            metadata,
        }
    }

    /// Get the final value of the path
    #[must_use]
    pub fn final_value(&self) -> Option<f64> {
        self.values.last().copied()
    }

    /// Get the initial value of the path
    #[must_use]
    pub fn initial_value(&self) -> Option<f64> {
        self.values.first().copied()
    }

    /// Calculate total return
    #[must_use]
    pub fn total_return(&self) -> Option<f64> {
        let initial = self.initial_value()?;
        let final_val = self.final_value()?;
        if initial > 0.0 {
            Some((final_val - initial) / initial)
        } else {
            None
        }
    }

    /// Calculate period-over-period returns
    #[must_use]
    pub fn period_returns(&self) -> Vec<f64> {
        if self.values.len() < 2 {
            return Vec::new();
        }

        self.values
            .windows(2)
            .filter_map(|w| {
                if w[0] > 0.0 {
                    Some((w[1] - w[0]) / w[0])
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate log returns
    #[must_use]
    pub fn log_returns(&self) -> Vec<f64> {
        if self.values.len() < 2 {
            return Vec::new();
        }

        self.values
            .windows(2)
            .filter_map(|w| {
                if w[0] > 0.0 && w[1] > 0.0 {
                    Some((w[1] / w[0]).ln())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Metadata for a simulation path
#[derive(Debug, Clone, Copy)]
pub struct PathMetadata {
    /// Unique identifier for this path
    pub path_id: usize,
    /// Seed used to generate this path
    pub seed: u64,
    /// Whether this is an antithetic path
    pub is_antithetic: bool,
}

/// Time horizon configuration for simulations
#[derive(Debug, Clone)]
pub struct TimeHorizon {
    /// Total duration in years
    pub duration: f64,
    /// Time step configuration
    pub step: TimeStep,
}

impl TimeHorizon {
    /// Create a time horizon spanning the given number of years
    #[must_use]
    pub fn years(n: u32) -> Self {
        Self {
            duration: f64::from(n),
            step: TimeStep::Monthly,
        }
    }

    /// Create a time horizon spanning the given number of quarters
    #[must_use]
    pub fn quarters(n: u32) -> Self {
        Self {
            duration: f64::from(n) * 0.25,
            step: TimeStep::Monthly,
        }
    }

    /// Create a time horizon spanning the given number of months
    #[must_use]
    pub fn months(n: u32) -> Self {
        Self {
            duration: f64::from(n) / 12.0,
            step: TimeStep::Weekly,
        }
    }

    /// Set the time step
    #[must_use]
    pub fn with_step(mut self, step: TimeStep) -> Self {
        self.step = step;
        self
    }

    /// Get the number of time steps
    #[must_use]
    pub fn n_steps(&self) -> usize {
        let steps_per_year = match self.step {
            TimeStep::Daily => 252.0,
            TimeStep::Weekly => 52.0,
            TimeStep::Monthly => 12.0,
            TimeStep::Quarterly => 4.0,
            TimeStep::Yearly => 1.0,
            TimeStep::Custom(n) => n,
        };
        (self.duration * steps_per_year).round() as usize
    }

    /// Get the time increment
    #[must_use]
    pub fn dt(&self) -> f64 {
        self.duration / self.n_steps() as f64
    }

    /// Get all time points
    #[must_use]
    pub fn time_points(&self) -> Vec<f64> {
        let n = self.n_steps();
        let dt = self.dt();
        (0..=n).map(|i| i as f64 * dt).collect()
    }
}

/// Time step granularity
#[derive(Debug, Clone, Copy)]
pub enum TimeStep {
    /// Daily (252 trading days per year)
    Daily,
    /// Weekly (52 weeks per year)
    Weekly,
    /// Monthly (12 months per year)
    Monthly,
    /// Quarterly (4 quarters per year)
    Quarterly,
    /// Yearly
    Yearly,
    /// Custom steps per year
    Custom(f64),
}

/// Computational budget for simulations
#[derive(Debug, Clone)]
pub enum Budget {
    /// Fixed number of simulations
    Simulations(usize),

    /// Fixed number of function evaluations
    Evaluations(usize),

    /// Run until convergence with patience
    Convergence {
        /// Number of stable iterations required
        patience: usize,
        /// Minimum change to consider significant
        min_delta: f64,
        /// Maximum simulations as safety limit
        max_simulations: usize,
    },
}

impl Budget {
    /// Get maximum number of simulations allowed
    #[must_use]
    pub fn max_simulations(&self) -> usize {
        match self {
            Self::Simulations(n) | Self::Evaluations(n) => *n,
            Self::Convergence {
                max_simulations, ..
            } => *max_simulations,
        }
    }
}

/// Percentile values for distributions
#[derive(Debug, Clone, Default)]
pub struct Percentiles {
    /// 1st percentile
    pub p1: f64,
    /// 5th percentile
    pub p5: f64,
    /// 10th percentile
    pub p10: f64,
    /// 25th percentile (Q1)
    pub p25: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 75th percentile (Q3)
    pub p75: f64,
    /// 90th percentile
    pub p90: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

impl Percentiles {
    /// Calculate percentiles from values
    #[must_use]
    pub fn from_values(values: &[f64]) -> Self {
        Self {
            p1: percentile(values, 0.01),
            p5: percentile(values, 0.05),
            p10: percentile(values, 0.10),
            p25: percentile(values, 0.25),
            p50: percentile(values, 0.50),
            p75: percentile(values, 0.75),
            p90: percentile(values, 0.90),
            p95: percentile(values, 0.95),
            p99: percentile(values, 0.99),
        }
    }

    /// Get interquartile range
    #[must_use]
    pub fn iqr(&self) -> f64 {
        self.p75 - self.p25
    }
}

/// Summary statistics for a distribution
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Sample mean
    pub mean: f64,
    /// Sample standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Skewness
    pub skewness: f64,
    /// Excess kurtosis
    pub kurtosis: f64,
    /// Number of samples
    pub n: usize,
}

impl Statistics {
    /// Calculate statistics from values
    #[must_use]
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let n = values.len();
        let n_f = n as f64;

        // Mean
        let mean = values.iter().sum::<f64>() / n_f;

        // Variance and standard deviation
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_f - 1.0).max(1.0);
        let std = variance.sqrt();

        // Min and max
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Skewness
        let skewness = if std > 0.0 {
            let m3 = values
                .iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f64>();
            m3 / n_f
        } else {
            0.0
        };

        // Kurtosis (excess kurtosis = kurtosis - 3)
        let kurtosis = if std > 0.0 {
            let m4 = values
                .iter()
                .map(|x| ((x - mean) / std).powi(4))
                .sum::<f64>();
            m4 / n_f - 3.0
        } else {
            0.0
        };

        Self {
            mean,
            std,
            min,
            max,
            skewness,
            kurtosis,
            n,
        }
    }

    /// Calculate coefficient of variation
    #[must_use]
    pub fn cv(&self) -> f64 {
        if self.mean.abs() > 1e-10 {
            self.std / self.mean.abs()
        } else {
            f64::INFINITY
        }
    }

    /// Calculate standard error of the mean
    #[must_use]
    pub fn sem(&self) -> f64 {
        if self.n > 0 {
            self.std / (self.n as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

/// Calculate a percentile from a slice of values
///
/// Uses linear interpolation between data points.
#[must_use]
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p = p.clamp(0.0, 1.0);
    let n = sorted.len();

    if n == 1 {
        return sorted[0];
    }

    let idx = p * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

#[path = "types_tests.rs"]
mod types_tests;
