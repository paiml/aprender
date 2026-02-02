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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_path() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![100.0, 105.0, 110.0],
            PathMetadata {
                path_id: 0,
                seed: 42,
                is_antithetic: false,
            },
        );

        assert_eq!(path.initial_value(), Some(100.0));
        assert_eq!(path.final_value(), Some(110.0));
        assert!((path.total_return().unwrap() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_period_returns() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![100.0, 110.0, 121.0],
            PathMetadata {
                path_id: 0,
                seed: 42,
                is_antithetic: false,
            },
        );

        let returns = path.period_returns();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 0.001);
        assert!((returns[1] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_log_returns() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![100.0, 110.0, 121.0],
            PathMetadata {
                path_id: 0,
                seed: 42,
                is_antithetic: false,
            },
        );

        let log_returns = path.log_returns();
        assert_eq!(log_returns.len(), 2);
        assert!((log_returns[0] - (1.1_f64).ln()).abs() < 0.001);
    }

    #[test]
    fn test_time_horizon_years() {
        let horizon = TimeHorizon::years(1);
        assert_eq!(horizon.n_steps(), 12); // Monthly default
        assert!((horizon.dt() - 1.0 / 12.0).abs() < 0.001);
    }

    #[test]
    fn test_time_horizon_quarters() {
        let horizon = TimeHorizon::quarters(4);
        assert!((horizon.duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_time_horizon_daily() {
        let horizon = TimeHorizon::years(1).with_step(TimeStep::Daily);
        assert_eq!(horizon.n_steps(), 252);
    }

    #[test]
    fn test_time_points() {
        let horizon = TimeHorizon::years(1).with_step(TimeStep::Quarterly);
        let points = horizon.time_points();
        assert_eq!(points.len(), 5); // 0, 0.25, 0.5, 0.75, 1.0
        assert!((points[0] - 0.0).abs() < 0.001);
        assert!((points[4] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_percentile_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!((percentile(&values, 0.0) - 1.0).abs() < 0.001);
        assert!((percentile(&values, 0.5) - 5.5).abs() < 0.001);
        assert!((percentile(&values, 1.0) - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_percentile_interpolation() {
        let values = vec![0.0, 1.0];
        assert!((percentile(&values, 0.25) - 0.25).abs() < 0.001);
        assert!((percentile(&values, 0.75) - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_statistics_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Statistics::from_values(&values);

        assert!((stats.mean - 3.0).abs() < 0.001);
        assert!((stats.min - 1.0).abs() < 0.001);
        assert!((stats.max - 5.0).abs() < 0.001);
        assert_eq!(stats.n, 5);
    }

    #[test]
    fn test_statistics_normal() {
        // Generate roughly normal samples
        let values: Vec<f64> = (0..1000)
            .map(|i| {
                let x = (i as f64 / 1000.0) * 2.0 - 1.0;
                x * x.signum() * (-x.abs()).exp()
            })
            .collect();

        let stats = Statistics::from_values(&values);
        assert!(stats.std > 0.0);
        assert!(stats.skewness.is_finite());
        assert!(stats.kurtosis.is_finite());
    }

    #[test]
    fn test_percentiles_from_values() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let pcts = Percentiles::from_values(&values);

        assert!(pcts.p50 > pcts.p25);
        assert!(pcts.p75 > pcts.p50);
        assert!(pcts.p99 > pcts.p95);
    }

    #[test]
    fn test_budget_max_simulations() {
        assert_eq!(Budget::Simulations(1000).max_simulations(), 1000);
        assert_eq!(
            Budget::Convergence {
                patience: 5,
                min_delta: 0.001,
                max_simulations: 5000
            }
            .max_simulations(),
            5000
        );
    }

    #[test]
    fn test_empty_statistics() {
        let stats = Statistics::from_values(&[]);
        assert_eq!(stats.n, 0);
        assert!((stats.mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_path_final_value() {
        let path = SimulationPath::new(
            vec![],
            vec![],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert_eq!(path.final_value(), None);
    }

    #[test]
    fn test_empty_path_initial_value() {
        let path = SimulationPath::new(
            vec![],
            vec![],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert_eq!(path.initial_value(), None);
    }

    #[test]
    fn test_total_return_empty() {
        let path = SimulationPath::new(
            vec![],
            vec![],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert_eq!(path.total_return(), None);
    }

    #[test]
    fn test_total_return_zero_initial() {
        let path = SimulationPath::new(
            vec![0.0, 1.0],
            vec![0.0, 5.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        // initial_value is 0.0, so total_return should be None
        assert_eq!(path.total_return(), None);
    }

    #[test]
    fn test_period_returns_fewer_than_two_values() {
        let path = SimulationPath::new(
            vec![0.0],
            vec![100.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert!(path.period_returns().is_empty());
    }

    #[test]
    fn test_period_returns_with_zero_value() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![0.0, 50.0, 100.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        let returns = path.period_returns();
        // First window [0.0, 50.0] => w[0] is 0.0, filtered out
        // Second window [50.0, 100.0] => (100.0 - 50.0) / 50.0 = 1.0
        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_log_returns_fewer_than_two_values() {
        let path = SimulationPath::new(
            vec![0.0],
            vec![100.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert!(path.log_returns().is_empty());
    }

    #[test]
    fn test_log_returns_with_zero_value() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![0.0, 50.0, 100.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        let log_ret = path.log_returns();
        // First window [0.0, 50.0] => w[0] is 0.0, filtered out
        // Second window [50.0, 100.0] => ln(100/50) = ln(2)
        assert_eq!(log_ret.len(), 1);
        assert!((log_ret[0] - 2.0_f64.ln()).abs() < 0.001);
    }

    #[test]
    fn test_log_returns_with_negative_value() {
        let path = SimulationPath::new(
            vec![0.0, 0.5],
            vec![100.0, -50.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        let log_ret = path.log_returns();
        // w[1] is negative, so filtered out
        assert!(log_ret.is_empty());
    }

    #[test]
    fn test_time_horizon_months() {
        let horizon = TimeHorizon::months(6);
        // duration = 6/12 = 0.5 years
        assert!((horizon.duration - 0.5).abs() < 0.001);
        // Default step for months() is Weekly
        let n_steps = horizon.n_steps();
        assert_eq!(n_steps, 26); // 0.5 * 52 = 26
    }

    #[test]
    fn test_time_horizon_yearly_step() {
        let horizon = TimeHorizon::years(3).with_step(TimeStep::Yearly);
        assert_eq!(horizon.n_steps(), 3);
        assert!((horizon.dt() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_time_horizon_custom_step() {
        let horizon = TimeHorizon::years(1).with_step(TimeStep::Custom(100.0));
        assert_eq!(horizon.n_steps(), 100);
    }

    #[test]
    fn test_budget_evaluations() {
        assert_eq!(Budget::Evaluations(500).max_simulations(), 500);
    }

    #[test]
    fn test_percentiles_iqr() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let pcts = Percentiles::from_values(&values);
        let iqr = pcts.iqr();
        assert!((iqr - (pcts.p75 - pcts.p25)).abs() < 1e-10);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_statistics_cv_near_zero_mean() {
        let stats = Statistics {
            mean: 0.0,
            std: 1.0,
            ..Statistics::default()
        };
        assert_eq!(stats.cv(), f64::INFINITY);
    }

    #[test]
    fn test_statistics_cv_normal() {
        let stats = Statistics {
            mean: 10.0,
            std: 2.0,
            ..Statistics::default()
        };
        assert!((stats.cv() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_statistics_sem_zero_n() {
        let stats = Statistics {
            n: 0,
            std: 1.0,
            ..Statistics::default()
        };
        assert_eq!(stats.sem(), f64::INFINITY);
    }

    #[test]
    fn test_statistics_sem_normal() {
        let stats = Statistics {
            n: 100,
            std: 10.0,
            ..Statistics::default()
        };
        assert!((stats.sem() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_percentile_empty() {
        assert!((percentile(&[], 0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_single_element() {
        assert!((percentile(&[42.0], 0.5) - 42.0).abs() < 1e-10);
        assert!((percentile(&[42.0], 0.0) - 42.0).abs() < 1e-10);
        assert!((percentile(&[42.0], 1.0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_statistics_single_value_zero_std() {
        let stats = Statistics::from_values(&[5.0]);
        assert!((stats.mean - 5.0).abs() < 1e-10);
        assert!((stats.std - 0.0).abs() < 1e-10);
        // Skewness and kurtosis should be 0 when std is 0
        assert!((stats.skewness - 0.0).abs() < 1e-10);
        assert!((stats.kurtosis - 0.0).abs() < 1e-10);
        assert_eq!(stats.n, 1);
    }

    #[test]
    fn test_path_metadata_antithetic() {
        let meta = PathMetadata {
            path_id: 5,
            seed: 123,
            is_antithetic: true,
        };
        assert!(meta.is_antithetic);
        assert_eq!(meta.path_id, 5);
        assert_eq!(meta.seed, 123);
    }

    #[test]
    fn test_simulation_path_debug_clone() {
        let path = SimulationPath::new(
            vec![0.0, 1.0],
            vec![100.0, 110.0],
            PathMetadata {
                path_id: 0,
                seed: 42,
                is_antithetic: false,
            },
        );
        let debug_str = format!("{:?}", path);
        assert!(debug_str.contains("SimulationPath"));

        let cloned = path.clone();
        assert_eq!(cloned.values.len(), path.values.len());
    }

    #[test]
    fn test_time_horizon_debug_clone() {
        let horizon = TimeHorizon::years(1);
        let debug_str = format!("{:?}", horizon);
        assert!(debug_str.contains("TimeHorizon"));

        let cloned = horizon.clone();
        assert!((cloned.duration - horizon.duration).abs() < 1e-10);
    }

    #[test]
    fn test_time_step_debug_clone() {
        let step = TimeStep::Daily;
        let debug_str = format!("{:?}", step);
        assert!(debug_str.contains("Daily"));

        let cloned = step;
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn test_budget_debug_clone() {
        let budget = Budget::Convergence {
            patience: 5,
            min_delta: 0.01,
            max_simulations: 1000,
        };
        let debug_str = format!("{:?}", budget);
        assert!(debug_str.contains("Convergence"));

        let cloned = budget.clone();
        assert_eq!(cloned.max_simulations(), 1000);
    }

    #[test]
    fn test_percentiles_default() {
        let pcts = Percentiles::default();
        assert!((pcts.p50 - 0.0).abs() < 1e-10);
        assert!((pcts.p1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_statistics_default() {
        let stats = Statistics::default();
        assert_eq!(stats.n, 0);
        assert!((stats.mean - 0.0).abs() < 1e-10);
        assert!((stats.std - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_horizon_time_points_count() {
        let horizon = TimeHorizon::years(1).with_step(TimeStep::Monthly);
        let points = horizon.time_points();
        // n_steps is 12, time_points has n+1 elements
        assert_eq!(points.len(), 13);
    }

    #[test]
    fn test_percentile_clamp_out_of_range() {
        let values = vec![1.0, 2.0, 3.0];
        // p < 0 should clamp to 0
        let p_neg = percentile(&values, -0.5);
        assert!((p_neg - 1.0).abs() < 0.001);
        // p > 1 should clamp to 1
        let p_over = percentile(&values, 1.5);
        assert!((p_over - 3.0).abs() < 0.001);
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_percentile_monotonic(values in prop::collection::vec(0.0..100.0f64, 10..100)) {
                let p25 = percentile(&values, 0.25);
                let p50 = percentile(&values, 0.50);
                let p75 = percentile(&values, 0.75);

                prop_assert!(p25 <= p50);
                prop_assert!(p50 <= p75);
            }

            #[test]
            fn prop_percentile_bounded(values in prop::collection::vec(0.0..100.0f64, 1..100)) {
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let pct = percentile(&values, p);
                    prop_assert!(pct >= min - 0.001);
                    prop_assert!(pct <= max + 0.001);
                }
            }

            #[test]
            fn prop_statistics_std_non_negative(values in prop::collection::vec(-100.0..100.0f64, 2..100)) {
                let stats = Statistics::from_values(&values);
                prop_assert!(stats.std >= 0.0);
            }

            #[test]
            fn prop_statistics_min_leq_max(values in prop::collection::vec(-100.0..100.0f64, 1..100)) {
                let stats = Statistics::from_values(&values);
                prop_assert!(stats.min <= stats.max);
            }
        }
    }
}
