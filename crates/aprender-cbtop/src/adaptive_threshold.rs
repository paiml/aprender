//! Adaptive Threshold Learning System (PMAT-037)
//!
//! Dynamic threshold learning that adjusts warning/critical bounds based on historical baseline data.
//!
//! # Features
//!
//! - Baseline learning from historical samples (μ±2σ)
//! - Percentile-based threshold computation
//! - Outlier filtering to prevent over-learning
//! - User override support for static thresholds
//!
//! # Falsification Criteria (F1291-F1300)
//!
//! See `tests/adaptive_threshold_f1291.rs` for falsification tests.

/// Minimum samples required for learning
pub const MIN_SAMPLES_FOR_LEARNING: usize = 10;

/// Default confidence level for bounds (95%)
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;

/// Default outlier threshold (3 standard deviations)
pub const DEFAULT_OUTLIER_THRESHOLD: f64 = 3.0;

/// Threshold direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdDirection {
    /// Upper bound (warn if above)
    Upper,
    /// Lower bound (warn if below)
    Lower,
    /// Both bounds (warn if outside range)
    Both,
}

impl ThresholdDirection {
    /// Get direction name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Upper => "upper",
            Self::Lower => "lower",
            Self::Both => "both",
        }
    }
}

/// Learned threshold bounds
#[derive(Debug, Clone)]
pub struct LearnedThreshold {
    /// Metric name
    pub metric: String,
    /// Sample mean
    pub mean: f64,
    /// Sample standard deviation
    pub std_dev: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Lower bound (warning)
    pub lower_bound: f64,
    /// Upper bound (warning)
    pub upper_bound: f64,
    /// Lower bound (critical)
    pub lower_critical: f64,
    /// Upper bound (critical)
    pub upper_critical: f64,
    /// Coefficient of variation
    pub cv: f64,
    /// Confidence level used
    pub confidence_level: f64,
    /// Threshold direction
    pub direction: ThresholdDirection,
}

impl LearnedThreshold {
    /// Check if value is within warning bounds
    pub fn is_warning(&self, value: f64) -> bool {
        match self.direction {
            ThresholdDirection::Upper => value > self.upper_bound,
            ThresholdDirection::Lower => value < self.lower_bound,
            ThresholdDirection::Both => value < self.lower_bound || value > self.upper_bound,
        }
    }

    /// Check if value is critical
    pub fn is_critical(&self, value: f64) -> bool {
        match self.direction {
            ThresholdDirection::Upper => value > self.upper_critical,
            ThresholdDirection::Lower => value < self.lower_critical,
            ThresholdDirection::Both => value < self.lower_critical || value > self.upper_critical,
        }
    }

    /// Check if value is normal
    pub fn is_normal(&self, value: f64) -> bool {
        !self.is_warning(value)
    }

    /// Export to JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"metric":"{}","mean":{},"std_dev":{},"cv":{},"lower_bound":{},"upper_bound":{},"sample_count":{}}}"#,
            self.metric,
            self.mean,
            self.std_dev,
            self.cv,
            self.lower_bound,
            self.upper_bound,
            self.sample_count
        )
    }
}

/// Threshold learner
#[derive(Debug)]
pub struct ThresholdLearner {
    /// Metric name
    metric: String,
    /// Historical samples
    samples: Vec<f64>,
    /// Maximum samples to keep
    max_samples: usize,
    /// Outlier threshold (std devs)
    outlier_threshold: f64,
    /// Confidence level
    confidence_level: f64,
    /// Warning multiplier (std devs from mean)
    warning_multiplier: f64,
    /// Critical multiplier (std devs from mean)
    critical_multiplier: f64,
    /// Threshold direction
    direction: ThresholdDirection,
    /// User override value (if set)
    override_value: Option<f64>,
}

impl ThresholdLearner {
    /// Create new learner
    pub fn new(metric: &str) -> Self {
        Self {
            metric: metric.to_string(),
            samples: Vec::new(),
            max_samples: 1000,
            outlier_threshold: DEFAULT_OUTLIER_THRESHOLD,
            confidence_level: DEFAULT_CONFIDENCE_LEVEL,
            warning_multiplier: 2.0,
            critical_multiplier: 3.0,
            direction: ThresholdDirection::Upper,
            override_value: None,
        }
    }

    /// Set threshold direction
    pub fn with_direction(mut self, direction: ThresholdDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set warning multiplier
    pub fn with_warning_multiplier(mut self, multiplier: f64) -> Self {
        self.warning_multiplier = multiplier.max(0.5);
        self
    }

    /// Set critical multiplier
    pub fn with_critical_multiplier(mut self, multiplier: f64) -> Self {
        self.critical_multiplier = multiplier.max(1.0);
        self
    }

    /// Set outlier threshold
    pub fn with_outlier_threshold(mut self, threshold: f64) -> Self {
        self.outlier_threshold = threshold.max(2.0);
        self
    }

    /// Set max samples
    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = max.max(10);
        self
    }

    /// Set user override
    pub fn with_override(mut self, value: f64) -> Self {
        self.override_value = Some(value);
        self
    }

    /// Clear override
    pub fn clear_override(&mut self) {
        self.override_value = None;
    }

    /// Add sample
    pub fn add_sample(&mut self, value: f64) {
        self.samples.push(value);

        // Trim to max samples
        if self.samples.len() > self.max_samples {
            self.samples.remove(0);
        }
    }

    /// Add multiple samples
    pub fn add_samples(&mut self, values: &[f64]) {
        for &v in values {
            self.add_sample(v);
        }
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Check if sufficient samples
    pub fn has_sufficient_samples(&self) -> bool {
        self.samples.len() >= MIN_SAMPLES_FOR_LEARNING
    }

    /// Calculate mean
    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate standard deviation
    fn std_dev(data: &[f64], mean: f64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Filter outliers
    pub fn filter_outliers(&self) -> Vec<f64> {
        if self.samples.len() < 3 {
            return self.samples.clone();
        }

        let mean = Self::mean(&self.samples);
        let std_dev = Self::std_dev(&self.samples, mean);

        if std_dev < 1e-10 {
            return self.samples.clone();
        }

        self.samples
            .iter()
            .filter(|&&x| ((x - mean) / std_dev).abs() <= self.outlier_threshold)
            .copied()
            .collect()
    }

    /// Learn baseline threshold
    pub fn learn_baseline(&self) -> Option<LearnedThreshold> {
        if !self.has_sufficient_samples() {
            return None;
        }

        // Filter outliers first
        let filtered = self.filter_outliers();
        if filtered.len() < MIN_SAMPLES_FOR_LEARNING {
            return None;
        }

        let mean = Self::mean(&filtered);
        let std_dev = Self::std_dev(&filtered, mean);
        let cv = if mean.abs() > 1e-10 {
            (std_dev / mean.abs()) * 100.0
        } else {
            0.0
        };

        // Compute bounds
        let (lower_bound, upper_bound) = match self.direction {
            ThresholdDirection::Upper => {
                (f64::NEG_INFINITY, mean + self.warning_multiplier * std_dev)
            }
            ThresholdDirection::Lower => (mean - self.warning_multiplier * std_dev, f64::INFINITY),
            ThresholdDirection::Both => (
                mean - self.warning_multiplier * std_dev,
                mean + self.warning_multiplier * std_dev,
            ),
        };

        let (lower_critical, upper_critical) = match self.direction {
            ThresholdDirection::Upper => {
                (f64::NEG_INFINITY, mean + self.critical_multiplier * std_dev)
            }
            ThresholdDirection::Lower => (mean - self.critical_multiplier * std_dev, f64::INFINITY),
            ThresholdDirection::Both => (
                mean - self.critical_multiplier * std_dev,
                mean + self.critical_multiplier * std_dev,
            ),
        };

        Some(LearnedThreshold {
            metric: self.metric.clone(),
            mean,
            std_dev,
            sample_count: filtered.len(),
            lower_bound,
            upper_bound,
            lower_critical,
            upper_critical,
            cv,
            confidence_level: self.confidence_level,
            direction: self.direction,
        })
    }

    /// Get percentile threshold
    pub fn percentile_threshold(&self, percentile: f64) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }

        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p = percentile.clamp(0.0, 100.0);
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    /// Get effective threshold (respects override)
    pub fn get_effective_threshold(&self) -> Option<f64> {
        if let Some(override_val) = self.override_value {
            return Some(override_val);
        }

        self.learn_baseline().map(|t| t.upper_bound)
    }

    /// Check value against learned threshold
    pub fn check(&self, value: f64) -> ThresholdCheck {
        // Override takes precedence
        if let Some(override_val) = self.override_value {
            let is_exceeded = match self.direction {
                ThresholdDirection::Upper => value > override_val,
                ThresholdDirection::Lower => value < override_val,
                ThresholdDirection::Both => false, // Not applicable for single override
            };
            return ThresholdCheck {
                value,
                threshold: override_val,
                is_warning: is_exceeded,
                is_critical: false,
                is_override: true,
            };
        }

        // Use learned threshold
        if let Some(learned) = self.learn_baseline() {
            ThresholdCheck {
                value,
                threshold: learned.upper_bound,
                is_warning: learned.is_warning(value),
                is_critical: learned.is_critical(value),
                is_override: false,
            }
        } else {
            // Insufficient data
            ThresholdCheck {
                value,
                threshold: 0.0,
                is_warning: false,
                is_critical: false,
                is_override: false,
            }
        }
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Result of threshold check
#[derive(Debug, Clone)]
pub struct ThresholdCheck {
    /// Checked value
    pub value: f64,
    /// Threshold used
    pub threshold: f64,
    /// Is warning triggered
    pub is_warning: bool,
    /// Is critical triggered
    pub is_critical: bool,
    /// Was override used
    pub is_override: bool,
}

impl ThresholdCheck {
    /// Check if passed (not warning or critical)
    pub fn passed(&self) -> bool {
        !self.is_warning && !self.is_critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_direction() {
        assert_eq!(ThresholdDirection::Upper.name(), "upper");
        assert_eq!(ThresholdDirection::Lower.name(), "lower");
        assert_eq!(ThresholdDirection::Both.name(), "both");
    }

    #[test]
    fn test_learner_creation() {
        let learner = ThresholdLearner::new("cpu_temp");
        assert_eq!(learner.sample_count(), 0);
        assert!(!learner.has_sufficient_samples());
    }

    #[test]
    fn test_add_samples() {
        let mut learner = ThresholdLearner::new("test");
        learner.add_sample(10.0);
        learner.add_sample(11.0);
        assert_eq!(learner.sample_count(), 2);
    }

    #[test]
    fn test_learn_baseline() {
        let mut learner = ThresholdLearner::new("test");

        // Add enough samples
        for i in 0..20 {
            learner.add_sample(100.0 + (i % 3) as f64);
        }

        let threshold = learner.learn_baseline().unwrap();
        assert!(threshold.mean > 99.0 && threshold.mean < 103.0);
        assert!(threshold.std_dev > 0.0);
    }

    #[test]
    fn test_override() {
        let mut learner = ThresholdLearner::new("test").with_override(50.0);

        for i in 0..20 {
            learner.add_sample(100.0 + i as f64);
        }

        let effective = learner.get_effective_threshold().unwrap();
        assert_eq!(effective, 50.0);
    }

    #[test]
    fn test_percentile() {
        let mut learner = ThresholdLearner::new("test");

        for i in 0..100 {
            learner.add_sample(i as f64);
        }

        let p50 = learner.percentile_threshold(50.0).unwrap();
        assert!((p50 - 50.0).abs() < 1.0);
    }
}
