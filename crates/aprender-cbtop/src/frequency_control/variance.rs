//! Frequency variance measurement and statistics.

/// Frequency variance measurement
#[derive(Debug, Clone, Default)]
pub struct FrequencyVariance {
    /// Mean frequency in MHz
    pub mean_mhz: f64,
    /// Standard deviation in MHz
    pub std_dev_mhz: f64,
    /// Coefficient of variation (%)
    pub cv_percent: f64,
    /// Minimum frequency in MHz
    pub min_mhz: f64,
    /// Maximum frequency in MHz
    pub max_mhz: f64,
    /// Number of samples
    pub sample_count: usize,
}

impl FrequencyVariance {
    /// Compute variance statistics from a set of frequency samples (MHz).
    pub fn from_samples(readings: &[f64]) -> Self {
        if readings.is_empty() {
            return Self::default();
        }
        let n = readings.len() as f64;
        let mean = readings.iter().sum::<f64>() / n;
        let var = readings.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std_dev = var.sqrt();
        let cv = if mean > 0.0 { std_dev / mean * 100.0 } else { 0.0 };
        Self {
            mean_mhz: mean,
            std_dev_mhz: std_dev,
            cv_percent: cv,
            min_mhz: readings.iter().copied().fold(f64::INFINITY, f64::min),
            max_mhz: readings.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            sample_count: readings.len(),
        }
    }

    /// Check if variance is acceptable (<3% CV)
    pub fn is_stable(&self) -> bool {
        self.cv_percent < 3.0
    }

    /// Get range in MHz
    pub fn range_mhz(&self) -> f64 {
        self.max_mhz - self.min_mhz
    }
}
