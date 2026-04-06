// Configuration for statistical regression detection
//
// Key Innovation: No magic numbers. Thresholds adapt to project-specific variance.
// (Section 6.4 of single-shot-compile-tooling-spec.md)

use serde::{Deserialize, Serialize};

/// Configuration for statistical regression detection
///
/// Instead of fixed percentage thresholds (e.g., "5% increase"), this uses:
/// - P-value threshold for statistical significance
/// - Noise filtering via Delta Debugging
/// - Dynamic thresholds based on baseline variance
///
/// # Example
/// ```
/// use renacer::regression::RegressionConfig;
///
/// let config = RegressionConfig::default();
/// assert_eq!(config.significance_level, 0.05); // 95% confidence
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Statistical significance level (alpha) for hypothesis testing
    ///
    /// - 0.05 (default): 95% confidence level, detects regressions with <5% false positive rate
    /// - 0.01: 99% confidence level, stricter (fewer false positives, more false negatives)
    /// - 0.10: 90% confidence level, looser (more false positives, fewer false negatives)
    ///
    /// Scientific Foundation:
    /// \[9\] Heger et al. (2013): Fixed thresholds (e.g., "5%") yield high false positives.
    ///     P-values adapt to baseline variance.
    pub significance_level: f64,

    /// Minimum sample size for statistical tests
    ///
    /// T-tests require at least 2 samples per distribution, but we recommend
    /// more for reliable results (Central Limit Theorem).
    ///
    /// Default: 5 samples minimum
    pub min_sample_size: usize,

    /// Enable Delta Debugging for noise filtering
    ///
    /// When enabled, filters out syscalls with high variance (noisy) before
    /// statistical testing. Reduces false positives from inherently variable
    /// syscalls (e.g., network I/O, random number generation).
    ///
    /// Scientific Foundation:
    /// \[7\] Zeller (2002): Delta Debugging minimizes differences between
    ///     failing/passing runs by isolating noise.
    ///
    /// Default: true
    pub enable_noise_filtering: bool,

    /// Noise threshold (coefficient of variation) for Delta Debugging
    ///
    /// Coefficient of Variation (CV) = `std_dev` / mean
    /// - CV > threshold: syscall is "noisy", filtered out before testing
    /// - CV <= threshold: syscall is "stable", included in testing
    ///
    /// Default: 0.5 (50% CV threshold)
    /// - CV = 0.2: baseline=`[10,12,8]` → std=2, mean=10, CV=0.2 (stable)
    /// - CV = 0.8: baseline=`[5,20,3]` → std=9, mean=9.3, CV=0.96 (noisy)
    pub noise_threshold: f64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,     // 95% confidence (standard in science)
            min_sample_size: 5,           // Reasonable for t-test reliability
            enable_noise_filtering: true, // Reduce false positives
            noise_threshold: 0.5,         // 50% CV threshold
        }
    }
}

impl RegressionConfig {
    /// Create a strict configuration (fewer false positives, more false negatives)
    ///
    /// Use when you want high confidence in detected regressions.
    pub fn strict() -> Self {
        Self {
            significance_level: 0.01, // 99% confidence
            min_sample_size: 10,
            enable_noise_filtering: true,
            noise_threshold: 0.3, // Stricter noise filtering (30% CV)
        }
    }

    /// Create a permissive configuration (more false positives, fewer false negatives)
    ///
    /// Use when you want to catch potential regressions early.
    pub fn permissive() -> Self {
        Self {
            significance_level: 0.10, // 90% confidence
            min_sample_size: 3,
            enable_noise_filtering: true,
            noise_threshold: 1.0, // Looser noise filtering (100% CV)
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.significance_level) {
            return Err(format!(
                "significance_level must be in [0, 1], got {}",
                self.significance_level
            ));
        }

        if self.min_sample_size < 2 {
            return Err(format!(
                "min_sample_size must be >= 2 for t-test, got {}",
                self.min_sample_size
            ));
        }

        if self.noise_threshold < 0.0 {
            return Err(format!(
                "noise_threshold must be non-negative, got {}",
                self.noise_threshold
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RegressionConfig::default();
        assert_eq!(config.significance_level, 0.05);
        assert_eq!(config.min_sample_size, 5);
        assert!(config.enable_noise_filtering);
        assert_eq!(config.noise_threshold, 0.5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_strict_config() {
        let config = RegressionConfig::strict();
        assert_eq!(config.significance_level, 0.01);
        assert_eq!(config.min_sample_size, 10);
        assert_eq!(config.noise_threshold, 0.3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_permissive_config() {
        let config = RegressionConfig::permissive();
        assert_eq!(config.significance_level, 0.10);
        assert_eq!(config.min_sample_size, 3);
        assert_eq!(config.noise_threshold, 1.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_invalid_significance_level() {
        let mut config = RegressionConfig::default();
        config.significance_level = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_invalid_min_sample_size() {
        let mut config = RegressionConfig::default();
        config.min_sample_size = 1;
        assert!(config.validate().is_err());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_invalid_noise_threshold() {
        let mut config = RegressionConfig::default();
        config.noise_threshold = -0.5;
        assert!(config.validate().is_err());
    }
}
