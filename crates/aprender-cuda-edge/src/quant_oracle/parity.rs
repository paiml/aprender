//! Quantization parity checking.
//!
//! Compares CPU and GPU quantization/dequantization results to ensure
//! they produce values within acceptable tolerance.

use serde::{Deserialize, Serialize};

use super::boundary::QuantFormat;

/// Configuration for parity checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityConfig {
    /// Quantization format being tested.
    pub format: QuantFormat,
    /// Custom tolerance override (uses format default if None).
    pub tolerance_override: Option<f64>,
}

impl ParityConfig {
    /// Create a parity config for the given format with default tolerance.
    #[must_use]
    pub fn new(format: QuantFormat) -> Self {
        Self {
            format,
            tolerance_override: None,
        }
    }

    /// Get the effective tolerance for this config.
    #[must_use]
    pub fn tolerance(&self) -> f64 {
        self.tolerance_override
            .unwrap_or_else(|| self.format.tolerance())
    }
}

/// A single parity violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityViolation {
    /// Index of the violating element.
    pub index: usize,
    /// CPU-computed value.
    pub cpu_value: f64,
    /// GPU-computed value.
    pub gpu_value: f64,
    /// Absolute difference.
    pub abs_diff: f64,
    /// Tolerance that was exceeded.
    pub tolerance: f64,
}

/// Report from a parity check session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParityReport {
    /// All detected violations.
    pub violations: Vec<ParityViolation>,
    /// Total elements compared.
    pub total_elements: usize,
    /// Maximum absolute difference observed.
    pub max_abs_diff: f64,
    /// Mean absolute difference.
    pub mean_abs_diff: f64,
}

impl ParityReport {
    /// Returns true if parity check passed (no violations).
    #[must_use]
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }

    /// Violation rate (violations / total elements).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn violation_rate(&self) -> f64 {
        if self.total_elements == 0 {
            return 0.0;
        }
        self.violations.len() as f64 / self.total_elements as f64
    }
}

/// Check element-wise parity between CPU and GPU values.
///
/// Returns a report with any violations where `|cpu - gpu| > tolerance`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn check_values_parity(
    cpu_values: &[f64],
    gpu_values: &[f64],
    config: &ParityConfig,
) -> ParityReport {
    let tolerance = config.tolerance();
    let mut violations = Vec::new();
    let mut max_abs_diff = 0.0_f64;
    let mut sum_abs_diff = 0.0_f64;

    let len = cpu_values.len().min(gpu_values.len());

    for i in 0..len {
        let cpu = cpu_values[i];
        let gpu = gpu_values[i];

        // Handle NaN specially
        if cpu.is_nan() && gpu.is_nan() {
            continue;
        }
        if cpu.is_nan() || gpu.is_nan() {
            violations.push(ParityViolation {
                index: i,
                cpu_value: cpu,
                gpu_value: gpu,
                abs_diff: f64::NAN,
                tolerance,
            });
            continue;
        }

        let abs_diff = (cpu - gpu).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        sum_abs_diff += abs_diff;

        if abs_diff > tolerance {
            violations.push(ParityViolation {
                index: i,
                cpu_value: cpu,
                gpu_value: gpu,
                abs_diff,
                tolerance,
            });
        }
    }

    let mean_abs_diff = if len > 0 {
        sum_abs_diff / len as f64
    } else {
        0.0
    };

    ParityReport {
        violations,
        total_elements: len,
        max_abs_diff,
        mean_abs_diff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_values_pass() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.0, 3.0];
        let config = ParityConfig::new(QuantFormat::F32);
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(report.passed());
    }

    #[test]
    fn small_diff_within_tolerance_passes() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.001, 2.001, 3.001];
        let config = ParityConfig::new(QuantFormat::Q4K); // tolerance 0.05
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(report.passed());
    }

    #[test]
    fn large_diff_fails() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.5, 3.0]; // 0.5 diff at index 1
        let config = ParityConfig::new(QuantFormat::Q4K);
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(!report.passed());
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].index, 1);
    }

    #[test]
    fn nan_vs_nan_is_ok() {
        let cpu = vec![f64::NAN];
        let gpu = vec![f64::NAN];
        let config = ParityConfig::new(QuantFormat::F32);
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(report.passed());
    }

    #[test]
    fn nan_vs_number_is_violation() {
        let cpu = vec![f64::NAN];
        let gpu = vec![1.0];
        let config = ParityConfig::new(QuantFormat::F32);
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(!report.passed());
    }

    #[test]
    fn violation_rate_computed() {
        let cpu = vec![1.0, 2.0, 3.0, 4.0];
        let gpu = vec![1.0, 3.0, 3.0, 5.0]; // 2 violations
        let config = ParityConfig::new(QuantFormat::Q4K);
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!((report.violation_rate() - 0.5).abs() < 0.01);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn identical_always_passes(values in proptest::collection::vec(-1e6f64..1e6, 1..100)) {
            let config = ParityConfig::new(QuantFormat::F32);
            let report = check_values_parity(&values, &values, &config);
            prop_assert!(report.passed());
        }
    }
}
