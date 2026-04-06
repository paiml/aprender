// Regression verdict assessment using statistical hypothesis testing
//
// This module integrates:
// - Statistical testing (aprender's t-tests)
// - Noise filtering (Zeller's Delta Debugging)
// - Dynamic thresholds (configuration-driven)
//
// to produce a final regression verdict with NO MAGIC NUMBERS.

use crate::regression::config::RegressionConfig;
use crate::regression::noise_filter::filter_noisy_syscalls;
use crate::regression::statistics::{compare_distributions, StatisticalTest};
use anyhow::Result;
use std::collections::HashMap;

/// Final regression verdict for a syscall trace comparison
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegressionVerdict {
    /// No statistically significant regression detected
    NoRegression,

    /// Statistically significant regression detected (p < `significance_level`)
    Regression {
        /// Syscall(s) that regressed
        regressed_syscalls: Vec<String>,
        /// Number of syscalls filtered as noisy
        filtered_count: usize,
    },

    /// Not enough data to make statistical determination
    InsufficientData { reason: String },
}

/// Detailed regression assessment result
#[derive(Debug, Clone)]
pub struct RegressionAssessment {
    /// Final verdict
    pub verdict: RegressionVerdict,

    /// Statistical tests performed (syscall → test result)
    pub tests: HashMap<String, StatisticalTest>,

    /// Syscalls filtered out as noisy
    pub filtered_syscalls: Vec<String>,

    /// Configuration used for assessment
    pub config: RegressionConfig,
}

impl RegressionAssessment {
    /// Generate human-readable report
    pub fn to_report_string(&self) -> String {
        let mut report = String::new();

        // Verdict header
        match &self.verdict {
            RegressionVerdict::NoRegression => {
                report.push_str("✅ NO REGRESSION DETECTED\n\n");
                report.push_str(&format!("Statistical tests performed: {}\n", self.tests.len()));
                report.push_str(&format!(
                    "Significance level: {} ({}% confidence)\n",
                    self.config.significance_level,
                    (1.0 - self.config.significance_level) * 100.0
                ));
            }
            RegressionVerdict::Regression { regressed_syscalls, filtered_count } => {
                report.push_str(&format!(
                    "❌ REGRESSION DETECTED ({} syscalls)\n\n",
                    regressed_syscalls.len()
                ));
                report
                    .push_str(&format!("Regressed syscalls: {}\n", regressed_syscalls.join(", ")));
                report.push_str(&format!("Filtered noisy syscalls: {filtered_count}\n"));
            }
            RegressionVerdict::InsufficientData { reason } => {
                report.push_str("⚠️  INSUFFICIENT DATA\n\n");
                report.push_str(&format!("Reason: {reason}\n"));
            }
        }

        // Filtered syscalls
        if !self.filtered_syscalls.is_empty() {
            report.push_str(&format!(
                "\n🔇 Filtered noisy syscalls ({}):\n",
                self.filtered_syscalls.len()
            ));
            for name in &self.filtered_syscalls {
                report.push_str(&format!("  - {name}\n"));
            }
        }

        // Statistical tests
        if !self.tests.is_empty() {
            report.push_str("\n📊 Statistical Tests:\n");
            for (name, test) in &self.tests {
                report.push_str(&format!(
                    "  {} (p={:.4}, baseline_median={:.1}, current_median={:.1})\n",
                    name, test.pvalue, test.baseline_median, test.current_median
                ));
            }
        }

        report
    }
}

/// Assess regression by comparing baseline and current syscall distributions
///
/// # Arguments
/// * `baseline` - Map of syscall name → baseline measurement samples
/// * `current` - Map of syscall name → current measurement samples
/// * `config` - Regression detection configuration
///
/// # Returns
/// `RegressionAssessment` with verdict, statistical tests, and filtered syscalls
///
/// # Example
/// ```
/// use renacer::regression::{assess_regression, RegressionConfig};
/// use std::collections::HashMap;
///
/// let mut baseline = HashMap::new();
/// baseline.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 12.0, 10.0]);
///
/// let mut current = HashMap::new();
/// current.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 13.0, 10.0]);
///
/// let assessment = assess_regression(&baseline, &current, &RegressionConfig::default()).expect("test");
/// // Should show no regression (distributions similar)
/// assert_eq!(assessment.verdict, renacer::regression::RegressionVerdict::NoRegression);
/// ```
pub fn assess_regression(
    baseline: &HashMap<String, Vec<f32>>,
    current: &HashMap<String, Vec<f32>>,
    config: &RegressionConfig,
) -> Result<RegressionAssessment> {
    // Validate configuration
    config.validate().map_err(|e| anyhow::anyhow!(e))?;

    // Step 1: Filter noisy syscalls (Delta Debugging)
    let (stable_baseline, filtered_syscalls) = if config.enable_noise_filtering {
        filter_noisy_syscalls(baseline, config.noise_threshold)
    } else {
        (baseline.clone(), Vec::new())
    };

    // Step 2: Run statistical tests on stable syscalls
    let mut tests = HashMap::new();
    let mut regressed_syscalls = Vec::new();

    for (name, baseline_samples) in &stable_baseline {
        // Check if syscall exists in current trace
        let Some(current_samples) = current.get(name) else {
            continue; // Syscall missing in current trace (separate analysis)
        };

        // Check minimum sample size
        if baseline_samples.len() < config.min_sample_size
            || current_samples.len() < config.min_sample_size
        {
            continue;
        }

        // Perform statistical test
        match compare_distributions(baseline_samples, current_samples) {
            Ok(test) => {
                // Check if regression is statistically significant
                if test.pvalue < config.significance_level as f32 {
                    // Regression detected: current significantly different from baseline
                    regressed_syscalls.push(name.clone());
                }
                tests.insert(name.clone(), test);
            }
            Err(e) => {
                tracing::warn!("Failed to compare distributions for {}: {}", name, e);
            }
        }
    }

    // Step 3: Determine verdict
    let verdict = if tests.is_empty() {
        RegressionVerdict::InsufficientData {
            reason: format!(
                "No syscalls passed noise filtering and sample size requirements \
                 (min_sample_size={}, filtered={})",
                config.min_sample_size,
                filtered_syscalls.len()
            ),
        }
    } else if regressed_syscalls.is_empty() {
        RegressionVerdict::NoRegression
    } else {
        RegressionVerdict::Regression {
            regressed_syscalls,
            filtered_count: filtered_syscalls.len(),
        }
    };

    Ok(RegressionAssessment { verdict, tests, filtered_syscalls, config: config.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assess_regression_no_regression() {
        let mut baseline = HashMap::new();
        baseline.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 12.0, 10.0]);

        let mut current = HashMap::new();
        current.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 13.0, 10.0]);

        let config = RegressionConfig::default();
        let assessment = assess_regression(&baseline, &current, &config).expect("test");

        assert_eq!(assessment.verdict, RegressionVerdict::NoRegression);
        assert_eq!(assessment.tests.len(), 1);
    }

    #[test]
    fn test_assess_regression_regression_detected() {
        let mut baseline = HashMap::new();
        baseline.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 12.0, 10.0]);

        let mut current = HashMap::new();
        current.insert("mmap".to_string(), vec![50.0, 52.0, 51.0, 53.0, 50.0]);

        let config = RegressionConfig::default();
        let assessment = assess_regression(&baseline, &current, &config).expect("test");

        match assessment.verdict {
            RegressionVerdict::Regression { ref regressed_syscalls, .. } => {
                assert!(regressed_syscalls.contains(&"mmap".to_string()));
            }
            _ => panic!("Expected Regression verdict"),
        }
    }

    #[test]
    fn test_assess_regression_filters_noisy() {
        let mut baseline = HashMap::new();
        baseline.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 12.0, 10.0]);
        baseline.insert("socket".to_string(), vec![5.0, 50.0, 3.0, 45.0, 2.0]); // Noisy!

        let mut current = HashMap::new();
        current.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 13.0, 10.0]);
        current.insert("socket".to_string(), vec![6.0, 51.0, 4.0, 46.0, 3.0]);

        let config = RegressionConfig::default();
        let assessment = assess_regression(&baseline, &current, &config).expect("test");

        // Should filter out socket as noisy
        assert!(assessment.filtered_syscalls.contains(&"socket".to_string()));

        // Should only test mmap
        assert_eq!(assessment.tests.len(), 1);
        assert!(assessment.tests.contains_key("mmap"));
    }

    #[test]
    fn test_assess_regression_insufficient_data() {
        let mut baseline = HashMap::new();
        baseline.insert("mmap".to_string(), vec![10.0]); // Only 1 sample

        let mut current = HashMap::new();
        current.insert("mmap".to_string(), vec![10.0]);

        let config = RegressionConfig::default();
        let assessment = assess_regression(&baseline, &current, &config).expect("test");

        match assessment.verdict {
            RegressionVerdict::InsufficientData { .. } => {
                // Expected
            }
            _ => panic!("Expected InsufficientData verdict"),
        }
    }

    #[test]
    fn test_report_string_no_regression() {
        let verdict = RegressionVerdict::NoRegression;
        let assessment = RegressionAssessment {
            verdict,
            tests: HashMap::new(),
            filtered_syscalls: vec![],
            config: RegressionConfig::default(),
        };

        let report = assessment.to_report_string();
        assert!(report.contains("NO REGRESSION DETECTED"));
    }

    #[test]
    fn test_report_string_regression() {
        let verdict = RegressionVerdict::Regression {
            regressed_syscalls: vec!["mmap".to_string()],
            filtered_count: 1,
        };
        let assessment = RegressionAssessment {
            verdict,
            tests: HashMap::new(),
            filtered_syscalls: vec!["socket".to_string()],
            config: RegressionConfig::default(),
        };

        let report = assessment.to_report_string();
        assert!(report.contains("REGRESSION DETECTED"));
        assert!(report.contains("mmap"));
        assert!(report.contains("socket"));
    }
}
