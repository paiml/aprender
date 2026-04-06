//! Regression detection for CI/CD integration (OPT-003).

use crate::error::CbtopError;
use serde::{Deserialize, Serialize};

use super::suite::{BaselineEntry, BaselineReport};

/// Entry describing a performance regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionEntry {
    /// Workload name
    pub workload: String,
    /// Problem size
    pub size: usize,
    /// Baseline GFLOP/s
    pub baseline_gflops: f64,
    /// Current GFLOP/s
    pub current_gflops: f64,
    /// Change percentage (negative = regression)
    pub change_percent: f64,
}

/// Results of regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Whether all checks passed (no regressions)
    pub passed: bool,
    /// Detected regressions
    pub regressions: Vec<RegressionEntry>,
    /// Detected improvements
    pub improvements: Vec<RegressionEntry>,
    /// Summary message
    pub summary: String,
}

/// Automated regression detection for CI/CD integration
pub struct RegressionDetector {
    /// Baseline to compare against
    baseline: BaselineReport,
    /// Threshold for regression detection (%)
    threshold_percent: f64,
}

impl RegressionDetector {
    /// Create new regression detector
    pub fn new(baseline: BaselineReport, threshold_percent: f64) -> Self {
        Self {
            baseline,
            threshold_percent,
        }
    }

    /// Load baseline from file and create detector
    pub fn from_file(path: &std::path::Path, threshold_percent: f64) -> Result<Self, CbtopError> {
        let baseline = BaselineReport::load(path)?;
        Ok(Self::new(baseline, threshold_percent))
    }

    /// Check current results against baseline
    pub fn check(&self, current: &BaselineReport) -> RegressionReport {
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        for current_entry in &current.entries {
            if let Some(baseline_entry) = self.find_baseline(current_entry) {
                if baseline_entry.gflops > 0.0 {
                    let change = (current_entry.gflops - baseline_entry.gflops)
                        / baseline_entry.gflops
                        * 100.0;

                    if change < -self.threshold_percent {
                        regressions.push(RegressionEntry {
                            workload: current_entry.workload.clone(),
                            size: current_entry.size,
                            baseline_gflops: baseline_entry.gflops,
                            current_gflops: current_entry.gflops,
                            change_percent: change,
                        });
                    } else if change > self.threshold_percent {
                        improvements.push(RegressionEntry {
                            workload: current_entry.workload.clone(),
                            size: current_entry.size,
                            baseline_gflops: baseline_entry.gflops,
                            current_gflops: current_entry.gflops,
                            change_percent: change,
                        });
                    }
                }
            }
        }

        let passed = regressions.is_empty();
        let summary = if passed {
            if improvements.is_empty() {
                "All benchmarks within threshold".to_string()
            } else {
                format!("{} improvements detected", improvements.len())
            }
        } else {
            format!(
                "FAILED: {} regressions detected (threshold: {}%)",
                regressions.len(),
                self.threshold_percent
            )
        };

        RegressionReport {
            passed,
            regressions,
            improvements,
            summary,
        }
    }

    fn find_baseline(&self, current: &BaselineEntry) -> Option<&BaselineEntry> {
        self.baseline.entries.iter().find(|b| {
            b.workload == current.workload && b.size == current.size && b.backend == current.backend
        })
    }
}

impl RegressionReport {
    /// Get exit code for CI (0 = pass, 1 = regression)
    pub fn exit_code(&self) -> i32 {
        if self.passed {
            0
        } else {
            1
        }
    }

    /// Format as human-readable report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Regression Check Report\n\n");
        report.push_str(&format!("**Status**: {}\n\n", self.summary));

        if !self.regressions.is_empty() {
            report.push_str("## Regressions\n\n");
            for r in &self.regressions {
                report.push_str(&format!(
                    "- **{}** @ {}: {:.1} -> {:.1} GFLOP/s ({:.1}%)\n",
                    r.workload, r.size, r.baseline_gflops, r.current_gflops, r.change_percent
                ));
            }
            report.push('\n');
        }

        if !self.improvements.is_empty() {
            report.push_str("## Improvements\n\n");
            for i in &self.improvements {
                report.push_str(&format!(
                    "- **{}** @ {}: {:.1} -> {:.1} GFLOP/s (+{:.1}%)\n",
                    i.workload, i.size, i.baseline_gflops, i.current_gflops, i.change_percent
                ));
            }
        }

        report
    }
}
