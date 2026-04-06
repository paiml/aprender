//! Transfer analysis and summary types for backend regression detection.

use super::types::{Backend, BackendComparison, SizeCliff, WorkloadType};

/// GPU transfer overhead analysis
#[derive(Debug, Clone)]
pub struct TransferAnalysis {
    /// Backend analyzed
    pub backend: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Average transfer overhead (0.0 - 1.0)
    pub average_overhead: f64,
    /// Total transfer time
    pub total_transfer_time_us: f64,
    /// Total compute time
    pub total_compute_time_us: f64,
    /// Sizes where transfer dominates (>50% overhead)
    pub sizes_dominated_by_transfer: Vec<(usize, f64)>,
}

impl TransferAnalysis {
    /// Check if transfer dominates compute
    pub fn transfer_dominated(&self) -> bool {
        self.average_overhead > 0.5
    }

    /// Get summary
    pub fn summary(&self) -> String {
        if self.transfer_dominated() {
            format!(
                "{} {}: Transfer overhead {:.1}% - consider larger batches",
                self.backend.name(),
                self.workload.name(),
                self.average_overhead * 100.0
            )
        } else {
            format!(
                "{} {}: Transfer overhead {:.1}% - GPU efficient",
                self.backend.name(),
                self.workload.name(),
                self.average_overhead * 100.0
            )
        }
    }
}

/// Summary of backend regression analysis
#[derive(Debug, Clone)]
pub struct BackendSummary {
    /// Total measurements
    pub measurement_count: usize,
    /// Number of backends tested
    pub backend_count: usize,
    /// Number of workloads tested
    pub workload_count: usize,
    /// Number of regressions detected
    pub regression_count: usize,
    /// Number of size cliffs detected
    pub cliff_count: usize,
    /// All regressions
    pub regressions: Vec<BackendComparison>,
    /// All cliffs
    pub cliffs: Vec<SizeCliff>,
}

impl BackendSummary {
    /// Check if any regressions detected
    pub fn has_regressions(&self) -> bool {
        self.regression_count > 0
    }

    /// Check if any cliffs detected
    pub fn has_cliffs(&self) -> bool {
        self.cliff_count > 0
    }

    /// Get status message
    pub fn status(&self) -> &'static str {
        if self.regression_count > 0 {
            "FAIL: Regressions detected"
        } else if self.cliff_count > 0 {
            "WARN: Size cliffs detected"
        } else {
            "PASS: No issues detected"
        }
    }
}
