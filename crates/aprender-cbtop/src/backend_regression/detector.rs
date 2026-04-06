//! Backend comparison, cliff detection, and recommendation logic.

use super::analysis::{BackendSummary, TransferAnalysis};
use super::types::{Backend, BackendComparison, BackendRecommendation, SizeCliff, WorkloadType};
use super::BackendRegressionDetector;

impl BackendRegressionDetector {
    /// Compare two backends for a specific workload/size
    pub fn compare_backends(
        &self,
        baseline: Backend,
        comparison: Backend,
        workload: WorkloadType,
        size: usize,
    ) -> Option<BackendComparison> {
        let baseline_m = self.find_measurement(baseline, workload, size)?;
        let comparison_m = self.find_measurement(comparison, workload, size)?;

        let efficiency_ratio = if baseline_m.efficiency_percent > 0.0 {
            comparison_m.efficiency_percent / baseline_m.efficiency_percent
        } else {
            0.0
        };

        let speedup = if comparison_m.latency_us > 0.0 {
            baseline_m.latency_us / comparison_m.latency_us
        } else {
            0.0
        };

        let is_regression = speedup < (1.0 - self.threshold_percent() / 100.0);

        Some(BackendComparison {
            baseline,
            comparison,
            workload,
            size,
            efficiency_ratio,
            speedup,
            is_regression,
            threshold: self.threshold_percent(),
        })
    }

    /// Detect size cliffs for a backend
    pub fn detect_size_cliffs(&self, backend: Backend, workload: WorkloadType) -> Vec<SizeCliff> {
        let mut measurements: Vec<_> = self
            .measurements()
            .iter()
            .filter(|m| m.backend == backend && m.workload == workload)
            .collect();

        measurements.sort_by_key(|m| m.size);

        let mut cliffs = Vec::new();

        for window in measurements.windows(2) {
            let before = &window[0];
            let after = &window[1];

            if before.efficiency_percent > 0.0 {
                let drop = (before.efficiency_percent - after.efficiency_percent)
                    / before.efficiency_percent
                    * 100.0;

                if drop > self.cliff_threshold_percent() {
                    cliffs.push(SizeCliff {
                        backend,
                        workload,
                        size_before: before.size,
                        size_after: after.size,
                        efficiency_before: before.efficiency_percent,
                        efficiency_after: after.efficiency_percent,
                        drop_percent: drop,
                    });
                }
            }
        }

        cliffs
    }

    /// Analyze GPU transfer overhead
    pub fn analyze_transfer_overhead(
        &self,
        backend: Backend,
        workload: WorkloadType,
    ) -> Option<TransferAnalysis> {
        if !backend.is_gpu() {
            return None;
        }

        let measurements: Vec<_> = self
            .measurements()
            .iter()
            .filter(|m| {
                m.backend == backend
                    && m.workload == workload
                    && m.transfer_time_us.is_some()
                    && m.compute_time_us.is_some()
            })
            .collect();

        if measurements.is_empty() {
            return None;
        }

        let mut total_transfer = 0.0;
        let mut total_compute = 0.0;
        let mut sizes_with_overhead = Vec::new();

        for m in &measurements {
            let transfer = m
                .transfer_time_us
                .expect("transfer_time_us MUST be set for GPU measurements");
            let compute = m
                .compute_time_us
                .expect("compute_time_us MUST be set for GPU measurements");
            total_transfer += transfer;
            total_compute += compute;

            let overhead = transfer / (transfer + compute);
            if overhead > 0.5 {
                sizes_with_overhead.push((m.size, overhead));
            }
        }

        let avg_overhead = total_transfer / (total_transfer + total_compute);

        Some(TransferAnalysis {
            backend,
            workload,
            average_overhead: avg_overhead,
            total_transfer_time_us: total_transfer,
            total_compute_time_us: total_compute,
            sizes_dominated_by_transfer: sizes_with_overhead,
        })
    }

    /// Recommend best backend for given workload/size
    pub fn recommend_backend(
        &self,
        workload: WorkloadType,
        size: usize,
    ) -> Option<BackendRecommendation> {
        let candidates: Vec<_> = self
            .measurements()
            .iter()
            .filter(|m| m.workload == workload && m.size == size)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let best = candidates.iter().max_by(|a, b| {
            a.throughput
                .partial_cmp(&b.throughput)
                .expect("throughput MUST be comparable (no NaN)")
        })?;

        let confidence = (best.efficiency_percent / 100.0).clamp(0.0, 1.0);

        let reason = if best.backend.is_gpu() {
            if let Some(overhead) = best.transfer_overhead() {
                if overhead > 0.3 {
                    format!(
                        "GPU selected but transfer overhead is {:.1}%",
                        overhead * 100.0
                    )
                } else {
                    "Best throughput with low transfer overhead".to_string()
                }
            } else {
                "Best throughput among available backends".to_string()
            }
        } else {
            "Best CPU backend for this size".to_string()
        };

        Some(BackendRecommendation {
            backend: best.backend,
            workload,
            size,
            expected_efficiency: best.efficiency_percent,
            confidence,
            reason,
        })
    }

    /// Get all comparisons for a workload
    pub fn compare_all_backends(&self, workload: WorkloadType) -> Vec<BackendComparison> {
        let sizes = self.unique_for(workload, |m| m.size);
        let backends = self.unique_for(workload, |m| m.backend);

        let mut comparisons = Vec::new();

        for size in &sizes {
            if let Some(scalar) = backends.iter().find(|b| **b == Backend::Scalar) {
                for backend in &backends {
                    if *backend != Backend::Scalar {
                        if let Some(cmp) = self.compare_backends(*scalar, *backend, workload, *size)
                        {
                            comparisons.push(cmp);
                        }
                    }
                }
            }
        }

        comparisons
    }

    /// Detect all regressions
    pub fn detect_regressions(&self) -> Vec<BackendComparison> {
        self.unique(|m| m.workload)
            .into_iter()
            .flat_map(|w| self.compare_all_backends(w))
            .filter(|cmp| cmp.is_regression)
            .collect()
    }

    /// Generate summary report
    pub fn summary(&self) -> BackendSummary {
        let workloads = self.unique(|m| m.workload);
        let backends = self.unique(|m| m.backend);
        let regressions = self.detect_regressions();

        let all_cliffs: Vec<_> = backends
            .iter()
            .flat_map(|b| {
                workloads
                    .iter()
                    .flat_map(move |w| self.detect_size_cliffs(*b, *w))
            })
            .collect();

        BackendSummary {
            measurement_count: self.measurements().len(),
            backend_count: backends.len(),
            workload_count: workloads.len(),
            regression_count: regressions.len(),
            cliff_count: all_cliffs.len(),
            regressions,
            cliffs: all_cliffs,
        }
    }
}
