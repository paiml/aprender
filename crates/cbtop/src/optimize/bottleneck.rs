//! Bottleneck analysis for performance identification (OPT-002).

use serde::{Deserialize, Serialize};

use super::suite::{BaselineEntry, BaselineReport, OptimizationSuite, WorkloadConfig};

/// Entry describing a performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckEntry {
    /// Workload name
    pub workload: String,
    /// Problem size where bottleneck occurs
    pub size: usize,
    /// Achieved efficiency (0.0 - 1.0)
    pub efficiency: f64,
    /// Achieved GFLOP/s
    pub gflops: f64,
    /// Recommendation for improvement
    pub recommendation: String,
    /// Severity level
    pub severity: BottleneckSeverity,
}

/// Severity level of a bottleneck
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Critical: < 25% efficiency
    Critical,
    /// Severe: < 50% efficiency
    Severe,
    /// Moderate: < 75% efficiency
    Moderate,
    /// Unstable: High CV (> 15%)
    Unstable,
}

/// Results of bottleneck analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Critical bottlenecks (< 25% efficiency)
    pub critical: Vec<BottleneckEntry>,
    /// Severe bottlenecks (< 50% efficiency)
    pub severe: Vec<BottleneckEntry>,
    /// Moderate bottlenecks (< 75% efficiency)
    pub moderate: Vec<BottleneckEntry>,
    /// Unstable operations (CV > 15%)
    pub unstable: Vec<BottleneckEntry>,
    /// Summary statistics
    pub summary: AnalysisSummary,
}

/// Summary statistics for the analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisSummary {
    /// Total configurations analyzed
    pub total_configs: usize,
    /// Number of critical bottlenecks
    pub critical_count: usize,
    /// Number of severe bottlenecks
    pub severe_count: usize,
    /// Number of moderate bottlenecks
    pub moderate_count: usize,
    /// Number of unstable operations
    pub unstable_count: usize,
    /// Average efficiency across all configs
    pub avg_efficiency: f64,
    /// Worst efficiency found
    pub min_efficiency: f64,
    /// Best efficiency found
    pub max_efficiency: f64,
}

impl OptimizationSuite {
    /// Analyze baseline for bottlenecks
    pub fn analyze_bottlenecks(&self, baseline: &BaselineReport) -> BottleneckAnalysis {
        let mut analysis = BottleneckAnalysis::default();
        let mut efficiencies = Vec::new();

        for entry in &baseline.entries {
            let workload = self.workloads.iter().find(|w| w.name == entry.workload);

            let efficiency = entry.efficiency;
            efficiencies.push(efficiency);

            // Check efficiency thresholds
            if efficiency < 0.25 {
                analysis.critical.push(BottleneckEntry {
                    workload: entry.workload.clone(),
                    size: entry.size,
                    efficiency,
                    gflops: entry.gflops,
                    recommendation: Self::recommend_optimization(
                        workload,
                        entry,
                        BottleneckSeverity::Critical,
                    ),
                    severity: BottleneckSeverity::Critical,
                });
            } else if efficiency < 0.50 {
                analysis.severe.push(BottleneckEntry {
                    workload: entry.workload.clone(),
                    size: entry.size,
                    efficiency,
                    gflops: entry.gflops,
                    recommendation: Self::recommend_optimization(
                        workload,
                        entry,
                        BottleneckSeverity::Severe,
                    ),
                    severity: BottleneckSeverity::Severe,
                });
            } else if efficiency < 0.75 {
                analysis.moderate.push(BottleneckEntry {
                    workload: entry.workload.clone(),
                    size: entry.size,
                    efficiency,
                    gflops: entry.gflops,
                    recommendation: Self::recommend_optimization(
                        workload,
                        entry,
                        BottleneckSeverity::Moderate,
                    ),
                    severity: BottleneckSeverity::Moderate,
                });
            }

            // Check stability (CV > 15%)
            if entry.cv_percent > 15.0 {
                analysis.unstable.push(BottleneckEntry {
                    workload: entry.workload.clone(),
                    size: entry.size,
                    efficiency,
                    gflops: entry.gflops,
                    recommendation: format!(
                        "High variance (CV={:.1}%) - check CPU governor with PERF-003 pattern, \
                         or reduce system load during benchmarks",
                        entry.cv_percent
                    ),
                    severity: BottleneckSeverity::Unstable,
                });
            }
        }

        // Calculate summary
        analysis.summary = AnalysisSummary {
            total_configs: baseline.entries.len(),
            critical_count: analysis.critical.len(),
            severe_count: analysis.severe.len(),
            moderate_count: analysis.moderate.len(),
            unstable_count: analysis.unstable.len(),
            avg_efficiency: if efficiencies.is_empty() {
                0.0
            } else {
                efficiencies.iter().sum::<f64>() / efficiencies.len() as f64
            },
            min_efficiency: efficiencies.iter().cloned().fold(f64::INFINITY, f64::min),
            max_efficiency: efficiencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        };

        analysis
    }

    fn recommend_optimization(
        workload: Option<&WorkloadConfig>,
        entry: &BaselineEntry,
        severity: BottleneckSeverity,
    ) -> String {
        let is_memory_bound = workload.map(|w| w.memory_bound).unwrap_or(false);
        let is_large = entry.size > 1_000_000;
        let is_very_large = entry.size > 4_000_000;

        match severity {
            BottleneckSeverity::Critical => {
                if entry.gflops < 1.0 {
                    "Critical: Near-zero throughput - verify SIMD codegen with `cargo asm`, \
                     check for scalar fallback"
                        .to_string()
                } else if is_memory_bound && is_very_large {
                    "Critical: Memory bandwidth limited at large size - implement cache-aware \
                     tiling (PERF-001 pattern), consider prefetching"
                        .to_string()
                } else {
                    "Critical: Profile with `perf record` or `renacer` to identify hotspot, \
                     check for branch mispredictions"
                        .to_string()
                }
            }
            BottleneckSeverity::Severe => {
                if is_memory_bound && is_large {
                    "Consider cache-aware tiling (PERF-001 pattern) for large memory-bound \
                     operations"
                        .to_string()
                } else if entry.cv_percent > 10.0 {
                    format!(
                        "High variance (CV={:.1}%) - set CPU governor to 'performance' \
                         (PERF-003 pattern)",
                        entry.cv_percent
                    )
                } else {
                    "Profile with `perf stat` to check IPC and cache misses".to_string()
                }
            }
            BottleneckSeverity::Moderate => {
                if is_memory_bound {
                    "Consider memory access pattern optimization (coalescing, prefetching)"
                        .to_string()
                } else {
                    "Near optimal - minor gains possible with micro-optimizations".to_string()
                }
            }
            BottleneckSeverity::Unstable => {
                "Reduce measurement variance before optimizing".to_string()
            }
        }
    }
}

impl BottleneckAnalysis {
    /// Format analysis as human-readable report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Bottleneck Analysis Report\n\n");
        report.push_str(&format!("**Configurations Analyzed**: {}\n", self.summary.total_configs));
        report.push_str(&format!(
            "**Average Efficiency**: {:.1}%\n",
            self.summary.avg_efficiency * 100.0
        ));
        report.push_str(&format!(
            "**Efficiency Range**: {:.1}% - {:.1}%\n\n",
            self.summary.min_efficiency * 100.0,
            self.summary.max_efficiency * 100.0
        ));

        if !self.critical.is_empty() {
            report.push_str("## Critical Bottlenecks (< 25% efficiency)\n\n");
            for b in &self.critical {
                report.push_str(&format!(
                    "- **{}** @ {} elements: {:.1}% efficiency ({:.1} GFLOP/s)\n  - {}\n\n",
                    b.workload,
                    b.size,
                    b.efficiency * 100.0,
                    b.gflops,
                    b.recommendation
                ));
            }
        }

        if !self.severe.is_empty() {
            report.push_str("## Severe Bottlenecks (< 50% efficiency)\n\n");
            for b in &self.severe {
                report.push_str(&format!(
                    "- **{}** @ {} elements: {:.1}% efficiency ({:.1} GFLOP/s)\n  - {}\n\n",
                    b.workload,
                    b.size,
                    b.efficiency * 100.0,
                    b.gflops,
                    b.recommendation
                ));
            }
        }

        if !self.moderate.is_empty() {
            report.push_str("## Moderate Bottlenecks (< 75% efficiency)\n\n");
            for b in &self.moderate {
                report.push_str(&format!(
                    "- **{}** @ {} elements: {:.1}% efficiency ({:.1} GFLOP/s)\n  - {}\n\n",
                    b.workload,
                    b.size,
                    b.efficiency * 100.0,
                    b.gflops,
                    b.recommendation
                ));
            }
        }

        if !self.unstable.is_empty() {
            report.push_str("## Unstable Operations (CV > 15%)\n\n");
            for b in &self.unstable {
                report.push_str(&format!(
                    "- **{}** @ {} elements: {}\n\n",
                    b.workload, b.size, b.recommendation
                ));
            }
        }

        if self.critical.is_empty()
            && self.severe.is_empty()
            && self.moderate.is_empty()
            && self.unstable.is_empty()
        {
            report.push_str(
                "**All operations performing at >= 75% efficiency with stable measurements.**\n",
            );
        }

        report
    }
}
