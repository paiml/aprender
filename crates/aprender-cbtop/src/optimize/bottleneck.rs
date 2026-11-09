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
            max_efficiency: efficiencies
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
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
        report.push_str(&format!(
            "**Configurations Analyzed**: {}\n",
            self.summary.total_configs
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(workload: &str, size: usize, gflops: f64, efficiency: f64, cv: f64) -> BaselineEntry {
        BaselineEntry {
            workload: workload.to_string(),
            size,
            backend: "simd".to_string(),
            gflops,
            efficiency,
            cv_percent: cv,
            score: 50,
        }
    }

    fn report(entries: Vec<BaselineEntry>) -> BaselineReport {
        BaselineReport {
            version: "test".to_string(),
            timestamp: "0".to_string(),
            entries,
            system: "test".to_string(),
        }
    }

    #[test]
    fn empty_baseline_has_zero_average_and_no_bottlenecks() {
        let a = OptimizationSuite::standard().analyze_bottlenecks(&report(vec![]));
        assert_eq!(a.summary.total_configs, 0);
        assert_eq!(a.summary.avg_efficiency, 0.0);
        assert!(a.summary.min_efficiency.is_infinite() && a.summary.min_efficiency > 0.0);
        assert!(a.critical.is_empty() && a.severe.is_empty() && a.moderate.is_empty());
    }

    #[test]
    fn efficiency_bands_split_at_25_50_75_percent() {
        let a = OptimizationSuite::standard().analyze_bottlenecks(&report(vec![
            entry("dot_product", 1000, 5.0, 0.24, 1.0),
            entry("dot_product", 1000, 5.0, 0.25, 1.0),
            entry("dot_product", 1000, 5.0, 0.49, 1.0),
            entry("dot_product", 1000, 5.0, 0.50, 1.0),
            entry("dot_product", 1000, 5.0, 0.74, 1.0),
            entry("dot_product", 1000, 5.0, 0.75, 1.0),
        ]));
        assert_eq!(
            (a.critical.len(), a.severe.len(), a.moderate.len()),
            (1, 2, 2)
        );
        assert_eq!(a.critical[0].severity, BottleneckSeverity::Critical);
        assert_eq!(a.severe[0].efficiency, 0.25);
        assert_eq!(a.moderate[1].efficiency, 0.74);
        let s = &a.summary;
        assert_eq!(
            (
                s.total_configs,
                s.critical_count,
                s.severe_count,
                s.moderate_count
            ),
            (6, 1, 2, 2)
        );
        assert_eq!((s.min_efficiency, s.max_efficiency), (0.24, 0.75));
        assert!((s.avg_efficiency - 0.495).abs() < 1e-12);
    }

    #[test]
    fn critical_recommendations_by_throughput_and_size() {
        let suite = OptimizationSuite::standard();
        let a = suite.analyze_bottlenecks(&report(vec![
            entry("elementwise_mul", 5_000_000, 0.5, 0.1, 1.0),
            entry("elementwise_mul", 5_000_000, 2.0, 0.1, 1.0),
            entry("dot_product", 5_000_000, 2.0, 0.1, 1.0),
        ]));
        assert!(a.critical[0]
            .recommendation
            .contains("Near-zero throughput"));
        assert!(a.critical[1].recommendation.contains("cache-aware tiling"));
        assert!(a.critical[2].recommendation.contains("perf record"));
    }

    #[test]
    fn severe_recommendations_memory_variance_and_default() {
        let a = OptimizationSuite::standard().analyze_bottlenecks(&report(vec![
            entry("elementwise_mul", 2_000_000, 3.0, 0.4, 1.0),
            entry("dot_product", 1000, 3.0, 0.4, 12.0),
            entry("dot_product", 1000, 3.0, 0.4, 1.0),
        ]));
        assert!(a.severe[0].recommendation.contains("cache-aware tiling"));
        assert!(a.severe[1].recommendation.contains("CV=12.0%"));
        assert!(a.severe[2].recommendation.contains("perf stat"));
    }

    #[test]
    fn moderate_recommendations_and_unknown_workload_is_not_memory_bound() {
        let a = OptimizationSuite::standard().analyze_bottlenecks(&report(vec![
            entry("elementwise_mul", 1000, 3.0, 0.6, 1.0),
            entry("dot_product", 1000, 3.0, 0.6, 1.0),
            entry("no_such_workload", 1000, 3.0, 0.6, 1.0),
        ]));
        assert!(a.moderate[0]
            .recommendation
            .contains("memory access pattern"));
        assert!(a.moderate[1].recommendation.contains("Near optimal"));
        assert!(a.moderate[2].recommendation.contains("Near optimal"));
    }

    #[test]
    fn unstable_flagged_above_15_percent_cv_even_when_efficient() {
        let a = OptimizationSuite::standard().analyze_bottlenecks(&report(vec![
            entry("dot_product", 1000, 9.0, 0.9, 15.0),
            entry("dot_product", 1000, 9.0, 0.9, 15.1),
        ]));
        assert_eq!(a.unstable.len(), 1);
        assert_eq!(a.summary.unstable_count, 1);
        assert_eq!(a.unstable[0].severity, BottleneckSeverity::Unstable);
        assert!(a.unstable[0].recommendation.contains("CV=15.1%"));
        let text = a.format_report();
        assert!(text.contains("## Unstable Operations"));
        assert!(!text.contains("All operations performing"));
    }

    #[test]
    fn format_report_clean_run_and_sections() {
        let clean = OptimizationSuite::standard().analyze_bottlenecks(&report(vec![entry(
            "dot_product",
            1000,
            9.0,
            0.9,
            1.0,
        )]));
        assert!(clean
            .format_report()
            .contains("All operations performing at >= 75%"));
        let bad = OptimizationSuite::standard().analyze_bottlenecks(&report(vec![
            entry("dot_product", 1000, 0.5, 0.1, 1.0),
            entry("dot_product", 1000, 3.0, 0.4, 1.0),
            entry("dot_product", 1000, 3.0, 0.6, 1.0),
        ]));
        let text = bad.format_report();
        for h in ["## Critical", "## Severe", "## Moderate"] {
            assert!(text.contains(h), "missing {h}");
        }
    }
}
