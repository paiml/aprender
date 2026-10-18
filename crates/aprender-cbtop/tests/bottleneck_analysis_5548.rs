//! Behaviour tests for `OptimizationSuite::analyze_bottlenecks` and
//! `BottleneckAnalysis::format_report` (#5548). Contract: cbtop-bottleneck-analysis-v1.

use cbtop::optimize::{
    BaselineEntry, BaselineReport, BottleneckSeverity, OptimizationSuite, WorkloadConfig,
};
use cbtop::WorkloadType;

fn workload(name: &str, memory_bound: bool) -> WorkloadConfig {
    WorkloadConfig {
        workload: WorkloadType::Gemm,
        name: name.into(),
        theoretical_peak_gflops: 100.0,
        memory_bound,
        bytes_per_flop: 8.0,
    }
}

fn entry(w: &str, size: usize, gflops: f64, eff: f64, cv: f64) -> BaselineEntry {
    BaselineEntry {
        workload: w.into(),
        size,
        backend: "simd".into(),
        gflops,
        efficiency: eff,
        cv_percent: cv,
        score: 0,
    }
}

fn report(entries: Vec<BaselineEntry>) -> BaselineReport {
    BaselineReport {
        version: "1".into(),
        timestamp: "t".into(),
        entries,
        system: "test".into(),
    }
}

fn suite() -> OptimizationSuite {
    let mut s = OptimizationSuite::standard();
    s.workloads = vec![workload("mem", true), workload("cpu", false)];
    s
}

#[test]
fn efficiency_thresholds_bucket_entries_and_boundaries_are_exclusive_below() {
    let a = suite().analyze_bottlenecks(&report(vec![
        entry("cpu", 1000, 5.0, 0.10, 1.0),
        entry("cpu", 1000, 5.0, 0.25, 1.0), // 0.25 is severe, not critical
        entry("cpu", 1000, 5.0, 0.49, 1.0),
        entry("cpu", 1000, 5.0, 0.50, 1.0), // moderate
        entry("cpu", 1000, 5.0, 0.74, 1.0),
        entry("cpu", 1000, 5.0, 0.75, 1.0), // healthy
    ]));
    assert_eq!(
        (
            a.critical.len(),
            a.severe.len(),
            a.moderate.len(),
            a.unstable.len()
        ),
        (1, 2, 2, 0)
    );
    assert_eq!(a.critical[0].severity, BottleneckSeverity::Critical);
    assert_eq!(a.summary.total_configs, 6);
    assert_eq!(a.summary.critical_count, 1);
    assert!((a.summary.min_efficiency - 0.10).abs() < 1e-12);
    assert!((a.summary.max_efficiency - 0.75).abs() < 1e-12);
    assert!((a.summary.avg_efficiency - 2.83 / 6.0).abs() < 1e-12);
}

#[test]
fn recommendations_follow_severity_size_and_memory_bound() {
    let a = suite().analyze_bottlenecks(&report(vec![
        entry("cpu", 10, 0.5, 0.10, 1.0), // critical, near-zero throughput
        entry("mem", 5_000_000, 5.0, 0.10, 1.0), // critical, memory bound, very large
        entry("cpu", 10, 5.0, 0.10, 1.0), // critical, generic
        entry("mem", 2_000_000, 5.0, 0.30, 1.0), // severe, memory bound, large
        entry("cpu", 10, 5.0, 0.30, 12.0), // severe, noisy
        entry("cpu", 10, 5.0, 0.30, 2.0), // severe, generic
        entry("mem", 10, 5.0, 0.60, 1.0), // moderate, memory bound
        entry("cpu", 10, 5.0, 0.60, 1.0), // moderate, compute bound
    ]));
    let c = &a.critical;
    assert!(c[0].recommendation.contains("Near-zero throughput"));
    assert!(c[1].recommendation.contains("cache-aware tiling"));
    assert!(c[2].recommendation.contains("perf record"));
    let s = &a.severe;
    assert!(s[0].recommendation.contains("cache-aware tiling"));
    assert!(s[1].recommendation.contains("CV=12.0%"));
    assert!(s[2].recommendation.contains("perf stat"));
    let m = &a.moderate;
    assert!(m[0].recommendation.contains("memory access pattern"));
    assert!(m[1].recommendation.contains("Near optimal"));
}

#[test]
fn unknown_workload_is_treated_as_not_memory_bound_and_cv_over_15_is_unstable() {
    let a = suite().analyze_bottlenecks(&report(vec![
        entry("ghost", 5_000_000, 5.0, 0.10, 16.0),
        entry("cpu", 10, 5.0, 0.90, 15.0), // exactly 15 is stable
    ]));
    assert!(a.critical[0].recommendation.contains("perf record"));
    assert_eq!(a.unstable.len(), 1);
    assert_eq!(a.unstable[0].severity, BottleneckSeverity::Unstable);
    assert!(a.unstable[0].recommendation.contains("CV=16.0%"));
}

#[test]
fn empty_baseline_gives_zero_average() {
    let a = suite().analyze_bottlenecks(&report(vec![]));
    assert_eq!(a.summary.total_configs, 0);
    assert_eq!(a.summary.avg_efficiency, 0.0);
    assert!(a.summary.min_efficiency.is_infinite() && a.summary.min_efficiency > 0.0);
}

#[test]
fn report_lists_each_section_or_the_all_clear() {
    let clean = suite().analyze_bottlenecks(&report(vec![entry("cpu", 10, 9.0, 0.95, 1.0)]));
    let r = clean.format_report();
    assert!(r.contains("**Configurations Analyzed**: 1"));
    assert!(r.contains("**Average Efficiency**: 95.0%"));
    assert!(r.contains("All operations performing at >= 75% efficiency"));

    let bad = suite().analyze_bottlenecks(&report(vec![
        entry("cpu", 10, 5.0, 0.10, 20.0),
        entry("cpu", 20, 5.0, 0.30, 1.0),
        entry("cpu", 30, 5.0, 0.60, 1.0),
    ]));
    let r = bad.format_report();
    for h in [
        "## Critical Bottlenecks (< 25% efficiency)",
        "## Severe Bottlenecks (< 50% efficiency)",
        "## Moderate Bottlenecks (< 75% efficiency)",
        "## Unstable Operations (CV > 15%)",
    ] {
        assert!(r.contains(h), "{h}");
    }
    assert!(r.contains("- **cpu** @ 20 elements: 30.0% efficiency (5.0 GFLOP/s)"));
    assert!(!r.contains("All operations performing"));
}
