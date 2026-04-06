//! Tests for benchmark module

use super::*;
use crate::efficiency::{
    ComputeDevice, CostMetrics, CpuInfo, EnergyMetrics, ModelParadigm, SimdCapability,
};

fn test_device() -> ComputeDevice {
    ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Test CPU"))
}

fn test_entry(run_id: &str, quality: f64, cost_usd: f64, joules: f64) -> BenchmarkEntry {
    BenchmarkEntry::new(
        run_id,
        ModelParadigm::DeepLearning,
        test_device(),
        quality,
        CostMetrics::new(cost_usd, 1.0, 1000, 10),
        EnergyMetrics::new(100.0, joules, 1000),
    )
}

#[test]
fn test_benchmark_entry_new() {
    let entry = test_entry("run-001", 0.95, 10.0, 1000.0);

    assert_eq!(entry.run_id, "run-001");
    assert!((entry.quality_score - 0.95).abs() < f64::EPSILON);
}

#[test]
fn test_benchmark_entry_efficiency_score() {
    let entry = test_entry("run-001", 0.90, 10.0, 1000.0);
    assert!((entry.efficiency_score() - 0.09).abs() < 0.01);
}

#[test]
fn test_benchmark_entry_dominates() {
    let a = test_entry("a", 0.95, 10.0, 1000.0);
    let b = test_entry("b", 0.90, 15.0, 1500.0);

    assert!(a.dominates(&b)); // Better quality, lower cost, lower energy
    assert!(!b.dominates(&a));
}

#[test]
fn test_benchmark_entry_no_domination() {
    let a = test_entry("a", 0.95, 20.0, 1000.0); // Higher quality, higher cost
    let b = test_entry("b", 0.90, 10.0, 1500.0); // Lower quality, lower cost

    assert!(!a.dominates(&b));
    assert!(!b.dominates(&a));
}

#[test]
fn test_benchmark_new() {
    let benchmark = CostPerformanceBenchmark::new();
    assert!(benchmark.is_empty());
    assert_eq!(benchmark.len(), 0);
}

#[test]
fn test_benchmark_add() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("run-001", 0.95, 10.0, 1000.0));

    assert_eq!(benchmark.len(), 1);
    assert!(!benchmark.is_empty());
}

#[test]
fn test_benchmark_pareto_frontier() {
    let mut benchmark = CostPerformanceBenchmark::new();

    // Entry A: High quality, high cost
    benchmark.add(test_entry("a", 0.98, 50.0, 5000.0));
    // Entry B: Medium quality, medium cost (Pareto optimal)
    benchmark.add(test_entry("b", 0.92, 20.0, 2000.0));
    // Entry C: Low quality, low cost (Pareto optimal)
    benchmark.add(test_entry("c", 0.85, 5.0, 500.0));
    // Entry D: Dominated by B (worse quality, same cost)
    benchmark.add(test_entry("d", 0.88, 20.0, 2000.0));

    let frontier = benchmark.pareto_frontier();
    assert_eq!(frontier.len(), 3);

    // D should not be in frontier (dominated by B)
    assert!(!frontier.iter().any(|e| e.run_id == "d"));
}

#[test]
fn test_benchmark_best_for_budget() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("cheap", 0.85, 5.0, 500.0));
    benchmark.add(test_entry("mid", 0.92, 15.0, 1500.0));
    benchmark.add(test_entry("expensive", 0.98, 50.0, 5000.0));

    let best = benchmark.best_for_budget(20.0).expect("operation should succeed");
    assert_eq!(best.run_id, "mid");

    let best = benchmark.best_for_budget(10.0).expect("operation should succeed");
    assert_eq!(best.run_id, "cheap");

    let best = benchmark.best_for_budget(1.0);
    assert!(best.is_none());
}

#[test]
fn test_benchmark_cheapest_for_quality() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("a", 0.95, 30.0, 3000.0));
    benchmark.add(test_entry("b", 0.92, 15.0, 1500.0));
    benchmark.add(test_entry("c", 0.90, 10.0, 1000.0));

    let cheapest = benchmark.cheapest_for_quality(0.90).expect("operation should succeed");
    assert_eq!(cheapest.run_id, "c");

    let cheapest = benchmark.cheapest_for_quality(0.93).expect("operation should succeed");
    assert_eq!(cheapest.run_id, "a");

    let cheapest = benchmark.cheapest_for_quality(0.99);
    assert!(cheapest.is_none());
}

#[test]
fn test_benchmark_most_efficient() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("low_eff", 0.50, 50.0, 5000.0)); // 0.01
    benchmark.add(test_entry("high_eff", 0.90, 10.0, 1000.0)); // 0.09

    let most = benchmark.most_efficient().expect("operation should succeed");
    assert_eq!(most.run_id, "high_eff");
}

#[test]
fn test_benchmark_best_quality() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("a", 0.85, 10.0, 1000.0));
    benchmark.add(test_entry("b", 0.98, 50.0, 5000.0));
    benchmark.add(test_entry("c", 0.92, 20.0, 2000.0));

    let best = benchmark.best_quality().expect("operation should succeed");
    assert_eq!(best.run_id, "b");
}

#[test]
fn test_benchmark_cheapest() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("a", 0.85, 10.0, 1000.0));
    benchmark.add(test_entry("b", 0.98, 50.0, 5000.0));
    benchmark.add(test_entry("c", 0.92, 5.0, 500.0));

    let cheapest = benchmark.cheapest().expect("operation should succeed");
    assert_eq!(cheapest.run_id, "c");
}

#[test]
fn test_benchmark_filter_by_paradigm() {
    let mut benchmark = CostPerformanceBenchmark::new();

    let mut entry1 = test_entry("dl", 0.90, 10.0, 1000.0);
    entry1.paradigm = ModelParadigm::DeepLearning;
    benchmark.add(entry1);

    let mut entry2 = test_entry("lora", 0.88, 5.0, 500.0);
    entry2.paradigm = ModelParadigm::lora(64, 64.0);
    benchmark.add(entry2);

    let dl_entries = benchmark.filter_by_paradigm(&ModelParadigm::DeepLearning);
    assert_eq!(dl_entries.len(), 1);
    assert_eq!(dl_entries[0].run_id, "dl");
}

#[test]
fn test_benchmark_filter_by_device() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("cpu-run", 0.90, 10.0, 1000.0));

    let cpu_entries = benchmark.filter_by_device_type(ComputeDevice::is_cpu);
    assert_eq!(cpu_entries.len(), 1);

    let gpu_entries = benchmark.filter_by_device_type(ComputeDevice::is_gpu);
    assert_eq!(gpu_entries.len(), 0);
}

#[test]
fn test_benchmark_statistics() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("a", 0.80, 10.0, 1000.0));
    benchmark.add(test_entry("b", 0.90, 20.0, 2000.0));
    benchmark.add(test_entry("c", 1.00, 30.0, 3000.0));

    let stats = benchmark.statistics();
    assert_eq!(stats.count, 3);
    assert!((stats.quality_min - 0.80).abs() < f64::EPSILON);
    assert!((stats.quality_max - 1.00).abs() < f64::EPSILON);
    assert!((stats.quality_avg - 0.90).abs() < 0.01);
    assert!((stats.cost_min - 10.0).abs() < f64::EPSILON);
    assert!((stats.cost_max - 30.0).abs() < f64::EPSILON);
}

#[test]
fn test_benchmark_comparison_report() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("run-001", 0.90, 10.0, 1000.0));
    benchmark.add(test_entry("run-002", 0.95, 20.0, 2000.0));

    let report = benchmark.comparison_report();
    assert!(report.contains("Benchmark Report"));
    assert!(report.contains("2 entries"));
    assert!(report.contains("Quality Scores"));
    assert!(report.contains("Pareto Frontier"));
}

#[test]
fn test_benchmark_empty_operations() {
    let benchmark = CostPerformanceBenchmark::new();

    assert!(benchmark.pareto_frontier().is_empty());
    assert!(benchmark.best_for_budget(100.0).is_none());
    assert!(benchmark.cheapest_for_quality(0.5).is_none());
    assert!(benchmark.most_efficient().is_none());
    assert!(benchmark.best_quality().is_none());
    assert!(benchmark.cheapest().is_none());
}

#[test]
fn test_benchmark_serialization() {
    let mut benchmark = CostPerformanceBenchmark::new();
    benchmark.add(test_entry("run-001", 0.90, 10.0, 1000.0));

    let json = serde_json::to_string(&benchmark).expect("JSON serialization should succeed");
    let parsed: CostPerformanceBenchmark =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed.entries[0].run_id, "run-001");
}
