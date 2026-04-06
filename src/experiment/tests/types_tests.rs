//! Tests for compute devices, metrics, and benchmark types.

use crate::experiment::*;
use std::collections::HashMap;

// -------------------------------------------------------------------------
// ComputeDevice Tests
// -------------------------------------------------------------------------

#[test]
fn test_compute_device_cpu_creation() {
    let cpu =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    match cpu {
        ComputeDevice::Cpu { cores, threads_per_core, architecture } => {
            assert_eq!(cores, 8);
            assert_eq!(threads_per_core, 2);
            assert_eq!(architecture, CpuArchitecture::X86_64);
        }
        _ => panic!("Expected CPU device"),
    }
}

#[test]
fn test_compute_device_gpu_creation() {
    let gpu = ComputeDevice::Gpu {
        name: "RTX 4090".to_string(),
        memory_gb: 24.0,
        compute_capability: Some("8.9".to_string()),
        vendor: GpuVendor::Nvidia,
    };

    match gpu {
        ComputeDevice::Gpu { name, memory_gb, vendor, .. } => {
            assert_eq!(name, "RTX 4090");
            assert_eq!(memory_gb, 24.0);
            assert_eq!(vendor, GpuVendor::Nvidia);
        }
        _ => panic!("Expected GPU device"),
    }
}

#[test]
fn test_compute_device_tpu_creation() {
    let tpu = ComputeDevice::Tpu { version: TpuVersion::V4, cores: 8 };

    match tpu {
        ComputeDevice::Tpu { version, cores } => {
            assert_eq!(version, TpuVersion::V4);
            assert_eq!(cores, 8);
        }
        _ => panic!("Expected TPU device"),
    }
}

#[test]
fn test_compute_device_apple_silicon_creation() {
    let m3 = ComputeDevice::AppleSilicon {
        chip: AppleChip::M3Max,
        neural_engine_cores: 16,
        gpu_cores: 40,
        memory_gb: 64,
    };

    match m3 {
        ComputeDevice::AppleSilicon { chip, gpu_cores, memory_gb, .. } => {
            assert_eq!(chip, AppleChip::M3Max);
            assert_eq!(gpu_cores, 40);
            assert_eq!(memory_gb, 64);
        }
        _ => panic!("Expected Apple Silicon device"),
    }
}

#[test]
fn test_compute_device_edge_creation() {
    let edge = ComputeDevice::Edge { name: "Raspberry Pi 5".to_string(), power_budget_watts: 15.0 };

    match edge {
        ComputeDevice::Edge { name, power_budget_watts } => {
            assert_eq!(name, "Raspberry Pi 5");
            assert_eq!(power_budget_watts, 15.0);
        }
        _ => panic!("Expected Edge device"),
    }
}

#[test]
fn test_compute_device_theoretical_flops_cpu() {
    let cpu =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let flops = cpu.theoretical_flops();
    assert!(flops > 0.0);
    // 8 cores * 2 threads * 32 FLOPS * 1e9 = 512e9
    assert_eq!(flops, 512e9);
}

#[test]
fn test_compute_device_theoretical_flops_gpu() {
    let gpu = ComputeDevice::Gpu {
        name: "RTX 4090".to_string(),
        memory_gb: 24.0,
        compute_capability: Some("8.9".to_string()),
        vendor: GpuVendor::Nvidia,
    };

    let flops = gpu.theoretical_flops();
    assert!(flops > 0.0);
    // 24 GB * 15 * 1e12 = 360e12
    assert_eq!(flops, 360e12);
}

#[test]
fn test_compute_device_theoretical_flops_tpu() {
    let tpu = ComputeDevice::Tpu { version: TpuVersion::V4, cores: 8 };

    let flops = tpu.theoretical_flops();
    assert!(flops > 0.0);
    // 8 * 275e12 = 2200e12
    assert_eq!(flops, 2200e12);
}

#[test]
fn test_compute_device_estimated_power() {
    let cpu =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };
    assert_eq!(cpu.estimated_power_watts(), 120.0); // 8 * 15

    let gpu = ComputeDevice::Gpu {
        name: "RTX 4090".to_string(),
        memory_gb: 24.0,
        compute_capability: None,
        vendor: GpuVendor::Nvidia,
    };
    assert_eq!(gpu.estimated_power_watts(), 720.0); // 24 * 30
}

// -------------------------------------------------------------------------
// EnergyMetrics Tests
// -------------------------------------------------------------------------

#[test]
fn test_energy_metrics_creation() {
    let metrics = EnergyMetrics::new(3600.0, 100.0, 150.0, 36.0);

    assert_eq!(metrics.total_joules, 3600.0);
    assert_eq!(metrics.average_power_watts, 100.0);
    assert_eq!(metrics.peak_power_watts, 150.0);
    assert_eq!(metrics.duration_seconds, 36.0);
    assert_eq!(metrics.pue, 1.0);
    assert!(metrics.co2_grams.is_none());
}

#[test]
fn test_energy_metrics_with_carbon_intensity() {
    let metrics =
        EnergyMetrics::new(3_600_000.0, 100.0, 150.0, 36000.0).with_carbon_intensity(400.0); // 400g CO2/kWh

    assert!(metrics.co2_grams.is_some());
    // 3600000 J = 1 kWh, * 400 g/kWh = 400g
    assert_eq!(metrics.co2_grams.unwrap(), 400.0);
}

#[test]
fn test_energy_metrics_with_pue() {
    let metrics = EnergyMetrics::new(3_600_000.0, 100.0, 150.0, 36000.0)
        .with_carbon_intensity(400.0)
        .with_pue(1.5);

    assert_eq!(metrics.pue, 1.5);
    // CO2 should be 400 * 1.5 = 600
    assert_eq!(metrics.co2_grams.unwrap(), 600.0);
}

#[test]
fn test_energy_metrics_flops_per_watt() {
    let metrics = EnergyMetrics::new(3600.0, 100.0, 150.0, 36.0);
    let flops_per_watt = metrics.flops_per_watt(1e15);

    assert_eq!(flops_per_watt, 1e13); // 1e15 / 100
}

#[test]
fn test_energy_metrics_flops_per_watt_zero_power() {
    let metrics = EnergyMetrics::new(0.0, 0.0, 0.0, 0.0);
    let flops_per_watt = metrics.flops_per_watt(1e15);

    assert_eq!(flops_per_watt, 0.0);
}

// -------------------------------------------------------------------------
// CostMetrics Tests
// -------------------------------------------------------------------------

#[test]
fn test_cost_metrics_creation() {
    let metrics = CostMetrics::new(10.0, 2.0, 0.5);

    assert_eq!(metrics.compute_cost_usd, 10.0);
    assert_eq!(metrics.storage_cost_usd, 2.0);
    assert_eq!(metrics.network_cost_usd, 0.5);
    assert_eq!(metrics.total_cost_usd, 12.5);
    assert_eq!(metrics.currency, "USD");
}

#[test]
fn test_cost_metrics_with_flops() {
    let metrics = CostMetrics::new(10.0, 0.0, 0.0).with_flops(1e18);

    assert!(metrics.cost_per_flop.is_some());
    assert_eq!(metrics.cost_per_flop.unwrap(), 1e-17);
}

#[test]
fn test_cost_metrics_with_samples() {
    let metrics = CostMetrics::new(10.0, 0.0, 0.0).with_samples(1000);

    assert!(metrics.cost_per_sample.is_some());
    assert_eq!(metrics.cost_per_sample.unwrap(), 0.01);
}

// -------------------------------------------------------------------------
// ModelParadigm Tests
// -------------------------------------------------------------------------

#[test]
fn test_model_paradigm_compute_intensity() {
    assert_eq!(ModelParadigm::TraditionalML.compute_intensity(), ComputeIntensity::Low);
    assert_eq!(ModelParadigm::DeepLearning.compute_intensity(), ComputeIntensity::High);
    assert_eq!(ModelParadigm::MoE.compute_intensity(), ComputeIntensity::VeryHigh);
    assert_eq!(ModelParadigm::FineTuning.compute_intensity(), ComputeIntensity::Medium);
}

#[test]
fn test_model_paradigm_benefits_from_gpu() {
    assert!(ModelParadigm::DeepLearning.benefits_from_gpu());
    assert!(ModelParadigm::FineTuning.benefits_from_gpu());
    assert!(ModelParadigm::MoE.benefits_from_gpu());
    assert!(!ModelParadigm::TraditionalML.benefits_from_gpu());
    assert!(!ModelParadigm::FederatedLearning.benefits_from_gpu());
}

// -------------------------------------------------------------------------
// CostPerformanceBenchmark Tests
// -------------------------------------------------------------------------

#[test]
fn test_benchmark_creation() {
    let benchmark = CostPerformanceBenchmark::new("test-benchmark");
    assert_eq!(benchmark.name, "test-benchmark");
    assert!(benchmark.points.is_empty());
}

#[test]
fn test_benchmark_add_point() {
    let mut benchmark = CostPerformanceBenchmark::new("test");
    benchmark.add_point(CostPerformancePoint {
        id: "p1".to_string(),
        performance: 0.95,
        cost: 100.0,
        energy_joules: 1000.0,
        latency_ms: Some(10.0),
        metadata: HashMap::new(),
    });

    assert_eq!(benchmark.points.len(), 1);
}

#[test]
fn test_pareto_frontier_single_point() {
    let mut benchmark = CostPerformanceBenchmark::new("test");
    benchmark.add_point(CostPerformancePoint {
        id: "p1".to_string(),
        performance: 0.95,
        cost: 100.0,
        energy_joules: 1000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let frontier = benchmark.compute_pareto_frontier();
    assert_eq!(frontier.len(), 1);
    assert_eq!(frontier[0], 0);
}

#[test]
fn test_pareto_frontier_dominated_point() {
    let mut benchmark = CostPerformanceBenchmark::new("test");

    // Point that dominates
    benchmark.add_point(CostPerformancePoint {
        id: "dominant".to_string(),
        performance: 0.95,
        cost: 50.0,
        energy_joules: 500.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    // Dominated point (worse on both dimensions)
    benchmark.add_point(CostPerformancePoint {
        id: "dominated".to_string(),
        performance: 0.90,
        cost: 100.0,
        energy_joules: 1000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let frontier = benchmark.compute_pareto_frontier().to_vec();
    assert_eq!(frontier.len(), 1);
    assert_eq!(benchmark.points[frontier[0]].id, "dominant");
}

#[test]
fn test_pareto_frontier_multiple_optimal() {
    let mut benchmark = CostPerformanceBenchmark::new("test");

    // High performance, high cost
    benchmark.add_point(CostPerformancePoint {
        id: "high-perf".to_string(),
        performance: 0.99,
        cost: 200.0,
        energy_joules: 2000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    // Low performance, low cost
    benchmark.add_point(CostPerformancePoint {
        id: "low-cost".to_string(),
        performance: 0.85,
        cost: 20.0,
        energy_joules: 200.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let frontier = benchmark.compute_pareto_frontier();
    assert_eq!(frontier.len(), 2); // Both are Pareto-optimal
}

#[test]
fn test_best_within_budget() {
    let mut benchmark = CostPerformanceBenchmark::new("test");

    benchmark.add_point(CostPerformancePoint {
        id: "expensive".to_string(),
        performance: 0.99,
        cost: 200.0,
        energy_joules: 2000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    benchmark.add_point(CostPerformancePoint {
        id: "cheap".to_string(),
        performance: 0.85,
        cost: 20.0,
        energy_joules: 200.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let best = benchmark.best_within_budget(50.0);
    assert!(best.is_some());
    assert_eq!(best.unwrap().id, "cheap");
}

#[test]
fn test_efficiency_scores() {
    let mut benchmark = CostPerformanceBenchmark::new("test");

    benchmark.add_point(CostPerformancePoint {
        id: "p1".to_string(),
        performance: 0.90,
        cost: 100.0,
        energy_joules: 1000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let scores = benchmark.efficiency_scores();
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[0].0, 0);
    // Use approximate comparison for floating point
    assert!((scores[0].1 - 0.009).abs() < 1e-10); // 0.90 / 100
}

// -------------------------------------------------------------------------
// PlatformEfficiency Tests
// -------------------------------------------------------------------------

#[test]
fn test_platform_efficiency_power_budget() {
    assert_eq!(PlatformEfficiency::Server.typical_power_budget_watts(), 500.0);
    assert_eq!(PlatformEfficiency::Laptop.typical_power_budget_watts(), 65.0);
    assert_eq!(PlatformEfficiency::Edge.typical_power_budget_watts(), 15.0);
    assert_eq!(PlatformEfficiency::Mobile.typical_power_budget_watts(), 5.0);
    assert_eq!(PlatformEfficiency::Embedded.typical_power_budget_watts(), 1.0);
}

// -------------------------------------------------------------------------
// Serialization Tests
// -------------------------------------------------------------------------

#[test]
fn test_compute_device_serialization() {
    let device = ComputeDevice::Gpu {
        name: "RTX 4090".to_string(),
        memory_gb: 24.0,
        compute_capability: Some("8.9".to_string()),
        vendor: GpuVendor::Nvidia,
    };

    let json = serde_json::to_string(&device).unwrap();
    let deserialized: ComputeDevice = serde_json::from_str(&json).unwrap();
    assert_eq!(device, deserialized);
}

// -------------------------------------------------------------------------
// Error Type Tests
// -------------------------------------------------------------------------

#[test]
fn test_experiment_error_display() {
    let err = ExperimentError::InvalidComputeDevice("bad config".to_string());
    assert_eq!(format!("{}", err), "Invalid compute device configuration: bad config");

    let err = ExperimentError::InvalidOrcid("bad-orcid".to_string());
    assert_eq!(format!("{}", err), "Invalid ORCID format: bad-orcid");
}

// -------------------------------------------------------------------------
// CostPerformanceBenchmark: Additional Coverage Tests
// -------------------------------------------------------------------------

#[test]
fn test_pareto_optimal_points_returns_correct_references() {
    let mut benchmark = CostPerformanceBenchmark::new("pareto-refs");

    // High performance, high cost (Pareto-optimal)
    benchmark.add_point(CostPerformancePoint {
        id: "high-perf".to_string(),
        performance: 0.99,
        cost: 200.0,
        energy_joules: 2000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    // Low performance, low cost (Pareto-optimal)
    benchmark.add_point(CostPerformancePoint {
        id: "low-cost".to_string(),
        performance: 0.85,
        cost: 20.0,
        energy_joules: 200.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    // Dominated point (worse than "high-perf" on both dimensions)
    benchmark.add_point(CostPerformancePoint {
        id: "dominated".to_string(),
        performance: 0.90,
        cost: 300.0,
        energy_joules: 3000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    // ACT: Call pareto_optimal_points (the 0% covered function)
    let optimal = benchmark.pareto_optimal_points();

    // ASSERT
    assert_eq!(optimal.len(), 2);
    let ids: Vec<&str> = optimal.iter().map(|p| p.id.as_str()).collect();
    assert!(ids.contains(&"high-perf"));
    assert!(ids.contains(&"low-cost"));
    assert!(!ids.contains(&"dominated"));
}

#[test]
fn test_pareto_optimal_points_single() {
    let mut benchmark = CostPerformanceBenchmark::new("single-pareto");
    benchmark.add_point(CostPerformancePoint {
        id: "only".to_string(),
        performance: 0.5,
        cost: 50.0,
        energy_joules: 500.0,
        latency_ms: Some(5.0),
        metadata: HashMap::new(),
    });

    let optimal = benchmark.pareto_optimal_points();
    assert_eq!(optimal.len(), 1);
    assert_eq!(optimal[0].id, "only");
}

#[test]
fn test_pareto_optimal_points_empty() {
    let mut benchmark = CostPerformanceBenchmark::new("empty-pareto");
    let optimal = benchmark.pareto_optimal_points();
    assert!(optimal.is_empty());
}

#[test]
fn test_efficiency_scores_zero_cost() {
    // Tests the p.cost == 0.0 branch that returns INFINITY
    let mut benchmark = CostPerformanceBenchmark::new("zero-cost");
    benchmark.add_point(CostPerformancePoint {
        id: "free".to_string(),
        performance: 0.95,
        cost: 0.0,
        energy_joules: 100.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let scores = benchmark.efficiency_scores();
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[0].0, 0);
    assert!(scores[0].1.is_infinite(), "Zero cost should give infinite efficiency");
}

#[test]
fn test_efficiency_scores_multiple_points() {
    let mut benchmark = CostPerformanceBenchmark::new("multi-efficiency");
    benchmark.add_point(CostPerformancePoint {
        id: "p1".to_string(),
        performance: 0.80,
        cost: 40.0,
        energy_joules: 400.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });
    benchmark.add_point(CostPerformancePoint {
        id: "p2".to_string(),
        performance: 0.90,
        cost: 10.0,
        energy_joules: 100.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let scores = benchmark.efficiency_scores();
    assert_eq!(scores.len(), 2);
    // p2 is more efficient (0.90/10 = 0.09) than p1 (0.80/40 = 0.02)
    assert!((scores[0].1 - 0.02).abs() < 1e-10); // p1: 0.80/40
    assert!((scores[1].1 - 0.09).abs() < 1e-10); // p2: 0.90/10
}

#[test]
fn test_best_within_budget_no_match() {
    let mut benchmark = CostPerformanceBenchmark::new("no-budget");
    benchmark.add_point(CostPerformancePoint {
        id: "expensive".to_string(),
        performance: 0.99,
        cost: 1000.0,
        energy_joules: 5000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    // Budget too low for any option
    let best = benchmark.best_within_budget(1.0);
    assert!(best.is_none());
}

#[test]
fn test_best_within_budget_multiple_candidates() {
    let mut benchmark = CostPerformanceBenchmark::new("budget-multi");
    benchmark.add_point(CostPerformancePoint {
        id: "good".to_string(),
        performance: 0.80,
        cost: 30.0,
        energy_joules: 300.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });
    benchmark.add_point(CostPerformancePoint {
        id: "better".to_string(),
        performance: 0.90,
        cost: 40.0,
        energy_joules: 400.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });
    benchmark.add_point(CostPerformancePoint {
        id: "too-expensive".to_string(),
        performance: 0.99,
        cost: 200.0,
        energy_joules: 2000.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let best = benchmark.best_within_budget(50.0);
    assert!(best.is_some());
    assert_eq!(best.unwrap().id, "better");
}

#[test]
fn test_pareto_frontier_cached() {
    // Test that compute_pareto_frontier returns cached result on second call
    let mut benchmark = CostPerformanceBenchmark::new("cached");
    benchmark.add_point(CostPerformancePoint {
        id: "a".to_string(),
        performance: 0.5,
        cost: 50.0,
        energy_joules: 500.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    let frontier1 = benchmark.compute_pareto_frontier().to_vec();
    let frontier2 = benchmark.compute_pareto_frontier().to_vec();
    assert_eq!(frontier1, frontier2);
}

#[test]
fn test_add_point_invalidates_pareto_cache() {
    let mut benchmark = CostPerformanceBenchmark::new("invalidate");
    benchmark.add_point(CostPerformancePoint {
        id: "a".to_string(),
        performance: 0.5,
        cost: 50.0,
        energy_joules: 500.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    // Compute frontier first
    let frontier1 = benchmark.compute_pareto_frontier().to_vec();
    assert_eq!(frontier1.len(), 1);

    // Add new dominant point - should invalidate cache
    benchmark.add_point(CostPerformancePoint {
        id: "b".to_string(),
        performance: 0.9,
        cost: 10.0,
        energy_joules: 100.0,
        latency_ms: None,
        metadata: HashMap::new(),
    });

    // Recompute - should have 1 point (b dominates a)
    let frontier2 = benchmark.compute_pareto_frontier().to_vec();
    assert_eq!(frontier2.len(), 1);
    assert_eq!(benchmark.points[frontier2[0]].id, "b");
}

#[test]
fn test_cost_performance_point_with_metadata() {
    let mut metadata = HashMap::new();
    metadata.insert("gpu".to_string(), "A100".to_string());
    metadata.insert("batch_size".to_string(), "32".to_string());

    let point = CostPerformancePoint {
        id: "meta-test".to_string(),
        performance: 0.95,
        cost: 100.0,
        energy_joules: 1000.0,
        latency_ms: Some(15.5),
        metadata,
    };

    assert_eq!(point.metadata.get("gpu"), Some(&"A100".to_string()));
    assert_eq!(point.latency_ms, Some(15.5));
}
