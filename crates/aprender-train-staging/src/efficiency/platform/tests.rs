//! Tests for platform efficiency module.

use super::*;

#[test]
fn test_server_efficiency_new() {
    let eff = ServerEfficiency::new(1000.0, 85.0, 500.0);

    assert!((eff.throughput_samples_per_sec - 1000.0).abs() < f64::EPSILON);
    assert!((eff.gpu_utilization_percent - 85.0).abs() < f64::EPSILON);
    assert!((eff.memory_bandwidth_gbps - 500.0).abs() < f64::EPSILON);
}

#[test]
fn test_server_efficiency_clamps_utilization() {
    let eff = ServerEfficiency::new(1000.0, 150.0, 500.0);
    assert!((eff.gpu_utilization_percent - 100.0).abs() < f64::EPSILON);

    let eff = ServerEfficiency::new(1000.0, -10.0, 500.0);
    assert!((eff.gpu_utilization_percent - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_server_efficiency_gpu_status() {
    let efficient = ServerEfficiency::new(1000.0, 80.0, 500.0);
    assert!(efficient.is_gpu_efficient());
    assert!(!efficient.is_gpu_underutilized());

    let underutilized = ServerEfficiency::new(1000.0, 40.0, 500.0);
    assert!(!underutilized.is_gpu_efficient());
    assert!(underutilized.is_gpu_underutilized());
}

#[test]
fn test_server_efficiency_throughput_efficiency() {
    let eff = ServerEfficiency::new(800.0, 80.0, 500.0);
    assert!((eff.throughput_efficiency() - 10.0).abs() < 0.01);
}

#[test]
fn test_server_efficiency_estimated_max() {
    let eff = ServerEfficiency::new(500.0, 50.0, 500.0);
    assert!((eff.estimated_max_throughput() - 1000.0).abs() < 0.01);
}

#[test]
fn test_server_efficiency_memory_bound() {
    let eff = ServerEfficiency::new(1000.0, 80.0, 950.0);
    assert!(eff.memory_bound(1000.0)); // 950 > 1000 * 0.9 = 900

    let eff = ServerEfficiency::new(1000.0, 80.0, 500.0);
    assert!(!eff.memory_bound(1000.0)); // 500 < 900
}

#[test]
fn test_edge_efficiency_new() {
    let eff = EdgeEfficiency::new(
        5 * 1024 * 1024,   // 5 MB
        100,               // 100 ms
        10.0,              // 10 ms p99
        128 * 1024 * 1024, // 128 MB
    );

    assert_eq!(eff.binary_size_bytes, 5 * 1024 * 1024);
    assert_eq!(eff.startup_latency_ms, 100);
    assert!((eff.inference_latency_p99_ms - 10.0).abs() < f64::EPSILON);
}

#[test]
fn test_edge_efficiency_size_conversions() {
    let eff = EdgeEfficiency::new(
        10 * 1024 * 1024, // 10 MB
        100,
        10.0,
        256 * 1024 * 1024, // 256 MB
    );

    assert!((eff.binary_size_mb() - 10.0).abs() < 0.01);
    assert!((eff.memory_footprint_mb() - 256.0).abs() < 0.01);
}

#[test]
fn test_edge_efficiency_max_throughput() {
    let eff = EdgeEfficiency::new(0, 0, 10.0, 0); // 10 ms latency
    assert!((eff.max_throughput_per_sec() - 100.0).abs() < 0.01);
}

#[test]
fn test_edge_efficiency_meets_budget() {
    let eff = EdgeEfficiency::new(
        4 * 1024 * 1024, // 4 MB
        400,             // 400 ms
        10.0,
        200 * 1024 * 1024, // 200 MB
    );

    let budget = WasmBudget::default(); // 5 MB, 500 ms, 256 MB
    assert!(eff.meets_wasm_budget(&budget));
}

#[test]
fn test_edge_efficiency_violates_budget() {
    let eff = EdgeEfficiency::new(
        10 * 1024 * 1024, // 10 MB - exceeds default 5 MB
        400,
        10.0,
        200 * 1024 * 1024,
    );

    let budget = WasmBudget::default();
    assert!(!eff.meets_wasm_budget(&budget));

    let violations = eff.budget_violations(&budget);
    assert_eq!(violations.len(), 1);
    assert!(matches!(violations[0], BudgetViolation::BinarySize { .. }));
}

#[test]
fn test_edge_efficiency_multiple_violations() {
    let eff = EdgeEfficiency::new(
        10 * 1024 * 1024, // 10 MB - exceeds 5 MB
        1000,             // 1000 ms - exceeds 500 ms
        10.0,
        300 * 1024 * 1024, // 300 MB - exceeds 256 MB
    );

    let budget = WasmBudget::default();
    let violations = eff.budget_violations(&budget);
    assert_eq!(violations.len(), 3);
}

#[test]
fn test_wasm_budget_presets() {
    let mobile = WasmBudget::mobile();
    let desktop = WasmBudget::desktop();
    let embedded = WasmBudget::embedded();

    // Mobile is strictest
    assert!(mobile.max_binary_size < desktop.max_binary_size);
    assert!(mobile.max_memory_bytes < desktop.max_memory_bytes);

    // Embedded is most constrained
    assert!(embedded.max_binary_size < mobile.max_binary_size);
    assert!(embedded.max_memory_bytes < mobile.max_memory_bytes);
}

#[test]
fn test_wasm_budget_default() {
    let budget = WasmBudget::default();
    assert_eq!(budget.max_binary_size, 5 * 1024 * 1024);
    assert_eq!(budget.max_startup_ms, 500);
    assert_eq!(budget.max_memory_bytes, 256 * 1024 * 1024);
}

#[test]
fn test_wasm_budget_sizes_mb() {
    let budget = WasmBudget::default();
    let (binary_mb, memory_mb) = budget.sizes_mb();
    assert!((binary_mb - 5.0).abs() < 0.01);
    assert!((memory_mb - 256.0).abs() < 0.01);
}

#[test]
fn test_budget_violation_display() {
    let violation =
        BudgetViolation::BinarySize { actual: 10 * 1024 * 1024, limit: 5 * 1024 * 1024 };
    let msg = format!("{violation}");
    assert!(msg.contains("10"));
    assert!(msg.contains('5'));
    assert!(msg.contains("MB"));
}

#[test]
fn test_platform_efficiency_server() {
    let server = PlatformEfficiency::Server(ServerEfficiency::new(1000.0, 80.0, 500.0));

    assert!(server.is_server());
    assert!(!server.is_edge());
    assert!((server.throughput() - 1000.0).abs() < f64::EPSILON);
}

#[test]
fn test_platform_efficiency_edge() {
    let edge = PlatformEfficiency::Edge(EdgeEfficiency::new(0, 0, 10.0, 128 * 1024 * 1024));

    assert!(!edge.is_server());
    assert!(edge.is_edge());
    assert!((edge.throughput() - 100.0).abs() < 0.01);
    assert_eq!(edge.memory_bytes(), 128 * 1024 * 1024);
}

#[test]
fn test_server_efficiency_serialization() {
    let eff = ServerEfficiency::new(1000.0, 85.0, 500.0);
    let json = serde_json::to_string(&eff).expect("JSON serialization should succeed");
    let parsed: ServerEfficiency =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert!(
        (parsed.throughput_samples_per_sec - eff.throughput_samples_per_sec).abs() < f64::EPSILON
    );
}

#[test]
fn test_edge_efficiency_serialization() {
    let eff = EdgeEfficiency::new(5 * 1024 * 1024, 100, 10.0, 128 * 1024 * 1024);
    let json = serde_json::to_string(&eff).expect("JSON serialization should succeed");
    let parsed: EdgeEfficiency =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(parsed.binary_size_bytes, eff.binary_size_bytes);
}

#[test]
fn test_server_efficiency_default() {
    let eff = ServerEfficiency::default();
    assert!((eff.throughput_samples_per_sec - 0.0).abs() < f64::EPSILON);
    assert!((eff.gpu_utilization_percent - 0.0).abs() < f64::EPSILON);
    assert!((eff.memory_bandwidth_gbps - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_server_efficiency_zero_utilization_throughput_efficiency() {
    let eff = ServerEfficiency::new(1000.0, 0.0, 500.0);
    assert!((eff.throughput_efficiency() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_server_efficiency_zero_utilization_estimated_max() {
    let eff = ServerEfficiency::new(1000.0, 0.0, 500.0);
    assert!((eff.estimated_max_throughput() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_edge_efficiency_default() {
    let eff = EdgeEfficiency::default();
    assert_eq!(eff.binary_size_bytes, 0);
    assert_eq!(eff.startup_latency_ms, 0);
    assert!((eff.inference_latency_p99_ms - 0.0).abs() < f64::EPSILON);
    assert_eq!(eff.memory_footprint_bytes, 0);
}

#[test]
fn test_edge_efficiency_zero_latency_max_throughput() {
    let eff = EdgeEfficiency::new(0, 0, 0.0, 0);
    assert!((eff.max_throughput_per_sec() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_budget_violation_display_startup_latency() {
    let violation = BudgetViolation::StartupLatency { actual: 1000, limit: 500 };
    let msg = format!("{violation}");
    assert!(msg.contains("1000"));
    assert!(msg.contains("500"));
    assert!(msg.contains("ms"));
}

#[test]
fn test_budget_violation_display_memory_footprint() {
    let violation =
        BudgetViolation::MemoryFootprint { actual: 512 * 1024 * 1024, limit: 256 * 1024 * 1024 };
    let msg = format!("{violation}");
    assert!(msg.contains("512"));
    assert!(msg.contains("256"));
    assert!(msg.contains("MB"));
}

#[test]
fn test_wasm_budget_new() {
    let budget = WasmBudget::new(1024, 100, 4096);
    assert_eq!(budget.max_binary_size, 1024);
    assert_eq!(budget.max_startup_ms, 100);
    assert_eq!(budget.max_memory_bytes, 4096);
}

#[test]
fn test_platform_efficiency_server_memory_bytes() {
    let server = PlatformEfficiency::Server(ServerEfficiency::new(1000.0, 80.0, 500.0));
    assert_eq!(server.memory_bytes(), 0);
}

#[test]
fn test_platform_efficiency_serde() {
    let server = PlatformEfficiency::Server(ServerEfficiency::new(1000.0, 80.0, 500.0));
    let json = serde_json::to_string(&server).expect("JSON serialization should succeed");
    let parsed: PlatformEfficiency =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert!(parsed.is_server());

    let edge = PlatformEfficiency::Edge(EdgeEfficiency::new(1024, 100, 10.0, 4096));
    let json = serde_json::to_string(&edge).expect("JSON serialization should succeed");
    let parsed: PlatformEfficiency =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert!(parsed.is_edge());
}

#[test]
fn test_wasm_budget_serde() {
    let budget = WasmBudget::mobile();
    let json = serde_json::to_string(&budget).expect("JSON serialization should succeed");
    let parsed: WasmBudget =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(budget.max_binary_size, parsed.max_binary_size);
}

#[test]
fn test_budget_violation_equality() {
    let v1 = BudgetViolation::BinarySize { actual: 10, limit: 5 };
    let v2 = BudgetViolation::BinarySize { actual: 10, limit: 5 };
    assert_eq!(v1, v2);

    let v3 = BudgetViolation::StartupLatency { actual: 10, limit: 5 };
    assert_ne!(v1, v3);
}

#[test]
fn test_server_efficiency_boundary_utilization() {
    // Test exactly at the 70% efficiency boundary
    let eff_70 = ServerEfficiency::new(1000.0, 70.0, 500.0);
    assert!(eff_70.is_gpu_efficient());
    assert!(!eff_70.is_gpu_underutilized());

    // Test exactly at the 50% underutilization boundary
    let eff_50 = ServerEfficiency::new(1000.0, 50.0, 500.0);
    assert!(!eff_50.is_gpu_efficient());
    assert!(!eff_50.is_gpu_underutilized());

    // Test just below 50%
    let eff_49 = ServerEfficiency::new(1000.0, 49.9, 500.0);
    assert!(eff_49.is_gpu_underutilized());
}

#[test]
fn test_server_efficiency_clone() {
    let eff = ServerEfficiency::new(1000.0, 80.0, 500.0);
    let cloned = eff.clone();
    assert_eq!(eff, cloned);
}

#[test]
fn test_edge_efficiency_clone() {
    let eff = EdgeEfficiency::new(1024, 100, 10.0, 4096);
    let cloned = eff.clone();
    assert_eq!(eff, cloned);
}

#[test]
fn test_wasm_budget_clone() {
    let budget = WasmBudget::desktop();
    let cloned = budget.clone();
    assert_eq!(budget, cloned);
}

#[test]
fn test_platform_efficiency_clone() {
    let server = PlatformEfficiency::Server(ServerEfficiency::new(1000.0, 80.0, 500.0));
    let cloned = server.clone();
    assert_eq!(server, cloned);
}

#[test]
fn test_budget_violation_clone() {
    let violation = BudgetViolation::BinarySize { actual: 10, limit: 5 };
    let cloned = violation.clone();
    assert_eq!(violation, cloned);
}
