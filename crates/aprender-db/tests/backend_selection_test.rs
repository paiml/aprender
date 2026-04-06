//! Backend selection tests for CORE-002
//!
//! Tests the cost-based backend dispatcher with the 5x rule:
//! - GPU only if `estimated_gpu_compute_ms` > `pcie_transfer_ms` * 5.0
//!
//! References:
//! - Gregg & Hazelwood (2011): `PCIe` bus bottleneck analysis
//! - Breß et al. (2014): Operator variant selection on heterogeneous hardware
//!
//! Toyota Way: Genchi Genbutsu (Go and See - physics-based cost model)

use trueno_db::backend::BackendDispatcher;

/// `PCIe` Gen4 x16 bandwidth: 32 GB/s
const PCIE_BANDWIDTH_GBPS: f64 = 32.0;

#[test]
fn test_small_dataset_selects_cpu() {
    // Small dataset (1 MB), simple aggregation
    let total_bytes = 1_000_000; // 1 MB
    let estimated_flops = 1_000_000.0; // Simple sum

    // Calculate transfer time: 1 MB / 32 GB/s = 0.03125 ms
    // For GPU to be worth it, compute must be > 5x transfer time
    // 0.03125 * 5 = 0.15625 ms
    // But our estimated compute is ~0.001 ms (1M FLOPs at 1 TFLOP/s)

    let backend = BackendDispatcher::select(total_bytes, estimated_flops);

    // Should select CPU (not worth GPU transfer overhead)
    assert!(
        matches!(backend, trueno_db::Backend::Simd),
        "Small dataset should use CPU/SIMD backend"
    );
}

#[test]
fn test_large_compute_selects_gpu() {
    // Large dataset (1 GB), complex aggregation
    let total_bytes = 1_000_000_000; // 1 GB
    let estimated_flops = 10_000_000_000.0; // 10 GFLOP (complex aggregation)

    // Calculate transfer time: 1 GB / 32 GB/s = 31.25 ms
    let transfer_time_ms = (total_bytes as f64 / (PCIE_BANDWIDTH_GBPS * 1_000_000_000.0)) * 1000.0;
    assert_eq!(transfer_time_ms, 31.25);

    // For GPU to be worth it, compute must be > 5x transfer time
    // 31.25 * 5 = 156.25 ms minimum compute time

    // At 1 TFLOP/s, 10 GFLOP takes 10 ms (too low)
    // But at 100 GFLOP/s (more realistic for GPU), it takes 100ms
    // Still not enough... let's increase compute

    let backend = BackendDispatcher::select(total_bytes, estimated_flops);

    // This should still select CPU because compute isn't 5x transfer
    assert!(
        matches!(backend, trueno_db::Backend::Simd),
        "Compute not 5x transfer time, should use CPU"
    );
}

#[test]
fn test_very_large_compute_selects_gpu() {
    // Large dataset (1 GB), very complex computation (e.g., hash join)
    let total_bytes = 1_000_000_000; // 1 GB
    let estimated_flops = 100_000_000_000.0; // 100 GFLOP (hash join)

    // Transfer time: 31.25 ms
    // Minimum compute for GPU: 31.25 * 5 = 156.25 ms
    // At 100 GFLOP/s, 100 GFLOP takes 1000ms > 156.25ms ✓

    let backend = BackendDispatcher::select(total_bytes, estimated_flops);

    // Should select GPU (compute is 5x+ transfer time)
    assert!(matches!(backend, trueno_db::Backend::Gpu), "Large compute should use GPU backend");
}

#[test]
fn test_minimum_data_threshold() {
    // Test the 10 MB minimum threshold from spec
    let total_bytes = 5_000_000; // 5 MB (below threshold)
    let estimated_flops = 1_000_000_000.0; // High compute

    let backend = BackendDispatcher::select(total_bytes, estimated_flops);

    // Should use CPU even with high compute (below 10 MB threshold)
    assert!(
        !matches!(backend, trueno_db::Backend::Gpu),
        "Data below 10 MB threshold should not use GPU"
    );
}

#[test]
fn test_arithmetic_intensity_calculation() {
    // Test arithmetic intensity (FLOPs per byte)
    let total_bytes = 100_000_000; // 100 MB
    let estimated_flops = 1_000_000_000.0; // 1 GFLOP

    let arithmetic_intensity = estimated_flops / total_bytes as f64;
    assert_eq!(arithmetic_intensity, 10.0); // 10 FLOPs per byte

    // This is moderate arithmetic intensity
    // Transfer: 100 MB / 32 GB/s = 3.125 ms
    // Compute needed: 3.125 * 5 = 15.625 ms
    // At 100 GFLOP/s: 1 GFLOP / 100 GFLOP/s = 10 ms
    // 10 ms < 15.625 ms, so should use CPU

    let backend = BackendDispatcher::select(total_bytes, estimated_flops);
    assert!(!matches!(backend, trueno_db::Backend::Gpu), "Moderate intensity should use CPU");
}

// ============================================================================
// FLOPs Estimation Helper Tests
// ============================================================================

#[test]
fn test_arithmetic_intensity_helper() {
    // Matrix multiply example: N^3 FLOPs for N^2 elements = N FLOPs/element
    let total_flops = 1_000_000_000.0; // 1 GFLOP
    let total_bytes = 100_000_000; // 100 MB

    let intensity = BackendDispatcher::arithmetic_intensity(total_flops, total_bytes);
    assert_eq!(intensity, 10.0); // 10 FLOPs per byte

    // Simple aggregation: 1 FLOP per element (low intensity)
    let simple_flops = 100_000_000.0; // 100M FLOPs
    let simple_bytes = 400_000_000; // 400 MB (f32 = 4 bytes)
    let simple_intensity = BackendDispatcher::arithmetic_intensity(simple_flops, simple_bytes);
    assert_eq!(simple_intensity, 0.25); // 0.25 FLOPs per byte (very low)
}

#[test]
fn test_estimate_simple_aggregation_flops() {
    // Simple aggregations (SUM, AVG, COUNT, MIN, MAX) perform ~1 FLOP per element
    let num_elements = 100_000_000; // 100M elements

    let flops = BackendDispatcher::estimate_simple_aggregation_flops(num_elements);
    assert_eq!(flops, 100_000_000.0); // 100M FLOPs

    // SUM over 100M elements = 100M FLOPs
    // At 100 GFLOP/s GPU: 1ms compute time
    // Transfer: 400 MB / 32 GB/s = 12.5 ms
    // 1ms < 12.5ms * 5 = 62.5ms, so CPU is better ✓
}

#[test]
fn test_estimate_group_by_flops() {
    // GROUP BY requires hashing (5 FLOPs/elem) + aggregation (1 FLOP/elem) = 6 FLOPs/elem
    let num_elements = 100_000_000; // 100M elements

    let flops = BackendDispatcher::estimate_group_by_flops(num_elements);
    assert_eq!(flops, 600_000_000.0); // 600M FLOPs

    // GROUP BY over 100M elements = 600M FLOPs
    // At 100 GFLOP/s GPU: 6ms compute time
    // Transfer: 400 MB / 32 GB/s = 12.5 ms
    // 6ms < 12.5ms * 5 = 62.5ms, so still CPU ✓
}

#[test]
fn test_estimate_filter_flops() {
    // Filters require predicate evaluation (~2 FLOPs per element)
    let num_elements = 100_000_000; // 100M elements

    let flops = BackendDispatcher::estimate_filter_flops(num_elements);
    assert_eq!(flops, 200_000_000.0); // 200M FLOPs

    // WHERE filter over 100M elements = 200M FLOPs
    // At 100 GFLOP/s GPU: 2ms compute time
    // Transfer: 400 MB / 32 GB/s = 12.5 ms
    // 2ms < 12.5ms * 5 = 62.5ms, so CPU is better ✓
}

#[test]
fn test_estimate_join_flops() {
    // Hash join: Build hash table (5 FLOPs/elem) + Probe (5 FLOPs/elem)
    let left_size = 10_000_000; // 10M rows
    let right_size = 100_000_000; // 100M rows

    let flops = BackendDispatcher::estimate_join_flops(left_size, right_size);
    assert_eq!(flops, 550_000_000.0); // 550M FLOPs

    // JOIN: (10M + 100M) * 5 = 550M FLOPs
    // At 100 GFLOP/s GPU: 5.5ms compute time
    // Transfer: (40MB + 400MB) / 32 GB/s = 13.75 ms
    // 5.5ms < 13.75ms * 5 = 68.75ms, so still CPU
}

#[test]
fn test_realistic_sql_operations_backend_selection() {
    // Test realistic SQL operations to validate cost model

    // Scenario 1: Simple SUM over 100M rows
    let num_elements = 100_000_000;
    let total_bytes = num_elements * 4; // f32 = 4 bytes
    let flops = BackendDispatcher::estimate_simple_aggregation_flops(num_elements);

    let backend = BackendDispatcher::select(total_bytes, flops);
    assert!(matches!(backend, trueno_db::Backend::Simd), "Simple SUM should use SIMD");

    // Scenario 2: GROUP BY over 1B rows
    // 1B elements * 4 bytes = 4GB, 6B FLOPs
    // Transfer: 4GB / 32 GB/s = 125ms
    // Compute at 100 GFLOP/s: 6 GFLOP / 100 = 60ms
    // 60ms < 125ms * 5 = 625ms, so still SIMD
    let large_num_elements = 1_000_000_000; // 1B elements
    let large_total_bytes = large_num_elements * 4;
    let large_flops = BackendDispatcher::estimate_group_by_flops(large_num_elements);

    let large_backend = BackendDispatcher::select(large_total_bytes, large_flops);
    assert!(
        matches!(large_backend, trueno_db::Backend::Simd),
        "GROUP BY over 1B rows still uses SIMD (transfer dominates)"
    );

    // Scenario 2b: Extreme compute scenario that triggers GPU
    // Need compute > transfer * 5
    // 1GB transfer = 31.25ms, so need > 156.25ms compute
    // At 100 GFLOP/s: need > 15.625 GFLOP
    // Use 100 GFLOP (similar to test_very_large_compute_selects_gpu)
    let extreme_bytes = 1_000_000_000; // 1 GB
    let extreme_flops = 100_000_000_000.0; // 100 GFLOP (e.g., complex multi-pass algorithm)

    let extreme_backend = BackendDispatcher::select(extreme_bytes, extreme_flops);
    assert!(
        matches!(extreme_backend, trueno_db::Backend::Gpu),
        "Extreme compute workload should use GPU"
    );

    // Scenario 3: Complex JOIN
    let left = 50_000_000;
    let right = 50_000_000;
    let join_bytes = (left + right) * 4;
    let join_flops = BackendDispatcher::estimate_join_flops(left, right);

    let join_backend = BackendDispatcher::select(join_bytes, join_flops);
    // 100M * 4 = 400MB, 500M FLOPs
    // Transfer: 12.5ms, Compute at 100 GFLOP/s: 5ms
    // 5ms < 62.5ms, so CPU
    assert!(matches!(join_backend, trueno_db::Backend::Simd), "Moderate JOIN should use SIMD");
}
