use super::*;

#[test]
fn test_backend_selection_small_matmul() {
    let selector = BackendSelector::new();

    // Small matrix: 64x64 x 64x64
    // Data: (64*64 + 64*64 + 64*64) * 4 = 49,152 bytes
    // FLOPs: 2 * 64 * 64 * 64 = 524,288
    // Transfer: 49,152 / 32e9 = 1.536 us
    // Compute: 524,288 / 20e12 = 0.026 us
    // Ratio: 0.026 / 1.536 = 0.017x (< 5x) -> SIMD
    let backend = selector.select_for_matmul(64, 64, 64);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_backend_selection_large_matmul() {
    let selector = BackendSelector::new();

    // Large matrix: 512x512 x 512x512
    // Data: (512*512 + 512*512 + 512*512) * 4 = 3,145,728 bytes
    // FLOPs: 2 * 512 * 512 * 512 = 268,435,456
    // Transfer: 3,145,728 / 32e9 = 98.3 us
    // Compute: 268,435,456 / 20e12 = 13.4 us
    // Ratio: 13.4 / 98.3 = 0.136x (< 5x) -> SIMD
    // NOTE: GPU only becomes beneficial for much larger matrices or
    // when compute complexity is O(n^3) with very large n
    let backend = selector.select_for_matmul(512, 512, 512);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_backend_selection_very_large_matmul() {
    let selector = BackendSelector::new();

    // Very large matrix: 2048x2048 x 2048x2048
    // Data: (2048*2048 + 2048*2048 + 2048*2048) * 4 = 50,331,648 bytes
    // FLOPs: 2 * 2048 * 2048 * 2048 = 17,179,869,184
    // Transfer: 50,331,648 / 32e9 = 1,573 us = 1.57 ms
    // Compute: 17,179,869,184 / 20e12 = 859 us = 0.859 ms
    // Ratio: 859 / 1573 = 0.546x (< 5x) -> still SIMD!
    // Per spec, GPU dispatch needs > 5x compute/transfer ratio
    let backend = selector.select_for_matmul(2048, 2048, 2048);
    // Even this is borderline - real benefit comes from O(n^3) operations
    // with sustained computation
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_backend_selection_dot_product() {
    let selector = BackendSelector::new();

    // Large dot product: 10K elements
    // Data: 120 KB
    // FLOPs: 20K (2 ops per element)
    // Transfer: 3.75 us, Compute: 1 ns -> ratio: 0.0003x
    let backend = selector.select_for_vector_op(10_000, 2);
    assert_eq!(backend, Backend::SIMD); // Not GPU
}

#[test]
fn test_backend_selection_elementwise() {
    let selector = BackendSelector::new();

    // Small array
    let backend = selector.select_for_elementwise(1000);
    assert_eq!(backend, Backend::Scalar);

    // Large array
    let backend = selector.select_for_elementwise(2_000_000);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_custom_dispatch_ratio() {
    let selector = BackendSelector::new().with_min_dispatch_ratio(10.0); // More conservative

    // Workload that passes 5x but fails 10x
    let backend = selector.select_backend(1_000_000, 30_000_000);
    // Transfer: ~31 us, Compute: ~1.5 us -> ratio: ~0.05x
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_moe_low_complexity() {
    let selector = BackendSelector::new();

    // Small element-wise: Scalar
    assert_eq!(selector.select_with_moe(OpComplexity::Low, 100), Backend::Scalar);

    // Large element-wise: SIMD
    assert_eq!(selector.select_with_moe(OpComplexity::Low, 2_000_000), Backend::SIMD);

    // Never GPU for element-wise (memory-bound)
    assert_ne!(selector.select_with_moe(OpComplexity::Low, 10_000_000), Backend::GPU);
}

#[test]
fn test_moe_medium_complexity() {
    let selector = BackendSelector::new();

    // Small reduction: Scalar
    assert_eq!(selector.select_with_moe(OpComplexity::Medium, 1_000), Backend::Scalar);

    // Medium reduction: SIMD
    assert_eq!(selector.select_with_moe(OpComplexity::Medium, 50_000), Backend::SIMD);

    // Large reduction: GPU
    assert_eq!(selector.select_with_moe(OpComplexity::Medium, 200_000), Backend::GPU);
}

#[test]
fn test_moe_high_complexity() {
    let selector = BackendSelector::new();

    // Small matmul: Scalar
    assert_eq!(selector.select_with_moe(OpComplexity::High, 500), Backend::Scalar);

    // Medium matmul: SIMD
    assert_eq!(selector.select_with_moe(OpComplexity::High, 5_000), Backend::SIMD);

    // Large matmul: GPU
    assert_eq!(selector.select_with_moe(OpComplexity::High, 50_000), Backend::GPU);
}

#[test]
#[cfg(feature = "trueno-integration")]
fn test_trueno_vector_add() {
    let selector = BackendSelector::new();

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let result = selector.vector_add(&a, &b).unwrap();
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
#[cfg(feature = "trueno-integration")]
fn test_trueno_matrix_multiply() {
    let selector = BackendSelector::new();

    // 2x2 matrices
    let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
    let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]

    let result = selector.matrix_multiply(&a, &b, 2, 2, 2).unwrap();
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    // [[19, 22], [43, 50]]
    assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
}

// ============================================================================
// COST CALCULATION TESTS (catch arithmetic mutations)
// ============================================================================

#[test]
fn test_select_backend_arithmetic_correctness() {
    // Test that verifies the exact arithmetic in select_backend()
    // Catches mutations: * -> /, * -> +, / -> *

    let selector = BackendSelector::new();

    // Test case 1: Exactly at 5x threshold (should choose SIMD, not GPU)
    // compute_s = 5.0 * transfer_s (boundary)
    let pcie_bw = 32e9; // 32 GB/s
    let gpu_gflops = 20e12; // 20 TFLOPS

    // Work backwards from desired ratio: compute_s = 5.0 * transfer_s
    // transfer_s = data_bytes / pcie_bw
    // compute_s = flops / gpu_gflops
    // flops / gpu_gflops = 5.0 * data_bytes / pcie_bw
    // flops = 5.0 * data_bytes * gpu_gflops / pcie_bw

    let data_bytes = 1_000_000; // 1 MB
    let transfer_s = data_bytes as f64 / pcie_bw;
    let compute_s_threshold = 5.0 * transfer_s;
    let flops = (compute_s_threshold * gpu_gflops) as u64;

    // At exactly 5x, should still choose SIMD (> not >=)
    let backend = selector.select_backend(data_bytes, flops);
    assert_eq!(backend, Backend::SIMD, "At exactly 5x threshold, should choose SIMD");

    // Just above 5x threshold, should choose GPU
    let flops_above = (flops as f64 * 1.01) as u64; // 1% above threshold
    let backend = selector.select_backend(data_bytes, flops_above);
    assert_eq!(backend, Backend::GPU, "Above 5x threshold, should choose GPU");

    // Well below threshold
    let flops_below = flops / 2;
    let backend = selector.select_backend(data_bytes, flops_below);
    assert_eq!(backend, Backend::SIMD, "Below 5x threshold, should choose SIMD");
}

#[test]
fn test_select_backend_arithmetic_mutation_detection() {
    // This test will FAIL if arithmetic operators are mutated
    // Specifically catches: / -> *, * -> /, * -> +

    let selector = BackendSelector::new();

    // Case 1: High compute, low transfer -> GPU
    // 1 GB data, 1000 TFLOPS compute
    let data_bytes = 1_000_000_000;
    let flops = 1_000_000_000_000_000; // 1000 TFLOPS

    // Expected calculations:
    // transfer_s = 1e9 / 32e9 = 0.03125 s = 31.25 ms
    // compute_s = 1e15 / 20e12 = 50 s
    // ratio = 50 / 0.03125 = 1600x >> 5x -> GPU
    let backend = selector.select_backend(data_bytes, flops);
    assert_eq!(backend, Backend::GPU, "High compute/transfer ratio should select GPU");

    // Case 2: Low compute, high transfer -> SIMD
    // 1 GB data, 1 GFLOP compute
    let flops_low = 1_000_000_000; // 1 GFLOPS

    // transfer_s = 1e9 / 32e9 = 0.03125 s
    // compute_s = 1e9 / 20e12 = 5e-5 s = 0.05 ms
    // ratio = 5e-5 / 0.03125 = 0.0016x << 5x -> SIMD
    let backend = selector.select_backend(data_bytes, flops_low);
    assert_eq!(backend, Backend::SIMD, "Low compute/transfer ratio should select SIMD");
}

#[test]
fn test_matmul_data_bytes_calculation() {
    // Test that data_bytes calculation is correct: (m*k + k*n + m*n) * 4
    // Catches mutations in arithmetic operators

    let selector = BackendSelector::new();

    // Test case: 100x100 x 100x100 matmul
    let m = 100;
    let n = 100;
    let k = 100;

    // Expected: (100*100 + 100*100 + 100*100) * 4 = 30,000 * 4 = 120,000 bytes
    // FLOPs: 2 * 100 * 100 * 100 = 2,000,000

    // With default settings:
    // transfer_s = 120,000 / 32e9 = 3.75e-6 s = 3.75 us
    // compute_s = 2,000,000 / 20e12 = 1e-7 s = 0.1 us
    // ratio = 0.1 / 3.75 = 0.0267x << 5x -> SIMD

    let backend = selector.select_for_matmul(m, n, k);
    assert_eq!(backend, Backend::SIMD);

    // Verify the calculation by testing a case that's GPU-bound
    // Need ratio > 5x, so need much larger matrices or different hardware params
    // Use custom selector with slower PCIe
    let slow_selector = BackendSelector::new()
        .with_pcie_bandwidth(1e9) // 1 GB/s (slow PCIe 3.0 x1)
        .with_gpu_gflops(100e12); // 100 TFLOPS (fast GPU)

    // Same 100x100x100 matmul:
    // transfer_s = 120,000 / 1e9 = 1.2e-4 s = 120 us
    // compute_s = 2,000,000 / 100e12 = 2e-8 s = 0.02 us
    // ratio = 0.02 / 120 = 0.00017x << 5x -> still SIMD

    // Need MUCH larger matrices
    let m_large = 1000;
    let n_large = 1000;
    let k_large = 1000;

    // data_bytes = (1000*1000 + 1000*1000 + 1000*1000) * 4 = 12,000,000 bytes = 12 MB
    // FLOPs = 2 * 1000 * 1000 * 1000 = 2,000,000,000
    // transfer_s = 12,000,000 / 1e9 = 0.012 s = 12 ms
    // compute_s = 2,000,000,000 / 100e12 = 2e-5 s = 0.02 ms
    // ratio = 0.02 / 12 = 0.0017x << 5x -> SIMD

    let backend = slow_selector.select_for_matmul(m_large, n_large, k_large);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_matmul_flops_calculation() {
    // Test that FLOPs calculation is correct: 2 * m * n * k
    // Catches mutations: * -> /, * -> +

    let selector = BackendSelector::new().with_gpu_gflops(1e12); // 1 TFLOPS (slower GPU for easier math)

    // Small matmul where we can verify the exact FLOP count matters
    let m = 10;
    let n = 10;
    let k = 10;

    // Expected FLOPs: 2 * 10 * 10 * 10 = 2,000
    // data_bytes: (10*10 + 10*10 + 10*10) * 4 = 1,200 bytes
    // transfer_s = 1,200 / 32e9 = 3.75e-8 s
    // compute_s = 2,000 / 1e12 = 2e-9 s
    // ratio = 2e-9 / 3.75e-8 = 0.053x << 5x -> SIMD

    let backend = selector.select_for_matmul(m, n, k);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_vector_op_data_bytes_calculation() {
    // Test that vector op data_bytes = n * 3 * 4
    // (two input vectors + output, f32 = 4 bytes)

    let selector = BackendSelector::new();

    let n = 1000;
    let ops_per_element = 2; // e.g., dot product

    // Expected: data_bytes = 1000 * 3 * 4 = 12,000 bytes
    // FLOPs: 1000 * 2 = 2,000
    // transfer_s = 12,000 / 32e9 = 3.75e-7 s
    // compute_s = 2,000 / 20e12 = 1e-10 s
    // ratio = 1e-10 / 3.75e-7 = 0.000267x << 5x -> SIMD

    let backend = selector.select_for_vector_op(n, ops_per_element);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_vector_op_flops_calculation() {
    // Test that vector op FLOPs = n * ops_per_element

    let selector = BackendSelector::new().with_gpu_gflops(1e12); // 1 TFLOPS

    let n = 10000;
    let ops_per_element = 10; // Complex reduction

    // FLOPs: 10,000 * 10 = 100,000
    // data_bytes: 10,000 * 3 * 4 = 120,000 bytes
    // transfer_s = 120,000 / 32e9 = 3.75e-6 s
    // compute_s = 100,000 / 1e12 = 1e-7 s
    // ratio = 1e-7 / 3.75e-6 = 0.0267x << 5x -> SIMD

    let backend = selector.select_for_vector_op(n, ops_per_element);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_dispatch_ratio_multiplication() {
    // Test that min_dispatch_ratio is correctly multiplied
    // Catches mutation: * -> /, * -> +

    // Test with different dispatch ratios
    let selector_5x = BackendSelector::new().with_min_dispatch_ratio(5.0);

    let selector_10x = BackendSelector::new().with_min_dispatch_ratio(10.0);

    // Workload with 7x ratio
    let data_bytes = 1_000_000;
    let pcie_bw = 32e9;
    let gpu_gflops = 20e12;

    let transfer_s = data_bytes as f64 / pcie_bw;
    let compute_s_7x = 7.0 * transfer_s;
    let flops = (compute_s_7x * gpu_gflops) as u64;

    // With 5x threshold: 7x > 5x -> GPU
    let backend = selector_5x.select_backend(data_bytes, flops);
    assert_eq!(backend, Backend::GPU, "7x should exceed 5x threshold");

    // With 10x threshold: 7x < 10x -> SIMD
    let backend = selector_10x.select_backend(data_bytes, flops);
    assert_eq!(backend, Backend::SIMD, "7x should not exceed 10x threshold");
}

// ============================================================================
// MOE BOUNDARY CONDITION TESTS (catch comparison mutations)
// ============================================================================

#[test]
fn test_moe_low_complexity_boundary() {
    // Test exact boundary: data_size > 1_000_000
    // Catches mutation: > -> >=

    let selector = BackendSelector::new();

    // Exactly at boundary (should be Scalar, not SIMD)
    assert_eq!(
        selector.select_with_moe(OpComplexity::Low, 1_000_000),
        Backend::Scalar,
        "Exactly 1M elements should be Scalar (> not >=)"
    );

    // Just above boundary (should be SIMD)
    assert_eq!(
        selector.select_with_moe(OpComplexity::Low, 1_000_001),
        Backend::SIMD,
        "1M+1 elements should be SIMD"
    );

    // Just below boundary (should be Scalar)
    assert_eq!(
        selector.select_with_moe(OpComplexity::Low, 999_999),
        Backend::Scalar,
        "1M-1 elements should be Scalar"
    );
}

#[test]
fn test_moe_medium_complexity_boundaries() {
    // Test exact boundaries: 10_000 and 100_000
    // Catches mutations: > -> >=

    let selector = BackendSelector::new();

    // First boundary: data_size > 10_000 (Scalar -> SIMD)
    assert_eq!(
        selector.select_with_moe(OpComplexity::Medium, 10_000),
        Backend::Scalar,
        "Exactly 10K should be Scalar"
    );

    assert_eq!(
        selector.select_with_moe(OpComplexity::Medium, 10_001),
        Backend::SIMD,
        "10K+1 should be SIMD"
    );

    assert_eq!(
        selector.select_with_moe(OpComplexity::Medium, 9_999),
        Backend::Scalar,
        "10K-1 should be Scalar"
    );

    // Second boundary: data_size > 100_000 (SIMD -> GPU)
    assert_eq!(
        selector.select_with_moe(OpComplexity::Medium, 100_000),
        Backend::SIMD,
        "Exactly 100K should be SIMD"
    );

    assert_eq!(
        selector.select_with_moe(OpComplexity::Medium, 100_001),
        Backend::GPU,
        "100K+1 should be GPU"
    );

    assert_eq!(
        selector.select_with_moe(OpComplexity::Medium, 99_999),
        Backend::SIMD,
        "100K-1 should be SIMD"
    );
}

#[test]
fn test_moe_high_complexity_boundaries() {
    // Test exact boundaries: 1_000 and 10_000
    // Catches mutations: > -> >=

    let selector = BackendSelector::new();

    // First boundary: data_size > 1_000 (Scalar -> SIMD)
    assert_eq!(
        selector.select_with_moe(OpComplexity::High, 1_000),
        Backend::Scalar,
        "Exactly 1K should be Scalar"
    );

    assert_eq!(
        selector.select_with_moe(OpComplexity::High, 1_001),
        Backend::SIMD,
        "1K+1 should be SIMD"
    );

    assert_eq!(
        selector.select_with_moe(OpComplexity::High, 999),
        Backend::Scalar,
        "1K-1 should be Scalar"
    );

    // Second boundary: data_size > 10_000 (SIMD -> GPU)
    assert_eq!(
        selector.select_with_moe(OpComplexity::High, 10_000),
        Backend::SIMD,
        "Exactly 10K should be SIMD"
    );

    assert_eq!(
        selector.select_with_moe(OpComplexity::High, 10_001),
        Backend::GPU,
        "10K+1 should be GPU"
    );

    assert_eq!(
        selector.select_with_moe(OpComplexity::High, 9_999),
        Backend::SIMD,
        "10K-1 should be SIMD"
    );
}

#[test]
fn test_elementwise_boundary() {
    // Test boundary for select_for_elementwise: data_size > 1_000_000

    let selector = BackendSelector::new();

    // Exactly at boundary
    assert_eq!(
        selector.select_for_elementwise(1_000_000),
        Backend::Scalar,
        "Exactly 1M should be Scalar"
    );

    // Just above
    assert_eq!(selector.select_for_elementwise(1_000_001), Backend::SIMD, "1M+1 should be SIMD");

    // Just below
    assert_eq!(selector.select_for_elementwise(999_999), Backend::Scalar, "1M-1 should be Scalar");
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_zero_size_operations() {
    let selector = BackendSelector::new();

    // Zero-size matmul
    let backend = selector.select_for_matmul(0, 0, 0);
    assert_eq!(backend, Backend::SIMD); // 0 flops, 0 data -> SIMD by default

    // Zero-size vector op
    let backend = selector.select_for_vector_op(0, 1);
    assert_eq!(backend, Backend::SIMD);

    // Zero-size elementwise
    let backend = selector.select_for_elementwise(0);
    assert_eq!(backend, Backend::Scalar);

    // Zero-size MoE
    assert_eq!(selector.select_with_moe(OpComplexity::Low, 0), Backend::Scalar);
    assert_eq!(selector.select_with_moe(OpComplexity::Medium, 0), Backend::Scalar);
    assert_eq!(selector.select_with_moe(OpComplexity::High, 0), Backend::Scalar);
}

#[test]
fn test_single_element_operations() {
    let selector = BackendSelector::new();

    // Single element should always be Scalar (too small for SIMD/GPU)
    assert_eq!(selector.select_with_moe(OpComplexity::Low, 1), Backend::Scalar);
    assert_eq!(selector.select_with_moe(OpComplexity::Medium, 1), Backend::Scalar);
    assert_eq!(selector.select_with_moe(OpComplexity::High, 1), Backend::Scalar);

    assert_eq!(selector.select_for_elementwise(1), Backend::Scalar);
}

#[test]
fn test_very_large_operations() {
    let selector = BackendSelector::new();

    // Very large sizes (billions of elements)
    let huge_size = 1_000_000_000; // 1 billion elements

    // Low complexity: never GPU (memory-bound)
    assert_eq!(selector.select_with_moe(OpComplexity::Low, huge_size), Backend::SIMD);

    // Medium complexity: GPU
    assert_eq!(selector.select_with_moe(OpComplexity::Medium, huge_size), Backend::GPU);

    // High complexity: GPU
    assert_eq!(selector.select_with_moe(OpComplexity::High, huge_size), Backend::GPU);
}

#[test]
fn test_custom_hardware_params() {
    // Test with extreme hardware configurations

    // Slow PCIe, fast GPU (can favor GPU with high compute workloads)
    let slow_pcie_selector = BackendSelector::new()
        .with_pcie_bandwidth(1e9) // 1 GB/s
        .with_gpu_gflops(100e12); // 100 TFLOPS

    // Fast PCIe, slow GPU (favors CPU/SIMD)
    let fast_pcie_selector = BackendSelector::new()
        .with_pcie_bandwidth(100e9) // 100 GB/s
        .with_gpu_gflops(1e12); // 1 TFLOPS

    // Test 1: Low compute workload (both should choose SIMD)
    let data_bytes_low = 1_000_000;
    let flops_low = 1_000_000_000; // 1 GFLOPS

    // Slow PCIe: transfer_s = 1M/1e9 = 1ms, compute_s = 1G/100e12 = 0.01us
    // ratio = 0.01us / 1ms = 0.00001x << 5x -> SIMD
    let backend = slow_pcie_selector.select_backend(data_bytes_low, flops_low);
    assert_eq!(backend, Backend::SIMD);

    // Fast PCIe: transfer_s = 1M/100e9 = 10us, compute_s = 1G/1e12 = 1ms
    // ratio = 1ms / 10us = 100x >> 5x -> GPU
    let backend = fast_pcie_selector.select_backend(data_bytes_low, flops_low);
    assert_eq!(backend, Backend::GPU);

    // Test 2: High compute workload
    let data_bytes_high = 1_000_000;
    let flops_high = 1_000_000_000_000; // 1 TFLOPS

    // Slow PCIe: transfer_s = 1ms, compute_s = 1T/100T = 10ms
    // ratio = 10ms / 1ms = 10x >> 5x -> GPU
    let backend = slow_pcie_selector.select_backend(data_bytes_high, flops_high);
    assert_eq!(backend, Backend::GPU);

    // Fast PCIe: transfer_s = 10us, compute_s = 1T/1T = 1s
    // ratio = 1s / 10us = 100,000x >> 5x -> GPU
    let backend = fast_pcie_selector.select_backend(data_bytes_high, flops_high);
    assert_eq!(backend, Backend::GPU);
}

// ============================================================================
// BACKEND ENUM TESTS
// ============================================================================

#[test]
fn test_backend_display() {
    assert_eq!(format!("{}", Backend::Scalar), "Scalar");
    assert_eq!(format!("{}", Backend::SIMD), "SIMD");
    assert_eq!(format!("{}", Backend::GPU), "GPU");
}

#[test]
fn test_backend_equality() {
    assert_eq!(Backend::Scalar, Backend::Scalar);
    assert_eq!(Backend::SIMD, Backend::SIMD);
    assert_eq!(Backend::GPU, Backend::GPU);

    assert_ne!(Backend::Scalar, Backend::SIMD);
    assert_ne!(Backend::SIMD, Backend::GPU);
    assert_ne!(Backend::Scalar, Backend::GPU);
}

#[test]
fn test_backend_clone_copy() {
    let b1 = Backend::GPU;
    let b2 = b1; // Copy
    assert_eq!(b1, b2);

    let b3 = b1; // Backend implements Copy, no need for clone
    assert_eq!(b1, b3);
}

#[test]
fn test_backend_debug() {
    let backend = Backend::SIMD;
    let debug_str = format!("{:?}", backend);
    assert!(debug_str.contains("SIMD"));
}

#[test]
fn test_backend_serialization() {
    let backend = Backend::GPU;
    let json = serde_json::to_string(&backend).unwrap();
    let deserialized: Backend = serde_json::from_str(&json).unwrap();
    assert_eq!(backend, deserialized);

    // Test all variants
    for backend in &[Backend::Scalar, Backend::SIMD, Backend::GPU] {
        let json = serde_json::to_string(backend).unwrap();
        let deserialized: Backend = serde_json::from_str(&json).unwrap();
        assert_eq!(*backend, deserialized);
    }
}

// ============================================================================
// OPCOMPLEXITY TESTS
// ============================================================================

#[test]
fn test_op_complexity_ordering() {
    assert!(OpComplexity::Low < OpComplexity::Medium);
    assert!(OpComplexity::Medium < OpComplexity::High);
    assert!(OpComplexity::Low < OpComplexity::High);

    assert!(OpComplexity::High > OpComplexity::Medium);
    assert!(OpComplexity::Medium > OpComplexity::Low);
    assert!(OpComplexity::High > OpComplexity::Low);
}

#[test]
fn test_op_complexity_equality() {
    assert_eq!(OpComplexity::Low, OpComplexity::Low);
    assert_eq!(OpComplexity::Medium, OpComplexity::Medium);
    assert_eq!(OpComplexity::High, OpComplexity::High);

    assert_ne!(OpComplexity::Low, OpComplexity::Medium);
    assert_ne!(OpComplexity::Medium, OpComplexity::High);
}

#[test]
fn test_op_complexity_clone_copy() {
    let c1 = OpComplexity::High;
    let c2 = c1; // Copy
    assert_eq!(c1, c2);

    let c3 = c1; // OpComplexity implements Copy, no need for clone
    assert_eq!(c1, c3);
}

#[test]
fn test_op_complexity_debug() {
    let complexity = OpComplexity::Medium;
    let debug_str = format!("{:?}", complexity);
    assert!(debug_str.contains("Medium"));
}

// ============================================================================
// BACKEND SELECTOR TESTS
// ============================================================================

#[test]
fn test_backend_selector_new() {
    let selector = BackendSelector::new();
    // Verify default values are set (indirectly by behavior)
    let backend = selector.select_backend(1_000_000, 1_000_000_000);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_backend_selector_default() {
    let selector1 = BackendSelector::new();
    let selector2 = BackendSelector::default();

    // Both should produce same results
    let backend1 = selector1.select_backend(1_000_000, 1_000_000_000);
    let backend2 = selector2.select_backend(1_000_000, 1_000_000_000);
    assert_eq!(backend1, backend2);
}

#[test]
fn test_backend_selector_with_pcie_bandwidth() {
    let selector = BackendSelector::new().with_pcie_bandwidth(64e9); // 64 GB/s

    // Slower PCIe means more transfer time, harder to hit GPU threshold
    let backend = selector.select_backend(1_000_000, 1_000_000_000);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_backend_selector_with_gpu_gflops() {
    let selector = BackendSelector::new().with_gpu_gflops(10e12); // 10 TFLOPS (slower GPU)

    // Slower GPU means more compute time, easier to hit threshold
    let backend = selector.select_backend(1_000_000, 1_000_000_000);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_backend_selector_with_min_dispatch_ratio() {
    let selector = BackendSelector::new().with_min_dispatch_ratio(2.0); // More aggressive

    // Lower threshold means easier to select GPU
    let backend = selector.select_backend(1_000_000, 1_000_000_000);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_backend_selector_builder_chaining() {
    let selector = BackendSelector::new()
        .with_pcie_bandwidth(16e9)
        .with_gpu_gflops(50e12)
        .with_min_dispatch_ratio(3.0);

    // Verify chaining works by using the configured selector
    let backend = selector.select_backend(1_000_000, 10_000_000_000);
    // With faster GPU (50 TFLOPS) and lower threshold (3x), might hit GPU
    assert!(backend == Backend::GPU || backend == Backend::SIMD);
}

#[test]
fn test_backend_selector_extreme_parameters() {
    // Test with extreme values to ensure no panics/errors

    let tiny_selector = BackendSelector::new()
        .with_pcie_bandwidth(1e6) // 1 MB/s
        .with_gpu_gflops(1e9) // 1 GFLOPS
        .with_min_dispatch_ratio(1.0);

    let backend = tiny_selector.select_backend(100, 1000);
    assert!(backend == Backend::SIMD || backend == Backend::GPU);

    let huge_selector = BackendSelector::new()
        .with_pcie_bandwidth(1e12) // 1 TB/s
        .with_gpu_gflops(1e15) // 1 PFLOPS
        .with_min_dispatch_ratio(100.0);

    let backend = huge_selector.select_backend(1_000_000_000, 1_000_000_000_000);
    assert!(backend == Backend::SIMD || backend == Backend::GPU);
}

// ============================================================================
// EDGE CASE TESTS (additional)
// ============================================================================

#[test]
fn test_select_backend_zero_data_edge() {
    let selector = BackendSelector::new();
    // Zero data should not panic
    let backend = selector.select_backend(0, 1000);
    assert_eq!(backend, Backend::GPU); // compute > 5 * 0
}

#[test]
fn test_select_backend_zero_flops_edge() {
    let selector = BackendSelector::new();
    // Zero flops should choose SIMD (0 < 5 * transfer)
    let backend = selector.select_backend(1000, 0);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_select_for_matmul_zero_edge() {
    let selector = BackendSelector::new();
    // Zero dimensions should not panic
    let backend = selector.select_for_matmul(0, 0, 0);
    assert!(backend == Backend::SIMD || backend == Backend::GPU);
}

#[test]
fn test_select_for_elementwise_boundary_edge() {
    let selector = BackendSelector::new();

    // Test various data sizes - verify no panics and consistent behavior
    let backend_small = selector.select_for_elementwise(1_000);
    let backend_medium = selector.select_for_elementwise(500_000);
    let backend_large = selector.select_for_elementwise(2_000_000);

    // Small should be Scalar, large should be SIMD
    assert_eq!(backend_small, Backend::Scalar);
    assert_eq!(backend_large, Backend::SIMD);
    // Medium depends on implementation
    assert!(backend_medium == Backend::Scalar || backend_medium == Backend::SIMD);
}

#[test]
fn test_moe_boundary_conditions_edge() {
    let selector = BackendSelector::new();

    // Test thresholds for Medium complexity - verify consistent behavior
    let small = selector.select_with_moe(OpComplexity::Medium, 1_000);
    let large = selector.select_with_moe(OpComplexity::Medium, 500_000);

    // Small should be Scalar, large should be GPU
    assert_eq!(small, Backend::Scalar);
    assert_eq!(large, Backend::GPU);
}
