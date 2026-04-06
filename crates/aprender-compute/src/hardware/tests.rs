use super::*;

#[test]
fn test_simd_detection() {
    let simd = detect_simd();
    // Should detect at least scalar
    assert!(simd.lanes() >= 1);
}

#[test]
fn test_simd_lanes() {
    assert_eq!(SimdWidth::Scalar.lanes(), 1);
    assert_eq!(SimdWidth::Avx2.lanes(), 8);
    assert_eq!(SimdWidth::Avx512.lanes(), 16);
}

#[test]
fn test_hardware_detection() {
    let cap = HardwareCapability::detect();
    assert!(cap.cpu.cores > 0);
    assert!(cap.cpu.peak_gflops > 0.0);
}

#[test]
fn test_bottleneck_classification() {
    let cap = HardwareCapability::detect();

    // Low arithmetic intensity = memory bound
    assert_eq!(cap.bottleneck(0.1, false), Bottleneck::Memory);

    // High arithmetic intensity = compute bound
    assert_eq!(cap.bottleneck(1000.0, false), Bottleneck::Compute);
}

#[test]
fn test_roofline_throughput() {
    let cap = HardwareCapability::detect();

    // Memory-bound workload
    let low_ai = cap.expected_throughput_gflops(0.1, false);
    // Compute-bound workload
    let high_ai = cap.expected_throughput_gflops(1000.0, false);

    // Low AI should give lower throughput (memory limited)
    assert!(low_ai < high_ai);
}

#[test]
fn test_toml_roundtrip() {
    let cap = HardwareCapability::detect();
    let toml_str = toml::to_string_pretty(&cap).unwrap();
    let parsed: HardwareCapability = toml::from_str(&toml_str).unwrap();
    assert_eq!(cap.cpu.cores, parsed.cpu.cores);
}

#[test]
fn test_byte_budget_in_hardware_toml() {
    let cap = HardwareCapability::detect();

    // Should have byte_budget populated
    assert!(cap.byte_budget.is_some());
    let budget = cap.byte_budget.unwrap();
    assert!(budget.gb_per_sec > 0.0);
    assert!(budget.gb_per_sec <= 25.0); // Capped at zstd max

    // Test TOML roundtrip preserves byte_budget
    let toml_str = toml::to_string_pretty(&cap).unwrap();
    assert!(toml_str.contains("[byte_budget]"));
    assert!(toml_str.contains("gb_per_sec"));

    let parsed: HardwareCapability = toml::from_str(&toml_str).unwrap();
    assert!(parsed.byte_budget.is_some());
    let parsed_budget = parsed.byte_budget.unwrap();
    assert!((parsed_budget.gb_per_sec - budget.gb_per_sec).abs() < 0.001);
}

#[test]
fn test_byte_budget_backward_compat() {
    // Test that hardware.toml without byte_budget still parses
    let toml_without_budget = r#"
timestamp = "2026-01-13T18:00:00Z"
hostname = "test"

[cpu]
vendor = "Intel"
model = "Test CPU"
cores = 4
threads = 8
simd = "Avx2"
base_freq_ghz = 3.5
peak_gflops = 112.0
memory_bw_gbps = 80.0

[roofline]
cpu_arithmetic_intensity = 1.4
"#;
    let parsed: HardwareCapability = toml::from_str(toml_without_budget).unwrap();
    // byte_budget should be None when not in TOML (backward compat)
    assert!(parsed.byte_budget.is_none());
}

// Additional coverage tests

#[test]
fn test_simd_width_bits() {
    assert_eq!(SimdWidth::Scalar.bits(), 32);
    assert_eq!(SimdWidth::Sse2.bits(), 128);
    assert_eq!(SimdWidth::Neon128.bits(), 128);
    assert_eq!(SimdWidth::WasmSimd128.bits(), 128);
    assert_eq!(SimdWidth::Avx2.bits(), 256);
    assert_eq!(SimdWidth::Avx512.bits(), 512);
}

#[test]
fn test_simd_width_compute_speedup() {
    assert!((SimdWidth::Scalar.compute_speedup() - 1.0).abs() < 0.01);
    assert!((SimdWidth::Sse2.compute_speedup() - 4.0).abs() < 0.01);
    assert!((SimdWidth::Neon128.compute_speedup() - 4.0).abs() < 0.01);
    assert!((SimdWidth::WasmSimd128.compute_speedup() - 4.0).abs() < 0.01);
    assert!((SimdWidth::Avx2.compute_speedup() - 10.0).abs() < 0.01);
    assert!((SimdWidth::Avx512.compute_speedup() - 12.0).abs() < 0.01);
}

#[test]
fn test_best_backend() {
    let cap = HardwareCapability::detect();
    let backend = cap.best_backend();
    // Should be a valid backend (even if None)
    assert!(matches!(
        backend,
        GpuBackend::None
            | GpuBackend::Cuda
            | GpuBackend::Metal
            | GpuBackend::Vulkan
            | GpuBackend::Wgpu
    ));
}

#[test]
fn test_load_or_detect_creates_new() {
    use std::path::PathBuf;
    let tmp_path = PathBuf::from("/tmp/trueno_test_nonexistent_12345.toml");
    // Ensure it doesn't exist
    let _ = std::fs::remove_file(&tmp_path);

    let cap = HardwareCapability::load_or_detect(&tmp_path);
    assert!(cap.cpu.cores > 0);

    // Cleanup
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_save_and_load() {
    use std::path::PathBuf;
    let tmp_path = PathBuf::from("/tmp/trueno_test_save_load.toml");

    let original = HardwareCapability::detect();
    original.save(&tmp_path).expect("Failed to save");

    let loaded = HardwareCapability::load_or_detect(&tmp_path);
    assert_eq!(original.cpu.cores, loaded.cpu.cores);
    assert_eq!(original.hostname, loaded.hostname);

    // Cleanup
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_expected_throughput_with_gpu() {
    // Create a capability with GPU
    let cap = HardwareCapability::detect();

    // Test GPU branch if available
    if cap.gpu.is_some() {
        let throughput_gpu = cap.expected_throughput_gflops(10.0, true);
        let throughput_cpu = cap.expected_throughput_gflops(10.0, false);
        // Both should be positive
        assert!(throughput_gpu > 0.0);
        assert!(throughput_cpu > 0.0);
    }

    // CPU path should always work
    let cpu_throughput = cap.expected_throughput_gflops(10.0, false);
    assert!(cpu_throughput > 0.0);
}

#[test]
fn test_expected_throughput_with_fake_gpu() {
    // Create a capability with a fake GPU to test GPU branch
    let mut cap = HardwareCapability::detect();
    cap.gpu = Some(GpuCapability {
        vendor: "Test".to_string(),
        model: "Fake GPU".to_string(),
        backend: GpuBackend::Cuda,
        compute_capability: Some("8.9".to_string()),
        peak_tflops_fp32: 100.0,
        peak_tflops_tensor: Some(400.0),
        memory_bw_gbps: 1000.0,
        vram_gb: 24.0,
    });

    // This should exercise the GPU branch
    let throughput_gpu = cap.expected_throughput_gflops(10.0, true);
    let throughput_cpu = cap.expected_throughput_gflops(10.0, false);

    // GPU with 1000 GB/s and AI=10: memory_bound = 10000 GFLOPS
    // GPU compute = 100 TFLOPS = 100000 GFLOPS
    // Result should be min(10000, 100000) = 10000
    assert!((throughput_gpu - 10000.0).abs() < 1.0);
    assert!(throughput_cpu > 0.0);
}

#[test]
fn test_load_invalid_toml() {
    let tmp_path = std::env::temp_dir().join("trueno_test_invalid.toml");

    // Write invalid TOML
    std::fs::write(&tmp_path, "this is not valid toml [[[").expect("Failed to write");

    // Should fall back to detect
    let cap = HardwareCapability::load_or_detect(&tmp_path);
    assert!(cap.cpu.cores > 0);

    // Cleanup
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_expected_throughput_no_gpu_fallback() {
    // Create a capability without GPU
    let mut cap = HardwareCapability::detect();
    cap.gpu = None;

    // Request GPU but none available - should fallback to CPU
    let throughput_gpu_request = cap.expected_throughput_gflops(10.0, true);
    let throughput_cpu = cap.expected_throughput_gflops(10.0, false);

    // Both should give the same result since there's no GPU
    assert!((throughput_gpu_request - throughput_cpu).abs() < 0.001);
    assert!(throughput_cpu > 0.0);
}

#[test]
fn test_bottleneck_with_gpu() {
    // Create capability with fake GPU to test GPU bottleneck path
    let mut cap = HardwareCapability::detect();
    cap.gpu = Some(GpuCapability {
        vendor: "Test".to_string(),
        model: "Fake GPU".to_string(),
        backend: GpuBackend::Cuda,
        compute_capability: Some("8.9".to_string()),
        peak_tflops_fp32: 100.0,
        peak_tflops_tensor: Some(400.0),
        memory_bw_gbps: 1000.0,
        vram_gb: 24.0,
    });
    cap.roofline.gpu_arithmetic_intensity = Some(50.0);

    // Low AI should be memory bound
    assert_eq!(cap.bottleneck(10.0, true), Bottleneck::Memory);

    // High AI should be compute bound
    assert_eq!(cap.bottleneck(100.0, true), Bottleneck::Compute);

    // Test edge case at threshold
    assert_eq!(cap.bottleneck(50.0, true), Bottleneck::Compute);
}

#[test]
fn test_bottleneck_gpu_without_ai() {
    // Create capability with GPU but no gpu_arithmetic_intensity
    let mut cap = HardwareCapability::detect();
    cap.gpu = Some(GpuCapability {
        vendor: "Test".to_string(),
        model: "Fake GPU".to_string(),
        backend: GpuBackend::Cuda,
        compute_capability: None,
        peak_tflops_fp32: 50.0,
        peak_tflops_tensor: None,
        memory_bw_gbps: 500.0,
        vram_gb: 8.0,
    });
    cap.roofline.gpu_arithmetic_intensity = None; // No GPU AI set

    // When gpu_arithmetic_intensity is None, uses f64::MAX as threshold
    // So any finite AI should be memory bound
    assert_eq!(cap.bottleneck(1000.0, true), Bottleneck::Memory);
}

#[test]
fn test_simd_width_neon() {
    // Test NEON SIMD width (4 lanes)
    assert_eq!(SimdWidth::Neon128.lanes(), 4);
}

#[test]
fn test_simd_width_sse2() {
    // Test SSE2 SIMD width (4 lanes)
    assert_eq!(SimdWidth::Sse2.lanes(), 4);
}

#[test]
fn test_best_backend_without_gpu() {
    let mut cap = HardwareCapability::detect();
    cap.gpu = None;

    // Should return None backend when no GPU
    assert_eq!(cap.best_backend(), GpuBackend::None);
}

#[test]
fn test_best_backend_with_gpu() {
    let mut cap = HardwareCapability::detect();
    cap.gpu = Some(GpuCapability {
        vendor: "NVIDIA".to_string(),
        model: "RTX 4090".to_string(),
        backend: GpuBackend::Cuda,
        compute_capability: Some("8.9".to_string()),
        peak_tflops_fp32: 82.58,
        peak_tflops_tensor: Some(330.3),
        memory_bw_gbps: 1008.0,
        vram_gb: 24.0,
    });

    assert_eq!(cap.best_backend(), GpuBackend::Cuda);
}

#[test]
fn test_gpu_backend_variants() {
    // Test all GPU backend variants
    assert_ne!(GpuBackend::None, GpuBackend::Cuda);
    assert_ne!(GpuBackend::Cuda, GpuBackend::Vulkan);
    assert_ne!(GpuBackend::Vulkan, GpuBackend::Metal);

    // Test debug formatting
    let debug_str = format!("{:?}", GpuBackend::Cuda);
    assert!(debug_str.contains("Cuda"));
}
