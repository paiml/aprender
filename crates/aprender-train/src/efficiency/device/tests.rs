//! Tests for device module.

use super::*;

#[test]
fn test_simd_capability_vector_width() {
    assert_eq!(SimdCapability::None.vector_width_bits(), 0);
    assert_eq!(SimdCapability::Sse4.vector_width_bits(), 128);
    assert_eq!(SimdCapability::Avx2.vector_width_bits(), 256);
    assert_eq!(SimdCapability::Avx512.vector_width_bits(), 512);
    assert_eq!(SimdCapability::Neon.vector_width_bits(), 128);
}

#[test]
fn test_simd_capability_display() {
    assert_eq!(format!("{}", SimdCapability::Avx2), "AVX2");
    assert_eq!(format!("{}", SimdCapability::Neon), "NEON");
}

#[test]
fn test_simd_capability_detect() {
    let simd = SimdCapability::detect();
    // Should return something (even None is valid)
    let _ = simd.vector_width_bits();
}

#[test]
fn test_cpu_info_new() {
    let cpu = CpuInfo::new(8, 16, SimdCapability::Avx2, "Intel Core i9-12900K");

    assert_eq!(cpu.cores, 8);
    assert_eq!(cpu.threads, 16);
    assert_eq!(cpu.simd, SimdCapability::Avx2);
    assert_eq!(cpu.model, "Intel Core i9-12900K");
}

#[test]
fn test_cpu_info_with_cache() {
    let cpu = CpuInfo::new(8, 16, SimdCapability::Avx2, "Test CPU").with_cache(30 * 1024 * 1024); // 30 MB

    assert_eq!(cpu.cache_bytes, 30 * 1024 * 1024);
}

#[test]
fn test_cpu_info_detect() {
    let cpu = CpuInfo::detect();

    // Should detect at least 1 core
    assert!(cpu.cores >= 1);
    assert!(cpu.threads >= cpu.cores);
    assert!(!cpu.model.is_empty());
}

#[test]
fn test_gpu_info_new() {
    let gpu = GpuInfo::new("NVIDIA RTX 4090", 24 * 1024 * 1024 * 1024);

    assert_eq!(gpu.name, "NVIDIA RTX 4090");
    assert_eq!(gpu.vram_bytes, 24 * 1024 * 1024 * 1024);
    assert!(gpu.compute_capability.is_none());
}

#[test]
fn test_gpu_info_with_compute_capability() {
    let gpu = GpuInfo::new("RTX 4090", 24 * 1024 * 1024 * 1024).with_compute_capability(8, 9);

    assert_eq!(gpu.compute_capability, Some((8, 9)));
    assert!(gpu.supports_compute_capability(8, 0));
    assert!(gpu.supports_compute_capability(8, 9));
    assert!(!gpu.supports_compute_capability(9, 0));
}

#[test]
fn test_gpu_info_vram_gb() {
    let gpu = GpuInfo::new("Test GPU", 8 * 1024 * 1024 * 1024);
    assert!((gpu.vram_gb() - 8.0).abs() < 0.01);
}

#[test]
fn test_tpu_info_new() {
    let tpu = TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024);

    assert_eq!(tpu.version, "v4");
    assert_eq!(tpu.cores, 8);
    assert!((tpu.hbm_gb() - 32.0).abs() < 0.01);
}

#[test]
fn test_apple_silicon_info() {
    let m2 = AppleSiliconInfo::new("Apple M2 Pro")
        .with_cores(8, 4, 19)
        .with_memory(32 * 1024 * 1024 * 1024);

    assert_eq!(m2.chip, "Apple M2 Pro");
    assert_eq!(m2.p_cores, 8);
    assert_eq!(m2.e_cores, 4);
    assert_eq!(m2.gpu_cores, 19);
    assert_eq!(m2.total_cpu_cores(), 12);
    assert!((m2.unified_memory_gb() - 32.0).abs() < 0.01);
}

#[test]
fn test_compute_device_detect_returns_cpu() {
    let devices = ComputeDevice::detect();

    // Should always detect at least one CPU
    assert!(!devices.is_empty());
    assert!(devices.iter().any(ComputeDevice::is_cpu));
}

#[test]
fn test_compute_device_is_methods() {
    let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Test"));
    let gpu = ComputeDevice::Gpu(GpuInfo::new("Test GPU", 8 * 1024 * 1024 * 1024));
    let tpu = ComputeDevice::Tpu(TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024));
    let apple = ComputeDevice::AppleSilicon(AppleSiliconInfo::new("M2"));

    assert!(cpu.is_cpu());
    assert!(!cpu.is_gpu());

    assert!(gpu.is_gpu());
    assert!(!gpu.is_cpu());

    assert!(tpu.is_tpu());
    assert!(!tpu.is_cpu());

    assert!(apple.is_apple_silicon());
    assert!(!apple.is_cpu());
}

#[test]
fn test_compute_device_memory_bytes() {
    let gpu = ComputeDevice::Gpu(GpuInfo::new("Test", 16 * 1024 * 1024 * 1024));
    assert_eq!(gpu.memory_bytes(), 16 * 1024 * 1024 * 1024);

    let tpu = ComputeDevice::Tpu(TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024));
    assert_eq!(tpu.memory_bytes(), 32 * 1024 * 1024 * 1024);

    let apple = ComputeDevice::AppleSilicon(
        AppleSiliconInfo::new("M2").with_memory(24 * 1024 * 1024 * 1024),
    );
    assert_eq!(apple.memory_bytes(), 24 * 1024 * 1024 * 1024);
}

#[test]
fn test_compute_device_name() {
    let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Intel i9"));
    assert_eq!(cpu.name(), "Intel i9");

    let gpu = ComputeDevice::Gpu(GpuInfo::new("RTX 4090", 24 * 1024 * 1024 * 1024));
    assert_eq!(gpu.name(), "RTX 4090");
}

#[test]
fn test_compute_device_display() {
    let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Intel i9"));
    let display = format!("{cpu}");
    assert!(display.contains("Intel i9"));
    assert!(display.contains("8 cores"));
    assert!(display.contains("AVX2"));

    let gpu = ComputeDevice::Gpu(
        GpuInfo::new("RTX 4090", 24 * 1024 * 1024 * 1024).with_compute_capability(8, 9),
    );
    let display = format!("{gpu}");
    assert!(display.contains("RTX 4090"));
    assert!(display.contains("24.0 GB"));
    assert!(display.contains("SM 8.9"));
}

#[test]
fn test_compute_device_relative_power() {
    let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Test"));
    let gpu = ComputeDevice::Gpu(GpuInfo::new("Test", 16 * 1024 * 1024 * 1024));

    // GPU should have higher relative power
    assert!(gpu.relative_compute_power() > cpu.relative_compute_power());
}

#[test]
fn test_compute_device_serialization() {
    let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Test CPU"));
    let json = serde_json::to_string(&cpu).expect("JSON serialization should succeed");
    let parsed: ComputeDevice =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert!(parsed.is_cpu());
    assert_eq!(parsed.name(), "Test CPU");
}

#[test]
fn test_simd_capability_default() {
    let simd: SimdCapability = Default::default();
    assert_eq!(simd, SimdCapability::None);
    assert_eq!(simd.vector_width_bits(), 0);
}

#[test]
fn test_simd_capability_serde_all_variants() {
    let variants = vec![
        SimdCapability::None,
        SimdCapability::Sse4,
        SimdCapability::Avx2,
        SimdCapability::Avx512,
        SimdCapability::Neon,
    ];

    for variant in variants {
        let json = serde_json::to_string(&variant).expect("JSON serialization should succeed");
        let parsed: SimdCapability =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(variant, parsed);
    }
}

#[test]
fn test_simd_capability_display_all_variants() {
    assert_eq!(format!("{}", SimdCapability::None), "none");
    assert_eq!(format!("{}", SimdCapability::Sse4), "SSE4");
    assert_eq!(format!("{}", SimdCapability::Avx2), "AVX2");
    assert_eq!(format!("{}", SimdCapability::Avx512), "AVX-512");
    assert_eq!(format!("{}", SimdCapability::Neon), "NEON");
}

#[test]
fn test_simd_capability_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(SimdCapability::Avx2);
    set.insert(SimdCapability::Neon);
    set.insert(SimdCapability::Avx2); // Duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn test_simd_capability_clone_copy() {
    let original = SimdCapability::Avx512;
    let copied = original;
    let cloned = original;
    assert_eq!(original, copied);
    assert_eq!(original, cloned);
}

#[test]
fn test_cpu_info_estimated_bandwidth() {
    let cpu = CpuInfo::new(8, 16, SimdCapability::Avx2, "Test");
    let bandwidth = cpu.estimated_memory_bandwidth_gbps();
    // 40.0 * (8.0 / 8.0).min(2.0) = 40.0 * 1.0 = 40.0
    assert!((bandwidth - 40.0).abs() < 0.01);

    let big_cpu = CpuInfo::new(32, 64, SimdCapability::Avx512, "Big CPU");
    let big_bandwidth = big_cpu.estimated_memory_bandwidth_gbps();
    // 40.0 * (32.0 / 8.0).min(2.0) = 40.0 * 2.0 = 80.0
    assert!((big_bandwidth - 80.0).abs() < 0.01);
}

#[test]
fn test_cpu_info_serde() {
    let cpu = CpuInfo::new(8, 16, SimdCapability::Avx2, "Test CPU").with_cache(30_000_000);
    let json = serde_json::to_string(&cpu).expect("JSON serialization should succeed");
    let parsed: CpuInfo = serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(cpu.cores, parsed.cores);
    assert_eq!(cpu.threads, parsed.threads);
    assert_eq!(cpu.simd, parsed.simd);
    assert_eq!(cpu.model, parsed.model);
    assert_eq!(cpu.cache_bytes, parsed.cache_bytes);
}

#[test]
fn test_cpu_info_debug() {
    let cpu = CpuInfo::new(4, 8, SimdCapability::Sse4, "Debug CPU");
    let debug = format!("{cpu:?}");
    assert!(debug.contains("Debug CPU"));
    assert!(debug.contains("Sse4"));
}

#[test]
fn test_gpu_info_with_index() {
    let gpu = GpuInfo::new("GPU 1", 8 * 1024 * 1024 * 1024).with_index(1);
    assert_eq!(gpu.index, 1);
}

#[test]
fn test_gpu_info_supports_capability_edge_cases() {
    let gpu = GpuInfo::new("Test", 8 * 1024 * 1024 * 1024).with_compute_capability(7, 5);
    // Same major, higher minor - should fail
    assert!(!gpu.supports_compute_capability(7, 6));
    // Same major, same minor - should pass
    assert!(gpu.supports_compute_capability(7, 5));
    // Lower major - should pass
    assert!(gpu.supports_compute_capability(6, 9));

    // GPU without compute capability
    let gpu_no_cap = GpuInfo::new("Test", 8 * 1024 * 1024 * 1024);
    assert!(!gpu_no_cap.supports_compute_capability(7, 0));
}

#[test]
fn test_gpu_info_serde() {
    let gpu = GpuInfo::new("Test GPU", 16 * 1024 * 1024 * 1024)
        .with_compute_capability(8, 6)
        .with_index(0);
    let json = serde_json::to_string(&gpu).expect("JSON serialization should succeed");
    let parsed: GpuInfo = serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(gpu.name, parsed.name);
    assert_eq!(gpu.vram_bytes, parsed.vram_bytes);
    assert_eq!(gpu.compute_capability, parsed.compute_capability);
    assert_eq!(gpu.index, parsed.index);
}

#[test]
fn test_tpu_info_serde() {
    let tpu = TpuInfo::new("v5e", 16, 64 * 1024 * 1024 * 1024);
    let json = serde_json::to_string(&tpu).expect("JSON serialization should succeed");
    let parsed: TpuInfo = serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(tpu.version, parsed.version);
    assert_eq!(tpu.cores, parsed.cores);
    assert_eq!(tpu.hbm_bytes, parsed.hbm_bytes);
}

#[test]
fn test_apple_silicon_info_serde() {
    let apple =
        AppleSiliconInfo::new("M3 Max").with_cores(12, 4, 40).with_memory(64 * 1024 * 1024 * 1024);
    let json = serde_json::to_string(&apple).expect("JSON serialization should succeed");
    let parsed: AppleSiliconInfo =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(apple.chip, parsed.chip);
    assert_eq!(apple.p_cores, parsed.p_cores);
    assert_eq!(apple.e_cores, parsed.e_cores);
    assert_eq!(apple.gpu_cores, parsed.gpu_cores);
    assert_eq!(apple.unified_memory_bytes, parsed.unified_memory_bytes);
}

#[test]
fn test_apple_silicon_default_neural_cores() {
    let apple = AppleSiliconInfo::new("M1");
    assert_eq!(apple.neural_cores, 16);
}

#[test]
fn test_compute_device_compute_units_all_variants() {
    let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "CPU"));
    assert_eq!(cpu.compute_units(), 16); // threads

    let gpu = ComputeDevice::Gpu(GpuInfo::new("GPU", 8 * 1024 * 1024 * 1024));
    assert_eq!(gpu.compute_units(), 0); // Unknown for GPU

    let tpu = ComputeDevice::Tpu(TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024));
    assert_eq!(tpu.compute_units(), 8); // cores

    let apple = ComputeDevice::AppleSilicon(AppleSiliconInfo::new("M2").with_cores(8, 4, 10));
    assert_eq!(apple.compute_units(), 22); // 8 + 4 + 10
}

#[test]
fn test_compute_device_memory_bytes_cpu() {
    let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "CPU"));
    // 8 cores * 4GB per core = 32GB
    let expected = 8_u64 * 4 * 1024 * 1024 * 1024;
    assert_eq!(cpu.memory_bytes(), expected);
}

#[test]
fn test_compute_device_display_tpu() {
    let tpu = ComputeDevice::Tpu(TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024));
    let display = format!("{tpu}");
    assert!(display.contains("TPU: v4"));
    assert!(display.contains("8 cores"));
    assert!(display.contains("32.0 GB"));
}

#[test]
fn test_compute_device_display_apple_silicon() {
    let apple = ComputeDevice::AppleSilicon(
        AppleSiliconInfo::new("Apple M2 Pro")
            .with_cores(8, 4, 19)
            .with_memory(32 * 1024 * 1024 * 1024),
    );
    let display = format!("{apple}");
    assert!(display.contains("Apple M2 Pro"));
    assert!(display.contains("8P+4E"));
    assert!(display.contains("19 GPU"));
    assert!(display.contains("32.0 GB"));
}

#[test]
fn test_compute_device_display_gpu_without_compute_capability() {
    let gpu = ComputeDevice::Gpu(GpuInfo::new("AMD RX 7900", 24 * 1024 * 1024 * 1024));
    let display = format!("{gpu}");
    assert!(display.contains("AMD RX 7900"));
    assert!(display.contains("24.0 GB"));
    assert!(!display.contains("SM")); // No compute capability
}

#[test]
fn test_compute_device_relative_power_tpu() {
    let tpu = ComputeDevice::Tpu(TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024));
    let power = tpu.relative_compute_power();
    // 50.0 * (8.0 / 8.0) = 50.0
    assert!((power - 50.0).abs() < 0.01);
}

#[test]
fn test_compute_device_relative_power_apple_silicon() {
    let apple = ComputeDevice::AppleSilicon(AppleSiliconInfo::new("M2").with_cores(8, 4, 10));
    let power = apple.relative_compute_power();
    // (8 * 1.5 + 4 * 0.5) / 8.0 + 10 * 0.5 = (12 + 2) / 8 + 5 = 1.75 + 5 = 6.75
    assert!((power - 6.75).abs() < 0.01);
}

#[test]
fn test_compute_device_serialization_all_variants() {
    let devices = vec![
        ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "CPU")),
        ComputeDevice::Gpu(
            GpuInfo::new("GPU", 8 * 1024 * 1024 * 1024).with_compute_capability(8, 0),
        ),
        ComputeDevice::Tpu(TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024)),
        ComputeDevice::AppleSilicon(AppleSiliconInfo::new("M2").with_cores(8, 4, 10)),
    ];

    for device in devices {
        let json = serde_json::to_string(&device).expect("JSON serialization should succeed");
        let parsed: ComputeDevice =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(device.name(), parsed.name());
    }
}

#[test]
fn test_compute_device_name_all_variants() {
    let tpu = ComputeDevice::Tpu(TpuInfo::new("v5p", 16, 64 * 1024 * 1024 * 1024));
    assert_eq!(tpu.name(), "v5p");

    let apple = ComputeDevice::AppleSilicon(AppleSiliconInfo::new("M3 Ultra"));
    assert_eq!(apple.name(), "M3 Ultra");
}

#[test]
fn test_apple_silicon_detect() {
    // This should return None on non-macOS or Some on macOS Apple Silicon
    let result = AppleSiliconInfo::detect();
    // We can't assert the result since it's platform-dependent
    // Just ensure it doesn't panic
    let _ = result;
}

#[test]
fn test_cpu_info_clone() {
    let original = CpuInfo::new(8, 16, SimdCapability::Avx2, "Original");
    let cloned = original.clone();
    assert_eq!(original.cores, cloned.cores);
    assert_eq!(original.model, cloned.model);
}

#[test]
fn test_gpu_info_clone() {
    let original = GpuInfo::new("Original", 16 * 1024 * 1024 * 1024)
        .with_compute_capability(8, 9)
        .with_index(1);
    let cloned = original.clone();
    assert_eq!(original.name, cloned.name);
    assert_eq!(original.index, cloned.index);
}

#[test]
fn test_tpu_info_clone() {
    let original = TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024);
    let cloned = original.clone();
    assert_eq!(original.version, cloned.version);
    assert_eq!(original.cores, cloned.cores);
}

#[test]
fn test_apple_silicon_clone() {
    let original =
        AppleSiliconInfo::new("M2 Pro").with_cores(8, 4, 19).with_memory(32 * 1024 * 1024 * 1024);
    let cloned = original.clone();
    assert_eq!(original.chip, cloned.chip);
    assert_eq!(original.gpu_cores, cloned.gpu_cores);
}

#[test]
fn test_compute_device_clone() {
    let original = ComputeDevice::Gpu(
        GpuInfo::new("Test", 16 * 1024 * 1024 * 1024).with_compute_capability(8, 0),
    );
    let cloned = original.clone();
    assert_eq!(original.name(), cloned.name());
}
