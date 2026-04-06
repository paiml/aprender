use super::*;

// =========================================================================
// H011: Pressure Level Tests
// =========================================================================

#[test]
fn h011_pressure_level_from_percent_ok() {
    assert_eq!(PressureLevel::from_available_percent(100.0), PressureLevel::Ok);
    assert_eq!(PressureLevel::from_available_percent(75.0), PressureLevel::Ok);
    assert_eq!(PressureLevel::from_available_percent(50.0), PressureLevel::Ok);
}

#[test]
fn h011_pressure_level_from_percent_elevated() {
    assert_eq!(PressureLevel::from_available_percent(49.9), PressureLevel::Elevated);
    assert_eq!(PressureLevel::from_available_percent(40.0), PressureLevel::Elevated);
    assert_eq!(PressureLevel::from_available_percent(30.0), PressureLevel::Elevated);
}

#[test]
fn h011_pressure_level_from_percent_warning() {
    assert_eq!(PressureLevel::from_available_percent(29.9), PressureLevel::Warning);
    assert_eq!(PressureLevel::from_available_percent(20.0), PressureLevel::Warning);
    assert_eq!(PressureLevel::from_available_percent(15.0), PressureLevel::Warning);
}

#[test]
fn h011_pressure_level_from_percent_critical() {
    assert_eq!(PressureLevel::from_available_percent(14.9), PressureLevel::Critical);
    assert_eq!(PressureLevel::from_available_percent(5.0), PressureLevel::Critical);
    assert_eq!(PressureLevel::from_available_percent(0.0), PressureLevel::Critical);
}

#[test]
fn h011_pressure_level_display() {
    assert_eq!(format!("{}", PressureLevel::Ok), "OK");
    assert_eq!(format!("{}", PressureLevel::Elevated), "ELEVATED");
    assert_eq!(format!("{}", PressureLevel::Warning), "WARNING");
    assert_eq!(format!("{}", PressureLevel::Critical), "CRITICAL");
}

#[test]
fn h011_pressure_level_recommendation() {
    assert!(PressureLevel::Ok.recommendation().contains("healthy"));
    assert!(PressureLevel::Critical.recommendation().contains("block"));
}

#[test]
fn h011_pressure_level_should_block() {
    assert!(!PressureLevel::Ok.should_block_allocations());
    assert!(!PressureLevel::Elevated.should_block_allocations());
    assert!(!PressureLevel::Warning.should_block_allocations());
    assert!(PressureLevel::Critical.should_block_allocations());
}

#[test]
fn h011_pressure_level_ansi_color() {
    // All levels should have ANSI color codes
    assert!(PressureLevel::Ok.ansi_color().contains("\x1b["));
    assert!(PressureLevel::Elevated.ansi_color().contains("\x1b["));
    assert!(PressureLevel::Warning.ansi_color().contains("\x1b["));
    assert!(PressureLevel::Critical.ansi_color().contains("\x1b["));
}

// =========================================================================
// H012: Memory Metrics Tests
// =========================================================================

#[test]
fn h012_memory_metrics_default() {
    let metrics = MemoryMetrics::default();
    assert_eq!(metrics.ram_used_bytes, 0);
    assert_eq!(metrics.ram_total_bytes, 0);
    assert_eq!(metrics.pressure_level, PressureLevel::Ok);
}

#[test]
fn h012_memory_metrics_new() {
    let metrics = MemoryMetrics::new();
    // On Linux, should have read some values
    #[cfg(target_os = "linux")]
    {
        assert!(metrics.ram_total_bytes > 0);
    }
}

#[test]
fn h012_memory_metrics_ram_usage_percent() {
    let mut metrics = MemoryMetrics::default();
    metrics.ram_used_bytes = 50 * 1024 * 1024 * 1024; // 50 GB
    metrics.ram_total_bytes = 100 * 1024 * 1024 * 1024; // 100 GB

    assert!((metrics.ram_usage_percent() - 50.0).abs() < 0.01);
}

#[test]
fn h012_memory_metrics_ram_usage_percent_zero() {
    let metrics = MemoryMetrics::default();
    assert!((metrics.ram_usage_percent() - 0.0).abs() < 0.01);
}

#[test]
fn h012_memory_metrics_available_percent() {
    let mut metrics = MemoryMetrics::default();
    metrics.ram_available_bytes = 75 * 1024 * 1024 * 1024; // 75 GB
    metrics.ram_total_bytes = 100 * 1024 * 1024 * 1024; // 100 GB

    assert!((metrics.ram_available_percent() - 75.0).abs() < 0.01);
}

#[test]
fn h012_memory_metrics_available_percent_zero_total() {
    let metrics = MemoryMetrics::default();
    // When total is 0, should return 100% available (not divide by zero)
    assert!((metrics.ram_available_percent() - 100.0).abs() < 0.01);
}

#[test]
fn h012_memory_metrics_swap_percent() {
    let mut metrics = MemoryMetrics::default();
    metrics.swap_used_bytes = 2 * 1024 * 1024 * 1024; // 2 GB
    metrics.swap_total_bytes = 16 * 1024 * 1024 * 1024; // 16 GB

    assert!((metrics.swap_usage_percent() - 12.5).abs() < 0.01);
}

#[test]
fn h012_memory_metrics_swap_percent_zero() {
    let metrics = MemoryMetrics::default();
    assert!((metrics.swap_usage_percent() - 0.0).abs() < 0.01);
}

#[test]
fn h012_memory_metrics_gb_helpers() {
    let mut metrics = MemoryMetrics::default();
    metrics.ram_used_bytes = 32 * 1024 * 1024 * 1024;
    metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024;
    metrics.swap_used_bytes = 1024 * 1024 * 1024;
    metrics.swap_total_bytes = 16 * 1024 * 1024 * 1024;

    assert!((metrics.ram_used_gb() - 32.0).abs() < 0.01);
    assert!((metrics.ram_total_gb() - 64.0).abs() < 0.01);
    assert!((metrics.swap_used_gb() - 1.0).abs() < 0.01);
    assert!((metrics.swap_total_gb() - 16.0).abs() < 0.01);
}

// =========================================================================
// H013: History Tracking Tests
// =========================================================================

#[test]
fn h013_memory_metrics_history_max() {
    let mut metrics = MemoryMetrics::default();
    metrics.ram_total_bytes = 100;
    metrics.swap_total_bytes = 100;

    // Add more than MAX_HISTORY_POINTS
    for i in 0..100 {
        metrics.ram_used_bytes = i;
        metrics.swap_used_bytes = i;
        metrics.update_history();
    }

    assert_eq!(metrics.ram_history.len(), MemoryMetrics::MAX_HISTORY_POINTS);
    assert_eq!(metrics.swap_history.len(), MemoryMetrics::MAX_HISTORY_POINTS);
}

// =========================================================================
// H014: GPU VRAM Metrics Tests
// =========================================================================

#[test]
fn h014_gpu_vram_new() {
    let vram =
        GpuVramMetrics::new(DeviceId::nvidia(0), 8 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024);
    assert_eq!(vram.device_id, DeviceId::nvidia(0));
    assert_eq!(vram.used_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(vram.total_bytes, 24 * 1024 * 1024 * 1024);
}

#[test]
fn h014_gpu_vram_usage_percent() {
    let vram =
        GpuVramMetrics::new(DeviceId::nvidia(0), 6 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024);
    assert!((vram.usage_percent() - 25.0).abs() < 0.01);
}

#[test]
fn h014_gpu_vram_usage_percent_zero() {
    let vram = GpuVramMetrics::new(DeviceId::nvidia(0), 0, 0);
    assert!((vram.usage_percent() - 0.0).abs() < 0.01);
}

#[test]
fn h014_gpu_vram_available() {
    let vram =
        GpuVramMetrics::new(DeviceId::nvidia(0), 8 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024);
    assert_eq!(vram.available_bytes(), 16 * 1024 * 1024 * 1024);
}

#[test]
fn h014_gpu_vram_gb_helpers() {
    let vram =
        GpuVramMetrics::new(DeviceId::nvidia(0), 8 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024);
    assert!((vram.used_gb() - 8.0).abs() < 0.01);
    assert!((vram.total_gb() - 24.0).abs() < 0.01);
}

#[test]
fn h014_gpu_vram_update_history() {
    let mut vram = GpuVramMetrics::new(DeviceId::nvidia(0), 0, 24 * 1024 * 1024 * 1024);

    for i in 0..100 {
        vram.update(i * 1024 * 1024 * 1024);
    }

    assert_eq!(vram.history.len(), GpuVramMetrics::MAX_HISTORY_POINTS);
}

// =========================================================================
// H015: Safe Jobs Calculation Tests
// =========================================================================

#[test]
fn h015_safe_jobs_calculation() {
    let mut metrics = MemoryMetrics::default();
    // 30 GB available = 10 safe jobs (at 3GB/job)
    metrics.ram_available_bytes = 30 * 1024 * 1024 * 1024;
    metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024;
    metrics.calculate_pressure();

    // Should be min(10, cpu_cores)
    assert!(metrics.safe_parallel_jobs >= 1);
    assert!(metrics.safe_parallel_jobs <= 10);
}

#[test]
fn h015_safe_jobs_minimum_one() {
    let mut metrics = MemoryMetrics::default();
    // Very low memory - should still allow at least 1 job
    metrics.ram_available_bytes = 512 * 1024 * 1024; // 512 MB
    metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024;
    metrics.calculate_pressure();

    assert_eq!(metrics.safe_parallel_jobs, 1);
}

// =========================================================================
// H016: Pressure Analysis Tests
// =========================================================================

#[test]
fn h016_pressure_analysis_from_metrics() {
    let mut metrics = MemoryMetrics::default();
    // 16 GB available of 64 GB = 25% available -> Warning level
    metrics.ram_available_bytes = 16 * 1024 * 1024 * 1024;
    metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024;
    metrics.calculate_pressure();

    let analysis = PressureAnalysis::from_metrics(&metrics);

    assert_eq!(analysis.level, PressureLevel::Warning);
    assert!(!analysis.block_builds);
    assert!(analysis.safe_jobs >= 1);
}

#[test]
fn h016_pressure_analysis_critical() {
    let mut metrics = MemoryMetrics::default();
    metrics.ram_available_bytes = 5 * 1024 * 1024 * 1024; // 5 GB
    metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024; // 64 GB (~7.8%)
    metrics.calculate_pressure();

    let analysis = PressureAnalysis::from_metrics(&metrics);

    assert_eq!(analysis.level, PressureLevel::Critical);
    assert!(analysis.block_builds);
}

// =========================================================================
// H017: Total VRAM Tests
// =========================================================================

#[test]
fn h017_total_vram_single_gpu() {
    let mut metrics = MemoryMetrics::default();
    metrics.gpu_vram.push(GpuVramMetrics::new(
        DeviceId::nvidia(0),
        8 * 1024 * 1024 * 1024,
        24 * 1024 * 1024 * 1024,
    ));

    assert_eq!(metrics.total_vram_used_bytes(), 8 * 1024 * 1024 * 1024);
    assert_eq!(metrics.total_vram_total_bytes(), 24 * 1024 * 1024 * 1024);
}

#[test]
fn h017_total_vram_multi_gpu() {
    let mut metrics = MemoryMetrics::default();
    metrics.gpu_vram.push(GpuVramMetrics::new(
        DeviceId::nvidia(0),
        8 * 1024 * 1024 * 1024,
        24 * 1024 * 1024 * 1024,
    ));
    metrics.gpu_vram.push(GpuVramMetrics::new(
        DeviceId::nvidia(1),
        4 * 1024 * 1024 * 1024,
        24 * 1024 * 1024 * 1024,
    ));

    assert_eq!(metrics.total_vram_used_bytes(), 12 * 1024 * 1024 * 1024);
    assert_eq!(metrics.total_vram_total_bytes(), 48 * 1024 * 1024 * 1024);
}

#[test]
fn h017_total_vram_empty() {
    let metrics = MemoryMetrics::default();
    assert_eq!(metrics.total_vram_used_bytes(), 0);
    assert_eq!(metrics.total_vram_total_bytes(), 0);
}
