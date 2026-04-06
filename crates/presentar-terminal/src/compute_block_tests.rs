use super::*;

// Tests from original mod tests
#[test]
fn test_simd_instruction_set_detect() {
    let isa = SimdInstructionSet::detect();
    // Should detect something (at minimum Scalar)
    assert!(isa.vector_width() >= 1);
}

#[test]
fn test_simd_instruction_set_names() {
    assert_eq!(SimdInstructionSet::Scalar.name(), "Scalar");
    assert_eq!(SimdInstructionSet::Avx2.name(), "AVX2");
    assert_eq!(SimdInstructionSet::Neon.name(), "NEON");
}

#[test]
fn test_simd_vector_widths() {
    assert_eq!(SimdInstructionSet::Scalar.vector_width(), 1);
    assert_eq!(SimdInstructionSet::Sse4.vector_width(), 4);
    assert_eq!(SimdInstructionSet::Avx2.vector_width(), 8);
    assert_eq!(SimdInstructionSet::Avx512.vector_width(), 16);
}

#[test]
fn test_compute_block_id_strings() {
    assert_eq!(ComputeBlockId::CpuSparklines.id_string(), "CB-CPU-001");
    assert_eq!(ComputeBlockId::MemSparklines.id_string(), "CB-MEM-001");
    assert_eq!(ComputeBlockId::ConnAge.id_string(), "CB-CONN-001");
    assert_eq!(ComputeBlockId::NetSparklines.id_string(), "CB-NET-001");
    assert_eq!(ComputeBlockId::ProcTreeView.id_string(), "CB-PROC-001");
}

#[test]
fn test_compute_block_simd_vectorizable() {
    assert!(ComputeBlockId::CpuSparklines.simd_vectorizable());
    assert!(ComputeBlockId::MemSparklines.simd_vectorizable());
    assert!(!ComputeBlockId::CpuLoadGauge.simd_vectorizable());
    assert!(!ComputeBlockId::ProcTreeView.simd_vectorizable());
}

#[test]
fn test_sparkline_block_new() {
    let block = SparklineBlock::new(60);
    assert!(block.history().is_empty());
}

#[test]
fn test_sparkline_block_push() {
    let mut block = SparklineBlock::new(5);
    for i in 0..10 {
        block.push(i as f32);
    }
    // Should only keep last 5
    assert_eq!(block.history().len(), 5);
    assert_eq!(block.history(), &[5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_sparkline_block_render() {
    let mut block = SparklineBlock::new(8);
    for v in [0.0, 25.0, 50.0, 75.0, 100.0] {
        block.push(v);
    }
    let rendered = block.render(5);
    assert_eq!(rendered.len(), 5);
    // First should be lowest, last should be highest
    assert_eq!(rendered[0], '▁');
    assert_eq!(rendered[4], '█');
}

#[test]
fn test_sparkline_block_empty() {
    let block = SparklineBlock::new(8);
    let rendered = block.render(5);
    assert_eq!(rendered, vec![' '; 5]);
}

#[test]
fn test_sparkline_block_compute() {
    let mut block = SparklineBlock::new(8);
    let output = block.compute(&50.0);
    assert!(!output.is_empty());
}

#[test]
fn test_sparkline_block_simd_supported() {
    let block = SparklineBlock::default();
    // Just verify it doesn't panic
    let _ = block.simd_supported();
    let _ = block.simd_instruction_set();
}

#[test]
fn test_load_trend_block_new() {
    let block = LoadTrendBlock::new(5);
    assert_eq!(block.trend(), TrendDirection::Flat);
}

#[test]
fn test_load_trend_block_up() {
    let mut block = LoadTrendBlock::new(3);
    for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
        block.compute(&v);
    }
    assert_eq!(block.trend(), TrendDirection::Up);
}

#[test]
fn test_load_trend_block_down() {
    let mut block = LoadTrendBlock::new(3);
    for v in [5.0, 4.0, 3.0, 2.0, 1.0] {
        block.compute(&v);
    }
    assert_eq!(block.trend(), TrendDirection::Down);
}

#[test]
fn test_load_trend_block_flat() {
    let mut block = LoadTrendBlock::new(3);
    for v in [5.0, 5.0, 5.0, 5.0, 5.0] {
        block.compute(&v);
    }
    assert_eq!(block.trend(), TrendDirection::Flat);
}

#[test]
fn test_trend_direction_arrows() {
    assert_eq!(TrendDirection::Up.arrow(), '↑');
    assert_eq!(TrendDirection::Down.arrow(), '↓');
    assert_eq!(TrendDirection::Flat.arrow(), '→');
}

#[test]
fn test_latency_budgets() {
    let sparkline = SparklineBlock::default();
    assert!(sparkline.latency_budget_us() > 0);

    let trend = LoadTrendBlock::default();
    assert!(trend.latency_budget_us() > 0);
}

#[test]
fn test_simd_instruction_set_default() {
    let isa = SimdInstructionSet::default();
    assert!(isa.vector_width() >= 1);
}

// Tests from original mod new_block_tests
#[test]
fn test_cpu_frequency_block_new() {
    let block = CpuFrequencyBlock::new();
    assert!(block.frequencies.is_empty());
}

#[test]
fn test_cpu_frequency_block_percentages() {
    let mut block = CpuFrequencyBlock::new();
    block.set_frequencies(vec![2000, 3000, 4000], vec![4000, 4000, 4000]);
    let pcts = block.frequency_percentages();
    assert_eq!(pcts.len(), 3);
    assert!((pcts[0] - 50.0).abs() < 0.1);
    assert!((pcts[1] - 75.0).abs() < 0.1);
    assert!((pcts[2] - 100.0).abs() < 0.1);
}

#[test]
fn test_cpu_frequency_block_scaling_states() {
    let mut block = CpuFrequencyBlock::new();
    block.set_frequencies(vec![1000, 2000, 3800, 4000], vec![4000, 4000, 4000, 4000]);
    let states = block.scaling_indicators();
    assert_eq!(states[0], FrequencyScalingState::Scaled);
    assert_eq!(states[1], FrequencyScalingState::Normal);
    assert_eq!(states[2], FrequencyScalingState::Turbo);
    assert_eq!(states[3], FrequencyScalingState::Turbo);
}

#[test]
fn test_frequency_scaling_state_indicators() {
    assert_eq!(FrequencyScalingState::Turbo.indicator(), '⚡');
    assert_eq!(FrequencyScalingState::High.indicator(), '↑');
    assert_eq!(FrequencyScalingState::Normal.indicator(), '→');
    assert_eq!(FrequencyScalingState::Scaled.indicator(), '↓');
    assert_eq!(FrequencyScalingState::Idle.indicator(), '·');
}

#[test]
fn test_cpu_governor_from_name() {
    assert_eq!(
        CpuGovernor::from_name("performance"),
        CpuGovernor::Performance
    );
    assert_eq!(CpuGovernor::from_name("powersave"), CpuGovernor::Powersave);
    assert_eq!(CpuGovernor::from_name("schedutil"), CpuGovernor::Schedutil);
    assert_eq!(CpuGovernor::from_name("unknown"), CpuGovernor::Unknown);
}

#[test]
fn test_cpu_governor_short_names() {
    assert_eq!(CpuGovernor::Performance.short_name(), "perf");
    assert_eq!(CpuGovernor::Powersave.short_name(), "psav");
    assert_eq!(CpuGovernor::Schedutil.short_name(), "schu");
}

#[test]
fn test_mem_pressure_level() {
    let mut block = MemPressureBlock::new();
    block.set_pressure(0.5, 0.3, 0.2, 0.1);
    assert_eq!(block.pressure_level(), MemoryPressureLevel::None);

    block.set_pressure(5.0, 3.0, 2.0, 1.0);
    assert_eq!(block.pressure_level(), MemoryPressureLevel::Low);

    block.set_pressure(15.0, 10.0, 8.0, 5.0);
    assert_eq!(block.pressure_level(), MemoryPressureLevel::Medium);

    block.set_pressure(30.0, 20.0, 15.0, 10.0);
    assert_eq!(block.pressure_level(), MemoryPressureLevel::High);

    block.set_pressure(60.0, 50.0, 40.0, 30.0);
    assert_eq!(block.pressure_level(), MemoryPressureLevel::Critical);
}

#[test]
fn test_mem_pressure_trend() {
    let mut block = MemPressureBlock::new();
    block.set_pressure(20.0, 15.0, 5.0, 10.0);
    assert_eq!(block.trend(), TrendDirection::Up);

    block.set_pressure(5.0, 10.0, 20.0, 2.0);
    assert_eq!(block.trend(), TrendDirection::Down);

    block.set_pressure(10.0, 10.0, 10.0, 5.0);
    assert_eq!(block.trend(), TrendDirection::Flat);
}

#[test]
fn test_huge_pages_block() {
    let mut block = HugePagesBlock::new();
    block.set_values(100, 50, 10, 2048);
    assert!((block.usage_percent() - 50.0).abs() < 0.1);
    assert_eq!(block.total_bytes(), 100 * 2048 * 1024);
    assert_eq!(block.used_bytes(), 50 * 2048 * 1024);
}

#[test]
fn test_huge_pages_block_empty() {
    let block = HugePagesBlock::new();
    assert_eq!(block.usage_percent(), 0.0);
}

#[test]
fn test_gpu_thermal_block() {
    let mut block = GpuThermalBlock::new();
    block.set_values(45.0, 100.0, 250.0);
    assert_eq!(block.thermal_state(), GpuThermalState::Cool);
    assert!((block.power_percent() - 40.0).abs() < 0.1);

    block.set_values(75.0, 200.0, 250.0);
    assert_eq!(block.thermal_state(), GpuThermalState::Warm);

    block.set_values(95.0, 250.0, 250.0);
    assert_eq!(block.thermal_state(), GpuThermalState::Critical);
}

#[test]
fn test_gpu_thermal_state_indicators() {
    assert_eq!(GpuThermalState::Cool.indicator(), '❄');
    assert_eq!(GpuThermalState::Normal.indicator(), '●');
    assert_eq!(GpuThermalState::Critical.indicator(), '🔥');
}

#[test]
fn test_gpu_vram_block() {
    let mut block = GpuVramBlock::new();
    let procs = vec![
        (1234, 1024, "firefox".to_string()),
        (5678, 512, "code".to_string()),
        (9012, 2048, "blender".to_string()),
    ];
    block.set_values(8192, 4096, procs);
    assert!((block.usage_percent() - 50.0).abs() < 0.1);

    let top = block.top_consumers(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].2, "blender");
    assert_eq!(top[1].2, "firefox");
}

#[test]
fn test_memory_pressure_level_severity() {
    assert_eq!(MemoryPressureLevel::None.severity(), 0);
    assert_eq!(MemoryPressureLevel::Low.severity(), 1);
    assert_eq!(MemoryPressureLevel::Medium.severity(), 2);
    assert_eq!(MemoryPressureLevel::High.severity(), 3);
    assert_eq!(MemoryPressureLevel::Critical.severity(), 4);
}

#[test]
fn test_gpu_thermal_state_severity() {
    assert_eq!(GpuThermalState::Cool.severity(), 0);
    assert_eq!(GpuThermalState::Normal.severity(), 1);
    assert_eq!(GpuThermalState::Warm.severity(), 2);
    assert_eq!(GpuThermalState::Hot.severity(), 3);
    assert_eq!(GpuThermalState::Critical.severity(), 4);
}

#[test]
fn test_cpu_frequency_block_compute() {
    let mut block = CpuFrequencyBlock::new();
    let input = (vec![2000, 4000], vec![4000, 4000]);
    let output = block.compute(&input);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0], FrequencyScalingState::Normal);
    assert_eq!(output[1], FrequencyScalingState::Turbo);
}

#[test]
fn test_cpu_governor_block_compute() {
    let mut block = CpuGovernorBlock::new();
    let output = block.compute(&"performance".to_string());
    assert_eq!(output, CpuGovernor::Performance);
}

#[test]
fn test_mem_pressure_block_compute() {
    let mut block = MemPressureBlock::new();
    let input = (30.0_f32, 25.0_f32, 20.0_f32, 15.0_f32);
    let output = block.compute(&input);
    assert_eq!(output, MemoryPressureLevel::High);
}

#[test]
fn test_huge_pages_block_compute() {
    let mut block = HugePagesBlock::new();
    let input = (100_u64, 75_u64, 5_u64, 2048_u64);
    let output = block.compute(&input);
    assert!((output - 25.0).abs() < 0.1);
}

#[test]
fn test_gpu_thermal_block_compute() {
    let mut block = GpuThermalBlock::new();
    let input = (85.0_f32, 200.0_f32, 250.0_f32);
    let output = block.compute(&input);
    assert_eq!(output, GpuThermalState::Hot);
}

#[test]
fn test_gpu_vram_block_compute() {
    let mut block = GpuVramBlock::new();
    let procs = vec![(1234_u32, 1024_u64, "test".to_string())];
    let input = (8192_u64, 4096_u64, procs);
    let output = block.compute(&input);
    assert!((output - 50.0).abs() < 0.1);
}

#[test]
fn test_latency_budgets_new_blocks() {
    assert!(CpuFrequencyBlock::new().latency_budget_us() > 0);
    assert!(CpuGovernorBlock::new().latency_budget_us() > 0);
    assert!(MemPressureBlock::new().latency_budget_us() > 0);
    assert!(HugePagesBlock::new().latency_budget_us() > 0);
    assert!(GpuThermalBlock::new().latency_budget_us() > 0);
    assert!(GpuVramBlock::new().latency_budget_us() > 0);
}

#[test]
fn test_cpu_governor_icons() {
    assert_eq!(CpuGovernor::Performance.icon(), '🚀');
    assert_eq!(CpuGovernor::Powersave.icon(), '🔋');
    assert_eq!(CpuGovernor::Unknown.icon(), '?');
}

#[test]
fn test_simd_all_names() {
    assert_eq!(SimdInstructionSet::Sse4.name(), "SSE4.1");
    assert_eq!(SimdInstructionSet::Avx512.name(), "AVX-512");
    assert_eq!(SimdInstructionSet::WasmSimd128.name(), "WASM SIMD128");
}

#[test]
fn test_simd_wasm_vector_width() {
    assert_eq!(SimdInstructionSet::Neon.vector_width(), 4);
    assert_eq!(SimdInstructionSet::WasmSimd128.vector_width(), 4);
}

#[test]
fn test_compute_block_id_all_strings() {
    // Test all compute block IDs have valid strings
    let ids = [
        ComputeBlockId::CpuLoadGauge,
        ComputeBlockId::CpuLoadTrend,
        ComputeBlockId::CpuFrequency,
        ComputeBlockId::CpuBoostIndicator,
        ComputeBlockId::CpuTemperature,
        ComputeBlockId::CpuTopConsumers,
        ComputeBlockId::MemZramRatio,
        ComputeBlockId::MemPressureGauge,
        ComputeBlockId::MemSwapThrashing,
        ComputeBlockId::MemCacheBreakdown,
        ComputeBlockId::MemHugePages,
        ComputeBlockId::ConnProc,
        ComputeBlockId::ConnGeo,
        ComputeBlockId::ConnLatency,
        ComputeBlockId::ConnService,
        ComputeBlockId::ConnHotIndicator,
        ComputeBlockId::ConnSparkline,
        ComputeBlockId::NetProtocolStats,
        ComputeBlockId::NetErrorRate,
        ComputeBlockId::NetDropRate,
        ComputeBlockId::NetLatencyGauge,
        ComputeBlockId::NetBandwidthUtil,
        ComputeBlockId::ProcSortIndicator,
        ComputeBlockId::ProcFilter,
        ComputeBlockId::ProcOomScore,
        ComputeBlockId::ProcNiceValue,
        ComputeBlockId::ProcThreadCount,
        ComputeBlockId::ProcCgroup,
    ];
    for id in ids {
        assert!(!id.id_string().is_empty());
    }
}

#[test]
fn test_compute_block_id_simd_categories() {
    // Test SIMD vectorizable blocks
    assert!(ComputeBlockId::NetSparklines.simd_vectorizable());
    assert!(ComputeBlockId::NetProtocolStats.simd_vectorizable());
    assert!(ComputeBlockId::NetErrorRate.simd_vectorizable());
    assert!(ComputeBlockId::NetDropRate.simd_vectorizable());
    assert!(ComputeBlockId::NetBandwidthUtil.simd_vectorizable());
    assert!(ComputeBlockId::ConnAge.simd_vectorizable());
    assert!(ComputeBlockId::ConnGeo.simd_vectorizable());
    assert!(ComputeBlockId::ConnLatency.simd_vectorizable());
    assert!(ComputeBlockId::ConnService.simd_vectorizable());
    assert!(ComputeBlockId::ConnHotIndicator.simd_vectorizable());
    assert!(ComputeBlockId::ConnSparkline.simd_vectorizable());

    // Test non-SIMD blocks
    assert!(!ComputeBlockId::MemZramRatio.simd_vectorizable());
    assert!(!ComputeBlockId::MemCacheBreakdown.simd_vectorizable());
    assert!(!ComputeBlockId::MemHugePages.simd_vectorizable());
    assert!(!ComputeBlockId::ConnProc.simd_vectorizable());
    assert!(!ComputeBlockId::NetLatencyGauge.simd_vectorizable());
    assert!(!ComputeBlockId::ProcSortIndicator.simd_vectorizable());
    assert!(!ComputeBlockId::ProcFilter.simd_vectorizable());
    assert!(!ComputeBlockId::ProcCgroup.simd_vectorizable());
}

#[test]
fn test_sparkline_block_default() {
    let block = SparklineBlock::default();
    assert!(block.history().is_empty());
    assert_eq!(block.max_samples, 60);
}

#[test]
fn test_sparkline_block_render_uniform() {
    let mut block = SparklineBlock::new(5);
    for _ in 0..5 {
        block.push(50.0);
    }
    let rendered = block.render(5);
    // All same value should render mid-level blocks
    for ch in &rendered {
        assert_ne!(*ch, ' ');
    }
}

#[test]
fn test_sparkline_block_sample_to_width_shorter() {
    let mut block = SparklineBlock::new(10);
    for i in 0..3 {
        block.push(i as f32);
    }
    // Render at width 5 - should pad with zeros
    let rendered = block.render(5);
    assert_eq!(rendered.len(), 5);
}

#[test]
fn test_sparkline_block_sample_to_width_longer() {
    let mut block = SparklineBlock::new(20);
    for i in 0..15 {
        block.push(i as f32 * 10.0);
    }
    // Render at width 5 - should downsample
    let rendered = block.render(5);
    assert_eq!(rendered.len(), 5);
}

#[test]
fn test_load_trend_block_default() {
    let block = LoadTrendBlock::default();
    assert_eq!(block.window_size, 5);
}

#[test]
fn test_load_trend_block_history_limit() {
    let mut block = LoadTrendBlock::new(3);
    // Push more than window_size * 2
    for i in 0..20 {
        block.compute(&(i as f32));
    }
    // History should be limited
    assert!(block.history.len() <= block.window_size * 2);
}

#[test]
fn test_load_trend_block_insufficient_history() {
    let mut block = LoadTrendBlock::new(5);
    block.compute(&1.0);
    // Only 1 sample, should be flat
    assert_eq!(block.trend(), TrendDirection::Flat);
}

#[test]
fn test_sparkline_block_find_min_max_empty() {
    let block = SparklineBlock::new(5);
    // Empty history should return defaults
    let (min, max) = block.find_min_max();
    assert_eq!(min, 0.0);
    assert_eq!(max, 1.0);
}

#[test]
fn test_sparkline_block_simd_instruction_set() {
    let block = SparklineBlock::new(10);
    let isa = block.simd_instruction_set();
    assert!(isa.vector_width() >= 1);
}

#[test]
fn test_load_trend_latency_budget() {
    let trend = LoadTrendBlock::new(5);
    assert_eq!(trend.latency_budget_us(), 10);
}

#[test]
fn test_sparkline_latency_budget() {
    let sparkline = SparklineBlock::new(60);
    assert_eq!(sparkline.latency_budget_us(), 100);
}

// Tests from original mod metrics_cache_tests
#[test]
fn test_metrics_cache_new() {
    let cache = MetricsCache::new();
    assert_eq!(cache.frame_id, 0);
    assert_eq!(cache.cpu.avg_usage, 0.0);
}

#[test]
fn test_metrics_cache_update_cpu() {
    let mut cache = MetricsCache::new();
    let cores = vec![10.0, 20.0, 30.0, 95.0];
    cache.update_cpu(&cores, [1.0, 2.0, 3.0], 4.5, 1);

    assert!((cache.cpu.avg_usage - 38.75).abs() < 0.1);
    assert_eq!(cache.cpu.max_core_usage, 95.0);
    assert_eq!(cache.cpu.hot_cores, 1);
    assert_eq!(cache.cpu.freq_ghz, 4.5);
    assert_eq!(cache.frame_id, 1);
}

#[test]
fn test_metrics_cache_update_memory() {
    let mut cache = MetricsCache::new();
    cache.update_memory(
        50_000_000_000,  // 50GB used
        100_000_000_000, // 100GB total
        20_000_000_000,  // 20GB cached
        1_000_000_000,   // 1GB swap used
        10_000_000_000,  // 10GB swap total
        2.5,             // ZRAM ratio
    );

    assert!((cache.memory.usage_percent - 50.0).abs() < 0.1);
    assert!((cache.memory.swap_percent - 10.0).abs() < 0.1);
    assert_eq!(cache.memory.zram_ratio, 2.5);
}

#[test]
fn test_metrics_cache_update_process() {
    let mut cache = MetricsCache::new();
    cache.update_process(
        1000, // total
        5,    // running
        900,  // sleeping
        Some((1234, 50.0, "chrome".to_string())),
        Some((5678, 25.0, "firefox".to_string())),
        150.0, // total CPU
    );

    assert_eq!(cache.process.total_count, 1000);
    assert_eq!(cache.process.running_count, 5);
    assert!(cache.process.top_cpu.is_some());
    assert_eq!(cache.process.top_cpu.as_ref().unwrap().2, "chrome");
}

#[test]
fn test_metrics_cache_update_gpu() {
    let mut cache = MetricsCache::new();
    cache.update_gpu(
        "RTX 4090".to_string(),
        80.0,  // usage
        50.0,  // vram
        75.0,  // temp
        300.0, // power
    );

    assert_eq!(cache.gpu.name, "RTX 4090");
    assert_eq!(cache.gpu.thermal_state, GpuThermalState::Warm);
}

#[test]
fn test_metrics_cache_staleness() {
    let mut cache = MetricsCache::new();
    cache.mark_updated(1000);

    // Not stale at same time
    assert!(!cache.is_stale(1000, 100));
    // Not stale within window
    assert!(!cache.is_stale(1050, 100));
    // Stale after window
    assert!(cache.is_stale(1200, 100));
}

#[test]
fn test_metrics_cache_block_compute() {
    let mut block = MetricsCacheBlock::new();
    block
        .cache_mut()
        .update_cpu(&[50.0, 60.0], [1.0, 2.0, 3.0], 4.0, 1);

    let output = block.compute(&());
    assert_eq!(output.frame_id, 1);
    assert!(output.cpu.avg_usage > 0.0);
}

#[test]
fn test_metrics_cache_block_latency() {
    let block = MetricsCacheBlock::new();
    assert_eq!(block.latency_budget_us(), 1);
}

#[test]
fn test_metrics_cache_empty_cores() {
    let mut cache = MetricsCache::new();
    cache.update_cpu(&[], [0.0, 0.0, 0.0], 0.0, 0);
    // Should not panic, just leave defaults
    assert_eq!(cache.cpu.avg_usage, 0.0);
}
