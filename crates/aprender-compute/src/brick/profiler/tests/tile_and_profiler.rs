//! TileStats and BrickProfiler core tests.

use super::super::*;

// ========================================================================
// TileStats Tests
// ========================================================================

#[test]
fn test_tile_stats_new() {
    let stats = TileStats::new(TileLevel::Macro);
    assert_eq!(stats.level, TileLevel::Macro);
    assert_eq!(stats.count, 0);
    assert_eq!(stats.total_ns, 0);
    assert_eq!(stats.min_ns, u64::MAX);
    assert_eq!(stats.max_ns, 0);
}

#[test]
fn test_tile_stats_add_sample() {
    let mut stats = TileStats::new(TileLevel::Midi);
    stats.add_sample(1000, 100, 200);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.total_ns, 1000);
    assert_eq!(stats.min_ns, 1000);
    assert_eq!(stats.max_ns, 1000);
    assert_eq!(stats.total_elements, 100);
    assert_eq!(stats.total_flops, 200);

    stats.add_sample(2000, 150, 300);
    assert_eq!(stats.count, 2);
    assert_eq!(stats.total_ns, 3000);
    assert_eq!(stats.min_ns, 1000);
    assert_eq!(stats.max_ns, 2000);
}

#[test]
fn test_tile_stats_avg_us() {
    let mut stats = TileStats::new(TileLevel::Micro);
    assert_eq!(stats.avg_us(), 0.0);

    stats.add_sample(2_000_000, 100, 200); // 2000 us
    assert!((stats.avg_us() - 2000.0).abs() < 0.01);

    stats.add_sample(4_000_000, 100, 200); // 4000 us
    assert!((stats.avg_us() - 3000.0).abs() < 0.01); // (2000 + 4000) / 2
}

#[test]
fn test_tile_stats_throughput() {
    let mut stats = TileStats::new(TileLevel::Macro);
    assert_eq!(stats.throughput(), 0.0);

    // 1 billion ns = 1 second, 1000 elements = 1000 elem/s
    stats.add_sample(1_000_000_000, 1000, 0);
    assert!((stats.throughput() - 1000.0).abs() < 1.0);
}

#[test]
fn test_tile_stats_gflops() {
    let mut stats = TileStats::new(TileLevel::Macro);
    assert_eq!(stats.gflops(), 0.0);

    // 1 second, 1 billion FLOPs = 1 GFLOP/s
    stats.add_sample(1_000_000_000, 100, 1_000_000_000);
    assert!((stats.gflops() - 1.0).abs() < 0.01);
}

#[test]
fn test_tile_stats_arithmetic_intensity() {
    let mut stats = TileStats::new(TileLevel::Midi);
    assert_eq!(stats.arithmetic_intensity(), 0.0);

    // 1000 elements * 4 bytes = 4000 bytes, 4000 flops = 1.0 AI
    stats.add_sample(1000, 1000, 4000);
    assert!((stats.arithmetic_intensity() - 1.0).abs() < 0.01);
}

#[test]
fn test_tile_level_name() {
    assert_eq!(TileLevel::Macro.name(), "macro");
    assert_eq!(TileLevel::Midi.name(), "midi");
    assert_eq!(TileLevel::Micro.name(), "micro");
}

// ========================================================================
// BrickProfiler Tests
// ========================================================================

#[test]
fn test_brick_profiler_new() {
    let profiler = BrickProfiler::new();
    assert!(!profiler.is_enabled());
    assert_eq!(profiler.total_tokens(), 0);
    assert_eq!(profiler.total_ns(), 0);
}

#[test]
fn test_brick_profiler_enabled() {
    let profiler = BrickProfiler::enabled();
    assert!(profiler.is_enabled());
}

#[test]
fn test_brick_profiler_enable_disable() {
    let mut profiler = BrickProfiler::new();
    assert!(!profiler.is_enabled());

    profiler.enable();
    assert!(profiler.is_enabled());

    profiler.disable();
    assert!(!profiler.is_enabled());
}

#[test]
fn test_brick_profiler_sync_mode() {
    let mut profiler = BrickProfiler::new();
    assert_eq!(profiler.sync_mode(), SyncMode::Deferred);

    profiler.set_sync_mode(SyncMode::Immediate);
    assert_eq!(profiler.sync_mode(), SyncMode::Immediate);
}

#[test]
fn test_brick_profiler_start_stop() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let timer = profiler.start("TestBrick");
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop(timer, 10);

    // Should have recorded something
    assert!(profiler.total_tokens() >= 10);
    assert!(profiler.total_ns() > 0);
}

#[test]
fn test_brick_profiler_start_stop_disabled() {
    let mut profiler = BrickProfiler::new();
    // profiler is disabled by default

    let timer = profiler.start("TestBrick");
    profiler.stop(timer, 10);

    // Should NOT have recorded anything
    assert_eq!(profiler.total_tokens(), 0);
    assert_eq!(profiler.total_ns(), 0);
}

#[test]
fn test_brick_profiler_brick_id_api() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let timer = profiler.start_brick(BrickId::RmsNorm);
    std::thread::sleep(std::time::Duration::from_micros(50));
    profiler.stop_brick(timer, 5);

    let stats = profiler.brick_stats(BrickId::RmsNorm);
    assert_eq!(stats.count, 1);
    assert!(stats.total_ns > 0);
}

#[test]
fn test_brick_profiler_deferred_api() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let start_ns = profiler.elapsed_ns();
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.record_deferred(BrickId::QkvProjection, start_ns, 100);

    assert!(profiler.has_pending());
    assert_eq!(profiler.pending_count(), 1);

    let end_ns = profiler.elapsed_ns();
    profiler.finalize(end_ns);

    assert!(!profiler.has_pending());
    assert_eq!(profiler.pending_count(), 0);

    let stats = profiler.brick_stats(BrickId::QkvProjection);
    assert_eq!(stats.count, 1);
}

#[test]
fn test_brick_profiler_reset() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let timer = profiler.start_brick(BrickId::AttentionSoftmax);
    profiler.stop_brick(timer, 10);

    assert!(profiler.total_tokens() > 0);

    profiler.reset();

    assert_eq!(profiler.total_tokens(), 0);
    assert_eq!(profiler.total_ns(), 0);
}

#[test]
fn test_brick_profiler_tile_profiling() {
    let mut profiler = BrickProfiler::new();
    assert!(!profiler.is_tile_profiling_enabled());

    profiler.enable_tile_profiling();
    assert!(profiler.is_tile_profiling_enabled());

    let timer = profiler.start_tile(TileLevel::Macro, 0, 0);
    std::thread::sleep(std::time::Duration::from_micros(50));
    profiler.stop_tile(timer, 1024, 2048);

    let stats = profiler.tile_stats(TileLevel::Macro);
    assert_eq!(stats.count, 1);
    assert!(stats.total_ns > 0);
}

#[test]
fn test_brick_profiler_tile_profiling_disabled() {
    let mut profiler = BrickProfiler::new();
    // Tile profiling disabled by default

    let timer = profiler.start_tile(TileLevel::Midi, 1, 1);
    profiler.stop_tile(timer, 512, 1024);

    let stats = profiler.tile_stats(TileLevel::Midi);
    assert_eq!(stats.count, 0);
}
