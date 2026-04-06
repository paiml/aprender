use super::super::*;

// ========================================================================
// PMAT-451: Compression Ratio and Bottleneck Tests
// ========================================================================

#[test]
fn test_brick_bottleneck_display() {
    assert_eq!(format!("{}", BrickBottleneck::Unknown), "unknown");
    assert_eq!(format!("{}", BrickBottleneck::Memory), "memory");
    assert_eq!(format!("{}", BrickBottleneck::Compute), "compute");
}

#[test]
fn test_brick_bottleneck_default() {
    let bottleneck = BrickBottleneck::default();
    assert_eq!(bottleneck, BrickBottleneck::Unknown);
}

#[test]
fn test_brick_stats_compression_ratio() {
    let mut stats = BrickStats::new("Compress");
    // 1000 bytes in, 250 bytes out = 4.0 compression ratio
    stats.add_sample_with_bytes(1_000_000, 100, 1000, 250);

    let ratio = stats.compression_ratio();
    assert!((ratio - 4.0).abs() < 0.001);
}

#[test]
fn test_brick_stats_compression_ratio_no_data() {
    let stats = BrickStats::new("Empty");
    // No compressed bytes = 1.0 ratio (no compression = 1:1)
    assert_eq!(stats.compression_ratio(), 1.0);
}

#[test]
fn test_brick_stats_throughput_gbps() {
    let mut stats = BrickStats::new("Throughput");
    // 1 GB (1e9 bytes) in 1 second (1e9 ns) = 1.0 GB/s
    stats.add_sample_with_bytes(1_000_000_000, 1000, 1_000_000_000, 0);

    let throughput = stats.throughput_gbps();
    assert!((throughput - 1.0).abs() < 0.01);
}

#[test]
fn test_brick_stats_throughput_gbps_zero_time() {
    let stats = BrickStats::new("Empty");
    // Zero time = 0.0 throughput (avoid division by zero)
    assert_eq!(stats.throughput_gbps(), 0.0);
}

#[test]
fn test_brick_stats_add_sample_with_bytes() {
    let mut stats = BrickStats::new("Bytes");

    stats.add_sample_with_bytes(1000, 10, 100, 25);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.total_ns, 1000);
    assert_eq!(stats.total_elements, 10);
    assert_eq!(stats.total_bytes, 100);
    assert_eq!(stats.total_compressed_bytes, 25);
    assert_eq!(stats.min_ns, 1000);
    assert_eq!(stats.max_ns, 1000);

    // Add second sample
    stats.add_sample_with_bytes(500, 5, 50, 20);
    assert_eq!(stats.count, 2);
    assert_eq!(stats.total_ns, 1500);
    assert_eq!(stats.total_elements, 15);
    assert_eq!(stats.total_bytes, 150);
    assert_eq!(stats.total_compressed_bytes, 45);
    assert_eq!(stats.min_ns, 500);
    assert_eq!(stats.max_ns, 1000);
}

#[test]
fn test_brick_stats_bottleneck() {
    let mut stats = BrickStats::new("Test");
    assert_eq!(stats.get_bottleneck(), BrickBottleneck::Unknown);

    stats.set_bottleneck(BrickBottleneck::Memory);
    assert_eq!(stats.get_bottleneck(), BrickBottleneck::Memory);

    stats.set_bottleneck(BrickBottleneck::Compute);
    assert_eq!(stats.get_bottleneck(), BrickBottleneck::Compute);
}

#[test]
fn test_brick_profiler_record_elapsed_with_bytes() {
    use std::time::Duration;
    let mut profiler = BrickProfiler::new();
    profiler.enable(); // Profiler is disabled by default

    profiler.record_elapsed_with_bytes(
        "Compress",
        Duration::from_nanos(1000),
        100,
        1_000_000,
        250_000,
    );
    profiler.record_elapsed_with_bytes(
        "Compress",
        Duration::from_nanos(2000),
        200,
        2_000_000,
        500_000,
    );

    let stats = profiler.stats("Compress").unwrap();
    assert_eq!(stats.count, 2);
    assert_eq!(stats.total_ns, 3000);
    assert_eq!(stats.total_elements, 300);
    assert_eq!(stats.total_bytes, 3_000_000);
    assert_eq!(stats.total_compressed_bytes, 750_000);
}

#[test]
fn test_brick_profiler_set_bottleneck() {
    use std::time::Duration;
    let mut profiler = BrickProfiler::new();
    profiler.enable(); // Profiler is disabled by default
    profiler.record_elapsed("TestBrick", Duration::from_nanos(1000), 100);
    profiler.set_brick_bottleneck("TestBrick", BrickBottleneck::Memory);

    let stats = profiler.stats("TestBrick").unwrap();
    assert_eq!(stats.get_bottleneck(), BrickBottleneck::Memory);
}

#[test]
fn test_brick_profiler_to_json_includes_pmat451_fields() {
    use std::time::Duration;
    let mut profiler = BrickProfiler::new();
    profiler.enable(); // Profiler is disabled by default
    profiler.record_elapsed_with_bytes(
        "Compress",
        Duration::from_micros(1000),
        100,
        1_000_000,
        250_000,
    );
    profiler.set_brick_bottleneck("Compress", BrickBottleneck::Memory);

    let json = profiler.to_json();

    // Verify new PMAT-451 fields are present
    assert!(json.contains("\"total_bytes\":"));
    assert!(json.contains("\"compression_ratio\":"));
    assert!(json.contains("\"throughput_gbps\":"));
    assert!(json.contains("\"bottleneck\":\"memory\""));
}

// ========================================================================
// PAR-200: BrickProfiler v2 Tests
// ========================================================================

#[test]
fn test_brick_id_category() {
    assert_eq!(BrickId::RmsNorm.category(), BrickCategory::Norm);
    assert_eq!(BrickId::LayerNorm.category(), BrickCategory::Norm);
    assert_eq!(BrickId::QkvProjection.category(), BrickCategory::Attention);
    assert_eq!(BrickId::AttentionSoftmax.category(), BrickCategory::Attention);
    assert_eq!(BrickId::GateProjection.category(), BrickCategory::Ffn);
    assert_eq!(BrickId::DownProjection.category(), BrickCategory::Ffn);
    assert_eq!(BrickId::Embedding.category(), BrickCategory::Other);
    assert_eq!(BrickId::Sampling.category(), BrickCategory::Other);
}

#[test]
fn test_brick_id_from_str() {
    assert_eq!(BrickId::from_str("RmsNorm"), Some(BrickId::RmsNorm));
    assert_eq!(BrickId::from_str("Rope"), Some(BrickId::RopeEmbedding));
    assert_eq!(BrickId::from_str("RoPE"), Some(BrickId::RopeEmbedding));
    assert_eq!(BrickId::from_str("SiLU"), Some(BrickId::Activation));
    assert_eq!(BrickId::from_str("Unknown"), None);
}

#[test]
fn test_brick_id_name() {
    assert_eq!(BrickId::RmsNorm.name(), "RmsNorm");
    assert_eq!(BrickId::QkvProjection.name(), "QkvProjection");
    assert_eq!(BrickId::Activation.name(), "Activation");
}

#[test]
fn test_brick_profiler_fast_path() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Use fast path API
    let timer = profiler.start_brick(BrickId::RmsNorm);
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop_brick(timer, 1);

    let stats = profiler.brick_stats(BrickId::RmsNorm);
    assert_eq!(stats.count, 1);
    assert!(stats.total_ns > 0);
    assert_eq!(profiler.total_tokens(), 1);
}

#[test]
fn test_brick_profiler_legacy_to_fast_path() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Use legacy string API with known brick name
    let timer = profiler.start("RmsNorm");
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop(timer, 1);

    // Should be routed to fast path array
    let stats = profiler.brick_stats(BrickId::RmsNorm);
    assert_eq!(stats.count, 1);
    assert!(stats.total_ns > 0);
}

#[test]
fn test_brick_profiler_dynamic_brick() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Use unknown brick name
    let timer = profiler.start("CustomOperation");
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop(timer, 1);

    // Should be in dynamic stats
    let stats = profiler.stats("CustomOperation").unwrap();
    assert_eq!(stats.count, 1);
}

#[test]
fn test_brick_profiler_deferred_sync() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.set_sync_mode(SyncMode::Deferred);
    profiler.reset_epoch();

    // Record deferred measurements
    let start1 = profiler.elapsed_ns();
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.record_deferred(BrickId::RmsNorm, start1, 1);

    let start2 = profiler.elapsed_ns();
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.record_deferred(BrickId::QkvProjection, start2, 1);

    // Should have pending measurements
    assert!(profiler.has_pending());
    assert_eq!(profiler.pending_count(), 2);

    // Finalize
    let end = profiler.elapsed_ns();
    profiler.finalize(end);

    // Should be finalized
    assert!(!profiler.has_pending());
    assert_eq!(profiler.brick_stats(BrickId::RmsNorm).count, 1);
    assert_eq!(profiler.brick_stats(BrickId::QkvProjection).count, 1);
}

#[test]
fn test_brick_profiler_category_stats() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Add samples to different categories
    let timer = profiler.start_brick(BrickId::RmsNorm);
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop_brick(timer, 1);

    let timer = profiler.start_brick(BrickId::QkvProjection);
    std::thread::sleep(std::time::Duration::from_micros(200));
    profiler.stop_brick(timer, 1);

    let timer = profiler.start_brick(BrickId::GateProjection);
    std::thread::sleep(std::time::Duration::from_micros(300));
    profiler.stop_brick(timer, 1);

    let cats = profiler.category_stats();

    // Verify category aggregation
    assert_eq!(cats[BrickCategory::Norm as usize].count, 1);
    assert_eq!(cats[BrickCategory::Attention as usize].count, 1);
    assert_eq!(cats[BrickCategory::Ffn as usize].count, 1);

    // Total should be sum of all categories
    let cat_total: u64 = cats.iter().map(|c| c.total_ns).sum();
    assert_eq!(cat_total, profiler.total_ns());
}

#[test]
fn test_brick_profiler_reset_v2() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let timer = profiler.start_brick(BrickId::RmsNorm);
    profiler.stop_brick(timer, 1);

    assert!(profiler.total_ns() > 0);

    profiler.reset();

    assert_eq!(profiler.total_ns(), 0);
    assert_eq!(profiler.total_tokens(), 0);
    assert_eq!(profiler.brick_stats(BrickId::RmsNorm).count, 0);
}

#[test]
fn test_sync_mode_default() {
    let profiler = BrickProfiler::new();
    assert_eq!(profiler.sync_mode(), SyncMode::Deferred);
}

#[test]
fn test_brick_id_count() {
    assert_eq!(BrickId::COUNT, 23);
    assert_eq!(BrickCategory::COUNT, 7);
}

// ========================================================================
// Reporting coverage tests
// ========================================================================

#[test]
fn test_profiler_summary_with_data() {
    use std::time::Duration;
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    profiler.record_elapsed("RmsNorm", Duration::from_micros(100), 50);
    profiler.record_elapsed("RmsNorm", Duration::from_micros(200), 50);
    profiler.record_elapsed("Embedding", Duration::from_micros(50), 10);

    let summary = profiler.summary();
    assert!(summary.contains("Brick Profiler Summary"));
    assert!(summary.contains("RmsNorm"));
    assert!(summary.contains("Category Breakdown"));
}

#[test]
fn test_profiler_summary_empty() {
    let profiler = BrickProfiler::new();
    let summary = profiler.summary();
    assert!(summary.contains("Brick Profiler Summary"));
    assert!(summary.contains("Total: 0 tokens"));
}

#[test]
fn test_profiler_print_category_stats() {
    use std::time::Duration;
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    profiler.record_elapsed("RmsNorm", Duration::from_micros(100), 50);
    profiler.record_elapsed("QkvProjection", Duration::from_micros(200), 50);

    // Just verify it doesn't panic
    profiler.print_category_stats();
}

#[test]
fn test_profiler_print_category_stats_empty() {
    let profiler = BrickProfiler::new();
    // Empty profiler - all categories have count 0
    profiler.print_category_stats();
}

#[test]
fn test_profiler_to_json_structure() {
    use std::time::Duration;
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    profiler.record_elapsed("RmsNorm", Duration::from_micros(100), 50);

    let json = profiler.to_json();
    assert!(json.contains("\"total_tokens\":"));
    assert!(json.contains("\"total_ns\":"));
    assert!(json.contains("\"total_throughput\":"));
    assert!(json.contains("\"bricks\":["));
    assert!(json.contains("\"name\":\"RmsNorm\""));
}

#[test]
fn test_profiler_to_json_with_dynamic_brick() {
    use std::time::Duration;
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    profiler.record_elapsed("CustomOp", Duration::from_micros(500), 100);

    let json = profiler.to_json();
    assert!(json.contains("\"name\":\"CustomOp\""));
    assert!(json.contains("\"count\":1"));
}

#[test]
fn test_profiler_write_json() {
    use std::time::Duration;
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.record_elapsed("RmsNorm", Duration::from_micros(100), 50);

    let tmp = std::env::temp_dir().join("trueno_test_profiler.json");
    profiler.write_json(&tmp).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert!(content.contains("\"total_tokens\":"));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_profiler_tile_summary_empty() {
    let profiler = BrickProfiler::new();
    let summary = profiler.tile_summary();
    assert!(summary.contains("Tile Profiling Summary"));
}

#[test]
fn test_profiler_tile_stats_to_json_empty() {
    let profiler = BrickProfiler::new();
    let json = profiler.tile_stats_to_json();
    assert!(json.contains("\"tile_profiling_enabled\":"));
    assert!(json.contains("\"tiles\":[]"));
}
