//! cache_analysis_f1121 - Part 2

use cbtop::{
    elementwise_working_set, matrix_working_set, optimal_matmul_tile, AccessPattern,
    BandwidthPrediction, CacheConfig, CacheLevel, WorkingSetAnalysis,
};

// =============================================================================
// F1125: Optimal Tile Size Tests
// =============================================================================

/// F1125.1: Tile size is positive
#[test]
fn f1125_tile_positive() {
    let config = CacheConfig::default();
    let tile = optimal_matmul_tile(&config, 4);

    assert!(tile > 0);
}

/// F1125.2: Tile size is cache-aligned
#[test]
fn f1125_tile_aligned() {
    let config = CacheConfig::default();
    let tile = optimal_matmul_tile(&config, 4);

    // Tile should be multiple of elements per cache line
    let elements_per_line = config.line_size / 4;
    assert_eq!(tile % elements_per_line, 0);
}

/// F1125.3: Tile fits in L2
#[test]
fn f1125_tile_fits_l2() {
    let config = CacheConfig::default();
    let tile = optimal_matmul_tile(&config, 4);

    // 3 tiles × tile² × 4 bytes should fit in 75% of L2
    let tile_bytes = 3 * tile * tile * 4;
    let target = config.l2_size * 3 / 4;

    assert!(tile_bytes <= target);
}

/// F1125.4: Different element sizes
#[test]
fn f1125_different_element_sizes() {
    let config = CacheConfig::default();

    let tile_f32 = optimal_matmul_tile(&config, 4);
    let tile_f64 = optimal_matmul_tile(&config, 8);

    // Larger elements should give smaller tile size
    assert!(tile_f32 >= tile_f64);
}

// =============================================================================
// F1126: Elementwise Working Set Tests
// =============================================================================

/// F1126.1: Single input single output
#[test]
fn f1126_single_io() {
    // sqrt: 1 input, 1 output
    let ws = elementwise_working_set(1000, 1, 1, 4);

    // 1000 × (1 + 1) × 4 = 8000 bytes
    assert_eq!(ws, 8000);
}

/// F1126.2: Multiple inputs
#[test]
fn f1126_multiple_inputs() {
    // add: 2 inputs, 1 output
    let ws = elementwise_working_set(1000, 2, 1, 4);

    // 1000 × (2 + 1) × 4 = 12000 bytes
    assert_eq!(ws, 12000);
}

/// F1126.3: Fused operation
#[test]
fn f1126_fused_op() {
    // fma: 3 inputs, 1 output
    let ws = elementwise_working_set(10000, 3, 1, 4);

    // 10000 × (3 + 1) × 4 = 160000 bytes
    assert_eq!(ws, 160000);
}

// =============================================================================
// F1127: Access Pattern Tests
// =============================================================================

/// F1127.1: Single iteration is streaming
#[test]
fn f1127_streaming_pattern() {
    let config = CacheConfig::default();
    let pattern = AccessPattern::estimate(1024, 1, &config);

    assert_eq!(pattern, AccessPattern::Streaming);
}

/// F1127.2: Small with reuse
#[test]
fn f1127_reuse_pattern() {
    let config = CacheConfig::default();
    // Small working set with multiple iterations
    let pattern = AccessPattern::estimate(1024, 10, &config);

    assert_eq!(pattern, AccessPattern::Reuse);
}

/// F1127.3: Large with iterations is random
#[test]
fn f1127_random_pattern() {
    let config = CacheConfig::default();
    // Large working set with multiple iterations
    let pattern = AccessPattern::estimate(100 * 1024 * 1024, 10, &config);

    assert_eq!(pattern, AccessPattern::Random);
}

/// F1127.4: Pattern names
#[test]
fn f1127_pattern_names() {
    assert_eq!(AccessPattern::Streaming.name(), "streaming");
    assert_eq!(AccessPattern::Reuse.name(), "reuse");
    assert_eq!(AccessPattern::Random.name(), "random");
}

/// F1127.5: Efficiency factors
#[test]
fn f1127_efficiency_factors() {
    // Reuse should be best, random worst
    assert!(
        AccessPattern::Reuse.efficiency_factor() > AccessPattern::Streaming.efficiency_factor()
    );
    assert!(
        AccessPattern::Streaming.efficiency_factor() > AccessPattern::Random.efficiency_factor()
    );
}

// =============================================================================
// F1128: Bandwidth Prediction Tests
// =============================================================================

/// F1128.1: L1 reuse near peak
#[test]
fn f1128_l1_reuse_prediction() {
    let config = CacheConfig::default();
    let prediction = BandwidthPrediction::predict(
        100.0, // 100 GB/s peak
        1024,  // 1KB working set (L1)
        AccessPattern::Reuse,
        &config,
    );

    // L1 with reuse should be near 100%
    assert!(prediction.efficiency_percent > 90.0);
    assert!(prediction.predicted_bandwidth_gbps > 90.0);
}

/// F1128.2: RAM random is slow
#[test]
fn f1128_ram_random_prediction() {
    let config = CacheConfig::default();
    let prediction = BandwidthPrediction::predict(
        100.0,
        100 * 1024 * 1024, // 100MB (RAM)
        AccessPattern::Random,
        &config,
    );

    // RAM with random should be very low
    assert!(prediction.efficiency_percent < 20.0);
}

/// F1128.3: Limiting factor identification
#[test]
fn f1128_limiting_factor() {
    let config = CacheConfig::default();

    // Random pattern should be the limiting factor for L1
    let prediction = BandwidthPrediction::predict(100.0, 1024, AccessPattern::Random, &config);

    assert!(prediction.limiting_factor.contains("random"));
}

/// F1128.4: Peak bandwidth preserved
#[test]
fn f1128_peak_preserved() {
    let config = CacheConfig::default();
    let prediction = BandwidthPrediction::predict(200.0, 1024, AccessPattern::Reuse, &config);

    assert_eq!(prediction.peak_bandwidth_gbps, 200.0);
}

// =============================================================================
// F1129: Edge Cases Tests
// =============================================================================

/// F1129.1: Zero elements
#[test]
fn f1129_zero_elements() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(0, 4, 1.0, &config);

    assert_eq!(analysis.working_set_bytes, 0);
    assert_eq!(analysis.cache_level, CacheLevel::L1);
}

/// F1129.2: Huge working set
#[test]
fn f1129_huge_working_set() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(1_000_000_000, 8, 1.0, &config);

    assert_eq!(analysis.cache_level, CacheLevel::Ram);
    assert!(analysis.tiling_recommended);
}

/// F1129.3: Very small access factor
#[test]
fn f1129_small_access_factor() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(1_000_000, 4, 0.001, &config);

    // Should still produce valid result
    assert!(analysis.working_set_bytes > 0);
}

// =============================================================================
// F1130: Integration Tests
// =============================================================================

/// F1130.1: §31.2 memory bandwidth cliff at 4M elements
#[test]
fn f1130_bandwidth_cliff() {
    let config = CacheConfig::default();

    // 4M elements × 8 bytes = 32MB (exactly L3 size)
    let analysis = WorkingSetAnalysis::analyze(4_000_000, 8, 1.0, &config);

    // Should trigger L3/RAM boundary
    assert!(matches!(
        analysis.cache_level,
        CacheLevel::L3 | CacheLevel::Ram
    ));
    assert!(analysis.tiling_recommended);
}

/// F1130.2: Optimal tile maintains L2 residency
#[test]
fn f1130_tile_l2_residency() {
    let config = CacheConfig::default();
    let tile = optimal_matmul_tile(&config, 4);

    // Use tile for matrix working set
    let ws = matrix_working_set(tile, tile, tile, 4);

    // Should fit in L2
    assert!(config.classify(ws) == CacheLevel::L2 || config.classify(ws) == CacheLevel::L1);
}

/// F1130.3: Real-world LLM attention pattern
#[test]
fn f1130_llm_attention() {
    let config = CacheConfig::zen4();

    // Attention: Q, K, V matrices for batch=32, seq=2048, dim=4096
    let batch = 32;
    let seq = 2048;
    let dim = 4096;
    let element_size = 2; // fp16

    // Q, K, V each are batch × seq × dim
    let _qkv_size = 3 * batch * seq * dim * element_size;

    let analysis = WorkingSetAnalysis::analyze(
        batch * seq * dim * 3, // total elements
        element_size,
        1.0,
        &config,
    );

    // This should overflow L3 and require tiling
    assert_eq!(analysis.cache_level, CacheLevel::Ram);
    assert!(analysis.tiling_recommended);
}

/// F1130.4: Streaming workload prediction
#[test]
fn f1130_streaming_workload() {
    let config = CacheConfig::default();

    // Large streaming workload (video processing)
    let prediction = BandwidthPrediction::predict(
        50.0,               // 50 GB/s memory bandwidth
        1024 * 1024 * 1024, // 1GB stream
        AccessPattern::Streaming,
        &config,
    );

    // Streaming should achieve ~50% of peak (prefetching helps)
    assert!(prediction.efficiency_percent > 4.0);
    assert!(prediction.efficiency_percent < 10.0); // RAM × streaming = 0.1 × 0.5 = 5%
}
