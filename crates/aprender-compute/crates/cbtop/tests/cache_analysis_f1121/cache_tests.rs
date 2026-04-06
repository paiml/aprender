//! cache_analysis_f1121 - Part 1

use cbtop::{matrix_working_set, CacheConfig, CacheLevel, WorkingSetAnalysis};

// =============================================================================
// F1121: Cache Level Classification Tests
// =============================================================================

/// F1121.1: L1 cache classification for small working sets
#[test]
fn f1121_l1_classification() {
    let config = CacheConfig::default();

    // 1KB should fit in L1 (32KB default)
    assert_eq!(config.classify(1024), CacheLevel::L1);

    // 20KB should fit in L1 (< 32KB * 0.75 = 24KB)
    assert_eq!(config.classify(20 * 1024), CacheLevel::L1);
}

/// F1121.2: L2 cache classification
#[test]
fn f1121_l2_classification() {
    let config = CacheConfig::default();

    // 100KB should fit in L2 (512KB default)
    assert_eq!(config.classify(100 * 1024), CacheLevel::L2);

    // 300KB should fit in L2
    assert_eq!(config.classify(300 * 1024), CacheLevel::L2);
}

/// F1121.3: L3 cache classification
#[test]
fn f1121_l3_classification() {
    let config = CacheConfig::default();

    // 10MB should fit in L3 (32MB default)
    assert_eq!(config.classify(10 * 1024 * 1024), CacheLevel::L3);

    // 20MB should fit in L3
    assert_eq!(config.classify(20 * 1024 * 1024), CacheLevel::L3);
}

/// F1121.4: RAM classification for large working sets
#[test]
fn f1121_ram_classification() {
    let config = CacheConfig::default();

    // 100MB should overflow to RAM
    assert_eq!(config.classify(100 * 1024 * 1024), CacheLevel::Ram);

    // 1GB should be in RAM
    assert_eq!(config.classify(1024 * 1024 * 1024), CacheLevel::Ram);
}

/// F1121.5: Cache level names
#[test]
fn f1121_cache_level_names() {
    assert_eq!(CacheLevel::L1.name(), "L1 cache");
    assert_eq!(CacheLevel::L2.name(), "L2 cache");
    assert_eq!(CacheLevel::L3.name(), "L3 cache");
    assert_eq!(CacheLevel::Ram.name(), "main memory");
}

/// F1121.6: Cache level latency ordering
#[test]
fn f1121_latency_ordering() {
    let l1_lat = CacheLevel::L1.typical_latency_cycles();
    let l2_lat = CacheLevel::L2.typical_latency_cycles();
    let l3_lat = CacheLevel::L3.typical_latency_cycles();
    let ram_lat = CacheLevel::Ram.typical_latency_cycles();

    assert!(l1_lat < l2_lat);
    assert!(l2_lat < l3_lat);
    assert!(l3_lat < ram_lat);
}

/// F1121.7: Relative bandwidth ordering
#[test]
fn f1121_bandwidth_ordering() {
    let l1_bw = CacheLevel::L1.relative_bandwidth();
    let l2_bw = CacheLevel::L2.relative_bandwidth();
    let l3_bw = CacheLevel::L3.relative_bandwidth();
    let ram_bw = CacheLevel::Ram.relative_bandwidth();

    assert!(l1_bw >= l2_bw);
    assert!(l2_bw >= l3_bw);
    assert!(l3_bw >= ram_bw);
    assert_eq!(l1_bw, 1.0); // L1 is baseline
}

// =============================================================================
// F1122: Cache Config Presets Tests
// =============================================================================

/// F1122.1: Default config has reasonable values
#[test]
fn f1122_default_config() {
    let config = CacheConfig::default();

    assert_eq!(config.l1_size, 32 * 1024);
    assert_eq!(config.l2_size, 512 * 1024);
    assert_eq!(config.l3_size, 32 * 1024 * 1024);
    assert_eq!(config.line_size, 64);
}

/// F1122.2: Zen4 preset
#[test]
fn f1122_zen4_config() {
    let config = CacheConfig::zen4();

    assert_eq!(config.l1_size, 32 * 1024);
    assert_eq!(config.l2_size, 1024 * 1024);
    assert_eq!(config.l3_size, 32 * 1024 * 1024);
    assert_eq!(config.l3_sharing, 8);
}

/// F1122.3: Sapphire Rapids preset
#[test]
fn f1122_sapphire_rapids_config() {
    let config = CacheConfig::sapphire_rapids();

    assert_eq!(config.l1_size, 48 * 1024);
    assert_eq!(config.l2_size, 2 * 1024 * 1024);
    assert_eq!(config.l3_size, 60 * 1024 * 1024);
    assert_eq!(config.l3_sharing, 16);
}

/// F1122.4: Apple M2 preset (no L3)
#[test]
fn f1122_apple_m2_config() {
    let config = CacheConfig::apple_m2();

    assert_eq!(config.l1_size, 128 * 1024);
    assert_eq!(config.l2_size, 16 * 1024 * 1024);
    assert_eq!(config.l3_size, 0); // No L3
    assert_eq!(config.line_size, 128); // Larger cache lines
}

/// F1122.5: Apple M2 classification skips L3
#[test]
fn f1122_m2_skips_l3() {
    let config = CacheConfig::apple_m2();

    // Large working set should go directly to RAM (no L3)
    assert_eq!(config.classify(20 * 1024 * 1024), CacheLevel::Ram);
}

/// F1122.6: L3 per core calculation
#[test]
fn f1122_l3_per_core() {
    let config = CacheConfig::default();

    // 32MB / 8 cores = 4MB per core
    assert_eq!(config.l3_per_core(), 4 * 1024 * 1024);
}

// =============================================================================
// F1123: Working Set Analysis Tests
// =============================================================================

/// F1123.1: Small working set fits in L1
#[test]
fn f1123_small_working_set() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(1000, 4, 1.0, &config);

    // 1000 elements * 4 bytes * 1.0 factor = 4KB
    assert_eq!(analysis.working_set_bytes, 4000);
    assert_eq!(analysis.cache_level, CacheLevel::L1);
    assert!(!analysis.tiling_recommended);
    assert!(analysis.recommended_tile_bytes.is_none());
}

/// F1123.2: Large working set triggers tiling
#[test]
fn f1123_large_working_set_tiling() {
    let config = CacheConfig::default();
    // 1M elements * 4 bytes * 2.0 factor = 8MB > L2
    let analysis = WorkingSetAnalysis::analyze(1_000_000, 4, 2.0, &config);

    assert!(analysis.tiling_recommended);
    assert!(analysis.recommended_tile_bytes.is_some());

    // Recommended tile should be 75% of L2
    let expected_tile = config.l2_size * 3 / 4;
    assert_eq!(analysis.recommended_tile_bytes.unwrap(), expected_tile);
}

/// F1123.3: Access factor multiplier
#[test]
fn f1123_access_factor() {
    let config = CacheConfig::default();

    let analysis1 = WorkingSetAnalysis::analyze(1000, 4, 1.0, &config);
    let analysis2 = WorkingSetAnalysis::analyze(1000, 4, 3.0, &config);

    // 3x access factor should triple working set
    assert_eq!(analysis1.working_set_bytes * 3, analysis2.working_set_bytes);
}

/// F1123.4: Utilization percentage
#[test]
fn f1123_utilization_percent() {
    let config = CacheConfig::default();

    // 16KB / 32KB L1 = 50% utilization
    let analysis = WorkingSetAnalysis::analyze(4096, 4, 1.0, &config);

    assert!(analysis.utilization_percent > 45.0);
    assert!(analysis.utilization_percent < 55.0);
}

/// F1123.5: Expected efficiency from cache level
#[test]
fn f1123_expected_efficiency() {
    let config = CacheConfig::default();

    let l1_analysis = WorkingSetAnalysis::analyze(1000, 4, 1.0, &config);
    let ram_analysis = WorkingSetAnalysis::analyze(100_000_000, 4, 1.0, &config);

    // L1 should have higher efficiency than RAM
    assert!(l1_analysis.expected_efficiency > ram_analysis.expected_efficiency);
}

/// F1123.6: Recommendation message contains size
#[test]
fn f1123_recommendation_message() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(1000, 4, 1.0, &config);

    let rec = analysis.recommendation();
    assert!(rec.contains("4000"));
    assert!(rec.contains("fits"));
}

// =============================================================================
// F1124: Matrix Working Set Tests
// =============================================================================

/// F1124.1: Square matrix working set
#[test]
fn f1124_square_matrix() {
    // C = A × B: 1024×1024 matrices
    let ws = matrix_working_set(1024, 1024, 1024, 4);

    // 3 matrices × 1024² × 4 bytes = 12MB
    assert_eq!(ws, 3 * 1024 * 1024 * 4);
}

/// F1124.2: Rectangular matrix working set
#[test]
fn f1124_rectangular_matrix() {
    // C (100×50) = A (100×200) × B (200×50)
    let ws = matrix_working_set(100, 50, 200, 4);

    // A: 100×200×4 = 80,000 bytes
    // B: 200×50×4 = 40,000 bytes
    // C: 100×50×4 = 20,000 bytes
    // Total: 140,000 bytes
    assert_eq!(ws, 140_000);
}

/// F1124.3: Double precision matrix
#[test]
fn f1124_double_precision() {
    let ws_f32 = matrix_working_set(512, 512, 512, 4);
    let ws_f64 = matrix_working_set(512, 512, 512, 8);

    // f64 should be exactly 2x f32
    assert_eq!(ws_f64, ws_f32 * 2);
}
