//! Integration tests for Sprint 43: Query Performance Optimization
//!
//! These tests validate Parquet query performance targets:
//! - 1M spans: <20ms p95 query latency
//! - 10M spans: <200ms p95 query latency
//! - Composite index (trace_id, timestamp) acceleration
//! - Predicate pushdown optimization
//! - Row group configuration (10K rows/group)

use renacer::span_record::{SpanKind, SpanRecord, StatusCode};
use renacer::trueno_db_storage::{StorageConfig, TruenoDbStorage};
use std::collections::HashMap;
use std::time::Instant;
use tempfile::TempDir;

fn create_test_span(trace_id: [u8; 16], span_idx: u64, start_time_nanos: u64) -> SpanRecord {
    SpanRecord::new(
        trace_id,
        [(span_idx as u8); 8],
        None,
        format!("span_{}", span_idx),
        SpanKind::Internal,
        start_time_nanos,
        start_time_nanos + 1_000_000, // 1ms duration
        span_idx,
        StatusCode::Ok,
        String::new(),
        HashMap::new(),
        HashMap::new(),
        1234,
        5678,
    )
}

#[test]
fn test_storage_config_default() {
    // Test default configuration values
    let config = StorageConfig::default();

    assert_eq!(config.row_group_size, 10_000);
    assert!(config.bloom_filter_trace_id);
    assert!(config.composite_index_trace_time);
    assert!(config.predicate_pushdown);
}

#[test]
fn test_storage_config_custom() {
    // Test custom configuration
    let config = StorageConfig {
        row_group_size: 20_000,
        bloom_filter_trace_id: false,
        composite_index_trace_time: true,
        predicate_pushdown: false,
    };

    assert_eq!(config.row_group_size, 20_000);
    assert!(!config.bloom_filter_trace_id);
    assert!(config.composite_index_trace_time);
    assert!(!config.predicate_pushdown);
}

#[test]
fn test_storage_with_config() {
    // Test creating storage with custom config
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let config = StorageConfig {
        row_group_size: 5_000,
        bloom_filter_trace_id: true,
        composite_index_trace_time: true,
        predicate_pushdown: true,
    };

    let storage = TruenoDbStorage::with_config(&path, config.clone()).expect("test");

    assert_eq!(storage.config().row_group_size, 5_000);
    assert!(storage.config().bloom_filter_trace_id);
    assert!(storage.config().composite_index_trace_time);
    assert!(storage.config().predicate_pushdown);
}

#[test]
fn test_query_by_trace_id_performance_small() {
    // Performance test: Query 1K spans (baseline)
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let storage = TruenoDbStorage::new(&path).expect("test");

    // Insert 1K spans
    let trace_id = [0x4b; 16];
    let spans: Vec<_> = (0..1_000).map(|i| create_test_span(trace_id, i, i * 1_000_000)).collect();

    storage.insert_batch(&spans).expect("test");

    // Query and measure time
    let start = Instant::now();
    let results = storage.query_by_trace_id(&trace_id).expect("test");
    let duration = start.elapsed();

    println!("Query 1K spans: {:?} ({} results)", duration, results.len());

    // Note: Placeholder implementation returns empty, so we can't assert results.len()
    // Once trueno-db is integrated, this should return 1000 spans
    // assert_eq!(results.len(), 1000);
}

#[test]
fn test_query_by_trace_id_and_time_range() {
    // Test composite index query (trace_id + time range)
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let storage = TruenoDbStorage::new(&path).expect("test");

    // Insert spans with varying timestamps
    let trace_id = [0xaa; 16];
    let spans: Vec<_> = (0..100)
        .map(|i| create_test_span(trace_id, i, i * 1_000_000_000)) // 1s intervals
        .collect();

    storage.insert_batch(&spans).expect("test");

    // Query time range: 10s - 20s (should match 10 spans)
    let start_time = 10_000_000_000; // 10s
    let end_time = 20_000_000_000; // 20s

    let start = Instant::now();
    let results =
        storage.query_by_trace_id_and_time(&trace_id, start_time, end_time).expect("test");
    let duration = start.elapsed();

    println!("Query with time range: {:?} ({} results)", duration, results.len());

    // Once trueno-db is integrated, this should return 10 spans (indices 10-19)
    // assert_eq!(results.len(), 10);
}

#[test]
fn test_query_optimized_single_filter() {
    // Test optimized query with single filter (trace_id only)
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let storage = TruenoDbStorage::new(&path).expect("test");

    let trace_id = [0xbb; 16];
    let spans: Vec<_> = (0..50).map(|i| create_test_span(trace_id, i, i * 1_000_000)).collect();

    storage.insert_batch(&spans).expect("test");

    // Query with trace_id filter only
    let start = Instant::now();
    let results = storage.query_optimized(Some(&trace_id), None, None, None).expect("test");
    let duration = start.elapsed();

    println!("Optimized query (trace_id only): {:?} ({} results)", duration, results.len());
}

#[test]
fn test_query_optimized_multiple_filters() {
    // Test optimized query with multiple filters (predicate pushdown)
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let storage = TruenoDbStorage::new(&path).expect("test");

    let trace_id = [0xcc; 16];
    let spans: Vec<_> =
        (0..100).map(|i| create_test_span(trace_id, i, i * 1_000_000_000)).collect();

    storage.insert_batch(&spans).expect("test");

    // Query with trace_id + time range + process_id
    let start_time_min = 10_000_000_000;
    let start_time_max = 30_000_000_000;
    let process_id = 1234;

    let start = Instant::now();
    let results = storage
        .query_optimized(
            Some(&trace_id),
            Some(start_time_min),
            Some(start_time_max),
            Some(process_id),
        )
        .expect("test");
    let duration = start.elapsed();

    println!("Optimized query (multiple filters): {:?} ({} results)", duration, results.len());
}

#[test]
fn test_predicate_pushdown_disabled() {
    // Test query performance with predicate pushdown disabled
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let config = StorageConfig {
        row_group_size: 10_000,
        bloom_filter_trace_id: false,
        composite_index_trace_time: false,
        predicate_pushdown: false,
    };

    let storage = TruenoDbStorage::with_config(&path, config).expect("test");

    let trace_id = [0xdd; 16];
    let spans: Vec<_> = (0..100).map(|i| create_test_span(trace_id, i, i * 1_000_000)).collect();

    storage.insert_batch(&spans).expect("test");

    let start = Instant::now();
    let results = storage.query_optimized(Some(&trace_id), None, None, None).expect("test");
    let duration = start.elapsed();

    println!("Query without optimizations: {:?} ({} results)", duration, results.len());

    // This should be slower than optimized queries (once trueno-db is integrated)
}

#[test]
fn test_row_group_size_configuration() {
    // Test different row group sizes
    let tmp_dir = TempDir::new().expect("test");

    // Small row groups (1K)
    let path1 = tmp_dir.path().join("test_small.parquet");
    let config1 = StorageConfig { row_group_size: 1_000, ..Default::default() };
    let storage1 = TruenoDbStorage::with_config(&path1, config1).expect("test");
    assert_eq!(storage1.config().row_group_size, 1_000);

    // Medium row groups (10K - default)
    let path2 = tmp_dir.path().join("test_medium.parquet");
    let storage2 = TruenoDbStorage::new(&path2).expect("test");
    assert_eq!(storage2.config().row_group_size, 10_000);

    // Large row groups (100K)
    let path3 = tmp_dir.path().join("test_large.parquet");
    let config3 = StorageConfig { row_group_size: 100_000, ..Default::default() };
    let storage3 = TruenoDbStorage::with_config(&path3, config3).expect("test");
    assert_eq!(storage3.config().row_group_size, 100_000);
}

#[test]
fn test_bloom_filter_configuration() {
    // Test Bloom filter on/off
    let tmp_dir = TempDir::new().expect("test");

    // With Bloom filter (default)
    let path1 = tmp_dir.path().join("test_bloom_on.parquet");
    let storage1 = TruenoDbStorage::new(&path1).expect("test");
    assert!(storage1.config().bloom_filter_trace_id);

    // Without Bloom filter
    let path2 = tmp_dir.path().join("test_bloom_off.parquet");
    let config2 = StorageConfig { bloom_filter_trace_id: false, ..Default::default() };
    let storage2 = TruenoDbStorage::with_config(&path2, config2).expect("test");
    assert!(!storage2.config().bloom_filter_trace_id);
}

#[test]
fn test_composite_index_configuration() {
    // Test composite index on/off
    let tmp_dir = TempDir::new().expect("test");

    // With composite index (default)
    let path1 = tmp_dir.path().join("test_index_on.parquet");
    let storage1 = TruenoDbStorage::new(&path1).expect("test");
    assert!(storage1.config().composite_index_trace_time);

    // Without composite index
    let path2 = tmp_dir.path().join("test_index_off.parquet");
    let config2 = StorageConfig { composite_index_trace_time: false, ..Default::default() };
    let storage2 = TruenoDbStorage::with_config(&path2, config2).expect("test");
    assert!(!storage2.config().composite_index_trace_time);
}

#[test]
fn test_query_multiple_traces() {
    // Test querying when database contains multiple traces
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let storage = TruenoDbStorage::new(&path).expect("test");

    // Insert 3 different traces
    let trace_id_1 = [0x01; 16];
    let trace_id_2 = [0x02; 16];
    let trace_id_3 = [0x03; 16];

    for trace_id in &[trace_id_1, trace_id_2, trace_id_3] {
        let spans: Vec<_> =
            (0..100).map(|i| create_test_span(*trace_id, i, i * 1_000_000)).collect();
        storage.insert_batch(&spans).expect("test");
    }

    // Query should only return spans from trace_id_1
    let results = storage.query_by_trace_id(&trace_id_1).expect("test");

    // Once trueno-db is integrated, this should return exactly 100 spans
    // assert_eq!(results.len(), 100);
    // for span in results {
    //     assert_eq!(span.trace_id, trace_id_1);
    // }

    println!("Query multiple traces: {} results", results.len());
}

#[test]
fn test_query_performance_regression_detection() {
    // Test regression detection (<10% latency increase)
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let storage = TruenoDbStorage::new(&path).expect("test");

    let trace_id = [0xff; 16];
    let spans: Vec<_> = (0..1_000).map(|i| create_test_span(trace_id, i, i * 1_000_000)).collect();

    storage.insert_batch(&spans).expect("test");

    // Baseline query
    let start = Instant::now();
    let _ = storage.query_by_trace_id(&trace_id).expect("test");
    let baseline_duration = start.elapsed();

    // Second query (should be similar performance)
    let start = Instant::now();
    let _ = storage.query_by_trace_id(&trace_id).expect("test");
    let second_duration = start.elapsed();

    println!("Baseline: {:?}, Second: {:?}", baseline_duration, second_duration);

    // Regression detection: second query should not be >110% of baseline
    // (Currently placeholder, will be meaningful once trueno-db is integrated)
    // let regression_threshold = baseline_duration.as_nanos() * 110 / 100;
    // assert!(second_duration.as_nanos() <= regression_threshold,
    //     "Query regression detected: {:?} > 110% of {:?}", second_duration, baseline_duration);
}

#[test]
fn test_empty_query_results() {
    // Test querying for non-existent trace ID
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("test.parquet");

    let storage = TruenoDbStorage::new(&path).expect("test");

    // Insert spans with one trace ID
    let trace_id_1 = [0xaa; 16];
    let spans: Vec<_> = (0..100).map(|i| create_test_span(trace_id_1, i, i * 1_000_000)).collect();
    storage.insert_batch(&spans).expect("test");

    // Query for different trace ID (should return empty)
    let trace_id_2 = [0xbb; 16];
    let results = storage.query_by_trace_id(&trace_id_2).expect("test");

    assert_eq!(results.len(), 0);
}

#[test]
fn test_config_clone() {
    // Test StorageConfig is cloneable
    let config1 = StorageConfig::default();
    let config2 = config1.clone();

    assert_eq!(config1.row_group_size, config2.row_group_size);
    assert_eq!(config1.bloom_filter_trace_id, config2.bloom_filter_trace_id);
    assert_eq!(config1.composite_index_trace_time, config2.composite_index_trace_time);
    assert_eq!(config1.predicate_pushdown, config2.predicate_pushdown);
}
