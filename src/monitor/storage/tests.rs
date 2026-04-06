//! Tests for metrics storage

use super::*;
use crate::monitor::{Metric, MetricRecord};

#[test]
fn test_in_memory_store_new() {
    let store = InMemoryStore::new();
    assert_eq!(store.count().expect("operation should succeed"), 0);
}

#[test]
fn test_in_memory_write_batch() {
    let mut store = InMemoryStore::new();
    let records =
        vec![MetricRecord::new(Metric::Loss, 0.5), MetricRecord::new(Metric::Accuracy, 0.85)];
    store.write_batch(&records).expect("file write should succeed");
    assert_eq!(store.count().expect("operation should succeed"), 2);
}

#[test]
fn test_in_memory_query_all() {
    let mut store = InMemoryStore::new();
    store
        .write_batch(&[
            MetricRecord::new(Metric::Loss, 0.5),
            MetricRecord::new(Metric::Loss, 0.4),
            MetricRecord::new(Metric::Accuracy, 0.85),
        ])
        .expect("operation should succeed");

    let loss_records = store.query_all(&Metric::Loss).expect("operation should succeed");
    assert_eq!(loss_records.len(), 2);
}

#[test]
fn test_in_memory_query_stats() {
    let mut store = InMemoryStore::new();
    store
        .write_batch(&[
            MetricRecord::new(Metric::Loss, 1.0),
            MetricRecord::new(Metric::Loss, 2.0),
            MetricRecord::new(Metric::Loss, 3.0),
        ])
        .expect("operation should succeed");

    let stats = store
        .query_stats(&Metric::Loss)
        .expect("operation should succeed")
        .expect("operation should succeed");
    assert!((stats.mean - 2.0).abs() < 1e-6);
    assert!((stats.min - 1.0).abs() < 1e-6);
    assert!((stats.max - 3.0).abs() < 1e-6);
    assert_eq!(stats.count, 3);
}

#[test]
fn test_in_memory_query_stats_empty() {
    let store = InMemoryStore::new();
    let stats = store.query_stats(&Metric::Loss).expect("operation should succeed");
    assert!(stats.is_none());
}

#[test]
fn test_json_file_store_create() {
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("metrics.json");

    let store = JsonFileStore::open(&path).expect("operation should succeed");
    assert_eq!(store.count().expect("operation should succeed"), 0);
}

#[test]
fn test_json_file_store_write_and_flush() {
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("metrics.json");

    {
        let mut store = JsonFileStore::open(&path).expect("operation should succeed");
        store
            .write_batch(&[MetricRecord::new(Metric::Loss, 0.5)])
            .expect("file write should succeed");
        store.flush().expect("operation should succeed");
    }

    // Reopen and verify
    let store = JsonFileStore::open(&path).expect("operation should succeed");
    assert_eq!(store.count().expect("operation should succeed"), 1);
}

#[test]
fn test_json_file_store_persistence() {
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("metrics.json");

    // Write some records
    {
        let mut store = JsonFileStore::open(&path).expect("operation should succeed");
        store
            .write_batch(&[
                MetricRecord::new(Metric::Loss, 0.5),
                MetricRecord::new(Metric::Accuracy, 0.85),
            ])
            .expect("operation should succeed");
        // Drop triggers flush
    }

    // Reopen and add more
    {
        let mut store = JsonFileStore::open(&path).expect("operation should succeed");
        assert_eq!(store.count().expect("operation should succeed"), 2);
        store
            .write_batch(&[MetricRecord::new(Metric::Loss, 0.4)])
            .expect("file write should succeed");
    }

    // Verify final count
    let store = JsonFileStore::open(&path).expect("operation should succeed");
    assert_eq!(store.count().expect("operation should succeed"), 3);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_in_memory_store_default() {
    let store = InMemoryStore::default();
    assert_eq!(store.count().expect("operation should succeed"), 0);
    assert!(store.all_records().is_empty());
}

#[test]
fn test_in_memory_store_all_records() {
    let mut store = InMemoryStore::new();
    store.write_batch(&[MetricRecord::new(Metric::Loss, 0.5)]).expect("file write should succeed");
    assert_eq!(store.all_records().len(), 1);
}

#[test]
fn test_in_memory_store_flush() {
    let mut store = InMemoryStore::new();
    assert!(store.flush().is_ok());
}

#[test]
fn test_in_memory_query_range() {
    let mut store = InMemoryStore::new();
    let mut r1 = MetricRecord::new(Metric::Loss, 0.5);
    r1.timestamp = 100;
    let mut r2 = MetricRecord::new(Metric::Loss, 0.4);
    r2.timestamp = 200;
    let mut r3 = MetricRecord::new(Metric::Loss, 0.3);
    r3.timestamp = 300;
    store.write_batch(&[r1, r2, r3]).expect("file write should succeed");

    let results = store.query_range(&Metric::Loss, 150, 250).expect("operation should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].timestamp, 200);
}

#[test]
fn test_in_memory_query_range_empty() {
    let store = InMemoryStore::new();
    let results = store.query_range(&Metric::Loss, 0, 1000).expect("operation should succeed");
    assert!(results.is_empty());
}

#[test]
fn test_in_memory_query_stats_single_value() {
    let mut store = InMemoryStore::new();
    store.write_batch(&[MetricRecord::new(Metric::Loss, 5.0)]).expect("file write should succeed");

    let stats = store
        .query_stats(&Metric::Loss)
        .expect("operation should succeed")
        .expect("operation should succeed");
    assert_eq!(stats.count, 1);
    assert!((stats.mean - 5.0).abs() < 1e-6);
    assert_eq!(stats.std, 0.0); // Single value has 0 std
}

#[test]
fn test_in_memory_query_stats_nan() {
    let mut store = InMemoryStore::new();
    store
        .write_batch(&[
            MetricRecord::new(Metric::Loss, f64::NAN),
            MetricRecord::new(Metric::Loss, 1.0),
        ])
        .expect("operation should succeed");

    let stats = store
        .query_stats(&Metric::Loss)
        .expect("operation should succeed")
        .expect("operation should succeed");
    assert!(stats.has_nan);
}

#[test]
fn test_in_memory_query_stats_inf() {
    let mut store = InMemoryStore::new();
    store
        .write_batch(&[
            MetricRecord::new(Metric::Loss, f64::INFINITY),
            MetricRecord::new(Metric::Loss, 1.0),
        ])
        .expect("operation should succeed");

    let stats = store
        .query_stats(&Metric::Loss)
        .expect("operation should succeed")
        .expect("operation should succeed");
    assert!(stats.has_inf);
}

#[test]
fn test_json_file_store_path() {
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("test.json");
    let store = JsonFileStore::open(&path).expect("operation should succeed");
    assert_eq!(store.path(), path);
}

#[test]
fn test_json_file_store_query_range() {
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("metrics.json");

    let mut store = JsonFileStore::open(&path).expect("operation should succeed");
    let mut r1 = MetricRecord::new(Metric::Loss, 0.5);
    r1.timestamp = 100;
    let mut r2 = MetricRecord::new(Metric::Loss, 0.4);
    r2.timestamp = 200;
    store.write_batch(&[r1, r2]).expect("file write should succeed");

    let results = store.query_range(&Metric::Loss, 50, 150).expect("operation should succeed");
    assert_eq!(results.len(), 1);
}

#[test]
fn test_json_file_store_query_all() {
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("metrics.json");

    let mut store = JsonFileStore::open(&path).expect("operation should succeed");
    store
        .write_batch(&[
            MetricRecord::new(Metric::Loss, 0.5),
            MetricRecord::new(Metric::Accuracy, 0.85),
        ])
        .expect("operation should succeed");

    let loss = store.query_all(&Metric::Loss).expect("operation should succeed");
    assert_eq!(loss.len(), 1);
}

#[test]
fn test_json_file_store_query_stats() {
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("metrics.json");

    let mut store = JsonFileStore::open(&path).expect("operation should succeed");
    store
        .write_batch(&[MetricRecord::new(Metric::Loss, 1.0), MetricRecord::new(Metric::Loss, 2.0)])
        .expect("operation should succeed");

    let stats = store
        .query_stats(&Metric::Loss)
        .expect("operation should succeed")
        .expect("operation should succeed");
    assert!((stats.mean - 1.5).abs() < 1e-6);
}

#[test]
fn test_json_file_store_flush_not_dirty() {
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("metrics.json");

    let mut store = JsonFileStore::open(&path).expect("operation should succeed");
    // Flush without writing anything
    assert!(store.flush().is_ok());
}

#[test]
fn test_storage_error_display() {
    let io_err = StorageError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
    assert!(io_err.to_string().contains("IO error"));

    let ser_err = StorageError::Serialization("bad json".to_string());
    assert!(ser_err.to_string().contains("Serialization"));

    let query_err = StorageError::Query("bad query".to_string());
    assert!(query_err.to_string().contains("Query"));

    let init_err = StorageError::NotInitialized;
    assert!(init_err.to_string().contains("not initialized"));
}

#[test]
fn test_storage_error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
    let storage_err: StorageError = io_err.into();
    matches!(storage_err, StorageError::Io(_));
}
