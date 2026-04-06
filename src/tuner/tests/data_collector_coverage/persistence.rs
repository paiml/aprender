//! save_apr/load_apr round-trip and error path tests

use super::*;

// ============================================================================
// save_apr() and load_apr() - round-trip
// ============================================================================

#[test]
fn save_and_load_apr_round_trip() {
    let c = make_collector_with_samples(5);
    let dir = std::env::temp_dir().join("trueno_test_save_load_apr");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("test_data.apr");

    c.save_apr(&path).expect("save should succeed");
    assert!(path.exists());

    let loaded = TunerDataCollector::load_apr(&path).expect("load should succeed");
    assert_eq!(loaded.len(), 5);
    assert_eq!(loaded.samples()[0].throughput_tps, c.samples()[0].throughput_tps);
    assert_eq!(loaded.samples()[4].throughput_tps, c.samples()[4].throughput_tps);

    // Loaded collector should have default state for non-persisted fields
    assert!(!loaded.is_online_learning_enabled());
    assert_eq!(loaded.retrain_threshold, 100);
    assert!(loaded.feedback.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_apr_creates_parent_directories() {
    let dir = std::env::temp_dir().join("trueno_test_nested_dir/a/b/c");
    let path = dir.join("model.apr");
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("trueno_test_nested_dir"));

    let c = make_collector_with_samples(1);
    c.save_apr(&path).expect("save to nested dir should succeed");
    assert!(path.exists());

    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("trueno_test_nested_dir"));
}

#[test]
fn save_apr_empty_collector() {
    let dir = std::env::temp_dir().join("trueno_test_save_empty");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("empty.apr");

    let c = TunerDataCollector::new();
    c.save_apr(&path).expect("save empty should succeed");

    let loaded = TunerDataCollector::load_apr(&path).expect("load empty should succeed");
    assert!(loaded.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// load_apr() - error paths
// ============================================================================

#[test]
fn load_apr_file_not_found() {
    let result = TunerDataCollector::load_apr("/tmp/trueno_nonexistent_file.apr");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("I/O error"));
}

#[test]
fn load_apr_bad_magic() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_bad_magic");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_magic.apr");

    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"XXXX").expect("write magic");
    file.write_all(&4u32.to_le_bytes()).expect("write len");
    file.write_all(b"test").expect("write data");
    file.write_all(&0u32.to_le_bytes()).expect("write crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("APR2"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_crc_mismatch() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_crc_mismatch");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_crc.apr");

    let json_bytes = b"[]";
    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    file.write_all(&(json_bytes.len() as u32).to_le_bytes()).expect("write len");
    file.write_all(json_bytes).expect("write data");
    // Write wrong CRC
    file.write_all(&0xDEADBEEFu32.to_le_bytes()).expect("write bad crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("CRC mismatch"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_truncated_file() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_truncated");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("truncated.apr");

    // Write only magic, no length or data
    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_invalid_json_in_valid_envelope() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_invalid_json_apr");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_json.apr");

    let json_bytes = b"not valid json at all";
    let crc = crate::tuner::helpers::crc32_hash(json_bytes);

    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    file.write_all(&(json_bytes.len() as u32).to_le_bytes()).expect("write len");
    file.write_all(json_bytes).expect("write data");
    file.write_all(&crc.to_le_bytes()).expect("write crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Serialization"));

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// save_apr error path: write to invalid path
// ============================================================================

#[test]
fn save_apr_returns_io_error_for_invalid_path() {
    let c = make_collector_with_samples(1);
    // Try to write to a path that cannot exist on any platform
    #[cfg(unix)]
    let invalid_path = "/proc/nonexistent/deep/path/file.apr";
    #[cfg(windows)]
    let invalid_path = r"\\?\INVALID:\nonexistent\deep\path\file.apr";
    let result = c.save_apr(invalid_path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("I/O error"));
}

// ============================================================================
// record_and_persist() and load_or_create() -- requires hardware-detect
//
// These tests share a single cache file (determined by cache_path()), so they
// are combined into one sequential test to avoid parallel race conditions.
// ============================================================================

#[cfg(feature = "hardware-detect")]
#[test]
fn record_and_persist_and_load_or_create_full_lifecycle() {
    use std::io::Write;

    let cache_path = TunerDataCollector::cache_path();

    // ------------------------------------------------------------------
    // Phase 1: load_or_create falls back to new when no cache exists
    // ------------------------------------------------------------------
    let _ = std::fs::remove_file(&cache_path);
    assert!(!cache_path.exists(), "pre-clean: cache file should not exist");

    let collector = TunerDataCollector::load_or_create();
    assert!(collector.is_empty(), "load_or_create should return empty when no cache");

    // ------------------------------------------------------------------
    // Phase 2: load_or_create falls back to new on corrupt cache
    // ------------------------------------------------------------------
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dir");
    }
    {
        let mut file = std::fs::File::create(&cache_path).expect("create corrupt file");
        file.write_all(b"GARBAGE_DATA_NOT_APR").expect("write garbage");
    }
    assert!(cache_path.exists(), "corrupt file should exist");

    let collector = TunerDataCollector::load_or_create();
    assert!(collector.is_empty(), "load_or_create should fall back to new on corrupt cache");

    // ------------------------------------------------------------------
    // Phase 3: record_and_persist writes sample and saves to cache
    // ------------------------------------------------------------------
    let _ = std::fs::remove_file(&cache_path);

    let mut profiler = BrickProfiler::new();
    profiler.enable();
    // Record 100 tokens over 1ms (= 100_000 tok/s)
    profiler.record_elapsed("TestBrick", std::time::Duration::from_millis(1), 100);

    let config = RunConfig::default();
    let mut collector = TunerDataCollector::new();

    let result = collector.record_and_persist(&profiler, &config, KernelType::TiledQ4K);
    assert!(result.is_ok(), "record_and_persist should succeed");
    assert_eq!(collector.len(), 1, "collector should have one sample after record_and_persist");

    // Verify file was written and is loadable
    assert!(cache_path.exists(), "cache file should exist after record_and_persist");
    let loaded = TunerDataCollector::load_apr(&cache_path).expect("load after persist");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded.samples()[0].throughput_tps, collector.samples()[0].throughput_tps);

    // ------------------------------------------------------------------
    // Phase 4: record_and_persist appends a second sample
    // ------------------------------------------------------------------
    collector
        .record_and_persist(&profiler, &config, KernelType::CoalescedQ4K)
        .expect("second record_and_persist");
    assert_eq!(collector.len(), 2, "collector should have two samples");

    let loaded = TunerDataCollector::load_apr(&cache_path).expect("load after second persist");
    assert_eq!(loaded.len(), 2, "persisted file should have two samples");

    // ------------------------------------------------------------------
    // Phase 5: load_or_create loads the persisted data
    // ------------------------------------------------------------------
    let loaded = TunerDataCollector::load_or_create();
    assert_eq!(loaded.len(), 2, "load_or_create should load the 2 persisted samples");
    assert_eq!(loaded.samples()[0].throughput_tps, collector.samples()[0].throughput_tps);
    assert_eq!(loaded.samples()[1].throughput_tps, collector.samples()[1].throughput_tps);

    // ------------------------------------------------------------------
    // Phase 6: load_or_create returns data with correct default state
    // ------------------------------------------------------------------
    assert!(!loaded.is_online_learning_enabled());
    assert_eq!(loaded.retrain_threshold, 100);
    assert!(loaded.feedback.is_empty());
    assert!(loaded.error_window.is_empty());

    // ------------------------------------------------------------------
    // Cleanup: remove cache file so we don't leave test artifacts
    // ------------------------------------------------------------------
    let _ = std::fs::remove_file(&cache_path);
}

// ============================================================================
// cache_path() and hardware_id() -- requires hardware-detect feature
// ============================================================================

#[cfg(feature = "hardware-detect")]
#[test]
fn cache_path_returns_valid_path_with_hardware_id() {
    let path = TunerDataCollector::cache_path();
    // Path should end with .apr extension
    assert!(
        path.extension().map_or(false, |ext| ext == "apr"),
        "cache path should have .apr extension: {:?}",
        path
    );
    // Path should contain "trueno" directory
    let path_str = path.to_string_lossy();
    assert!(path_str.contains("trueno"), "cache path should contain 'trueno': {}", path_str);
    // Filename should contain hardware ID prefix
    let filename = path.file_name().unwrap().to_string_lossy();
    assert!(
        filename.starts_with("training_data_"),
        "filename should start with 'training_data_': {}",
        filename
    );
}

#[cfg(feature = "hardware-detect")]
#[test]
fn hardware_id_returns_stable_hex_string() {
    let id1 = TunerDataCollector::hardware_id();
    let id2 = TunerDataCollector::hardware_id();
    // Should be deterministic
    assert_eq!(id1, id2, "hardware_id should be stable across calls");
    // Should be 8 hex characters
    assert_eq!(id1.len(), 8, "hardware_id should be 8 hex chars: {}", id1);
    assert!(id1.chars().all(|c| c.is_ascii_hexdigit()), "hardware_id should be hex: {}", id1);
}
