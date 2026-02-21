use super::*;

#[test]
fn test_model_provenance_format_full() {
    let prov = ModelProvenance::new("TestModel", "v2.0", "Apache-2.0")
        .with_source("https://example.com/model")
        .with_authors("John Doe");

    let formatted = prov.format();
    assert!(formatted.contains("TestModel"));
    assert!(formatted.contains("v2.0"));
    assert!(formatted.contains("Apache-2.0"));
    assert!(formatted.contains("John Doe"));
}

#[test]
fn test_model_provenance_json_with_optionals() {
    let prov = ModelProvenance::new("Model", "1.0", "MIT")
        .with_source("https://source.url")
        .with_authors("Authors");

    let json = prov.to_json();
    assert!(json.contains("\"source\":"));
    assert!(json.contains("\"authors\":"));
}

#[test]
fn test_model_cache_debug() {
    let cache = ModelCache::default();
    assert!(format!("{:?}", cache).contains("ModelCache"));
}

#[test]
fn test_model_source_debug() {
    let source = ModelSource::Local(PathBuf::from("./test"));
    assert!(format!("{:?}", source).contains("Local"));
}

#[test]
fn test_execution_mode_debug() {
    let mode = ExecutionMode::Interactive;
    assert!(format!("{:?}", mode).contains("Interactive"));
}

#[test]
fn test_adaptive_output_debug() {
    let output = AdaptiveOutput::new();
    assert!(format!("{:?}", output).contains("AdaptiveOutput"));
}

#[test]
fn test_recoverable_error_debug() {
    let err = RecoverableError::new("test");
    assert!(format!("{:?}", err).contains("RecoverableError"));
}

#[test]
fn test_performance_metrics_debug() {
    let metrics = PerformanceMetrics::default();
    assert!(format!("{:?}", metrics).contains("PerformanceMetrics"));
}

#[test]
fn test_perf_timer_debug() {
    let timer = PerfTimer::new();
    assert!(format!("{:?}", timer).contains("PerfTimer"));
}

#[test]
fn test_prerequisite_check_debug() {
    let check = PrerequisiteCheck::satisfied("test");
    assert!(format!("{:?}", check).contains("PrerequisiteCheck"));
}

#[test]
fn test_model_cache_clone() {
    let cache1 = ModelCache::default();
    let cache2 = cache1.clone();
    assert_eq!(cache1.cache_dir, cache2.cache_dir);
}

#[test]
fn test_model_source_clone() {
    let source1 = ModelSource::Local(PathBuf::from("./test"));
    let source2 = source1.clone();
    assert!(matches!(source2, ModelSource::Local(_)));
}

#[test]
fn test_execution_mode_eq() {
    assert_eq!(ExecutionMode::Interactive, ExecutionMode::Interactive);
    assert_ne!(ExecutionMode::Interactive, ExecutionMode::Batch);
}

#[test]
fn test_performance_metrics_clone() {
    let metrics1 = PerformanceMetrics {
        backend: "test".to_string(),
        ..Default::default()
    };
    let metrics2 = metrics1.clone();
    assert_eq!(metrics1.backend, metrics2.backend);
}

#[test]
fn test_prerequisite_check_clone() {
    let check1 = PrerequisiteCheck::satisfied("test");
    let check2 = check1.clone();
    assert_eq!(check1.name, check2.name);
}

// ========================================================================
// Phase 1 additional coverage tests
// ========================================================================

#[test]
fn test_model_cache_ensure_dir_error_path() {
    // Use an invalid path that cannot be created to test the error branch
    let cache = ModelCache::new(PathBuf::from("/proc/0/nonexistent/deep/path"));
    let result = cache.ensure_dir();
    assert!(result.is_err());
}

#[test]
fn test_model_provenance_format_no_optionals() {
    let prov = ModelProvenance::new("TestModel", "v1.0", "MIT");
    let formatted = prov.format();
    assert!(formatted.contains("TestModel"));
    assert!(formatted.contains("v1.0"));
    assert!(formatted.contains("MIT"));
    // Should NOT contain Authors or Source lines since they are None
    assert!(!formatted.contains("Authors:"));
    assert!(!formatted.contains("Source:"));
}

#[test]
fn test_model_provenance_json_no_optionals() {
    let prov = ModelProvenance::new("Bare", "0.1", "BSD");
    let json = prov.to_json();
    assert!(json.contains("\"name\":\"Bare\""));
    assert!(json.contains("\"version\":\"0.1\""));
    assert!(json.contains("\"license\":\"BSD\""));
    // No authors or source in JSON
    assert!(!json.contains("\"authors\""));
    assert!(!json.contains("\"source\""));
    // Verify it ends with proper closing brace
    assert!(json.ends_with('}'));
}

#[test]
fn test_model_provenance_format_with_authors_no_source() {
    let prov = ModelProvenance::new("M", "1", "MIT").with_authors("Auth");
    let formatted = prov.format();
    assert!(formatted.contains("Authors: Auth"));
    assert!(!formatted.contains("Source:"));
}

#[test]
fn test_model_provenance_format_with_source_no_authors() {
    let prov = ModelProvenance::new("M", "1", "MIT").with_source("https://x.com");
    let formatted = prov.format();
    assert!(!formatted.contains("Authors:"));
    assert!(formatted.contains("Source: https://x.com"));
}

#[test]
fn test_model_provenance_json_with_authors_no_source() {
    let prov = ModelProvenance::new("M", "1", "MIT").with_authors("Auth");
    let json = prov.to_json();
    assert!(json.contains("\"authors\":\"Auth\""));
    assert!(!json.contains("\"source\""));
}

#[test]
fn test_model_provenance_json_with_source_no_authors() {
    let prov = ModelProvenance::new("M", "1", "MIT").with_source("https://x.com");
    let json = prov.to_json();
    assert!(!json.contains("\"authors\""));
    assert!(json.contains("\"source\":\"https://x.com\""));
}

#[test]
fn test_out_of_memory_large_values() {
    let err = recovery::out_of_memory(8_000_000_000, 4_000_000_000);
    assert!(err.message.contains("8000"));
    assert!(err.message.contains("4000"));
    assert!(err.recovery.is_some());
}

#[test]
fn test_check_command_existing() {
    // "sh" should exist on all unix-like systems
    let check = check_command("sh");
    assert!(check.satisfied);
    assert_eq!(check.name, "sh");
}

#[test]
fn test_adaptive_output_interactive_mode_status() {
    let output = AdaptiveOutput::new().with_mode(ExecutionMode::Interactive);
    // Should print status in interactive mode (no panic)
    output.status("Loading model...");
    output.progress(1, 10, "processing");
}

#[test]
fn test_adaptive_output_interactive_json_suppresses() {
    // JSON mode should suppress status even in interactive mode
    let output = AdaptiveOutput::new()
        .with_mode(ExecutionMode::Interactive)
        .with_json();
    output.status("This should not print");
    output.progress(5, 10, "this too");
}

#[test]
fn test_perf_timer_print_verbose_no_checkpoints() {
    let timer = PerfTimer::new();
    // Should not panic with empty checkpoints
    timer.print_verbose();
}

#[test]
fn test_perf_timer_print_verbose_multiple() {
    let mut timer = PerfTimer::new();
    std::thread::sleep(Duration::from_millis(2));
    timer.checkpoint("step1");
    std::thread::sleep(Duration::from_millis(2));
    timer.checkpoint("step2");
    std::thread::sleep(Duration::from_millis(2));
    timer.checkpoint("step3");
    // Verify all checkpoints print without panic
    timer.print_verbose();
}

#[test]
fn test_model_source_hf_clone_variants() {
    let source = ModelSource::HuggingFace {
        repo_id: "org/repo".to_string(),
        filename: "model.safetensors".to_string(),
    };
    let cloned = source.clone();
    if let ModelSource::HuggingFace { repo_id, filename } = cloned {
        assert_eq!(repo_id, "org/repo");
        assert_eq!(filename, "model.safetensors");
    } else {
        panic!("Expected HuggingFace variant after clone");
    }
}

#[test]
fn test_model_source_url_clone() {
    let source = ModelSource::Url("https://example.com/model.bin".to_string());
    let cloned = source.clone();
    if let ModelSource::Url(url) = cloned {
        assert_eq!(url, "https://example.com/model.bin");
    } else {
        panic!("Expected Url variant after clone");
    }
}

#[test]
fn test_prerequisite_check_with_version() {
    let mut check = PrerequisiteCheck::satisfied("rustc");
    check.version = Some("1.75.0".to_string());
    assert!(check.satisfied);
    assert_eq!(check.version, Some("1.75.0".to_string()));
}

#[test]
fn test_recoverable_error_format_with_recovery() {
    let err = RecoverableError::new("disk full").with_recovery("Free up disk space");
    let formatted = err.format();
    assert!(formatted.contains("disk full"));
    assert!(formatted.contains("Suggested fix: Free up disk space"));
}

#[test]
fn test_performance_metrics_high_throughput() {
    let metrics = PerformanceMetrics {
        load_time: Duration::from_millis(100),
        time_to_first_token: Duration::from_millis(10),
        tokens_generated: 1000,
        generation_time: Duration::from_secs(1),
        peak_memory: 100_000,
        backend: "AVX-512".to_string(),
    };
    assert!((metrics.tokens_per_second() - 1000.0).abs() < 1.0);
    let json = metrics.to_json();
    assert!(json.contains("\"backend\":\"AVX-512\""));
}

#[test]
fn test_checksum_mismatch_message() {
    let err = recovery::checksum_mismatch("aabbccdd", "11223344");
    assert!(err.message.contains("aabbccdd"));
    assert!(err.message.contains("11223344"));
    assert!(err.auto_recoverable);
    assert!(err.recovery.is_some());
}
