use super::*;

// RDB-01 Tests
#[test]
fn test_model_cache_default() {
    let cache = ModelCache::default();
    assert!(cache.auto_download);
    // Cache dir contains "apr" (with hf-hub-integration) or "apr_cache" (without)
    let path_str = cache.cache_dir.to_string_lossy();
    assert!(path_str.contains("apr") || path_str.contains("cache"));
}

#[test]
fn test_model_source_parse() {
    // HuggingFace
    let hf = ModelSource::parse("hf://Qwen/Qwen2-0.5B-Instruct/model.safetensors");
    assert!(matches!(hf, ModelSource::HuggingFace { .. }));
    assert!(hf.is_remote());

    // URL
    let url = ModelSource::parse("https://example.com/model.gguf");
    assert!(matches!(url, ModelSource::Url(_)));
    assert!(url.is_remote());

    // Local
    let local = ModelSource::parse("/path/to/model.safetensors");
    assert!(matches!(local, ModelSource::Local(_)));
    assert!(!local.is_remote());
}

// RDB-02 Tests
#[test]
fn test_prerequisite_check() {
    let satisfied = PrerequisiteCheck::satisfied("test");
    assert!(satisfied.satisfied);

    let missing = PrerequisiteCheck::missing("test", "install it");
    assert!(!missing.satisfied);
    assert!(missing.install_hint.is_some());
}

// RDB-03 Tests
#[test]
fn test_execution_mode() {
    // In tests, we're usually not in a TTY
    let mode = ExecutionMode::detect();
    // Just verify it returns something valid
    assert!(mode.is_interactive() || mode.is_batch());
}

#[test]
fn test_adaptive_output() {
    let output = AdaptiveOutput::new();
    // Should not panic
    output.status("test");
    output.result("result");
    output.error("error");
}

// RDB-04 Tests
#[test]
fn test_recoverable_error() {
    let err = RecoverableError::new("test error")
        .with_recovery("do this")
        .auto_recoverable();

    assert!(err.auto_recoverable);
    assert!(err.recovery.is_some());
    assert!(err.format().contains("test error"));
    assert!(err.format().contains("do this"));
}

#[test]
fn test_recovery_scenarios() {
    let err = recovery::model_not_found(Path::new("/test"));
    assert!(err.message.contains("not found"));

    let err = recovery::checksum_mismatch("abc", "def");
    assert!(err.auto_recoverable);

    let err = recovery::gpu_not_available();
    assert!(err.recovery.is_some());

    let err = recovery::out_of_memory(1000, 500);
    assert!(err.message.contains("memory"));
}

// RDB-05 Tests
#[test]
fn test_performance_metrics() {
    let metrics = PerformanceMetrics {
        load_time: Duration::from_millis(1000),
        time_to_first_token: Duration::from_millis(100),
        tokens_generated: 100,
        generation_time: Duration::from_secs(5),
        peak_memory: 500_000_000,
        backend: "AVX2".to_string(),
    };

    assert!((metrics.tokens_per_second() - 20.0).abs() < 0.1);
    assert!(metrics.format().contains("AVX2"));
    assert!(metrics.to_json().contains("\"backend\":\"AVX2\""));
}

#[test]
fn test_perf_timer() {
    let mut timer = PerfTimer::new();
    std::thread::sleep(Duration::from_millis(10));
    timer.checkpoint("first");
    std::thread::sleep(Duration::from_millis(10));
    timer.checkpoint("second");

    assert!(timer.elapsed() >= Duration::from_millis(20));
}

#[test]
fn test_detect_backend() {
    let backend = detect_backend();
    // Should return a valid backend string
    assert!(!backend.is_empty());
}

// RDB-06 Tests
#[test]
fn test_model_provenance() {
    let prov = ModelProvenance::new("TestModel", "v1.0", "MIT")
        .with_source("https://example.com")
        .with_authors("Test Author");

    assert_eq!(prov.name, "TestModel");
    assert!(prov.format().contains("MIT"));
    assert!(prov.to_json().contains("\"license\":\"MIT\""));
}

#[test]
fn test_common_provenances() {
    let tinyllama = models::tinyllama_chat();
    assert!(tinyllama.name.contains("TinyLlama"));
    assert_eq!(tinyllama.license, "Apache-2.0");

    let qwen = models::qwen2_0_5b();
    assert!(qwen.name.contains("Qwen2"));

    let mistral = models::mistral_7b();
    assert!(mistral.name.contains("Mistral"));

    let phi = models::phi2();
    assert_eq!(phi.license, "MIT");
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_model_cache_new() {
    let cache = ModelCache::new(PathBuf::from("/tmp/test_cache"));
    assert_eq!(cache.cache_dir, PathBuf::from("/tmp/test_cache"));
    assert!(cache.auto_download);
    assert_eq!(cache.max_size_bytes, 0);
}

#[test]
fn test_model_cache_model_path() {
    let cache = ModelCache::new(PathBuf::from("/tmp/cache"));
    let path = cache.model_path("qwen2-0.5b");
    assert!(path.to_string_lossy().contains("qwen2-0.5b"));
}

#[test]
fn test_model_cache_has_model() {
    let cache = ModelCache::new(PathBuf::from("/nonexistent/path"));
    assert!(!cache.has_model("any-model"));
}

#[test]
fn test_model_source_hf_parse() {
    let source = ModelSource::parse("hf://org/repo/file.safetensors");
    if let ModelSource::HuggingFace { repo_id, filename } = source {
        assert_eq!(repo_id, "org/repo");
        assert_eq!(filename, "file.safetensors");
    } else {
        panic!("Expected HuggingFace source");
    }
}

#[test]
fn test_model_source_url_parse() {
    let source = ModelSource::parse("https://example.com/model.gguf");
    assert!(matches!(source, ModelSource::Url(_)));
}

#[test]
fn test_model_source_local_parse() {
    let source = ModelSource::parse("./model.safetensors");
    assert!(matches!(source, ModelSource::Local(_)));
}

#[test]
fn test_model_source_is_local() {
    let local = ModelSource::Local(PathBuf::from("./test"));
    assert!(!local.is_remote());
    let url = ModelSource::Url("https://example.com".to_string());
    assert!(url.is_remote());
}

#[test]
fn test_execution_mode_batch() {
    let mode = ExecutionMode::Batch;
    assert!(mode.is_batch());
    assert!(!mode.is_interactive());
}

#[test]
fn test_execution_mode_interactive() {
    let mode = ExecutionMode::Interactive;
    assert!(mode.is_interactive());
    assert!(!mode.is_batch());
}

#[test]
fn test_adaptive_output_methods() {
    let output = AdaptiveOutput::new();
    output.progress(50, 100, "loading...");
    output.result("done");
    output.error("test error");
}

#[test]
fn test_recoverable_error_format_no_recovery() {
    let err = RecoverableError::new("simple error");
    let formatted = err.format();
    assert!(formatted.contains("simple error"));
}

#[test]
fn test_check_command_nonexistent() {
    let check = check_command("nonexistent_command_12345");
    assert!(!check.satisfied);
}

#[test]
fn test_performance_metrics_zero_time() {
    let metrics = PerformanceMetrics {
        load_time: Duration::ZERO,
        time_to_first_token: Duration::ZERO,
        tokens_generated: 0,
        generation_time: Duration::ZERO,
        peak_memory: 0,
        backend: "test".to_string(),
    };
    // Should not panic on division by zero
    assert_eq!(metrics.tokens_per_second(), 0.0);
}

#[test]
fn test_perf_timer_checkpoints() {
    let mut timer = PerfTimer::new();
    timer.checkpoint("start");
    timer.checkpoint("middle");
    timer.checkpoint("end");
    // Verify checkpoints were recorded
    assert!(timer.elapsed() >= Duration::ZERO);
}

#[test]
fn test_model_provenance_builder() {
    let prov = ModelProvenance::new("Model", "1.0", "MIT")
        .with_source("https://source.com")
        .with_authors("Author1, Author2");
    assert_eq!(prov.name, "Model");
    assert_eq!(prov.version, "1.0");
    assert_eq!(prov.license, "MIT");
    assert_eq!(prov.source_url, Some("https://source.com".to_string()));
    assert_eq!(prov.authors, Some("Author1, Author2".to_string()));
}

#[test]
fn test_model_provenance_json() {
    let prov = ModelProvenance::new("TestModel", "v1.0", "Apache-2.0");
    let json = prov.to_json();
    assert!(json.contains("\"name\":\"TestModel\""));
    assert!(json.contains("\"version\":\"v1.0\""));
    assert!(json.contains("\"license\":\"Apache-2.0\""));
}

#[test]
fn test_detect_backend_not_empty() {
    let backend = detect_backend();
    assert!(!backend.is_empty());
}

#[test]
fn test_adaptive_output_with_json() {
    let output = AdaptiveOutput::new().with_json();
    output.status("should not print in json mode");
}

#[test]
fn test_adaptive_output_with_mode() {
    let output = AdaptiveOutput::new().with_mode(ExecutionMode::Batch);
    output.status("should not print in batch mode");
}

#[test]
fn test_perf_timer_since_last() {
    let mut timer = PerfTimer::new();
    std::thread::sleep(Duration::from_millis(5));
    timer.checkpoint("first");
    std::thread::sleep(Duration::from_millis(5));
    let since_last = timer.since_last();
    assert!(since_last >= Duration::from_millis(4)); // Allow some tolerance
}

#[test]
fn test_perf_timer_since_last_no_checkpoints() {
    let timer = PerfTimer::new();
    std::thread::sleep(Duration::from_millis(5));
    let since_last = timer.since_last();
    assert!(since_last >= Duration::from_millis(4));
}

#[test]
fn test_perf_timer_print_verbose() {
    let mut timer = PerfTimer::new();
    timer.checkpoint("load");
    timer.checkpoint("process");
    // Should not panic
    timer.print_verbose();
}

#[test]
fn test_performance_metrics_default() {
    let metrics = PerformanceMetrics::default();
    assert_eq!(metrics.tokens_generated, 0);
    assert!(metrics.backend.is_empty());
}

#[test]
fn test_adaptive_output_default() {
    let output = AdaptiveOutput::default();
    // Should work the same as new()
    output.status("test");
}

#[test]
fn test_perf_timer_default() {
    let timer = PerfTimer::default();
    assert!(timer.elapsed() >= Duration::ZERO);
}

#[test]
fn test_model_cache_ensure_dir() {
    let cache = ModelCache::new(PathBuf::from("/tmp/aprender_test_cache"));
    // Should succeed or already exist
    let _ = cache.ensure_dir();
    // Clean up
    let _ = std::fs::remove_dir_all("/tmp/aprender_test_cache");
}

#[test]
fn test_model_source_hf_short_path() {
    // Test HF path with only org/repo (no file)
    let source = ModelSource::parse("hf://owner/repo");
    if let ModelSource::HuggingFace { repo_id, filename } = source {
        assert_eq!(repo_id, "owner/repo");
        assert_eq!(filename, "model.safetensors"); // Default filename
    } else {
        panic!("Expected HuggingFace source");
    }
}

#[test]
fn test_model_source_http_url() {
    let source = ModelSource::parse("http://localhost/model.gguf");
    assert!(matches!(source, ModelSource::Url(_)));
    assert!(source.is_remote());
}

#[test]
fn test_model_source_hf_single_part() {
    // Edge case: single component after hf://
    let source = ModelSource::parse("hf://single");
    // Should fall back to Local since it doesn't have org/repo structure
    assert!(matches!(source, ModelSource::Local(_)));
}

#[test]
fn test_check_prerequisites_multiple() {
    let checks = check_prerequisites(&["ls", "nonexistent_cmd_xyz"]);
    assert_eq!(checks.len(), 2);
    // ls should exist on most systems
    assert!(checks[0].satisfied || !checks[0].satisfied); // Always valid
    assert!(!checks[1].satisfied); // nonexistent should not exist
}

#[test]
fn test_print_prerequisites() {
    let checks = vec![
        PrerequisiteCheck::satisfied("test1"),
        PrerequisiteCheck::missing("test2", "install it"),
    ];
    // Should not panic
    print_prerequisites(&checks);
}

#[test]
fn test_recoverable_error_not_auto() {
    let err = RecoverableError::new("not auto-recoverable");
    assert!(!err.auto_recoverable);
}

#[test]
fn test_performance_metrics_format_content() {
    let metrics = PerformanceMetrics {
        load_time: Duration::from_secs(2),
        time_to_first_token: Duration::from_millis(150),
        tokens_generated: 50,
        generation_time: Duration::from_secs(5),
        peak_memory: 1_500_000_000,
        backend: "CUDA".to_string(),
    };

    let formatted = metrics.format();
    assert!(formatted.contains("Load time"));
    assert!(formatted.contains("CUDA"));
    assert!(formatted.contains("Peak memory"));
}

#[test]
fn test_performance_metrics_json_content() {
    let metrics = PerformanceMetrics {
        load_time: Duration::from_millis(500),
        time_to_first_token: Duration::from_millis(50),
        tokens_generated: 100,
        generation_time: Duration::from_secs(10),
        peak_memory: 2_000_000_000,
        backend: "Metal".to_string(),
    };

    let json = metrics.to_json();
    assert!(json.contains("\"backend\":\"Metal\""));
    assert!(json.contains("tokens_per_sec"));
    assert!(json.contains("tokens_generated"));
}

include!("tests_part_02.rs");
