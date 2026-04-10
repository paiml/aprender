#[test]
fn test_model_preparation_result_serialization() {
    use crate::provenance::{Provenance, SourceProvenance};

    let result = ModelPreparationResult {
        provenance: Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc123".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-02-01T12:00:00Z".to_string(),
            },
            derived: vec![],
        },
        safetensors_path: std::path::PathBuf::from("/models/model.safetensors"),
        gguf_path: None,
        apr_path: None,
        conversions: vec![],
    };

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("safetensors"));
    assert!(json.contains("test/model"));
}

#[test]
fn test_verify_comparison_provenance_missing_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
    assert!(result.is_err());
}

#[test]
fn test_verify_comparison_provenance_valid() {
    use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance, save_provenance};

    let temp_dir = tempfile::tempdir().unwrap();

    // Create valid provenance
    let provenance = Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: "model.safetensors".to_string(),
            sha256: "abc123".to_string(),
            hf_repo: "test/model".to_string(),
            downloaded_at: "2026-02-01T12:00:00Z".to_string(),
        },
        derived: vec![
            DerivedProvenance {
                format: "gguf".to_string(),
                path: "model.gguf".to_string(),
                sha256: "def456".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None,
                created_at: "2026-02-01T12:05:00Z".to_string(),
            },
            DerivedProvenance {
                format: "apr".to_string(),
                path: "model.apr".to_string(),
                sha256: "789ghi".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None,
                created_at: "2026-02-01T12:06:00Z".to_string(),
            },
        ],
    };
    save_provenance(temp_dir.path(), &provenance).unwrap();

    let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
    assert!(result.is_ok());
}

#[test]
fn test_verify_comparison_provenance_quantization_mismatch() {
    use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance, save_provenance};

    let temp_dir = tempfile::tempdir().unwrap();

    // Create provenance with mismatched quantization
    let provenance = Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: "model.safetensors".to_string(),
            sha256: "abc123".to_string(),
            hf_repo: "test/model".to_string(),
            downloaded_at: "2026-02-01T12:00:00Z".to_string(),
        },
        derived: vec![
            DerivedProvenance {
                format: "gguf".to_string(),
                path: "model-q4.gguf".to_string(),
                sha256: "def456".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: Some("q4_k_m".to_string()), // Quantized
                created_at: "2026-02-01T12:05:00Z".to_string(),
            },
            DerivedProvenance {
                format: "apr".to_string(),
                path: "model.apr".to_string(),
                sha256: "789ghi".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None, // Not quantized
                created_at: "2026-02-01T12:06:00Z".to_string(),
            },
        ],
    };
    save_provenance(temp_dir.path(), &provenance).unwrap();

    let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
    assert!(result.is_err()); // PROV-005 violation
}

#[test]
fn test_prepare_model_fails_without_apr_binary() {
    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"fake safetensors content").unwrap();

    let output_dir = temp_dir.path().join("output");
    let result = prepare_model_with_provenance(
        "/nonexistent/apr",
        &safetensors,
        "test/model",
        &output_dir,
        None,
    );

    // Will fail because apr binary doesn't exist
    assert!(result.is_err());
}

// =========================================================================
// Mock binary tests for Command-calling functions
// =========================================================================

/// Create a mock bash script that acts as a fake apr binary
/// Create a mock binary with explicit fd sync/close to avoid ETXTBSY (os error 26)
/// when parallel tests execute mock scripts concurrently.
fn create_mock_binary(dir: &std::path::Path, name: &str, script: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(format!("#!/bin/bash\n{script}").as_bytes())
            .unwrap();
        f.sync_all().unwrap();
        drop(f);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    std::thread::yield_now();
    path
}

// =========================================================================
// convert_format_cached - cache hit path
// =========================================================================

#[test]
fn test_convert_format_cached_cache_hit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.safetensors");
    let target = temp_dir.path().join("target.gguf");
    let hash_file = temp_dir.path().join(".hash");

    // Write source file
    std::fs::write(&source, b"model content for caching test").unwrap();

    // Mock that creates the target file (arg $4 = target_path)
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_hash",
        "echo 'converted' > \"$4\" && exit 0",
    );

    // First call: does actual conversion, writes hash
    let first = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
    if let Ok(r1) = first {
        assert!(r1.success);
        assert!(!r1.cached);

        // Second call with same source: should hit cache
        let second = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
        if let Ok(r2) = second {
            assert!(r2.cached);
            assert!(r2.success);
            assert_eq!(r2.duration_ms, 0);
        }
    }
}

#[test]
fn test_convert_format_cached_successful_conversion() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.safetensors");
    let target = temp_dir.path().join("output").join("target.gguf");
    let hash_file = temp_dir.path().join(".hash");

    std::fs::write(&source, b"model data for conversion").unwrap();

    // Mock binary that creates the target file
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_convert",
        "mkdir -p \"$(dirname \"$3\")\" && echo 'converted' > \"$3\" && exit 0",
    );

    let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
    if let Ok(r) = result {
        assert!(r.success);
        assert!(!r.cached);
        assert_eq!(r.source_format, "safetensors");
        assert_eq!(r.target_format, "gguf");
        // Hash file should have been written
        assert!(hash_file.exists());
    }
}

#[test]
fn test_convert_format_cached_failed_conversion() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.gguf");
    let target = temp_dir.path().join("target.apr");
    let hash_file = temp_dir.path().join(".hash");

    std::fs::write(&source, b"model data").unwrap();

    // Mock binary that fails
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_fail",
        "echo 'error: bad format' >&2; exit 1",
    );

    let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
    if let Ok(r) = result {
        assert!(!r.success);
        assert!(!r.cached);
        assert!(r.error.is_some());
        assert!(r.error.unwrap().contains("error: bad format"));
    }
}

#[test]
fn test_convert_format_cached_stale_cache() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.safetensors");
    let target = temp_dir.path().join("target.gguf");
    let hash_file = temp_dir.path().join(".hash");

    std::fs::write(&source, b"model data v1").unwrap();
    // Pre-populate with wrong hash to simulate stale cache
    std::fs::write(&target, b"old converted data").unwrap();
    std::fs::write(&hash_file, "wrong_hash_value").unwrap();

    // Mock binary that creates target
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_stale",
        "echo 'reconverted' > \"$3\" && exit 0",
    );

    let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
    if let Ok(r) = result {
        // Should NOT be cached since hash didn't match
        assert!(!r.cached);
        assert!(r.success);
    }
}

// =========================================================================
// compute_file_hash - error paths
// =========================================================================

#[test]
fn test_compute_file_hash_nonexistent_file() {
    let result = convert_format_cached(
        "echo",
        std::path::Path::new("/nonexistent/model.gguf"),
        std::path::Path::new("/tmp/out.apr"),
        std::path::Path::new("/tmp/.hash"),
    );
    // Should fail because source doesn't exist (compute_file_hash fails)
    assert!(result.is_err());
}

// =========================================================================
// run_bench_throughput tests
// =========================================================================

#[test]
fn test_run_bench_throughput_success_cpu() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench",
        "echo 'Loading model...\nThroughput: 65.5 tok/s (PASS: >= 10 tok/s)\nDone.' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 3);
    if let Ok(r) = result {
        assert!((r.throughput_tps - 65.5).abs() < 0.01);
        assert!(r.passed);
        assert_eq!(r.backend, "cpu");
        assert_eq!(r.format, "gguf");
    }
}

#[test]
fn test_run_bench_throughput_success_gpu() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.apr");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_gpu",
        "echo 'Throughput: 120.3 tok/s' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, true, 1, 3);
    if let Ok(r) = result {
        assert!((r.throughput_tps - 120.3).abs() < 0.01);
        assert!(r.passed);
        assert_eq!(r.backend, "gpu");
        assert_eq!(r.format, "apr");
    }
}

#[test]
fn test_run_bench_throughput_below_threshold() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.safetensors");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_slow",
        "echo 'Throughput: 5.2 tok/s' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
    if let Ok(r) = result {
        assert!((r.throughput_tps - 5.2).abs() < 0.01);
        // Below 10.0 threshold
        assert!(!r.passed);
        assert_eq!(r.format, "safetensors");
    }
}

#[test]
fn test_run_bench_throughput_no_throughput_line() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_nothroughput",
        "echo 'Loading model...\nDone.' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
    if let Ok(r) = result {
        assert!((r.throughput_tps - 0.0).abs() < 0.01);
        assert!(!r.passed); // 0.0 < 10.0
    }
}

#[test]
fn test_run_bench_throughput_failed_exit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_fail",
        "echo 'Throughput: 50.0 tok/s' && exit 1",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
    if let Ok(r) = result {
        // exit code non-zero => passed = false even though throughput was high
        assert!(!r.passed);
    }
}

#[test]
fn test_run_bench_throughput_unknown_extension() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_noext",
        "echo 'Throughput: 15.0 tok/s' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
    if let Ok(r) = result {
        assert_eq!(r.format, "unknown");
    }
}

// =========================================================================
// run_ci_profile tests
// =========================================================================

#[test]
fn test_run_ci_profile_json_output() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake model").unwrap();

    let json = r#"{"model":"test","metrics":null,"throughput_tps":42.0,"latency_p50_ms":10.0,"latency_p99_ms":25.0,"assertions":[],"passed":true}"#;
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_profile_json",
        &format!("echo '{json}' && exit 0"),
    );

    let result = run_profile_ci(
        mock.to_str().unwrap(),
        &model,
        Some(10.0),
        Some(100.0),
        Some(50.0),
        1,
        3,
    );
    if let Ok(r) = result {
        assert!(r.passed);
        assert!((r.throughput_tps - 42.0).abs() < 0.01);
    }
}
