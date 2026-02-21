
// ========================================================================
// NEW: pipeline non-inference paths (pipeline.rs)
// ========================================================================

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_gguf_inference_no_inference_needs_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    // Create a fake GGUF file that's large enough
    let gguf_path = temp_dir.path().join(config.tier.gguf_filename());
    let fake_data = vec![0u8; 2_000_000]; // 2MB
    std::fs::write(&gguf_path, &fake_data).unwrap();

    let result = super::pipeline::run_gguf_inference(&config);
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_gguf_inference_no_inference_too_small() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    // Create a GGUF file that's too small
    let gguf_path = temp_dir.path().join(config.tier.gguf_filename());
    std::fs::write(&gguf_path, b"tiny").unwrap();

    let result = super::pipeline::run_gguf_inference(&config);
    assert!(result.is_err());
}

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_gguf_inference_no_inference_missing_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    // No file exists
    let result = super::pipeline::run_gguf_inference(&config);
    assert!(result.is_err());
}

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_convert_no_inference() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    // Create a fake GGUF file for conversion
    let gguf_path = temp_dir.path().join(config.tier.gguf_filename());
    let fake_data = vec![0u8; 1_000];
    std::fs::write(&gguf_path, &fake_data).unwrap();

    let result = super::pipeline::run_convert(&config);
    assert!(result.is_ok());
    assert!(result.unwrap());

    // Should have created a placeholder APR file
    let apr_basename = config.tier.gguf_filename().replace(".gguf", ".apr");
    let apr_path = temp_dir.path().join(apr_basename);
    assert!(apr_path.exists());
    let content = std::fs::read_to_string(&apr_path).unwrap();
    assert!(content.contains("APR-PLACEHOLDER-V2"));
}

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_apr_inference_no_inference_missing_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    // No APR file - should return Ok(false)
    let result = super::pipeline::run_apr_inference(&config);
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_apr_inference_no_inference_with_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    // Create a placeholder APR file
    let apr_basename = config.tier.gguf_filename().replace(".gguf", ".apr");
    let apr_path = temp_dir.path().join(&apr_basename);
    std::fs::write(&apr_path, "APR-PLACEHOLDER").unwrap();

    let result = super::pipeline::run_apr_inference(&config);
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_apr_inference_no_inference_with_zram_flag() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        zram: true, // Tests the ZRAM flag printing path
        ..Default::default()
    };

    let apr_basename = config.tier.gguf_filename().replace(".gguf", ".apr");
    let apr_path = temp_dir.path().join(&apr_basename);
    std::fs::write(&apr_path, "APR-PLACEHOLDER").unwrap();

    let result = super::pipeline::run_apr_inference(&config);
    assert!(result.is_ok());
}

// ========================================================================
// NEW: pipeline run_import (pipeline.rs lines 11-70)
// ========================================================================

#[test]
fn test_run_import_model_already_exists() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    // Create the model file so import returns early
    let gguf_path = temp_dir.path().join(config.tier.gguf_filename());
    std::fs::write(&gguf_path, b"existing model data").unwrap();

    let result = super::pipeline::run_import(&config);
    assert!(result.is_ok());
    assert!(result.unwrap()); // Should return true for already-exists
}

// ========================================================================
// NEW: validate_falsification with gguf_inference failure
// ========================================================================

#[test]
fn test_falsification_gguf_inference_failure_full_run() {
    let results = ShowcaseResults {
        import: true,
        gguf_inference: false, // Failed
        convert: true,
        apr_inference: true,
        benchmark: Some(BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: None,
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(120.0),
            ollama_ttft_ms: None,
            speedup_vs_llama: Some(25.7),
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.5,
            runs: 30,
        }),
        ..Default::default()
    };
    assert!(validate_falsification(&results, &full_run_config()).is_err());
}

#[test]
fn test_falsification_apr_inference_failure_full_run() {
    let results = ShowcaseResults {
        import: true,
        gguf_inference: true,
        convert: true,
        apr_inference: false, // Failed
        benchmark: Some(BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: None,
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(120.0),
            ollama_ttft_ms: None,
            speedup_vs_llama: Some(25.7),
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.5,
            runs: 30,
        }),
        ..Default::default()
    };
    assert!(validate_falsification(&results, &full_run_config()).is_err());
}

// ========================================================================
// NEW: ModelTier specific size/filename tests
// ========================================================================

#[test]
fn test_model_tier_tiny_specific_values() {
    let tier = ModelTier::Tiny;
    assert!(tier.size_gb() < 1.0);
    assert!(tier.gguf_filename().contains("0.5b"));
    assert!(tier.model_path().contains("0.5B"));
}

#[test]
fn test_model_tier_large_specific_values() {
    let tier = ModelTier::Large;
    assert!(tier.size_gb() > 10.0);
    assert!(tier.gguf_filename().contains("32B"));
    assert!(tier.model_path().contains("32B"));
}

// ========================================================================
// NEW: ShowcaseConfig model field matches tier
// ========================================================================

#[test]
fn test_showcase_config_default_model_matches_small_tier() {
    let config = ShowcaseConfig::default();
    assert_eq!(config.model, ModelTier::Small.model_path());
}

#[test]
fn test_showcase_config_with_tier_model_field_matches() {
    for tier in &[
        ModelTier::Tiny,
        ModelTier::Small,
        ModelTier::Medium,
        ModelTier::Large,
    ] {
        let config = ShowcaseConfig::with_tier(*tier);
        assert_eq!(config.model, tier.model_path());
    }
}

// ========================================================================
// NEW: BenchmarkComparison serde round-trip with all None optionals
// ========================================================================

#[test]
fn test_benchmark_comparison_serde_all_none() {
    let bench = BenchmarkComparison {
        apr_tps: 50.0,
        llama_cpp_tps: None,
        ollama_tps: None,
        apr_ttft_ms: 60.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
        apr_tps_stddev: 2.0,
        runs: 30,
    };
    let json = serde_json::to_string(&bench).unwrap();
    let parsed: BenchmarkComparison = serde_json::from_str(&json).unwrap();
    assert!((parsed.apr_tps - 50.0).abs() < 0.001);
    assert!(parsed.llama_cpp_tps.is_none());
    assert!(parsed.ollama_tps.is_none());
    assert!(parsed.speedup_vs_llama.is_none());
    assert!(parsed.speedup_vs_ollama.is_none());
}

#[test]
fn test_benchmark_comparison_serde_pretty_print() {
    let bench = BenchmarkComparison {
        apr_tps: 44.0,
        llama_cpp_tps: Some(35.0),
        ollama_tps: None,
        apr_ttft_ms: 78.0,
        llama_cpp_ttft_ms: Some(120.0),
        ollama_ttft_ms: None,
        speedup_vs_llama: Some(25.7),
        speedup_vs_ollama: None,
        apr_tps_stddev: 1.5,
        runs: 30,
    };
    let pretty = serde_json::to_string_pretty(&bench).unwrap();
    assert!(pretty.contains('\n'));
    assert!(pretty.contains("apr_tps"));
    // Verify round-trip with pretty-printed JSON
    let parsed: BenchmarkComparison = serde_json::from_str(&pretty).unwrap();
    assert_eq!(parsed.runs, 30);
}

// ========================================================================
// NEW: run_convert with existing APR file (early return)
// ========================================================================

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_convert_apr_already_exists() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    // Create both GGUF and APR files
    let gguf_path = temp_dir.path().join(config.tier.gguf_filename());
    std::fs::write(&gguf_path, b"fake gguf").unwrap();

    let apr_basename = config.tier.gguf_filename().replace(".gguf", ".apr");
    let apr_path = temp_dir.path().join(&apr_basename);
    std::fs::write(&apr_path, b"existing apr").unwrap();

    // The non-inference convert path still writes a placeholder.
    // But the inference path would short-circuit. Let's verify the function runs:
    let result = super::pipeline::run_convert(&config);
    assert!(result.is_ok());
}
