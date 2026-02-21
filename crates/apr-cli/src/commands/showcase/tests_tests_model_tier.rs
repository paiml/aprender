
#[test]
fn test_model_tier_model_path_contains_qwen() {
    assert!(ModelTier::Tiny.model_path().contains("Qwen"));
    assert!(ModelTier::Small.model_path().contains("Qwen"));
    assert!(ModelTier::Medium.model_path().contains("Qwen"));
    assert!(ModelTier::Large.model_path().contains("Qwen"));
}

#[test]
fn test_model_tier_model_path_distinct() {
    let paths: Vec<&str> = vec![
        ModelTier::Tiny.model_path(),
        ModelTier::Small.model_path(),
        ModelTier::Medium.model_path(),
        ModelTier::Large.model_path(),
    ];
    // All paths should be unique
    for i in 0..paths.len() {
        for j in (i + 1)..paths.len() {
            assert_ne!(paths[i], paths[j], "Model paths should be unique");
        }
    }
}

#[test]
fn test_model_tier_gguf_filename_has_extension() {
    assert!(ModelTier::Tiny.gguf_filename().ends_with(".gguf"));
    assert!(ModelTier::Small.gguf_filename().ends_with(".gguf"));
    assert!(ModelTier::Medium.gguf_filename().ends_with(".gguf"));
    assert!(ModelTier::Large.gguf_filename().ends_with(".gguf"));
}

#[test]
fn test_model_tier_gguf_filename_distinct() {
    let filenames: Vec<&str> = vec![
        ModelTier::Tiny.gguf_filename(),
        ModelTier::Small.gguf_filename(),
        ModelTier::Medium.gguf_filename(),
        ModelTier::Large.gguf_filename(),
    ];
    for i in 0..filenames.len() {
        for j in (i + 1)..filenames.len() {
            assert_ne!(
                filenames[i], filenames[j],
                "GGUF filenames should be unique"
            );
        }
    }
}

#[test]
fn test_model_tier_gguf_filename_contains_q4_k() {
    // All filenames should contain the quantization level
    assert!(ModelTier::Tiny
        .gguf_filename()
        .to_lowercase()
        .contains("q4_k"));
    assert!(ModelTier::Small
        .gguf_filename()
        .to_lowercase()
        .contains("q4_k"));
    assert!(
        ModelTier::Medium.gguf_filename().contains("Q4_K")
            || ModelTier::Medium.gguf_filename().contains("q4_k")
    );
    assert!(
        ModelTier::Large.gguf_filename().contains("Q4_K")
            || ModelTier::Large.gguf_filename().contains("q4_k")
    );
}

#[test]
fn test_model_tier_size_gb_monotonically_increasing() {
    assert!(ModelTier::Tiny.size_gb() < ModelTier::Small.size_gb());
    assert!(ModelTier::Small.size_gb() < ModelTier::Medium.size_gb());
    assert!(ModelTier::Medium.size_gb() < ModelTier::Large.size_gb());
}

#[test]
fn test_model_tier_size_gb_all_positive() {
    assert!(ModelTier::Tiny.size_gb() > 0.0);
    assert!(ModelTier::Small.size_gb() > 0.0);
    assert!(ModelTier::Medium.size_gb() > 0.0);
    assert!(ModelTier::Large.size_gb() > 0.0);
}

#[test]
fn test_model_tier_params_values() {
    assert_eq!(ModelTier::Tiny.params(), "0.5B");
    assert_eq!(ModelTier::Small.params(), "1.5B");
    assert_eq!(ModelTier::Medium.params(), "7B");
    assert_eq!(ModelTier::Large.params(), "32B");
}

#[test]
fn test_model_tier_equality() {
    assert_eq!(ModelTier::Tiny, ModelTier::Tiny);
    assert_ne!(ModelTier::Tiny, ModelTier::Small);
    assert_ne!(ModelTier::Small, ModelTier::Medium);
    assert_ne!(ModelTier::Medium, ModelTier::Large);
}

#[test]
fn test_model_tier_copy_clone() {
    let tier = ModelTier::Medium;
    let copied = tier;
    let cloned = tier.clone();
    assert_eq!(copied, ModelTier::Medium);
    assert_eq!(cloned, ModelTier::Medium);
}

#[test]
fn test_model_tier_debug() {
    let debug = format!("{:?}", ModelTier::Large);
    assert_eq!(debug, "Large");
}

// ========================================================================
// ShowcaseConfig Tests
// ========================================================================

#[test]
fn test_showcase_config_default_tier() {
    let config = ShowcaseConfig::default();
    assert_eq!(config.tier, ModelTier::Small);
}

#[test]
fn test_showcase_config_default_model_dir() {
    let config = ShowcaseConfig::default();
    assert_eq!(config.model_dir, std::path::PathBuf::from("./models"));
}

#[test]
fn test_showcase_config_default_auto_verify_false() {
    let config = ShowcaseConfig::default();
    assert!(!config.auto_verify);
}

#[test]
fn test_showcase_config_default_step_none() {
    let config = ShowcaseConfig::default();
    assert!(config.step.is_none());
}

#[test]
fn test_showcase_config_default_baselines() {
    let config = ShowcaseConfig::default();
    assert_eq!(config.baselines.len(), 2);
    assert!(config.baselines.contains(&Baseline::LlamaCpp));
    assert!(config.baselines.contains(&Baseline::Ollama));
}

#[test]
fn test_showcase_config_default_bench_runs_30() {
    let config = ShowcaseConfig::default();
    assert_eq!(config.bench_runs, 30);
}

#[test]
fn test_showcase_config_default_zram_enabled() {
    let config = ShowcaseConfig::default();
    assert!(config.zram);
}

#[test]
fn test_showcase_config_default_gpu_disabled() {
    let config = ShowcaseConfig::default();
    assert!(!config.gpu);
}

#[test]
fn test_showcase_config_default_verbose_quiet() {
    let config = ShowcaseConfig::default();
    assert!(!config.verbose);
    assert!(!config.quiet);
}

#[test]
fn test_showcase_config_with_tier() {
    let config = ShowcaseConfig::with_tier(ModelTier::Large);
    assert_eq!(config.tier, ModelTier::Large);
    assert!(config.model.contains("32B") || config.model.contains("bartowski"));
    // Other defaults should still apply
    assert_eq!(config.quant, "Q4_K_M");
    assert_eq!(config.bench_runs, 30);
}

#[test]
fn test_showcase_config_with_tier_tiny() {
    let config = ShowcaseConfig::with_tier(ModelTier::Tiny);
    assert_eq!(config.tier, ModelTier::Tiny);
    assert!(config.model.contains("0.5B"));
}

#[test]
fn test_showcase_config_with_tier_preserves_defaults() {
    let config = ShowcaseConfig::with_tier(ModelTier::Medium);
    assert_eq!(config.model_dir, std::path::PathBuf::from("./models"));
    assert!(!config.auto_verify);
    assert!(config.step.is_none());
    assert!(config.zram);
    assert_eq!(config.bench_runs, 30);
}

// ========================================================================
// ShowcaseStep Tests
// ========================================================================

#[test]
fn test_showcase_step_equality() {
    assert_eq!(ShowcaseStep::Import, ShowcaseStep::Import);
    assert_ne!(ShowcaseStep::Import, ShowcaseStep::Convert);
    assert_ne!(ShowcaseStep::GgufInference, ShowcaseStep::AprInference);
}

#[test]
fn test_showcase_step_copy_clone() {
    let step = ShowcaseStep::Benchmark;
    let copied = step;
    let cloned = step.clone();
    assert_eq!(copied, ShowcaseStep::Benchmark);
    assert_eq!(cloned, ShowcaseStep::Benchmark);
}

#[test]
fn test_showcase_step_debug_all_variants() {
    // Verify all variants can be debug-printed without panic
    let steps = vec![
        ShowcaseStep::Import,
        ShowcaseStep::GgufInference,
        ShowcaseStep::Convert,
        ShowcaseStep::AprInference,
        ShowcaseStep::Benchmark,
        ShowcaseStep::Chat,
        ShowcaseStep::Visualize,
        ShowcaseStep::ZramDemo,
        ShowcaseStep::CudaDemo,
        ShowcaseStep::BrickDemo,
        ShowcaseStep::All,
    ];
    for step in &steps {
        let debug = format!("{:?}", step);
        assert!(!debug.is_empty());
    }
}

// ========================================================================
// Baseline Tests
// ========================================================================

#[test]
fn test_baseline_equality() {
    assert_eq!(Baseline::LlamaCpp, Baseline::LlamaCpp);
    assert_eq!(Baseline::Ollama, Baseline::Ollama);
    assert_ne!(Baseline::LlamaCpp, Baseline::Ollama);
}

#[test]
fn test_baseline_copy_clone() {
    let b = Baseline::LlamaCpp;
    let copied = b;
    let cloned = b.clone();
    assert_eq!(copied, Baseline::LlamaCpp);
    assert_eq!(cloned, Baseline::LlamaCpp);
}

// ========================================================================
// ExportFormat Tests
// ========================================================================

#[test]
fn test_export_format_equality() {
    assert_eq!(ExportFormat::None, ExportFormat::None);
    assert_eq!(ExportFormat::Json, ExportFormat::Json);
    assert_eq!(ExportFormat::Csv, ExportFormat::Csv);
    assert_ne!(ExportFormat::None, ExportFormat::Json);
    assert_ne!(ExportFormat::Json, ExportFormat::Csv);
}

#[test]
fn test_export_format_copy_clone() {
    let fmt = ExportFormat::Csv;
    let copied = fmt;
    let cloned = fmt.clone();
    assert_eq!(copied, ExportFormat::Csv);
    assert_eq!(cloned, ExportFormat::Csv);
}

// ========================================================================
// BenchmarkComparison Comprehensive Tests
// ========================================================================

#[test]
fn test_benchmark_comparison_no_baselines() {
    let bench = BenchmarkComparison {
        apr_tps: 50.0,
        llama_cpp_tps: None,
        ollama_tps: None,
        apr_ttft_ms: 80.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
        apr_tps_stddev: 2.0,
        runs: 30,
    };

    assert!(bench.llama_cpp_tps.is_none());
    assert!(bench.speedup_vs_llama.is_none());
    assert!(bench.speedup_vs_ollama.is_none());
}

#[test]
fn test_benchmark_comparison_negative_speedup() {
    // APR slower than baseline
    let bench = BenchmarkComparison {
        apr_tps: 30.0,
        llama_cpp_tps: Some(35.0),
        ollama_tps: None,
        apr_ttft_ms: 100.0,
        llama_cpp_ttft_ms: Some(80.0),
        ollama_ttft_ms: None,
        speedup_vs_llama: Some(-14.3),
        speedup_vs_ollama: None,
        apr_tps_stddev: 1.0,
        runs: 30,
    };

    assert!(bench.speedup_vs_llama.unwrap() < 0.0);
}

#[test]
fn test_benchmark_comparison_deserialization() {
    let json = r#"{
        "apr_tps": 100.5,
        "llama_cpp_tps": null,
        "ollama_tps": 80.0,
        "apr_ttft_ms": 50.0,
        "llama_cpp_ttft_ms": null,
        "ollama_ttft_ms": 70.0,
        "speedup_vs_llama": null,
        "speedup_vs_ollama": 25.6,
        "apr_tps_stddev": 3.2,
        "runs": 50
    }"#;
    let bench: BenchmarkComparison = serde_json::from_str(json).unwrap();
    assert!((bench.apr_tps - 100.5).abs() < 0.001);
    assert!(bench.llama_cpp_tps.is_none());
    assert!((bench.ollama_tps.unwrap() - 80.0).abs() < 0.001);
    assert_eq!(bench.runs, 50);
}

#[test]
fn test_benchmark_comparison_clone() {
    let original = BenchmarkComparison {
        apr_tps: 44.0,
        llama_cpp_tps: Some(35.0),
        ollama_tps: Some(32.0),
        apr_ttft_ms: 78.0,
        llama_cpp_ttft_ms: Some(120.0),
        ollama_ttft_ms: Some(150.0),
        speedup_vs_llama: Some(25.7),
        speedup_vs_ollama: Some(37.5),
        apr_tps_stddev: 1.5,
        runs: 30,
    };
    let cloned = original.clone();
    assert!((cloned.apr_tps - 44.0).abs() < 0.001);
    assert_eq!(cloned.runs, 30);
    assert!((cloned.speedup_vs_llama.unwrap() - 25.7).abs() < 0.001);
}

// ========================================================================
// BenchMeasurement Comprehensive Tests
// ========================================================================

#[test]
fn test_bench_measurement_tps_high_throughput() {
    let m = BenchMeasurement {
        tokens_generated: 1000,
        duration: Duration::from_millis(100),
        ttft: Duration::from_millis(5),
    };
    // 1000 / 0.1 = 10000 tok/s
    assert!((m.tokens_per_second() - 10000.0).abs() < 0.1);
}

#[test]
fn test_bench_measurement_tps_one_token() {
    let m = BenchMeasurement {
        tokens_generated: 1,
        duration: Duration::from_secs(1),
        ttft: Duration::from_secs(1),
    };
    assert!((m.tokens_per_second() - 1.0).abs() < 0.01);
}

#[test]
fn test_bench_measurement_tps_zero_tokens() {
    let m = BenchMeasurement {
        tokens_generated: 0,
        duration: Duration::from_secs(1),
        ttft: Duration::from_millis(50),
    };
    assert_eq!(m.tokens_per_second(), 0.0);
}

#[test]
fn test_bench_measurement_clone() {
    let original = BenchMeasurement {
        tokens_generated: 50,
        duration: Duration::from_millis(500),
        ttft: Duration::from_millis(25),
    };
    let cloned = original.clone();
    assert_eq!(cloned.tokens_generated, 50);
    assert_eq!(cloned.duration, Duration::from_millis(500));
    assert_eq!(cloned.ttft, Duration::from_millis(25));
}

// ========================================================================
// calculate_stddev Tests
// ========================================================================

#[test]
fn test_calculate_stddev_large_spread() {
    let values = vec![1.0, 100.0];
    let stddev = calculate_stddev(&values);
    // Mean = 50.5, variance = ((1-50.5)^2 + (100-50.5)^2)/1 = 4900.5
    // stddev = sqrt(4900.5) ~ 70.0
    assert!((stddev - 70.0).abs() < 0.5);
}

#[test]
fn test_calculate_stddev_negative_values() {
    let values = vec![-10.0, 0.0, 10.0];
    let stddev = calculate_stddev(&values);
    // Mean = 0, variance = (100 + 0 + 100)/2 = 100
    // stddev = 10.0
    assert!((stddev - 10.0).abs() < 0.01);
}
