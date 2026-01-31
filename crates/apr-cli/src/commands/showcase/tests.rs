// File-level imports from showcase module
#[allow(unused_imports)]
use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use super::benchmark::{calculate_stddev, generate_jitter};
    use super::demo::{run_zram_demo, run_cuda_demo};
    use std::time::Duration;
    use super::super::benchmark::{export_benchmark_results, format_benchmark_csv};

    /// Helper: create a full-run config for testing validation
    fn full_run_config() -> ShowcaseConfig {
        ShowcaseConfig {
            auto_verify: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_showcase_config_default() {
        let config = ShowcaseConfig::default();
        assert!(config.model.contains("Qwen2.5-Coder"));
        assert_eq!(config.quant, "Q4_K_M");
        assert_eq!(config.bench_runs, 30);
        assert!(config.zram);
    }

    #[test]
    fn test_benchmark_comparison_speedup() {
        let comparison = BenchmarkComparison {
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

        assert!(comparison.speedup_vs_llama.unwrap() >= 25.0);
        assert!(comparison.speedup_vs_ollama.unwrap() >= 25.0);
    }

    #[test]
    fn test_falsification_passes_with_valid_metrics() {
        let results = ShowcaseResults {
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            benchmark: Some(BenchmarkComparison {
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
            }),
            visualize: true,
            chat: true,
            ..Default::default()
        };

        assert!(validate_falsification(&results, &full_run_config()).is_ok());
    }

    #[test]
    fn test_falsification_fails_below_25_percent() {
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 40.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: None,
                apr_ttft_ms: 80.0,
                llama_cpp_ttft_ms: Some(100.0),
                ollama_ttft_ms: None,
                speedup_vs_llama: Some(14.3),
                speedup_vs_ollama: None,
                apr_tps_stddev: 1.0,
                runs: 30,
            }),
            ..Default::default()
        };

        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    #[test]
    fn test_falsification_fails_insufficient_runs() {
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 44.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(32.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(150.0),
                speedup_vs_llama: Some(25.7),
                speedup_vs_ollama: Some(37.5),
                apr_tps_stddev: 1.5,
                runs: 10, // Below 30!
            }),
            ..Default::default()
        };

        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    #[test]
    fn test_falsification_fails_high_variance() {
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 44.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(32.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(150.0),
                speedup_vs_llama: Some(25.7),
                speedup_vs_ollama: Some(37.5),
                apr_tps_stddev: 5.0, // CV = 5/44 = 11.4% > 5%
                runs: 30,
            }),
            ..Default::default()
        };

        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    #[test]
    fn test_calculate_stddev() {
        let values = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let stddev = calculate_stddev(&values);
        assert!((stddev - 3.162).abs() < 0.01);
    }

    #[test]
    fn test_calculate_stddev_empty() {
        assert_eq!(calculate_stddev(&[]), 0.0);
        assert_eq!(calculate_stddev(&[42.0]), 0.0);
    }

    #[test]
    fn test_bench_measurement_tps() {
        let measurement = BenchMeasurement {
            tokens_generated: 100,
            duration: Duration::from_secs(2),
            ttft: Duration::from_millis(50),
        };
        assert!((measurement.tokens_per_second() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_generate_jitter_range() {
        for _ in 0..100 {
            let jitter = generate_jitter();
            assert!(jitter >= -1.0);
            assert!(jitter <= 1.0);
        }
    }

    // === Falsification Point 49: CV <5% ===
    #[test]
    fn test_cv_calculation_at_boundary() {
        // CV = stddev/mean * 100 = 2.2/44.0 * 100 = 5.0% (exactly at limit)
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 44.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(32.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(150.0),
                speedup_vs_llama: Some(25.7),
                speedup_vs_ollama: Some(37.5),
                apr_tps_stddev: 2.199, // CV = 4.998% (just under 5%)
                runs: 30,
            }),
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            visualize: true,
            chat: true,
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_ok());
    }

    // === Falsification Point 50: 30+ runs ===
    #[test]
    fn test_exactly_30_runs_passes() {
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 44.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(32.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(150.0),
                speedup_vs_llama: Some(25.7),
                speedup_vs_ollama: Some(37.5),
                apr_tps_stddev: 1.5,
                runs: 30, // Exactly 30 - should pass
            }),
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            visualize: true,
            chat: true,
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_ok());
    }

    #[test]
    fn test_29_runs_fails() {
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 44.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(32.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(150.0),
                speedup_vs_llama: Some(25.7),
                speedup_vs_ollama: Some(37.5),
                apr_tps_stddev: 1.5,
                runs: 29, // One less than required
            }),
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            visualize: true,
            chat: true,
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    // === Falsification Point 41/42: 25% speedup ===
    #[test]
    fn test_speedup_exactly_25_percent_passes() {
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 43.75, // 35 * 1.25 = 43.75
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(32.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(150.0),
                speedup_vs_llama: Some(25.0), // Exactly 25%
                speedup_vs_ollama: Some(36.7),
                apr_tps_stddev: 1.5,
                runs: 30,
            }),
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            visualize: true,
            chat: true,
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_ok());
    }

    #[test]
    fn test_speedup_24_9_percent_fails() {
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 43.7,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(32.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(150.0),
                speedup_vs_llama: Some(24.9), // Below 25%
                speedup_vs_ollama: Some(36.5),
                apr_tps_stddev: 1.5,
                runs: 30,
            }),
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            visualize: true,
            chat: true,
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    // === Falsification: No benchmark data ===
    #[test]
    fn test_no_benchmark_fails() {
        let results = ShowcaseResults {
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            benchmark: None, // Missing benchmark
            visualize: true,
            chat: true,
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    // === BenchMeasurement tests ===
    #[test]
    fn test_bench_measurement_zero_duration() {
        let measurement = BenchMeasurement {
            tokens_generated: 100,
            duration: Duration::from_millis(0),
            ttft: Duration::from_millis(50),
        };
        // With zero duration, tps returns 0.0 (safe behavior)
        let tps = measurement.tokens_per_second();
        assert_eq!(tps, 0.0);
    }

    #[test]
    fn test_bench_measurement_fractional_seconds() {
        let measurement = BenchMeasurement {
            tokens_generated: 50,
            duration: Duration::from_millis(500),
            ttft: Duration::from_millis(25),
        };
        // 50 tokens / 0.5 seconds = 100 tok/s
        assert!((measurement.tokens_per_second() - 100.0).abs() < 0.01);
    }

    // === stddev edge cases ===
    #[test]
    fn test_calculate_stddev_identical_values() {
        let values = vec![42.0, 42.0, 42.0, 42.0, 42.0];
        let stddev = calculate_stddev(&values);
        assert_eq!(stddev, 0.0);
    }

    #[test]
    fn test_calculate_stddev_two_values() {
        let values = vec![10.0, 20.0];
        let stddev = calculate_stddev(&values);
        // Mean = 15, variance = ((10-15)^2 + (20-15)^2)/(n-1) = 50/1 = 50
        // Sample stddev = sqrt(50) ≈ 7.07
        assert!((stddev - 7.07).abs() < 0.01);
    }

    // === ShowcaseResults default ===
    #[test]
    fn test_showcase_results_default() {
        let results = ShowcaseResults::default();
        assert!(!results.import);
        assert!(!results.gguf_inference);
        assert!(!results.convert);
        assert!(!results.apr_inference);
        assert!(results.benchmark.is_none());
        assert!(!results.visualize);
        assert!(!results.chat);
    }

    // === Speedup calculation formula verification ===
    #[test]
    fn test_speedup_formula() {
        // speedup = (new - old) / old * 100
        // APR: 44 tok/s, llama.cpp: 35 tok/s
        // speedup = (44 - 35) / 35 * 100 = 9/35 * 100 = 25.71%
        let apr: f64 = 44.0;
        let baseline: f64 = 35.0;
        let speedup = (apr - baseline) / baseline * 100.0;
        assert!((speedup - 25.71).abs() < 0.1);
    }

    // === Ollama missing but llama.cpp present ===
    #[test]
    fn test_only_llama_cpp_baseline_passes() {
        let results = ShowcaseResults {
            benchmark: Some(BenchmarkComparison {
                apr_tps: 44.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: None, // Ollama not tested
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: None,
                speedup_vs_llama: Some(25.7),
                speedup_vs_ollama: None, // No Ollama speedup
                apr_tps_stddev: 1.5,
                runs: 30,
            }),
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            visualize: true,
            chat: true,
            ..Default::default()
        };
        // Should pass - only llama.cpp baseline required
        assert!(validate_falsification(&results, &full_run_config()).is_ok());
    }

    // === Step failures should fail falsification ===
    #[test]
    fn test_import_failure_fails_falsification() {
        let results = ShowcaseResults {
            import: false, // Failed
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            benchmark: Some(BenchmarkComparison {
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
            }),
            visualize: true,
            chat: true,
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    #[test]
    fn test_convert_failure_fails_falsification() {
        let results = ShowcaseResults {
            import: true,
            gguf_inference: true,
            convert: false, // Failed
            apr_inference: true,
            benchmark: Some(BenchmarkComparison {
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
            }),
            visualize: true,
            chat: true,
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    // === Export Format Tests (Point 85) ===

    #[test]
    fn test_export_format_default() {
        assert_eq!(ExportFormat::default(), ExportFormat::None);
    }

    #[test]
    fn test_showcase_config_includes_export_fields() {
        let config = ShowcaseConfig::default();
        assert_eq!(config.export_format, ExportFormat::None);
        assert!(config.export_path.is_none());
    }

    #[test]
    fn test_benchmark_comparison_json_serialization() {
        let bench = BenchmarkComparison {
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

        let json = serde_json::to_string(&bench).unwrap();
        assert!(json.contains("\"apr_tps\":44.0"));
        assert!(json.contains("\"runs\":30"));

        // Round-trip
        let parsed: BenchmarkComparison = serde_json::from_str(&json).unwrap();
        assert!((parsed.apr_tps - 44.0).abs() < 0.001);
        assert_eq!(parsed.runs, 30);
    }

    #[test]
    fn test_format_benchmark_csv_all_baselines() {
        let bench = BenchmarkComparison {
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

        let csv = format_benchmark_csv(&bench);

        // Check header
        assert!(csv.starts_with("system,tokens_per_sec,ttft_ms,speedup_pct,stddev,runs\n"));

        // Check APR row
        assert!(csv.contains("APR,44.00,78.00,,1.50,30"));

        // Check baseline rows
        assert!(csv.contains("llama.cpp,35.00,120.00,25.70,N/A,N/A"));
        assert!(csv.contains("Ollama,32.00,150.00,37.50,N/A,N/A"));
    }

    #[test]
    fn test_format_benchmark_csv_no_baselines() {
        let bench = BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.5,
            runs: 30,
        };

        let csv = format_benchmark_csv(&bench);

        // Check header and APR row only
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2); // Header + APR
        assert!(lines[1].contains("APR"));
    }

    #[test]
    fn test_format_benchmark_csv_llama_only() {
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

        let csv = format_benchmark_csv(&bench);

        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // Header + APR + llama.cpp
        assert!(csv.contains("llama.cpp"));
        assert!(!csv.contains("Ollama"));
    }

    #[test]
    fn test_export_json_to_file() {
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

        let temp_dir = tempfile::tempdir().unwrap();
        let export_path = temp_dir.path().join("benchmark.json");

        let config = ShowcaseConfig {
            export_format: ExportFormat::Json,
            export_path: Some(export_path.clone()),
            model_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        export_benchmark_results(&bench, &config).unwrap();

        // Verify file exists and contains valid JSON
        assert!(export_path.exists());
        let content = std::fs::read_to_string(&export_path).unwrap();
        let parsed: BenchmarkComparison = serde_json::from_str(&content).unwrap();
        assert!((parsed.apr_tps - 44.0).abs() < 0.001);
    }

    #[test]
    fn test_export_csv_to_file() {
        let bench = BenchmarkComparison {
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

        let temp_dir = tempfile::tempdir().unwrap();
        let export_path = temp_dir.path().join("benchmark.csv");

        let config = ShowcaseConfig {
            export_format: ExportFormat::Csv,
            export_path: Some(export_path.clone()),
            model_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        export_benchmark_results(&bench, &config).unwrap();

        // Verify file exists and contains CSV header
        assert!(export_path.exists());
        let content = std::fs::read_to_string(&export_path).unwrap();
        assert!(content.starts_with("system,tokens_per_sec"));
        assert!(content.contains("APR,44.00"));
    }

    #[test]
    fn test_export_none_creates_no_file() {
        let bench = BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.5,
            runs: 30,
        };

        let temp_dir = tempfile::tempdir().unwrap();
        let json_path = temp_dir.path().join("benchmark-results.json");
        let csv_path = temp_dir.path().join("benchmark-results.csv");

        let config = ShowcaseConfig {
            export_format: ExportFormat::None,
            export_path: None,
            model_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        export_benchmark_results(&bench, &config).unwrap();

        // No files should be created
        assert!(!json_path.exists());
        assert!(!csv_path.exists());
    }

    // === ZRAM Demo Tests (Point 79-82) ===

    #[test]
    fn test_zram_demo_result_fields() {
        let result = ZramDemoResult {
            lz4_ratio: 2.5,
            zstd_ratio: 3.2,
            zero_page_gbps: 175.0,
            lz4_gbps: 3.5,
            simd_backend: "Avx2".to_string(),
            context_extension: 2.5,
        };

        // Verify all fields are accessible
        assert!(result.lz4_ratio > 2.0);
        assert!(result.zstd_ratio > result.lz4_ratio);
        assert!(result.zero_page_gbps > 150.0); // Point 81 target
        assert!(result.lz4_gbps > 3.0); // Target throughput
        assert!(!result.simd_backend.is_empty());
        assert!(result.context_extension >= 2.0); // Point 80 target
    }

    #[test]
    #[cfg(feature = "zram")]
    fn test_zram_demo_runs_successfully() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShowcaseConfig {
            model_dir: temp_dir.path().to_path_buf(),
            zram: true,
            ..Default::default()
        };

        // Run ZRAM demo - should complete without error
        let result = run_zram_demo(&config);
        assert!(result.is_ok(), "ZRAM demo should complete successfully");

        let zram_result = result.unwrap();
        // Verify compression ratios are positive
        assert!(zram_result.lz4_ratio > 0.0, "LZ4 ratio should be positive");
        assert!(
            zram_result.zstd_ratio > 0.0,
            "ZSTD ratio should be positive"
        );
        // Verify throughput measurements
        assert!(
            zram_result.zero_page_gbps > 0.0,
            "Zero-page throughput should be measurable"
        );
        assert!(
            zram_result.lz4_gbps > 0.0,
            "LZ4 throughput should be measurable"
        );
        // Verify context extension (Point 80)
        assert!(
            zram_result.context_extension > 0.0,
            "Context extension should be calculated"
        );
    }

    #[test]
    #[cfg(feature = "zram")]
    fn test_zram_context_extension_point_80() {
        // Point 80: ZRAM ≥2x context extension
        // With typical compression ratios, we should achieve at least 2x context extension
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShowcaseConfig {
            model_dir: temp_dir.path().to_path_buf(),
            zram: true,
            ..Default::default()
        };

        let result = run_zram_demo(&config).expect("ZRAM demo should succeed");

        // Context extension should be at least 2.0x for LZ4/ZSTD
        // This verifies that ZRAM enables 16K -> 32K+ token context
        assert!(
            result.context_extension >= 2.0,
            "Point 80 FAILED: context extension {:.2}x < 2.0x target",
            result.context_extension
        );
    }

    #[test]
    #[cfg(feature = "zram")]
    fn test_zram_zero_page_optimization() {
        use trueno_zram_core::{Algorithm as ZramAlgorithm, CompressorBuilder, PAGE_SIZE};

        // Zero pages should compress extremely well
        let compressor = CompressorBuilder::new()
            .algorithm(ZramAlgorithm::Lz4)
            .build()
            .unwrap();

        let zero_page = [0u8; PAGE_SIZE];
        let compressed = compressor.compress(&zero_page).unwrap();

        // Zero page should compress to very small size (same-fill optimization)
        let ratio = PAGE_SIZE as f64 / compressed.data.len() as f64;
        assert!(
            ratio > 100.0,
            "Zero page ratio should be >100x, got {:.1}x",
            ratio
        );
    }

    #[test]
    #[cfg(feature = "zram")]
    fn test_zram_compression_stats_reporting() {
        use trueno_zram_core::{Algorithm as ZramAlgorithm, CompressorBuilder, PAGE_SIZE};

        let compressor = CompressorBuilder::new()
            .algorithm(ZramAlgorithm::Lz4)
            .build()
            .unwrap();

        // Compress some pages
        let test_page = [0x42u8; PAGE_SIZE];
        let _ = compressor.compress(&test_page).unwrap();
        let _ = compressor.compress(&test_page).unwrap();

        // Verify stats are tracked (Point 82)
        let stats = compressor.stats();
        assert!(stats.pages_compressed >= 2, "Should track compressed pages");
        assert!(stats.bytes_in > 0, "Should track bytes in");
        assert!(stats.bytes_out > 0, "Should track bytes out");
    }

    // === CUDA Demo Tests (Point 78) ===

    #[test]
    fn test_cuda_demo_result_fields() {
        let result = CudaDemoResult {
            device_count: 1,
            device_name: "NVIDIA GeForce RTX 4090".to_string(),
            total_vram_gb: 24.0,
            free_vram_gb: 20.0,
            cuda_available: true,
            graph_capture_available: true,
            graph_speedup: 70.0,
            dp4a_available: true,
            dp4a_arithmetic_intensity: 1.78,
        };

        assert_eq!(result.device_count, 1);
        assert!(result.device_name.contains("RTX"));
        assert!(result.total_vram_gb >= 20.0); // RTX 4090 has 24GB
        assert!(result.free_vram_gb > 0.0);
        assert!(result.cuda_available);
        // Section 5.2/5.3 fields
        assert!(result.graph_capture_available);
        assert!(result.graph_speedup > 1.0);
        assert!(result.dp4a_available);
        assert!(result.dp4a_arithmetic_intensity > 0.0);
    }

    #[test]
    fn test_cuda_demo_disabled_result() {
        let result = CudaDemoResult {
            device_count: 0,
            device_name: "disabled".to_string(),
            total_vram_gb: 0.0,
            free_vram_gb: 0.0,
            cuda_available: false,
            graph_capture_available: false,
            graph_speedup: 1.0,
            dp4a_available: false,
            dp4a_arithmetic_intensity: 0.0,
        };

        assert_eq!(result.device_count, 0);
        assert!(!result.cuda_available);
        assert!(!result.graph_capture_available);
        assert!(!result.dp4a_available);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_demo_runs_successfully() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShowcaseConfig {
            model_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        // Run CUDA demo - should complete without error
        let result = run_cuda_demo(&config);
        assert!(result.is_ok(), "CUDA demo should complete: {:?}", result);

        let cuda_result = result.unwrap();
        // Either CUDA is available with devices, or it's not
        if cuda_result.cuda_available {
            assert!(cuda_result.device_count > 0);
            assert!(!cuda_result.device_name.is_empty());
            assert!(cuda_result.total_vram_gb > 0.0);
        }
    }
}
