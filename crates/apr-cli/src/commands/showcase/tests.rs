// File-level imports from showcase module
#[allow(unused_imports)]
use super::*;

#[cfg(test)]
mod tests {
    use super::super::benchmark::{export_benchmark_results, format_benchmark_csv};
    use super::benchmark::{calculate_stddev, generate_jitter};
    use super::demo::{run_cuda_demo, run_zram_demo};
    use super::*;
    use std::time::Duration;

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

    // ========================================================================
    // ModelTier Tests
    // ========================================================================

    #[test]
    fn test_model_tier_default_is_small() {
        assert_eq!(ModelTier::default(), ModelTier::Small);
    }

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

    #[test]
    fn test_calculate_stddev_many_values() {
        // All same value should give 0 stddev
        let values: Vec<f64> = vec![5.0; 100];
        let stddev = calculate_stddev(&values);
        assert_eq!(stddev, 0.0);
    }

    // ========================================================================
    // generate_jitter Tests
    // ========================================================================

    #[test]
    fn test_generate_jitter_bounded() {
        // Run many times to check bounds
        for _ in 0..1000 {
            let j = generate_jitter();
            assert!(j >= -1.0, "jitter {j} below -1.0");
            assert!(j <= 1.0, "jitter {j} above 1.0");
        }
    }

    // ========================================================================
    // extract_json_field Tests
    // ========================================================================

    #[test]
    fn test_extract_json_field_basic() {
        let json = r#"{"eval_count":42,"eval_duration":1000000000}"#;
        let count = super::benchmark::extract_json_field(json, "eval_count");
        assert_eq!(count, Some(42.0));

        let duration = super::benchmark::extract_json_field(json, "eval_duration");
        assert_eq!(duration, Some(1_000_000_000.0));
    }

    #[test]
    fn test_extract_json_field_with_spaces() {
        let json = r#"{"eval_count": 42, "eval_duration": 1000000000}"#;
        let count = super::benchmark::extract_json_field(json, "eval_count");
        assert_eq!(count, Some(42.0));
    }

    #[test]
    fn test_extract_json_field_float() {
        let json = r#"{"value": 3.14}"#;
        let val = super::benchmark::extract_json_field(json, "value");
        assert!((val.unwrap() - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_extract_json_field_missing() {
        let json = r#"{"other": 42}"#;
        let val = super::benchmark::extract_json_field(json, "missing");
        assert!(val.is_none());
    }

    #[test]
    fn test_extract_json_field_empty_json() {
        let json = r#"{}"#;
        let val = super::benchmark::extract_json_field(json, "field");
        assert!(val.is_none());
    }

    #[test]
    fn test_extract_json_field_nested() {
        // Should find the first occurrence at top level
        let json = r#"{"eval_count":100,"nested":{"eval_count":200}}"#;
        let val = super::benchmark::extract_json_field(json, "eval_count");
        assert_eq!(val, Some(100.0));
    }

    // ========================================================================
    // format_benchmark_csv Comprehensive Tests
    // ========================================================================

    #[test]
    fn test_format_benchmark_csv_header_structure() {
        let bench = BenchmarkComparison {
            apr_tps: 1.0,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 1.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 0.1,
            runs: 1,
        };
        let csv = format_benchmark_csv(&bench);
        let header = csv.lines().next().unwrap();
        assert_eq!(
            header,
            "system,tokens_per_sec,ttft_ms,speedup_pct,stddev,runs"
        );
    }

    #[test]
    fn test_format_benchmark_csv_apr_row_format() {
        let bench = BenchmarkComparison {
            apr_tps: 123.45,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 67.89,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 2.34,
            runs: 50,
        };
        let csv = format_benchmark_csv(&bench);
        assert!(csv.contains("APR,123.45,67.89,,2.34,50"));
    }

    #[test]
    fn test_format_benchmark_csv_ollama_only() {
        let bench = BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: None,
            ollama_tps: Some(32.0),
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: Some(150.0),
            speedup_vs_llama: None,
            speedup_vs_ollama: Some(37.5),
            apr_tps_stddev: 1.5,
            runs: 30,
        };
        let csv = format_benchmark_csv(&bench);

        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // Header + APR + Ollama
        assert!(!csv.contains("llama.cpp"));
        assert!(csv.contains("Ollama"));
    }

    // ========================================================================
    // ShowcaseResults Default Tests
    // ========================================================================

    #[test]
    fn test_showcase_results_default_all_false() {
        let r = ShowcaseResults::default();
        assert!(!r.import);
        assert!(!r.gguf_inference);
        assert!(!r.convert);
        assert!(!r.apr_inference);
        assert!(!r.visualize);
        assert!(!r.chat);
    }

    #[test]
    fn test_showcase_results_default_all_none() {
        let r = ShowcaseResults::default();
        assert!(r.benchmark.is_none());
        assert!(r.zram_demo.is_none());
        assert!(r.cuda_demo.is_none());
        assert!(r.brick_demo.is_none());
    }

    // ========================================================================
    // BrickDemoResult Tests
    // ========================================================================

    #[test]
    fn test_brick_demo_result_default() {
        let r = BrickDemoResult::default();
        assert_eq!(r.layers_measured, 0);
        assert!(r.layer_timings_us.is_empty());
        assert!(r.bottleneck.is_none());
        assert_eq!(r.total_us, 0.0);
        assert_eq!(r.tokens_per_sec, 0.0);
        assert!(!r.assertions_passed);
    }

    #[test]
    fn test_brick_demo_result_with_data() {
        let r = BrickDemoResult {
            layers_measured: 28,
            layer_timings_us: vec![100.0, 120.0, 110.0],
            bottleneck: Some(("FfnBrick".to_string(), 120.0)),
            total_us: 3080.0,
            tokens_per_sec: 324.7,
            assertions_passed: true,
        };
        assert_eq!(r.layers_measured, 28);
        assert_eq!(r.layer_timings_us.len(), 3);
        assert!(r.bottleneck.is_some());
        let (name, time) = r.bottleneck.unwrap();
        assert_eq!(name, "FfnBrick");
        assert!((time - 120.0).abs() < 0.001);
        assert!(r.assertions_passed);
    }

    #[test]
    fn test_brick_demo_result_clone() {
        let original = BrickDemoResult {
            layers_measured: 10,
            layer_timings_us: vec![50.0, 55.0],
            bottleneck: Some(("Attn".to_string(), 55.0)),
            total_us: 525.0,
            tokens_per_sec: 1904.8,
            assertions_passed: true,
        };
        let cloned = original.clone();
        assert_eq!(cloned.layers_measured, 10);
        assert_eq!(cloned.layer_timings_us.len(), 2);
    }

    // ========================================================================
    // ZramDemoResult Tests
    // ========================================================================

    #[test]
    fn test_zram_demo_result_clone() {
        let original = ZramDemoResult {
            lz4_ratio: 2.5,
            zstd_ratio: 3.2,
            zero_page_gbps: 175.0,
            lz4_gbps: 3.5,
            simd_backend: "Avx2".to_string(),
            context_extension: 2.5,
        };
        let cloned = original.clone();
        assert!((cloned.lz4_ratio - 2.5).abs() < 0.001);
        assert_eq!(cloned.simd_backend, "Avx2");
    }

    #[test]
    fn test_zram_demo_result_context_extension_calculation() {
        // Verify context extension represents multiplier
        let result = ZramDemoResult {
            lz4_ratio: 3.0,
            zstd_ratio: 4.0,
            zero_page_gbps: 200.0,
            lz4_gbps: 5.0,
            simd_backend: "Neon".to_string(),
            context_extension: 2.5,
        };
        // 16K * 2.5 = 40K tokens
        let extended_tokens = 16_000.0 * result.context_extension;
        assert!((extended_tokens - 40_000.0).abs() < 0.1);
    }

    // ========================================================================
    // CudaDemoResult Tests
    // ========================================================================

    #[test]
    fn test_cuda_demo_result_clone() {
        let original = CudaDemoResult {
            device_count: 2,
            device_name: "Tesla V100".to_string(),
            total_vram_gb: 32.0,
            free_vram_gb: 28.0,
            cuda_available: true,
            graph_capture_available: true,
            graph_speedup: 50.0,
            dp4a_available: true,
            dp4a_arithmetic_intensity: 2.0,
        };
        let cloned = original.clone();
        assert_eq!(cloned.device_count, 2);
        assert_eq!(cloned.device_name, "Tesla V100");
        assert!(cloned.cuda_available);
    }

    #[test]
    fn test_cuda_demo_result_no_gpu() {
        let result = CudaDemoResult {
            device_count: 0,
            device_name: String::new(),
            total_vram_gb: 0.0,
            free_vram_gb: 0.0,
            cuda_available: false,
            graph_capture_available: false,
            graph_speedup: 0.0,
            dp4a_available: false,
            dp4a_arithmetic_intensity: 0.0,
        };
        assert!(!result.cuda_available);
        assert_eq!(result.device_count, 0);
        assert!(!result.graph_capture_available);
        assert!(!result.dp4a_available);
    }

    // ========================================================================
    // Validation: Single Step Mode Tests
    // ========================================================================

    #[test]
    fn test_falsification_standalone_cuda_demo_passes() {
        let results = ShowcaseResults::default();
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::CudaDemo),
            ..Default::default()
        };
        // Standalone demos should pass without full pipeline
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    #[test]
    fn test_falsification_standalone_zram_demo_passes() {
        let results = ShowcaseResults::default();
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::ZramDemo),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    #[test]
    fn test_falsification_standalone_brick_demo_passes() {
        let results = ShowcaseResults::default();
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::BrickDemo),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    #[test]
    fn test_falsification_single_import_step_passes() {
        let results = ShowcaseResults {
            import: true,
            ..Default::default()
        };
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::Import),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    #[test]
    fn test_falsification_single_import_step_fails() {
        let results = ShowcaseResults {
            import: false,
            ..Default::default()
        };
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::Import),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_err());
    }

    #[test]
    fn test_falsification_single_convert_step_passes() {
        let results = ShowcaseResults {
            convert: true,
            ..Default::default()
        };
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::Convert),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    #[test]
    fn test_falsification_single_gguf_step_fails() {
        let results = ShowcaseResults {
            gguf_inference: false,
            ..Default::default()
        };
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::GgufInference),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_err());
    }

    #[test]
    fn test_falsification_single_apr_inference_step_fails() {
        let results = ShowcaseResults {
            apr_inference: false,
            ..Default::default()
        };
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::AprInference),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_err());
    }

    // ========================================================================
    // Validation: Coefficient of Variation Edge Cases
    // ========================================================================

    #[test]
    fn test_falsification_zero_tps_high_cv() {
        let results = ShowcaseResults {
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            benchmark: Some(BenchmarkComparison {
                apr_tps: 0.0,
                llama_cpp_tps: None,
                ollama_tps: None,
                apr_ttft_ms: 0.0,
                llama_cpp_ttft_ms: None,
                ollama_ttft_ms: None,
                speedup_vs_llama: None,
                speedup_vs_ollama: None,
                apr_tps_stddev: 1.0,
                runs: 30,
            }),
            ..Default::default()
        };
        // CV = 1.0/0.0 * 100 = 100% (capped) > 5%
        assert!(super::validation::validate_falsification(&results, &full_run_config()).is_err());
    }

    // ========================================================================
    // Validation: Multiple Failures
    // ========================================================================

    #[test]
    fn test_falsification_multiple_step_failures() {
        let results = ShowcaseResults {
            import: false,
            gguf_inference: false,
            convert: false,
            apr_inference: false,
            benchmark: None,
            ..Default::default()
        };
        let err = super::validation::validate_falsification(&results, &full_run_config());
        assert!(err.is_err());
    }

    // ========================================================================
    // Export: Default Path Tests
    // ========================================================================

    #[test]
    fn test_export_json_default_path() {
        let bench = BenchmarkComparison {
            apr_tps: 50.0,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 60.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.0,
            runs: 30,
        };

        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShowcaseConfig {
            export_format: ExportFormat::Json,
            export_path: None, // Should use default
            model_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        export_benchmark_results(&bench, &config).unwrap();

        // Default path should be model_dir/benchmark-results.json
        let expected_path = temp_dir.path().join("benchmark-results.json");
        assert!(expected_path.exists());
    }

    #[test]
    fn test_export_csv_default_path() {
        let bench = BenchmarkComparison {
            apr_tps: 50.0,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 60.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.0,
            runs: 30,
        };

        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShowcaseConfig {
            export_format: ExportFormat::Csv,
            export_path: None,
            model_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        export_benchmark_results(&bench, &config).unwrap();

        let expected_path = temp_dir.path().join("benchmark-results.csv");
        assert!(expected_path.exists());
    }

    // ========================================================================
    // NEW: print_benchmark_results coverage (benchmark.rs lines 592-646)
    // ========================================================================

    #[test]
    fn test_print_benchmark_results_apr_only() {
        // Exercises the APR-only path (no baselines, no speedup annotations)
        let comparison = BenchmarkComparison {
            apr_tps: 50.0,
            apr_ttft_ms: 70.0,
            apr_tps_stddev: 2.0,
            runs: 30,
            llama_cpp_tps: None,
            llama_cpp_ttft_ms: None,
            ollama_tps: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
        };
        // Should not panic
        super::benchmark::print_benchmark_results(&comparison);
    }

    #[test]
    fn test_print_benchmark_results_with_llama_cpp() {
        // Exercises llama.cpp row + speedup annotation
        let comparison = BenchmarkComparison {
            apr_tps: 50.0,
            apr_ttft_ms: 70.0,
            apr_tps_stddev: 2.0,
            runs: 30,
            llama_cpp_tps: Some(35.0),
            llama_cpp_ttft_ms: Some(120.0),
            ollama_tps: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: Some(42.9),
            speedup_vs_ollama: None,
        };
        super::benchmark::print_benchmark_results(&comparison);
    }

    #[test]
    fn test_print_benchmark_results_with_ollama() {
        // Exercises Ollama row + speedup annotation
        let comparison = BenchmarkComparison {
            apr_tps: 50.0,
            apr_ttft_ms: 70.0,
            apr_tps_stddev: 2.0,
            runs: 30,
            llama_cpp_tps: None,
            llama_cpp_ttft_ms: None,
            ollama_tps: Some(32.0),
            ollama_ttft_ms: Some(150.0),
            speedup_vs_llama: None,
            speedup_vs_ollama: Some(56.25),
        };
        super::benchmark::print_benchmark_results(&comparison);
    }

    #[test]
    fn test_print_benchmark_results_all_baselines() {
        // Exercises both baseline rows and both speedup annotations
        let comparison = BenchmarkComparison {
            apr_tps: 50.0,
            apr_ttft_ms: 70.0,
            apr_tps_stddev: 2.0,
            runs: 30,
            llama_cpp_tps: Some(35.0),
            llama_cpp_ttft_ms: Some(120.0),
            ollama_tps: Some(32.0),
            ollama_ttft_ms: Some(150.0),
            speedup_vs_llama: Some(42.9),
            speedup_vs_ollama: Some(56.25),
        };
        super::benchmark::print_benchmark_results(&comparison);
    }

    #[test]
    fn test_print_benchmark_results_speedup_below_target() {
        // Exercises the FAIL branch for speedup < 25%
        let comparison = BenchmarkComparison {
            apr_tps: 38.0,
            apr_ttft_ms: 90.0,
            apr_tps_stddev: 3.0,
            runs: 30,
            llama_cpp_tps: Some(35.0),
            llama_cpp_ttft_ms: Some(100.0),
            ollama_tps: Some(36.0),
            ollama_ttft_ms: Some(110.0),
            speedup_vs_llama: Some(8.6),  // Below 25%
            speedup_vs_ollama: Some(5.6), // Below 25%
        };
        super::benchmark::print_benchmark_results(&comparison);
    }

    #[test]
    fn test_print_benchmark_results_llama_no_ttft() {
        // Exercises llama.cpp with None ttft (defaults to 0.0)
        let comparison = BenchmarkComparison {
            apr_tps: 50.0,
            apr_ttft_ms: 70.0,
            apr_tps_stddev: 2.0,
            runs: 30,
            llama_cpp_tps: Some(35.0),
            llama_cpp_ttft_ms: None,
            ollama_tps: Some(32.0),
            ollama_ttft_ms: None,
            speedup_vs_llama: Some(42.9),
            speedup_vs_ollama: Some(56.25),
        };
        super::benchmark::print_benchmark_results(&comparison);
    }

    // ========================================================================
    // NEW: extract_json_field edge cases (benchmark.rs lines 469-482)
    // ========================================================================

    #[test]
    fn test_extract_json_field_zero_value() {
        let json = r#"{"count":0}"#;
        let val = super::benchmark::extract_json_field(json, "count");
        assert_eq!(val, Some(0.0));
    }

    #[test]
    fn test_extract_json_field_large_value() {
        let json = r#"{"duration":999999999999}"#;
        let val = super::benchmark::extract_json_field(json, "duration");
        assert_eq!(val, Some(999_999_999_999.0));
    }

    #[test]
    fn test_extract_json_field_trailing_comma() {
        let json = r#"{"value":42,"other":99}"#;
        let val = super::benchmark::extract_json_field(json, "value");
        assert_eq!(val, Some(42.0));
    }

    #[test]
    fn test_extract_json_field_string_value_returns_none() {
        // Non-numeric value after field name
        let json = r#"{"name":"hello"}"#;
        let val = super::benchmark::extract_json_field(json, "name");
        // "hello" starts with '"' which is not digit/dot, so empty parse -> None
        assert!(val.is_none());
    }

    #[test]
    fn test_extract_json_field_boolean_returns_none() {
        let json = r#"{"enabled":true}"#;
        let val = super::benchmark::extract_json_field(json, "enabled");
        // 't' is not digit/dot
        assert!(val.is_none());
    }

    #[test]
    fn test_extract_json_field_null_returns_none() {
        let json = r#"{"value":null}"#;
        let val = super::benchmark::extract_json_field(json, "value");
        // 'n' is not digit/dot
        assert!(val.is_none());
    }

    #[test]
    fn test_extract_json_field_decimal_leading_dot() {
        // ".5" is a valid float parse
        let json = r#"{"value":.5}"#;
        let val = super::benchmark::extract_json_field(json, "value");
        assert_eq!(val, Some(0.5));
    }

    #[test]
    fn test_extract_json_field_real_ollama_response() {
        // Realistic Ollama API response format
        let json = r#"{"model":"qwen2.5-coder:1.5b","created_at":"2025-01-09T12:00:00Z","response":"def hello()","done":true,"total_duration":1234567890,"load_duration":100000000,"prompt_eval_count":10,"prompt_eval_duration":200000000,"eval_count":25,"eval_duration":900000000}"#;
        let eval_count = super::benchmark::extract_json_field(json, "eval_count");
        assert_eq!(eval_count, Some(25.0));
        let eval_duration = super::benchmark::extract_json_field(json, "eval_duration");
        assert_eq!(eval_duration, Some(900_000_000.0));
        let prompt_eval_duration =
            super::benchmark::extract_json_field(json, "prompt_eval_duration");
        assert_eq!(prompt_eval_duration, Some(200_000_000.0));
    }

    // ========================================================================
    // NEW: format_benchmark_csv edge cases (benchmark.rs lines 69-103)
    // ========================================================================

    #[test]
    fn test_format_benchmark_csv_llama_no_speedup() {
        // llama.cpp present but no speedup calculated
        let bench = BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: None,
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(120.0),
            ollama_ttft_ms: None,
            speedup_vs_llama: None, // No speedup
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.5,
            runs: 30,
        };
        let csv = format_benchmark_csv(&bench);
        // llama row should have empty speedup field
        assert!(csv.contains("llama.cpp,35.00,120.00,,N/A,N/A"));
    }

    #[test]
    fn test_format_benchmark_csv_llama_zero_ttft() {
        // llama.cpp with None ttft defaults to 0.0
        let bench = BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: None,
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: None, // defaults to 0.0
            ollama_ttft_ms: None,
            speedup_vs_llama: Some(25.7),
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.5,
            runs: 30,
        };
        let csv = format_benchmark_csv(&bench);
        assert!(csv.contains("llama.cpp,35.00,0.00,25.70,N/A,N/A"));
    }

    #[test]
    fn test_format_benchmark_csv_ollama_zero_ttft() {
        // Ollama with None ttft defaults to 0.0
        let bench = BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: None,
            ollama_tps: Some(32.0),
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None, // defaults to 0.0
            speedup_vs_llama: None,
            speedup_vs_ollama: Some(37.5),
            apr_tps_stddev: 1.5,
            runs: 30,
        };
        let csv = format_benchmark_csv(&bench);
        assert!(csv.contains("Ollama,32.00,0.00,37.50,N/A,N/A"));
    }

    #[test]
    fn test_format_benchmark_csv_ollama_no_speedup() {
        let bench = BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: None,
            ollama_tps: Some(32.0),
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: Some(150.0),
            speedup_vs_llama: None,
            speedup_vs_ollama: None, // No speedup
            apr_tps_stddev: 1.5,
            runs: 30,
        };
        let csv = format_benchmark_csv(&bench);
        // Ollama row should have empty speedup
        assert!(csv.contains("Ollama,32.00,150.00,,N/A,N/A"));
    }

    #[test]
    fn test_format_benchmark_csv_zero_values() {
        let bench = BenchmarkComparison {
            apr_tps: 0.0,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 0.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 0.0,
            runs: 0,
        };
        let csv = format_benchmark_csv(&bench);
        assert!(csv.contains("APR,0.00,0.00,,0.00,0"));
    }

    #[test]
    fn test_format_benchmark_csv_high_precision_values() {
        let bench = BenchmarkComparison {
            apr_tps: 123.456789,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 0.001,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 0.12345,
            runs: 1,
        };
        let csv = format_benchmark_csv(&bench);
        // Should be formatted to 2 decimal places
        assert!(csv.contains("APR,123.46,0.00,,0.12,1"));
    }

    // ========================================================================
    // NEW: run_chat coverage (demo.rs line 372-382)
    // ========================================================================

    #[test]
    fn test_run_chat_returns_ok_true() {
        let config = ShowcaseConfig::default();
        let result = super::demo::run_chat(&config);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_run_chat_with_all_tiers() {
        for tier in &[
            ModelTier::Tiny,
            ModelTier::Small,
            ModelTier::Medium,
            ModelTier::Large,
        ] {
            let config = ShowcaseConfig::with_tier(*tier);
            let result = super::demo::run_chat(&config);
            assert!(result.is_ok());
        }
    }

    // ========================================================================
    // NEW: run_brick_demo non-inference path (demo.rs lines 1159-1167)
    // ========================================================================

    #[test]
    #[cfg(not(feature = "inference"))]
    fn test_run_brick_demo_no_inference_feature() {
        let config = ShowcaseConfig::default();
        let result = super::demo::run_brick_demo(&config);
        assert!(result.is_ok());
        let demo_result = result.unwrap();
        assert_eq!(demo_result.layers_measured, 0);
        assert!(demo_result.layer_timings_us.is_empty());
        assert!(demo_result.bottleneck.is_none());
        assert_eq!(demo_result.total_us, 0.0);
        assert_eq!(demo_result.tokens_per_sec, 0.0);
        assert!(!demo_result.assertions_passed);
    }

    // ========================================================================
    // NEW: run_cuda_demo non-cuda path (demo.rs lines 789-805)
    // ========================================================================

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_run_cuda_demo_no_cuda_feature() {
        let config = ShowcaseConfig::default();
        let result = run_cuda_demo(&config);
        assert!(result.is_ok());
        let cuda_result = result.unwrap();
        assert_eq!(cuda_result.device_count, 0);
        assert_eq!(cuda_result.device_name, "disabled");
        assert_eq!(cuda_result.total_vram_gb, 0.0);
        assert_eq!(cuda_result.free_vram_gb, 0.0);
        assert!(!cuda_result.cuda_available);
        assert!(!cuda_result.graph_capture_available);
        assert!((cuda_result.graph_speedup - 1.0).abs() < f64::EPSILON);
        assert!(!cuda_result.dp4a_available);
        assert_eq!(cuda_result.dp4a_arithmetic_intensity, 0.0);
    }

    // ========================================================================
    // NEW: run_zram_demo non-zram path (demo.rs lines 571-584)
    // ========================================================================

    #[test]
    #[cfg(not(feature = "zram"))]
    fn test_run_zram_demo_no_zram_feature() {
        let config = ShowcaseConfig::default();
        let result = run_zram_demo(&config);
        assert!(result.is_ok());
        let zram_result = result.unwrap();
        assert_eq!(zram_result.lz4_ratio, 0.0);
        assert_eq!(zram_result.zstd_ratio, 0.0);
        assert_eq!(zram_result.zero_page_gbps, 0.0);
        assert_eq!(zram_result.lz4_gbps, 0.0);
        assert_eq!(zram_result.simd_backend, "disabled");
        assert_eq!(zram_result.context_extension, 0.0);
    }

    // ========================================================================
    // NEW: print_header coverage (validation.rs lines 8-37)
    // ========================================================================

    #[test]
    fn test_print_header_tiny() {
        super::validation::print_header(ModelTier::Tiny);
    }

    #[test]
    fn test_print_header_small() {
        super::validation::print_header(ModelTier::Small);
    }

    #[test]
    fn test_print_header_medium() {
        super::validation::print_header(ModelTier::Medium);
    }

    #[test]
    fn test_print_header_large() {
        super::validation::print_header(ModelTier::Large);
    }

    // ========================================================================
    // NEW: print_summary coverage (validation.rs lines 39-96)
    // ========================================================================

    #[test]
    fn test_print_summary_all_pass() {
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
            zram_demo: Some(ZramDemoResult {
                lz4_ratio: 2.5,
                zstd_ratio: 3.2,
                zero_page_gbps: 175.0,
                lz4_gbps: 3.5,
                simd_backend: "Avx2".to_string(),
                context_extension: 2.5,
            }),
            cuda_demo: Some(CudaDemoResult {
                device_count: 1,
                device_name: "RTX 4090".to_string(),
                total_vram_gb: 24.0,
                free_vram_gb: 20.0,
                cuda_available: true,
                graph_capture_available: true,
                graph_speedup: 70.0,
                dp4a_available: true,
                dp4a_arithmetic_intensity: 1.78,
            }),
            brick_demo: None,
        };
        let config = ShowcaseConfig::default();
        // Should not panic
        super::validation::print_summary(&results, &config);
    }

    #[test]
    fn test_print_summary_all_fail() {
        let results = ShowcaseResults::default();
        let config = ShowcaseConfig::default();
        super::validation::print_summary(&results, &config);
    }

    #[test]
    fn test_print_summary_no_benchmark() {
        let results = ShowcaseResults {
            import: true,
            gguf_inference: false,
            convert: true,
            apr_inference: false,
            benchmark: None,
            visualize: false,
            chat: false,
            zram_demo: None,
            cuda_demo: None,
            brick_demo: None,
        };
        let config = ShowcaseConfig::default();
        super::validation::print_summary(&results, &config);
    }

    #[test]
    fn test_print_summary_benchmark_below_threshold() {
        // Exercises the speedup check branches where pass = false
        let results = ShowcaseResults {
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            benchmark: Some(BenchmarkComparison {
                apr_tps: 36.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(32.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(150.0),
                speedup_vs_llama: Some(2.9),   // Below 25%
                speedup_vs_ollama: Some(12.5), // Below 25%
                apr_tps_stddev: 1.5,
                runs: 30,
            }),
            visualize: true,
            chat: true,
            zram_demo: None,
            cuda_demo: None,
            brick_demo: None,
        };
        let config = ShowcaseConfig::default();
        super::validation::print_summary(&results, &config);
    }

    #[test]
    fn test_print_summary_zram_below_thresholds() {
        // Exercises ZRAM summary with failing checks
        let results = ShowcaseResults {
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            benchmark: None,
            visualize: false,
            chat: false,
            zram_demo: Some(ZramDemoResult {
                lz4_ratio: 1.5, // Below 2.0 threshold
                zstd_ratio: 1.8,
                zero_page_gbps: 100.0, // Below 150.0 threshold
                lz4_gbps: 2.0,
                simd_backend: "Scalar".to_string(),
                context_extension: 1.5, // Below 2.0 threshold
            }),
            cuda_demo: None,
            brick_demo: None,
        };
        let config = ShowcaseConfig::default();
        super::validation::print_summary(&results, &config);
    }

    #[test]
    fn test_print_summary_cuda_no_device() {
        let results = ShowcaseResults {
            import: false,
            gguf_inference: false,
            convert: false,
            apr_inference: false,
            benchmark: None,
            visualize: false,
            chat: false,
            zram_demo: None,
            cuda_demo: Some(CudaDemoResult {
                device_count: 0,
                device_name: "N/A".to_string(),
                total_vram_gb: 0.0,
                free_vram_gb: 0.0,
                cuda_available: false,
                graph_capture_available: false,
                graph_speedup: 1.0,
                dp4a_available: false,
                dp4a_arithmetic_intensity: 0.0,
            }),
            brick_demo: None,
        };
        let config = ShowcaseConfig::default();
        super::validation::print_summary(&results, &config);
    }

    // ========================================================================
    // NEW: mod.rs run() coverage - no step path (mod.rs lines 49-67)
    // ========================================================================

    #[test]
    fn test_run_no_step_no_auto_verify() {
        // This path prints available steps and returns Ok(())
        let config = ShowcaseConfig {
            step: None,
            auto_verify: false,
            ..Default::default()
        };
        let result = super::run(&config);
        assert!(result.is_ok());
    }

    // ========================================================================
    // NEW: validate_falsification single Benchmark step (validation.rs)
    // ========================================================================

    #[test]
    fn test_falsification_single_benchmark_step_no_data() {
        // Benchmark step explicitly requested but no data collected
        let results = ShowcaseResults::default();
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::Benchmark),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_err());
    }

    #[test]
    fn test_falsification_single_benchmark_step_valid() {
        let results = ShowcaseResults {
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
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::Benchmark),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    #[test]
    fn test_falsification_single_visualize_step_passes() {
        // Visualize step is not validated strictly - should pass even with defaults
        let results = ShowcaseResults::default();
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::Visualize),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    #[test]
    fn test_falsification_single_chat_step_passes() {
        let results = ShowcaseResults::default();
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::Chat),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    // ========================================================================
    // NEW: validate_falsification --step all (treated like full run)
    // ========================================================================

    #[test]
    fn test_falsification_step_all_passes() {
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
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::All),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_ok());
    }

    #[test]
    fn test_falsification_step_all_fails_with_failures() {
        let results = ShowcaseResults {
            import: false,
            ..Default::default()
        };
        let config = ShowcaseConfig {
            step: Some(ShowcaseStep::All),
            ..Default::default()
        };
        assert!(super::validation::validate_falsification(&results, &config).is_err());
    }

    // ========================================================================
    // NEW: validate_falsification both baselines fail (Points 41+42)
    // ========================================================================

    #[test]
    fn test_falsification_both_speedups_below_25() {
        let results = ShowcaseResults {
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            benchmark: Some(BenchmarkComparison {
                apr_tps: 37.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(36.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(80.0),
                ollama_ttft_ms: Some(82.0),
                speedup_vs_llama: Some(5.7),  // Below 25%
                speedup_vs_ollama: Some(2.8), // Below 25%
                apr_tps_stddev: 1.0,
                runs: 30,
            }),
            ..Default::default()
        };
        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    #[test]
    fn test_falsification_ollama_speedup_below_25_llama_passes() {
        let results = ShowcaseResults {
            import: true,
            gguf_inference: true,
            convert: true,
            apr_inference: true,
            benchmark: Some(BenchmarkComparison {
                apr_tps: 44.0,
                llama_cpp_tps: Some(35.0),
                ollama_tps: Some(42.0),
                apr_ttft_ms: 78.0,
                llama_cpp_ttft_ms: Some(120.0),
                ollama_ttft_ms: Some(80.0),
                speedup_vs_llama: Some(25.7), // Passes
                speedup_vs_ollama: Some(4.8), // Fails
                apr_tps_stddev: 1.5,
                runs: 30,
            }),
            ..Default::default()
        };
        // Ollama below threshold means failure
        assert!(validate_falsification(&results, &full_run_config()).is_err());
    }

    // ========================================================================
    // NEW: run_benchmark non-inference path (benchmark.rs lines 402-445)
    // ========================================================================

    #[test]
    #[cfg(not(feature = "inference"))]
    fn test_run_benchmark_no_inference_no_baselines() {
        let config = ShowcaseConfig {
            baselines: vec![], // No baselines to avoid system calls
            ..Default::default()
        };
        let result = super::benchmark::run_benchmark(&config);
        assert!(result.is_ok());
        let bench = result.unwrap();
        // Non-inference path uses simulated values around 44.0 tok/s
        assert!(bench.apr_tps > 0.0);
        assert!(bench.apr_ttft_ms > 0.0);
        assert_eq!(bench.apr_tps_stddev, 2.0);
        assert_eq!(bench.runs, 30);
        assert!(bench.llama_cpp_tps.is_none());
        assert!(bench.ollama_tps.is_none());
    }

    // ========================================================================
    // NEW: run_visualize non-visualization path (demo.rs lines 67-87)
    // ========================================================================

    #[test]
    #[cfg(not(feature = "visualization"))]
    fn test_run_visualize_no_visualization_feature() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShowcaseConfig {
            model_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let result = super::demo::run_visualize(&config, None);
        assert!(result.is_ok());
        assert!(result.unwrap());
        // Should have created an SVG file
        let svg_path = temp_dir.path().join("showcase-performance.svg");
        assert!(svg_path.exists());
        let content = std::fs::read_to_string(&svg_path).unwrap();
        assert!(content.contains("<svg"));
        assert!(content.contains("APR Inference"));
    }

    #[test]
    #[cfg(not(feature = "visualization"))]
    fn test_run_visualize_with_benchmark_data() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShowcaseConfig {
            model_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
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
        let result = super::demo::run_visualize(&config, Some(&bench));
        assert!(result.is_ok());
    }

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
}
