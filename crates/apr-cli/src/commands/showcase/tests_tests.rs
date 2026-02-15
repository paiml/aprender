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

include!("tests_tests_part_02.rs");
include!("tests_tests_part_03.rs");
include!("tests_tests_part_04.rs");
include!("tests_tests_part_05.rs");
include!("tests_tests_part_06.rs");
include!("tests_tests_part_07.rs");
