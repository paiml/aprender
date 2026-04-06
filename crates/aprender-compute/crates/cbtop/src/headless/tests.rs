use super::*;

#[test]
fn test_system_info_detect() {
    let info = SystemInfo::detect();
    assert!(info.cores > 0);
}

#[test]
fn test_latency_stats_calculation() {
    let latencies = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = HeadlessBenchmark::calculate_latency_stats(&latencies);
    assert!((stats.mean - 3.0).abs() < 0.01);
    assert!((stats.min - 1.0).abs() < 0.01);
    assert!((stats.max - 5.0).abs() < 0.01);
}

#[test]
fn test_benchmark_result_json_format() {
    let result = BenchmarkResult {
        version: "0.1.0".to_string(),
        timestamp: "2026-01-11T10:00:00Z".to_string(),
        duration_secs: 5.0,
        system: SystemInfo {
            cpu: "Test CPU".to_string(),
            cores: 4,
            memory_gb: 16,
            gpu: None,
            cpu_governor: None,
        },
        benchmark: BenchmarkConfig {
            backend: "Simd".to_string(),
            workload: "Gemm".to_string(),
            size: 1000000,
            iterations: 500,
        },
        results: BenchmarkResults {
            gflops: 25.0,
            throughput_ops_sec: 1000.0,
            latency_ms: LatencyStats {
                mean: 1.0,
                min: 0.5,
                max: 2.0,
                p50: 0.9,
                p95: 1.5,
                p99: 1.8,
                cv_percent: 10.0,
            },
        },
        score: ScoreInfo {
            total: 85,
            grade: "B".to_string(),
            performance: 35,
            efficiency: 20,
            correctness: 20,
            stability: 10,
        },
        warnings: vec![],
    };

    let json = result.format(OutputFormat::Json);
    assert!(json.contains("\"gflops\": 25.0"));
    assert!(json.contains("\"total\": 85"));
}

#[test]
fn test_regression_detection() {
    let baseline = BenchmarkResult {
        version: "0.1.0".to_string(),
        timestamp: "2026-01-11T10:00:00Z".to_string(),
        duration_secs: 5.0,
        system: SystemInfo {
            cpu: "Test".to_string(),
            cores: 4,
            memory_gb: 16,
            gpu: None,
            cpu_governor: None,
        },
        benchmark: BenchmarkConfig {
            backend: "Simd".to_string(),
            workload: "Gemm".to_string(),
            size: 1000000,
            iterations: 500,
        },
        results: BenchmarkResults {
            gflops: 25.0,
            throughput_ops_sec: 1000.0,
            latency_ms: LatencyStats {
                mean: 1.0,
                min: 0.5,
                max: 2.0,
                p50: 0.9,
                p95: 1.5,
                p99: 1.8,
                cv_percent: 10.0,
            },
        },
        score: ScoreInfo {
            total: 85,
            grade: "B".to_string(),
            performance: 35,
            efficiency: 20,
            correctness: 20,
            stability: 10,
        },
        warnings: vec![],
    };

    let mut current = baseline.clone();
    current.results.gflops = 22.0; // 12% regression

    let regression = current.check_regression(&baseline, 5.0);
    assert!(regression.is_regression);
    assert_eq!(regression.status, "REGRESSION");
    assert!(regression.change_percent < -10.0);
}

#[test]
fn test_headless_benchmark_short_run() {
    let benchmark = HeadlessBenchmark::new(
        ComputeBackend::Simd,
        WorkloadType::Gemm,
        10000,
        Duration::from_millis(100),
    );

    let result = benchmark.run().unwrap();
    assert!(result.results.gflops > 0.0);
    assert!(result.benchmark.iterations > 0);
}

// HL-007: Library API tests
#[test]
fn test_benchmark_builder_defaults() {
    let benchmark = Benchmark::builder().build().unwrap();

    let result = benchmark.run().unwrap();
    assert!(result.results.gflops > 0.0);
    assert_eq!(result.benchmark.workload, "Gemm");
    assert_eq!(result.benchmark.backend, "Simd");
}

#[test]
fn test_benchmark_builder_with_options() {
    let benchmark = Benchmark::builder()
        .workload("elementwise")
        .size(10000)
        .duration_secs(1)
        .backend_str("simd")
        .build()
        .unwrap();

    let result = benchmark.run().unwrap();
    assert!(result.results.gflops > 0.0);
    assert_eq!(result.benchmark.workload, "Elementwise");
    assert_eq!(result.benchmark.size, 10000);
}

#[test]
fn test_benchmark_with_baseline() {
    let benchmark = Benchmark::builder()
        .workload("gemm")
        .size(10000)
        .duration(Duration::from_millis(100))
        .build()
        .unwrap();

    // Run first benchmark as baseline
    let baseline = benchmark.run().unwrap();

    // Run with baseline comparison
    let (result, regression) = benchmark.run_with_baseline(&baseline, 50.0).unwrap();

    // With 50% threshold, small variations should pass
    assert!(!regression.is_regression || regression.change_percent.abs() > 50.0);
    assert!(result.results.gflops > 0.0);
}
