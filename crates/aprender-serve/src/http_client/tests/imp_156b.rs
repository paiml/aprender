
/// IMP-156b: Test latency comparison
#[test]
fn test_imp_156b_latency_comparison() {
    let realizar = LatencyPercentiles {
        p50_ms: 12.0,
        p95_ms: 25.0,
        p99_ms: 45.0,
        min_ms: 8.0,
        max_ms: 60.0,
        mean_ms: 15.0,
        stddev_ms: 8.0,
    };
    let reference = LatencyPercentiles {
        p50_ms: 10.0,
        p95_ms: 20.0,
        p99_ms: 40.0,
        min_ms: 7.0,
        max_ms: 55.0,
        mean_ms: 12.0,
        stddev_ms: 6.0,
    };

    let comparison = LatencyComparison::new(realizar, reference);

    assert!(
        (comparison.p50_gap_percent - 20.0).abs() < 1.0,
        "IMP-156b: P50 gap should be ~20%"
    );
    assert!(
        comparison.parity_achieved(),
        "IMP-156b: Should be at parity (within 20%)"
    );

    println!("\nIMP-156b: Latency Comparison:");
    println!(
        "  Realizar P50: {:.1}ms",
        comparison.realizar_percentiles.p50_ms
    );
    println!(
        "  Reference P50: {:.1}ms",
        comparison.reference_percentiles.p50_ms
    );
    println!("  P50 gap: {:+.1}%", comparison.p50_gap_percent);
    println!("  P99 gap: {:+.1}%", comparison.p99_gap_percent);
    println!("  Parity: {}", comparison.parity_achieved());
}

/// IMP-156c: Real-world latency comparison vs llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_156c_latency_vs_llamacpp() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Count from 1 to 5:".to_string(),
        max_tokens: 20,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect multiple samples for percentile calculation
    let mut latencies_ms = Vec::new();
    for _ in 0..10 {
        let start = std::time::Instant::now();
        let _ = client.llamacpp_completion("http://127.0.0.1:8082", &request);
        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let percentiles = LatencyPercentiles::from_samples(&latencies_ms);

    println!("\nIMP-156c: llama.cpp Latency Percentiles:");
    println!("  P50: {:.2}ms", percentiles.p50_ms);
    println!("  P95: {:.2}ms", percentiles.p95_ms);
    println!("  P99: {:.2}ms", percentiles.p99_ms);
    println!("  Tail ratio: {:.2}x", percentiles.tail_latency_ratio());
}

/// IMP-156d: Latency SLA gate
#[derive(Debug, Clone)]
pub struct LatencySLAGate {
    pub p50_limit_ms: f64,
    pub p99_limit_ms: f64,
    pub measured_p50_ms: f64,
    pub measured_p99_ms: f64,
    pub p50_pass: bool,
    pub p99_pass: bool,
    pub overall_pass: bool,
}

impl LatencySLAGate {
    pub fn new(p50_limit_ms: f64, p99_limit_ms: f64, measured: &LatencyPercentiles) -> Self {
        let p50_pass = measured.p50_ms <= p50_limit_ms;
        let p99_pass = measured.p99_ms <= p99_limit_ms;
        Self {
            p50_limit_ms,
            p99_limit_ms,
            measured_p50_ms: measured.p50_ms,
            measured_p99_ms: measured.p99_ms,
            p50_pass,
            p99_pass,
            overall_pass: p50_pass && p99_pass,
        }
    }
}

/// IMP-156d: Test latency SLA gate
#[test]
fn test_imp_156d_latency_sla() {
    let good_latency = LatencyPercentiles {
        p50_ms: 8.0,
        p95_ms: 15.0,
        p99_ms: 25.0,
        min_ms: 5.0,
        max_ms: 40.0,
        mean_ms: 10.0,
        stddev_ms: 5.0,
    };

    // SLA: P50 < 10ms, P99 < 30ms
    let gate = LatencySLAGate::new(10.0, 30.0, &good_latency);
    assert!(gate.overall_pass, "IMP-156d: Good latency should pass SLA");

    let bad_latency = LatencyPercentiles {
        p50_ms: 15.0,
        p95_ms: 40.0,
        p99_ms: 80.0,
        min_ms: 10.0,
        max_ms: 100.0,
        mean_ms: 20.0,
        stddev_ms: 15.0,
    };

    let fail_gate = LatencySLAGate::new(10.0, 30.0, &bad_latency);
    assert!(
        !fail_gate.overall_pass,
        "IMP-156d: Bad latency should fail SLA"
    );
    assert!(!fail_gate.p50_pass, "IMP-156d: P50 15ms > 10ms limit");
    assert!(!fail_gate.p99_pass, "IMP-156d: P99 80ms > 30ms limit");

    println!("\nIMP-156d: Latency SLA Gate:");
    println!(
        "  Good: P50={:.0}ms (limit {:.0}), P99={:.0}ms (limit {:.0}) -> {}",
        gate.measured_p50_ms,
        gate.p50_limit_ms,
        gate.measured_p99_ms,
        gate.p99_limit_ms,
        if gate.overall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Bad: P50={:.0}ms (limit {:.0}), P99={:.0}ms (limit {:.0}) -> {}",
        fail_gate.measured_p50_ms,
        fail_gate.p50_limit_ms,
        fail_gate.measured_p99_ms,
        fail_gate.p99_limit_ms,
        if fail_gate.overall_pass {
            "PASS"
        } else {
            "FAIL"
        }
    );
}

// =========================================================================
// IMP-157: Environment Metadata Capture (EXTREME TDD)
// Per spec QA-033: Environment metadata captured per Vitek & Kalibera [8]
// =========================================================================

/// IMP-157a: System environment metadata
#[derive(Debug, Clone)]
pub struct EnvironmentMetadata {
    pub os_name: String,
    pub os_version: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub rust_version: String,
    pub realizar_version: String,
    pub timestamp: String,
    pub hostname: String,
}

impl EnvironmentMetadata {
    pub fn capture() -> Self {
        Self {
            os_name: std::env::consts::OS.to_string(),
            os_version: std::env::consts::ARCH.to_string(),
            cpu_model: "Unknown".to_string(), // Would need sysinfo crate
            cpu_cores: std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1),
            memory_gb: 0.0, // Would need sysinfo crate
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            realizar_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            hostname: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::json!({
            "os": {
                "name": self.os_name,
                "version": self.os_version
            },
            "cpu": {
                "model": self.cpu_model,
                "cores": self.cpu_cores
            },
            "memory_gb": self.memory_gb,
            "software": {
                "rust_version": self.rust_version,
                "realizar_version": self.realizar_version
            },
            "timestamp": self.timestamp,
            "hostname": self.hostname
        })
        .to_string()
    }
}

/// IMP-157a: Test environment metadata capture
#[test]
fn test_imp_157a_environment_capture() {
    let env = EnvironmentMetadata::capture();

    assert!(
        !env.os_name.is_empty(),
        "IMP-157a: OS name should not be empty"
    );
    assert!(env.cpu_cores > 0, "IMP-157a: CPU cores should be > 0");
    assert!(
        !env.realizar_version.is_empty(),
        "IMP-157a: Version should not be empty"
    );

    let json = env.to_json();
    assert!(json.contains("os"), "IMP-157a: JSON should have os field");
    assert!(json.contains("cpu"), "IMP-157a: JSON should have cpu field");

    println!("\nIMP-157a: Environment Metadata:");
    println!("  OS: {} {}", env.os_name, env.os_version);
    println!("  CPU cores: {}", env.cpu_cores);
    println!("  Rust: {}", env.rust_version);
    println!("  Realizar: {}", env.realizar_version);
    println!("  Timestamp: {}", env.timestamp);
}

/// IMP-157b: Benchmark configuration metadata
#[derive(Debug, Clone)]
pub struct BenchmarkMetadata {
    pub benchmark_name: String,
    pub model_path: String,
    pub model_size_mb: f64,
    pub quantization: String,
    pub batch_size: usize,
    pub max_tokens: usize,
    pub cv_threshold: f64,
    pub warmup_iterations: usize,
}

impl BenchmarkMetadata {
    pub fn new(name: &str) -> Self {
        Self {
            benchmark_name: name.to_string(),
            model_path: String::new(),
            model_size_mb: 0.0,
            quantization: "Q4_K".to_string(),
            batch_size: 1,
            max_tokens: 100,
            cv_threshold: 0.10,
            warmup_iterations: 3,
        }
    }

    pub fn with_model(mut self, path: &str, size_mb: f64, quant: &str) -> Self {
        self.model_path = path.to_string();
        self.model_size_mb = size_mb;
        self.quantization = quant.to_string();
        self
    }
}

/// IMP-157b: Test benchmark metadata
#[test]
fn test_imp_157b_benchmark_metadata() {
    let meta = BenchmarkMetadata::new("performance_parity").with_model(
        "phi-2-q4k.gguf",
        1.6 * 1024.0,
        "Q4_K_M",
    );

    assert_eq!(meta.benchmark_name, "performance_parity");
    assert!(
        meta.model_size_mb > 1000.0,
        "IMP-157b: Model should be > 1GB"
    );
    assert_eq!(meta.quantization, "Q4_K_M");

    println!("\nIMP-157b: Benchmark Metadata:");
    println!("  Name: {}", meta.benchmark_name);
    println!(
        "  Model: {} ({:.1} MB)",
        meta.model_path, meta.model_size_mb
    );
    println!("  Quantization: {}", meta.quantization);
    println!("  Batch size: {}", meta.batch_size);
    println!("  CV threshold: {:.0}%", meta.cv_threshold * 100.0);
}

/// IMP-157c: Full benchmark result with metadata
#[derive(Debug, Clone)]
pub struct FullBenchmarkResult {
    pub environment: EnvironmentMetadata,
    pub benchmark: BenchmarkMetadata,
    pub throughput_tps: f64,
    pub latency: LatencyPercentiles,
    pub iterations: usize,
    pub cv_achieved: f64,
}

impl FullBenchmarkResult {
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "environment": serde_json::from_str::<serde_json::Value>(&self.environment.to_json()).unwrap_or_default(),
            "benchmark": {
                "name": self.benchmark.benchmark_name,
                "model_path": self.benchmark.model_path,
                "model_size_mb": self.benchmark.model_size_mb,
                "quantization": self.benchmark.quantization
            },
            "results": {
                "throughput_tps": self.throughput_tps,
                "latency_p50_ms": self.latency.p50_ms,
                "latency_p95_ms": self.latency.p95_ms,
                "latency_p99_ms": self.latency.p99_ms,
                "iterations": self.iterations,
                "cv_achieved": self.cv_achieved
            }
        }).to_string()
    }
}

/// IMP-157c: Test full benchmark result
#[test]
fn test_imp_157c_full_benchmark_result() {
    let result = FullBenchmarkResult {
        environment: EnvironmentMetadata::capture(),
        benchmark: BenchmarkMetadata::new("parity_test"),
        throughput_tps: 150.0,
        latency: LatencyPercentiles {
            p50_ms: 10.0,
            p95_ms: 20.0,
            p99_ms: 35.0,
            min_ms: 8.0,
            max_ms: 50.0,
            mean_ms: 12.0,
            stddev_ms: 5.0,
        },
        iterations: 25,
        cv_achieved: 0.08,
    };

    let json = result.to_json();
    assert!(
        json.contains("environment"),
        "IMP-157c: Should have environment"
    );
    assert!(
        json.contains("throughput_tps"),
        "IMP-157c: Should have throughput"
    );
    assert!(
        json.contains("latency_p50_ms"),
        "IMP-157c: Should have latency"
    );

    println!("\nIMP-157c: Full Benchmark Result JSON:");
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::from_str::<serde_json::Value>(&json).expect("test")
        )
        .unwrap_or(json)
    );
}

/// IMP-157d: Reproducibility hash
#[derive(Debug, Clone)]
pub struct ReproducibilityHash {
    pub config_hash: String,
    pub environment_hash: String,
    pub combined_hash: String,
}

impl ReproducibilityHash {
    pub fn compute(env: &EnvironmentMetadata, bench: &BenchmarkMetadata) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut config_hasher = DefaultHasher::new();
        bench.benchmark_name.hash(&mut config_hasher);
        bench.quantization.hash(&mut config_hasher);
        bench.max_tokens.hash(&mut config_hasher);
        let config_hash = format!("{:016x}", config_hasher.finish());

        let mut env_hasher = DefaultHasher::new();
        env.os_name.hash(&mut env_hasher);
        env.cpu_cores.hash(&mut env_hasher);
        env.rust_version.hash(&mut env_hasher);
        let env_hash = format!("{:016x}", env_hasher.finish());

        let mut combined_hasher = DefaultHasher::new();
        config_hash.hash(&mut combined_hasher);
        env_hash.hash(&mut combined_hasher);
        let combined = format!("{:016x}", combined_hasher.finish());

        Self {
            config_hash,
            environment_hash: env_hash,
            combined_hash: combined,
        }
    }
}
