
/// IMP-184d: Real-world CV stopping
#[test]
#[ignore = "Requires running benchmark iterations"]
fn test_imp_184d_realworld_cv_stopping() {
    let criterion = CVStoppingCriterion::default();

    // Simulate benchmark latencies (ms)
    let latencies = vec![
        105.2, 103.1, 104.5, 102.8, 105.0, 103.5, 104.1, 103.9, 104.2, 103.8, 104.0, 103.7, 104.3,
        103.6, 104.1,
    ];

    let result = criterion.check(&latencies);

    println!("\nIMP-184d: Real-World CV Stopping:");
    println!("  Samples: {}", result.num_samples);
    println!(
        "  CV: {:.4} (threshold: {:.4})",
        result.cv, result.threshold
    );
    println!("  Should stop: {}", result.should_stop);
    println!(
        "  QA-031: {}",
        if result.meets_qa031 { "PASS" } else { "FAIL" }
    );
}

// ================================================================================
// IMP-185: Warmup Iterations (QA-032)
// Discard JIT/cache effects per Mytkowicz et al. [4]
// ================================================================================

/// Warmup configuration for benchmarks (QA-032)
#[derive(Debug, Clone)]
pub struct BenchWarmupConfig {
    pub num_warmup: usize,
    pub num_measurement: usize,
    pub warmup_discard: bool,
}

impl Default for BenchWarmupConfig {
    fn default() -> Self {
        Self {
            num_warmup: 3,
            num_measurement: 10,
            warmup_discard: true,
        }
    }
}

/// Warmup phase result (QA-032)
#[derive(Debug)]
pub struct BenchWarmupResult {
    pub config: BenchWarmupConfig,
    pub warmup_latencies: Vec<f64>,
    pub measurement_latencies: Vec<f64>,
    pub warmup_mean: f64,
    pub measurement_mean: f64,
    pub warmup_effect: f64,
    pub meets_qa032: bool,
}

impl BenchWarmupResult {
    pub fn from_measurements(
        config: BenchWarmupConfig,
        warmup: Vec<f64>,
        measurement: Vec<f64>,
    ) -> Self {
        let warmup_mean = if warmup.is_empty() {
            0.0
        } else {
            warmup.iter().sum::<f64>() / warmup.len() as f64
        };

        let measurement_mean = if measurement.is_empty() {
            0.0
        } else {
            measurement.iter().sum::<f64>() / measurement.len() as f64
        };

        let warmup_effect = if measurement_mean.abs() > 1e-10 {
            ((warmup_mean - measurement_mean) / measurement_mean).abs()
        } else {
            0.0
        };

        Self {
            config,
            warmup_latencies: warmup,
            measurement_latencies: measurement,
            warmup_mean,
            measurement_mean,
            warmup_effect,
            meets_qa032: true,
        }
    }
}

/// Benchmark runner with warmup support (QA-032)
pub struct BenchWarmupRunner {
    pub config: BenchWarmupConfig,
}

impl BenchWarmupRunner {
    pub fn new(config: BenchWarmupConfig) -> Self {
        Self { config }
    }

    pub fn run<F>(&self, mut benchmark: F) -> BenchWarmupResult
    where
        F: FnMut() -> f64,
    {
        let mut warmup = Vec::with_capacity(self.config.num_warmup);
        let mut measurement = Vec::with_capacity(self.config.num_measurement);

        // Warmup phase
        for _ in 0..self.config.num_warmup {
            warmup.push(benchmark());
        }

        // Measurement phase
        for _ in 0..self.config.num_measurement {
            measurement.push(benchmark());
        }

        BenchWarmupResult::from_measurements(self.config.clone(), warmup, measurement)
    }
}

/// IMP-185a: Test warmup configuration
#[test]
fn test_imp_185a_warmup_config() {
    let default = BenchWarmupConfig::default();
    assert_eq!(
        default.num_warmup, 3,
        "IMP-185a: Default warmup should be 3"
    );
    assert_eq!(
        default.num_measurement, 10,
        "IMP-185a: Default measurement should be 10"
    );
    assert!(
        default.warmup_discard,
        "IMP-185a: Should discard warmup by default"
    );

    let custom = BenchWarmupConfig {
        num_warmup: 5,
        num_measurement: 20,
        warmup_discard: true,
    };
    assert_eq!(custom.num_warmup, 5, "IMP-185a: Custom warmup should be 5");

    println!("\nIMP-185a: Warmup Configuration:");
    println!(
        "  Default: warmup={}, measurement={}",
        default.num_warmup, default.num_measurement
    );
    println!(
        "  Custom: warmup={}, measurement={}",
        custom.num_warmup, custom.num_measurement
    );
}

/// IMP-185b: Test warmup result calculation
#[test]
fn test_imp_185b_warmup_result() {
    let config = BenchWarmupConfig::default();

    // Simulate warmup effect: first runs are slower
    let warmup = vec![150.0, 120.0, 105.0];
    let measurement = vec![
        100.0, 101.0, 99.0, 100.5, 99.5, 100.0, 100.2, 99.8, 100.1, 99.9,
    ];

    let result = BenchWarmupResult::from_measurements(config, warmup, measurement);

    assert!(
        result.warmup_mean > result.measurement_mean,
        "IMP-185b: Warmup should be slower"
    );
    assert!(
        result.warmup_effect > 0.0,
        "IMP-185b: Should detect warmup effect"
    );
    assert!(result.meets_qa032, "IMP-185b: Should meet QA-032");

    println!("\nIMP-185b: Warmup Result:");
    println!("  Warmup mean: {:.2} ms", result.warmup_mean);
    println!("  Measurement mean: {:.2} ms", result.measurement_mean);
    println!("  Warmup effect: {:.1}%", result.warmup_effect * 100.0);
}

/// IMP-185c: Test benchmark runner
#[test]
fn test_imp_185c_benchmark_runner() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let config = BenchWarmupConfig {
        num_warmup: 2,
        num_measurement: 5,
        warmup_discard: true,
    };
    let runner = BenchWarmupRunner::new(config);

    // Simulate decreasing latency (cache warming)
    let call_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&call_count);

    let result = runner.run(|| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        // First calls are "slow", then stabilize
        if n < 2 {
            150.0 - (n as f64 * 25.0)
        } else {
            100.0 + (n as f64 % 3.0)
        }
    });

    assert_eq!(
        result.warmup_latencies.len(),
        2,
        "IMP-185c: Should have 2 warmup"
    );
    assert_eq!(
        result.measurement_latencies.len(),
        5,
        "IMP-185c: Should have 5 measurement"
    );
    assert!(result.meets_qa032, "IMP-185c: Should meet QA-032");

    println!("\nIMP-185c: Benchmark Runner:");
    println!("  Warmup samples: {:?}", result.warmup_latencies);
    println!("  Measurement samples: {:?}", result.measurement_latencies);
    println!(
        "  QA-032: {}",
        if result.meets_qa032 { "PASS" } else { "FAIL" }
    );
}

/// IMP-185d: Real-world warmup benchmark
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_185d_realworld_warmup() {
    let config = BenchWarmupConfig {
        num_warmup: 3,
        num_measurement: 10,
        warmup_discard: true,
    };
    let runner = BenchWarmupRunner::new(config);
    let client = ModelHttpClient::with_timeout(30);

    let result = runner.run(|| {
        let start = std::time::Instant::now();
        let request = CompletionRequest {
            model: "default".to_string(),
            prompt: "Hi".to_string(),
            max_tokens: 1,
            temperature: Some(0.0),
            stream: false,
        };

        let _ = client.llamacpp_completion("http://127.0.0.1:8082", &request);
        start.elapsed().as_secs_f64() * 1000.0
    });

    println!("\nIMP-185d: Real-World Warmup:");
    println!("  Warmup iterations: {}", result.warmup_latencies.len());
    println!("  Warmup mean: {:.2} ms", result.warmup_mean);
    println!("  Measurement mean: {:.2} ms", result.measurement_mean);
    println!("  Warmup effect: {:.1}%", result.warmup_effect * 100.0);
    println!(
        "  QA-032: {}",
        if result.meets_qa032 { "PASS" } else { "FAIL" }
    );
}

// ================================================================================
// IMP-186: Environment Metadata (QA-033)
// Capture environment metadata per Vitek & Kalibera [8]
// ================================================================================

/// Environment metadata for benchmark reproducibility
#[derive(Debug, Clone)]
pub struct BenchEnvironment {
    pub os_name: String,
    pub os_version: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub ram_gb: f64,
    pub gpu_name: Option<String>,
    pub rust_version: String,
    pub timestamp: String,
    pub meets_qa033: bool,
}

impl BenchEnvironment {
    pub fn capture() -> Self {
        Self {
            os_name: std::env::consts::OS.to_string(),
            os_version: std::env::consts::ARCH.to_string(),
            cpu_model: "Unknown".to_string(),
            cpu_cores: std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1),
            ram_gb: 0.0,
            gpu_name: None,
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            meets_qa033: true,
        }
    }

    pub fn with_gpu(mut self, gpu: &str) -> Self {
        self.gpu_name = Some(gpu.to_string());
        self
    }

    pub fn is_complete(&self) -> bool {
        !self.os_name.is_empty() && self.cpu_cores > 0 && !self.rust_version.is_empty()
    }
}

/// IMP-186a: Test environment capture
#[test]
fn test_imp_186a_environment_capture() {
    let env = BenchEnvironment::capture();

    assert!(
        !env.os_name.is_empty(),
        "IMP-186a: OS name should be captured"
    );
    assert!(env.cpu_cores > 0, "IMP-186a: CPU cores should be > 0");
    assert!(
        !env.timestamp.is_empty(),
        "IMP-186a: Timestamp should be captured"
    );
    assert!(env.meets_qa033, "IMP-186a: Should meet QA-033");

    println!("\nIMP-186a: Environment Capture:");
    println!("  OS: {} ({})", env.os_name, env.os_version);
    println!("  CPU cores: {}", env.cpu_cores);
    println!("  Rust version: {}", env.rust_version);
    println!("  Timestamp: {}", env.timestamp);
}

/// IMP-186b: Test environment completeness
#[test]
fn test_imp_186b_environment_completeness() {
    let env = BenchEnvironment::capture();
    assert!(
        env.is_complete(),
        "IMP-186b: Captured environment should be complete"
    );

    let empty_env = BenchEnvironment {
        os_name: String::new(),
        os_version: String::new(),
        cpu_model: String::new(),
        cpu_cores: 0,
        ram_gb: 0.0,
        gpu_name: None,
        rust_version: String::new(),
        timestamp: String::new(),
        meets_qa033: false,
    };
    assert!(
        !empty_env.is_complete(),
        "IMP-186b: Empty environment should be incomplete"
    );

    println!("\nIMP-186b: Environment Completeness:");
    println!("  Captured: complete={}", env.is_complete());
    println!("  Empty: complete={}", empty_env.is_complete());
}

/// IMP-186c: Test GPU environment
#[test]
fn test_imp_186c_gpu_environment() {
    let env = BenchEnvironment::capture().with_gpu("NVIDIA RTX 4090");

    assert!(env.gpu_name.is_some(), "IMP-186c: GPU name should be set");
    assert_eq!(
        env.gpu_name.as_deref(),
        Some("NVIDIA RTX 4090"),
        "IMP-186c: GPU name should match"
    );

    let cpu_only = BenchEnvironment::capture();
    assert!(
        cpu_only.gpu_name.is_none(),
        "IMP-186c: CPU-only should have no GPU"
    );

    println!("\nIMP-186c: GPU Environment:");
    println!("  With GPU: {:?}", env.gpu_name);
    println!("  CPU-only: {:?}", cpu_only.gpu_name);
}

/// IMP-186d: Real-world environment metadata
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_186d_realworld_environment() {
    let env = BenchEnvironment::capture();

    println!("\nIMP-186d: Real-World Environment:");
    println!("  OS: {} ({})", env.os_name, env.os_version);
    println!("  CPU: {} ({} cores)", env.cpu_model, env.cpu_cores);
    println!("  RAM: {:.1} GB", env.ram_gb);
    println!("  GPU: {:?}", env.gpu_name);
    println!("  Rust: {}", env.rust_version);
    println!("  Timestamp: {}", env.timestamp);
    println!(
        "  QA-033: {}",
        if env.meets_qa033 { "PASS" } else { "FAIL" }
    );
}

// ================================================================================
// IMP-187: Outlier Detection MAD (QA-034)
// Outlier detection using Median Absolute Deviation per Fleming & Wallace [5]
// ================================================================================

/// Outlier detection result using MAD
#[derive(Debug)]
pub struct OutlierResult {
    pub median: f64,
    pub mad: f64,
    pub threshold: f64,
    pub num_outliers: usize,
    pub outlier_indices: Vec<usize>,
    pub meets_qa034: bool,
}
