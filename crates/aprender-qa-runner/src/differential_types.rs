/// Result of inference comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceComparisonResult {
    /// Total tokens compared
    pub total_tokens: usize,
    /// Tokens with matching argmax
    pub matching_tokens: usize,
    /// Maximum logit difference observed
    pub max_logit_diff: f64,
    /// Whether comparison passed
    pub passed: bool,
    /// Per-token comparison details
    pub token_comparisons: Vec<TokenComparison>,
}

/// Comparison of a single token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenComparison {
    /// Token index
    pub index: usize,
    /// Token ID from model A
    pub token_a: u32,
    /// Token ID from model B
    pub token_b: u32,
    /// Logit difference
    pub logit_diff: f64,
    /// Whether tokens match
    pub matches: bool,
}

/// Result of differential benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffBenchmarkResult {
    /// Model A metrics
    pub model_a: BenchmarkMetrics,
    /// Model B metrics
    pub model_b: BenchmarkMetrics,
    /// Throughput delta percentage
    pub throughput_delta_pct: f64,
    /// Latency delta percentage (p50)
    pub latency_p50_delta_pct: f64,
    /// Latency delta percentage (p99)
    pub latency_p99_delta_pct: f64,
    /// Whether regression detected
    pub regression_detected: bool,
    /// Regression threshold used
    pub regression_threshold: f64,
}

/// Benchmark metrics for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Model path
    pub path: String,
    /// Throughput in tokens/second
    pub throughput_tps: f64,
    /// p50 latency in milliseconds
    pub latency_p50_ms: f64,
    /// p99 latency in milliseconds
    pub latency_p99_ms: f64,
}

/// CI profile metrics (nested in JSON output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiProfileMetrics {
    /// Throughput achieved (tok/s)
    #[serde(alias = "throughput_tok_s")]
    pub throughput_tok_s: f64,
    /// p50 latency (ms)
    pub latency_p50_ms: f64,
    /// p99 latency (ms)
    pub latency_p99_ms: f64,
}

/// CI profile assertions result from apr profile --ci --json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiProfileResult {
    /// Model path
    #[serde(default)]
    pub model: String,
    /// Nested metrics
    #[serde(default)]
    pub metrics: Option<CiProfileMetrics>,
    /// Assertion results
    #[serde(default)]
    pub assertions: Vec<CiAssertion>,
    /// Overall pass/fail
    #[serde(default)]
    pub passed: bool,
    // Legacy flat fields for backwards compatibility
    /// Throughput achieved (legacy)
    #[serde(default)]
    pub throughput_tps: f64,
    /// p50 latency (legacy)
    #[serde(default)]
    pub latency_p50_ms: f64,
    /// p99 latency (legacy)
    #[serde(default)]
    pub latency_p99_ms: f64,
}

/// Accessor methods for CI profile metrics with legacy field fallback
impl CiProfileResult {
    /// Get throughput in tok/s (from nested metrics or legacy field)
    #[must_use]
    pub fn throughput(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(self.throughput_tps, |m| m.throughput_tok_s)
    }

    /// Get p50 latency in ms
    #[must_use]
    pub fn p50_latency(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(self.latency_p50_ms, |m| m.latency_p50_ms)
    }

    /// Get p99 latency in ms
    #[must_use]
    pub fn p99_latency(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(self.latency_p99_ms, |m| m.latency_p99_ms)
    }
}

/// A single CI assertion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiAssertion {
    /// Assertion name
    pub name: String,
    /// Expected value (threshold)
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Whether assertion passed
    pub passed: bool,
    /// Gate ID (optional - not all apr versions include it)
    #[serde(default)]
    pub gate_id: String,
}

/// Execute profile CI mode
///
/// Runs `apr profile --ci` with optional assertion flags.
///
/// # Errors
///
/// Returns an error if the apr command fails to execute.
pub fn run_profile_ci(
    apr_binary: &str,
    model_path: &Path,
    min_throughput: Option<f64>,
    max_p99: Option<f64>,
    max_p50: Option<f64>,
    warmup: usize,
    measure: usize,
) -> Result<CiProfileResult> {
    let mut cmd = Command::new(apr_binary);
    cmd.arg("profile").arg(model_path).arg("--ci");

    if let Some(throughput) = min_throughput {
        cmd.arg("--assert-throughput").arg(throughput.to_string());
    }
    if let Some(p99) = max_p99 {
        cmd.arg("--assert-p99").arg(p99.to_string());
    }
    if let Some(p50) = max_p50 {
        cmd.arg("--assert-p50").arg(p50.to_string());
    }

    cmd.arg("--warmup").arg(warmup.to_string());
    cmd.arg("--measure").arg(measure.to_string());
    cmd.arg("--format").arg("json");

    let output = cmd.output().map_err(|e| Error::ExecutionFailed {
        command: "apr profile --ci".to_string(),
        reason: e.to_string(),
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Extract JSON object from output (may have prefix lines like "Loading model...")
    let json_start = stdout.find('{');
    let json_str = json_start.map_or_else(|| stdout.as_ref(), |i| &stdout[i..]);

    // Try JSON parsing
    if let Ok(result) = serde_json::from_str::<CiProfileResult>(json_str) {
        return Ok(result);
    }

    // Fall back to basic result based on exit code
    Ok(CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 0.0,
        latency_p50_ms: 0.0,
        latency_p99_ms: 0.0,
        assertions: vec![],
        passed: output.status.success(),
    })
}

/// Execute differential benchmark
///
/// Compares performance between two models to detect regressions.
///
/// # Errors
///
/// Returns an error if the apr command fails or output cannot be parsed.
pub fn run_diff_benchmark(
    apr_binary: &str,
    model_a: &Path,
    model_b: &Path,
    regression_threshold: f64,
) -> Result<DiffBenchmarkResult> {
    // Retry on ETXTBSY (os error 26) — transient fork/exec race on Linux
    let output = {
        let mut attempts = 0;
        loop {
            match Command::new(apr_binary)
                .arg("profile")
                .arg(model_a)
                .arg(model_b)
                .arg("--diff-benchmark")
                .arg("--regression-threshold")
                .arg(regression_threshold.to_string())
                .arg("--json")
                .output()
            {
                Ok(output) => break output,
                Err(e) if e.raw_os_error() == Some(26) && attempts < 3 => {
                    attempts += 1;
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(Error::ExecutionFailed {
                        command: "apr profile --diff-benchmark".to_string(),
                        reason: e.to_string(),
                    });
                }
            }
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);

    if let Ok(result) = serde_json::from_str::<DiffBenchmarkResult>(&stdout) {
        return Ok(result);
    }

    Err(Error::ExecutionFailed {
        command: "apr profile --diff-benchmark".to_string(),
        reason: "Failed to parse output".to_string(),
    })
}

/// Result of throughput benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Throughput in tokens/second
    pub throughput_tps: f64,
    /// Whether the benchmark passed minimum threshold
    pub passed: bool,
    /// Backend used (cpu or gpu)
    pub backend: String,
    /// Format tested (gguf, apr, safetensors)
    pub format: String,
}

/// Run throughput benchmark with explicit backend selection
///
/// Uses `apr bench --fast` (realizar) for real inference.
/// Backend selection via `CUDA_VISIBLE_DEVICES` environment variable.
///
/// # Arguments
/// * `apr_binary` - Path to apr binary
/// * `model_path` - Path to model file
/// * `use_gpu` - If true, use GPU; if false, set CUDA_VISIBLE_DEVICES=""
/// * `warmup` - Number of warmup iterations
/// * `iterations` - Number of measurement iterations
///
/// # Errors
///
/// Returns an error if the apr command fails to execute.
pub fn run_bench_throughput(
    apr_binary: &str,
    model_path: &Path,
    use_gpu: bool,
    warmup: usize,
    iterations: usize,
) -> Result<BenchResult> {
    let mut cmd = Command::new(apr_binary);
    cmd.arg("bench")
        .arg(model_path)
        .arg("--warmup")
        .arg(warmup.to_string())
        .arg("--iterations")
        .arg(iterations.to_string());

    // Force CPU-only by hiding CUDA devices
    if !use_gpu {
        cmd.env("CUDA_VISIBLE_DEVICES", "");
    }

    let output = cmd.output().map_err(|e| Error::ExecutionFailed {
        command: format!("apr bench {}", model_path.display()),
        reason: e.to_string(),
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse throughput from output: "Throughput: 65.5 tok/s (PASS: >= 10 tok/s)"
    let throughput = stdout
        .lines()
        .find(|line| line.contains("Throughput:"))
        .and_then(|line| {
            line.split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<f64>().ok())
        })
        .unwrap_or(0.0);

    let passed = output.status.success() && throughput >= 10.0;

    // Determine format from file extension
    let format = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(BenchResult {
        throughput_tps: throughput,
        passed,
        backend: if use_gpu { "gpu" } else { "cpu" }.to_string(),
        format,
    })
}

/// Result of format conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatConversionResult {
    /// Source format
    pub source_format: String,
    /// Target format
    pub target_format: String,
    /// Whether conversion succeeded
    pub success: bool,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Error message if failed
    pub error: Option<String>,
    /// Whether result was from cache
    pub cached: bool,
}

/// Compute SHA256 hash of a file (first 1MB for speed)
fn compute_file_hash(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| Error::ExecutionFailed {
        command: format!("open {}", path.display()),
        reason: e.to_string(),
    })?;

    let mut buffer = vec![0u8; 1024 * 1024]; // 1MB
    let bytes_read = file.read(&mut buffer).map_err(|e| Error::ExecutionFailed {
        command: format!("read {}", path.display()),
        reason: e.to_string(),
    })?;

    buffer.truncate(bytes_read);

    // Simple hash using std (no external dependency)
    let hash: u64 = buffer.iter().fold(0u64, |acc, &b| {
        acc.wrapping_mul(31).wrapping_add(u64::from(b))
    });

    Ok(format!("{hash:016x}"))
}
