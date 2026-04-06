
impl ConvoyTestResult {
    /// Create a new convoy test result from measurements
    #[must_use]
    pub fn new(
        config: &ConvoyTestConfig,
        baseline_short_latencies: &[f64],
        convoy_short_latencies: &[f64],
        hol_blocking_times: &[f64],
        kv_fragmentation_pct: f64,
    ) -> Self {
        let baseline_short_p99 = percentile(baseline_short_latencies, 99.0);
        let convoy_short_p99 = percentile(convoy_short_latencies, 99.0);

        let p99_increase_pct = if baseline_short_p99 > 0.0 {
            ((convoy_short_p99 - baseline_short_p99) / baseline_short_p99) * 100.0
        } else {
            0.0
        };

        let max_hol_blocking = hol_blocking_times.iter().copied().fold(0.0_f64, f64::max);
        let avg_hol_blocking = if hol_blocking_times.is_empty() {
            0.0
        } else {
            hol_blocking_times.iter().sum::<f64>() / hol_blocking_times.len() as f64
        };

        let mut failure_reasons = Vec::new();

        if p99_increase_pct > config.max_p99_increase_pct {
            failure_reasons.push(format!(
                "P99 increase {p99_increase_pct:.1}% exceeds threshold {:.1}%",
                config.max_p99_increase_pct
            ));
        }

        if max_hol_blocking > config.max_hol_blocking_ms {
            failure_reasons.push(format!(
                "Max HOL blocking {max_hol_blocking:.1}ms exceeds threshold {:.1}ms",
                config.max_hol_blocking_ms
            ));
        }

        if kv_fragmentation_pct > config.max_kv_fragmentation_pct {
            failure_reasons.push(format!(
                "KV fragmentation {kv_fragmentation_pct:.1}% exceeds threshold {:.1}%",
                config.max_kv_fragmentation_pct
            ));
        }

        Self {
            long_requests: config.long_requests,
            short_requests: config.short_requests,
            baseline_short_p99_ms: baseline_short_p99,
            convoy_short_p99_ms: convoy_short_p99,
            p99_increase_pct,
            max_hol_blocking_ms: max_hol_blocking,
            avg_hol_blocking_ms: avg_hol_blocking,
            kv_fragmentation_pct,
            passed: failure_reasons.is_empty(),
            failure_reasons,
        }
    }
}

// ============================================================================
// Saturation Test (Section 2.5)
// ============================================================================

/// Configuration for saturation stress test per spec Section 2.5
#[derive(Debug, Clone)]
pub struct SaturationTestConfig {
    /// CPU load percentage (default: 50%)
    pub cpu_load_pct: u8,
    /// Maximum acceptable throughput degradation (default: 30%)
    pub max_throughput_degradation_pct: f64,
    /// Maximum acceptable p99 latency increase (default: 100%)
    pub max_p99_increase_pct: f64,
}

impl Default for SaturationTestConfig {
    fn default() -> Self {
        Self {
            cpu_load_pct: 50,
            max_throughput_degradation_pct: 30.0,
            max_p99_increase_pct: 100.0,
        }
    }
}

/// Saturation test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationTestResult {
    /// CPU load used
    pub cpu_load_pct: u8,

    /// Baseline throughput (tok/s)
    pub baseline_throughput: f64,
    /// Stressed throughput (tok/s)
    pub stressed_throughput: f64,
    /// Throughput degradation percentage
    pub throughput_degradation_pct: f64,

    /// Baseline p99 latency (ms)
    pub baseline_p99_ms: f64,
    /// Stressed p99 latency (ms)
    pub stressed_p99_ms: f64,
    /// P99 latency increase percentage
    pub p99_increase_pct: f64,

    /// Pass/fail status
    pub passed: bool,
    /// Failure reasons (if any)
    pub failure_reasons: Vec<String>,
}

impl SaturationTestResult {
    /// Create a new saturation test result
    #[must_use]
    pub fn new(
        config: &SaturationTestConfig,
        baseline_throughputs: &[f64],
        stressed_throughputs: &[f64],
        baseline_latencies: &[f64],
        stressed_latencies: &[f64],
    ) -> Self {
        let baseline_throughput = if baseline_throughputs.is_empty() {
            0.0
        } else {
            baseline_throughputs.iter().sum::<f64>() / baseline_throughputs.len() as f64
        };

        let stressed_throughput = if stressed_throughputs.is_empty() {
            0.0
        } else {
            stressed_throughputs.iter().sum::<f64>() / stressed_throughputs.len() as f64
        };

        let throughput_degradation_pct = if baseline_throughput > 0.0 {
            ((baseline_throughput - stressed_throughput) / baseline_throughput) * 100.0
        } else {
            0.0
        };

        let baseline_p99 = percentile(baseline_latencies, 99.0);
        let stressed_p99 = percentile(stressed_latencies, 99.0);

        let p99_increase_pct = if baseline_p99 > 0.0 {
            ((stressed_p99 - baseline_p99) / baseline_p99) * 100.0
        } else {
            0.0
        };

        let mut failure_reasons = Vec::new();

        if throughput_degradation_pct > config.max_throughput_degradation_pct {
            failure_reasons.push(format!(
                "Throughput degradation {throughput_degradation_pct:.1}% exceeds threshold {:.1}%",
                config.max_throughput_degradation_pct
            ));
        }

        if p99_increase_pct > config.max_p99_increase_pct {
            failure_reasons.push(format!(
                "P99 increase {p99_increase_pct:.1}% exceeds threshold {:.1}%",
                config.max_p99_increase_pct
            ));
        }

        Self {
            cpu_load_pct: config.cpu_load_pct,
            baseline_throughput,
            stressed_throughput,
            throughput_degradation_pct,
            baseline_p99_ms: baseline_p99,
            stressed_p99_ms: stressed_p99,
            p99_increase_pct,
            passed: failure_reasons.is_empty(),
            failure_reasons,
        }
    }
}

// ============================================================================
// Benchmark Runner (Full Harness)
// ============================================================================

/// Hardware specification for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    /// CPU model
    pub cpu: String,
    /// GPU model (if any)
    pub gpu: Option<String>,
    /// Total memory in GB
    pub memory_gb: u64,
    /// Storage type
    pub storage: String,
}

impl Default for HardwareSpec {
    fn default() -> Self {
        Self {
            cpu: "Unknown".to_string(),
            gpu: None,
            memory_gb: 0,
            storage: "Unknown".to_string(),
        }
    }
}

/// Sampling method configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Sampling method (e.g., "dynamic_cv")
    pub method: String,
    /// CV threshold for stopping
    pub cv_threshold: f64,
    /// Actual iterations run
    pub actual_iterations: usize,
    /// CV at stop point
    pub cv_at_stop: f64,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            method: "dynamic_cv".to_string(),
            cv_threshold: 0.05,
            actual_iterations: 0,
            cv_at_stop: 0.0,
            warmup_iterations: 100,
        }
    }
}

/// Thermal validity info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalInfo {
    /// Whether thermal conditions were valid
    pub valid: bool,
    /// Temperature variance (°C)
    pub temp_variance_c: f64,
    /// Maximum temperature observed (°C)
    pub max_temp_c: f64,
}

impl Default for ThermalInfo {
    fn default() -> Self {
        Self {
            valid: true,
            temp_variance_c: 0.0,
            max_temp_c: 0.0,
        }
    }
}

/// TTFT results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtftResults {
    /// P50 (median)
    pub p50: f64,
    /// P95
    pub p95: f64,
    /// P99
    pub p99: f64,
    /// P99.9
    pub p999: f64,
}

/// ITL results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItlResults {
    /// Median ITL
    pub median: f64,
    /// Standard deviation (jitter)
    pub std_dev: f64,
    /// P99 ITL
    pub p99: f64,
}

/// Throughput results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputResults {
    /// Median throughput (tok/s)
    pub median: f64,
    /// 95% confidence interval
    pub ci_95: (f64, f64),
}

/// Memory results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResults {
    /// Model size (MB)
    pub model_mb: u64,
    /// Peak RSS (MB)
    pub peak_rss_mb: u64,
    /// KV-cache waste percentage
    pub kv_waste_pct: f64,
}

/// Energy results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyResults {
    /// Total energy (Joules)
    pub total_joules: f64,
    /// Energy per token (J/tok)
    pub token_joules: f64,
    /// Idle power (Watts)
    pub idle_watts: f64,
}

/// Cold start results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartResults {
    /// Median cold start time (ms)
    pub median: f64,
    /// P99 cold start time (ms)
    pub p99: f64,
}

/// Quality validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityValidation {
    /// KL-divergence vs FP32
    pub kl_divergence_vs_fp32: f64,
    /// Perplexity on WikiText-2 (optional)
    pub perplexity_wikitext2: Option<f64>,
}

/// Full benchmark results per JSON schema v1.1 (Appendix B)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBenchmarkResult {
    /// Schema version
    pub version: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Model configuration
    pub config: BenchmarkConfig,
    /// Hardware specification
    pub hardware: HardwareSpec,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Thermal information
    pub thermal: ThermalInfo,
    /// All results
    pub results: BenchmarkResults,
    /// Quality validation
    pub quality: QualityValidation,
}

/// Consolidated benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Time-to-first-token metrics
    pub ttft_ms: TtftResults,
    /// Inter-token latency metrics
    pub itl_ms: ItlResults,
    /// Throughput metrics
    pub throughput_tok_s: ThroughputResults,
    /// Memory metrics
    pub memory_mb: MemoryResults,
    /// Energy metrics
    pub energy: EnergyResults,
    /// Cold start metrics
    pub cold_start_ms: ColdStartResults,
}
