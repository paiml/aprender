
impl ItlMetrics {
    /// Create ITL metrics from raw measurements
    #[must_use]
    pub fn from_measurements(itl_times_ms: &[f64]) -> Self {
        if itl_times_ms.is_empty() {
            return Self::default();
        }

        let mut sorted = itl_times_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let median_ms = if n.is_multiple_of(2) {
            f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
        } else {
            sorted[n / 2]
        };

        let mean = itl_times_ms.iter().sum::<f64>() / n as f64;
        let variance = itl_times_ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (n as f64 - 1.0).max(1.0);
        let std_dev_ms = variance.sqrt();

        let percentile_99 = ((n as f64 * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        let percentile_999 = ((n as f64 * 0.999).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);

        Self {
            median_ms,
            std_dev_ms,
            p99_ms: sorted[percentile_99],
            p999_ms: sorted[percentile_999],
        }
    }

    /// Check if jitter is acceptable (std_dev < threshold)
    #[must_use]
    pub fn is_low_jitter(&self, threshold_ms: f64) -> bool {
        self.std_dev_ms < threshold_ms
    }
}

// ============================================================================
// KL-Divergence Quality Validation (Section 6.1)
// ============================================================================

/// Result of quantization quality validation
#[derive(Debug, Clone, PartialEq)]
pub enum QualityResult {
    /// Quality is acceptable
    Pass {
        /// Measured KL-divergence (nats)
        kl_divergence: f64,
    },
    /// Quality degradation detected
    Fail {
        /// Measured KL-divergence (nats)
        kl_divergence: f64,
        /// Threshold that was exceeded
        threshold: f64,
        /// Descriptive message
        message: &'static str,
    },
}

/// Compute softmax of logits
fn softmax(logits: &[f32]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f64> = logits
        .iter()
        .map(|x| ((*x - max_logit) as f64).exp())
        .collect();
    let sum: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|x| x / sum).collect()
}

/// Validate quantization quality using KL-Divergence
///
/// Per LLM.int8() [13], epsilon checks fail on outlier features.
/// KL-divergence provides a proper information-theoretic measure.
///
/// # Arguments
///
/// * `fp32_logits` - Reference logits from FP32 model
/// * `quantized_logits` - Logits from quantized model
/// * `threshold` - Maximum acceptable KL-divergence (nats)
///
/// # Returns
///
/// `QualityResult::Pass` if KL-divergence < threshold, `Fail` otherwise.
#[must_use]
pub fn validate_quantization_quality(
    fp32_logits: &[f32],
    quantized_logits: &[f32],
    threshold: f64,
) -> QualityResult {
    if fp32_logits.len() != quantized_logits.len() {
        return QualityResult::Fail {
            kl_divergence: f64::INFINITY,
            threshold,
            message: "Logit vector lengths do not match",
        };
    }

    if fp32_logits.is_empty() {
        return QualityResult::Pass { kl_divergence: 0.0 };
    }

    // Convert to probability distributions
    let fp32_probs = softmax(fp32_logits);
    let quant_probs = softmax(quantized_logits);

    // Compute KL(P_fp32 || P_quant)
    let kl_div: f64 = fp32_probs
        .iter()
        .zip(&quant_probs)
        .map(|(p, q)| {
            if *p > 1e-10 && *q > 1e-10 {
                p * (p / q).ln()
            } else {
                0.0
            }
        })
        .sum();

    if kl_div < threshold {
        QualityResult::Pass {
            kl_divergence: kl_div,
        }
    } else {
        QualityResult::Fail {
            kl_divergence: kl_div,
            threshold,
            message: "Quantization quality degradation detected",
        }
    }
}

// ============================================================================
// Benchmark Result (Section 4.1)
// ============================================================================

/// Configuration for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Model identifier
    pub model: String,
    /// Model format (apr, gguf, safetensors)
    pub format: String,
    /// Quantization level
    pub quantization: String,
    /// Runtime name
    pub runtime: String,
    /// Runtime version
    pub runtime_version: String,
}

/// Complete benchmark result per spec Section 4.1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Configuration used
    pub config: BenchmarkConfig,
    /// Cold start time (ms)
    pub cold_start_ms: f64,
    /// Model load time (ms)
    pub model_load_ms: f64,
    /// Time-to-first-token measurements (ms)
    pub ttft_ms: Vec<f64>,
    /// Inter-token latency measurements (ms)
    pub itl_ms: Vec<f64>,
    /// Generation throughput measurements (tok/s)
    pub generation_tok_s: Vec<f64>,
    /// Peak memory usage (MB)
    pub peak_memory_mb: u64,
    /// KV-cache fragmentation percentage
    pub kv_cache_waste_pct: f64,
    /// Total energy consumed (Joules)
    pub energy_joules: f64,
    /// Total tokens generated
    pub tokens_generated: u64,
    /// Actual number of iterations (dynamic sampling)
    pub actual_iterations: usize,
    /// CV at stop point
    pub cv_at_stop: f64,
    /// Unix timestamp
    pub timestamp: u64,
}

/// Summary statistics for benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    // TTFT metrics
    /// TTFT p50 (ms)
    pub ttft_p50: f64,
    /// TTFT p95 (ms)
    pub ttft_p95: f64,
    /// TTFT p99 (ms)
    pub ttft_p99: f64,
    /// TTFT p99.9 (ms)
    pub ttft_p999: f64,

    // ITL metrics
    /// ITL median (ms)
    pub itl_median: f64,
    /// ITL standard deviation (jitter)
    pub itl_std_dev: f64,

    // Throughput metrics
    /// Throughput median (tok/s)
    pub throughput_median: f64,
    /// Throughput 95% CI (lower, upper)
    pub throughput_ci_95: (f64, f64),

    // Energy metrics
    /// Energy per token (J/tok)
    pub token_joules: f64,

    // Memory metrics
    /// KV-cache waste percentage
    pub memory_waste_pct: f64,

    // Statistical validity
    /// Number of iterations run
    pub iterations: usize,
    /// Final CV value
    pub cv_final: f64,
}

impl BenchmarkResult {
    /// Generate summary statistics from raw measurements
    #[must_use]
    pub fn summary(&self) -> BenchmarkSummary {
        BenchmarkSummary {
            ttft_p50: percentile(&self.ttft_ms, 50.0),
            ttft_p95: percentile(&self.ttft_ms, 95.0),
            ttft_p99: percentile(&self.ttft_ms, 99.0),
            ttft_p999: percentile(&self.ttft_ms, 99.9),

            itl_median: percentile(&self.itl_ms, 50.0),
            itl_std_dev: compute_std_dev(&self.itl_ms),

            throughput_median: percentile(&self.generation_tok_s, 50.0),
            throughput_ci_95: bootstrap_ci(&self.generation_tok_s, 0.95, 1000),

            token_joules: if self.tokens_generated > 0 {
                self.energy_joules / self.tokens_generated as f64
            } else {
                0.0
            },

            memory_waste_pct: self.kv_cache_waste_pct,
            iterations: self.actual_iterations,
            cv_final: self.cv_at_stop,
        }
    }
}

/// Compute percentile of a dataset
fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((sorted.len() as f64 * p / 100.0).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

/// Compute standard deviation
fn compute_std_dev(data: &[f64]) -> f64 {
    compute_variance(data).sqrt()
}

/// Bootstrap confidence interval
fn bootstrap_ci(data: &[f64], confidence: f64, n_resamples: usize) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }

    let mut bootstrap_means = Vec::with_capacity(n_resamples);
    let n = data.len();

    for i in 0..n_resamples {
        // Simple deterministic pseudo-random for reproducibility
        // Uses a basic LCG instead of hash for clippy compliance
        let seed = (i as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);

        let mut sum = 0.0;
        for j in 0..n {
            let idx = ((seed.wrapping_mul(j as u64 + 1)) as usize) % n;
            sum += data[idx];
        }
        bootstrap_means.push(sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower_idx = ((n_resamples as f64 * alpha / 2.0).floor() as usize).min(n_resamples - 1);
    let upper_idx =
        ((n_resamples as f64 * (1.0 - alpha / 2.0)).ceil() as usize).min(n_resamples - 1);

    (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
}

// ============================================================================
// Convoy Test (Section 2.4)
// ============================================================================

/// Workload type for convoy testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// Short QA: 32 input tokens, 64 output tokens
    ShortQa,
    /// Long Context: 2048 input tokens, 512 output tokens
    LongContext,
}

impl WorkloadType {
    /// Get input token count for this workload type
    #[must_use]
    pub const fn input_tokens(&self) -> usize {
        match self {
            Self::ShortQa => 32,
            Self::LongContext => 2048,
        }
    }

    /// Get output token count for this workload type
    #[must_use]
    pub const fn output_tokens(&self) -> usize {
        match self {
            Self::ShortQa => 64,
            Self::LongContext => 512,
        }
    }
}

/// Configuration for convoy test per spec Section 2.4
#[derive(Debug, Clone)]
pub struct ConvoyTestConfig {
    /// Number of long-context requests (default: 10)
    pub long_requests: usize,
    /// Number of short-QA requests (default: 100)
    pub short_requests: usize,
    /// Maximum acceptable p99 latency increase (default: 50%)
    pub max_p99_increase_pct: f64,
    /// Maximum acceptable head-of-line blocking time (ms)
    pub max_hol_blocking_ms: f64,
    /// Maximum acceptable KV-cache fragmentation (%)
    pub max_kv_fragmentation_pct: f64,
}

impl Default for ConvoyTestConfig {
    fn default() -> Self {
        Self {
            long_requests: 10,
            short_requests: 100,
            max_p99_increase_pct: 50.0,
            max_hol_blocking_ms: 500.0,
            max_kv_fragmentation_pct: 15.0,
        }
    }
}

/// Result of a single request in convoy test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvoyRequestResult {
    /// Request type
    pub workload_type: String,
    /// Time spent waiting (head-of-line blocking)
    pub queue_time_ms: f64,
    /// Time to first token
    pub ttft_ms: f64,
    /// Total latency
    pub total_latency_ms: f64,
}

/// Overall convoy test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvoyTestResult {
    /// Number of long-context requests in test
    pub long_requests: usize,
    /// Number of short-QA requests in test
    pub short_requests: usize,

    /// Baseline: Short-QA p99 without convoy
    pub baseline_short_p99_ms: f64,
    /// Convoy: Short-QA p99 with convoy
    pub convoy_short_p99_ms: f64,
    /// P99 increase percentage
    pub p99_increase_pct: f64,

    /// Maximum head-of-line blocking observed
    pub max_hol_blocking_ms: f64,
    /// Average head-of-line blocking
    pub avg_hol_blocking_ms: f64,

    /// KV-cache fragmentation during convoy
    pub kv_fragmentation_pct: f64,

    /// Pass/fail status
    pub passed: bool,
    /// Failure reasons (if any)
    pub failure_reasons: Vec<String>,
}
