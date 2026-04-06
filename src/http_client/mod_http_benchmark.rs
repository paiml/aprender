
// ============================================================================
// HTTP Benchmark Runner with CV-based Stopping (Hoefler & Belli SC'15)
// ============================================================================

/// Configuration for HTTP benchmark runs
///
/// Per spec v1.0.1, uses canonical inputs and CV-based stopping criterion.
#[derive(Debug, Clone)]
pub struct HttpBenchmarkConfig {
    /// CV-based stopping criterion (per Hoefler & Belli SC'15)
    pub cv_criterion: CvStoppingCriterion,
    /// Warmup iterations (not counted in stats)
    pub warmup_iterations: usize,
    /// Prompt for inference (uses canonical input by default)
    pub prompt: String,
    /// Max tokens to generate (uses canonical input by default)
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// Whether to run preflight validation
    pub run_preflight: bool,
    /// Whether to filter outliers using MAD
    pub filter_outliers: bool,
    /// Outlier k-factor (3.0 = 99.7% for normal distribution)
    pub outlier_k_factor: f64,
}

impl Default for HttpBenchmarkConfig {
    fn default() -> Self {
        Self {
            cv_criterion: CvStoppingCriterion::default(), // 5% CV, 5-30 samples
            warmup_iterations: 2,
            prompt: canonical_inputs::LATENCY_PROMPT.to_string(),
            max_tokens: canonical_inputs::MAX_TOKENS,
            temperature: 0.0, // Deterministic by default
            run_preflight: true,
            filter_outliers: true,
            outlier_k_factor: 3.0,
        }
    }
}

impl HttpBenchmarkConfig {
    /// Create config with relaxed CV threshold (for quicker benchmarks)
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            cv_criterion: CvStoppingCriterion::new(3, 10, 0.20), // 20% CV
            warmup_iterations: 1,
            run_preflight: false,
            filter_outliers: false,
            ..Default::default()
        }
    }

    /// Create config optimized for reproducibility (strict CV)
    #[must_use]
    pub fn reproducible() -> Self {
        Self {
            cv_criterion: CvStoppingCriterion::new(10, 50, 0.03), // 3% CV, more samples
            warmup_iterations: 3,
            run_preflight: true,
            filter_outliers: true,
            outlier_k_factor: 2.5, // Stricter outlier detection
            ..Default::default()
        }
    }

    /// Backward-compatible accessors for min_samples
    #[must_use]
    pub fn min_samples(&self) -> usize {
        self.cv_criterion.min_samples
    }

    /// Backward-compatible accessor for max_samples
    #[must_use]
    pub fn max_samples(&self) -> usize {
        self.cv_criterion.max_samples
    }

    /// Backward-compatible accessor for cv_threshold
    #[must_use]
    pub fn cv_threshold(&self) -> f64 {
        self.cv_criterion.cv_threshold
    }
}

/// Results from a benchmark run
///
/// Per spec v1.0.1, includes quality metrics for reproducibility assessment.
#[derive(Debug, Clone)]
pub struct HttpBenchmarkResult {
    /// Collected latency samples (ms) - raw, before outlier filtering
    pub latency_samples: Vec<f64>,
    /// Filtered latency samples (ms) - after outlier removal
    pub latency_samples_filtered: Vec<f64>,
    /// Mean latency (ms) - computed from filtered samples
    pub mean_latency_ms: f64,
    /// P50 latency (ms)
    pub p50_latency_ms: f64,
    /// P99 latency (ms)
    pub p99_latency_ms: f64,
    /// Standard deviation (ms)
    pub std_dev_ms: f64,
    /// Coefficient of variation at stop
    pub cv_at_stop: f64,
    /// Throughput (tokens/sec)
    pub throughput_tps: f64,
    /// Cold start latency (first iteration after warmup)
    pub cold_start_ms: f64,
    /// Number of samples collected (raw)
    pub sample_count: usize,
    /// Number of samples after outlier filtering
    pub filtered_sample_count: usize,
    /// Whether CV threshold was achieved
    pub cv_converged: bool,
    /// Quality metrics for reproducibility assessment
    pub quality_metrics: QualityMetrics,
}

/// HTTP benchmark runner with CV-based stopping and preflight validation
///
/// Per spec v1.0.1, implements Toyota Way principles:
/// - Jidoka: Fail-fast via preflight validation
/// - Poka-yoke: Type-safe configuration
/// - Genchi Genbutsu: Verify actual server state before benchmark
pub struct HttpBenchmarkRunner {
    client: ModelHttpClient,
    config: HttpBenchmarkConfig,
    preflight_runner: Option<PreflightRunner>,
    outlier_detector: OutlierDetector,
}
