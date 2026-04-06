//! Concurrent load testing engine for LLM inference endpoints.
//!
//! Generates concurrent chat completion requests, collects timing metrics,
//! and produces percentile-based latency reports.

use super::client::{BrickTrace, ChatMessage, ChatRequest, LlmClient, LlmClientError, Role};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// Validation mode (Feature 5: Inline Correctness)
// =============================================================================

/// Validation mode for inline correctness checking during load tests.
#[derive(Debug, Clone, Default)]
pub enum ValidationMode {
    /// No validation (zero overhead).
    #[default]
    None,
    /// Check: non-empty content, finish_reason present, token count > 0.
    Basic,
    /// Basic + response contains substring.
    Contains(String),
    /// Basic + response matches regex.
    Pattern(String),
}

impl ValidationMode {
    /// Parse from CLI string: "none", "basic", "contains:X", "pattern:X".
    pub fn parse(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "basic" => Self::Basic,
            s if s.starts_with("contains:") => Self::Contains(s[9..].to_string()),
            s if s.starts_with("pattern:") => Self::Pattern(s[8..].to_string()),
            _ => Self::None,
        }
    }

    /// Whether response content needs to be captured for validation.
    fn needs_content(&self) -> bool {
        matches!(self, Self::Contains(_) | Self::Pattern(_))
    }
}

/// Request scheduling mode for load generation (GH-25).
#[derive(Debug, Clone, Default)]
pub enum RequestRate {
    /// Closed-loop: each worker sends the next request immediately after receiving a response.
    /// Maximizes throughput but doesn't model realistic workloads.
    #[default]
    Max,
    /// Poisson arrival rate: requests arrive at the specified rate with exponentially distributed
    /// inter-arrival times. Models realistic web traffic.
    Poisson(f64),
    /// Constant rate: requests arrive at fixed intervals (1/rate seconds apart).
    Constant(f64),
}

/// Configuration for a load test run.
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    /// Number of concurrent workers.
    pub concurrency: usize,
    /// Total duration of the load test.
    pub duration: Duration,
    /// Prompts to cycle through.
    pub prompts: Vec<ChatRequest>,
    /// Name of the runtime being tested (for reporting).
    pub runtime_name: String,
    /// Duration of warmup phase (excluded from metrics). Default: zero (no warmup).
    pub warmup_duration: Duration,
    /// Use SSE streaming for per-token TPOT measurement. Default: false.
    pub stream: bool,
    /// Trace level for BrickProfiler data collection. Default: None.
    pub trace_level: Option<String>,
    /// TTFT SLO threshold in ms. Requests exceeding this are SLO violations.
    pub slo_ttft_ms: Option<f64>,
    /// TPOT SLO threshold in ms. Requests exceeding this are SLO violations.
    pub slo_tpot_ms: Option<f64>,
    /// End-to-end latency SLO threshold in ms.
    pub slo_latency_ms: Option<f64>,
    /// Request scheduling mode (GH-25). Default: Max (closed-loop).
    pub rate: RequestRate,
    /// Number of transformer layers in the model (e.g. 28 for Qwen 1.5B).
    /// When set, computes per-layer decode time for cross-runtime comparison.
    /// Per-layer time is overhead-free (derived from wall-clock TPOT, not per-brick sync).
    pub num_layers: Option<u32>,
    /// Inline correctness validation mode. Default: None (zero overhead).
    pub validate: ValidationMode,
    /// Multiplier of median ITL to classify as latency spike. Default: 5.0.
    pub spike_threshold: f64,
    /// Exit threshold for quality pass rate (e.g., 0.95). None = don't fail.
    pub fail_on_quality: Option<f64>,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            duration: Duration::from_secs(30),
            prompts: vec![default_prompt()],
            runtime_name: "unknown".to_string(),
            warmup_duration: Duration::ZERO,
            stream: false,
            trace_level: None,
            slo_ttft_ms: None,
            slo_tpot_ms: None,
            slo_latency_ms: None,
            rate: RequestRate::Max,
            num_layers: None,
            validate: ValidationMode::None,
            spike_threshold: 5.0,
            fail_on_quality: None,
        }
    }
}

/// Results from a load test run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResult {
    /// Total requests sent.
    pub total_requests: u64,
    /// Successful completions.
    pub successful: u64,
    /// Failed requests.
    pub failed: u64,
    /// Requests per second (throughput).
    pub throughput_rps: f64,
    /// Median latency (ms).
    pub latency_p50_ms: f64,
    /// 95th percentile latency (ms).
    pub latency_p95_ms: f64,
    /// 99th percentile latency (ms).
    pub latency_p99_ms: f64,
    /// Median time to first byte (ms).
    pub ttft_p50_ms: f64,
    /// Total tokens per second (sum of all completion tokens / wall time).
    /// NOTE: Not comparable across backends with different response lengths.
    pub tokens_per_sec: f64,
    /// Average completion tokens per response (GH-23).
    #[serde(default)]
    pub avg_tok_per_req: f64,
    /// Median inter-token latency in ms: (latency - ttft) / (tokens - 1) (GH-23).
    /// Comparable across backends regardless of response length.
    #[serde(default)]
    pub itl_p50_ms: f64,
    /// Decode throughput: 1000 / itl_p50_ms. True generation speed (GH-23).
    #[serde(default)]
    pub decode_tok_per_sec: f64,
    /// Prefill throughput: prompt_tokens / ttft_seconds. Measures prompt processing speed.
    #[serde(default)]
    pub prefill_tok_per_sec: f64,
    /// ISO 8601 timestamp of the run.
    pub timestamp: String,
    /// Name of the runtime tested.
    pub runtime_name: String,
    /// Total elapsed wall time (seconds).
    pub elapsed_secs: f64,
    /// Concurrency level used.
    pub concurrency: usize,
    // --- Extended percentiles (Benchmarking v2.0) ---
    /// TTFT 90th percentile (ms).
    #[serde(default)]
    pub ttft_p90_ms: f64,
    /// TTFT 95th percentile (ms).
    #[serde(default)]
    pub ttft_p95_ms: f64,
    /// TTFT 99th percentile (ms).
    #[serde(default)]
    pub ttft_p99_ms: f64,
    /// Time per output token P50 (ms). TPOT = (latency - TTFB) / (tokens - 1).
    #[serde(default)]
    pub tpot_p50_ms: f64,
    /// TPOT 90th percentile (ms).
    #[serde(default)]
    pub tpot_p90_ms: f64,
    /// TPOT 95th percentile (ms).
    #[serde(default)]
    pub tpot_p95_ms: f64,
    /// TPOT 99th percentile (ms).
    #[serde(default)]
    pub tpot_p99_ms: f64,
    /// Minimum latency (ms).
    #[serde(default)]
    pub latency_min_ms: f64,
    /// Maximum latency (ms).
    #[serde(default)]
    pub latency_max_ms: f64,
    /// Latency standard deviation (ms).
    #[serde(default)]
    pub latency_stddev_ms: f64,
    /// Error rate: failed / total.
    #[serde(default)]
    pub error_rate: f64,
    /// Total prompt (input) tokens across all requests.
    #[serde(default)]
    pub prompt_tokens_total: u64,
    /// Total completion (output) tokens across all requests.
    #[serde(default)]
    pub completion_tokens_total: u64,
    /// Percentage of requests truncated by max_tokens (finish_reason="length").
    #[serde(default)]
    pub truncated_pct: f64,
    /// Mean SSE chunks per completion token (1.0 = no batching, <1.0 = server batches tokens).
    #[serde(default)]
    pub sse_batch_ratio: f64,
    /// Goodput: percentage of requests meeting all configured SLOs.
    #[serde(default)]
    pub goodput_pct: f64,
    /// Output token count distribution: [min, p50, p90, max].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_dist: Option<[f64; 4]>,
    /// Per-layer decode time in microseconds: TPOT_us / num_layers.
    /// Overhead-free (derived from wall-clock ITL, not per-brick sync).
    /// Comparable across all runtimes regardless of internal profiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_us_per_layer: Option<f64>,
    /// Number of transformer layers used to compute per-layer time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<u32>,
    /// Aggregated BrickProfiler per-operation timing (GH-114).
    /// Present when --trace-level brick is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brick_trace_summary: Option<Vec<BrickTraceOpSummary>>,
    /// Per-request raw timing data for distribution analysis.
    /// Each entry: [latency_ms, ttft_ms, completion_tokens, itl_ms].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub request_details: Vec<RequestDetail>,
    /// Inline quality validation results (Feature 5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualityResult>,
    /// Tail latency analysis with jitter and drift detection (Feature 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_analysis: Option<TailAnalysis>,
    /// GPU telemetry collected during benchmark (Feature 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_telemetry: Option<GpuTelemetry>,
    /// Dataset statistics when --dataset was used (Feature 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_stats: Option<DatasetStats>,
    /// Time from process launch to first successful health check (ms).
    /// Present when benchmark was run with --start-command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cold_start_ms: Option<f64>,
}

/// Per-request timing for distribution analysis and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestDetail {
    /// Total request latency (ms).
    pub latency_ms: f64,
    /// Time to first token (ms).
    pub ttft_ms: f64,
    /// Completion tokens generated.
    pub completion_tokens: u32,
    /// Prompt tokens (if reported by server).
    pub prompt_tokens: u32,
    /// Mean inter-token latency for this request (ms). 0 if < 2 tokens.
    pub itl_ms: f64,
    /// Why generation stopped: "stop" (natural) or "length" (truncated by max_tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

// =============================================================================
// Quality validation result (Feature 5)
// =============================================================================

/// Quality validation result from inline correctness checking during load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityResult {
    /// Validation level used (e.g., "basic", "contains:X").
    pub validation_level: String,
    /// Total responses validated.
    pub total_validated: u64,
    /// Responses that passed validation.
    pub passed: u64,
    /// Responses that failed validation.
    pub failed: u64,
    /// Pass rate: passed / total_validated.
    pub pass_rate: f64,
    /// Individual failure details.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failures: Vec<QualityFailure>,
}

/// A single quality validation failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFailure {
    /// Index of the request that failed.
    pub request_idx: usize,
    /// Reason for failure (e.g., "empty_content", "zero_tokens", "missing_pattern").
    pub reason: String,
}

// =============================================================================
// Tail latency analysis (Feature 3)
// =============================================================================

/// Tail latency analysis with jitter and drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailAnalysis {
    /// ITL P99.9 (ms).
    pub itl_p999_ms: f64,
    /// ITL P99.99 (ms).
    pub itl_p9999_ms: f64,
    /// TTFT P99.9 (ms).
    pub ttft_p999_ms: f64,
    /// TTFT P99.99 (ms).
    pub ttft_p9999_ms: f64,
    /// End-to-end latency P99.9 (ms).
    pub latency_p999_ms: f64,
    /// End-to-end latency P99.99 (ms).
    pub latency_p9999_ms: f64,
    /// Tail ratio: P99/P50 for ITL. >5 = concern, >10 = critical.
    pub tail_ratio_itl: f64,
    /// Tail ratio: P99/P50 for TTFT.
    pub tail_ratio_ttft: f64,
    /// Tail ratio: P99/P50 for end-to-end latency.
    pub tail_ratio_latency: f64,
    /// Jitter metrics.
    pub jitter: JitterAnalysis,
    /// Drift metrics (latency trend over time).
    pub drift: DriftAnalysis,
}

/// Jitter (ITL variance) analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterAnalysis {
    /// Coefficient of variation: stddev(ITL) / mean(ITL).
    pub itl_cv: f64,
    /// Inter-quartile range of ITL (ms).
    pub itl_iqr_ms: f64,
    /// Number of latency spikes detected.
    pub spike_count: usize,
    /// Threshold used for spike detection (ms).
    pub spike_threshold_ms: f64,
    /// Individual spike details (capped at 20).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spikes: Vec<LatencySpike>,
}

/// A single latency spike event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySpike {
    /// Index of the request.
    pub request_idx: usize,
    /// ITL of the spiking request (ms).
    pub itl_ms: f64,
}

/// Drift detection: is latency degrading over time?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAnalysis {
    /// Linear regression slope of ITL over time (ms/min). Positive = degradation.
    pub itl_slope_ms_per_min: f64,
    /// Linear regression slope of TTFT over time (ms/min).
    pub ttft_slope_ms_per_min: f64,
    /// True if slope is significant (r^2 > 0.5 and positive slope).
    pub degradation_detected: bool,
}

// =============================================================================
// GPU telemetry (Feature 2)
// =============================================================================

/// GPU telemetry collected via nvidia-smi during benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTelemetry {
    /// Number of samples collected.
    pub samples: usize,
    /// GPU compute utilization (%).
    pub gpu_utilization_pct: TelemetryStat,
    /// GPU memory used (MB).
    pub memory_used_mb: TelemetryStat,
    /// Total GPU memory (MB).
    pub memory_total_mb: f64,
    /// Power draw (watts).
    pub power_draw_w: TelemetryStat,
    /// GPU temperature (Celsius).
    pub temperature_c: TelemetryStat,
    /// GPU clock speed (MHz).
    pub clock_gpu_mhz: TelemetryStat,
    /// Number of throttle events (clock drop >10%).
    pub throttle_events: usize,
    /// Total energy consumed (Wh).
    pub energy_total_wh: f64,
    /// Energy per completion token (mJ).
    pub energy_per_token_mj: f64,
    /// Energy per request (mJ).
    pub energy_per_request_mj: f64,
}

/// Min/mean/max summary statistic for telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryStat {
    /// Mean value.
    pub mean: f64,
    /// Maximum value.
    pub max: f64,
    /// Minimum value.
    pub min: f64,
}

// =============================================================================
// Concurrency sweep (Feature 1)
// =============================================================================

/// Result of a concurrency sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    /// Results at each concurrency level.
    pub levels: Vec<SweepLevel>,
    /// Optimal concurrency (highest throughput before saturation).
    pub optimal_concurrency: usize,
    /// Throughput at optimal concurrency (req/s).
    pub optimal_throughput_rps: f64,
    /// Concurrency levels on the Pareto frontier (throughput up, latency acceptable).
    pub pareto_frontier: Vec<usize>,
}

/// Result at a single concurrency level in a sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepLevel {
    /// Concurrency level tested.
    pub concurrency: usize,
    /// Aggregate throughput (req/s).
    pub throughput_rps: f64,
    /// P99 latency (ms).
    pub latency_p99_ms: f64,
    /// Decode throughput (tok/s).
    pub decode_tok_s: f64,
    /// Whether this level is saturated.
    pub saturated: bool,
    /// Reason for saturation (if saturated).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saturation_reason: Option<String>,
    /// Full load test result at this level.
    pub result: LoadTestResult,
}

// =============================================================================
// Dataset stats (Feature 4)
// =============================================================================

/// Statistics about the dataset used for load testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Source file path.
    pub source: String,
    /// Total number of prompts in the dataset.
    pub total_prompts: usize,
    /// Input token distribution: [min, p50, p90, max].
    pub input_tokens: [f64; 4],
    /// Requested max_tokens distribution: [min, p50, p90, max].
    pub max_tokens_requested: [f64; 4],
}

/// Aggregated BrickProfiler operation timing across benchmark requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrickTraceOpSummary {
    /// Operation name (e.g., "attention_qkv", "mlp_gate_up").
    pub name: String,
    /// Mean time in microseconds across all requests.
    pub mean_us: f64,
    /// Minimum time in microseconds.
    pub min_us: f64,
    /// Maximum time in microseconds.
    pub max_us: f64,
    /// Percentage of total inference time.
    pub pct_of_total: f64,
    /// Number of samples.
    pub samples: usize,
}

/// Load test executor.
#[derive(Debug)]
pub struct LoadTest {
    client: LlmClient,
    config: LoadTestConfig,
}

/// Individual request timing record.
#[derive(Debug, Clone)]
struct RequestRecord {
    latency: Duration,
    ttfb: Duration,
    tokens: u32,
    prompt_tokens: u32,
    success: bool,
    /// Per-token timestamps from SSE streaming (empty if non-streaming).
    token_timestamps: Vec<Duration>,
    /// Brick trace data from response (when trace_level was set).
    brick_trace: Option<BrickTrace>,
    /// Why generation stopped (e.g., "stop", "length").
    finish_reason: Option<String>,
    /// Response content (only captured when validation requires content inspection).
    response_content: Option<String>,
}

impl LoadTest {
    /// Create a new load test.
    pub fn new(client: LlmClient, config: LoadTestConfig) -> Self {
        Self { client, config }
    }

    /// Run the load test and return aggregated results.
    pub async fn run(&self) -> Result<LoadTestResult, LlmClientError> {
        // Warmup phase: send requests but discard results
        if self.config.warmup_duration > Duration::ZERO {
            self.run_phase(self.config.warmup_duration).await?;
        }

        // Measurement phase: use actual wall time
        let measure_start = Instant::now();
        let all_records = self.run_phase(self.config.duration).await?;
        let elapsed = measure_start.elapsed().as_secs_f64();

        let mut result = aggregate_results(
            &all_records,
            elapsed,
            &self.config.runtime_name,
            self.config.concurrency,
            self.config.slo_ttft_ms,
            self.config.slo_tpot_ms,
            self.config.slo_latency_ms,
            self.config.num_layers,
        );

        // Feature 5: Inline quality validation
        if !matches!(self.config.validate, ValidationMode::None) {
            result.quality = Some(compute_quality(&all_records, &self.config.validate));
        }

        // Feature 3: Tail latency analysis (always computed, cheap)
        result.tail_analysis = Some(compute_tail_analysis(
            &all_records,
            self.config.spike_threshold,
        ));

        Ok(result)
    }

    /// Run a single phase (warmup or measurement) for the given duration.
    async fn run_phase(&self, duration: Duration) -> Result<Vec<RequestRecord>, LlmClientError> {
        match self.config.rate {
            RequestRate::Max => self.run_phase_max(duration).await,
            RequestRate::Poisson(rate) => self.run_phase_rate(duration, rate, true).await,
            RequestRate::Constant(rate) => self.run_phase_rate(duration, rate, false).await,
        }
    }

    /// Closed-loop: N workers each send back-to-back requests.
    async fn run_phase_max(
        &self,
        duration: Duration,
    ) -> Result<Vec<RequestRecord>, LlmClientError> {
        let deadline = Instant::now() + duration;
        let mut handles = Vec::new();
        let use_stream = self.config.stream;
        let trace_level = self.config.trace_level.clone();
        let capture_content = self.config.validate.needs_content();

        for worker_id in 0..self.config.concurrency {
            let client = self.client.clone();
            let prompts = self.config.prompts.clone();
            let trace_level = trace_level.clone();

            handles.push(tokio::spawn(async move {
                let mut records = Vec::new();
                let mut prompt_idx = worker_id % prompts.len().max(1);

                while Instant::now() < deadline {
                    let prompt = &prompts[prompt_idx % prompts.len()];
                    records.push(
                        send_one_request(
                            &client,
                            prompt,
                            use_stream,
                            trace_level.as_deref(),
                            capture_content,
                        )
                        .await,
                    );
                    prompt_idx += 1;
                }
                records
            }));
        }

        collect_handles(handles).await
    }

    /// Open-loop: scheduler dispatches requests at the configured rate,
    /// with a bounded concurrency pool. Poisson uses exponential inter-arrival;
    /// constant uses fixed intervals.
    async fn run_phase_rate(
        &self,
        duration: Duration,
        rate: f64,
        poisson: bool,
    ) -> Result<Vec<RequestRecord>, LlmClientError> {
        let deadline = Instant::now() + duration;
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.concurrency));
        let results: Arc<tokio::sync::Mutex<Vec<RequestRecord>>> =
            Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let prompt_idx = Arc::new(AtomicUsize::new(0));
        let capture_content = self.config.validate.needs_content();

        let mut rng_state: u64 = Instant::now().elapsed().as_nanos() as u64;

        while Instant::now() < deadline {
            // Wait for a concurrency slot
            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) if Instant::now() < deadline => p,
                _ => break,
            };

            let idx = prompt_idx.fetch_add(1, Ordering::Relaxed);
            let prompt = self.config.prompts[idx % self.config.prompts.len()].clone();
            let client = self.client.clone();
            let use_stream = self.config.stream;
            let trace_level = self.config.trace_level.clone();
            let results = results.clone();

            tokio::spawn(async move {
                let record = send_one_request(
                    &client,
                    &prompt,
                    use_stream,
                    trace_level.as_deref(),
                    capture_content,
                )
                .await;
                results.lock().await.push(record);
                drop(permit);
            });

            // Inter-arrival delay
            let delay = if poisson {
                // Exponential distribution: -ln(U)/rate, where U ~ Uniform(0,1)
                rng_state = xorshift64(rng_state);
                let u = (rng_state as f64) / (u64::MAX as f64);
                let u = u.max(1e-10); // avoid ln(0)
                Duration::from_secs_f64(-u.ln() / rate)
            } else {
                Duration::from_secs_f64(1.0 / rate)
            };
            tokio::time::sleep(delay).await;
        }

        // Wait for in-flight requests (with timeout)
        let drain_deadline = Instant::now() + Duration::from_secs(30);
        while semaphore.available_permits() < self.config.concurrency
            && Instant::now() < drain_deadline
        {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let records = Arc::try_unwrap(results)
            .map(|mutex| mutex.into_inner())
            .unwrap_or_else(|arc| {
                // Fallback: clone if other refs still exist
                arc.blocking_lock().clone()
            });
        Ok(records)
    }
}

/// Send a single request (streaming or non-streaming) and return a RequestRecord.
async fn send_one_request(
    client: &LlmClient,
    prompt: &ChatRequest,
    use_stream: bool,
    trace_level: Option<&str>,
    capture_content: bool,
) -> RequestRecord {
    if use_stream {
        match client.chat_completion_stream(prompt).await {
            Ok(streamed) => {
                let token_count = streamed.token_timestamps.len() as u32;
                let usage_tokens = streamed
                    .usage
                    .as_ref()
                    .map_or(token_count, |u| u.completion_tokens);
                let prompt_tokens = streamed.usage.as_ref().map_or_else(
                    || estimate_prompt_tokens(&prompt.messages),
                    |u| u.prompt_tokens,
                );
                let content = if capture_content {
                    Some(streamed.content.clone())
                } else {
                    None
                };
                RequestRecord {
                    latency: streamed.latency,
                    ttfb: streamed.ttft,
                    tokens: usage_tokens,
                    prompt_tokens,
                    success: true,
                    token_timestamps: streamed.token_timestamps,
                    brick_trace: None,
                    finish_reason: streamed.finish_reason,
                    response_content: content,
                }
            }
            Err(_) => failed_record(),
        }
    } else {
        let result = if let Some(tl) = trace_level {
            client.send_with_trace(prompt, tl).await
        } else {
            client.send(prompt).await
        };
        match result {
            Ok(timed) => {
                let usage = timed.response.usage.as_ref();
                let tokens = usage.map_or(0, |u| u.completion_tokens);
                let prompt_tokens = usage.map_or(0, |u| u.prompt_tokens);
                let finish_reason = timed
                    .response
                    .choices
                    .first()
                    .and_then(|c| c.finish_reason.clone());
                let content = if capture_content {
                    timed
                        .response
                        .choices
                        .first()
                        .map(|c| c.message.content.clone())
                } else {
                    None
                };
                RequestRecord {
                    latency: timed.latency,
                    ttfb: timed.ttfb,
                    tokens,
                    prompt_tokens,
                    success: true,
                    token_timestamps: Vec::new(),
                    brick_trace: timed.brick_trace,
                    finish_reason,
                    response_content: content,
                }
            }
            Err(_) => failed_record(),
        }
    }
}

fn failed_record() -> RequestRecord {
    RequestRecord {
        latency: Duration::from_millis(0),
        ttfb: Duration::from_millis(0),
        tokens: 0,
        prompt_tokens: 0,
        success: false,
        token_timestamps: Vec::new(),
        brick_trace: None,
        finish_reason: None,
        response_content: None,
    }
}

async fn collect_handles(
    handles: Vec<tokio::task::JoinHandle<Vec<RequestRecord>>>,
) -> Result<Vec<RequestRecord>, LlmClientError> {
    let mut all_records = Vec::new();
    for handle in handles {
        if let Ok(records) = handle.await {
            all_records.extend(records);
        }
    }
    Ok(all_records)
}

/// Fast xorshift64 PRNG for Poisson inter-arrival times.
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// Aggregate individual request records into summary statistics.
fn aggregate_results(
    records: &[RequestRecord],
    elapsed_secs: f64,
    runtime_name: &str,
    concurrency: usize,
    slo_ttft_ms: Option<f64>,
    slo_tpot_ms: Option<f64>,
    slo_latency_ms: Option<f64>,
    num_layers: Option<u32>,
) -> LoadTestResult {
    let total = records.len() as u64;
    let successful = records.iter().filter(|r| r.success).count() as u64;
    let failed = total - successful;

    let mut latencies: Vec<f64> = records
        .iter()
        .filter(|r| r.success)
        .map(|r| r.latency.as_secs_f64() * 1000.0)
        .collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut ttfbs: Vec<f64> = records
        .iter()
        .filter(|r| r.success)
        .map(|r| r.ttfb.as_secs_f64() * 1000.0)
        .collect();
    ttfbs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total_tokens: u64 = records
        .iter()
        .filter(|r| r.success)
        .map(|r| u64::from(r.tokens))
        .sum();
    let total_prompt_tokens: u64 = records
        .iter()
        .filter(|r| r.success)
        .map(|r| u64::from(r.prompt_tokens))
        .sum();

    let throughput_rps = if elapsed_secs > 0.0 {
        successful as f64 / elapsed_secs
    } else {
        0.0
    };

    let tokens_per_sec = if elapsed_secs > 0.0 {
        total_tokens as f64 / elapsed_secs
    } else {
        0.0
    };

    // GH-23: Normalized metrics for cross-backend comparison
    let avg_tok_per_req = if successful > 0 {
        total_tokens as f64 / successful as f64
    } else {
        0.0
    };

    // Inter-token latency and decode throughput.
    let multi_token_records: Vec<&RequestRecord> = records
        .iter()
        .filter(|r| r.success && r.tokens >= 2)
        .collect();

    let has_streaming_timestamps = multi_token_records
        .iter()
        .any(|r| r.token_timestamps.len() >= 2);

    let is_streaming = has_streaming_timestamps
        || multi_token_records.iter().any(|r| {
            let ratio = r.ttfb.as_secs_f64() / r.latency.as_secs_f64().max(1e-9);
            ratio < 0.95
        });

    let mut itls =
        compute_per_token_latencies(&multi_token_records, has_streaming_timestamps, is_streaming);
    itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let itl_p50_ms = percentile(&itls, 0.50);
    let decode_tok_per_sec = if itl_p50_ms > 0.0 {
        1000.0 / itl_p50_ms
    } else {
        0.0
    };

    // TPOT uses same computation as ITL (identical for streaming timestamps, same fallback).
    let mut tpots =
        compute_per_token_latencies(&multi_token_records, has_streaming_timestamps, is_streaming);
    tpots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Prefill throughput: prompt_tokens / ttft_seconds for each request, then median
    let mut prefill_rates: Vec<f64> = records
        .iter()
        .filter(|r| r.success && r.prompt_tokens > 0 && r.ttfb > Duration::ZERO)
        .map(|r| r.prompt_tokens as f64 / r.ttfb.as_secs_f64())
        .collect();
    prefill_rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let prefill_tok_per_sec = percentile(&prefill_rates, 0.50);

    // Latency statistics
    let latency_min_ms = latencies.first().copied().unwrap_or(0.0);
    let latency_max_ms = latencies.last().copied().unwrap_or(0.0);
    let latency_stddev_ms = stddev(&latencies);

    let error_rate = if total > 0 {
        failed as f64 / total as f64
    } else {
        0.0
    };

    let now = chrono::Utc::now().to_rfc3339();

    // Truncation rate: finish_reason="length" means max_tokens hit
    let success_records: Vec<&RequestRecord> = records.iter().filter(|r| r.success).collect();
    let truncated = success_records
        .iter()
        .filter(|r| r.finish_reason.as_deref() == Some("length"))
        .count();
    let truncated_pct = if !success_records.is_empty() {
        truncated as f64 / success_records.len() as f64 * 100.0
    } else {
        0.0
    };

    // Output token distribution
    let mut tok_counts: Vec<f64> = success_records.iter().map(|r| r.tokens as f64).collect();
    tok_counts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let output_tokens_dist = if !tok_counts.is_empty() {
        Some([
            tok_counts[0],
            percentile(&tok_counts, 0.50),
            percentile(&tok_counts, 0.90),
            tok_counts[tok_counts.len() - 1],
        ])
    } else {
        None
    };

    // SSE batch ratio: chunks/tokens. <0.8 means server batches tokens.
    let sse_batch_ratio = if has_streaming_timestamps {
        let total_chunks: usize = multi_token_records
            .iter()
            .filter(|r| !r.token_timestamps.is_empty())
            .map(|r| r.token_timestamps.len())
            .sum();
        let total_toks: u64 = multi_token_records
            .iter()
            .filter(|r| !r.token_timestamps.is_empty())
            .map(|r| u64::from(r.tokens))
            .sum();
        if total_toks > 0 {
            total_chunks as f64 / total_toks as f64
        } else {
            0.0
        }
    } else {
        0.0
    };

    // GH-114: Aggregate brick trace data across requests
    let brick_trace_summary = aggregate_brick_traces(records);

    let request_details = build_request_details(records);
    let goodput_pct = compute_goodput(&request_details, slo_ttft_ms, slo_tpot_ms, slo_latency_ms);

    // Per-layer decode time: TPOT_us / num_layers.
    // Derived from wall-clock ITL — no per-brick sync overhead.
    // Same metric for all runtimes (realizr, llama.cpp, ollama).
    let decode_us_per_layer = num_layers.and_then(|n| {
        if n > 0 && itl_p50_ms > 0.0 {
            Some(itl_p50_ms * 1000.0 / f64::from(n))
        } else {
            None
        }
    });

    LoadTestResult {
        total_requests: total,
        successful,
        failed,
        throughput_rps,
        latency_p50_ms: percentile(&latencies, 0.50),
        latency_p95_ms: percentile(&latencies, 0.95),
        latency_p99_ms: percentile(&latencies, 0.99),
        ttft_p50_ms: percentile(&ttfbs, 0.50),
        tokens_per_sec,
        avg_tok_per_req,
        itl_p50_ms,
        decode_tok_per_sec,
        prefill_tok_per_sec,
        timestamp: now,
        runtime_name: runtime_name.to_string(),
        elapsed_secs,
        concurrency,
        // Extended percentiles
        ttft_p90_ms: percentile(&ttfbs, 0.90),
        ttft_p95_ms: percentile(&ttfbs, 0.95),
        ttft_p99_ms: percentile(&ttfbs, 0.99),
        tpot_p50_ms: percentile(&tpots, 0.50),
        tpot_p90_ms: percentile(&tpots, 0.90),
        tpot_p95_ms: percentile(&tpots, 0.95),
        tpot_p99_ms: percentile(&tpots, 0.99),
        latency_min_ms,
        latency_max_ms,
        latency_stddev_ms,
        error_rate,
        prompt_tokens_total: total_prompt_tokens,
        completion_tokens_total: total_tokens,
        truncated_pct,
        sse_batch_ratio,
        goodput_pct,
        decode_us_per_layer,
        num_layers,
        output_tokens_dist,
        brick_trace_summary,
        request_details,
        quality: None,
        tail_analysis: None,
        gpu_telemetry: None,
        dataset_stats: None,
        cold_start_ms: None,
    }
}

// =============================================================================
// Feature 5: Inline Quality Validation
// =============================================================================

/// Validate all successful responses against the configured validation mode.
fn compute_quality(records: &[RequestRecord], mode: &ValidationMode) -> QualityResult {
    let validation_level = match mode {
        ValidationMode::None => "none".to_string(),
        ValidationMode::Basic => "basic".to_string(),
        ValidationMode::Contains(s) => format!("contains:{s}"),
        ValidationMode::Pattern(p) => format!("pattern:{p}"),
    };

    let compiled_regex = if let ValidationMode::Pattern(p) = mode {
        regex::Regex::new(p).ok()
    } else {
        None
    };

    let mut failures = Vec::new();
    let mut passed_count = 0u64;

    for (idx, record) in records.iter().enumerate() {
        if !record.success {
            continue; // Failed requests already counted in error_rate
        }

        let mut fail_reason = None;

        // Basic checks (all validation levels)
        if record.tokens == 0 {
            fail_reason = Some("zero_tokens".to_string());
        } else if record.finish_reason.is_none() {
            fail_reason = Some("no_finish_reason".to_string());
        }

        // Content checks (contains/pattern only)
        if fail_reason.is_none() {
            match mode {
                ValidationMode::Contains(substring) => {
                    if let Some(ref content) = record.response_content {
                        if content.is_empty() {
                            fail_reason = Some("empty_content".to_string());
                        } else if !content.contains(substring.as_str()) {
                            fail_reason = Some(format!("missing_substring:{substring}"));
                        }
                    } else if record.tokens == 0 {
                        fail_reason = Some("empty_content".to_string());
                    }
                }
                ValidationMode::Pattern(pattern) => {
                    if let Some(ref content) = record.response_content {
                        if content.is_empty() {
                            fail_reason = Some("empty_content".to_string());
                        } else if let Some(ref re) = compiled_regex {
                            if !re.is_match(content) {
                                fail_reason = Some(format!("missing_pattern:{pattern}"));
                            }
                        }
                    } else if record.tokens == 0 {
                        fail_reason = Some("empty_content".to_string());
                    }
                }
                _ => {}
            }
        }

        if let Some(reason) = fail_reason {
            failures.push(QualityFailure {
                request_idx: idx,
                reason,
            });
        } else {
            passed_count += 1;
        }
    }

    let total_validated = passed_count + failures.len() as u64;
    let pass_rate = if total_validated > 0 {
        passed_count as f64 / total_validated as f64
    } else {
        1.0
    };

    QualityResult {
        validation_level,
        total_validated,
        passed: passed_count,
        failed: failures.len() as u64,
        pass_rate,
        failures,
    }
}

// =============================================================================
// Feature 3: Tail Latency Analysis
// =============================================================================

/// Compute tail latency analysis with jitter and drift detection.
fn compute_tail_analysis(records: &[RequestRecord], spike_threshold_mult: f64) -> TailAnalysis {
    let success_records: Vec<&RequestRecord> = records.iter().filter(|r| r.success).collect();

    // Compute per-request ITL values
    let multi_token: Vec<&RequestRecord> = success_records
        .iter()
        .copied()
        .filter(|r| r.tokens >= 2)
        .collect();

    let has_timestamps = multi_token.iter().any(|r| r.token_timestamps.len() >= 2);
    let is_streaming = has_timestamps
        || multi_token.iter().any(|r| {
            let ratio = r.ttfb.as_secs_f64() / r.latency.as_secs_f64().max(1e-9);
            ratio < 0.95
        });

    let mut itls = compute_per_token_latencies(&multi_token.to_vec(), has_timestamps, is_streaming);
    itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut ttfbs: Vec<f64> = success_records
        .iter()
        .map(|r| r.ttfb.as_secs_f64() * 1000.0)
        .collect();
    ttfbs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut latencies: Vec<f64> = success_records
        .iter()
        .map(|r| r.latency.as_secs_f64() * 1000.0)
        .collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Extended percentiles
    let itl_p50 = percentile(&itls, 0.50);
    let itl_p99 = percentile(&itls, 0.99);
    let ttft_p50 = percentile(&ttfbs, 0.50);
    let ttft_p99 = percentile(&ttfbs, 0.99);
    let lat_p50 = percentile(&latencies, 0.50);
    let lat_p99 = percentile(&latencies, 0.99);

    // Tail ratios
    let tail_ratio_itl = if itl_p50 > 0.0 {
        itl_p99 / itl_p50
    } else {
        0.0
    };
    let tail_ratio_ttft = if ttft_p50 > 0.0 {
        ttft_p99 / ttft_p50
    } else {
        0.0
    };
    let tail_ratio_latency = if lat_p50 > 0.0 {
        lat_p99 / lat_p50
    } else {
        0.0
    };

    // Jitter: coefficient of variation
    let itl_mean = if itls.is_empty() {
        0.0
    } else {
        itls.iter().sum::<f64>() / itls.len() as f64
    };
    let itl_sd = stddev(&itls);
    let itl_cv = if itl_mean > 0.0 {
        itl_sd / itl_mean
    } else {
        0.0
    };

    // IQR
    let itl_iqr_ms = percentile(&itls, 0.75) - percentile(&itls, 0.25);

    // Spike detection
    let spike_threshold_ms = itl_p50 * spike_threshold_mult;
    let mut spikes = Vec::new();
    for (idx, &itl) in itls.iter().enumerate() {
        if itl > spike_threshold_ms && spike_threshold_ms > 0.0 && spikes.len() < 20 {
            spikes.push(LatencySpike {
                request_idx: idx,
                itl_ms: itl,
            });
        }
    }

    // Drift detection: simple linear regression of ITL over request index
    let drift = compute_drift(&itls, &ttfbs);

    TailAnalysis {
        itl_p999_ms: percentile(&itls, 0.999),
        itl_p9999_ms: percentile(&itls, 0.9999),
        ttft_p999_ms: percentile(&ttfbs, 0.999),
        ttft_p9999_ms: percentile(&ttfbs, 0.9999),
        latency_p999_ms: percentile(&latencies, 0.999),
        latency_p9999_ms: percentile(&latencies, 0.9999),
        tail_ratio_itl,
        tail_ratio_ttft,
        tail_ratio_latency,
        jitter: JitterAnalysis {
            itl_cv,
            itl_iqr_ms,
            spike_count: spikes.len(),
            spike_threshold_ms,
            spikes,
        },
        drift,
    }
}

/// Simple linear regression: y = a + b*x. Returns (slope, r_squared).
fn linear_regression(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    if n < 3.0 {
        return (0.0, 0.0);
    }
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;
    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        ss_xy += (x - x_mean) * (y - y_mean);
        ss_xx += (x - x_mean).powi(2);
        ss_yy += (y - y_mean).powi(2);
    }
    let slope = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
    let r_squared = if ss_yy > 0.0 {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    } else {
        0.0
    };
    (slope, r_squared)
}

/// Compute drift analysis from time-ordered ITL and TTFT values.
fn compute_drift(itls: &[f64], ttfbs: &[f64]) -> DriftAnalysis {
    let (itl_slope, itl_r2) = linear_regression(itls);
    let (ttft_slope, _ttft_r2) = linear_regression(ttfbs);

    // Convert slope from ms/request to ms/min.
    // Slope is per-index; multiply by N for per-run-duration scale.
    let itl_slope_per_min = itl_slope * (itls.len() as f64).max(1.0);
    let ttft_slope_per_min = ttft_slope * (ttfbs.len() as f64).max(1.0);

    // Drift detection requires:
    // 1. At least 30 data points (small N gives spurious r^2)
    // 2. Positive slope (degradation, not improvement)
    // 3. Strong correlation (r^2 > 0.5)
    // 4. Practically significant slope (> 1% of median ITL per minute)
    let itl_median = if itls.is_empty() {
        0.0
    } else {
        let mid = itls.len() / 2;
        itls[mid]
    };
    let min_slope = itl_median * 0.01; // 1% of median per run
    let degradation_detected =
        itls.len() >= 30 && itl_slope > 0.0 && itl_r2 > 0.5 && itl_slope_per_min > min_slope;

    DriftAnalysis {
        itl_slope_ms_per_min: itl_slope_per_min,
        ttft_slope_ms_per_min: ttft_slope_per_min,
        degradation_detected,
    }
}

/// Aggregate BrickProfiler traces across requests (GH-114).
/// Compute per-token latencies (ITL/TPOT) from request records.
/// Uses real SSE timestamps when available, falls back to request-level approximation.
fn compute_per_token_latencies(
    multi_token_records: &[&RequestRecord],
    has_streaming_timestamps: bool,
    is_streaming: bool,
) -> Vec<f64> {
    if has_streaming_timestamps {
        // Per-request mean from SSE timestamps: (last_ts - first_ts) / (n - 1).
        // Robust to token batching (servers often send 2+ tokens per SSE chunk).
        multi_token_records
            .iter()
            .filter(|r| r.token_timestamps.len() >= 2)
            .map(|r| {
                let first = &r.token_timestamps[0];
                let last = &r.token_timestamps[r.token_timestamps.len() - 1];
                let decode_ms = (*last).checked_sub(*first).unwrap().as_secs_f64() * 1000.0;
                decode_ms / (r.token_timestamps.len() - 1) as f64
            })
            .collect()
    } else if is_streaming {
        // Streaming without timestamps: ITL from decode phase
        multi_token_records
            .iter()
            .map(|r| {
                let decode_ms = (r.latency.as_secs_f64() - r.ttfb.as_secs_f64()) * 1000.0;
                decode_ms / (r.tokens as f64 - 1.0)
            })
            .collect()
    } else {
        // Non-streaming: per-request throughput as proxy
        multi_token_records
            .iter()
            .map(|r| r.latency.as_secs_f64() * 1000.0 / r.tokens as f64)
            .collect()
    }
}

/// Build per-request detail records for distribution analysis and SLO evaluation.
fn build_request_details(records: &[RequestRecord]) -> Vec<RequestDetail> {
    records
        .iter()
        .filter(|r| r.success)
        .map(|r| {
            let itl_ms = if r.token_timestamps.len() >= 2 {
                let first = &r.token_timestamps[0];
                let last = &r.token_timestamps[r.token_timestamps.len() - 1];
                let decode_ms = (*last).checked_sub(*first).unwrap().as_secs_f64() * 1000.0;
                decode_ms / (r.token_timestamps.len() - 1) as f64
            } else if r.tokens >= 2 {
                let decode_ms = (r.latency.as_secs_f64() - r.ttfb.as_secs_f64()) * 1000.0;
                decode_ms / (r.tokens as f64 - 1.0)
            } else {
                0.0
            };
            RequestDetail {
                latency_ms: r.latency.as_secs_f64() * 1000.0,
                ttft_ms: r.ttfb.as_secs_f64() * 1000.0,
                completion_tokens: r.tokens,
                prompt_tokens: r.prompt_tokens,
                itl_ms,
                finish_reason: r.finish_reason.clone(),
            }
        })
        .collect()
}

/// Compute goodput: percentage of requests meeting all configured SLO thresholds.
fn compute_goodput(
    details: &[RequestDetail],
    slo_ttft_ms: Option<f64>,
    slo_tpot_ms: Option<f64>,
    slo_latency_ms: Option<f64>,
) -> f64 {
    if slo_ttft_ms.is_none() && slo_tpot_ms.is_none() && slo_latency_ms.is_none() {
        return 0.0;
    }
    if details.is_empty() {
        return 0.0;
    }
    let passing = details
        .iter()
        .filter(|d| {
            let ttft_ok = slo_ttft_ms.map_or(true, |t| d.ttft_ms <= t);
            let tpot_ok = slo_tpot_ms.map_or(true, |t| d.itl_ms <= t || d.itl_ms == 0.0);
            let lat_ok = slo_latency_ms.map_or(true, |t| d.latency_ms <= t);
            ttft_ok && tpot_ok && lat_ok
        })
        .count();
    passing as f64 / details.len() as f64 * 100.0
}

fn aggregate_brick_traces(records: &[RequestRecord]) -> Option<Vec<BrickTraceOpSummary>> {
    let traces: Vec<&BrickTrace> = records
        .iter()
        .filter(|r| r.success)
        .filter_map(|r| r.brick_trace.as_ref())
        .collect();

    if traces.is_empty() {
        return None;
    }

    // Collect per-operation times across all requests
    let mut op_times: HashMap<String, Vec<f64>> = HashMap::new();
    let mut total_time_sum: f64 = 0.0;

    for trace in &traces {
        total_time_sum += trace.total_time_us as f64;
        for op in &trace.breakdown {
            op_times
                .entry(op.name.clone())
                .or_default()
                .push(op.time_us as f64);
        }
    }

    let avg_total = total_time_sum / traces.len() as f64;
    let mut summaries: Vec<BrickTraceOpSummary> = op_times
        .into_iter()
        .map(|(name, times)| {
            let n = times.len();
            let sum: f64 = times.iter().sum();
            let mean = sum / n as f64;
            let min = times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let pct = if avg_total > 0.0 {
                (mean / avg_total) * 100.0
            } else {
                0.0
            };
            BrickTraceOpSummary {
                name,
                mean_us: mean,
                min_us: min,
                max_us: max,
                pct_of_total: pct,
                samples: n,
            }
        })
        .collect();

    // Sort by percentage descending (hottest first)
    summaries.sort_by(|a, b| {
        b.pct_of_total
            .partial_cmp(&a.pct_of_total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(summaries)
}

/// Compute a percentile from a sorted slice using linear interpolation.
/// Standard method matching numpy/R default. Returns 0.0 for empty slices.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (sorted.len() as f64 - 1.0) * p;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute standard deviation of a slice of f64 values.
fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// Estimate prompt tokens from message content when server doesn't report usage.
/// Uses words × 1.3 heuristic (approximation for GPT-2/BPE tokenizers).
fn estimate_prompt_tokens(messages: &[ChatMessage]) -> u32 {
    let total_words: usize = messages
        .iter()
        .map(|m| m.content.split_whitespace().count() + 4) // +4 for role/delimiter tokens
        .sum();
    (total_words as f64 * 1.3) as u32
}

/// Default prompt for load testing.
fn default_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "What is 2 + 2? Reply with just the number.".to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(16),
        stream: Some(false),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42.0], 0.5), 42.0);
        assert_eq!(percentile(&[42.0], 0.99), 42.0);
    }

    #[test]
    fn test_percentile_multiple() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        // Linear interpolation: idx = 99 * p, lerp between floor and ceil
        // p50: idx=49.5, lerp(50, 51, 0.5) = 50.5
        assert!((percentile(&data, 0.50) - 50.5).abs() < 0.01);
        // p95: idx=94.05, lerp(95, 96, 0.05) = 95.05
        assert!((percentile(&data, 0.95) - 95.05).abs() < 0.01);
        // p99: idx=98.01, lerp(99, 100, 0.01) = 99.01
        assert!((percentile(&data, 0.99) - 99.01).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_empty() {
        let result = aggregate_results(&[], 10.0, "test", 1, None, None, None, None);
        assert_eq!(result.total_requests, 0);
        assert_eq!(result.successful, 0);
        assert_eq!(result.failed, 0);
        assert_eq!(result.throughput_rps, 0.0);
        assert_eq!(result.latency_p50_ms, 0.0);
        assert_eq!(result.error_rate, 0.0);
        assert_eq!(result.prompt_tokens_total, 0);
        assert_eq!(result.completion_tokens_total, 0);
    }

    #[test]
    fn test_aggregate_all_success() {
        let records: Vec<RequestRecord> = (0..10)
            .map(|i| RequestRecord {
                latency: Duration::from_millis(100 + i * 10),
                ttfb: Duration::from_millis(50 + i * 5),
                tokens: 20,
                prompt_tokens: 10,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            })
            .collect();
        let result = aggregate_results(&records, 10.0, "realizar", 2, None, None, None, None);
        assert_eq!(result.total_requests, 10);
        assert_eq!(result.successful, 10);
        assert_eq!(result.failed, 0);
        assert!((result.throughput_rps - 1.0).abs() < f64::EPSILON);
        assert!(result.latency_p50_ms > 0.0);
        assert!(result.tokens_per_sec > 0.0);
        // GH-23: normalized metrics
        assert!((result.avg_tok_per_req - 20.0).abs() < f64::EPSILON);
        assert!(result.itl_p50_ms > 0.0);
        assert!(result.decode_tok_per_sec > 0.0);
        assert_eq!(result.runtime_name, "realizar");
        assert_eq!(result.concurrency, 2);
        // Extended percentiles
        assert!(result.ttft_p90_ms > 0.0);
        assert!(result.ttft_p95_ms > 0.0);
        assert!(result.ttft_p99_ms > 0.0);
        assert!(result.tpot_p50_ms > 0.0);
        assert!(result.latency_min_ms > 0.0);
        assert!(result.latency_max_ms >= result.latency_min_ms);
        assert!(result.latency_stddev_ms >= 0.0);
        assert!((result.error_rate).abs() < f64::EPSILON);
        assert_eq!(result.prompt_tokens_total, 100);
        assert_eq!(result.completion_tokens_total, 200);
    }

    #[test]
    fn test_aggregate_mixed() {
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(0),
                ttfb: Duration::from_millis(0),
                tokens: 0,
                prompt_tokens: 0,
                success: false,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
        ];
        let result = aggregate_results(&records, 5.0, "ollama", 1, None, None, None, None);
        assert_eq!(result.total_requests, 2);
        assert_eq!(result.successful, 1);
        assert_eq!(result.failed, 1);
        assert!((result.error_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_config() {
        let config = LoadTestConfig::default();
        assert_eq!(config.concurrency, 1);
        assert_eq!(config.duration, Duration::from_secs(30));
        assert_eq!(config.prompts.len(), 1);
        assert_eq!(config.warmup_duration, Duration::ZERO);
    }

    #[test]
    fn test_default_prompt() {
        let p = default_prompt();
        assert_eq!(p.messages.len(), 1);
        assert_eq!(p.messages[0].role, Role::User);
        assert_eq!(p.temperature, Some(0.0));
    }

    #[test]
    fn test_load_test_result_serialization() {
        let result = LoadTestResult {
            total_requests: 100,
            successful: 95,
            failed: 5,
            throughput_rps: 10.0,
            latency_p50_ms: 150.0,
            latency_p95_ms: 300.0,
            latency_p99_ms: 500.0,
            ttft_p50_ms: 80.0,
            tokens_per_sec: 200.0,
            avg_tok_per_req: 15.0,
            itl_p50_ms: 5.0,
            decode_tok_per_sec: 200.0,
            prefill_tok_per_sec: 0.0,
            timestamp: "2026-03-01T00:00:00Z".to_string(),
            runtime_name: "realizar".to_string(),
            elapsed_secs: 10.0,
            concurrency: 4,
            ttft_p90_ms: 90.0,
            ttft_p95_ms: 95.0,
            ttft_p99_ms: 99.0,
            tpot_p50_ms: 6.0,
            tpot_p90_ms: 8.0,
            tpot_p95_ms: 9.0,
            tpot_p99_ms: 12.0,
            latency_min_ms: 50.0,
            latency_max_ms: 800.0,
            latency_stddev_ms: 120.0,
            error_rate: 0.05,
            prompt_tokens_total: 950,
            completion_tokens_total: 1425,
            truncated_pct: 0.0,
            sse_batch_ratio: 0.0,
            goodput_pct: 0.0,
            decode_us_per_layer: None,
            num_layers: None,
            output_tokens_dist: None,
            brick_trace_summary: None,
            request_details: Vec::new(),
            quality: None,
            tail_analysis: None,
            gpu_telemetry: None,
            dataset_stats: None,
            cold_start_ms: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: LoadTestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_requests, 100);
        assert_eq!(back.runtime_name, "realizar");
        assert!((back.avg_tok_per_req - 15.0).abs() < f64::EPSILON);
        assert!((back.itl_p50_ms - 5.0).abs() < f64::EPSILON);
        assert!((back.decode_tok_per_sec - 200.0).abs() < f64::EPSILON);
        assert!((back.tpot_p50_ms - 6.0).abs() < f64::EPSILON);
        assert!((back.error_rate - 0.05).abs() < f64::EPSILON);
        assert_eq!(back.prompt_tokens_total, 950);
        assert_eq!(back.completion_tokens_total, 1425);
    }

    #[test]
    fn test_load_test_result_backwards_compat() {
        // Old JSON without new fields should deserialize with defaults
        let json = r#"{
            "total_requests": 50,
            "successful": 50,
            "failed": 0,
            "throughput_rps": 5.0,
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 200.0,
            "latency_p99_ms": 300.0,
            "ttft_p50_ms": 50.0,
            "tokens_per_sec": 100.0,
            "timestamp": "2026-01-01T00:00:00Z",
            "runtime_name": "old",
            "elapsed_secs": 10.0,
            "concurrency": 1
        }"#;
        let result: LoadTestResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.total_requests, 50);
        assert_eq!(result.tpot_p50_ms, 0.0);
        assert_eq!(result.error_rate, 0.0);
        assert_eq!(result.prompt_tokens_total, 0);
    }

    #[test]
    fn test_percentile_boundary() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 1.0), 3.0);
    }

    #[test]
    fn test_itl_streaming() {
        // GH-23: Streaming mode — ITL = (latency - ttfb) / (tokens - 1)
        // Request: 200ms latency, 50ms ttfb, 16 tokens
        // ttfb/latency = 0.25 < 0.95 → streaming detected
        // Decode time = 200 - 50 = 150ms, ITL = 150 / 15 = 10ms
        let records = vec![RequestRecord {
            latency: Duration::from_millis(200),
            ttfb: Duration::from_millis(50),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert!((result.itl_p50_ms - 10.0).abs() < 0.1);
        assert!((result.decode_tok_per_sec - 100.0).abs() < 1.0);
        assert!((result.avg_tok_per_req - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_itl_non_streaming() {
        // GH-23: Non-streaming — ttfb ≈ latency, fallback to latency/tokens
        // Request: 1600ms latency, 1599ms ttfb, 16 tokens
        // ttfb/latency = 0.999 > 0.95 → non-streaming detected
        // ITL proxy = 1600 / 16 = 100ms
        let records = vec![RequestRecord {
            latency: Duration::from_millis(1600),
            ttfb: Duration::from_millis(1599),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert!((result.itl_p50_ms - 100.0).abs() < 0.1);
        assert!((result.decode_tok_per_sec - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_itl_single_token_excluded() {
        // GH-23: Requests with < 2 tokens should be excluded from ITL
        // (can't compute inter-token latency with 0 or 1 token)
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(100),
            tokens: 1,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert_eq!(result.itl_p50_ms, 0.0);
        assert_eq!(result.decode_tok_per_sec, 0.0);
        assert!((result.avg_tok_per_req - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_zero_elapsed() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 10,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 0.0, "test", 1, None, None, None, None);
        assert_eq!(result.throughput_rps, 0.0);
        assert_eq!(result.tokens_per_sec, 0.0);
    }

    #[test]
    fn test_stddev() {
        assert_eq!(stddev(&[]), 0.0);
        assert_eq!(stddev(&[5.0]), 0.0);
        // [10, 20, 30]: mean=20, var=((100+0+100)/2)=100, stddev=10
        let sd = stddev(&[10.0, 20.0, 30.0]);
        assert!((sd - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_tpot_computation() {
        // TPOT = (latency - ttfb) / (tokens - 1)
        // Streaming: 200ms latency, 50ms ttfb, 16 tokens
        // TPOT = (200 - 50) / 15 = 10ms
        let records = vec![RequestRecord {
            latency: Duration::from_millis(200),
            ttfb: Duration::from_millis(50),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert!((result.tpot_p50_ms - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_latency_min_max_stddev() {
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(300),
                ttfb: Duration::from_millis(100),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
        ];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert!((result.latency_min_ms - 100.0).abs() < 0.1);
        assert!((result.latency_max_ms - 300.0).abs() < 0.1);
        assert!(result.latency_stddev_ms > 0.0);
    }

    #[test]
    fn test_prompt_tokens_tracking() {
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 20,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 15,
                prompt_tokens: 25,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
        ];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert_eq!(result.prompt_tokens_total, 45);
        assert_eq!(result.completion_tokens_total, 25);
    }

    #[test]
    fn test_tpot_from_streaming_timestamps() {
        // GH-24: When token_timestamps are available, TPOT uses real per-token deltas.
        // 5 tokens arriving at 50ms, 60ms, 70ms, 80ms, 90ms
        // Inter-token deltas: 10ms, 10ms, 10ms, 10ms → mean TPOT = 10ms
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 5,
            prompt_tokens: 10,
            success: true,
            token_timestamps: vec![
                Duration::from_millis(50),
                Duration::from_millis(60),
                Duration::from_millis(70),
                Duration::from_millis(80),
                Duration::from_millis(90),
            ],
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        // Real TPOT from timestamps: mean of [10, 10, 10, 10] = 10ms
        assert!((result.tpot_p50_ms - 10.0).abs() < 0.1);
        // ITL also uses real timestamps
        assert!((result.itl_p50_ms - 10.0).abs() < 0.1);
        assert!((result.decode_tok_per_sec - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_tpot_mixed_streaming_and_non_streaming() {
        // GH-24: When some records have timestamps and some don't,
        // only records with timestamps >= 2 are used for streaming TPOT.
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(200),
                ttfb: Duration::from_millis(50),
                tokens: 4,
                prompt_tokens: 10,
                success: true,
                token_timestamps: vec![
                    Duration::from_millis(50),
                    Duration::from_millis(70),
                    Duration::from_millis(90),
                    Duration::from_millis(110),
                ],
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 5,
                prompt_tokens: 10,
                success: true,
                token_timestamps: Vec::new(), // non-streaming request
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
        ];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        // Only the first record with timestamps is used for TPOT
        // Deltas: [20, 20, 20] → mean TPOT = 20ms
        assert!((result.tpot_p50_ms - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_stream_config_default() {
        let config = LoadTestConfig::default();
        assert!(!config.stream);
    }

    #[test]
    fn test_tpot_non_streaming_uses_latency_per_token() {
        // Non-streaming: ttfb ≈ latency → TPOT should use latency/tokens (not near-zero).
        // Before fix: TPOT = (latency - ttfb)/(tokens-1) = (1600-1599)/15 = 0.067ms (WRONG)
        // After fix: TPOT = latency/tokens = 1600/16 = 100ms (correct, matches ITL)
        let records = vec![RequestRecord {
            latency: Duration::from_millis(1600),
            ttfb: Duration::from_millis(1599),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        // Both TPOT and ITL should be latency/tokens = 100ms
        assert!(
            (result.tpot_p50_ms - 100.0).abs() < 0.1,
            "tpot={}",
            result.tpot_p50_ms
        );
        assert!(
            (result.itl_p50_ms - 100.0).abs() < 0.1,
            "itl={}",
            result.itl_p50_ms
        );
    }

    #[test]
    fn test_itl_robust_to_token_batching() {
        // Server sends tokens in pairs (batch=2): timestamps are [100, 100, 200, 200, 300]
        // Old code (flat_map): deltas = [0, 100, 0, 100] → P50 = 50ms (bimodal, fragile)
        // New code (per-request mean): (300-100)/4 = 50ms (robust)
        // With batch=3: timestamps = [100, 100, 100, 300, 300, 300]
        // Old code: deltas = [0, 0, 200, 0, 0] → P50 = 0ms (WRONG)
        // New code: (300-100)/5 = 40ms (correct)
        let records = vec![RequestRecord {
            latency: Duration::from_millis(350),
            ttfb: Duration::from_millis(100),
            tokens: 6,
            prompt_tokens: 10,
            success: true,
            token_timestamps: vec![
                Duration::from_millis(100), // batch 1
                Duration::from_millis(100),
                Duration::from_millis(100),
                Duration::from_millis(300), // batch 2
                Duration::from_millis(300),
                Duration::from_millis(300),
            ],
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        // Per-request mean: (300-100)/5 = 40ms
        assert!(
            (result.itl_p50_ms - 40.0).abs() < 0.1,
            "itl={}",
            result.itl_p50_ms
        );
        assert!(
            (result.tpot_p50_ms - 40.0).abs() < 0.1,
            "tpot={}",
            result.tpot_p50_ms
        );
        assert!(
            (result.decode_tok_per_sec - 25.0).abs() < 0.5,
            "decode={}",
            result.decode_tok_per_sec
        );
    }

    #[test]
    fn test_request_details_populated() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(200),
            ttfb: Duration::from_millis(50),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert_eq!(result.request_details.len(), 1);
        let detail = &result.request_details[0];
        assert!((detail.latency_ms - 200.0).abs() < 0.1);
        assert!((detail.ttft_ms - 50.0).abs() < 0.1);
        assert_eq!(detail.completion_tokens, 16);
        assert_eq!(detail.prompt_tokens, 10);
        assert!(detail.itl_ms > 0.0);
    }

    // =========================================================================
    // Feature 5: Quality validation tests
    // =========================================================================

    #[test]
    fn test_quality_basic_all_pass() {
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(120),
                ttfb: Duration::from_millis(60),
                tokens: 8,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            },
        ];
        let quality = compute_quality(&records, &ValidationMode::Basic);
        assert_eq!(quality.total_validated, 2);
        assert_eq!(quality.passed, 2);
        assert_eq!(quality.failed, 0);
        assert!((quality.pass_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_basic_zero_tokens() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(100),
            tokens: 0,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: Some("stop".to_string()),
            response_content: None,
        }];
        let quality = compute_quality(&records, &ValidationMode::Basic);
        assert_eq!(quality.failed, 1);
        assert_eq!(quality.failures[0].reason, "zero_tokens");
    }

    #[test]
    fn test_quality_basic_no_finish_reason() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 10,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let quality = compute_quality(&records, &ValidationMode::Basic);
        assert_eq!(quality.failed, 1);
        assert_eq!(quality.failures[0].reason, "no_finish_reason");
    }

    #[test]
    fn test_quality_contains_match() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 10,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: Some("stop".to_string()),
            response_content: Some("hello world".to_string()),
        }];
        let quality = compute_quality(&records, &ValidationMode::Contains("hello".to_string()));
        assert_eq!(quality.passed, 1);
        assert_eq!(quality.failed, 0);
    }

    #[test]
    fn test_quality_contains_mismatch() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 10,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: Some("stop".to_string()),
            response_content: Some("goodbye world".to_string()),
        }];
        let quality = compute_quality(&records, &ValidationMode::Contains("hello".to_string()));
        assert_eq!(quality.failed, 1);
        assert!(quality.failures[0].reason.starts_with("missing_substring:"));
    }

    #[test]
    fn test_quality_none_skipped() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 0,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        // ValidationMode::None should still return results if called directly
        let quality = compute_quality(&records, &ValidationMode::None);
        // But in practice, LoadTest::run() skips calling compute_quality when mode is None
        assert_eq!(quality.validation_level, "none");
    }

    #[test]
    fn test_quality_skips_failed_requests() {
        let records = vec![
            failed_record(), // success: false
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            },
        ];
        let quality = compute_quality(&records, &ValidationMode::Basic);
        // Only the successful request should be validated
        assert_eq!(quality.total_validated, 1);
        assert_eq!(quality.passed, 1);
    }

    // =========================================================================
    // Feature 3: Tail latency analysis tests
    // =========================================================================

    #[test]
    fn test_tail_analysis_basic() {
        let records: Vec<RequestRecord> = (0..100)
            .map(|i| RequestRecord {
                latency: Duration::from_millis(100 + i),
                ttfb: Duration::from_millis(50 + i / 2),
                tokens: 20,
                prompt_tokens: 10,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            })
            .collect();
        let tail = compute_tail_analysis(&records, 5.0);
        // P99.9 should be near the max
        assert!(tail.latency_p999_ms > 0.0);
        assert!(tail.ttft_p999_ms > 0.0);
        // Tail ratios should be computed
        assert!(tail.tail_ratio_latency > 0.0);
    }

    #[test]
    fn test_spike_detection() {
        // Create records with one outlier
        let mut records: Vec<RequestRecord> = (0..50)
            .map(|_| RequestRecord {
                latency: Duration::from_millis(200),
                ttfb: Duration::from_millis(50),
                tokens: 16,
                prompt_tokens: 10,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            })
            .collect();
        // Add a spike (10x normal latency)
        records.push(RequestRecord {
            latency: Duration::from_millis(2000),
            ttfb: Duration::from_millis(50),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: Some("stop".to_string()),
            response_content: None,
        });
        let tail = compute_tail_analysis(&records, 5.0);
        // The spike should be detected (its ITL is much higher than median)
        assert!(tail.jitter.spike_threshold_ms > 0.0);
    }

    #[test]
    fn test_linear_regression() {
        // Perfect positive slope: y = 2x
        let values: Vec<f64> = (0..10).map(|x| 2.0 * x as f64).collect();
        let (slope, r2) = linear_regression(&values);
        assert!((slope - 2.0).abs() < 0.01);
        assert!((r2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_flat() {
        let values = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (slope, _r2) = linear_regression(&values);
        assert!(slope.abs() < 0.01);
    }

    #[test]
    fn test_validation_mode_parse() {
        assert!(matches!(
            ValidationMode::parse("none"),
            ValidationMode::None
        ));
        assert!(matches!(
            ValidationMode::parse("basic"),
            ValidationMode::Basic
        ));
        if let ValidationMode::Contains(s) = ValidationMode::parse("contains:hello") {
            assert_eq!(s, "hello");
        } else {
            panic!("Expected Contains");
        }
        if let ValidationMode::Pattern(p) = ValidationMode::parse("pattern:\\d+") {
            assert_eq!(p, "\\d+");
        } else {
            panic!("Expected Pattern");
        }
    }

    #[test]
    fn test_tail_analysis_empty() {
        let records: Vec<RequestRecord> = Vec::new();
        let tail = compute_tail_analysis(&records, 5.0);
        assert_eq!(tail.itl_p999_ms, 0.0);
        assert_eq!(tail.jitter.spike_count, 0);
        assert!(!tail.drift.degradation_detected);
    }
}
