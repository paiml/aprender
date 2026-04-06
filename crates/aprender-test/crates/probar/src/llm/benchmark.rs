//! Full benchmark orchestrator for LLM inference endpoints.
//!
//! Orchestrates the complete lifecycle: server start, health poll, warmup,
//! multi-run measurement, statistical analysis, baseline comparison, and teardown.

use super::client::{ChatRequest, LlmClient, LlmClientError};
use super::loadtest::{LoadTest, LoadTestConfig, LoadTestResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for a full benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Base URL of the LLM API server.
    pub url: String,
    /// Model name.
    pub model: String,
    /// Shell command to start the server (optional).
    pub start_command: Option<String>,
    /// Maximum time to wait for server readiness.
    pub health_timeout: Duration,
    /// Warmup duration (excluded from metrics).
    pub warmup: Duration,
    /// Per-run measurement duration.
    pub duration: Duration,
    /// Number of concurrent workers.
    pub concurrency: usize,
    /// Number of measurement runs.
    pub runs: usize,
    /// Cooldown between runs.
    pub cooldown: Duration,
    /// Prompts to use.
    pub prompts: Vec<ChatRequest>,
    /// Name of the runtime being benchmarked.
    pub runtime_name: String,
    /// Baseline result for regression detection.
    pub baseline: Option<LoadTestResult>,
    /// Percentage threshold for regression detection.
    pub fail_on_regression: Option<f64>,
    /// Use SSE streaming for per-token TPOT measurement (GH-24).
    pub stream: bool,
    /// Trace level for BrickProfiler data collection (GH-114).
    pub trace_level: Option<String>,
    /// Number of transformer layers for per-layer decode time computation.
    pub num_layers: Option<u32>,
}

/// Complete benchmark report with per-run results and cross-run statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Individual run results.
    pub runs: Vec<LoadTestResult>,
    /// Aggregate statistics across runs.
    pub aggregate: AggregateStats,
    /// Regressions detected vs. baseline.
    pub regressions: Vec<Regression>,
}

/// Cross-run aggregate statistics with confidence intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStats {
    /// Throughput (req/s) statistics.
    pub throughput_rps: StatSummary,
    /// Median latency statistics.
    pub latency_p50: StatSummary,
    /// Tokens per second statistics.
    pub tokens_per_sec: StatSummary,
    /// TTFT P50 statistics.
    pub ttft_p50: StatSummary,
    /// TPOT P50 statistics.
    pub tpot_p50: StatSummary,
}

/// Summary statistics for a single metric across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatSummary {
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation.
    pub stddev: f64,
    /// Lower bound of 95% confidence interval.
    pub ci_95_lower: f64,
    /// Upper bound of 95% confidence interval.
    pub ci_95_upper: f64,
    /// Individual run values.
    pub values: Vec<f64>,
}

/// A metric regression compared to baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    /// Metric name (e.g., "throughput_rps").
    pub metric: String,
    /// Baseline value.
    pub baseline_value: f64,
    /// Current value.
    pub current_value: f64,
    /// Percentage change (negative = regression for throughput, positive = regression for latency).
    pub change_pct: f64,
    /// Whether this regression exceeds the configured threshold.
    pub exceeds_threshold: bool,
}

/// Benchmark executor.
pub struct Benchmark {
    config: BenchmarkConfig,
    child: Option<tokio::process::Child>,
}

impl std::fmt::Debug for Benchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Benchmark")
            .field("config", &self.config)
            .field("child", &self.child.as_ref().map(|c| c.id()))
            .finish()
    }
}

impl Benchmark {
    /// Create a new benchmark from configuration.
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            child: None,
        }
    }

    /// Run the full benchmark lifecycle.
    pub async fn run(&mut self) -> Result<BenchmarkReport, LlmClientError> {
        // Phase 1: Start server (if configured)
        if let Some(ref cmd) = self.config.start_command {
            let child = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map_err(|e| {
                    LlmClientError::HealthCheckFailed(format!("Failed to start server: {e}"))
                })?;
            self.child = Some(child);
        }

        let client = LlmClient::new(&self.config.url, &self.config.model);

        // Phase 1b: Wait for server readiness
        let ready_time = client
            .wait_ready(self.config.health_timeout, Duration::from_secs(2))
            .await?;
        eprintln!("Server ready in {:.1}s", ready_time.as_secs_f64());

        // Phase 2: Warmup (excluded from metrics)
        if self.config.warmup > Duration::ZERO {
            eprintln!("Warming up for {:.0}s...", self.config.warmup.as_secs_f64());
            let warmup_config = LoadTestConfig {
                concurrency: self.config.concurrency,
                duration: self.config.warmup,
                prompts: self.config.prompts.clone(),
                runtime_name: self.config.runtime_name.clone(),
                warmup_duration: Duration::ZERO,
                stream: self.config.stream,
                trace_level: None, // No tracing during warmup
                slo_ttft_ms: None,
                slo_tpot_ms: None,
                slo_latency_ms: None,
                num_layers: self.config.num_layers,
                rate: super::loadtest::RequestRate::Max,
                validate: super::loadtest::ValidationMode::None,
                spike_threshold: 5.0,
                fail_on_quality: None,
            };
            let warmup_test = LoadTest::new(client.clone(), warmup_config);
            let _ = warmup_test.run().await;
        }

        // Phase 3: Measure (repeated N times)
        let mut run_results = Vec::with_capacity(self.config.runs);
        for i in 0..self.config.runs {
            eprintln!(
                "Run {}/{} ({:.0}s)...",
                i + 1,
                self.config.runs,
                self.config.duration.as_secs_f64()
            );
            let measure_config = LoadTestConfig {
                concurrency: self.config.concurrency,
                duration: self.config.duration,
                prompts: self.config.prompts.clone(),
                runtime_name: self.config.runtime_name.clone(),
                warmup_duration: Duration::ZERO,
                stream: self.config.stream,
                trace_level: self.config.trace_level.clone(),
                slo_ttft_ms: None,
                slo_tpot_ms: None,
                slo_latency_ms: None,
                rate: super::loadtest::RequestRate::Max,
                num_layers: self.config.num_layers,
                validate: super::loadtest::ValidationMode::None,
                spike_threshold: 5.0,
                fail_on_quality: None,
            };
            let load_test = LoadTest::new(client.clone(), measure_config);
            let result = load_test.run().await?;
            run_results.push(result);

            // Cooldown between runs (except after last)
            if i + 1 < self.config.runs && self.config.cooldown > Duration::ZERO {
                tokio::time::sleep(self.config.cooldown).await;
            }
        }

        // Phase 4: Analyze
        let aggregate = compute_aggregate(&run_results);
        let regressions = if let Some(ref baseline) = self.config.baseline {
            let threshold = self.config.fail_on_regression.unwrap_or(10.0);
            super::report::compare_to_baseline(&aggregate, baseline, threshold)
        } else {
            Vec::new()
        };

        // Phase 5: Teardown
        self.teardown().await;

        Ok(BenchmarkReport {
            runs: run_results,
            aggregate,
            regressions,
        })
    }

    /// Kill the server process if we started one.
    async fn teardown(&mut self) {
        if let Some(ref mut child) = self.child {
            // Send SIGKILL and wait for exit
            let _ = child.kill().await;
            let _ = child.wait().await;
            eprintln!("Server process terminated");
        }
        self.child = None;
    }
}

impl Drop for Benchmark {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.start_kill();
        }
    }
}

/// Compute aggregate statistics across multiple benchmark runs.
pub fn compute_aggregate(runs: &[LoadTestResult]) -> AggregateStats {
    let throughput_values: Vec<f64> = runs.iter().map(|r| r.throughput_rps).collect();
    let latency_values: Vec<f64> = runs.iter().map(|r| r.latency_p50_ms).collect();
    let tps_values: Vec<f64> = runs.iter().map(|r| r.tokens_per_sec).collect();
    let ttft_values: Vec<f64> = runs.iter().map(|r| r.ttft_p50_ms).collect();
    let tpot_values: Vec<f64> = runs.iter().map(|r| r.tpot_p50_ms).collect();

    AggregateStats {
        throughput_rps: stat_summary(&throughput_values),
        latency_p50: stat_summary(&latency_values),
        tokens_per_sec: stat_summary(&tps_values),
        ttft_p50: stat_summary(&ttft_values),
        tpot_p50: stat_summary(&tpot_values),
    }
}

/// Compute summary statistics with 95% confidence interval.
fn stat_summary(values: &[f64]) -> StatSummary {
    let n = values.len();
    if n == 0 {
        return StatSummary {
            mean: 0.0,
            stddev: 0.0,
            ci_95_lower: 0.0,
            ci_95_upper: 0.0,
            values: Vec::new(),
        };
    }

    let mean = values.iter().sum::<f64>() / n as f64;

    let stddev = if n > 1 {
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        variance.sqrt()
    } else {
        0.0
    };

    // 95% CI using t-distribution critical value
    // For small N, use t-value lookup; for N > 30, approximate with 1.96
    let t_value = t_critical_95(n);
    let margin = t_value * stddev / (n as f64).sqrt();

    StatSummary {
        mean,
        stddev,
        ci_95_lower: mean - margin,
        ci_95_upper: mean + margin,
        values: values.to_vec(),
    }
}

/// T-distribution critical value for 95% CI (two-tailed).
/// Lookup table for common degrees of freedom (df = n - 1).
fn t_critical_95(n: usize) -> f64 {
    match n {
        0 | 1 => 0.0, // Undefined, no CI possible
        2 => 12.706,
        3 => 4.303,
        4 => 3.182,
        5 => 2.776,
        6 => 2.571,
        7 => 2.447,
        8 => 2.365,
        9 => 2.306,
        10 => 2.262,
        11..=15 => 2.145,
        16..=20 => 2.086,
        21..=30 => 2.042,
        _ => 1.96, // Normal approximation for N > 30
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sample_run(throughput: f64, latency: f64, tps: f64) -> LoadTestResult {
        LoadTestResult {
            total_requests: 100,
            successful: 100,
            failed: 0,
            throughput_rps: throughput,
            latency_p50_ms: latency,
            latency_p95_ms: latency * 2.0,
            latency_p99_ms: latency * 3.0,
            ttft_p50_ms: 50.0,
            tokens_per_sec: tps,
            avg_tok_per_req: 20.0,
            itl_p50_ms: 5.0,
            decode_tok_per_sec: 200.0,
            prefill_tok_per_sec: 0.0,
            timestamp: "2026-03-03T00:00:00Z".to_string(),
            runtime_name: "test".to_string(),
            elapsed_secs: 60.0,
            concurrency: 1,
            ttft_p90_ms: 60.0,
            ttft_p95_ms: 70.0,
            ttft_p99_ms: 80.0,
            tpot_p50_ms: 5.0,
            tpot_p90_ms: 7.0,
            tpot_p95_ms: 8.0,
            tpot_p99_ms: 10.0,
            latency_min_ms: latency * 0.5,
            latency_max_ms: latency * 4.0,
            latency_stddev_ms: latency * 0.3,
            error_rate: 0.0,
            prompt_tokens_total: 1000,
            completion_tokens_total: 2000,
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
        }
    }

    #[test]
    fn test_stat_summary_empty() {
        let s = stat_summary(&[]);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.stddev, 0.0);
    }

    #[test]
    fn test_stat_summary_single() {
        let s = stat_summary(&[42.0]);
        assert!((s.mean - 42.0).abs() < f64::EPSILON);
        assert_eq!(s.stddev, 0.0);
        assert!((s.ci_95_lower - 42.0).abs() < f64::EPSILON);
        assert!((s.ci_95_upper - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stat_summary_multiple() {
        let s = stat_summary(&[10.0, 20.0, 30.0]);
        assert!((s.mean - 20.0).abs() < f64::EPSILON);
        assert!(s.stddev > 0.0);
        assert!(s.ci_95_lower < s.mean);
        assert!(s.ci_95_upper > s.mean);
        assert_eq!(s.values.len(), 3);
    }

    #[test]
    fn test_stat_summary_ci_narrows_with_more_runs() {
        let few = stat_summary(&[10.0, 12.0, 11.0]);
        let many = stat_summary(&[10.0, 12.0, 11.0, 10.5, 11.5, 10.8, 11.2, 10.3, 11.7, 10.9]);
        let few_width = few.ci_95_upper - few.ci_95_lower;
        let many_width = many.ci_95_upper - many.ci_95_lower;
        assert!(many_width < few_width, "CI should narrow with more samples");
    }

    #[test]
    fn test_compute_aggregate() {
        let runs = vec![
            sample_run(10.0, 100.0, 200.0),
            sample_run(11.0, 95.0, 210.0),
            sample_run(10.5, 98.0, 205.0),
        ];
        let agg = compute_aggregate(&runs);
        assert!((agg.throughput_rps.mean - 10.5).abs() < 0.01);
        assert!(agg.throughput_rps.stddev > 0.0);
        assert!(agg.latency_p50.mean > 0.0);
        assert!(agg.tokens_per_sec.mean > 0.0);
        assert!(agg.ttft_p50.mean > 0.0);
        assert!(agg.tpot_p50.mean > 0.0);
    }

    #[test]
    fn test_t_critical_values() {
        assert_eq!(t_critical_95(0), 0.0);
        assert_eq!(t_critical_95(1), 0.0);
        assert!((t_critical_95(2) - 12.706).abs() < 0.001);
        assert!((t_critical_95(3) - 4.303).abs() < 0.001);
        assert!((t_critical_95(100) - 1.96).abs() < 0.001);
    }

    #[test]
    fn test_benchmark_report_serialization() {
        let runs = vec![sample_run(10.0, 100.0, 200.0)];
        let agg = compute_aggregate(&runs);
        let report = BenchmarkReport {
            runs,
            aggregate: agg,
            regressions: vec![Regression {
                metric: "throughput_rps".to_string(),
                baseline_value: 12.0,
                current_value: 10.0,
                change_pct: -16.7,
                exceeds_threshold: true,
            }],
        };
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("throughput_rps"));
        assert!(json.contains("ci_95_lower"));
        let back: BenchmarkReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.runs.len(), 1);
        assert_eq!(back.regressions.len(), 1);
        assert!(back.regressions[0].exceeds_threshold);
    }
}
