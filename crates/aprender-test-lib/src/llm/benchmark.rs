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
    /// Server spawn → first healthy answer, ms (V2, #4953). Present only when
    /// this benchmark started the server (`start_command`); a server it did
    /// not start has no observed load.
    #[serde(default)]
    pub load_ms: Option<f64>,
    /// Whether the server this benchmark started was still alive after the
    /// last measured run. `None` when it started none. A restart between
    /// requests would hide a per-request load.
    #[serde(default)]
    pub server_alive_after: Option<bool>,
    /// The server's own `/health` `uptime_sec`, read after the last run. `None`
    /// when this run started no server, or the server does not report it.
    #[serde(default)]
    pub server_uptime_after_s: Option<f64>,
    /// Seconds from ready to that read, on this producer's clock. A server
    /// that never restarted has `server_uptime_after_s >= ready_to_end_s`.
    #[serde(default)]
    pub ready_to_end_s: Option<f64>,
    /// The server's own `/health` `compute_mode` (`gpu`/`cpu`), read after the
    /// last run: what the server reports, not what the build or the command
    /// line asked for. The Qwen3.5 hybrid reports its live backend, so a GPU
    /// path it refused at the first request reads `cpu` here; other model
    /// paths report the device of the resident model.
    #[serde(default)]
    pub server_compute_mode_after: Option<String>,
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
    ///
    /// With `start_command`, the server runs in its own process group, so the
    /// terminal's Ctrl-C reaches only this process, and a default SIGINT or
    /// SIGTERM would end it without `Drop`, leaving the server holding its
    /// port and its GPU. So the run listens for both: on either, it kills the
    /// server's group and returns [`LlmClientError::Interrupted`] (#4953).
    /// tokio keeps a handler installed for the life of the process, so after
    /// this returns, a SIGINT or SIGTERM no longer ends the process by default.
    pub async fn run(&mut self) -> Result<BenchmarkReport, LlmClientError> {
        if self.config.start_command.is_none() {
            return self.measure().await;
        }
        let stop = stop_signals();
        self.run_until(stop).await
    }

    /// [`Self::measure`], unless `stop` resolves first: then tear the started
    /// server down and name what stopped the run.
    async fn run_until(
        &mut self,
        stop: impl std::future::Future<Output = &'static str>,
    ) -> Result<BenchmarkReport, LlmClientError> {
        let stopped_by = tokio::select! {
            r = self.measure() => return r,
            why = stop => why,
        };
        self.teardown().await;
        Err(LlmClientError::Interrupted(stopped_by))
    }

    async fn measure(&mut self) -> Result<BenchmarkReport, LlmClientError> {
        let client = LlmClient::new(&self.config.url, &self.config.model);

        // Phase 1: Start server (if configured). A server that ALREADY answers
        // before we spawn ours would make the readiness poll return at once and
        // load_ms measure nothing (#4953), so that is an error, not a fast load.
        // The probe gets a 5 s timeout: a refused connect returns at once, but a
        // blackholed --url would otherwise hold the run for 3 × 120 s.
        let probe = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map(|c| LlmClient::with_client(&self.config.url, &self.config.model, c))
            .unwrap_or_else(|_| client.clone());
        if self.config.start_command.is_some() && matches!(probe.health_check().await, Ok(true)) {
            return Err(LlmClientError::HealthCheckFailed(format!(
                "a server already answers at {} before --start ran; its load is not this run's",
                self.config.url
            )));
        }
        let spawned_at = std::time::Instant::now();
        if let Some(ref cmd) = self.config.start_command {
            let mut command = tokio::process::Command::new("sh");
            command
                .arg("-c")
                .arg(cmd)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            // Its own process group, so teardown reaches the server and not only
            // the `sh` that started it (#4953). The terminal's Ctrl-C then
            // reaches only this process; `run` handles it.
            #[cfg(unix)]
            command.process_group(0);
            let child = command.spawn().map_err(|e| {
                LlmClientError::HealthCheckFailed(format!("Failed to start server: {e}"))
            })?;
            self.child = Some(child);
        }

        // Phase 1b: Wait for server readiness
        // A 2 s poll is fine for "is it up", not for a load time: it rounds a
        // load up to the next poll. When this run owns the server, poll at
        // 50 ms and record spawn → ready as load_ms (V2, #4953).
        let poll = if self.child.is_some() {
            Duration::from_millis(50)
        } else {
            Duration::from_secs(2)
        };
        let ready_time = client.wait_ready(self.config.health_timeout, poll).await?;
        let load_ms = self
            .child
            .is_some()
            .then(|| spawned_at.elapsed().as_secs_f64() * 1000.0);
        eprintln!("Server ready in {:.1}s", ready_time.as_secs_f64());
        let ready_at = std::time::Instant::now();

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

        // Process continuity (#4953): `/health` `uptime_sec` counts from the
        // moment the server finished loading. Read now, it must cover the whole
        // span since ready; a restart in between (a supervisor, a wrapper)
        // resets it, which a liveness check on our own child cannot see.
        // The same read names the device that served the runs.
        let (server_uptime_after_s, ready_to_end_s, server_compute_mode_after) =
            if self.child.is_some() {
                let span = ready_at.elapsed().as_secs_f64();
                let health = client.get_json("/health").await.ok().flatten();
                let field = |k: &str| health.as_ref().and_then(|v| v.get(k)).cloned();
                (
                    field("uptime_sec").and_then(|v| v.as_f64()),
                    Some(span),
                    field("compute_mode").and_then(|v| v.as_str().map(str::to_string)),
                )
            } else {
                (None, None, None)
            };

        // Phase 5: Teardown. Read liveness FIRST: `try_wait` is Ok(None) while
        // the process we spawned is still running.
        let server_alive_after = self
            .child
            .as_mut()
            .map(|c| matches!(c.try_wait(), Ok(None)));
        self.teardown().await;

        Ok(BenchmarkReport {
            runs: run_results,
            aggregate,
            regressions,
            load_ms,
            server_alive_after,
            server_uptime_after_s,
            ready_to_end_s,
            server_compute_mode_after,
        })
    }

    /// Kill the server process if we started one.
    async fn teardown(&mut self) {
        if let Some(ref mut child) = self.child {
            kill_group(child);
            let _ = child.kill().await;
            let _ = child.wait().await;
            eprintln!("Server process terminated");
        }
        self.child = None;
    }
}

/// SIGKILL the started command's whole process group. Killing `sh` alone
/// orphans the server it started, which keeps the port and the GPU (#4953).
/// A server that calls `setsid` or daemonizes leaves the group and escapes
/// this; `apr serve` does neither.
fn kill_group(child: &tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    #[cfg(not(unix))]
    let _ = child;
}

/// Resolves with the name of the first SIGINT or SIGTERM. Both handlers are
/// installed by this call, before the server is spawned, so no signal falls
/// in a gap. A handler that cannot be installed never resolves. SIGHUP is not
/// handled.
#[cfg(unix)]
fn stop_signals() -> impl std::future::Future<Output = &'static str> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut int = signal(SignalKind::interrupt()).ok();
    let mut term = signal(SignalKind::terminate()).ok();
    async move {
        tokio::select! {
            Some(()) = next_signal(&mut int) => "SIGINT",
            Some(()) = next_signal(&mut term) => "SIGTERM",
            else => std::future::pending().await,
        }
    }
}

#[cfg(unix)]
async fn next_signal(s: &mut Option<tokio::signal::unix::Signal>) -> Option<()> {
    match s {
        Some(s) => s.recv().await,
        None => std::future::pending().await,
    }
}

/// Resolves on the first Ctrl-C; never, if its handler cannot be installed.
#[cfg(not(unix))]
fn stop_signals() -> impl std::future::Future<Output = &'static str> {
    async {
        if tokio::signal::ctrl_c().await.is_ok() {
            "Ctrl-C"
        } else {
            std::future::pending().await
        }
    }
}

impl Drop for Benchmark {
    fn drop(&mut self) {
        // Also on an early return (a readiness timeout) or a panic, which skip
        // teardown: the group, not only `sh`.
        if let Some(ref mut child) = self.child {
            kill_group(child);
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
            load_ms: Some(4000.0),
            server_alive_after: Some(true),
            server_uptime_after_s: Some(60.0),
            ready_to_end_s: Some(30.0),
            server_compute_mode_after: Some("gpu".to_string()),
        };
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("throughput_rps"));
        assert!(json.contains("ci_95_lower"));
        let back: BenchmarkReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.runs.len(), 1);
        assert_eq!(back.regressions.len(), 1);
        assert!(back.regressions[0].exceeds_threshold);
        assert_eq!(back.load_ms, Some(4000.0));
        assert_eq!(back.server_alive_after, Some(true));
        assert_eq!(back.server_uptime_after_s, Some(60.0));
        assert_eq!(back.ready_to_end_s, Some(30.0));
        assert_eq!(back.server_compute_mode_after.as_deref(), Some("gpu"));
    }

    /// A report written before V2 (#4953) carries neither field and must still
    /// load as a baseline, with both absent rather than invented.
    #[test]
    fn a_pre_v2_report_loads_with_no_load_ms() {
        let runs = vec![sample_run(10.0, 100.0, 200.0)];
        let mut v = serde_json::to_value(BenchmarkReport {
            aggregate: compute_aggregate(&runs),
            runs,
            regressions: Vec::new(),
            load_ms: Some(1.0),
            server_alive_after: Some(true),
            server_uptime_after_s: Some(1.0),
            ready_to_end_s: Some(1.0),
            server_compute_mode_after: Some("gpu".to_string()),
        })
        .unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("load_ms");
        o.remove("server_alive_after");
        o.remove("server_uptime_after_s");
        o.remove("ready_to_end_s");
        o.remove("server_compute_mode_after");
        let back: BenchmarkReport = serde_json::from_value(v).unwrap();
        assert_eq!(back.load_ms, None);
        assert_eq!(back.server_alive_after, None);
        assert_eq!(back.server_uptime_after_s, None);
        assert_eq!(back.ready_to_end_s, None);
        assert_eq!(back.server_compute_mode_after, None);
    }

    /// The two tests below run `Benchmark::run` with `start_command`, which
    /// installs this process's SIGINT and SIGTERM handlers, and one of them
    /// sends those signals to this process. One at a time, so no run sees a
    /// signal meant for another.
    static SIGNAL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn owned_server_config(url: String, start: &str, health_timeout: Duration) -> BenchmarkConfig {
        BenchmarkConfig {
            url,
            model: "m".to_string(),
            start_command: Some(start.to_string()),
            health_timeout,
            warmup: Duration::ZERO,
            duration: Duration::from_secs(1),
            concurrency: 1,
            runs: 0,
            cooldown: Duration::ZERO,
            prompts: Vec::new(),
            runtime_name: "test".to_string(),
            baseline: None,
            fail_on_regression: None,
            stream: false,
            trace_level: None,
            num_layers: None,
        }
    }

    fn current_thread_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    /// A listener that answers every request with 200 and `{}`: a server that
    /// is already up.
    fn stub_health_server() -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for mut conn in listener.incoming().flatten() {
                let mut req = Vec::new();
                let mut buf = [0u8; 1024];
                while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                    match conn.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => req.extend_from_slice(&buf[..n]),
                    }
                }
                let _ = conn.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                      Content-Length: 2\r\nConnection: close\r\n\r\n{}",
                );
            }
        });
        url
    }

    /// A server already answering before `--start` would make the readiness
    /// poll return at once and load_ms measure nothing, so the run refuses
    /// before it spawns anything (#4953, FALSIFY-SERVE-009).
    #[test]
    fn a_server_already_answering_is_refused_before_start() {
        let _serial = SIGNAL_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let url = stub_health_server();
        let mut bench = Benchmark::new(owned_server_config(url, "true", Duration::from_secs(5)));
        let result = current_thread_runtime().block_on(bench.run());
        let Err(e) = result else {
            panic!("a run against a server that already answers must be refused");
        };
        assert!(e.to_string().contains("already answers"), "{e}");
        assert!(bench.child.is_none(), "the start command ran anyway");
    }

    /// True once `pid` has exited (gone, or a zombie not yet reaped).
    #[cfg(target_os = "linux")]
    fn exited(pid: u32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Err(_) => true,
            Ok(stat) => stat
                .rsplit_once(") ")
                .is_some_and(|(_, rest)| rest.starts_with('Z')),
        }
    }

    #[cfg(target_os = "linux")]
    async fn read_pid(path: &std::path::Path) -> u32 {
        for _ in 0..500 {
            let pid = std::fs::read_to_string(path)
                .ok()
                .and_then(|s| s.trim().parse().ok());
            if let Some(pid) = pid {
                return pid;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the start command never wrote {}", path.display());
    }

    /// The started server is in its own process group, so the terminal's
    /// Ctrl-C reaches only this process. A SIGINT or SIGTERM mid-run must still
    /// kill the server, here a grandchild of `sh` as `apr serve` is under
    /// `--start`, and end the run with Interrupted (#4953, FALSIFY-SERVE-010).
    #[cfg(target_os = "linux")]
    #[test]
    fn a_signal_mid_run_kills_the_started_server() {
        let _serial = SIGNAL_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rt = current_thread_runtime();
        for sig in ["SIGINT", "SIGTERM"] {
            let dir = tempfile::tempdir().unwrap();
            let pidfile = dir.path().join("server.pid");
            // A port nothing answers on: the run waits for readiness until the signal.
            let url = format!(
                "http://{}",
                std::net::TcpListener::bind("127.0.0.1:0")
                    .unwrap()
                    .local_addr()
                    .unwrap()
            );
            let start = format!("sleep 60 & echo $! > '{}'; wait", pidfile.display());
            let mut bench =
                Benchmark::new(owned_server_config(url, &start, Duration::from_secs(20)));
            let signal_this_process = async {
                let server = read_pid(&pidfile).await;
                assert!(
                    !exited(server),
                    "{sig}: server {server} not running before the signal"
                );
                let sent = tokio::process::Command::new("kill")
                    .arg(format!("-{}", &sig[3..]))
                    .arg(std::process::id().to_string())
                    .status()
                    .await
                    .unwrap();
                assert!(sent.success(), "{sig}: kill failed");
                server
            };
            let (result, server) =
                rt.block_on(async { tokio::join!(bench.run(), signal_this_process) });
            let Err(e) = result else {
                panic!("{sig}: an interrupted run returned a report");
            };
            assert!(
                e.to_string().contains(&format!("interrupted by {sig}")),
                "{e}"
            );
            assert!(bench.child.is_none(), "{sig}: the server was not torn down");
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while !exited(server) && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(exited(server), "{sig}: server {server} outlived the run");
        }
    }
}
