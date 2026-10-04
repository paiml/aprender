//! Full benchmark orchestrator for LLM inference endpoints.
//!
//! Orchestrates the complete lifecycle: server start, health poll, warmup,
//! multi-run measurement, statistical analysis, baseline comparison, and teardown.

use super::client::{ChatRequest, LlmClient, LlmClientError, ReadyProbe};
use super::loadtest::{LoadTest, LoadTestConfig, LoadTestResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

/// Configuration for a full benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Base URL of the LLM API server.
    pub url: String,
    /// Model name.
    pub model: String,
    /// Shell command to start the server (optional).
    pub start_command: Option<String>,
    /// File that receives a started server's stdout and stderr. Without it
    /// the output is discarded, and a failed start has no log to cite.
    pub start_log: Option<PathBuf>,
    /// Maximum time to wait for server readiness.
    pub health_timeout: Duration,
    /// Interval between readiness probes. It bounds how precisely a started
    /// server's cold start is known.
    pub health_poll: Duration,
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
        let client = LlmClient::new(&self.config.url, &self.config.model);

        // Phase 1: Start the server (if configured) and wait for readiness
        let readiness = self.bring_up(&client).await?;

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
        stamp_cold_start(&mut run_results, readiness);

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

    /// Start the server when configured, and wait until it is ready.
    ///
    /// Returns when a server this run started became ready. A server that
    /// was already running has no spawn to time from, so it returns `None`.
    async fn bring_up(&mut self, client: &LlmClient) -> Result<Option<Readiness>, LlmClientError> {
        let Some(cmd) = self.config.start_command.as_deref() else {
            let ready_time = client
                .wait_ready(self.config.health_timeout, self.config.health_poll)
                .await?;
            eprintln!("Server ready in {:.1}s", ready_time.as_secs_f64());
            return Ok(None);
        };
        // A server that already answers, ready or still loading, would be
        // timed in place of the one this run starts, whose bind would then fail.
        let occupant = client.probe_ready().await;
        if let Some(status) = occupant.status {
            return Err(LlmClientError::HealthCheckFailed(format!(
                "a server already answers at {} ({status}); --start would time it, not the one it starts",
                occupant.url
            )));
        }
        let log = self.config.start_log.as_deref();
        let (child, spawned) = spawn_server(cmd, log)?;
        let child = self.child.insert(child);
        let readiness = await_started_server(
            client,
            child,
            spawned,
            self.config.health_timeout,
            self.config.health_poll,
            log,
        )
        .await?;
        eprintln!(
            "Server ready {:.0} ms after start (known to within {:.0} ms): {} answered {}",
            readiness.ready_ms,
            readiness.resolution_ms(),
            readiness.probe.url,
            readiness
                .probe
                .status
                .map_or_else(|| "nothing".to_string(), |s| s.to_string())
        );
        Ok(Some(readiness))
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

/// When a server the benchmark started became ready, bracketed from spawn.
///
/// The true ready time lies in `(not_ready_ms, ready_ms]`. `ready_ms` is when
/// the first passing probe returned, so it can only overstate the load time.
/// `not_ready_ms` is when the last failing probe was sent, or 0 when the first
/// probe passed. The gap between them is how well the load time is known. It
/// is at least one poll interval whenever a probe failed.
///
/// `probe` is what the passing probe saw, and `refusal` what the last failing
/// one saw (`None` when the first probe passed). A refusal with a status says
/// the server was listening while it loaded; one without says nothing had
/// bound yet, which is how a server that loads before it binds looks.
#[derive(Debug, Clone)]
struct Readiness {
    ready_ms: f64,
    not_ready_ms: f64,
    probe: ReadyProbe,
    refusal: Option<ReadyProbe>,
}

impl Readiness {
    fn resolution_ms(&self) -> f64 {
        self.ready_ms - self.not_ready_ms
    }
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Start `cmd` under `sh -c`, writing its stdout and stderr to `log` when one
/// is given and discarding them otherwise. Returns the child and the instant
/// just before the spawn, which the cold start is timed from.
fn spawn_server(
    cmd: &str,
    log: Option<&Path>,
) -> Result<(tokio::process::Child, Instant), LlmClientError> {
    let (stdout, stderr) = match log {
        Some(path) => {
            let open_failed = |e: std::io::Error| {
                LlmClientError::HealthCheckFailed(format!(
                    "cannot open start log {}: {e}",
                    path.display()
                ))
            };
            let out = std::fs::File::create(path).map_err(open_failed)?;
            let err = out.try_clone().map_err(open_failed)?;
            (Stdio::from(out), Stdio::from(err))
        }
        None => (Stdio::null(), Stdio::null()),
    };
    let spawned = Instant::now();
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdout(stdout)
        .stderr(stderr)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| LlmClientError::HealthCheckFailed(format!("Failed to start server: {e}")))?;
    Ok((child, spawned))
}

/// Poll a server this benchmark started until it is ready.
///
/// A start command that exits non-zero fails the wait at once with its exit
/// status and the last line it logged, rather than running out the timeout:
/// a failed start carries its own evidence (#3943). An exit status of 0 keeps
/// polling, because the command may have left the server running in the
/// background.
async fn await_started_server(
    client: &LlmClient,
    child: &mut tokio::process::Child,
    spawned: Instant,
    timeout: Duration,
    poll: Duration,
    log: Option<&Path>,
) -> Result<Readiness, LlmClientError> {
    let mut not_ready_ms = 0.0;
    let mut refusal = None;
    loop {
        let sent = spawned.elapsed();
        if sent > timeout {
            return Err(LlmClientError::HealthCheckFailed(format!(
                "server not ready {:.1}s after start{}",
                timeout.as_secs_f64(),
                log_evidence(log)
            )));
        }
        let probe = client.probe_ready().await;
        if probe.is_ready() {
            return Ok(Readiness {
                ready_ms: millis(spawned.elapsed()),
                not_ready_ms,
                probe,
                refusal,
            });
        }
        refusal = Some(probe);
        not_ready_ms = millis(sent);
        let exited = child.try_wait().ok().flatten();
        if let Some(status) = exited.filter(|status| !status.success()) {
            return Err(LlmClientError::HealthCheckFailed(format!(
                "start command ended ({status}) before the server was ready{}",
                log_evidence(log)
            )));
        }
        tokio::time::sleep(poll).await;
    }
}

/// The start log's last non-empty line, as a suffix for an error message.
fn log_evidence(log: Option<&Path>) -> String {
    let Some(path) = log else {
        return "; its output was discarded (no start log)".to_string();
    };
    match std::fs::read(path) {
        Err(e) => format!("; cannot read {}: {e}", path.display()),
        Ok(bytes) => last_nonempty_line(&String::from_utf8_lossy(&bytes)).map_or_else(
            || format!("; {} is empty", path.display()),
            |line| format!("; last line of {}: {line}", path.display()),
        ),
    }
}

fn last_nonempty_line(text: &str) -> Option<&str> {
    text.lines()
        .rev()
        .map(str::trim_end)
        .find(|line| !line.is_empty())
}

/// Record a started server's readiness on every run it served. Runs against
/// a server the benchmark found running keep every cold-start field unset.
fn stamp_cold_start(runs: &mut [LoadTestResult], readiness: Option<Readiness>) {
    let Some(readiness) = readiness else {
        return;
    };
    for run in runs {
        run.cold_start_ms = Some(readiness.ready_ms);
        run.cold_start_resolution_ms = Some(readiness.resolution_ms());
        run.cold_start_probe = Some(readiness.probe.clone());
        run.cold_start_refusal.clone_from(&readiness.refusal);
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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
            cold_start_resolution_ms: None,
            cold_start_probe: None,
            cold_start_refusal: None,
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

    /// A loopback server whose `/health` answers 503 to the first `failures`
    /// probes and 200 after, and 404 on every other path. Returns its base URL
    /// and a count of the `/health` probes it has answered.
    async fn spawn_health_endpoint(failures: usize) -> (String, Arc<AtomicUsize>) {
        spawn_health_endpoint_after(0, failures).await
    }

    /// As [`spawn_health_endpoint`], but the first `absent` `/health` probes
    /// get 404 before the 503s start. A 404 is no verdict, so to a probe the
    /// URL looks like one where nothing serves yet, as before a start.
    async fn spawn_health_endpoint_after(
        absent: usize,
        failures: usize,
    ) -> (String, Arc<AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let probes = Arc::new(AtomicUsize::new(0));
        let answered = Arc::clone(&probes);
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let answered = Arc::clone(&answered);
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut chunk = [0_u8; 1024];
                    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                        match sock.read(&mut chunk).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => head.extend_from_slice(&chunk[..n]),
                        }
                    }
                    let status_line = if !head.starts_with(b"GET /health ") {
                        "404 Not Found"
                    } else {
                        match answered.fetch_add(1, Ordering::SeqCst) {
                            n if n < absent => "404 Not Found",
                            n if n - absent < failures => "503 Service Unavailable",
                            _ => "200 OK",
                        }
                    };
                    let response = format!(
                        "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n\
                         Content-Length: 2\r\nConnection: close\r\n\r\n{{}}"
                    );
                    let _ = sock.write_all(response.as_bytes()).await;
                    let _ = sock.flush().await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        (format!("http://{addr}"), probes)
    }

    fn test_config(url: &str, start_command: Option<&str>) -> BenchmarkConfig {
        BenchmarkConfig {
            url: url.to_string(),
            model: "m".to_string(),
            start_command: start_command.map(str::to_string),
            start_log: None,
            health_timeout: Duration::from_secs(30),
            health_poll: Duration::from_millis(20),
            warmup: Duration::ZERO,
            duration: Duration::ZERO,
            concurrency: 1,
            runs: 1,
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

    #[tokio::test]
    async fn started_server_cold_start_is_stamped_on_every_run() {
        let (base, probes) = spawn_health_endpoint_after(1, 2).await;
        let mut config = test_config(&base, Some("exec sleep 30"));
        config.runs = 2;
        let mut bench = Benchmark::new(config);
        let report = bench.run().await.expect("benchmark");
        // The first probe found nothing answering, so the run started its own.
        // Two failed after the start and the fourth passed.
        assert_eq!(probes.load(Ordering::SeqCst), 4);
        assert!(bench.child.is_none(), "teardown reaps the started server");
        assert_eq!(report.runs.len(), 2);
        let poll_ms = 20.0;
        for run in &report.runs {
            let ready = run
                .cold_start_ms
                .expect("a started server's runs carry its cold start");
            let resolution = run
                .cold_start_resolution_ms
                .expect("and how precisely it is known");
            // Sleeps never end early, so these lower bounds cannot flake.
            assert!(ready >= 2.0 * poll_ms, "ready {ready} ms");
            assert!(resolution >= poll_ms, "resolution {resolution} ms");
            assert!(
                ready - resolution >= poll_ms,
                "last failing probe at {ready} - {resolution} ms"
            );
        }
    }

    #[tokio::test]
    async fn found_server_runs_have_no_cold_start() {
        let (base, probes) = spawn_health_endpoint(0).await;
        let mut bench = Benchmark::new(test_config(&base, None));
        let report = bench.run().await.expect("benchmark");
        assert_eq!(probes.load(Ordering::SeqCst), 1);
        assert_eq!(report.runs.len(), 1);
        assert!(report.runs[0].cold_start_ms.is_none());
        assert!(report.runs[0].cold_start_resolution_ms.is_none());
        assert!(report.runs[0].cold_start_probe.is_none());
        assert!(report.runs[0].cold_start_refusal.is_none());
    }

    #[tokio::test]
    async fn start_refuses_a_url_that_already_answers() {
        let (base, probes) = spawn_health_endpoint(0).await;
        let mut bench = Benchmark::new(test_config(&base, Some("exec sleep 30")));
        let err = bench.run().await.expect_err("occupied URL").to_string();
        assert!(err.contains("already answers"), "{err}");
        assert!(err.contains("(200)"), "{err}");
        assert!(bench.child.is_none(), "nothing was started");
        assert_eq!(probes.load(Ordering::SeqCst), 1);
    }

    /// A server still loading answers 503, and it holds the port all the same.
    #[tokio::test]
    async fn start_refuses_a_url_that_answers_not_ready() {
        let (base, probes) = spawn_health_endpoint(usize::MAX).await;
        let mut bench = Benchmark::new(test_config(&base, Some("exec sleep 30")));
        let err = bench.run().await.expect_err("occupied URL").to_string();
        assert!(err.contains("already answers"), "{err}");
        assert!(err.contains("(503)"), "{err}");
        assert!(bench.child.is_none(), "nothing was started");
        assert_eq!(probes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn readiness_brackets_the_first_passing_probe() {
        let (base, probes) = spawn_health_endpoint(3).await;
        let client = LlmClient::new(&base, "m");
        let (mut child, spawned) = spawn_server("exec sleep 30", None).expect("spawn");
        let poll = Duration::from_millis(20);
        let readiness = await_started_server(
            &client,
            &mut child,
            spawned,
            Duration::from_secs(30),
            poll,
            None,
        )
        .await
        .expect("ready");
        let _ = child.kill().await;
        assert_eq!(probes.load(Ordering::SeqCst), 4);
        assert!(
            readiness.not_ready_ms >= 2.0 * millis(poll),
            "{readiness:?}"
        );
        assert!(readiness.resolution_ms() >= millis(poll), "{readiness:?}");
        let health = format!("{base}/health");
        assert_eq!(
            readiness.probe,
            ReadyProbe {
                url: health.clone(),
                status: Some(200)
            }
        );
        assert_eq!(
            readiness.refusal,
            Some(ReadyProbe {
                url: health,
                status: Some(503)
            }),
            "the last failing probe is kept"
        );
    }

    #[tokio::test]
    async fn a_first_probe_that_passes_leaves_no_refusal() {
        let (base, probes) = spawn_health_endpoint(0).await;
        let client = LlmClient::new(&base, "m");
        let (mut child, spawned) = spawn_server("exec sleep 30", None).expect("spawn");
        let readiness = await_started_server(
            &client,
            &mut child,
            spawned,
            Duration::from_secs(30),
            Duration::from_millis(20),
            None,
        )
        .await
        .expect("ready");
        let _ = child.kill().await;
        assert_eq!(probes.load(Ordering::SeqCst), 1);
        assert!(readiness.refusal.is_none(), "{readiness:?}");
        assert_eq!(readiness.not_ready_ms, 0.0, "{readiness:?}");
        assert_eq!(readiness.resolution_ms(), readiness.ready_ms);
    }

    #[tokio::test]
    async fn readiness_timeout_cites_the_start_log() {
        let (base, _) = spawn_health_endpoint(usize::MAX).await;
        let client = LlmClient::new(&base, "m");
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("server.log");
        let (mut child, spawned) = spawn_server(
            "echo 'load_tensors: 42%'; exec sleep 30",
            Some(log.as_path()),
        )
        .expect("spawn");
        // Wait for the line itself, so the assertion never races the shell.
        let deadline = Instant::now() + Duration::from_secs(30);
        while !std::fs::read_to_string(&log)
            .unwrap_or_default()
            .contains("42%")
        {
            assert!(Instant::now() < deadline, "the start command never logged");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let err = await_started_server(
            &client,
            &mut child,
            spawned,
            Duration::from_millis(200),
            Duration::from_millis(20),
            Some(log.as_path()),
        )
        .await
        .expect_err("never ready")
        .to_string();
        let _ = child.kill().await;
        assert!(err.contains("not ready"), "{err}");
        assert!(err.contains("load_tensors: 42%"), "{err}");
    }

    #[tokio::test]
    async fn failed_start_command_ends_the_wait_with_its_evidence() {
        let (base, _) = spawn_health_endpoint(usize::MAX).await;
        let client = LlmClient::new(&base, "m");
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("server.log");
        let (mut child, spawned) = spawn_server(
            "echo 'error: model file not found' >&2; exit 3",
            Some(log.as_path()),
        )
        .expect("spawn");
        let err = await_started_server(
            &client,
            &mut child,
            spawned,
            Duration::from_secs(30),
            Duration::from_millis(20),
            Some(log.as_path()),
        )
        .await
        .expect_err("start failed")
        .to_string();
        // The timeout message would say "not ready"; this one names the exit.
        assert!(err.contains("exit status: 3"), "{err}");
        assert!(err.contains("error: model file not found"), "{err}");
    }

    #[tokio::test]
    async fn start_command_that_exits_zero_keeps_polling() {
        let (base, probes) = spawn_health_endpoint(3).await;
        let client = LlmClient::new(&base, "m");
        let (mut child, spawned) = spawn_server("exit 0", None).expect("spawn");
        // Exited before the first probe: a server it left in the background
        // could still come up, so the wait goes on.
        assert!(child.wait().await.expect("wait").success());
        await_started_server(
            &client,
            &mut child,
            spawned,
            Duration::from_secs(30),
            Duration::from_millis(20),
            None,
        )
        .await
        .expect("ready after the command exited 0");
        assert_eq!(probes.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn stamp_cold_start_sets_every_field_or_none() {
        let mut runs = vec![sample_run(1.0, 1.0, 1.0), sample_run(2.0, 2.0, 2.0)];
        stamp_cold_start(&mut runs, None);
        for run in &runs {
            assert!(run.cold_start_ms.is_none());
            assert!(run.cold_start_resolution_ms.is_none());
            assert!(run.cold_start_probe.is_none());
            assert!(run.cold_start_refusal.is_none());
            let json = serde_json::to_value(run).expect("serialize");
            assert!(json.get("cold_start_probe").is_none(), "{json}");
            assert!(json.get("cold_start_refusal").is_none(), "{json}");
        }
        let probe = ReadyProbe {
            url: "http://127.0.0.1:8090/health".to_string(),
            status: Some(200),
        };
        let refusal = ReadyProbe {
            url: "http://127.0.0.1:8090".to_string(),
            status: None,
        };
        let readiness = Readiness {
            ready_ms: 1840.0,
            not_ready_ms: 1790.0,
            probe: probe.clone(),
            refusal: Some(refusal.clone()),
        };
        stamp_cold_start(&mut runs, Some(readiness));
        for run in &runs {
            assert_eq!(run.cold_start_ms, Some(1840.0));
            assert_eq!(run.cold_start_resolution_ms, Some(50.0));
            assert_eq!(run.cold_start_probe.as_ref(), Some(&probe));
            assert_eq!(run.cold_start_refusal.as_ref(), Some(&refusal));
            // The wire shape the parity producer reads.
            let json = serde_json::to_value(run).expect("serialize");
            assert_eq!(json["cold_start_probe"]["url"], probe.url.as_str());
            assert_eq!(json["cold_start_probe"]["status"], 200);
            assert!(json["cold_start_refusal"]["status"].is_null(), "{json}");
        }
    }

    #[test]
    fn log_evidence_is_the_last_nonempty_line() {
        assert_eq!(
            last_nonempty_line("a\nload 42%  \n\n  \n"),
            Some("load 42%")
        );
        assert_eq!(last_nonempty_line("\n \n"), None);
        assert!(log_evidence(None).contains("discarded"));
    }
}
