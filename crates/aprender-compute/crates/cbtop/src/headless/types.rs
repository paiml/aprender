//! Types, structs, and data models for headless benchmark mode.

use crate::brick::BrickScore;
use serde::{Deserialize, Serialize};

/// Output format for benchmark results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Text,
}

/// CPU frequency governor status (PERF-003)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuGovernorInfo {
    pub governor: String,
    pub is_performance: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_freq_mhz: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_freq_mhz: Option<u32>,
}

/// System information for benchmark context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu: String,
    pub cores: usize,
    pub memory_gb: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// PERF-003: CPU governor status for deterministic benchmarks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_governor: Option<CpuGovernorInfo>,
}

impl SystemInfo {
    pub fn detect() -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

        // Try to get CPU info from /proc/cpuinfo on Linux
        let cpu = Self::detect_cpu();

        // Get memory info
        let memory_gb = Self::detect_memory_gb();

        // PERF-003: Detect CPU governor for deterministic benchmarks
        let cpu_governor = Self::detect_cpu_governor();

        Self {
            cpu,
            cores,
            memory_gb,
            gpu: None, // GPU detection requires CUDA/wgpu initialization
            cpu_governor,
        }
    }

    fn detect_cpu() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        if let Some(name) = line.split(':').nth(1) {
                            return name.trim().to_string();
                        }
                    }
                }
            }
        }
        "Unknown CPU".to_string()
    }

    fn detect_memory_gb() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb / 1024 / 1024; // Convert KB to GB
                            }
                        }
                    }
                }
            }
        }
        0
    }

    /// PERF-003: Detect CPU frequency governor for deterministic benchmarks
    /// Warns if governor is not set to "performance" mode
    fn detect_cpu_governor() -> Option<CpuGovernorInfo> {
        #[cfg(target_os = "linux")]
        {
            // Read governor from first CPU core (cpu0)
            let governor_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor";
            let cur_freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
            let max_freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq";

            if let Ok(governor) = std::fs::read_to_string(governor_path) {
                let governor = governor.trim().to_string();
                let is_performance = governor == "performance";

                let current_freq_mhz = std::fs::read_to_string(cur_freq_path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .map(|khz| khz / 1000);

                let max_freq_mhz = std::fs::read_to_string(max_freq_path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .map(|khz| khz / 1000);

                return Some(CpuGovernorInfo {
                    governor,
                    is_performance,
                    current_freq_mhz,
                    max_freq_mhz,
                });
            }
        }

        None
    }

    /// PERF-003: Check if CPU is in optimal state for benchmarking
    pub fn check_benchmark_readiness(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if let Some(ref gov) = self.cpu_governor {
            if !gov.is_performance {
                warnings.push(format!(
                    "CPU governor is '{}' (not 'performance'). For deterministic benchmarks, run: \
                     sudo cpupower frequency-set -g performance",
                    gov.governor
                ));
            }

            if let (Some(cur), Some(max)) = (gov.current_freq_mhz, gov.max_freq_mhz) {
                let ratio = cur as f64 / max as f64;
                if ratio < 0.9 {
                    warnings.push(format!(
                        "CPU running at {}MHz ({:.0}% of max {}MHz). Thermal throttling may affect results.",
                        cur, ratio * 100.0, max
                    ));
                }
            }
        }

        warnings
    }
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub backend: String,
    pub workload: String,
    pub size: usize,
    pub iterations: u64,
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub cv_percent: f64,
}

/// Score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreInfo {
    pub total: u8,
    pub grade: String,
    pub performance: u8,
    pub efficiency: u8,
    pub correctness: u8,
    pub stability: u8,
}

impl From<BrickScore> for ScoreInfo {
    fn from(score: BrickScore) -> Self {
        Self {
            total: score.total(),
            grade: format!("{:?}", score.grade()),
            performance: score.performance,
            efficiency: score.efficiency,
            correctness: score.correctness,
            stability: score.stability,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub version: String,
    pub timestamp: String,
    pub duration_secs: f64,
    pub system: SystemInfo,
    pub benchmark: BenchmarkConfig,
    pub results: BenchmarkResults,
    pub score: ScoreInfo,
    /// PERF-003: Warnings about benchmark environment
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Core benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub gflops: f64,
    pub throughput_ops_sec: f64,
    pub latency_ms: LatencyStats,
}

impl BenchmarkResult {
    /// Format result for output
    pub fn format(&self, format: OutputFormat) -> String {
        match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
            }
            OutputFormat::Text => self.format_text(),
        }
    }

    fn format_text(&self) -> String {
        format!(
            r#"
=== cbtop Benchmark Results ===

System:
  CPU: {}
  Cores: {}
  Memory: {} GB

Benchmark:
  Backend: {}
  Workload: {}
  Size: {} elements
  Iterations: {}
  Duration: {:.2}s

Results:
  GFLOP/s: {:.2}
  Throughput: {:.0} ops/sec
  Latency (ms):
    Mean: {:.3}
    P50:  {:.3}
    P95:  {:.3}
    P99:  {:.3}
    CV:   {:.1}%

Score: {}/100 (Grade: {})
  Performance:  {}/40
  Efficiency:   {}/25
  Correctness:  {}/20
  Stability:    {}/15
{}
"#,
            self.system.cpu,
            self.system.cores,
            self.system.memory_gb,
            self.benchmark.backend,
            self.benchmark.workload,
            self.benchmark.size,
            self.benchmark.iterations,
            self.duration_secs,
            self.results.gflops,
            self.results.throughput_ops_sec,
            self.results.latency_ms.mean,
            self.results.latency_ms.p50,
            self.results.latency_ms.p95,
            self.results.latency_ms.p99,
            self.results.latency_ms.cv_percent,
            self.score.total,
            self.score.grade,
            self.score.performance,
            self.score.efficiency,
            self.score.correctness,
            self.score.stability,
            // PERF-003: Show warnings if any
            if self.warnings.is_empty() {
                String::new()
            } else {
                format!(
                    "\nWarnings:\n{}",
                    self.warnings
                        .iter()
                        .map(|w| format!("  - {}", w))
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            },
        )
    }

    /// Check for regression against baseline
    pub fn check_regression(&self, baseline: &BenchmarkResult, threshold: f64) -> RegressionResult {
        let change_percent =
            (self.results.gflops - baseline.results.gflops) / baseline.results.gflops * 100.0;

        RegressionResult {
            baseline_gflops: baseline.results.gflops,
            current_gflops: self.results.gflops,
            change_percent,
            threshold_percent: threshold,
            is_regression: change_percent < -threshold,
            status: if change_percent < -threshold {
                "REGRESSION".to_string()
            } else if change_percent > threshold {
                "IMPROVEMENT".to_string()
            } else {
                "STABLE".to_string()
            },
        }
    }

    /// Compare multiple benchmark results
    pub fn compare(results: &[(String, BenchmarkResult)]) -> ComparisonResult {
        let comparisons: Vec<_> = results
            .iter()
            .map(|(name, r)| BackendComparison {
                backend: name.clone(),
                gflops: r.results.gflops,
                score: r.score.total,
                latency_mean_ms: r.results.latency_ms.mean,
            })
            .collect();

        let best = comparisons
            .iter()
            .max_by(|a, b| a.gflops.partial_cmp(&b.gflops).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.backend.clone())
            .unwrap_or_default();

        ComparisonResult { backends: comparisons, recommended: best }
    }
}

/// Regression check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    pub baseline_gflops: f64,
    pub current_gflops: f64,
    pub change_percent: f64,
    pub threshold_percent: f64,
    pub is_regression: bool,
    pub status: String,
}

impl RegressionResult {
    pub fn format(&self, format: OutputFormat) -> String {
        match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
            }
            OutputFormat::Text => {
                format!(
                    r#"
=== Regression Check ===

Baseline: {:.2} GFLOP/s
Current:  {:.2} GFLOP/s
Change:   {:+.1}%
Threshold: {:.1}%

Status: {}
"#,
                    self.baseline_gflops,
                    self.current_gflops,
                    self.change_percent,
                    self.threshold_percent,
                    self.status,
                )
            }
        }
    }
}

/// Backend comparison for --compare mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendComparison {
    pub backend: String,
    pub gflops: f64,
    pub score: u8,
    pub latency_mean_ms: f64,
}

/// Comparison result for multiple backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub backends: Vec<BackendComparison>,
    pub recommended: String,
}

impl ComparisonResult {
    pub fn format(&self, format: OutputFormat) -> String {
        match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
            }
            OutputFormat::Text => {
                let mut s = String::from("\n=== Backend Comparison ===\n\n");
                s.push_str("Backend      GFLOP/s   Score   Latency\n");
                s.push_str("----------------------------------------\n");
                for c in &self.backends {
                    s.push_str(&format!(
                        "{:<12} {:>7.2}   {:>3}     {:.3}ms\n",
                        c.backend, c.gflops, c.score, c.latency_mean_ms
                    ));
                }
                s.push_str(&format!("\nRecommended: {}\n", self.recommended));
                s
            }
        }
    }
}
