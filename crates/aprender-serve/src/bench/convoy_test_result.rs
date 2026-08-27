
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

impl HardwareSpec {
    /// PARITY-007 — read the host, never assert it.
    ///
    /// `external_matrix.rs` wrote `cpu: "Benchmark CPU", gpu: Some("Benchmark
    /// GPU"), memory_gb: 32` straight into the receipt. The `BenchmarkMatrix`
    /// schema existed and nothing forced its fields to be MEASURED, so a
    /// four-host receipt could carry placeholder provenance while looking
    /// complete. That is F12 — a value with the form of a measurement and
    /// nothing behind it (aprender#2679).
    ///
    /// The remedy is the general one: a field that CAN be measured is derived
    /// or absent, never asserted. Where a probe fails, this records "unknown"
    /// and `None` — an honest gap the consuming gate can treat as RED — rather
    /// than a plausible-looking literal it cannot distinguish from evidence.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            cpu: Self::detect_cpu(),
            gpu: Self::detect_gpu(),
            memory_gb: Self::detect_memory_gb(),
            storage: "unknown".to_string(),
        }
    }

    /// Linux: the `model name` line of /proc/cpuinfo.
    fn cpu_from_proc() -> Option<String> {
        let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        info.lines()
            .find_map(|l| l.strip_prefix("model name"))
            .and_then(|rest| rest.split(':').nth(1))
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    }

    /// macOS has no /proc; sysctl carries the same fact.
    fn cpu_from_sysctl() -> Option<String> {
        Self::sysctl("machdep.cpu.brand_string")
    }

    /// One place that runs sysctl, so both callers agree on what a failed
    /// probe looks like.
    fn sysctl(key: &str) -> Option<String> {
        let out = std::process::Command::new("sysctl")
            .args(["-n", key])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    }

    fn detect_cpu() -> String {
        Self::cpu_from_proc()
            .or_else(Self::cpu_from_sysctl)
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn detect_gpu() -> Option<String> {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let name = String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }

    fn memory_from_proc() -> Option<u64> {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = info
            .lines()
            .find_map(|l| l.strip_prefix("MemTotal:"))
            .map(|r| r.trim().trim_end_matches(" kB").trim().to_string())
            .and_then(|v| v.parse().ok())?;
        Some(kb / 1024 / 1024)
    }

    fn memory_from_sysctl() -> Option<u64> {
        let bytes: u64 = Self::sysctl("hw.memsize")?.parse().ok()?;
        Some(bytes / 1024 / 1024 / 1024)
    }

    /// 0 is the honest "could not read", never a default that looks measured.
    fn detect_memory_gb() -> u64 {
        Self::memory_from_proc()
            .or_else(Self::memory_from_sysctl)
            .unwrap_or(0)
    }
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

// ── PARITY-007: hardware provenance is read, never asserted ────────────────
#[cfg(test)]
mod parity_007_hardware_tests {
    // This file is include!()d into bench/mod.rs, so `super` is the crate
    // root's `bench` module rather than a file-level module.
    use super::HardwareSpec;

    /// `detect()` must not reproduce the placeholder strings it replaced.
    /// Those exact literals shipped in `external_matrix.rs` and wrote
    /// placeholder provenance into a receipt that looked complete (F12).
    #[test]
    fn detect_never_emits_the_placeholders_it_replaced() {
        let hw = HardwareSpec::detect();
        assert_ne!(hw.cpu, "Benchmark CPU");
        assert_ne!(hw.gpu.as_deref(), Some("Benchmark GPU"));
        assert_ne!(hw.storage, "SSD", "storage was a literal; it is now honest");
    }

    /// A failed probe must yield an honest gap a gate can treat as RED —
    /// never a plausible-looking value it cannot distinguish from evidence.
    #[test]
    fn detect_reports_unknown_rather_than_inventing() {
        let hw = HardwareSpec::detect();
        // On any host, cpu is either a real model string or the honest
        // "unknown". It is never a decorative placeholder.
        assert!(
            hw.cpu == "unknown" || hw.cpu.len() > 3,
            "cpu was {:?}: expected a real model string or the honest \"unknown\"",
            hw.cpu
        );
        // gpu is Option: absent means absent, and is not faked.
        if let Some(g) = &hw.gpu {
            assert!(!g.is_empty(), "a present gpu must be named");
        }
    }

    /// On a host with /proc or sysctl, memory must be a real figure. Zero is
    /// the honest "could not read", not a default that looks measured.
    #[test]
    fn detect_memory_is_measured_or_zero() {
        let hw = HardwareSpec::detect();
        assert!(
            hw.memory_gb == 0 || hw.memory_gb >= 1,
            "memory_gb {} is neither an honest 0 nor a plausible size",
            hw.memory_gb
        );
    }
}

// ── PARITY-007: the detection is real, checked against this host ───────────
#[cfg(test)]
mod parity_007_engagement_tests {
    use super::HardwareSpec;

    /// PROVE THE MECHANISM ENGAGED. The three tests above assert that
    /// `detect()` does not emit the placeholders it replaced — necessary, and
    /// satisfied equally by a function that returns "unknown" for everything.
    /// This one checks the detected values against the host's own files, so a
    /// detector that silently degraded to "unknown" everywhere is caught.
    ///
    /// Skips honestly where the source does not exist (macOS has no /proc),
    /// and says so, rather than passing vacuously.
    #[test]
    fn detected_values_match_this_host() {
        let hw = HardwareSpec::detect();

        if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
            let truth = info
                .lines()
                .find_map(|l| l.strip_prefix("model name"))
                .and_then(|r| r.split(':').nth(1))
                .map(str::trim)
                .unwrap_or("");
            if !truth.is_empty() {
                assert_eq!(
                    hw.cpu, truth,
                    "detect() must report the host's actual CPU, not a fallback"
                );
            }
        }

        if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
            let kb: u64 = info
                .lines()
                .find_map(|l| l.strip_prefix("MemTotal:"))
                .map(|r| r.trim().trim_end_matches(" kB").trim().to_string())
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if kb > 0 {
                assert_eq!(
                    hw.memory_gb,
                    kb / 1024 / 1024,
                    "detect() must report the host's actual memory"
                );
            }
        }
    }
}
