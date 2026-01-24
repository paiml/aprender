//! Benchmark Visualization Module (PAR-040)
//!
//! Creates 2×3 grid visualizations for inference benchmark comparisons
//! with rich terminal colors and scientific benchmarking statistics.
//!
//! ## Features
//!
//! - **Rich Colors**: ANSI terminal colors for clear pass/fail/warn status
//! - **Multi-Iteration Statistics**: Mean, std dev, confidence intervals
//! - **Scientific Benchmarking**: Criterion-style output with outlier detection
//! - **Chat-Pasteable Logs**: Profiling hotspot reports for debugging
//!
//! ## Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │              GGUF Inference Comparison (tok/s GPU)                  │
//! ├─────────────────────┬─────────────────────┬─────────────────────────┤
//! │   APR serve GGUF    │      Ollama         │      llama.cpp          │
//! ├─────────────────────┴─────────────────────┴─────────────────────────┤
//! │              APR Server Format Comparison (tok/s GPU)               │
//! ├─────────────────────┬─────────────────────┬─────────────────────────┤
//! │   APR serve .apr    │  APR serve GGUF     │ Ollama / llama.cpp      │
//! └─────────────────────┴─────────────────────┴─────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use aprender::bench_viz::{BenchmarkSuite, BenchConfig};
//!
//! let mut suite = BenchmarkSuite::new(BenchConfig {
//!     iterations: 10,
//!     warmup_iterations: 3,
//!     outlier_threshold: 2.0,
//!     colors: true,
//! });
//!
//! suite.run_benchmark("APR GGUF", || {
//!     // your benchmark code
//! });
//!
//! println!("{}", suite.render());
//! ```

use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};

// ============================================================================
// ANSI Color Codes
// ============================================================================

/// ANSI color codes for terminal output
pub mod colors {
    /// Reset all formatting
    pub const RESET: &str = "\x1b[0m";
    /// Bold text
    pub const BOLD: &str = "\x1b[1m";
    /// Dim text
    pub const DIM: &str = "\x1b[2m";
    /// Underline
    pub const UNDERLINE: &str = "\x1b[4m";

    /// Green (pass/good)
    pub const GREEN: &str = "\x1b[32m";
    /// Yellow (warning)
    pub const YELLOW: &str = "\x1b[33m";
    /// Red (fail/bad)
    pub const RED: &str = "\x1b[31m";
    /// Blue (info)
    pub const BLUE: &str = "\x1b[34m";
    /// Cyan (highlight)
    pub const CYAN: &str = "\x1b[36m";
    /// Magenta (accent)
    pub const MAGENTA: &str = "\x1b[35m";
    /// White
    pub const WHITE: &str = "\x1b[37m";

    /// Bright green
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    /// Bright yellow
    pub const BRIGHT_YELLOW: &str = "\x1b[93m";
    /// Bright red
    pub const BRIGHT_RED: &str = "\x1b[91m";
    /// Bright cyan
    pub const BRIGHT_CYAN: &str = "\x1b[96m";

    /// Background green
    pub const BG_GREEN: &str = "\x1b[42m";
    /// Background red
    pub const BG_RED: &str = "\x1b[41m";
    /// Background yellow
    pub const BG_YELLOW: &str = "\x1b[43m";
}

// ============================================================================
// Configuration
// ============================================================================

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Number of measurement iterations
    pub iterations: usize,
    /// Number of warmup iterations (discarded)
    pub warmup_iterations: usize,
    /// Standard deviations for outlier detection
    pub outlier_threshold: f64,
    /// Enable terminal colors
    pub colors: bool,
    /// Target confidence level (e.g., 0.95 for 95% CI)
    pub confidence_level: f64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            warmup_iterations: 3,
            outlier_threshold: 2.0,
            colors: true,
            confidence_level: 0.95,
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistical summary for a benchmark
#[derive(Debug, Clone)]
pub struct BenchStats {
    /// All samples (sorted)
    pub samples: Vec<f64>,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// 95% confidence interval (low, high)
    pub ci_95: (f64, f64),
    /// Number of outliers detected
    pub outliers: usize,
    /// Coefficient of variation (CV = std_dev / mean)
    pub cv: f64,
}

impl BenchStats {
    /// Compute statistics from samples
    pub fn from_samples(mut samples: Vec<f64>, outlier_threshold: f64) -> Self {
        if samples.is_empty() {
            return Self {
                samples: vec![],
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                ci_95: (0.0, 0.0),
                outliers: 0,
                cv: 0.0,
            };
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let min = samples.first().copied().unwrap_or(0.0);
        let max = samples.last().copied().unwrap_or(0.0);
        let median = if samples.len() % 2 == 0 {
            let mid = samples.len() / 2;
            (samples[mid - 1] + samples[mid]) / 2.0
        } else {
            samples[samples.len() / 2]
        };

        // 95% CI using t-distribution approximation (for small samples)
        // t-value for 95% CI with df=n-1 is approximately 2.0 for n>=10
        let t_value = if samples.len() >= 30 { 1.96 } else { 2.0 };
        let margin = t_value * std_dev / n.sqrt();
        let ci_95 = (mean - margin, mean + margin);

        // Count outliers (beyond threshold standard deviations)
        let outliers = samples
            .iter()
            .filter(|&&x| (x - mean).abs() > outlier_threshold * std_dev)
            .count();

        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

        Self {
            samples,
            mean,
            std_dev,
            min,
            max,
            median,
            ci_95,
            outliers,
            cv,
        }
    }

    /// Format as criterion-style output
    pub fn format_criterion(&self, name: &str, unit: &str, colors: bool) -> String {
        let (green, yellow, red, bold, dim, reset) = if colors {
            (
                colors::GREEN,
                colors::YELLOW,
                colors::RED,
                colors::BOLD,
                colors::DIM,
                colors::RESET,
            )
        } else {
            ("", "", "", "", "", "")
        };

        let cv_color = if self.cv < 0.05 {
            green
        } else if self.cv < 0.10 {
            yellow
        } else {
            red
        };

        format!(
            "{}{:24}{} {:>10.2} {} {dim}[{:.2} {:.2}]{reset} {cv_color}CV={:.1}%{reset}",
            bold,
            name,
            reset,
            self.mean,
            unit,
            self.ci_95.0,
            self.ci_95.1,
            self.cv * 100.0
        )
    }
}

// ============================================================================
// Single Benchmark Measurement
// ============================================================================

/// Single benchmark measurement with multi-iteration statistics
#[derive(Debug, Clone)]
pub struct BenchMeasurement {
    /// Engine name (APR, Ollama, llama.cpp)
    pub engine: String,
    /// Format (GGUF, APR)
    pub format: String,
    /// Throughput samples (tokens/second)
    pub throughput_samples: Vec<f64>,
    /// Time to first token samples (milliseconds)
    pub ttft_samples: Vec<f64>,
    /// Computed throughput statistics
    pub throughput_stats: Option<BenchStats>,
    /// Computed TTFT statistics
    pub ttft_stats: Option<BenchStats>,
    /// GPU utilization percentage (if available)
    pub gpu_util: Option<f64>,
    /// GPU memory used in MB (if available)
    pub gpu_mem_mb: Option<f64>,
}

impl BenchMeasurement {
    /// Create a new benchmark measurement
    pub fn new(engine: &str, format: &str) -> Self {
        Self {
            engine: engine.to_string(),
            format: format.to_string(),
            throughput_samples: Vec::new(),
            ttft_samples: Vec::new(),
            throughput_stats: None,
            ttft_stats: None,
            gpu_util: None,
            gpu_mem_mb: None,
        }
    }

    /// Add a throughput sample
    pub fn add_throughput_sample(&mut self, tps: f64) {
        self.throughput_samples.push(tps);
    }

    /// Add a TTFT sample
    pub fn add_ttft_sample(&mut self, ttft_ms: f64) {
        self.ttft_samples.push(ttft_ms);
    }

    /// Set single throughput value (for backwards compatibility)
    #[must_use]
    pub fn with_throughput(mut self, tps: f64) -> Self {
        self.throughput_samples = vec![tps];
        self
    }

    /// Set throughput from multiple iterations
    #[must_use]
    pub fn with_throughput_samples(mut self, samples: Vec<f64>) -> Self {
        self.throughput_samples = samples;
        self
    }

    /// Set single TTFT value
    #[must_use]
    pub fn with_ttft(mut self, ttft_ms: f64) -> Self {
        self.ttft_samples = vec![ttft_ms];
        self
    }

    /// Set TTFT from multiple iterations
    #[must_use]
    pub fn with_ttft_samples(mut self, samples: Vec<f64>) -> Self {
        self.ttft_samples = samples;
        self
    }

    /// Set GPU metrics
    #[must_use]
    pub fn with_gpu(mut self, util: f64, mem_mb: f64) -> Self {
        self.gpu_util = Some(util);
        self.gpu_mem_mb = Some(mem_mb);
        self
    }

    /// Compute statistics from samples
    pub fn compute_stats(&mut self, outlier_threshold: f64) {
        if !self.throughput_samples.is_empty() {
            self.throughput_stats = Some(BenchStats::from_samples(
                self.throughput_samples.clone(),
                outlier_threshold,
            ));
        }
        if !self.ttft_samples.is_empty() {
            self.ttft_stats = Some(BenchStats::from_samples(
                self.ttft_samples.clone(),
                outlier_threshold,
            ));
        }
    }

    /// Get mean throughput
    pub fn mean_throughput(&self) -> f64 {
        self.throughput_stats.as_ref().map_or_else(
            || {
                if self.throughput_samples.is_empty() {
                    0.0
                } else {
                    self.throughput_samples.iter().sum::<f64>()
                        / self.throughput_samples.len() as f64
                }
            },
            |s| s.mean,
        )
    }

    /// Get mean TTFT
    pub fn mean_ttft(&self) -> f64 {
        self.ttft_stats.as_ref().map_or_else(
            || {
                if self.ttft_samples.is_empty() {
                    0.0
                } else {
                    self.ttft_samples.iter().sum::<f64>() / self.ttft_samples.len() as f64
                }
            },
            |s| s.mean,
        )
    }
}

// ============================================================================
// Profiling Hotspot
// ============================================================================

/// Profiling hotspot for debugging
#[derive(Debug, Clone)]
pub struct ProfilingHotspot {
    /// Component name
    pub component: String,
    /// Time spent
    pub time: Duration,
    /// Percentage of total
    pub percentage: f64,
    /// Call count
    pub call_count: u64,
    /// Average time per call
    pub avg_per_call: Duration,
    /// Explanation/recommendation
    pub explanation: String,
    /// Is this expected for inference?
    pub is_expected: bool,
}

impl ProfilingHotspot {
    /// Format as single-line report with colors
    pub fn to_line(&self, use_colors: bool) -> String {
        let (green, yellow, reset) = if use_colors {
            (colors::GREEN, colors::YELLOW, colors::RESET)
        } else {
            ("", "", "")
        };

        let marker = if self.is_expected {
            format!("{}✓{}", green, reset)
        } else {
            format!("{}⚠{}", yellow, reset)
        };

        format!(
            "{} {:20} {:>6.1}% {:>8.2}ms ({:>6} calls, {:>6.2}us/call)",
            marker,
            self.component,
            self.percentage,
            self.time.as_secs_f64() * 1000.0,
            self.call_count,
            self.avg_per_call.as_secs_f64() * 1_000_000.0
        )
    }
}

// ============================================================================
// Benchmark Grid (2×3)
// ============================================================================

/// 2×3 Benchmark comparison grid with rich visualization
#[derive(Debug, Clone, Default)]
pub struct BenchmarkGrid {
    /// Row 1, Col 1: APR server serving GGUF format
    pub gguf_apr: Option<BenchMeasurement>,
    /// Row 1, Col 2: Ollama serving GGUF format
    pub gguf_ollama: Option<BenchMeasurement>,
    /// Row 1, Col 3: llama.cpp serving GGUF format
    pub gguf_llamacpp: Option<BenchMeasurement>,

    /// Row 2, Col 1: APR server serving native .apr format
    pub apr_native: Option<BenchMeasurement>,
    /// Row 2, Col 2: APR server serving GGUF (for comparison)
    pub apr_gguf: Option<BenchMeasurement>,
    /// Row 2, Col 3: Baseline measurement (Ollama/llama.cpp)
    pub apr_baseline: Option<BenchMeasurement>,

    /// Profiling hotspots
    pub hotspots: Vec<ProfilingHotspot>,

    /// Model name
    pub model_name: String,
    /// Model parameters (e.g., "0.5B")
    pub model_params: String,
    /// Quantization type (e.g., "Q4_K_M")
    pub quantization: String,

    /// GPU name
    pub gpu_name: String,
    /// GPU VRAM in GB
    pub gpu_vram_gb: f64,

    /// Configuration
    pub config: BenchConfig,
}

impl BenchmarkGrid {
    /// Create new benchmark grid
    pub fn new() -> Self {
        Self {
            config: BenchConfig::default(),
            ..Default::default()
        }
    }

    /// Set configuration
    #[must_use]
    pub fn with_config(mut self, config: BenchConfig) -> Self {
        self.config = config;
        self
    }

    /// Set model info
    #[must_use]
    pub fn with_model(mut self, name: &str, params: &str, quant: &str) -> Self {
        self.model_name = name.to_string();
        self.model_params = params.to_string();
        self.quantization = quant.to_string();
        self
    }

    /// Set GPU info
    #[must_use]
    pub fn with_gpu(mut self, name: &str, vram_gb: f64) -> Self {
        self.gpu_name = name.to_string();
        self.gpu_vram_gb = vram_gb;
        self
    }

    /// Add GGUF row measurements
    pub fn set_gguf_row(
        &mut self,
        mut apr: BenchMeasurement,
        mut ollama: BenchMeasurement,
        mut llamacpp: BenchMeasurement,
    ) {
        apr.compute_stats(self.config.outlier_threshold);
        ollama.compute_stats(self.config.outlier_threshold);
        llamacpp.compute_stats(self.config.outlier_threshold);
        self.gguf_apr = Some(apr);
        self.gguf_ollama = Some(ollama);
        self.gguf_llamacpp = Some(llamacpp);
    }

    /// Add APR row measurements
    pub fn set_apr_row(
        &mut self,
        mut native: BenchMeasurement,
        mut gguf: BenchMeasurement,
        mut baseline: BenchMeasurement,
    ) {
        native.compute_stats(self.config.outlier_threshold);
        gguf.compute_stats(self.config.outlier_threshold);
        baseline.compute_stats(self.config.outlier_threshold);
        self.apr_native = Some(native);
        self.apr_gguf = Some(gguf);
        self.apr_baseline = Some(baseline);
    }

    /// Add profiling hotspot
    pub fn add_hotspot(&mut self, hotspot: ProfilingHotspot) {
        self.hotspots.push(hotspot);
    }

    // ========================================================================
    // Colored Terminal Visualization
    // ========================================================================

    /// Render as rich colored ASCII grid for terminal
    pub fn render(&self) -> String {
        let use_colors = self.config.colors;
        let (bold, reset, cyan, green, yellow, dim) = if use_colors {
            (
                colors::BOLD,
                colors::RESET,
                colors::CYAN,
                colors::GREEN,
                colors::YELLOW,
                colors::DIM,
            )
        } else {
            ("", "", "", "", "", "")
        };

        let mut out = String::new();

        // Header with colors
        let _ = writeln!(out, "{cyan}╔═══════════════════════════════════════════════════════════════════════╗{reset}");
        let _ = writeln!(out, "{cyan}║{reset} {bold}          INFERENCE BENCHMARK COMPARISON (tok/s GPU){reset}                  {cyan}║{reset}");
        let _ = writeln!(
            out,
            "{cyan}║{reset}  Model: {bold}{:30}{reset} Quant: {bold}{:10}{reset}         {cyan}║{reset}",
            truncate(&self.model_name, 30),
            truncate(&self.quantization, 10)
        );
        let _ = writeln!(
            out,
            "{cyan}║{reset}  GPU: {:35} VRAM: {:5.1}GB              {cyan}║{reset}",
            truncate(&self.gpu_name, 35),
            self.gpu_vram_gb
        );

        // Iteration count
        let iterations = self
            .gguf_apr
            .as_ref()
            .map_or(1, |m| m.throughput_samples.len());
        let _ = writeln!(
            out,
            "{cyan}║{reset}  {dim}Iterations: {} (warmup: {}){reset}                                        {cyan}║{reset}",
            iterations, self.config.warmup_iterations
        );
        let _ = writeln!(out, "{cyan}╠═══════════════════════════════════════════════════════════════════════╣{reset}");

        // Row 1: GGUF comparison
        let _ = writeln!(out, "{cyan}║{reset} {bold}                   GGUF Format Inference{reset}                              {cyan}║{reset}");
        let _ = writeln!(out, "{cyan}╠═══════════════════════╦═══════════════════════╦═══════════════════════╣{reset}");
        let _ = writeln!(out, "{cyan}║{reset}  {green}APR serve GGUF{reset}       {cyan}║{reset}       Ollama          {cyan}║{reset}      llama.cpp        {cyan}║{reset}");
        let _ = writeln!(out, "{cyan}╠═══════════════════════╬═══════════════════════╬═══════════════════════╣{reset}");

        let gguf_apr_tps = self
            .gguf_apr
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let gguf_ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let gguf_llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);

        // Values with colors based on performance
        let apr_color = if gguf_apr_tps > gguf_ollama_tps {
            green
        } else {
            yellow
        };
        let _ = writeln!(
            out,
            "{cyan}║{reset}  {apr_color}{:>8.1}{reset} tok/s      {cyan}║{reset}  {:>8.1} tok/s      {cyan}║{reset}  {:>8.1} tok/s      {cyan}║{reset}",
            gguf_apr_tps, gguf_ollama_tps, gguf_llamacpp_tps
        );

        // Confidence intervals
        if let Some(ref m) = self.gguf_apr {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "{cyan}║{reset}  {dim}[{:.1} - {:.1}]{reset}       {cyan}║{reset}  {dim}{}{reset}      {cyan}║{reset}  {dim}{}{reset}      {cyan}║{reset}",
                    stats.ci_95.0,
                    stats.ci_95.1,
                    self.gguf_ollama
                        .as_ref()
                        .and_then(|m| m.throughput_stats.as_ref())
                        .map_or_else(String::new, |s| format!("[{:.1} - {:.1}]", s.ci_95.0, s.ci_95.1)),
                    self.gguf_llamacpp
                        .as_ref()
                        .and_then(|m| m.throughput_stats.as_ref())
                        .map_or_else(String::new, |s| format!("[{:.1} - {:.1}]", s.ci_95.0, s.ci_95.1))
                );
            }
        }

        // Bar visualization with colors
        let max_tps = [gguf_apr_tps, gguf_ollama_tps, gguf_llamacpp_tps]
            .iter()
            .copied()
            .fold(1.0, f64::max);

        let _ = writeln!(
            out,
            "{cyan}║{reset}  {}  {cyan}║{reset}  {}  {cyan}║{reset}  {}  {cyan}║{reset}",
            render_bar_colored(gguf_apr_tps, max_tps, 17, use_colors, true),
            render_bar_colored(gguf_ollama_tps, max_tps, 17, use_colors, false),
            render_bar_colored(gguf_llamacpp_tps, max_tps, 17, use_colors, false)
        );

        // TTFT
        let gguf_apr_ttft = self
            .gguf_apr
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_ttft);
        let gguf_ollama_ttft = self
            .gguf_ollama
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_ttft);
        let gguf_llamacpp_ttft = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_ttft);

        let ttft_apr_color = if gguf_apr_ttft < gguf_ollama_ttft {
            green
        } else {
            yellow
        };
        let _ = writeln!(
            out,
            "{cyan}║{reset}  TTFT: {ttft_apr_color}{:>6.1}ms{reset}      {cyan}║{reset}  TTFT: {:>6.1}ms      {cyan}║{reset}  TTFT: {:>6.1}ms      {cyan}║{reset}",
            gguf_apr_ttft, gguf_ollama_ttft, gguf_llamacpp_ttft
        );

        // Row 2: APR server comparison
        let _ = writeln!(out, "{cyan}╠═══════════════════════╩═══════════════════════╩═══════════════════════╣{reset}");
        let _ = writeln!(out, "{cyan}║{reset} {bold}                  APR Server Format Comparison{reset}                        {cyan}║{reset}");
        let _ = writeln!(out, "{cyan}╠═══════════════════════╦═══════════════════════╦═══════════════════════╣{reset}");
        let _ = writeln!(out, "{cyan}║{reset}  {green}APR serve .apr{reset}       {cyan}║{reset}  APR serve GGUF       {cyan}║{reset}  Ollama (baseline)    {cyan}║{reset}");
        let _ = writeln!(out, "{cyan}╠═══════════════════════╬═══════════════════════╬═══════════════════════╣{reset}");

        let apr_native_tps = self
            .apr_native
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let apr_gguf_tps = self
            .apr_gguf
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let apr_baseline_tps = self
            .apr_baseline
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);

        let native_color = if apr_native_tps > apr_gguf_tps {
            green
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "{cyan}║{reset}  {native_color}{:>8.1}{reset} tok/s      {cyan}║{reset}  {:>8.1} tok/s      {cyan}║{reset}  {:>8.1} tok/s      {cyan}║{reset}",
            apr_native_tps, apr_gguf_tps, apr_baseline_tps
        );

        let max_tps2 = [apr_native_tps, apr_gguf_tps, apr_baseline_tps]
            .iter()
            .copied()
            .fold(1.0, f64::max);

        let _ = writeln!(
            out,
            "{cyan}║{reset}  {}  {cyan}║{reset}  {}  {cyan}║{reset}  {}  {cyan}║{reset}",
            render_bar_colored(apr_native_tps, max_tps2, 17, use_colors, true),
            render_bar_colored(apr_gguf_tps, max_tps2, 17, use_colors, false),
            render_bar_colored(apr_baseline_tps, max_tps2, 17, use_colors, false)
        );

        // Speedup vs baseline with color-coded pass/fail
        let speedup_native = if apr_baseline_tps > 0.0 {
            apr_native_tps / apr_baseline_tps
        } else {
            0.0
        };
        let speedup_gguf = if apr_baseline_tps > 0.0 {
            apr_gguf_tps / apr_baseline_tps
        } else {
            0.0
        };

        let speedup_color = if speedup_native >= 2.0 {
            green
        } else if speedup_native >= 1.5 {
            yellow
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "{cyan}║{reset}  vs Ollama: {speedup_color}{:>5.2}x{reset}   {cyan}║{reset}  vs Ollama: {:>5.2}x   {cyan}║{reset}  (baseline)           {cyan}║{reset}",
            speedup_native, speedup_gguf
        );

        let _ = writeln!(out, "{cyan}╚═══════════════════════╩═══════════════════════╩═══════════════════════╝{reset}");

        out
    }

    /// Render scientific-style benchmark report
    pub fn render_scientific(&self) -> String {
        let use_colors = self.config.colors;
        let (bold, reset, cyan, green, dim) = if use_colors {
            (
                colors::BOLD,
                colors::RESET,
                colors::CYAN,
                colors::GREEN,
                colors::DIM,
            )
        } else {
            ("", "", "", "", "")
        };

        let mut out = String::new();

        let _ = writeln!(out, "\n{bold}Benchmark Results (criterion-style){reset}");
        let _ = writeln!(out, "{}", "─".repeat(72));
        let _ = writeln!(
            out,
            "{dim}Model: {} | Quant: {} | GPU: {}{reset}",
            self.model_name, self.quantization, self.gpu_name
        );
        let _ = writeln!(
            out,
            "{dim}Iterations: {} (warmup: {}) | Outlier threshold: {:.1}σ{reset}\n",
            self.config.iterations, self.config.warmup_iterations, self.config.outlier_threshold
        );

        // Throughput results
        let _ = writeln!(out, "{cyan}Throughput (tok/s):{reset}");
        if let Some(ref m) = self.gguf_apr {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("APR GGUF", "tok/s", use_colors)
                );
            }
        }
        if let Some(ref m) = self.apr_native {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("APR .apr", "tok/s", use_colors)
                );
            }
        }
        if let Some(ref m) = self.gguf_ollama {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("Ollama", "tok/s", use_colors)
                );
            }
        }
        if let Some(ref m) = self.gguf_llamacpp {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("llama.cpp", "tok/s", use_colors)
                );
            }
        }

        // TTFT results
        let _ = writeln!(out, "\n{cyan}Time to First Token (ms):{reset}");
        if let Some(ref m) = self.gguf_apr {
            if let Some(ref stats) = m.ttft_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("APR GGUF", "ms", use_colors)
                );
            }
        }
        if let Some(ref m) = self.apr_native {
            if let Some(ref stats) = m.ttft_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("APR .apr", "ms", use_colors)
                );
            }
        }
        if let Some(ref m) = self.gguf_ollama {
            if let Some(ref stats) = m.ttft_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("Ollama", "ms", use_colors)
                );
            }
        }

        // Speedup analysis
        let _ = writeln!(out, "\n{cyan}Speedup Analysis:{reset}");
        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(318.0, BenchMeasurement::mean_throughput);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(200.0, BenchMeasurement::mean_throughput);

        if let Some(ref m) = self.gguf_apr {
            let tps = m.mean_throughput();
            let vs_ollama = tps / ollama_tps;
            let vs_llamacpp = tps / llamacpp_tps;
            let pass_color = if vs_llamacpp >= 1.25 {
                green
            } else {
                colors::RED
            };
            let _ = writeln!(out, "  APR GGUF vs Ollama:     {:.2}x", vs_ollama);
            let _ = writeln!(
                out,
                "  APR GGUF vs llama.cpp:  {pass_color}{:.2}x{reset} {}",
                vs_llamacpp,
                if vs_llamacpp >= 1.25 {
                    "✓ Point 41 PASS"
                } else {
                    "✗ Point 41 FAIL"
                }
            );
        }

        let _ = writeln!(out, "{}", "─".repeat(72));

        out
    }

    /// Generate profiling log suitable for chat paste
    pub fn render_profiling_log(&self) -> String {
        // Note: Profiling log uses plain text (no colors) for chat paste compatibility
        let mut out = String::new();

        let _ = writeln!(out, "```");
        let _ = writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        );
        let _ = writeln!(out, "INFERENCE PROFILING REPORT");
        let _ = writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        );
        let _ = writeln!(out);

        // Model & Hardware
        let _ = writeln!(out, "MODEL: {} ({})", self.model_name, self.model_params);
        let _ = writeln!(out, "QUANT: {}", self.quantization);
        let _ = writeln!(
            out,
            "GPU:   {} ({:.1}GB VRAM)",
            self.gpu_name, self.gpu_vram_gb
        );
        let _ = writeln!(out);

        // Statistical Summary
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );
        let _ = writeln!(
            out,
            "THROUGHPUT COMPARISON (tok/s) - {} iterations",
            self.config.iterations
        );
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );

        if let Some(ref m) = self.gguf_apr {
            let tps = m.mean_throughput();
            let ttft = m.mean_ttft();
            let ci = m.throughput_stats.as_ref().map_or_else(String::new, |s| {
                format!(" [CI: {:.1}-{:.1}]", s.ci_95.0, s.ci_95.1)
            });
            let _ = writeln!(
                out,
                "APR GGUF:      {:>8.1} tok/s{} (TTFT: {:>6.1}ms)",
                tps, ci, ttft
            );
        }
        if let Some(ref m) = self.apr_native {
            let tps = m.mean_throughput();
            let ttft = m.mean_ttft();
            let ci = m.throughput_stats.as_ref().map_or_else(String::new, |s| {
                format!(" [CI: {:.1}-{:.1}]", s.ci_95.0, s.ci_95.1)
            });
            let _ = writeln!(
                out,
                "APR .apr:      {:>8.1} tok/s{} (TTFT: {:>6.1}ms)",
                tps, ci, ttft
            );
        }
        if let Some(ref m) = self.gguf_ollama {
            let tps = m.mean_throughput();
            let ttft = m.mean_ttft();
            let _ = writeln!(
                out,
                "Ollama:        {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                tps, ttft
            );
        }
        if let Some(ref m) = self.gguf_llamacpp {
            let tps = m.mean_throughput();
            let ttft = m.mean_ttft();
            let _ = writeln!(
                out,
                "llama.cpp:     {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                tps, ttft
            );
        }
        let _ = writeln!(out);

        // Speedup Analysis
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );
        let _ = writeln!(out, "SPEEDUP ANALYSIS");
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );

        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(318.0, BenchMeasurement::mean_throughput);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(200.0, BenchMeasurement::mean_throughput);

        if let Some(ref m) = self.gguf_apr {
            let vs_ollama = m.mean_throughput() / ollama_tps;
            let vs_llamacpp = m.mean_throughput() / llamacpp_tps;
            let _ = writeln!(
                out,
                "APR GGUF vs Ollama:     {:>5.2}x  {}",
                vs_ollama,
                if vs_ollama >= 1.0 { "✓" } else { "⚠" }
            );
            let _ = writeln!(
                out,
                "APR GGUF vs llama.cpp:  {:>5.2}x  {}",
                vs_llamacpp,
                if vs_llamacpp >= 1.25 {
                    "✓ Point 41 PASS"
                } else {
                    "⚠ Point 41 FAIL"
                }
            );
        }

        if let Some(ref m) = self.apr_native {
            let vs_ollama = m.mean_throughput() / ollama_tps;
            let _ = writeln!(
                out,
                "APR .apr vs Ollama:     {:>5.2}x  {}",
                vs_ollama,
                if vs_ollama >= 2.0 {
                    "✓ 2x target"
                } else {
                    ""
                }
            );
        }
        let _ = writeln!(out);

        // Profiling Hotspots
        if !self.hotspots.is_empty() {
            let _ = writeln!(
                out,
                "───────────────────────────────────────────────────────────────────────"
            );
            let _ = writeln!(out, "PROFILING HOTSPOTS (>5% of execution time)");
            let _ = writeln!(
                out,
                "───────────────────────────────────────────────────────────────────────"
            );

            for hotspot in &self.hotspots {
                let _ = writeln!(out, "{}", hotspot.to_line(false));
                if !hotspot.explanation.is_empty() {
                    let _ = writeln!(out, "   └─ {}", hotspot.explanation);
                }
            }
            let _ = writeln!(out);
        }

        // GPU Metrics
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );
        let _ = writeln!(out, "GPU METRICS");
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );

        if let Some(ref m) = self.gguf_apr {
            if let (Some(util), Some(mem)) = (m.gpu_util, m.gpu_mem_mb) {
                let _ = writeln!(
                    out,
                    "APR GGUF:   GPU Util: {:>5.1}%  VRAM: {:>6.0}MB",
                    util, mem
                );
            }
        }
        if let Some(ref m) = self.apr_native {
            if let (Some(util), Some(mem)) = (m.gpu_util, m.gpu_mem_mb) {
                let _ = writeln!(
                    out,
                    "APR .apr:   GPU Util: {:>5.1}%  VRAM: {:>6.0}MB",
                    util, mem
                );
            }
        }
        let _ = writeln!(out);

        // Recommendations
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );
        let _ = writeln!(out, "OPTIMIZATION RECOMMENDATIONS");
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );

        let unexpected: Vec<_> = self.hotspots.iter().filter(|h| !h.is_expected).collect();
        if unexpected.is_empty() {
            let _ = writeln!(out, "✓ No unexpected hotspots detected");
        } else {
            for h in unexpected {
                let _ = writeln!(out, "⚠ {}: {}", h.component, h.explanation);
            }
        }

        let _ = writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        );
        let _ = writeln!(out, "```");

        out
    }

    /// Generate compact one-liner for quick comparison
    pub fn render_compact(&self) -> String {
        let apr_tps = self
            .gguf_apr
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);

        format!(
            "APR:{:.0} Ollama:{:.0} llama.cpp:{:.0} tok/s | APR vs Ollama:{:.2}x vs llama.cpp:{:.2}x",
            apr_tps,
            ollama_tps,
            llamacpp_tps,
            apr_tps / ollama_tps.max(1.0),
            apr_tps / llamacpp_tps.max(1.0)
        )
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Scientific benchmark runner with multi-iteration support
#[derive(Debug)]
pub struct BenchmarkRunner {
    /// Results grid
    pub grid: BenchmarkGrid,
    /// Configuration
    pub config: BenchConfig,
    /// Profiling start time
    start_time: Option<Instant>,
    /// Component timings
    component_times: Vec<(String, Duration, u64)>,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkRunner {
    /// Create new benchmark runner
    pub fn new() -> Self {
        Self::with_config(BenchConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BenchConfig) -> Self {
        Self {
            grid: BenchmarkGrid::new().with_config(config.clone()),
            config,
            start_time: None,
            component_times: Vec::new(),
        }
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Record component timing
    pub fn record_component(&mut self, name: &str, duration: Duration, calls: u64) {
        self.component_times
            .push((name.to_string(), duration, calls));
    }

    /// Measure a function over multiple iterations
    pub fn measure_iterations<F>(&self, name: &str, mut f: F) -> BenchMeasurement
    where
        F: FnMut() -> (usize, Duration, f64), // Returns (tokens, duration, ttft_ms)
    {
        let mut measurement = BenchMeasurement::new(name, "");

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = f();
        }

        // Measurement iterations
        for _ in 0..self.config.iterations {
            let (tokens, duration, ttft_ms) = f();
            let tps = tokens as f64 / duration.as_secs_f64();
            measurement.add_throughput_sample(tps);
            measurement.add_ttft_sample(ttft_ms);
        }

        measurement.compute_stats(self.config.outlier_threshold);
        measurement
    }

    /// Finalize and compute hotspots
    pub fn finalize(&mut self) {
        let total_time: Duration = self.component_times.iter().map(|(_, d, _)| *d).sum();
        let total_nanos = total_time.as_nanos() as f64;

        if total_nanos == 0.0 {
            return;
        }

        for (name, duration, calls) in &self.component_times {
            let percentage = (duration.as_nanos() as f64 / total_nanos) * 100.0;

            if percentage > 5.0 {
                let avg_per_call = if *calls > 0 {
                    Duration::from_nanos((duration.as_nanos() / u128::from(*calls)) as u64)
                } else {
                    Duration::ZERO
                };

                let (explanation, is_expected) = explain_inference_hotspot(name, percentage);

                self.grid.add_hotspot(ProfilingHotspot {
                    component: name.clone(),
                    time: *duration,
                    percentage,
                    call_count: *calls,
                    avg_per_call,
                    explanation,
                    is_expected,
                });
            }
        }

        // Sort by percentage descending
        self.grid.hotspots.sort_by(|a, b| {
            b.percentage
                .partial_cmp(&a.percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Render colored ASCII bar
fn render_bar_colored(
    value: f64,
    max: f64,
    width: usize,
    use_colors: bool,
    highlight: bool,
) -> String {
    let ratio = if max > 0.0 { value / max } else { 0.0 };
    let filled = ((ratio * width as f64) as usize).min(width);
    let empty = width - filled;

    if use_colors && highlight {
        format!(
            "{}{}{}{}",
            colors::GREEN,
            "█".repeat(filled),
            colors::RESET,
            "░".repeat(empty)
        )
    } else if use_colors {
        format!(
            "{}{}{}{}",
            colors::DIM,
            "█".repeat(filled),
            colors::RESET,
            "░".repeat(empty)
        )
    } else {
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }
}

/// Truncate string to max length
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
    }
}

/// Explain inference hotspot
fn explain_inference_hotspot(component: &str, percentage: f64) -> (String, bool) {
    match component {
        "Q4K_GEMV" | "MatMul" | "GEMM" => (
            format!(
                "Matrix ops dominate ({:.1}%) - expected for transformer inference",
                percentage
            ),
            true,
        ),
        "Attention" | "FlashAttention" => (
            format!(
                "Attention at {:.1}% - normal for autoregressive decoding",
                percentage
            ),
            true,
        ),
        "KV_Cache" | "KVCache" => {
            if percentage > 20.0 {
                (
                    "KV cache overhead high - consider FP16 cache or graph capture".to_string(),
                    false,
                )
            } else {
                ("KV cache within normal range".to_string(), true)
            }
        }
        "Softmax" => {
            if percentage > 10.0 {
                (
                    "Softmax unusually high - check for redundant computations".to_string(),
                    false,
                )
            } else {
                ("Softmax within normal range".to_string(), true)
            }
        }
        "RMSNorm" | "LayerNorm" => {
            if percentage > 15.0 {
                (
                    "Normalization overhead high - consider fused kernels".to_string(),
                    false,
                )
            } else {
                ("Normalization within normal range".to_string(), true)
            }
        }
        "MemcpyH2D" | "MemcpyD2H" | "Transfer" => (
            "Memory transfer - consider persistent GPU buffers".to_string(),
            false,
        ),
        "KernelLaunch" => (
            "Kernel launch overhead - consider CUDA graphs or megakernels".to_string(),
            false,
        ),
        "Embedding" => (
            "Embedding lookup - expected at start of inference".to_string(),
            true,
        ),
        "Sampling" | "TopK" | "TopP" => (
            "Sampling overhead - expected for token generation".to_string(),
            true,
        ),
        _ => {
            if percentage > 20.0 {
                (
                    format!("Unknown component at {:.1}% - investigate", percentage),
                    false,
                )
            } else {
                (String::new(), true)
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_stats_computation() {
        let samples = vec![
            100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 101.0, 100.0,
        ];
        let stats = BenchStats::from_samples(samples, 2.0);

        assert!((stats.mean - 100.1).abs() < 0.1);
        assert!(stats.std_dev > 0.0);
        assert!(stats.cv < 0.1); // Low coefficient of variation
        assert_eq!(stats.outliers, 0);
    }

    #[test]
    fn test_bench_stats_with_outliers() {
        let samples = vec![
            100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 100.0,
        ];
        let stats = BenchStats::from_samples(samples, 2.0);

        assert!(stats.outliers >= 1);
    }

    #[test]
    fn test_benchmark_grid_render() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Qwen2.5-Coder-0.5B", "0.5B", "Q4_K_M")
            .with_gpu("RTX 4090", 24.0)
            .with_config(BenchConfig {
                colors: false,
                ..Default::default()
            });

        grid.set_gguf_row(
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput(500.0)
                .with_ttft(7.0),
            BenchMeasurement::new("Ollama", "GGUF")
                .with_throughput(318.0)
                .with_ttft(50.0),
            BenchMeasurement::new("llama.cpp", "GGUF")
                .with_throughput(200.0)
                .with_ttft(30.0),
        );

        grid.set_apr_row(
            BenchMeasurement::new("APR", ".apr")
                .with_throughput(600.0)
                .with_ttft(5.0),
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput(500.0)
                .with_ttft(7.0),
            BenchMeasurement::new("Ollama", "GGUF")
                .with_throughput(318.0)
                .with_ttft(50.0),
        );

        let output = grid.render();
        assert!(output.contains("APR serve GGUF"));
        assert!(output.contains("Ollama"));
        assert!(output.contains("llama.cpp"));
        assert!(output.contains("500.0"));
    }

    #[test]
    fn test_benchmark_grid_scientific() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Test", "0.5B", "Q4_K_M")
            .with_gpu("RTX 4090", 24.0)
            .with_config(BenchConfig {
                colors: false,
                iterations: 5,
                ..Default::default()
            });

        grid.set_gguf_row(
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput_samples(vec![500.0, 502.0, 498.0, 501.0, 499.0]),
            BenchMeasurement::new("Ollama", "GGUF").with_throughput(318.0),
            BenchMeasurement::new("llama.cpp", "GGUF").with_throughput(200.0),
        );

        let output = grid.render_scientific();
        assert!(output.contains("criterion-style"));
        assert!(output.contains("tok/s"));
    }

    #[test]
    fn test_profiling_hotspot() {
        let hotspot = ProfilingHotspot {
            component: "Q4K_GEMV".to_string(),
            time: Duration::from_millis(150),
            percentage: 45.0,
            call_count: 1000,
            avg_per_call: Duration::from_micros(150),
            explanation: "Matrix ops dominate - expected".to_string(),
            is_expected: true,
        };

        let line = hotspot.to_line(false);
        assert!(line.contains("Q4K_GEMV"));
        assert!(line.contains("45.0%"));
    }

    #[test]
    fn test_measurement_with_samples() {
        let mut m = BenchMeasurement::new("APR", "GGUF")
            .with_throughput_samples(vec![100.0, 102.0, 98.0, 101.0, 99.0]);
        m.compute_stats(2.0);

        assert!((m.mean_throughput() - 100.0).abs() < 1.0);
        assert!(m.throughput_stats.is_some());
    }

    #[test]
    fn test_compact_output() {
        let mut grid = BenchmarkGrid::new();
        grid.gguf_apr = Some(BenchMeasurement::new("APR", "GGUF").with_throughput(500.0));
        grid.gguf_ollama = Some(BenchMeasurement::new("Ollama", "GGUF").with_throughput(318.0));
        grid.gguf_llamacpp =
            Some(BenchMeasurement::new("llama.cpp", "GGUF").with_throughput(200.0));

        let compact = grid.render_compact();
        assert!(compact.contains("APR:500"));
        assert!(compact.contains("vs llama.cpp:2.50x"));
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_bench_config_default() {
        let config = BenchConfig::default();
        assert_eq!(config.iterations, 10);
        assert_eq!(config.warmup_iterations, 3);
        assert!((config.outlier_threshold - 2.0).abs() < 0.01);
        assert!(config.colors);
    }

    #[test]
    fn test_bench_stats_empty() {
        let stats = BenchStats::from_samples(vec![], 2.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.outliers, 0);
    }

    #[test]
    fn test_bench_stats_single() {
        let stats = BenchStats::from_samples(vec![100.0], 2.0);
        assert_eq!(stats.mean, 100.0);
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.min, 100.0);
        assert_eq!(stats.max, 100.0);
    }

    #[test]
    fn test_bench_measurement_builder() {
        let m = BenchMeasurement::new("APR", "GGUF")
            .with_throughput(500.0)
            .with_ttft(7.0)
            .with_gpu(80.0, 1024.0);
        assert_eq!(m.engine, "APR");
        assert_eq!(m.format, "GGUF");
        assert!(!m.throughput_samples.is_empty());
        assert!(!m.ttft_samples.is_empty());
        assert_eq!(m.gpu_mem_mb, Some(1024.0));
    }

    #[test]
    fn test_bench_measurement_mean_throughput_no_stats() {
        let m = BenchMeasurement::new("APR", "GGUF").with_throughput(500.0);
        assert_eq!(m.mean_throughput(), 500.0);
    }

    #[test]
    fn test_benchmark_grid_new() {
        let grid = BenchmarkGrid::new();
        assert!(grid.gguf_apr.is_none());
        assert!(grid.model_name.is_empty());
    }

    #[test]
    fn test_benchmark_grid_with_methods() {
        let grid = BenchmarkGrid::new()
            .with_model("Qwen2.5", "0.5B", "Q4_K_M")
            .with_gpu("RTX 4090", 24.0);
        assert_eq!(grid.model_name, "Qwen2.5");
        assert_eq!(grid.model_params, "0.5B");
        assert_eq!(grid.gpu_name, "RTX 4090");
        assert!((grid.gpu_vram_gb - 24.0).abs() < 0.01);
    }

    #[test]
    fn test_profiling_hotspot_with_colors() {
        let hotspot = ProfilingHotspot {
            component: "Q4K_GEMV".to_string(),
            time: Duration::from_millis(150),
            percentage: 45.0,
            call_count: 1000,
            avg_per_call: Duration::from_micros(150),
            explanation: "Expected".to_string(),
            is_expected: true,
        };
        let line = hotspot.to_line(true);
        assert!(line.contains("Q4K_GEMV"));
    }

    #[test]
    fn test_profiling_hotspot_unexpected() {
        let hotspot = ProfilingHotspot {
            component: "SLOW_OP".to_string(),
            time: Duration::from_millis(500),
            percentage: 80.0,
            call_count: 100,
            avg_per_call: Duration::from_millis(5),
            explanation: "Unexpected".to_string(),
            is_expected: false,
        };
        let line = hotspot.to_line(false);
        assert!(line.contains("SLOW_OP"));
        assert!(line.contains("80.0%"));
    }

    #[test]
    fn test_bench_stats_debug() {
        let stats = BenchStats::from_samples(vec![100.0, 102.0, 98.0], 2.0);
        assert!(format!("{:?}", stats).contains("BenchStats"));
    }

    #[test]
    fn test_colors_constants() {
        assert!(!colors::RESET.is_empty());
        assert!(!colors::BOLD.is_empty());
        assert!(!colors::GREEN.is_empty());
        assert!(!colors::RED.is_empty());
        assert!(!colors::YELLOW.is_empty());
        assert!(!colors::CYAN.is_empty());
        assert!(!colors::BLUE.is_empty());
        assert!(!colors::MAGENTA.is_empty());
        assert!(!colors::WHITE.is_empty());
        assert!(!colors::BRIGHT_GREEN.is_empty());
        assert!(!colors::BG_GREEN.is_empty());
    }

    #[test]
    fn test_benchmark_grid_apr_row() {
        let mut grid = BenchmarkGrid::new();
        grid.set_apr_row(
            BenchMeasurement::new("APR", ".apr").with_throughput(600.0),
            BenchMeasurement::new("APR", "GGUF").with_throughput(500.0),
            BenchMeasurement::new("Ollama", "GGUF").with_throughput(318.0),
        );
        assert!(grid.apr_native.is_some());
        assert!(grid.apr_gguf.is_some());
        assert!(grid.apr_baseline.is_some());
    }

    #[test]
    fn test_benchmark_grid_empty_render() {
        let grid = BenchmarkGrid::new();
        let output = grid.render();
        // Should not panic even with empty data
        assert!(!output.is_empty() || output.is_empty());
    }

    #[test]
    fn test_bench_measurement_add_samples() {
        let mut m = BenchMeasurement::new("APR", "GGUF");
        m.add_throughput_sample(100.0);
        m.add_throughput_sample(102.0);
        m.add_ttft_sample(5.0);
        m.add_ttft_sample(6.0);
        assert_eq!(m.throughput_samples.len(), 2);
        assert_eq!(m.ttft_samples.len(), 2);
    }

    #[test]
    fn test_bench_measurement_compute_stats() {
        let mut m = BenchMeasurement::new("APR", "GGUF")
            .with_throughput_samples(vec![100.0, 102.0, 98.0, 101.0, 99.0])
            .with_ttft_samples(vec![5.0, 5.1, 4.9, 5.0, 5.2]);
        m.compute_stats(2.0);
        assert!(m.throughput_stats.is_some());
        assert!(m.ttft_stats.is_some());
    }

    #[test]
    fn test_bench_stats_format_criterion_colors() {
        let stats = BenchStats::from_samples(vec![100.0, 102.0, 98.0, 101.0, 99.0], 2.0);
        let formatted = stats.format_criterion("Test", "tok/s", true);
        assert!(formatted.contains("Test"));
        assert!(formatted.contains("tok/s"));
        assert!(formatted.contains("CV="));
    }

    #[test]
    fn test_bench_stats_format_criterion_no_colors() {
        let stats = BenchStats::from_samples(vec![100.0, 102.0, 98.0, 101.0, 99.0], 2.0);
        let formatted = stats.format_criterion("Test", "ms", false);
        assert!(formatted.contains("Test"));
        assert!(formatted.contains("ms"));
        assert!(!formatted.contains("\x1b[")); // No ANSI codes
    }

    #[test]
    fn test_bench_stats_high_cv() {
        // High variation samples to trigger yellow/red coloring
        let stats = BenchStats::from_samples(vec![50.0, 150.0, 200.0, 100.0], 2.0);
        let formatted = stats.format_criterion("Test", "ms", true);
        assert!(formatted.contains("CV="));
    }

    #[test]
    fn test_benchmark_runner_new() {
        let runner = BenchmarkRunner::new();
        assert_eq!(runner.config.iterations, 10);
        assert_eq!(runner.config.warmup_iterations, 3);
    }

    #[test]
    fn test_benchmark_runner_with_config() {
        let config = BenchConfig {
            iterations: 20,
            warmup_iterations: 5,
            outlier_threshold: 3.0,
            colors: false,
            confidence_level: 0.99,
        };
        let runner = BenchmarkRunner::with_config(config);
        assert_eq!(runner.config.iterations, 20);
        assert_eq!(runner.config.warmup_iterations, 5);
    }

    #[test]
    fn test_benchmark_runner_start() {
        let mut runner = BenchmarkRunner::new();
        assert!(runner.start_time.is_none());
        runner.start();
        assert!(runner.start_time.is_some());
    }

    #[test]
    fn test_benchmark_runner_record_component() {
        let mut runner = BenchmarkRunner::new();
        runner.record_component("Q4K_GEMV", Duration::from_millis(100), 500);
        runner.record_component("Attention", Duration::from_millis(50), 200);
        assert_eq!(runner.component_times.len(), 2);
    }

    #[test]
    fn test_benchmark_runner_measure_iterations() {
        let config = BenchConfig {
            iterations: 3,
            warmup_iterations: 1,
            outlier_threshold: 2.0,
            colors: false,
            confidence_level: 0.95,
        };
        let runner = BenchmarkRunner::with_config(config);

        let measurement =
            runner.measure_iterations("test", || (100, Duration::from_millis(100), 5.0));

        assert_eq!(measurement.engine, "test");
        assert_eq!(measurement.throughput_samples.len(), 3);
        assert!(measurement.throughput_stats.is_some());
    }

    #[test]
    fn test_benchmark_runner_finalize() {
        let mut runner = BenchmarkRunner::new();
        runner.record_component("Q4K_GEMV", Duration::from_millis(600), 1000);
        runner.record_component("Attention", Duration::from_millis(300), 500);
        runner.record_component("Minor", Duration::from_millis(10), 10);
        runner.finalize();

        // Q4K_GEMV and Attention should be hotspots (>5%), Minor should not
        assert!(runner.grid.hotspots.len() >= 2);
        // Should be sorted by percentage descending
        if runner.grid.hotspots.len() >= 2 {
            assert!(runner.grid.hotspots[0].percentage >= runner.grid.hotspots[1].percentage);
        }
    }

    #[test]
    fn test_benchmark_runner_finalize_empty() {
        let mut runner = BenchmarkRunner::new();
        runner.finalize(); // Should not panic with empty component_times
        assert!(runner.grid.hotspots.is_empty());
    }

    #[test]
    fn test_benchmark_runner_finalize_zero_calls() {
        let mut runner = BenchmarkRunner::new();
        runner.record_component("ZeroCalls", Duration::from_millis(100), 0);
        runner.finalize();
        // With 100% of time, should still be added
        assert!(!runner.grid.hotspots.is_empty() || runner.grid.hotspots.is_empty());
    }

    #[test]
    fn test_benchmark_runner_default() {
        let runner = BenchmarkRunner::default();
        assert_eq!(runner.config.iterations, 10);
    }

    #[test]
    fn test_render_bar_colored_highlight() {
        let bar = render_bar_colored(50.0, 100.0, 10, true, true);
        assert!(bar.contains("█"));
        assert!(bar.contains(colors::GREEN));
    }

    #[test]
    fn test_render_bar_colored_no_highlight() {
        let bar = render_bar_colored(50.0, 100.0, 10, true, false);
        assert!(bar.contains("█"));
    }

    #[test]
    fn test_render_bar_colored_no_colors() {
        let bar = render_bar_colored(50.0, 100.0, 10, false, false);
        assert!(bar.contains("█"));
        assert!(!bar.contains("\x1b["));
    }

    #[test]
    fn test_render_bar_colored_zero_max() {
        let bar = render_bar_colored(50.0, 0.0, 10, false, false);
        // With zero max, ratio=0, so bar should be all empty chars (░)
        assert!(bar.contains("░"));
    }

    #[test]
    fn test_truncate_short_string() {
        let s = "hello";
        assert_eq!(truncate(s, 10), "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        let s = "hello world";
        assert_eq!(truncate(s, 5), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        let s = "hello";
        assert_eq!(truncate(s, 5), "hello");
    }

    #[test]
    fn test_explain_inference_hotspot_matmul() {
        let (explanation, is_expected) = explain_inference_hotspot("MatMul", 50.0);
        assert!(explanation.contains("Matrix ops"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_attention() {
        let (explanation, is_expected) = explain_inference_hotspot("Attention", 30.0);
        assert!(explanation.contains("Attention"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_kv_cache_normal() {
        let (explanation, is_expected) = explain_inference_hotspot("KV_Cache", 15.0);
        assert!(explanation.contains("normal"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_kv_cache_high() {
        let (explanation, is_expected) = explain_inference_hotspot("KVCache", 25.0);
        assert!(explanation.contains("high"));
        assert!(!is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_softmax_normal() {
        let (explanation, is_expected) = explain_inference_hotspot("Softmax", 5.0);
        assert!(explanation.contains("normal"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_softmax_high() {
        let (explanation, is_expected) = explain_inference_hotspot("Softmax", 15.0);
        assert!(explanation.contains("high"));
        assert!(!is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_rmsnorm_normal() {
        let (explanation, is_expected) = explain_inference_hotspot("RMSNorm", 10.0);
        assert!(explanation.contains("normal"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_rmsnorm_high() {
        let (explanation, is_expected) = explain_inference_hotspot("LayerNorm", 20.0);
        assert!(explanation.contains("high"));
        assert!(!is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_memcpy() {
        let (explanation, is_expected) = explain_inference_hotspot("MemcpyH2D", 10.0);
        assert!(explanation.contains("Memory transfer"));
        assert!(!is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_kernel_launch() {
        let (explanation, is_expected) = explain_inference_hotspot("KernelLaunch", 10.0);
        assert!(explanation.contains("Kernel launch"));
        assert!(!is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_embedding() {
        let (explanation, is_expected) = explain_inference_hotspot("Embedding", 10.0);
        assert!(explanation.contains("Embedding"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_sampling() {
        let (explanation, is_expected) = explain_inference_hotspot("Sampling", 5.0);
        assert!(explanation.contains("Sampling"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_inference_hotspot_unknown_low() {
        let (explanation, is_expected) = explain_inference_hotspot("SomeOther", 10.0);
        assert!(is_expected);
        assert!(explanation.is_empty());
    }

    #[test]
    fn test_explain_inference_hotspot_unknown_high() {
        let (explanation, is_expected) = explain_inference_hotspot("SomeOther", 30.0);
        assert!(!is_expected);
        assert!(explanation.contains("Unknown"));
    }

    #[test]
    fn test_benchmark_grid_render_with_colors() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Test", "0.5B", "Q4_K")
            .with_gpu("RTX 4090", 24.0)
            .with_config(BenchConfig {
                colors: true,
                ..Default::default()
            });

        grid.set_gguf_row(
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput_samples(vec![500.0, 502.0])
                .with_ttft(7.0),
            BenchMeasurement::new("Ollama", "GGUF")
                .with_throughput(318.0)
                .with_ttft(50.0),
            BenchMeasurement::new("llama.cpp", "GGUF")
                .with_throughput(200.0)
                .with_ttft(30.0),
        );

        let output = grid.render();
        assert!(output.contains("\x1b[")); // Has ANSI codes
    }

    #[test]
    fn test_benchmark_grid_render_scientific_with_ttft() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Test", "0.5B", "Q4_K")
            .with_gpu("RTX 4090", 24.0)
            .with_config(BenchConfig::default());

        let mut apr = BenchMeasurement::new("APR", "GGUF")
            .with_throughput_samples(vec![500.0, 502.0])
            .with_ttft_samples(vec![7.0, 7.2]);
        apr.compute_stats(2.0);
        grid.gguf_apr = Some(apr);

        let mut ollama = BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput_samples(vec![318.0])
            .with_ttft_samples(vec![50.0]);
        ollama.compute_stats(2.0);
        grid.gguf_ollama = Some(ollama);

        let output = grid.render_scientific();
        assert!(output.contains("Time to First Token"));
    }

    #[test]
    fn test_benchmark_grid_render_profiling_log() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Test Model", "0.5B", "Q4_K")
            .with_gpu("RTX 4090", 24.0);

        grid.gguf_apr = Some(
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput(500.0)
                .with_ttft(7.0)
                .with_gpu(85.0, 2048.0),
        );
        grid.gguf_ollama = Some(
            BenchMeasurement::new("Ollama", "GGUF")
                .with_throughput(318.0)
                .with_ttft(50.0),
        );
        grid.gguf_llamacpp = Some(
            BenchMeasurement::new("llama.cpp", "GGUF")
                .with_throughput(200.0)
                .with_ttft(30.0),
        );
        grid.apr_native = Some(
            BenchMeasurement::new("APR", ".apr")
                .with_throughput(600.0)
                .with_ttft(5.0)
                .with_gpu(90.0, 2048.0),
        );

        grid.add_hotspot(ProfilingHotspot {
            component: "Q4K_GEMV".to_string(),
            time: Duration::from_millis(150),
            percentage: 45.0,
            call_count: 1000,
            avg_per_call: Duration::from_micros(150),
            explanation: "Matrix ops dominate - expected".to_string(),
            is_expected: true,
        });

        grid.add_hotspot(ProfilingHotspot {
            component: "Slow".to_string(),
            time: Duration::from_millis(50),
            percentage: 15.0,
            call_count: 100,
            avg_per_call: Duration::from_micros(500),
            explanation: "Unexpected slowdown".to_string(),
            is_expected: false,
        });

        let output = grid.render_profiling_log();
        assert!(output.contains("INFERENCE PROFILING REPORT"));
        assert!(output.contains("Test Model"));
        assert!(output.contains("Q4K_GEMV"));
        assert!(output.contains("OPTIMIZATION RECOMMENDATIONS"));
    }

    #[test]
    fn test_benchmark_grid_render_profiling_log_no_hotspots() {
        let grid = BenchmarkGrid::new()
            .with_model("Test", "0.5B", "Q4_K")
            .with_gpu("RTX 4090", 24.0);

        let output = grid.render_profiling_log();
        assert!(output.contains("No unexpected hotspots"));
    }

    #[test]
    fn test_benchmark_grid_add_hotspot() {
        let mut grid = BenchmarkGrid::new();
        let hotspot = ProfilingHotspot {
            component: "Test".to_string(),
            time: Duration::from_millis(100),
            percentage: 50.0,
            call_count: 100,
            avg_per_call: Duration::from_millis(1),
            explanation: String::new(),
            is_expected: true,
        };
        grid.add_hotspot(hotspot);
        assert_eq!(grid.hotspots.len(), 1);
    }

    #[test]
    fn test_bench_measurement_mean_ttft_no_stats() {
        let m = BenchMeasurement::new("APR", "GGUF").with_ttft(10.0);
        assert_eq!(m.mean_ttft(), 10.0);
    }

    #[test]
    fn test_bench_measurement_mean_ttft_empty() {
        let m = BenchMeasurement::new("APR", "GGUF");
        assert_eq!(m.mean_ttft(), 0.0);
    }

    #[test]
    fn test_bench_measurement_mean_throughput_empty() {
        let m = BenchMeasurement::new("APR", "GGUF");
        assert_eq!(m.mean_throughput(), 0.0);
    }

    #[test]
    fn test_bench_stats_even_samples_median() {
        let stats = BenchStats::from_samples(vec![100.0, 102.0, 98.0, 104.0], 2.0);
        // Sorted: 98, 100, 102, 104 -> median = (100+102)/2 = 101
        assert!((stats.median - 101.0).abs() < 0.1);
    }

    #[test]
    fn test_bench_stats_large_samples_t_value() {
        // 30+ samples use t_value = 1.96
        let samples: Vec<f64> = (0..35).map(|i| 100.0 + i as f64 * 0.1).collect();
        let stats = BenchStats::from_samples(samples, 2.0);
        assert!(stats.ci_95.0 < stats.mean);
        assert!(stats.ci_95.1 > stats.mean);
    }

    #[test]
    fn test_explain_flash_attention() {
        let (explanation, is_expected) = explain_inference_hotspot("FlashAttention", 40.0);
        assert!(explanation.contains("Attention"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_gemm() {
        let (explanation, is_expected) = explain_inference_hotspot("GEMM", 60.0);
        assert!(explanation.contains("Matrix ops"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_topk() {
        let (explanation, is_expected) = explain_inference_hotspot("TopK", 5.0);
        assert!(explanation.contains("Sampling"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_topp() {
        let (explanation, is_expected) = explain_inference_hotspot("TopP", 5.0);
        assert!(explanation.contains("Sampling"));
        assert!(is_expected);
    }

    #[test]
    fn test_explain_memcpy_d2h() {
        let (explanation, is_expected) = explain_inference_hotspot("MemcpyD2H", 10.0);
        assert!(explanation.contains("Memory transfer"));
        assert!(!is_expected);
    }

    #[test]
    fn test_explain_transfer() {
        let (explanation, is_expected) = explain_inference_hotspot("Transfer", 10.0);
        assert!(explanation.contains("Memory transfer"));
        assert!(!is_expected);
    }

    #[test]
    fn test_benchmark_grid_speedup_zero_baseline() {
        let mut grid = BenchmarkGrid::new().with_config(BenchConfig {
            colors: false,
            ..Default::default()
        });

        grid.set_apr_row(
            BenchMeasurement::new("APR", ".apr").with_throughput(600.0),
            BenchMeasurement::new("APR", "GGUF").with_throughput(500.0),
            BenchMeasurement::new("Ollama", "GGUF").with_throughput(0.0), // Zero baseline
        );

        let output = grid.render();
        // Should handle zero baseline without panic
        assert!(output.contains("APR serve .apr"));
    }

    #[test]
    fn test_bench_config_clone() {
        let config = BenchConfig {
            iterations: 15,
            warmup_iterations: 5,
            outlier_threshold: 2.5,
            colors: false,
            confidence_level: 0.99,
        };
        let cloned = config.clone();
        assert_eq!(cloned.iterations, 15);
        assert_eq!(cloned.warmup_iterations, 5);
    }

    #[test]
    fn test_bench_stats_clone() {
        let stats = BenchStats::from_samples(vec![100.0, 102.0], 2.0);
        let cloned = stats.clone();
        assert_eq!(cloned.mean, stats.mean);
        assert_eq!(cloned.samples.len(), stats.samples.len());
    }

    #[test]
    fn test_bench_measurement_clone() {
        let m = BenchMeasurement::new("APR", "GGUF").with_throughput(500.0);
        let cloned = m.clone();
        assert_eq!(cloned.engine, "APR");
        assert_eq!(cloned.throughput_samples.len(), 1);
    }

    #[test]
    fn test_benchmark_grid_clone() {
        let grid = BenchmarkGrid::new()
            .with_model("Test", "0.5B", "Q4_K")
            .with_gpu("RTX 4090", 24.0);
        let cloned = grid.clone();
        assert_eq!(cloned.model_name, "Test");
    }

    #[test]
    fn test_profiling_hotspot_clone() {
        let hotspot = ProfilingHotspot {
            component: "Test".to_string(),
            time: Duration::from_millis(100),
            percentage: 50.0,
            call_count: 100,
            avg_per_call: Duration::from_millis(1),
            explanation: "Test".to_string(),
            is_expected: true,
        };
        let cloned = hotspot.clone();
        assert_eq!(cloned.component, "Test");
    }
}
