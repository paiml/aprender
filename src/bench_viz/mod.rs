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

include!("benchmark_grid.rs");
include!("grid_impl.rs");
include!("benchmark_runner.rs");
