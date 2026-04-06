//! Benchmark visualization using trueno-viz.
//!
//! Provides terminal-based visualizations for benchmark results including
//! histograms, sparklines, and performance comparisons.

// Statistical calculations require numeric casts that may lose precision
// on 64-bit systems, but this is acceptable for visualization purposes
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

#[cfg(feature = "visualization")]
use trueno_viz::{
    output::{TerminalEncoder, TerminalMode},
    plots::{BinStrategy, Histogram},
    prelude::Rgba,
};

#[cfg(feature = "visualization")]
use crate::error::{RealizarError, Result};

/// Benchmark result data for visualization.
#[derive(Debug, Clone)]
pub struct BenchmarkData {
    /// Name of the benchmark
    pub name: String,
    /// Latency samples in microseconds
    pub latencies_us: Vec<f64>,
    /// Throughput samples (ops/sec)
    pub throughput: Option<Vec<f64>>,
}

impl BenchmarkData {
    /// Create new benchmark data.
    #[must_use]
    pub fn new(name: impl Into<String>, latencies_us: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            latencies_us,
            throughput: None,
        }
    }

    /// Add throughput data.
    #[must_use]
    pub fn with_throughput(mut self, throughput: Vec<f64>) -> Self {
        self.throughput = Some(throughput);
        self
    }

    /// Calculate statistics.
    #[must_use]
    pub fn stats(&self) -> BenchmarkStats {
        let n = self.latencies_us.len();
        if n == 0 {
            return BenchmarkStats::default();
        }

        let mut sorted = self.latencies_us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = sorted.iter().sum();
        let mean = sum / n as f64;

        let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        BenchmarkStats {
            count: n,
            mean,
            std_dev,
            min: sorted.first().copied().unwrap_or(0.0),
            max: sorted.last().copied().unwrap_or(0.0),
            p50,
            p95,
            p99,
        }
    }
}

/// Calculate percentile from sorted data.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Statistics for benchmark results.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStats {
    /// Number of samples
    pub count: usize,
    /// Mean latency (us)
    pub mean: f64,
    /// Standard deviation (us)
    pub std_dev: f64,
    /// Minimum latency (us)
    pub min: f64,
    /// Maximum latency (us)
    pub max: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

impl std::fmt::Display for BenchmarkStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  samples: {}", self.count)?;
        writeln!(f, "  mean:    {:.2} us", self.mean)?;
        writeln!(f, "  std_dev: {:.2} us", self.std_dev)?;
        writeln!(f, "  min:     {:.2} us", self.min)?;
        writeln!(f, "  p50:     {:.2} us", self.p50)?;
        writeln!(f, "  p95:     {:.2} us", self.p95)?;
        writeln!(f, "  p99:     {:.2} us", self.p99)?;
        write!(f, "  max:     {:.2} us", self.max)
    }
}

/// Render a latency histogram to terminal.
///
/// # Errors
///
/// Returns error if visualization fails or data is empty.
#[cfg(feature = "visualization")]
pub fn render_histogram_terminal(data: &BenchmarkData, width: u32) -> Result<String> {
    if data.latencies_us.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "No latency data to visualize".to_string(),
        });
    }

    // Convert to f32 for trueno-viz
    let latencies: Vec<f32> = data.latencies_us.iter().map(|&x| x as f32).collect();

    let hist = Histogram::new()
        .data(&latencies)
        .bins(BinStrategy::Sturges)
        .color(Rgba::rgb(70, 130, 180)) // Steel blue
        .dimensions(width * 8, 200) // Scale up for better resolution
        .build()
        .map_err(|e| RealizarError::InvalidShape {
            reason: format!("Failed to build histogram: {e}"),
        })?;

    let fb = hist
        .to_framebuffer()
        .map_err(|e| RealizarError::InvalidShape {
            reason: format!("Failed to render histogram: {e}"),
        })?;

    let encoder = TerminalEncoder::new()
        .mode(TerminalMode::Ascii)
        .width(width);

    Ok(encoder.render(&fb))
}

/// Render a latency histogram with ANSI colors.
///
/// # Errors
///
/// Returns error if visualization fails or data is empty.
#[cfg(feature = "visualization")]
pub fn render_histogram_ansi(data: &BenchmarkData, width: u32) -> Result<String> {
    if data.latencies_us.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "No latency data to visualize".to_string(),
        });
    }

    let latencies: Vec<f32> = data.latencies_us.iter().map(|&x| x as f32).collect();

    let hist = Histogram::new()
        .data(&latencies)
        .bins(BinStrategy::Sturges)
        .color(Rgba::rgb(70, 130, 180))
        .dimensions(width * 8, 200)
        .build()
        .map_err(|e| RealizarError::InvalidShape {
            reason: format!("Failed to build histogram: {e}"),
        })?;

    let fb = hist
        .to_framebuffer()
        .map_err(|e| RealizarError::InvalidShape {
            reason: format!("Failed to render histogram: {e}"),
        })?;

    let encoder = TerminalEncoder::new()
        .mode(TerminalMode::UnicodeHalfBlock)
        .width(width);

    Ok(encoder.render(&fb))
}

/// Sparkline bar characters (8 levels).
const SPARKLINE_BARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Render an ASCII sparkline for quick visualization.
///
/// This is a lightweight alternative that doesn't require trueno-viz.
#[must_use]
pub fn render_sparkline(values: &[f64], width: usize) -> String {
    if values.is_empty() {
        return String::new();
    }

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    // Sample values to fit width
    let step = values.len().max(1) / width.max(1);
    let step = step.max(1);

    let mut result = String::with_capacity(width);

    for i in 0..width {
        let idx = (i * step).min(values.len() - 1);
        let value = values[idx];

        let normalized = if range > 0.0 {
            (value - min) / range
        } else {
            0.5
        };

        let bar_idx = (normalized * (SPARKLINE_BARS.len() - 1) as f64).round() as usize;
        result.push(SPARKLINE_BARS[bar_idx.min(SPARKLINE_BARS.len() - 1)]);
    }

    result
}

/// Render a simple ASCII histogram (no dependencies).
///
/// This provides basic visualization without trueno-viz.
#[must_use]
pub fn render_ascii_histogram(values: &[f64], bins: usize, width: usize) -> String {
    use std::fmt::Write;

    if values.is_empty() {
        return String::new();
    }

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    let bin_width = range / bins as f64;

    // Count values in each bin
    let mut counts = vec![0usize; bins];
    for &v in values {
        let bin = if bin_width > 0.0 {
            ((v - min) / bin_width).floor() as usize
        } else {
            0
        };
        let bin = bin.min(bins - 1);
        counts[bin] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);
    let scale = width as f64 / max_count as f64;

    let mut result = String::new();

    for (i, &count) in counts.iter().enumerate() {
        let bar_len = (count as f64 * scale).round() as usize;
        let bin_start = min + i as f64 * bin_width;
        let bin_end = bin_start + bin_width;

        let _ = writeln!(
            result,
            "{:>8.1}-{:<8.1} |{}",
            bin_start,
            bin_end,
            "█".repeat(bar_len)
        );
    }

    result
}

/// Print benchmark results with optional visualization.
pub fn print_benchmark_results(data: &BenchmarkData, use_ansi: bool) {
    let stats = data.stats();

    println!("Benchmark: {}", data.name);
    println!("{stats}");
    println!();

    // Sparkline (always available)
    println!("  trend: {}", render_sparkline(&data.latencies_us, 40));
    println!();

    // ASCII histogram (always available)
    println!("  distribution:");
    let hist = render_ascii_histogram(&data.latencies_us, 10, 40);
    for line in hist.lines() {
        println!("    {line}");
    }

    // Full visualization if available
    #[cfg(feature = "visualization")]
    {
        println!();
        println!("  visual:");
        let rendered = if use_ansi {
            render_histogram_ansi(data, 60)
        } else {
            render_histogram_terminal(data, 60)
        };

        if let Ok(viz) = rendered {
            for line in viz.lines() {
                println!("    {line}");
            }
        }
    }

    let _ = use_ansi; // Suppress unused warning when feature disabled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_data_creation() {
        let data = BenchmarkData::new("test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(data.name, "test");
        assert_eq!(data.latencies_us.len(), 5);
    }

    #[test]
    fn test_benchmark_stats() {
        let data = BenchmarkData::new("test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = data.stats();

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!((stats.min - 1.0).abs() < 0.01);
        assert!((stats.max - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_stats() {
        let data = BenchmarkData::new("empty", vec![]);
        let stats = data.stats();
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_sparkline() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let sparkline = render_sparkline(&values, 9);
        assert_eq!(sparkline.chars().count(), 9);
        assert!(sparkline.contains('▁')); // Min
        assert!(sparkline.contains('█')); // Max
    }

    #[test]
    fn test_sparkline_empty() {
        let sparkline = render_sparkline(&[], 10);
        assert!(sparkline.is_empty());
    }

    #[test]
    fn test_sparkline_constant() {
        let values = vec![5.0; 10];
        let sparkline = render_sparkline(&values, 10);
        // All same value should produce uniform bars
        let unique: std::collections::HashSet<char> = sparkline.chars().collect();
        assert_eq!(unique.len(), 1);
    }

    #[test]
    fn test_ascii_histogram() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let hist = render_ascii_histogram(&values, 10, 40);

        assert!(!hist.is_empty());
        assert!(hist.contains('█'));
        assert_eq!(hist.lines().count(), 10);
    }

    #[test]
    fn test_ascii_histogram_empty() {
        let hist = render_ascii_histogram(&[], 10, 40);
        assert!(hist.is_empty());
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&sorted, 50.0) - 3.0).abs() < 0.01);
        assert!((percentile(&sorted, 100.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_empty() {
        assert!((percentile(&[], 50.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_display() {
        let data = BenchmarkData::new("test", vec![1.0, 2.0, 3.0]);
        let stats = data.stats();
        let display = format!("{stats}");
        assert!(display.contains("mean"));
        assert!(display.contains("p50"));
        assert!(display.contains("p99"));
    }

    #[test]
    fn test_with_throughput() {
        let data = BenchmarkData::new("test", vec![1.0, 2.0]).with_throughput(vec![1000.0, 2000.0]);
        assert!(data.throughput.is_some());
        assert_eq!(data.throughput.expect("test").len(), 2);
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_histogram_terminal() {
        let data = BenchmarkData::new("test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = render_histogram_terminal(&data, 40);
        assert!(result.is_ok());
        assert!(!result.expect("test").is_empty());
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_histogram_empty_error() {
        let data = BenchmarkData::new("empty", vec![]);
        let result = render_histogram_terminal(&data, 40);
        assert!(result.is_err());
    }
}
