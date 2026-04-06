//! Benchmark Visualization Module (PAR-040)
//!
//! Creates 2×3 grid visualizations for inference benchmark comparisons
//! and generates profiling logs suitable for chat paste debugging.
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

use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};

// ============================================================================
// Benchmark Result Types
// ============================================================================

/// Single benchmark measurement
#[derive(Debug, Clone)]
pub struct BenchMeasurement {
    /// Engine name (APR, Ollama, llama.cpp)
    pub engine: String,
    /// Format (GGUF, APR)
    pub format: String,
    /// Throughput in tokens/second
    pub tokens_per_sec: f64,
    /// Time to first token in milliseconds
    pub ttft_ms: f64,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Total duration
    pub duration: Duration,
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
            tokens_per_sec: 0.0,
            ttft_ms: 0.0,
            tokens_generated: 0,
            duration: Duration::ZERO,
            gpu_util: None,
            gpu_mem_mb: None,
        }
    }

    /// Set throughput
    #[must_use]
    pub fn with_throughput(mut self, tps: f64) -> Self {
        self.tokens_per_sec = tps;
        self
    }

    /// Set TTFT
    #[must_use]
    pub fn with_ttft(mut self, ttft_ms: f64) -> Self {
        self.ttft_ms = ttft_ms;
        self
    }

    /// Set tokens generated
    #[must_use]
    pub fn with_tokens(mut self, count: usize, duration: Duration) -> Self {
        self.tokens_generated = count;
        self.duration = duration;
        if duration.as_secs_f64() > 0.0 {
            self.tokens_per_sec = count as f64 / duration.as_secs_f64();
        }
        self
    }

    /// Set GPU metrics
    #[must_use]
    pub fn with_gpu(mut self, util: f64, mem_mb: f64) -> Self {
        self.gpu_util = Some(util);
        self.gpu_mem_mb = Some(mem_mb);
        self
    }
}

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
    /// Format as single-line report
    pub fn to_line(&self) -> String {
        let marker = if self.is_expected { "✓" } else { "⚠" };
        format!(
            "{} {:20} {:>6.1}% {:>8.2}ms ({:>6} calls, {:>6.2}µs/call)",
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

/// 2×3 Benchmark comparison grid
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
}

include!("bench_viz_render_profiling_benchmark.rs");
include!("bench_viz_runner.rs");
include!("bench_viz_benchmark_grid.rs");
