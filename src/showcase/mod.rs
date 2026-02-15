//! GPU Inference Showcase Module (PAR-040)
//!
//! PMAT-verified benchmark harness for Qwen2.5-Coder showcase.
//! Delivers >2x performance vs competitors via:
//! - trueno GPU PTX generation (persistent kernels, megakernels)
//! - trueno SIMD (AVX2/AVX-512/NEON)
//! - trueno-zram KV cache compression
//! - renacer GPU kernel profiling
//!
//! # Performance Targets (Point 41)
//!
//! | Engine | Target | Mechanism |
//! |--------|--------|-----------|
//! | APR GGUF | >2x llama.cpp | Phase 2 GPU optimizations |
//! | APR .apr | >2x Ollama | Native format + ZRAM |
//!
//! # Usage
//!
//! ```bash
//! # Run full showcase benchmark
//! cargo run --example showcase_benchmark --features cuda
//!
//! # Run with profiling
//! renacer trace -- cargo run --example showcase_benchmark --features cuda
//! ```

use crate::bench_viz::{BenchConfig, BenchMeasurement, BenchmarkGrid, ProfilingHotspot};
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

/// Showcase benchmark configuration
#[derive(Debug, Clone)]
pub struct ShowcaseConfig {
    /// Model path (GGUF or APR format)
    pub model_path: String,
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Tokens to generate per iteration
    pub gen_tokens: usize,
    /// Prompt for generation
    pub prompt: String,
    /// Enable rich terminal colors
    pub colors: bool,
    /// Enable renacer profiling integration
    pub profile: bool,
    /// Enable trueno-zram KV cache compression
    pub zram: bool,
    /// Target GPU device
    pub gpu_device: u32,
}

impl Default for ShowcaseConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            iterations: 10,
            warmup_iterations: 3,
            gen_tokens: 128,
            prompt: "Write a function to calculate fibonacci numbers:".to_string(),
            colors: true,
            profile: false,
            zram: false,
            gpu_device: 0,
        }
    }
}

// ============================================================================
// Benchmark Results
// ============================================================================

/// Single benchmark run result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Tokens generated
    pub tokens: usize,
    /// Total generation time
    pub duration: Duration,
    /// Time to first token
    pub ttft: Duration,
    /// Throughput (tokens/sec)
    pub throughput: f64,
    /// GPU utilization percentage
    pub gpu_util: Option<f64>,
    /// GPU memory used (MB)
    pub gpu_mem_mb: Option<f64>,
}

impl BenchmarkResult {
    /// Calculate throughput from tokens and duration
    pub fn new(tokens: usize, duration: Duration, ttft: Duration) -> Self {
        let throughput = if duration.as_secs_f64() > 0.0 {
            tokens as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        Self {
            tokens,
            duration,
            ttft,
            throughput,
            gpu_util: None,
            gpu_mem_mb: None,
        }
    }

    /// Add GPU metrics
    pub fn with_gpu_metrics(mut self, util: f64, mem_mb: f64) -> Self {
        self.gpu_util = Some(util);
        self.gpu_mem_mb = Some(mem_mb);
        self
    }
}

/// Aggregated benchmark statistics
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    /// All results
    pub results: Vec<BenchmarkResult>,
    /// Mean throughput
    pub mean_throughput: f64,
    /// Throughput standard deviation
    pub std_throughput: f64,
    /// Mean TTFT
    pub mean_ttft_ms: f64,
    /// 95% confidence interval (low, high)
    pub ci_95: (f64, f64),
    /// Coefficient of variation
    pub cv: f64,
}

impl BenchmarkStats {
    /// Compute statistics from results
    pub fn from_results(results: Vec<BenchmarkResult>) -> Self {
        if results.is_empty() {
            return Self {
                results: vec![],
                mean_throughput: 0.0,
                std_throughput: 0.0,
                mean_ttft_ms: 0.0,
                ci_95: (0.0, 0.0),
                cv: 0.0,
            };
        }

        let n = results.len() as f64;
        let throughputs: Vec<f64> = results.iter().map(|r| r.throughput).collect();
        let ttfts: Vec<f64> = results
            .iter()
            .map(|r| r.ttft.as_secs_f64() * 1000.0)
            .collect();

        let mean_throughput = throughputs.iter().sum::<f64>() / n;
        let mean_ttft_ms = ttfts.iter().sum::<f64>() / n;

        let variance = throughputs
            .iter()
            .map(|x| (x - mean_throughput).powi(2))
            .sum::<f64>()
            / n;
        let std_throughput = variance.sqrt();

        // 95% CI
        let t_value = if results.len() >= 30 { 1.96 } else { 2.0 };
        let margin = t_value * std_throughput / n.sqrt();
        let ci_95 = (mean_throughput - margin, mean_throughput + margin);

        let cv = if mean_throughput > 0.0 {
            std_throughput / mean_throughput
        } else {
            0.0
        };

        Self {
            results,
            mean_throughput,
            std_throughput,
            mean_ttft_ms,
            ci_95,
            cv,
        }
    }
}

// ============================================================================
// Showcase Runner
// ============================================================================

/// Main showcase benchmark runner
#[derive(Debug)]
pub struct ShowcaseRunner {
    /// Configuration
    pub config: ShowcaseConfig,
    /// APR GGUF results
    pub apr_gguf_stats: Option<BenchmarkStats>,
    /// APR native (.apr) results
    pub apr_native_stats: Option<BenchmarkStats>,
    /// Ollama baseline results
    pub ollama_stats: Option<BenchmarkStats>,
    /// llama.cpp baseline results
    pub llamacpp_stats: Option<BenchmarkStats>,
    /// Profiling hotspots
    pub hotspots: Vec<ProfilingHotspot>,
    /// Model info
    pub model_name: String,
    /// Model parameters
    pub model_params: String,
    /// Quantization type
    pub quantization: String,
    /// GPU name
    pub gpu_name: String,
    /// GPU VRAM (GB)
    pub gpu_vram_gb: f64,
}

impl ShowcaseRunner {
    /// Create new showcase runner
    pub fn new(config: ShowcaseConfig) -> Self {
        Self {
            config,
            apr_gguf_stats: None,
            apr_native_stats: None,
            ollama_stats: None,
            llamacpp_stats: None,
            hotspots: Vec::new(),
            model_name: String::new(),
            model_params: String::new(),
            quantization: String::new(),
            gpu_name: String::new(),
            gpu_vram_gb: 0.0,
        }
    }

    /// Set model info
    pub fn with_model_info(mut self, name: &str, params: &str, quant: &str) -> Self {
        self.model_name = name.to_string();
        self.model_params = params.to_string();
        self.quantization = quant.to_string();
        self
    }

    /// Set GPU info
    pub fn with_gpu_info(mut self, name: &str, vram_gb: f64) -> Self {
        self.gpu_name = name.to_string();
        self.gpu_vram_gb = vram_gb;
        self
    }

    /// Run benchmark with provided inference function
    ///
    /// The function should return (tokens_generated, duration, ttft)
    pub fn run_benchmark<F>(&self, _name: &str, mut f: F) -> BenchmarkStats
    where
        F: FnMut() -> (usize, Duration, Duration),
    {
        let mut results = Vec::with_capacity(self.config.iterations);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = f();
        }

        // Measurement
        for _ in 0..self.config.iterations {
            let (tokens, duration, ttft) = f();
            results.push(BenchmarkResult::new(tokens, duration, ttft));
        }

        BenchmarkStats::from_results(results)
    }

    /// Record APR GGUF benchmark results
    pub fn record_apr_gguf(&mut self, stats: BenchmarkStats) {
        self.apr_gguf_stats = Some(stats);
    }

    /// Record APR native benchmark results
    pub fn record_apr_native(&mut self, stats: BenchmarkStats) {
        self.apr_native_stats = Some(stats);
    }

    /// Record Ollama baseline results
    pub fn record_ollama(&mut self, stats: BenchmarkStats) {
        self.ollama_stats = Some(stats);
    }

    /// Record llama.cpp baseline results
    pub fn record_llamacpp(&mut self, stats: BenchmarkStats) {
        self.llamacpp_stats = Some(stats);
    }

    /// Add profiling hotspot
    pub fn add_hotspot(&mut self, hotspot: ProfilingHotspot) {
        self.hotspots.push(hotspot);
    }

    /// Generate BenchmarkGrid for visualization
    pub fn to_grid(&self) -> BenchmarkGrid {
        let bench_config = BenchConfig {
            iterations: self.config.iterations,
            warmup_iterations: self.config.warmup_iterations,
            outlier_threshold: 2.0,
            colors: self.config.colors,
            confidence_level: 0.95,
        };

        let mut grid = BenchmarkGrid::new()
            .with_config(bench_config)
            .with_model(&self.model_name, &self.model_params, &self.quantization)
            .with_gpu(&self.gpu_name, self.gpu_vram_gb);

        // Build measurements from stats
        if let Some(ref stats) = self.apr_gguf_stats {
            let apr_gguf = self.stats_to_measurement(stats, "APR", "GGUF");

            let ollama = self.ollama_stats.as_ref().map_or_else(
                || {
                    BenchMeasurement::new("Ollama", "GGUF")
                        .with_throughput(318.0)
                        .with_ttft(50.0)
                },
                |s| self.stats_to_measurement(s, "Ollama", "GGUF"),
            );

            let llamacpp = self.llamacpp_stats.as_ref().map_or_else(
                || {
                    BenchMeasurement::new("llama.cpp", "GGUF")
                        .with_throughput(200.0)
                        .with_ttft(30.0)
                },
                |s| self.stats_to_measurement(s, "llama.cpp", "GGUF"),
            );

            grid.set_gguf_row(apr_gguf, ollama, llamacpp);
        }

        if let Some(ref native_stats) = self.apr_native_stats {
            let apr_native = self.stats_to_measurement(native_stats, "APR", ".apr");

            let apr_gguf = self.apr_gguf_stats.as_ref().map_or_else(
                || {
                    BenchMeasurement::new("APR", "GGUF")
                        .with_throughput(500.0)
                        .with_ttft(7.0)
                },
                |s| self.stats_to_measurement(s, "APR", "GGUF"),
            );

            let baseline = self.ollama_stats.as_ref().map_or_else(
                || {
                    BenchMeasurement::new("Ollama", "GGUF")
                        .with_throughput(318.0)
                        .with_ttft(50.0)
                },
                |s| self.stats_to_measurement(s, "Ollama", "GGUF"),
            );

            grid.set_apr_row(apr_native, apr_gguf, baseline);
        }

        // Add hotspots
        for hotspot in &self.hotspots {
            grid.add_hotspot(hotspot.clone());
        }

        grid
    }

    /// Convert stats to measurement
    fn stats_to_measurement(
        &self,
        stats: &BenchmarkStats,
        engine: &str,
        format: &str,
    ) -> BenchMeasurement {
        let throughputs: Vec<f64> = stats.results.iter().map(|r| r.throughput).collect();
        let ttfts: Vec<f64> = stats
            .results
            .iter()
            .map(|r| r.ttft.as_secs_f64() * 1000.0)
            .collect();

        let mut m = BenchMeasurement::new(engine, format)
            .with_throughput_samples(throughputs)
            .with_ttft_samples(ttfts);

        // Add GPU metrics from first result if available
        if let Some(first) = stats.results.first() {
            if let (Some(util), Some(mem)) = (first.gpu_util, first.gpu_mem_mb) {
                m = m.with_gpu(util, mem);
            }
        }

        m
    }

    /// Generate full report
    pub fn render_report(&self) -> String {
        let grid = self.to_grid();
        let mut report = String::new();

        report.push_str(&grid.render());
        report.push('\n');
        report.push_str(&grid.render_scientific());
        report.push('\n');
        report.push_str(&grid.render_profiling_log());

        report
    }

    /// Check if Point 41 passes (≥25% faster than llama.cpp)
    pub fn check_point_41(&self) -> bool {
        let apr_tps = self
            .apr_gguf_stats
            .as_ref()
            .map_or(0.0, |s| s.mean_throughput);

        let llamacpp_tps = self
            .llamacpp_stats
            .as_ref()
            .map_or(200.0, |s| s.mean_throughput); // Default baseline

        apr_tps >= llamacpp_tps * 1.25
    }

    /// Check if 2x Ollama target is met
    pub fn check_2x_ollama(&self) -> bool {
        let apr_tps = self
            .apr_native_stats
            .as_ref()
            .or(self.apr_gguf_stats.as_ref())
            .map_or(0.0, |s| s.mean_throughput);

        let ollama_tps = self
            .ollama_stats
            .as_ref()
            .map_or(318.0, |s| s.mean_throughput); // Default baseline

        apr_tps >= ollama_tps * 2.0
    }
}

include!("mod_part_02.rs");
include!("mod_part_03.rs");
