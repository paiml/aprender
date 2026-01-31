//! Type definitions for the showcase demo

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Showcase configuration
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used by CLI but not all consumed yet
pub struct ShowcaseConfig {
    /// Model tier (whisper-style: tiny, small, medium, large)
    pub tier: ModelTier,
    /// Model to use (derived from tier, can be overridden)
    pub model: String,
    /// Quantization level
    pub quant: String,
    /// Output directory for models
    pub model_dir: PathBuf,
    /// Run all steps automatically
    pub auto_verify: bool,
    /// Specific step to run
    pub step: Option<ShowcaseStep>,
    /// Baselines to compare against
    pub baselines: Vec<Baseline>,
    /// Enable ZRAM compression
    pub zram: bool,
    /// Number of benchmark runs (spec: minimum 30)
    pub bench_runs: usize,
    /// Export format for benchmark results (Point 85)
    pub export_format: ExportFormat,
    /// Export path for benchmark results (default: model_dir/benchmark-results.{json,csv})
    pub export_path: Option<PathBuf>,
    /// Force GPU acceleration
    pub gpu: bool,
    /// Verbose output
    pub verbose: bool,
    /// Quiet mode (errors only)
    pub quiet: bool,
}

impl Default for ShowcaseConfig {
    fn default() -> Self {
        let tier = ModelTier::Small; // Default to 1.5B for fast iteration
        Self {
            tier,
            model: tier.model_path().to_string(),
            quant: "Q4_K_M".to_string(),
            model_dir: PathBuf::from("./models"),
            auto_verify: false,
            step: None,
            baselines: vec![Baseline::LlamaCpp, Baseline::Ollama],
            zram: true,
            bench_runs: 30,
            export_format: ExportFormat::None,
            export_path: None,
            gpu: false,
            verbose: false,
            quiet: false,
        }
    }
}

impl ShowcaseConfig {
    /// Create config for a specific tier
    #[must_use]
    #[allow(dead_code)] // Convenience constructor for future use
    pub fn with_tier(tier: ModelTier) -> Self {
        Self {
            tier,
            model: tier.model_path().to_string(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShowcaseStep {
    Import,
    GgufInference,
    Convert,
    AprInference,
    Benchmark,
    Chat,
    Visualize,
    ZramDemo,
    CudaDemo,
    /// Brick architecture demo - per-layer timing with bottleneck detection
    BrickDemo,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Baseline {
    LlamaCpp,
    Ollama,
}

/// Whisper-style model tiers for Qwen2.5-Coder
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelTier {
    /// 0.5B - Quick testing (~400MB)
    Tiny,
    /// 1.5B - Development (~1.1GB)
    #[default]
    Small,
    /// 7B - Production (~4.5GB)
    Medium,
    /// 32B - Showcase demo (~19GB)
    Large,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // Idiomatic &self for enum methods
impl ModelTier {
    /// Get HuggingFace model path for this tier
    #[must_use]
    pub fn model_path(&self) -> &'static str {
        match self {
            Self::Tiny => "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            Self::Small => "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            Self::Medium => "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            Self::Large => "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        }
    }

    /// Get GGUF filename for this tier (Q4_K_M quantization)
    #[must_use]
    pub fn gguf_filename(&self) -> &'static str {
        match self {
            Self::Tiny => "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
            Self::Small => "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
            Self::Medium => "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            Self::Large => "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        }
    }

    /// Get approximate model size in GB
    #[must_use]
    pub fn size_gb(&self) -> f32 {
        match self {
            Self::Tiny => 0.4,
            Self::Small => 1.1,
            Self::Medium => 4.5,
            Self::Large => 19.0,
        }
    }

    /// Get parameter count description
    #[must_use]
    pub fn params(&self) -> &'static str {
        match self {
            Self::Tiny => "0.5B",
            Self::Small => "1.5B",
            Self::Medium => "7B",
            Self::Large => "32B",
        }
    }
}

/// Export format for benchmark results (Point 85)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)] // Variants are constructed by CLI parser
pub enum ExportFormat {
    #[default]
    None,
    Json,
    Csv,
}

/// Benchmark results for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub apr_tps: f64,
    pub llama_cpp_tps: Option<f64>,
    pub ollama_tps: Option<f64>,
    pub apr_ttft_ms: f64,
    pub llama_cpp_ttft_ms: Option<f64>,
    pub ollama_ttft_ms: Option<f64>,
    pub speedup_vs_llama: Option<f64>,
    pub speedup_vs_ollama: Option<f64>,
    /// Standard deviation of APR measurements
    pub apr_tps_stddev: f64,
    /// Number of runs performed
    pub runs: usize,
}

/// Single benchmark measurement
#[derive(Debug, Clone)]
pub struct BenchMeasurement {
    pub tokens_generated: usize,
    pub duration: Duration,
    pub ttft: Duration,
}

impl BenchMeasurement {
    pub fn tokens_per_second(&self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            self.tokens_generated as f64 / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct ShowcaseResults {
    pub(super) import: bool,
    pub(super) gguf_inference: bool,
    pub(super) convert: bool,
    pub(super) apr_inference: bool,
    pub(super) benchmark: Option<BenchmarkComparison>,
    pub(super) visualize: bool,
    pub(super) chat: bool,
    pub(super) zram_demo: Option<ZramDemoResult>,
    pub(super) cuda_demo: Option<CudaDemoResult>,
    pub(super) brick_demo: Option<BrickDemoResult>,
}

/// ComputeBrick demo result - per-layer timing with bottleneck detection
#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // Fields used in tests and for future summary output
pub struct BrickDemoResult {
    /// Total layers measured
    pub layers_measured: usize,
    /// Per-layer timing in µs
    pub layer_timings_us: Vec<f64>,
    /// Identified bottleneck (brick name, time in µs)
    pub bottleneck: Option<(String, f64)>,
    /// Total inference time in µs
    pub total_us: f64,
    /// Tokens per second achieved
    pub tokens_per_sec: f64,
    /// All assertions passed
    pub assertions_passed: bool,
}

/// ZRAM compression demo result
#[derive(Debug, Clone)]
pub struct ZramDemoResult {
    /// LZ4 compression ratio achieved
    pub lz4_ratio: f64,
    /// ZSTD compression ratio achieved
    pub zstd_ratio: f64,
    /// Zero-page throughput in GB/s
    pub zero_page_gbps: f64,
    /// LZ4 throughput in GB/s
    pub lz4_gbps: f64,
    /// SIMD backend used
    pub simd_backend: String,
    /// Context extension factor (Point 80)
    pub context_extension: f64,
}

/// CUDA GPU demo result (Point 78, Sections 5.2/5.3)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used in tests and for future summary output
pub struct CudaDemoResult {
    /// Number of CUDA devices detected
    pub device_count: usize,
    /// Primary GPU name (e.g., "NVIDIA GeForce RTX 4090")
    pub device_name: String,
    /// Total VRAM in GB
    pub total_vram_gb: f64,
    /// Free VRAM in GB
    pub free_vram_gb: f64,
    /// CUDA available flag
    pub cuda_available: bool,
    /// CUDA Graph capture available (Section 5.2 - P0)
    pub graph_capture_available: bool,
    /// CUDA Graph speedup factor vs eager execution
    pub graph_speedup: f64,
    /// Coalesced DP4A kernel available (Section 5.3 - P0)
    pub dp4a_available: bool,
    /// DP4A arithmetic intensity (flops/byte)
    pub dp4a_arithmetic_intensity: f64,
}
