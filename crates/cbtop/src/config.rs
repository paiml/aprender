//! Configuration for cbtop
//!
//! Handles CLI arguments and config file parsing.

use std::path::PathBuf;

/// cbtop configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// Refresh rate in milliseconds
    pub refresh_ms: u64,
    /// GPU device index
    pub device_index: u32,
    /// Compute backend
    pub backend: ComputeBackend,
    /// Load profile
    pub load_profile: LoadProfile,
    /// Workload type
    pub workload: WorkloadType,
    /// Problem size in elements
    pub problem_size: usize,
    /// Thread count for SIMD
    pub threads: usize,
    /// Enable deterministic mode for testing
    pub deterministic: bool,
    /// Show FPS statistics
    pub show_fps: bool,
    /// Config file path
    pub config_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            refresh_ms: 100,
            device_index: 0,
            backend: ComputeBackend::All,
            load_profile: LoadProfile::Idle,
            workload: WorkloadType::Gemm,
            problem_size: 1_048_576,
            threads: num_cpus::get(),
            deterministic: false,
            show_fps: false,
            config_path: None,
        }
    }
}

/// Compute backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ComputeBackend {
    /// CPU SIMD (SSE2/AVX2/AVX-512/NEON)
    Simd,
    /// Cross-platform GPU (Vulkan/Metal/DX12)
    Wgpu,
    /// Native NVIDIA CUDA
    Cuda,
    /// All backends simultaneously
    #[default]
    All,
}

/// Load profile intensity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadProfile {
    /// No load
    #[default]
    Idle,
    /// 25% intensity
    Light,
    /// 50% intensity
    Medium,
    /// 75% intensity
    Heavy,
    /// 100% intensity
    Stress,
}

impl LoadProfile {
    /// Convert to intensity value (0.0 - 1.0)
    pub fn intensity(&self) -> f64 {
        match self {
            Self::Idle => 0.0,
            Self::Light => 0.25,
            Self::Medium => 0.50,
            Self::Heavy => 0.75,
            Self::Stress => 1.0,
        }
    }
}

/// Workload type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum WorkloadType {
    /// Matrix multiplication
    #[default]
    Gemm,
    /// 2D convolution
    Conv2d,
    /// Transformer attention
    Attention,
    /// Memory bandwidth stress
    Bandwidth,
    /// Element-wise operations
    Elementwise,
    /// Reduction operations
    Reduction,
    /// Cycle through all
    All,
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    }
}
