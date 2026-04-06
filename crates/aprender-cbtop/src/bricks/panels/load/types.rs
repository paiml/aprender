//! Types for the load control panel.

/// Compute backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    #[default]
    Auto,
    CpuScalar,
    CpuSimd,
    GpuCuda,
    GpuWgpu,
}

impl ComputeBackend {
    /// Display name for the backend
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::CpuScalar => "CPU (Scalar)",
            Self::CpuSimd => "CPU (SIMD)",
            Self::GpuCuda => "GPU (CUDA)",
            Self::GpuWgpu => "GPU (wgpu)",
        }
    }

    /// All available backends
    pub const ALL: [ComputeBackend; 5] =
        [Self::Auto, Self::CpuScalar, Self::CpuSimd, Self::GpuCuda, Self::GpuWgpu];
}

/// Workload type for load testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkloadType {
    #[default]
    Gemm,
    Softmax,
    LayerNorm,
    Attention,
    Lz4Compress,
    Mixed,
}

impl WorkloadType {
    /// Display name for the workload
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemm => "GEMM (Matrix Multiply)",
            Self::Softmax => "Softmax",
            Self::LayerNorm => "Layer Normalization",
            Self::Attention => "Attention",
            Self::Lz4Compress => "LZ4 Compression",
            Self::Mixed => "Mixed Workload",
        }
    }

    /// Short name for compact display
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Gemm => "GEMM",
            Self::Softmax => "Softmax",
            Self::LayerNorm => "LayerNorm",
            Self::Attention => "Attention",
            Self::Lz4Compress => "LZ4",
            Self::Mixed => "Mixed",
        }
    }

    /// All available workloads
    pub const ALL: [WorkloadType; 6] = [
        Self::Gemm,
        Self::Softmax,
        Self::LayerNorm,
        Self::Attention,
        Self::Lz4Compress,
        Self::Mixed,
    ];
}

/// Load test run statistics
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Iterations completed
    pub iterations: u64,
    /// Total elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Operations per second
    pub ops_per_sec: f64,
    /// Current throughput in GB/s
    pub throughput_gbs: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// P99 latency in microseconds
    pub p99_latency_us: f64,
}
