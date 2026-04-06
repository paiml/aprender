//! Core types for cross-backend regression detection.

/// Backend identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Scalar (no SIMD)
    Scalar,
    /// SSE2 (128-bit)
    Sse2,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// NEON (ARM)
    Neon,
    /// CUDA (NVIDIA GPU)
    Cuda,
    /// Metal (Apple GPU)
    Metal,
    /// Vulkan (Cross-platform GPU)
    Vulkan,
    /// WebGPU
    WebGpu,
}

impl Backend {
    /// Get backend name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::Sse2 => "SSE2",
            Self::Avx2 => "AVX2",
            Self::Avx512 => "AVX-512",
            Self::Neon => "NEON",
            Self::Cuda => "CUDA",
            Self::Metal => "Metal",
            Self::Vulkan => "Vulkan",
            Self::WebGpu => "WebGPU",
        }
    }

    /// Is this a GPU backend?
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::Metal | Self::Vulkan | Self::WebGpu)
    }

    /// Is this a SIMD backend?
    pub fn is_simd(&self) -> bool {
        matches!(self, Self::Sse2 | Self::Avx2 | Self::Avx512 | Self::Neon)
    }

    /// Get expected speedup over scalar (theoretical)
    pub fn theoretical_speedup(&self) -> f64 {
        match self {
            Self::Scalar => 1.0,
            Self::Sse2 => 4.0,    // 128-bit / 32-bit = 4
            Self::Avx2 => 8.0,    // 256-bit / 32-bit = 8
            Self::Avx512 => 16.0, // 512-bit / 32-bit = 16
            Self::Neon => 4.0,    // 128-bit / 32-bit = 4
            Self::Cuda => 100.0,  // Variable, placeholder
            Self::Metal => 50.0,  // Variable, placeholder
            Self::Vulkan => 50.0, // Variable, placeholder
            Self::WebGpu => 30.0, // Variable, placeholder
        }
    }
}

/// Workload type for benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    /// Matrix multiplication
    Gemm,
    /// 2D convolution
    Conv2d,
    /// Element-wise operations
    Elementwise,
    /// Reduction (sum, mean)
    Reduction,
    /// Attention mechanism
    Attention,
    /// Memory bandwidth test
    Bandwidth,
}

impl WorkloadType {
    /// Get workload name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemm => "GEMM",
            Self::Conv2d => "Conv2D",
            Self::Elementwise => "Elementwise",
            Self::Reduction => "Reduction",
            Self::Attention => "Attention",
            Self::Bandwidth => "Bandwidth",
        }
    }
}

/// Performance measurement for a single backend/size combination
#[derive(Debug, Clone)]
pub struct BackendMeasurement {
    /// Backend used
    pub backend: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Problem size (elements)
    pub size: usize,
    /// Latency in microseconds
    pub latency_us: f64,
    /// Throughput (ops/sec or elements/sec)
    pub throughput: f64,
    /// Efficiency (% of theoretical peak)
    pub efficiency_percent: f64,
    /// GPU transfer time (if applicable)
    pub transfer_time_us: Option<f64>,
    /// Compute time (excluding transfer)
    pub compute_time_us: Option<f64>,
}

impl BackendMeasurement {
    /// Create new measurement
    pub fn new(
        backend: Backend,
        workload: WorkloadType,
        size: usize,
        latency_us: f64,
        throughput: f64,
    ) -> Self {
        Self {
            backend,
            workload,
            size,
            latency_us,
            throughput,
            efficiency_percent: 0.0,
            transfer_time_us: None,
            compute_time_us: None,
        }
    }

    /// Set efficiency
    pub fn with_efficiency(mut self, efficiency: f64) -> Self {
        self.efficiency_percent = efficiency;
        self
    }

    /// Set GPU timing breakdown
    pub fn with_gpu_timing(mut self, transfer_us: f64, compute_us: f64) -> Self {
        self.transfer_time_us = Some(transfer_us);
        self.compute_time_us = Some(compute_us);
        self
    }

    /// Get transfer overhead ratio (transfer / total)
    pub fn transfer_overhead(&self) -> Option<f64> {
        match (self.transfer_time_us, self.compute_time_us) {
            (Some(t), Some(c)) if t + c > 0.0 => Some(t / (t + c)),
            _ => None,
        }
    }
}

/// Comparison result between two backends
#[derive(Debug, Clone)]
pub struct BackendComparison {
    /// Baseline backend
    pub baseline: Backend,
    /// Comparison backend
    pub comparison: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Problem size
    pub size: usize,
    /// Efficiency ratio (comparison / baseline)
    pub efficiency_ratio: f64,
    /// Speedup (baseline_latency / comparison_latency)
    pub speedup: f64,
    /// Is this a regression? (efficiency_ratio < 1.0 - threshold)
    pub is_regression: bool,
    /// Regression threshold used
    pub threshold: f64,
}

impl BackendComparison {
    /// Get summary message
    pub fn summary(&self) -> String {
        if self.is_regression {
            format!(
                "REGRESSION: {} -> {} on {} size={}: {:.1}% slower",
                self.baseline.name(),
                self.comparison.name(),
                self.workload.name(),
                self.size,
                (1.0 - self.speedup) * 100.0
            )
        } else {
            format!(
                "OK: {} -> {} on {} size={}: {:.1}x speedup",
                self.baseline.name(),
                self.comparison.name(),
                self.workload.name(),
                self.size,
                self.speedup
            )
        }
    }
}

/// Size cliff detection result
#[derive(Debug, Clone)]
pub struct SizeCliff {
    /// Backend where cliff occurs
    pub backend: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Size before cliff
    pub size_before: usize,
    /// Size after cliff
    pub size_after: usize,
    /// Efficiency before cliff
    pub efficiency_before: f64,
    /// Efficiency after cliff
    pub efficiency_after: f64,
    /// Drop percentage
    pub drop_percent: f64,
}

impl SizeCliff {
    /// Get summary message
    pub fn summary(&self) -> String {
        format!(
            "CLIFF: {} {} at {}\u{2192}{}: {:.1}% efficiency drop",
            self.backend.name(),
            self.workload.name(),
            self.size_before,
            self.size_after,
            self.drop_percent
        )
    }
}

/// Backend recommendation
#[derive(Debug, Clone)]
pub struct BackendRecommendation {
    /// Recommended backend
    pub backend: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Problem size
    pub size: usize,
    /// Expected efficiency
    pub expected_efficiency: f64,
    /// Confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Reason for recommendation
    pub reason: String,
}
