//! Execution strategy (Geometry equivalent in Grammar of Graphics).

use super::resources::ResourceMapping;
use super::workload::WorkloadSpec;

/// SIMD width specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdWidth {
    /// Auto-detect best available
    Auto,
    /// SSE2 (128-bit)
    Sse2,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// ARM NEON (128-bit)
    Neon,
    /// WASM SIMD128
    Wasm,
}

/// GPU device specification
#[derive(Debug, Clone, PartialEq)]
pub enum GpuDevice {
    /// Auto-select best available
    Auto,
    /// Specific device by ID
    Id(u32),
    /// CUDA device
    Cuda(u32),
    /// wgpu device
    Wgpu(u32),
}

/// Kernel specification for GPU
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSpec {
    /// Kernel name
    pub name: String,
    /// Block size (threads per block)
    pub block_size: (u32, u32, u32),
    /// Grid size (number of blocks)
    pub grid_size: Option<(u32, u32, u32)>,
    /// Shared memory per block
    pub shared_mem: usize,
}

/// Execution strategy (analogous to Geometry)
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStrategy {
    /// Sequential execution (baseline)
    Sequential,
    /// SIMD vectorization
    Simd { width: SimdWidth },
    /// Multi-threaded parallel
    Parallel { threads: usize, chunk_size: usize },
    /// GPU acceleration
    Gpu {
        device: GpuDevice,
        kernel: Option<KernelSpec>,
    },
    /// Distributed across nodes
    Distributed { nodes: Vec<String> },
    /// Hybrid CPU+GPU
    Hybrid { cpu_fraction: f64 },
}

impl ExecutionStrategy {
    /// Create SIMD strategy with auto width
    pub fn simd_auto() -> Self {
        ExecutionStrategy::Simd {
            width: SimdWidth::Auto,
        }
    }

    /// Create SIMD strategy with specific width
    pub fn simd(width: SimdWidth) -> Self {
        ExecutionStrategy::Simd { width }
    }

    /// Create parallel strategy
    pub fn parallel(threads: usize) -> Self {
        ExecutionStrategy::Parallel {
            threads,
            chunk_size: 1024,
        }
    }

    /// Create GPU strategy with auto device
    pub fn gpu_auto() -> Self {
        ExecutionStrategy::Gpu {
            device: GpuDevice::Auto,
            kernel: None,
        }
    }

    /// Create GPU strategy with specific device
    pub fn gpu(device: GpuDevice) -> Self {
        ExecutionStrategy::Gpu {
            device,
            kernel: None,
        }
    }
}

/// Strategy layer (analogous to ggplot2 Layer)
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyLayer {
    /// Execution strategy
    pub strategy: ExecutionStrategy,
    /// Layer-specific workload override
    pub workload: Option<WorkloadSpec>,
    /// Layer-specific resource mapping
    pub resources: ResourceMapping,
    /// Layer priority (higher = try first)
    pub priority: i32,
}

impl StrategyLayer {
    /// Create new strategy layer
    pub fn new(strategy: ExecutionStrategy) -> Self {
        Self {
            strategy,
            workload: None,
            resources: ResourceMapping::default(),
            priority: 0,
        }
    }

    /// Set layer priority
    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}
