//! Backend Selection and Cost Model
//!
//! Automatic selection between CPU (SIMD), CUDA (PTX), and wgpu (WGSL) backends
//! based on the 5× PCIe rule and roofline analysis.
//!
//! # Philosophy
//!
//! Uses Gregg & Hazelwood (2011) "5× PCIe rule": GPU worthwhile when
//! compute time exceeds 5× data transfer time.
//!
//! # References
//!
//! - Gregg, C., & Hazelwood, K. (2011). Where is the Data? Why You Cannot
//!   Debate CPU vs. GPU Performance Without the Answer. IEEE ISPASS.
//! - Volkov, V. (2010). Better Performance at Lower Occupancy.

#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

use super::profiler::BlisProfiler;
use super::{gemm_blis, TruenoError};

///
/// Maps to different ISA targets:
/// - Cpu: x86 asm (AVX2/AVX-512), ARM asm (NEON)
/// - Gpu: PTX (CUDA), wgpu compute shaders
/// - Wgpu: WGSL for cross-platform GPU (Vulkan/Metal/DX12/WebGPU)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBackend {
    /// CPU SIMD backend (AVX2, AVX-512, NEON, SSE2)
    Cpu,
    /// NVIDIA GPU backend (PTX)
    Gpu,
    /// Cross-platform GPU backend (wgpu/WGSL)
    Wgpu,
    /// Scalar fallback (no SIMD)
    Scalar,
}

/// ComputeBrick hierarchy level
///
/// Maps BLIS loop structure to brick abstraction:
/// - Nano: Microkernel (MR×NR×K) - register file
/// - Micro: Midi loop (MC×NC×KC) - L1/L2 cache
/// - Meso: Macro loop (full M×N×K) - L3/DRAM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrickLevel {
    /// Register-level compute (MR×NR tile)
    Nano,
    /// Cache-level compute (MC×NC block)
    Micro,
    /// Memory-level compute (full matrix)
    Meso,
}

/// Cost model for backend selection
///
/// Based on Gregg & Hazelwood (2011): GPU worthwhile when compute > 5× transfer
#[derive(Debug, Clone)]
pub struct BackendCostModel {
    /// PCIe bandwidth in GB/s (e.g., 15.75 for PCIe 3.0 x16)
    pub pcie_bandwidth_gbps: f64,
    /// GPU peak TFLOP/s
    pub gpu_peak_tflops: f64,
    /// CPU peak GFLOP/s
    pub cpu_peak_gflops: f64,
    /// Minimum problem size for GPU (elements)
    pub gpu_min_elements: usize,
}

/// Modern AVX2 CPU peak compute in GFLOP/s
const DEFAULT_CPU_PEAK_GFLOPS: f64 = 400.0;

impl Default for BackendCostModel {
    fn default() -> Self {
        Self {
            pcie_bandwidth_gbps: 15.75, // PCIe 3.0 x16
            gpu_peak_tflops: 10.0,      // Mid-range GPU
            cpu_peak_gflops: DEFAULT_CPU_PEAK_GFLOPS,
            gpu_min_elements: 1_000_000, // ~1M elements
        }
    }
}

impl BackendCostModel {
    /// Select optimal backend based on 5× PCIe rule
    ///
    /// # References
    ///
    /// Gregg, C., & Hazelwood, K. (2011). Where is the Data? Why You Cannot
    /// Debate CPU vs. GPU Performance Without the Answer. IEEE ISPASS.
    pub fn select_backend(&self, m: usize, n: usize, k: usize) -> ComputeBackend {
        let flops = 2 * m * n * k;
        let bytes = 4 * (m * k + k * n + m * n); // f32 = 4 bytes
        let arithmetic_intensity = flops as f64 / bytes as f64;

        // Ridge point: where compute = memory bandwidth
        let ridge_point = self.gpu_peak_tflops * 1000.0 / self.pcie_bandwidth_gbps;

        // GPU worthwhile if:
        // 1. High arithmetic intensity (compute-bound)
        // 2. Problem size exceeds minimum threshold
        // 3. Transfer time is amortized (5× rule)
        let elements = m * n * k;
        if arithmetic_intensity > ridge_point && elements > self.gpu_min_elements {
            // Check if wgpu available at runtime
            #[cfg(feature = "wgpu")]
            return ComputeBackend::Wgpu;

            #[cfg(all(not(feature = "wgpu"), feature = "cuda"))]
            return ComputeBackend::Gpu;

            #[allow(unreachable_code)]
            ComputeBackend::Cpu
        } else {
            // CPU is better for small problems or memory-bound workloads
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    return ComputeBackend::Cpu;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                return ComputeBackend::Cpu;
            }
            #[allow(unreachable_code)]
            ComputeBackend::Scalar
        }
    }

    /// Estimate execution time in microseconds
    pub fn estimate_time_us(&self, m: usize, n: usize, k: usize, backend: ComputeBackend) -> f64 {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let bytes = 4.0 * (m * k + k * n + m * n) as f64;

        match backend {
            ComputeBackend::Gpu | ComputeBackend::Wgpu => {
                // Transfer time + compute time
                let transfer_us = bytes / (self.pcie_bandwidth_gbps * 1e3);
                let compute_us = flops / (self.gpu_peak_tflops * 1e6);
                transfer_us + compute_us
            }
            ComputeBackend::Cpu => flops / (self.cpu_peak_gflops * 1e3),
            ComputeBackend::Scalar => {
                // Assume 1 GFLOP/s for scalar
                flops / 1e3
            }
        }
    }
}

/// Unified profiler for all backends
///
/// Collects metrics across CPU (RDTSC), GPU (CUDA events), and wgpu (timestamp queries)
#[derive(Debug, Clone, Default)]
pub struct UnifiedBrickProfiler {
    /// CPU profiling stats
    pub cpu_stats: BlisProfiler,
    /// Selected backend for this run
    pub backend: Option<ComputeBackend>,
    /// Total elements processed
    pub total_elements: u64,
    /// Backend selection decisions
    pub selection_history: Vec<(usize, usize, usize, ComputeBackend)>,
}

impl UnifiedBrickProfiler {
    /// Create a new unified profiler
    pub fn new() -> Self {
        Self {
            cpu_stats: BlisProfiler::enabled(),
            backend: None,
            total_elements: 0,
            selection_history: Vec::new(),
        }
    }

    /// Record backend selection
    pub fn record_selection(&mut self, m: usize, n: usize, k: usize, backend: ComputeBackend) {
        self.backend = Some(backend);
        self.total_elements += (m * n) as u64;
        self.selection_history.push((m, n, k, backend));
    }

    /// Get roofline analysis for current backend
    pub fn roofline_analysis(&self, m: usize, n: usize, k: usize) -> RooflineResult {
        let cost = BackendCostModel::default();
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let bytes = 4.0 * (m * k + k * n + m * n) as f64;
        let ai = flops / bytes;

        let ridge_point = match self.backend.unwrap_or(ComputeBackend::Cpu) {
            ComputeBackend::Gpu | ComputeBackend::Wgpu => {
                cost.gpu_peak_tflops * 1000.0 / cost.pcie_bandwidth_gbps
            }
            ComputeBackend::Cpu | ComputeBackend::Scalar => {
                cost.cpu_peak_gflops / 50.0 // ~50 GB/s memory bandwidth
            }
        };

        if ai < ridge_point {
            RooflineResult::MemoryBound { ai, ridge_point }
        } else {
            RooflineResult::ComputeBound { ai, ridge_point }
        }
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("Unified Brick Profiler Summary\n");
        s.push_str("==============================\n");
        s.push_str(&format!("Backend: {:?}\n", self.backend.unwrap_or(ComputeBackend::Scalar)));
        s.push_str(&format!("Total elements: {}\n", self.total_elements));
        s.push_str(&format!("Selections: {} decisions\n", self.selection_history.len()));
        s.push_str("\nCPU Stats:\n");
        s.push_str(&self.cpu_stats.summary());
        s
    }
}

/// Roofline model result
#[derive(Debug, Clone, Copy)]
pub enum RooflineResult {
    /// Workload is memory-bound (AI < ridge point)
    MemoryBound {
        /// Arithmetic intensity (FLOP/byte)
        ai: f64,
        /// Ridge point where compute = memory
        ridge_point: f64,
    },
    /// Workload is compute-bound (AI > ridge point)
    ComputeBound {
        /// Arithmetic intensity (FLOP/byte)
        ai: f64,
        /// Ridge point where compute = memory
        ridge_point: f64,
    },
}

impl RooflineResult {
    /// Get arithmetic intensity
    pub fn arithmetic_intensity(&self) -> f64 {
        match self {
            RooflineResult::MemoryBound { ai, .. } => *ai,
            RooflineResult::ComputeBound { ai, .. } => *ai,
        }
    }

    /// Check if compute-bound
    pub fn is_compute_bound(&self) -> bool {
        matches!(self, RooflineResult::ComputeBound { .. })
    }
}

/// PTX microkernel definition (for documentation and future CUDA support)
///
/// This is a specification for the GPU microkernel. Actual PTX code generation
/// would be done by the trueno-ptx crate.
///
/// # References
///
/// - NVIDIA PTX ISA Reference Manual
/// - Volkov, V. (2010). Better Performance at Lower Occupancy.
#[derive(Debug, Clone)]
pub struct PtxMicrokernelSpec {
    /// PTX version (e.g., "8.0")
    pub ptx_version: &'static str,
    /// Target SM architecture (e.g., "sm_80")
    pub sm_target: &'static str,
    /// Register count per thread
    pub registers_per_thread: u32,
    /// Shared memory bytes per block
    pub smem_bytes: usize,
    /// Thread block dimensions
    pub block_dim: (u32, u32, u32),
    /// Tile dimensions (MR, NR)
    pub tile_dim: (usize, usize),
}

impl Default for PtxMicrokernelSpec {
    fn default() -> Self {
        Self {
            ptx_version: "8.0",
            sm_target: "sm_80",
            registers_per_thread: 64,
            smem_bytes: 48 * 1024, // 48KB shared memory
            block_dim: (16, 16, 1),
            tile_dim: (16, 16), // 16x16 output tile per warp
        }
    }
}

/// WGSL microkernel specification (for wgpu backend)
///
/// Defines the compute shader for matrix multiplication.
#[derive(Debug, Clone)]
pub struct WgslMicrokernelSpec {
    /// Workgroup size (x, y, z)
    pub workgroup_size: (u32, u32, u32),
    /// Tile dimensions (MR, NR)
    pub tile_dim: (usize, usize),
    /// Use shared memory for tiling
    pub use_shared_memory: bool,
}

impl Default for WgslMicrokernelSpec {
    fn default() -> Self {
        Self { workgroup_size: (8, 8, 1), tile_dim: (8, 8), use_shared_memory: true }
    }
}

impl WgslMicrokernelSpec {
    /// Generate WGSL shader source
    ///
    /// This generates a basic tiled GEMM shader. For production use,
    /// this would be optimized with coalesced memory access and bank conflict avoidance.
    pub fn generate_wgsl(&self) -> String {
        format!(
            r#"// WGSL GEMM Microkernel
// Generated by trueno BLIS module
// Tile: {}x{}, Workgroup: {}x{}x{}

struct GemmParams {{
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
}}

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

var<workgroup> tile_a: array<f32, {tile_a_size}>;
var<workgroup> tile_b: array<f32, {tile_b_size}>;

@compute @workgroup_size({wx}, {wy}, {wz})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {{
    let row = global_id.y;
    let col = global_id.x;

    if (row >= params.m || col >= params.n) {{
        return;
    }}

    var sum: f32 = 0.0;

    // Tile over K dimension
    let num_tiles = (params.k + {tile_k}u - 1u) / {tile_k}u;

    for (var t: u32 = 0u; t < num_tiles; t++) {{
        let k_base = t * {tile_k}u;

        // Load tile_a and tile_b into shared memory
        // (simplified - production code would have proper coalescing)
        let k_idx = k_base + local_id.x;
        if (row < params.m && k_idx < params.k) {{
            tile_a[local_id.y * {tile_k}u + local_id.x] = a[row * params.k + k_idx];
        }}
        if (k_idx < params.k && col < params.n) {{
            tile_b[local_id.y * {tile_k}u + local_id.x] = b[k_idx * params.n + col];
        }}

        workgroupBarrier();

        // Compute partial sum
        for (var kk: u32 = 0u; kk < {tile_k}u; kk++) {{
            if (k_base + kk < params.k) {{
                sum += tile_a[local_id.y * {tile_k}u + kk] * tile_b[kk * {tile_k}u + local_id.x];
            }}
        }}

        workgroupBarrier();
    }}

    // Store result
    let c_idx = row * params.n + col;
    c[c_idx] = params.alpha * sum + params.beta * c[c_idx];
}}
"#,
            self.tile_dim.0,
            self.tile_dim.1,
            self.workgroup_size.0,
            self.workgroup_size.1,
            self.workgroup_size.2,
            tile_a_size = self.tile_dim.0 * self.tile_dim.0,
            tile_b_size = self.tile_dim.0 * self.tile_dim.1,
            wx = self.workgroup_size.0,
            wy = self.workgroup_size.1,
            wz = self.workgroup_size.2,
            tile_k = self.tile_dim.0,
        )
    }
}

/// GEMM with automatic backend selection
///
/// Uses the 5× PCIe rule to select between CPU (asm) and GPU (PTX/WGSL) backends.
pub fn gemm_auto(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    profiler: Option<&mut UnifiedBrickProfiler>,
) -> Result<(), TruenoError> {
    let cost_model = BackendCostModel::default();
    let backend = cost_model.select_backend(m, n, k);

    if let Some(prof) = profiler {
        prof.record_selection(m, n, k, backend);
    }

    match backend {
        ComputeBackend::Cpu | ComputeBackend::Scalar => {
            // Use BLIS CPU implementation
            gemm_blis(m, n, k, a, b, c, None)
        }
        ComputeBackend::Gpu => {
            // PTX backend (stub - requires CUDA support)
            // For now, fall back to CPU
            gemm_blis(m, n, k, a, b, c, None)
        }
        ComputeBackend::Wgpu => {
            // WGSL backend (stub - requires wgpu support)
            // For now, fall back to CPU
            gemm_blis(m, n, k, a, b, c, None)
        }
    }
}
