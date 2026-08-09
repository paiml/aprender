//! Hardware Capability Detection (PMAT-447)
//!
//! Detects CPU SIMD capabilities, GPU presence, and calculates
//! theoretical peak performance for roofline analysis.
//!
//! Integrates with `pmat brick-score` for hardware-aware profiling.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Get hostname (native only, returns "wasm" on WASM targets)
#[cfg(not(target_arch = "wasm32"))]
fn get_hostname() -> String {
    hostname::get().map(|h| h.to_string_lossy().to_string()).unwrap_or_else(|e| {
        eprintln!("warning: failed to get hostname: {e}");
        "unknown".to_string()
    })
}

/// Get hostname (WASM fallback)
#[cfg(target_arch = "wasm32")]
fn get_hostname() -> String {
    "wasm".to_string()
}

/// SIMD instruction set width
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimdWidth {
    /// No SIMD (scalar)
    Scalar,
    /// ARM NEON (128-bit, 4×f32)
    Neon128,
    /// SSE2 (128-bit, 4×f32)
    Sse2,
    /// AVX2 (256-bit, 8×f32)
    Avx2,
    /// AVX-512 (512-bit, 16×f32)
    Avx512,
    /// WebAssembly SIMD (128-bit, 4×f32)
    WasmSimd128,
}

impl SimdWidth {
    /// Number of f32 lanes
    pub fn lanes(&self) -> usize {
        match self {
            SimdWidth::Scalar => 1,
            SimdWidth::Neon128 | SimdWidth::Sse2 | SimdWidth::WasmSimd128 => 4,
            SimdWidth::Avx2 => 8,
            SimdWidth::Avx512 => 16,
        }
    }

    /// Bit width
    pub fn bits(&self) -> usize {
        self.lanes() * 32
    }

    /// Typical speedup factor for compute-bound operations
    pub fn compute_speedup(&self) -> f64 {
        match self {
            SimdWidth::Scalar => 1.0,
            SimdWidth::Neon128 | SimdWidth::Sse2 | SimdWidth::WasmSimd128 => 4.0,
            SimdWidth::Avx2 => 10.0,   // 8-12x measured in trueno-zram
            SimdWidth::Avx512 => 12.0, // 8-13x measured
        }
    }
}

/// GPU compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// No GPU available
    None,
    /// NVIDIA CUDA
    Cuda,
    /// WebGPU (cross-platform)
    Wgpu,
    /// Apple Metal
    Metal,
    /// Vulkan compute
    Vulkan,
}

/// CPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCapability {
    /// CPU vendor (Intel, AMD, Apple, etc.)
    pub vendor: String,
    /// CPU model name
    pub model: String,
    /// Number of physical cores
    pub cores: usize,
    /// Number of logical threads
    pub threads: usize,
    /// Best available SIMD width
    pub simd: SimdWidth,
    /// Base frequency in GHz
    pub base_freq_ghz: f64,
    /// Theoretical peak GFLOP/s (FMA)
    pub peak_gflops: f64,
    /// Memory bandwidth in GB/s (estimated)
    pub memory_bw_gbps: f64,
}

/// GPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapability {
    /// GPU vendor
    pub vendor: String,
    /// GPU model name
    pub model: String,
    /// Compute backend
    pub backend: GpuBackend,
    /// CUDA compute capability (e.g., "8.9" for RTX 4090)
    pub compute_capability: Option<String>,
    /// Peak FP32 TFLOP/s
    pub peak_tflops_fp32: f64,
    /// Peak Tensor Core TFLOP/s (NVIDIA only)
    pub peak_tflops_tensor: Option<f64>,
    /// Memory bandwidth in GB/s
    pub memory_bw_gbps: f64,
    /// VRAM in GB
    pub vram_gb: f64,
}

/// Complete hardware capability profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapability {
    /// Detection timestamp
    pub timestamp: String,
    /// Hostname
    pub hostname: String,
    /// CPU capabilities
    pub cpu: CpuCapability,
    /// GPU capabilities (if present)
    pub gpu: Option<GpuCapability>,
    /// Roofline model parameters
    pub roofline: RooflineParams,
    /// PMAT-452: Byte budget configuration for compression/I/O workloads
    #[serde(default)]
    pub byte_budget: Option<crate::brick::ByteBudget>,
}

/// Roofline model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineParams {
    /// CPU arithmetic intensity threshold (GFLOP/s ÷ GB/s)
    pub cpu_arithmetic_intensity: f64,
    /// GPU arithmetic intensity threshold
    pub gpu_arithmetic_intensity: Option<f64>,
}

impl HardwareCapability {
    /// Detect hardware capabilities at runtime
    pub fn detect() -> Self {
        let cpu = detect_cpu();
        let gpu = detect_gpu();

        let cpu_ai = cpu.peak_gflops / cpu.memory_bw_gbps;
        let gpu_ai = gpu.as_ref().map(|g| g.peak_tflops_fp32 * 1000.0 / g.memory_bw_gbps);
        // PMAT-452: Extract memory bandwidth before moving cpu
        let byte_budget_throughput = cpu.memory_bw_gbps.min(25.0);

        HardwareCapability {
            timestamp: chrono::Utc::now().to_rfc3339(),
            hostname: get_hostname(),
            cpu,
            gpu,
            roofline: RooflineParams {
                cpu_arithmetic_intensity: cpu_ai,
                gpu_arithmetic_intensity: gpu_ai,
            },
            // PMAT-452: Default byte budget based on memory bandwidth
            byte_budget: Some(crate::brick::ByteBudget::from_throughput(byte_budget_throughput)),
        }
    }

    /// Load from TOML file or detect if missing
    pub fn load_or_detect(path: &Path) -> Self {
        if path.exists() {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(cap) = toml::from_str(&content) {
                    return cap;
                }
            }
        }
        let cap = Self::detect();
        // Try to cache it
        let _ = cap.save(path);
        cap
    }

    /// Save to TOML file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, content)
    }

    /// Get the best available backend for a workload
    pub fn best_backend(&self) -> GpuBackend {
        self.gpu.as_ref().map(|g| g.backend).unwrap_or(GpuBackend::None)
    }

    /// Calculate expected throughput for a brick given its arithmetic intensity
    pub fn expected_throughput_gflops(&self, arithmetic_intensity: f64, use_gpu: bool) -> f64 {
        if use_gpu {
            if let Some(gpu) = &self.gpu {
                let memory_bound = gpu.memory_bw_gbps * arithmetic_intensity;
                let compute_bound = gpu.peak_tflops_fp32 * 1000.0;
                memory_bound.min(compute_bound)
            } else {
                self.cpu_expected_throughput(arithmetic_intensity)
            }
        } else {
            self.cpu_expected_throughput(arithmetic_intensity)
        }
    }

    fn cpu_expected_throughput(&self, arithmetic_intensity: f64) -> f64 {
        let memory_bound = self.cpu.memory_bw_gbps * arithmetic_intensity;
        let compute_bound = self.cpu.peak_gflops;
        memory_bound.min(compute_bound)
    }

    /// Determine if workload is memory-bound or compute-bound
    pub fn bottleneck(&self, arithmetic_intensity: f64, use_gpu: bool) -> Bottleneck {
        let threshold = if use_gpu {
            self.roofline.gpu_arithmetic_intensity.unwrap_or(f64::MAX)
        } else {
            self.roofline.cpu_arithmetic_intensity
        };

        if arithmetic_intensity < threshold {
            Bottleneck::Memory
        } else {
            Bottleneck::Compute
        }
    }
}

/// Workload bottleneck classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bottleneck {
    /// Limited by memory bandwidth
    Memory,
    /// Limited by compute throughput
    Compute,
}

/// Detect CPU capabilities
fn detect_cpu() -> CpuCapability {
    let simd = detect_simd();
    let cores = num_cpus::get_physical();
    let threads = num_cpus::get();

    // Estimate frequency (fallback to 3.0 GHz if unknown)
    let base_freq_ghz = 3.0;

    // Calculate peak GFLOP/s: cores × lanes × 2 (FMA) × freq
    let peak_gflops = (cores as f64) * (simd.lanes() as f64) * 2.0 * base_freq_ghz;

    // Estimate memory bandwidth (DDR5-5600 dual channel ≈ 89.6 GB/s)
    let memory_bw_gbps = 80.0; // Conservative estimate

    CpuCapability {
        vendor: "Unknown".to_string(),
        model: "Unknown".to_string(),
        cores,
        threads,
        simd,
        base_freq_ghz,
        peak_gflops,
        memory_bw_gbps,
    }
}

/// Detect best available SIMD width
fn detect_simd() -> SimdWidth {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdWidth::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdWidth::Avx2;
        }
        if is_x86_feature_detected!("sse2") {
            return SimdWidth::Sse2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return SimdWidth::Neon128;
    }

    #[cfg(target_arch = "wasm32")]
    {
        return SimdWidth::WasmSimd128;
    }

    #[allow(unreachable_code)]
    SimdWidth::Scalar
}

/// Detect GPU capabilities
fn detect_gpu() -> Option<GpuCapability> {
    // Check for CUDA first (highest performance)
    #[cfg(feature = "cuda")]
    {
        if let Some(gpu) = detect_cuda_gpu() {
            return Some(gpu);
        }
    }

    // Fallback: no GPU detected
    None
}

#[cfg(feature = "cuda")]
fn detect_cuda_gpu() -> Option<GpuCapability> {
    // This would use cuDeviceGetAttribute in a real implementation
    // For now, return None and let the caller provide GPU info
    None
}

/// Default hardware.toml path
pub fn default_hardware_path() -> std::path::PathBuf {
    #[cfg(feature = "hardware-detect")]
    {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".pmat")
            .join("hardware.toml")
    }
    #[cfg(not(feature = "hardware-detect"))]
    {
        std::path::PathBuf::from(".pmat").join("hardware.toml")
    }
}

#[cfg(test)]
mod tests;
