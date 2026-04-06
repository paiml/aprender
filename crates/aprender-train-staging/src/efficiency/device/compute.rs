//! Compute device abstraction.

use serde::{Deserialize, Serialize};

use super::apple::AppleSiliconInfo;
use super::cpu::CpuInfo;
use super::gpu::GpuInfo;
use super::tpu::TpuInfo;

/// Compute device abstraction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputeDevice {
    /// CPU device
    Cpu(CpuInfo),
    /// GPU device (NVIDIA, AMD, Intel)
    Gpu(GpuInfo),
    /// TPU device (Google)
    Tpu(TpuInfo),
    /// Apple Silicon with unified memory
    AppleSilicon(AppleSiliconInfo),
}

impl ComputeDevice {
    /// Auto-detect available compute devices
    pub fn detect() -> Vec<Self> {
        let mut devices = Vec::new();

        // Always detect CPU
        devices.push(Self::Cpu(CpuInfo::detect()));

        // Check for Apple Silicon
        if let Some(apple) = AppleSiliconInfo::detect() {
            devices.push(Self::AppleSilicon(apple));
        }

        // GPU detection would require platform-specific APIs (CUDA, ROCm, Metal)
        // For now, we only auto-detect CPU and Apple Silicon

        devices
    }

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    /// Check if this is a TPU device
    pub fn is_tpu(&self) -> bool {
        matches!(self, Self::Tpu(_))
    }

    /// Check if this is Apple Silicon
    pub fn is_apple_silicon(&self) -> bool {
        matches!(self, Self::AppleSilicon(_))
    }

    /// Get available memory in bytes
    pub fn memory_bytes(&self) -> u64 {
        match self {
            Self::Cpu(info) => {
                // Return system memory (approximation)
                // In real implementation, would query actual system RAM
                u64::from(info.cores) * 4 * 1024 * 1024 * 1024 // ~4GB per core estimate
            }
            Self::Gpu(info) => info.vram_bytes,
            Self::Tpu(info) => info.hbm_bytes,
            Self::AppleSilicon(info) => info.unified_memory_bytes,
        }
    }

    /// Get device name
    pub fn name(&self) -> &str {
        match self {
            Self::Cpu(info) => &info.model,
            Self::Gpu(info) => &info.name,
            Self::Tpu(info) => &info.version,
            Self::AppleSilicon(info) => &info.chip,
        }
    }

    /// Get compute cores/units
    pub fn compute_units(&self) -> u32 {
        match self {
            Self::Cpu(info) => info.threads,
            Self::Gpu(_) => 0, // Would need more info
            Self::Tpu(info) => info.cores,
            Self::AppleSilicon(info) => info.total_cpu_cores() + info.gpu_cores,
        }
    }

    /// Estimate relative compute power (normalized, CPU = 1.0)
    pub fn relative_compute_power(&self) -> f64 {
        match self {
            Self::Cpu(info) => f64::from(info.threads) / 8.0, // Normalize to 8-thread CPU
            Self::Gpu(info) => {
                // Rough estimate based on VRAM (proxy for capability)
                10.0 * (info.vram_gb() / 8.0) // 8GB GPU ~ 10x CPU
            }
            Self::Tpu(info) => {
                // TPUs are highly optimized for matrix ops
                50.0 * (f64::from(info.cores) / 8.0)
            }
            Self::AppleSilicon(info) => {
                // P-cores are faster, E-cores are efficient
                (f64::from(info.p_cores) * 1.5 + f64::from(info.e_cores) * 0.5) / 8.0
                    + f64::from(info.gpu_cores) * 0.5
            }
        }
    }
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu(info) => write!(
                f,
                "CPU: {} ({} cores, {} threads, {})",
                info.model, info.cores, info.threads, info.simd
            ),
            Self::Gpu(info) => {
                write!(f, "GPU: {} ({:.1} GB VRAM", info.name, info.vram_gb())?;
                if let Some((major, minor)) = info.compute_capability {
                    write!(f, ", SM {major}.{minor}")?;
                }
                write!(f, ")")
            }
            Self::Tpu(info) => write!(
                f,
                "TPU: {} ({} cores, {:.1} GB HBM)",
                info.version,
                info.cores,
                info.hbm_gb()
            ),
            Self::AppleSilicon(info) => write!(
                f,
                "Apple Silicon: {} ({}P+{}E cores, {} GPU cores, {:.1} GB)",
                info.chip,
                info.p_cores,
                info.e_cores,
                info.gpu_cores,
                info.unified_memory_gb()
            ),
        }
    }
}
