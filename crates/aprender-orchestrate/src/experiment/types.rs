//! Core type definitions for experiment tracking.
//!
//! This module contains the fundamental types for compute devices, architectures,
//! and model paradigms used throughout the experiment tracking system.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during experiment tracking operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ExperimentError {
    #[error("Invalid compute device configuration: {0}")]
    InvalidComputeDevice(String),

    #[error("Metrics collection failed: {0}")]
    MetricsCollectionFailed(String),

    #[error("Pareto frontier calculation failed: {0}")]
    ParetoFrontierFailed(String),

    #[error("Invalid ORCID format: {0}")]
    InvalidOrcid(String),

    #[error("Citation generation failed: {0}")]
    CitationGenerationFailed(String),

    #[error("Sovereign distribution validation failed: {0}")]
    SovereignValidationFailed(String),

    #[error("Experiment storage error: {0}")]
    StorageError(String),
}

/// Compute device abstraction for heterogeneous hardware
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputeDevice {
    /// Standard CPU execution
    Cpu { cores: u32, threads_per_core: u32, architecture: CpuArchitecture },
    /// NVIDIA/AMD GPU acceleration
    Gpu { name: String, memory_gb: f32, compute_capability: Option<String>, vendor: GpuVendor },
    /// Google TPU accelerator
    Tpu { version: TpuVersion, cores: u32 },
    /// Apple Silicon unified memory
    AppleSilicon { chip: AppleChip, neural_engine_cores: u32, gpu_cores: u32, memory_gb: u32 },
    /// Edge/embedded devices
    Edge { name: String, power_budget_watts: f32 },
}

/// CPU architecture variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuArchitecture {
    X86_64,
    Aarch64,
    Riscv64,
    Wasm32,
}

/// GPU vendor identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
}

/// TPU version variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpuVersion {
    V2,
    V3,
    V4,
    V5e,
    V5p,
}

/// Apple Silicon chip variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppleChip {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M4,
    M4Pro,
    M4Max,
}

impl ComputeDevice {
    /// Calculate theoretical FLOPS for the device
    pub fn theoretical_flops(&self) -> f64 {
        match self {
            ComputeDevice::Cpu { cores, threads_per_core, architecture } => {
                let base_flops = match architecture {
                    CpuArchitecture::X86_64 => 32.0,  // AVX2: 8 FP32 * 4 ops
                    CpuArchitecture::Aarch64 => 16.0, // NEON: 4 FP32 * 4 ops
                    CpuArchitecture::Riscv64 => 8.0,
                    CpuArchitecture::Wasm32 => 4.0,
                };
                (*cores as f64) * (*threads_per_core as f64) * base_flops * 1e9
            }
            ComputeDevice::Gpu { memory_gb, vendor, .. } => {
                // Rough estimate based on memory bandwidth
                let bandwidth_factor = match vendor {
                    GpuVendor::Nvidia => 15.0,
                    GpuVendor::Amd => 12.0,
                    GpuVendor::Intel => 8.0,
                    GpuVendor::Apple => 10.0,
                };
                (*memory_gb as f64) * bandwidth_factor * 1e12
            }
            ComputeDevice::Tpu { version, cores } => {
                let flops_per_core = match version {
                    TpuVersion::V2 => 45e12,
                    TpuVersion::V3 => 90e12,
                    TpuVersion::V4 => 275e12,
                    TpuVersion::V5e => 197e12,
                    TpuVersion::V5p => 459e12,
                };
                (*cores as f64) * flops_per_core
            }
            ComputeDevice::AppleSilicon { chip, gpu_cores, .. } => {
                let flops_per_gpu_core = match chip {
                    AppleChip::M1 | AppleChip::M1Pro | AppleChip::M1Max | AppleChip::M1Ultra => {
                        128e9
                    }
                    AppleChip::M2 | AppleChip::M2Pro | AppleChip::M2Max | AppleChip::M2Ultra => {
                        150e9
                    }
                    AppleChip::M3 | AppleChip::M3Pro | AppleChip::M3Max => 180e9,
                    AppleChip::M4 | AppleChip::M4Pro | AppleChip::M4Max => 200e9,
                };
                (*gpu_cores as f64) * flops_per_gpu_core
            }
            ComputeDevice::Edge { power_budget_watts, .. } => {
                // Assume ~10 GFLOPS per watt for edge devices
                (*power_budget_watts as f64) * 10e9
            }
        }
    }

    /// Estimate power consumption in watts
    pub fn estimated_power_watts(&self) -> f32 {
        match self {
            ComputeDevice::Cpu { cores, .. } => (*cores as f32) * 15.0,
            ComputeDevice::Gpu { memory_gb, vendor, .. } => {
                let base = match vendor {
                    GpuVendor::Nvidia => 30.0,
                    GpuVendor::Amd => 35.0,
                    GpuVendor::Intel => 25.0,
                    GpuVendor::Apple => 20.0,
                };
                *memory_gb * base
            }
            ComputeDevice::Tpu { version, cores } => {
                let per_core = match version {
                    TpuVersion::V2 => 40.0,
                    TpuVersion::V3 => 50.0,
                    TpuVersion::V4 => 60.0,
                    TpuVersion::V5e => 45.0,
                    TpuVersion::V5p => 70.0,
                };
                (*cores as f32) * per_core
            }
            ComputeDevice::AppleSilicon { chip, .. } => match chip {
                AppleChip::M1 => 20.0,
                AppleChip::M1Pro => 30.0,
                AppleChip::M1Max => 40.0,
                AppleChip::M1Ultra => 60.0,
                AppleChip::M2 => 22.0,
                AppleChip::M2Pro => 32.0,
                AppleChip::M2Max => 45.0,
                AppleChip::M2Ultra => 65.0,
                AppleChip::M3 => 24.0,
                AppleChip::M3Pro => 35.0,
                AppleChip::M3Max => 50.0,
                AppleChip::M4 => 25.0,
                AppleChip::M4Pro => 38.0,
                AppleChip::M4Max => 55.0,
            },
            ComputeDevice::Edge { power_budget_watts, .. } => *power_budget_watts,
        }
    }
}

/// Model training paradigm classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelParadigm {
    /// Traditional ML (sklearn-style)
    TraditionalML,
    /// Deep Learning from scratch
    DeepLearning,
    /// Fine-tuning pretrained models
    FineTuning,
    /// Knowledge distillation
    Distillation,
    /// Mixture of Experts
    MoE,
    /// Reinforcement Learning
    ReinforcementLearning,
    /// Federated Learning
    FederatedLearning,
    /// Neural Architecture Search
    Nas,
    /// Continual/Lifelong Learning
    ContinualLearning,
    /// Meta-Learning
    MetaLearning,
}

impl ModelParadigm {
    /// Get typical compute intensity for this paradigm
    pub fn compute_intensity(&self) -> ComputeIntensity {
        match self {
            ModelParadigm::TraditionalML => ComputeIntensity::Low,
            ModelParadigm::DeepLearning => ComputeIntensity::High,
            ModelParadigm::FineTuning => ComputeIntensity::Medium,
            ModelParadigm::Distillation => ComputeIntensity::Medium,
            ModelParadigm::MoE => ComputeIntensity::VeryHigh,
            ModelParadigm::ReinforcementLearning => ComputeIntensity::High,
            ModelParadigm::FederatedLearning => ComputeIntensity::Medium,
            ModelParadigm::Nas => ComputeIntensity::VeryHigh,
            ModelParadigm::ContinualLearning => ComputeIntensity::Medium,
            ModelParadigm::MetaLearning => ComputeIntensity::High,
        }
    }

    /// Check if paradigm typically benefits from GPU
    pub fn benefits_from_gpu(&self) -> bool {
        matches!(
            self,
            ModelParadigm::DeepLearning
                | ModelParadigm::FineTuning
                | ModelParadigm::MoE
                | ModelParadigm::ReinforcementLearning
                | ModelParadigm::Nas
        )
    }
}

/// Compute intensity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeIntensity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Platform efficiency classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlatformEfficiency {
    /// Server-grade hardware (datacenter)
    Server,
    /// Desktop workstation
    Workstation,
    /// Laptop
    Laptop,
    /// Edge device
    Edge,
    /// Mobile device
    Mobile,
    /// Embedded system
    Embedded,
}

impl PlatformEfficiency {
    /// Get typical power budget for platform
    pub fn typical_power_budget_watts(&self) -> f32 {
        match self {
            PlatformEfficiency::Server => 500.0,
            PlatformEfficiency::Workstation => 350.0,
            PlatformEfficiency::Laptop => 65.0,
            PlatformEfficiency::Edge => 15.0,
            PlatformEfficiency::Mobile => 5.0,
            PlatformEfficiency::Embedded => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ExperimentError Tests
    // =========================================================================

    #[test]
    fn test_experiment_error_display() {
        let err = ExperimentError::InvalidComputeDevice("test".into());
        assert!(err.to_string().contains("Invalid compute device"));

        let err = ExperimentError::MetricsCollectionFailed("test".into());
        assert!(err.to_string().contains("Metrics collection failed"));

        let err = ExperimentError::ParetoFrontierFailed("test".into());
        assert!(err.to_string().contains("Pareto frontier"));

        let err = ExperimentError::InvalidOrcid("test".into());
        assert!(err.to_string().contains("Invalid ORCID"));

        let err = ExperimentError::CitationGenerationFailed("test".into());
        assert!(err.to_string().contains("Citation generation"));

        let err = ExperimentError::SovereignValidationFailed("test".into());
        assert!(err.to_string().contains("Sovereign distribution"));

        let err = ExperimentError::StorageError("test".into());
        assert!(err.to_string().contains("storage error"));
    }

    #[test]
    fn test_experiment_error_equality() {
        let err1 = ExperimentError::InvalidComputeDevice("a".into());
        let err2 = ExperimentError::InvalidComputeDevice("a".into());
        assert_eq!(err1, err2);
    }

    // =========================================================================
    // ComputeDevice Tests
    // =========================================================================

    #[test]
    fn test_cpu_theoretical_flops_x86() {
        let cpu = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };
        let flops = cpu.theoretical_flops();
        assert!(flops > 0.0);
        assert_eq!(flops, 8.0 * 2.0 * 32.0 * 1e9);
    }

    #[test]
    fn test_cpu_theoretical_flops_aarch64() {
        let cpu = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 1,
            architecture: CpuArchitecture::Aarch64,
        };
        let flops = cpu.theoretical_flops();
        assert_eq!(flops, 8.0 * 1.0 * 16.0 * 1e9);
    }

    #[test]
    fn test_cpu_theoretical_flops_riscv() {
        let cpu = ComputeDevice::Cpu {
            cores: 4,
            threads_per_core: 1,
            architecture: CpuArchitecture::Riscv64,
        };
        let flops = cpu.theoretical_flops();
        assert_eq!(flops, 4.0 * 1.0 * 8.0 * 1e9);
    }

    #[test]
    fn test_cpu_theoretical_flops_wasm() {
        let cpu = ComputeDevice::Cpu {
            cores: 1,
            threads_per_core: 1,
            architecture: CpuArchitecture::Wasm32,
        };
        let flops = cpu.theoretical_flops();
        assert_eq!(flops, 1.0 * 1.0 * 4.0 * 1e9);
    }

    #[test]
    fn test_gpu_theoretical_flops_nvidia() {
        let gpu = ComputeDevice::Gpu {
            name: "RTX 4090".into(),
            memory_gb: 24.0,
            compute_capability: Some("8.9".into()),
            vendor: GpuVendor::Nvidia,
        };
        let flops = gpu.theoretical_flops();
        assert_eq!(flops, 24.0 * 15.0 * 1e12);
    }

    #[test]
    fn test_gpu_theoretical_flops_amd() {
        let gpu = ComputeDevice::Gpu {
            name: "RX 7900".into(),
            memory_gb: 20.0,
            compute_capability: None,
            vendor: GpuVendor::Amd,
        };
        let flops = gpu.theoretical_flops();
        assert_eq!(flops, 20.0 * 12.0 * 1e12);
    }

    #[test]
    fn test_gpu_theoretical_flops_intel() {
        let gpu = ComputeDevice::Gpu {
            name: "Arc A770".into(),
            memory_gb: 16.0,
            compute_capability: None,
            vendor: GpuVendor::Intel,
        };
        let flops = gpu.theoretical_flops();
        assert_eq!(flops, 16.0 * 8.0 * 1e12);
    }

    #[test]
    fn test_gpu_theoretical_flops_apple() {
        let gpu = ComputeDevice::Gpu {
            name: "Apple GPU".into(),
            memory_gb: 32.0,
            compute_capability: None,
            vendor: GpuVendor::Apple,
        };
        let flops = gpu.theoretical_flops();
        assert_eq!(flops, 32.0 * 10.0 * 1e12);
    }

    #[test]
    fn test_tpu_theoretical_flops() {
        let tpu_v2 = ComputeDevice::Tpu { version: TpuVersion::V2, cores: 1 };
        assert_eq!(tpu_v2.theoretical_flops(), 45e12);

        let tpu_v3 = ComputeDevice::Tpu { version: TpuVersion::V3, cores: 2 };
        assert_eq!(tpu_v3.theoretical_flops(), 2.0 * 90e12);

        let tpu_v4 = ComputeDevice::Tpu { version: TpuVersion::V4, cores: 1 };
        assert_eq!(tpu_v4.theoretical_flops(), 275e12);

        let tpu_v5e = ComputeDevice::Tpu { version: TpuVersion::V5e, cores: 1 };
        assert_eq!(tpu_v5e.theoretical_flops(), 197e12);

        let tpu_v5p = ComputeDevice::Tpu { version: TpuVersion::V5p, cores: 1 };
        assert_eq!(tpu_v5p.theoretical_flops(), 459e12);
    }

    #[test]
    fn test_apple_silicon_theoretical_flops() {
        let m1 = ComputeDevice::AppleSilicon {
            chip: AppleChip::M1,
            neural_engine_cores: 16,
            gpu_cores: 8,
            memory_gb: 16,
        };
        assert_eq!(m1.theoretical_flops(), 8.0 * 128e9);

        let m2_max = ComputeDevice::AppleSilicon {
            chip: AppleChip::M2Max,
            neural_engine_cores: 16,
            gpu_cores: 38,
            memory_gb: 96,
        };
        assert_eq!(m2_max.theoretical_flops(), 38.0 * 150e9);

        let m3_pro = ComputeDevice::AppleSilicon {
            chip: AppleChip::M3Pro,
            neural_engine_cores: 16,
            gpu_cores: 18,
            memory_gb: 36,
        };
        assert_eq!(m3_pro.theoretical_flops(), 18.0 * 180e9);

        let m4_max = ComputeDevice::AppleSilicon {
            chip: AppleChip::M4Max,
            neural_engine_cores: 16,
            gpu_cores: 40,
            memory_gb: 128,
        };
        assert_eq!(m4_max.theoretical_flops(), 40.0 * 200e9);
    }

    #[test]
    fn test_edge_device_theoretical_flops() {
        let edge = ComputeDevice::Edge { name: "Jetson Nano".into(), power_budget_watts: 10.0 };
        assert_eq!(edge.theoretical_flops(), 10.0 * 10e9);
    }

    #[test]
    fn test_cpu_estimated_power() {
        let cpu = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };
        assert_eq!(cpu.estimated_power_watts(), 8.0 * 15.0);
    }

    #[test]
    fn test_gpu_estimated_power() {
        let nvidia = ComputeDevice::Gpu {
            name: "RTX".into(),
            memory_gb: 24.0,
            compute_capability: None,
            vendor: GpuVendor::Nvidia,
        };
        assert_eq!(nvidia.estimated_power_watts(), 24.0 * 30.0);

        let amd = ComputeDevice::Gpu {
            name: "RX".into(),
            memory_gb: 16.0,
            compute_capability: None,
            vendor: GpuVendor::Amd,
        };
        assert_eq!(amd.estimated_power_watts(), 16.0 * 35.0);
    }

    #[test]
    fn test_tpu_estimated_power() {
        let tpu = ComputeDevice::Tpu { version: TpuVersion::V4, cores: 4 };
        assert_eq!(tpu.estimated_power_watts(), 4.0 * 60.0);
    }

    #[test]
    fn test_apple_silicon_estimated_power() {
        let m1 = ComputeDevice::AppleSilicon {
            chip: AppleChip::M1,
            neural_engine_cores: 16,
            gpu_cores: 8,
            memory_gb: 16,
        };
        assert_eq!(m1.estimated_power_watts(), 20.0);

        let m2_ultra = ComputeDevice::AppleSilicon {
            chip: AppleChip::M2Ultra,
            neural_engine_cores: 32,
            gpu_cores: 76,
            memory_gb: 192,
        };
        assert_eq!(m2_ultra.estimated_power_watts(), 65.0);

        let m4 = ComputeDevice::AppleSilicon {
            chip: AppleChip::M4,
            neural_engine_cores: 16,
            gpu_cores: 10,
            memory_gb: 24,
        };
        assert_eq!(m4.estimated_power_watts(), 25.0);
    }

    #[test]
    fn test_edge_device_estimated_power() {
        let edge = ComputeDevice::Edge { name: "Pi".into(), power_budget_watts: 5.0 };
        assert_eq!(edge.estimated_power_watts(), 5.0);
    }

    // =========================================================================
    // ModelParadigm Tests
    // =========================================================================

    #[test]
    fn test_model_paradigm_compute_intensity() {
        assert_eq!(ModelParadigm::TraditionalML.compute_intensity(), ComputeIntensity::Low);
        assert_eq!(ModelParadigm::DeepLearning.compute_intensity(), ComputeIntensity::High);
        assert_eq!(ModelParadigm::FineTuning.compute_intensity(), ComputeIntensity::Medium);
        assert_eq!(ModelParadigm::Distillation.compute_intensity(), ComputeIntensity::Medium);
        assert_eq!(ModelParadigm::MoE.compute_intensity(), ComputeIntensity::VeryHigh);
        assert_eq!(
            ModelParadigm::ReinforcementLearning.compute_intensity(),
            ComputeIntensity::High
        );
        assert_eq!(ModelParadigm::FederatedLearning.compute_intensity(), ComputeIntensity::Medium);
        assert_eq!(ModelParadigm::Nas.compute_intensity(), ComputeIntensity::VeryHigh);
        assert_eq!(ModelParadigm::ContinualLearning.compute_intensity(), ComputeIntensity::Medium);
        assert_eq!(ModelParadigm::MetaLearning.compute_intensity(), ComputeIntensity::High);
    }

    #[test]
    fn test_model_paradigm_benefits_from_gpu() {
        assert!(ModelParadigm::DeepLearning.benefits_from_gpu());
        assert!(ModelParadigm::FineTuning.benefits_from_gpu());
        assert!(ModelParadigm::MoE.benefits_from_gpu());
        assert!(ModelParadigm::ReinforcementLearning.benefits_from_gpu());
        assert!(ModelParadigm::Nas.benefits_from_gpu());

        assert!(!ModelParadigm::TraditionalML.benefits_from_gpu());
        assert!(!ModelParadigm::Distillation.benefits_from_gpu());
        assert!(!ModelParadigm::FederatedLearning.benefits_from_gpu());
        assert!(!ModelParadigm::ContinualLearning.benefits_from_gpu());
        assert!(!ModelParadigm::MetaLearning.benefits_from_gpu());
    }

    // =========================================================================
    // PlatformEfficiency Tests
    // =========================================================================

    #[test]
    fn test_platform_efficiency_power_budget() {
        assert_eq!(PlatformEfficiency::Server.typical_power_budget_watts(), 500.0);
        assert_eq!(PlatformEfficiency::Workstation.typical_power_budget_watts(), 350.0);
        assert_eq!(PlatformEfficiency::Laptop.typical_power_budget_watts(), 65.0);
        assert_eq!(PlatformEfficiency::Edge.typical_power_budget_watts(), 15.0);
        assert_eq!(PlatformEfficiency::Mobile.typical_power_budget_watts(), 5.0);
        assert_eq!(PlatformEfficiency::Embedded.typical_power_budget_watts(), 1.0);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_compute_device_serialization() {
        let cpu = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };
        let json = serde_json::to_string(&cpu).expect("json serialize failed");
        let deserialized: ComputeDevice =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(cpu, deserialized);
    }

    #[test]
    fn test_model_paradigm_serialization() {
        let paradigm = ModelParadigm::DeepLearning;
        let json = serde_json::to_string(&paradigm).expect("json serialize failed");
        let deserialized: ModelParadigm =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(paradigm, deserialized);
    }
}
