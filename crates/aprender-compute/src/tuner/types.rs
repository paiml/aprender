#![allow(missing_docs)]
//! Tuner Type Definitions
//!
//! Core enums for quantization, kernel selection, and bottleneck classification.

use crate::brick::BrickBottleneck;
use serde::{Deserialize, Serialize};

// ============================================================================
// QuantType
// ============================================================================

/// Quantization type for feature encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QuantType {
    Q4_0,
    Q4_1,
    #[default]
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    F16,
    F32,
}

impl QuantType {
    /// One-hot encoding index (0-7)
    pub fn to_index(self) -> usize {
        match self {
            QuantType::Q4_0 => 0,
            QuantType::Q4_1 => 1,
            QuantType::Q4K => 2,
            QuantType::Q5K => 3,
            QuantType::Q6K => 4,
            QuantType::Q8_0 => 5,
            QuantType::F16 => 6,
            QuantType::F32 => 7,
        }
    }

    /// Bytes per parameter (approximate)
    pub fn bytes_per_param(self) -> f32 {
        contract_pre_bytes_per_param!();
        match self {
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q4K => 0.5625, // 4.5 bits
            QuantType::Q5K => 0.6875,                                     // 5.5 bits
            QuantType::Q6K => 0.8125,                                     // 6.5 bits
            QuantType::Q8_0 => 1.0,
            QuantType::F16 => 2.0,
            QuantType::F32 => 4.0,
        }
    }
}

// ============================================================================
// KernelType
// ============================================================================

/// Kernel type for feature encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum KernelType {
    // Q4K variants
    #[default]
    TiledQ4K,
    CoalescedQ4K,
    VectorizedQ4K,
    BatchedQ4K,
    Dp4aQ4K,
    FusedRmsNormQ4K,
    // Q6K variants
    CoalescedQ6K,
    // Attention variants
    IncrementalAttention,
    MultiWarpAttention,
    BatchedAttention,
    // Normalization
    RmsNorm,
    VectorizedRmsNorm,
    BatchedRmsNorm,
    // Fused attention projection
    FusedQKVHwDp4aQ4KGemv,
    // Other
    Generic,
    Unknown,
}

impl KernelType {
    /// One-hot encoding index (0-16)
    pub fn to_index(self) -> usize {
        match self {
            KernelType::TiledQ4K => 0,
            KernelType::CoalescedQ4K => 1,
            KernelType::VectorizedQ4K => 2,
            KernelType::BatchedQ4K => 3,
            KernelType::Dp4aQ4K => 4,
            KernelType::FusedRmsNormQ4K => 5,
            KernelType::CoalescedQ6K => 6,
            KernelType::IncrementalAttention => 7,
            KernelType::MultiWarpAttention => 8,
            KernelType::BatchedAttention => 9,
            KernelType::RmsNorm => 10,
            KernelType::VectorizedRmsNorm => 11,
            KernelType::BatchedRmsNorm => 12,
            KernelType::FusedQKVHwDp4aQ4KGemv => 13,
            KernelType::Generic => 14,
            KernelType::Unknown => 15,
        }
    }

    /// Convert kernel index to type (inverse of to_index())
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => KernelType::TiledQ4K,
            1 => KernelType::CoalescedQ4K,
            2 => KernelType::VectorizedQ4K,
            3 => KernelType::BatchedQ4K,
            4 => KernelType::Dp4aQ4K,
            5 => KernelType::FusedRmsNormQ4K,
            6 => KernelType::CoalescedQ6K,
            7 => KernelType::IncrementalAttention,
            8 => KernelType::MultiWarpAttention,
            9 => KernelType::BatchedAttention,
            10 => KernelType::RmsNorm,
            11 => KernelType::VectorizedRmsNorm,
            12 => KernelType::BatchedRmsNorm,
            13 => KernelType::FusedQKVHwDp4aQ4KGemv,
            14 => KernelType::Generic,
            15.. => KernelType::Unknown,
        }
    }

    /// Number of kernel types
    pub const COUNT: usize = 17;
}

// ============================================================================
// BottleneckClass
// ============================================================================

/// Bottleneck classification for ML model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BottleneckClass {
    #[default]
    Unknown,
    MemoryBound,
    ComputeBound,
    LaunchBound,
    AttentionBound,
}

impl BottleneckClass {
    /// Convert from BrickBottleneck
    pub fn from_brick_bottleneck(b: BrickBottleneck) -> Self {
        match b {
            BrickBottleneck::Memory => BottleneckClass::MemoryBound,
            BrickBottleneck::Compute => BottleneckClass::ComputeBound,
            BrickBottleneck::Unknown => BottleneckClass::Unknown,
        }
    }

    /// Recommended action for this bottleneck
    pub fn recommended_action(self) -> &'static str {
        match self {
            BottleneckClass::MemoryBound => {
                "Increase batch size (M) to amortize weight reads across sequences"
            }
            BottleneckClass::ComputeBound => {
                "Rare for inference; check for redundant computation or use tensor cores"
            }
            BottleneckClass::LaunchBound => {
                "Enable CUDA graphs or fuse kernels to reduce launch overhead"
            }
            BottleneckClass::AttentionBound => {
                "Use Flash Decoding, reduce sequence length, or use batched attention"
            }
            BottleneckClass::Unknown => "Run profiling to identify bottleneck",
        }
    }

    /// One-hot encoding index (0-4)
    pub fn to_index(self) -> usize {
        match self {
            BottleneckClass::Unknown => 0,
            BottleneckClass::MemoryBound => 1,
            BottleneckClass::ComputeBound => 2,
            BottleneckClass::LaunchBound => 3,
            BottleneckClass::AttentionBound => 4,
        }
    }
}

impl std::fmt::Display for BottleneckClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckClass::Unknown => write!(f, "Unknown"),
            BottleneckClass::MemoryBound => write!(f, "MemoryBound"),
            BottleneckClass::ComputeBound => write!(f, "ComputeBound"),
            BottleneckClass::LaunchBound => write!(f, "LaunchBound"),
            BottleneckClass::AttentionBound => write!(f, "AttentionBound"),
        }
    }
}
