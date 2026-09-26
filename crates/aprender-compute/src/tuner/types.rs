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
    /// One-hot encoding index (0-16). `KernelType`'s declaration order matches
    /// the index assignment exactly, so this is just the discriminant.
    pub fn to_index(self) -> usize {
        self as usize
    }

    /// All variants that occupy indices 0..14, in `to_index()` order. `Unknown`
    /// (index 15) is deliberately excluded: it is the catch-all fallback below.
    const INDEXED: [KernelType; 15] = [
        KernelType::TiledQ4K,
        KernelType::CoalescedQ4K,
        KernelType::VectorizedQ4K,
        KernelType::BatchedQ4K,
        KernelType::Dp4aQ4K,
        KernelType::FusedRmsNormQ4K,
        KernelType::CoalescedQ6K,
        KernelType::IncrementalAttention,
        KernelType::MultiWarpAttention,
        KernelType::BatchedAttention,
        KernelType::RmsNorm,
        KernelType::VectorizedRmsNorm,
        KernelType::BatchedRmsNorm,
        KernelType::FusedQKVHwDp4aQ4KGemv,
        KernelType::Generic,
    ];

    /// Convert kernel index to type (inverse of to_index()). Any index at or
    /// beyond 15 (including the `Unknown` index itself) maps to `Unknown`.
    pub fn from_index(idx: usize) -> Self {
        Self::INDEXED.get(idx).copied().unwrap_or(KernelType::Unknown)
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

#[cfg(test)]
mod cb200_kernel_type_index_tests {
    use super::KernelType;

    /// `to_index()`/`from_index()` must round-trip for every real variant,
    /// and every out-of-range index (including the `Unknown` slot itself)
    /// must map to `Unknown`.
    #[test]
    fn kernel_type_index_round_trip_matches_declaration_order() {
        let variants = [
            KernelType::TiledQ4K,
            KernelType::CoalescedQ4K,
            KernelType::VectorizedQ4K,
            KernelType::BatchedQ4K,
            KernelType::Dp4aQ4K,
            KernelType::FusedRmsNormQ4K,
            KernelType::CoalescedQ6K,
            KernelType::IncrementalAttention,
            KernelType::MultiWarpAttention,
            KernelType::BatchedAttention,
            KernelType::RmsNorm,
            KernelType::VectorizedRmsNorm,
            KernelType::BatchedRmsNorm,
            KernelType::FusedQKVHwDp4aQ4KGemv,
            KernelType::Generic,
        ];
        for (expected_idx, variant) in variants.into_iter().enumerate() {
            assert_eq!(variant.to_index(), expected_idx);
            assert_eq!(KernelType::from_index(expected_idx), variant);
        }
        assert_eq!(KernelType::Unknown.to_index(), 15);
        for idx in 15..20 {
            assert_eq!(KernelType::from_index(idx), KernelType::Unknown);
        }
    }
}
