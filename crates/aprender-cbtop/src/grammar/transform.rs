//! Data transform (Statistics equivalent in Grammar of Graphics).

use super::workload::Operation;

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Symmetric quantization
    Symmetric,
    /// Asymmetric quantization
    Asymmetric,
    /// Block-wise quantization (GGML-style)
    BlockWise { block_size: usize },
}

/// Data transform (analogous to Statistics)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DataTransform {
    /// No transformation
    #[default]
    Identity,
    /// Quantize to lower precision
    Quantize { bits: u8, scheme: QuantScheme },
    /// Tile for cache efficiency
    Tile { tile_size: usize },
    /// Transpose for memory layout
    Transpose { order: Vec<usize> },
    /// Pad for alignment
    Pad { alignment: usize },
    /// Fuse multiple operations
    Fuse { ops: Vec<Operation> },
}

impl DataTransform {
    /// Create identity transform
    pub fn identity() -> Self {
        DataTransform::Identity
    }

    /// Create tiling transform
    pub fn tile(size: usize) -> Self {
        DataTransform::Tile { tile_size: size }
    }

    /// Create quantization transform
    pub fn quantize(bits: u8) -> Self {
        DataTransform::Quantize {
            bits,
            scheme: QuantScheme::Symmetric,
        }
    }

    /// Create padding transform
    pub fn pad(alignment: usize) -> Self {
        DataTransform::Pad { alignment }
    }
}
