#![allow(missing_docs)]
//! Tiling error types.

use super::geometry::TcbLevel;
use std::fmt;

/// Tiling configuration errors
#[derive(Debug, Clone)]
pub enum TilingError {
    /// Tile hierarchy is invalid (e.g., micro > midi)
    InvalidHierarchy { reason: String },
    /// Tile dimensions not divisible
    DivisibilityError { level: &'static str, dimension: &'static str, larger: u32, smaller: u32 },
    /// Tile doesn't fit in cache
    CacheOverflow { level: TcbLevel, required_bytes: usize, available_bytes: usize },
    /// Alignment violation
    AlignmentError { required: u32, actual: u32 },
    /// Quantization alignment violated
    QuantAlignmentError { format: &'static str, required_k: u32, actual_k: u32 },
}

impl fmt::Display for TilingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TilingError::InvalidHierarchy { reason } => {
                write!(f, "Invalid tiling hierarchy: {}", reason)
            }
            TilingError::DivisibilityError { level, dimension, larger, smaller } => {
                write!(
                    f,
                    "Tiling divisibility error at {}: {} ({}) not divisible by {}",
                    level, dimension, larger, smaller
                )
            }
            TilingError::CacheOverflow { level, required_bytes, available_bytes } => {
                write!(
                    f,
                    "Tile exceeds {:?} cache: {} bytes required, {} available",
                    level, required_bytes, available_bytes
                )
            }
            TilingError::AlignmentError { required, actual } => {
                write!(f, "Alignment error: required {} bytes, actual {} bytes", required, actual)
            }
            TilingError::QuantAlignmentError { format, required_k, actual_k } => {
                write!(
                    f,
                    "Quantization alignment error for {}: K must be multiple of {}, got {}",
                    format, required_k, actual_k
                )
            }
        }
    }
}

impl std::error::Error for TilingError {}
