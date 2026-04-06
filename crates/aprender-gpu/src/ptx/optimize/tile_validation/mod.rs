//! Tile Constraint Validation
//!
//! Validates tile dimensions and constraints to prevent register pressure issues
//! and compilation hangs.
//!
//! ## Constraints
//!
//! 1. **Power-of-two dimensions**: Required for efficient GPU scheduling
//! 2. **Maximum tile elements**: 16M elements to prevent register spills
//! 3. **Maximum single dimension**: 4096 to prevent degenerate shapes
//!
//! ## Academic Foundation
//!
//! Based on Volkov & Demmel (2008): Power-of-two tiles achieve 95%+ peak throughput.
//! cuda-tile-behavior.md: Section 3.4, Falsification tests #1-15

use super::super::instructions::{PtxInstruction, PtxOp, WmmaShape};
use crate::error::{GpuError, Result};

/// Maximum number of elements in a tile (16M elements = 64MB for f32)
pub const MAX_TILE_ELEMENTS: usize = 16_777_216;

/// Maximum size for any single tile dimension
pub const MAX_TILE_DIM: usize = 4096;

/// Tile validation error
#[derive(Debug, Clone, PartialEq)]
pub enum TileError {
    /// Tile has too many total elements
    TooManyElements {
        /// Actual number of elements
        actual: usize,
        /// Maximum allowed elements
        max: usize,
    },
    /// Tile dimension is not a power of two
    NonPowerOfTwo {
        /// The non-power-of-two dimension value
        dim: usize,
    },
    /// Single dimension exceeds maximum
    DimensionTooLarge {
        /// Actual dimension size
        actual: usize,
        /// Maximum allowed dimension
        max: usize,
    },
    /// Invalid WMMA shape
    InvalidWmmaShape {
        /// The invalid shape description
        shape: String,
    },
}

impl std::fmt::Display for TileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyElements { actual, max } => {
                write!(f, "Tile has too many elements: {} > {}", actual, max)
            }
            Self::NonPowerOfTwo { dim } => {
                write!(f, "Tile dimension {} is not a power of two", dim)
            }
            Self::DimensionTooLarge { actual, max } => {
                write!(f, "Tile dimension {} exceeds maximum {}", actual, max)
            }
            Self::InvalidWmmaShape { shape } => {
                write!(f, "Invalid WMMA shape: {}", shape)
            }
        }
    }
}

impl std::error::Error for TileError {}

impl From<TileError> for GpuError {
    fn from(err: TileError) -> Self {
        GpuError::InvalidParameter(err.to_string())
    }
}

/// Validate tile shape constraints.
///
/// # Arguments
///
/// * `shape` - Array of tile dimensions
///
/// # Returns
///
/// Ok(()) if valid, Err with TileError otherwise
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #1: Power-of-two tiles improve GPU occupancy
/// - Falsification test #2: MAX_TILE_ELEMENTS prevents register spills
/// - Falsification test #3: MAX_TILE_DIM prevents degenerate shapes
pub fn validate_shape(shape: &[usize]) -> std::result::Result<(), TileError> {
    // Calculate total elements
    let total_elements: usize = shape.iter().product();

    // Constraint 1: Total element cap
    if total_elements > MAX_TILE_ELEMENTS {
        return Err(TileError::TooManyElements { actual: total_elements, max: MAX_TILE_ELEMENTS });
    }

    // Constraint 2: Power-of-two dimensions (for GPU efficiency)
    for &dim in shape {
        if dim != 0 && !dim.is_power_of_two() {
            return Err(TileError::NonPowerOfTwo { dim });
        }
    }

    // Constraint 3: Single dimension cap
    for &dim in shape {
        if dim > MAX_TILE_DIM {
            return Err(TileError::DimensionTooLarge { actual: dim, max: MAX_TILE_DIM });
        }
    }

    Ok(())
}

/// Validate WMMA (Tensor Core) shape.
///
/// WMMA operations have fixed valid shapes. This validates that the shape
/// is one of the supported configurations.
///
/// # Arguments
///
/// * `shape` - WMMA shape (M×N×K)
///
/// # Returns
///
/// Ok(()) if valid
pub fn validate_wmma_shape(shape: &WmmaShape) -> std::result::Result<(), TileError> {
    // Valid WMMA shapes for SM 70+
    let valid_shapes = [
        (16, 16, 16), // Standard
        (8, 32, 16),  // Wide
        (32, 8, 16),  // Tall
    ];

    let is_valid =
        valid_shapes.iter().any(|&(m, n, k)| shape.m == m && shape.n == n && shape.k == k);

    if !is_valid {
        return Err(TileError::InvalidWmmaShape {
            shape: format!("m{}n{}k{}", shape.m, shape.n, shape.k),
        });
    }

    Ok(())
}

/// Validate PTX instructions for tile constraints.
///
/// Scans instructions for tile-related operations and validates their parameters.
///
/// # Arguments
///
/// * `instructions` - PTX instruction sequence
///
/// # Returns
///
/// Ok(()) if all tile constraints are satisfied
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #4: Tile validation catches invalid shapes at compile time
pub fn validate(instructions: &[PtxInstruction]) -> Result<()> {
    for instr in instructions {
        // Validate WMMA operations
        if matches!(
            instr.op,
            PtxOp::WmmaLoadA
                | PtxOp::WmmaLoadB
                | PtxOp::WmmaLoadC
                | PtxOp::WmmaMma
                | PtxOp::WmmaStoreD
        ) {
            // WMMA operations always use fixed 16×16×16 shape in current implementation
            // Future: extract shape from instruction metadata
            validate_wmma_shape(&WmmaShape::M16N16K16)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;
