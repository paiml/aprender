//! PTX Optimization Passes
//!
//! Inspired by NVIDIA CUDA Tile IR patterns (spec: cuda-tile-behavior.md v1.2.1).
//!
//! ## Available Passes
//!
//! - **FMA Fusion**: Detect `mul` + `add` patterns and fuse to `fma`
//! - **Tile Validation**: Validate tile constraints to prevent register pressure issues
//! - **Loop Splitting**: Split loops at conditional boundaries for GPU efficiency
//! - **Token-Based Ordering (TKO)**: Memory dependency tracking for barrier elimination
//! - **Barrier Safety**: Static analysis to detect PARITY-114 early-exit-before-barrier bugs
//!
//! ## Usage
//!
//! ```rust,ignore
//! use trueno_gpu::ptx::optimize::{fma_fusion, tile_validation, loop_split, tko, barrier_safety};
//!
//! let instructions = vec![/* PTX instructions */];
//! let fused = fma_fusion::pass(instructions);
//! tile_validation::validate(&fused)?;
//!
//! // Analyze for loop splitting opportunities
//! let splits = loop_split::analyze(&fused, &loop_split::LoopSplitConfig::default());
//!
//! // Track memory dependencies with tokens
//! let t1 = tko::Token::new();
//! let t2 = tko::Token::new();
//! let joined = tko::join_tokens(&[t1, t2]);
//!
//! // Validate barrier safety (PARITY-114 prevention)
//! let ptx_source = "...";
//! barrier_safety::validate(ptx_source)?;
//! ```
//!
//! ## Academic Foundation
//!
//! - FMA fusion based on Click & Paleczny (1995) SSA pattern matching
//! - Tile constraints based on Volkov & Demmel (2008) GPU optimization
//! - Loop splitting from NVIDIA CUDA Tile IR LoopSplit.cpp
//! - Token-based ordering from NVIDIA CUDA Tile IR memory consistency model
//! - Barrier safety from PARITY-114 Five Whys analysis (2026)

pub mod barrier_safety;
pub mod fma_fusion;
pub mod loop_split;
pub mod tile_validation;
pub mod tko;

use super::instructions::PtxInstruction;
use crate::error::Result;

/// Apply all optimization passes to a sequence of PTX instructions.
///
/// # Arguments
///
/// * `instructions` - The PTX instructions to optimize
///
/// # Returns
///
/// Optimized instruction sequence
///
/// # cuda-tile-behavior.md References
///
/// - Section 3.5: FMA Fusion Detection
/// - Section 3.4: Tile Dimension Constraints
pub fn optimize(instructions: Vec<PtxInstruction>) -> Result<Vec<PtxInstruction>> {
    // Pass 1: FMA fusion
    let fused = fma_fusion::pass(instructions);

    // Pass 2: Tile validation (just validates, doesn't transform)
    tile_validation::validate(&fused)?;

    Ok(fused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::instructions::PtxOp;
    use crate::ptx::types::PtxType;

    #[test]
    fn test_optimize_empty() {
        let instructions = vec![];
        let result = optimize(instructions).expect("test");
        assert!(result.is_empty());
    }

    #[test]
    fn test_optimize_passthrough() {
        // Instructions that shouldn't be modified
        let instructions = vec![PtxInstruction::new(PtxOp::Ret, PtxType::Pred)];
        let result = optimize(instructions).expect("test");
        assert_eq!(result.len(), 1);
    }
}
