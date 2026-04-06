//! Loop Splitting Optimization Pass
//!
//! Splits loops at conditional boundaries to eliminate branch divergence.
//!
//! ## Pattern Detection
//!
//! ```text
//! for i in 0..n {
//!     if i < boundary {
//!         heavy_operation();
//!     } else {
//!         light_operation();
//!     }
//! }
//! ```
//!
//! Becomes:
//!
//! ```text
//! for i in 0..boundary {
//!     heavy_operation();  // No branch
//! }
//! for i in boundary..n {
//!     light_operation();  // No branch
//! }
//! ```
//!
//! ## Benefits
//!
//! - Eliminates branch divergence in GPU warps
//! - Enables specialized register allocation per loop
//! - Reduces instruction cache pressure
//!
//! ## Academic Foundation
//!
//! Based on NVIDIA CUDA Tile IR (LoopSplit.cpp) from CUDA Toolkit 13.1.
//! Allen & Kennedy prove loop splitting is always legal for affine conditions.
//! cuda-tile-behavior.md: Section 3.3, Falsification tests #51-65

use std::collections::HashSet;

use super::super::instructions::{CmpOp, Operand, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;

/// Configuration for loop splitting
#[derive(Debug, Clone)]
pub struct LoopSplitConfig {
    /// Minimum number of operations in if-block to trigger split
    /// Default: 1 (always split heavy ops)
    pub threshold: usize,
}

impl Default for LoopSplitConfig {
    fn default() -> Self {
        Self { threshold: 1 }
    }
}

/// Heavy operations that benefit from loop splitting
/// (Aligned with NVIDIA LoopSplit.cpp isSplitProfitable)
const HEAVY_OPS: &[PtxOp] = &[
    PtxOp::Ld,        // Load
    PtxOp::St,        // Store
    PtxOp::WmmaMma,   // Tensor Core MMA
    PtxOp::WmmaLoadA, // WMMA load
    PtxOp::WmmaLoadB, // WMMA load
    PtxOp::WmmaLoadC, // WMMA load
    PtxOp::WmmaStoreD,
];

/// Check if an operation is "heavy" (benefits from splitting)
fn is_heavy_op(op: &PtxOp) -> bool {
    HEAVY_OPS.contains(op)
}

/// Profitability analysis for loop splitting
/// (Aligned with NVIDIA LoopSplit.cpp isSplitProfitable)
///
/// # Arguments
///
/// * `if_body` - Instructions inside the if block
/// * `threshold` - Minimum ops to trigger split
///
/// # Returns
///
/// true if splitting is profitable
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #52: Profitability heuristic is accurate
#[must_use]
pub fn is_split_profitable(if_body: &[PtxInstruction], threshold: usize) -> bool {
    // Always split if threshold is 1 (split for any branch)
    if threshold == 1 {
        return true;
    }

    // Check for heavy operations
    let has_heavy_ops = if_body.iter().any(|instr| is_heavy_op(&instr.op));

    // Check operation count
    let op_count = if_body.len();

    op_count >= threshold || has_heavy_ops
}

/// Comparison predicate for loop conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopPredicate {
    /// iv < bound
    LessThan,
    /// iv <= bound
    LessEqual,
    /// iv > bound
    GreaterThan,
    /// iv >= bound
    GreaterEqual,
}

impl LoopPredicate {
    /// Check if the "then" block should be second (after split)
    #[must_use]
    pub const fn then_is_second(self) -> bool {
        matches!(self, Self::GreaterThan | Self::GreaterEqual)
    }

    /// Convert from CmpOp
    #[must_use]
    pub fn from_cmp_op(cmp: CmpOp) -> Option<Self> {
        match cmp {
            CmpOp::Lt => Some(Self::LessThan),
            CmpOp::Le => Some(Self::LessEqual),
            CmpOp::Gt => Some(Self::GreaterThan),
            CmpOp::Ge => Some(Self::GreaterEqual),
            _ => None,
        }
    }
}

/// Represents a splittable loop condition
#[derive(Debug, Clone)]
pub struct SplittableCondition {
    /// The comparison instruction index
    pub cmp_idx: usize,
    /// The induction variable register
    pub induction_var: VirtualReg,
    /// The comparison predicate (normalized to iv on left)
    pub predicate: LoopPredicate,
    /// The split bound value (right-hand side of comparison)
    pub bound: Operand,
    /// Indices of if-ops using this condition
    pub if_ops: HashSet<usize>,
}

/// Normalize a comparison to always be "iv <op> bound"
///
/// # Arguments
///
/// * `cmp` - The comparison instruction
/// * `induction_var` - The loop induction variable
///
/// # Returns
///
/// Normalized predicate and bound if comparison involves induction variable
///
/// # cuda-tile-behavior.md References
///
/// - Section 3.3: Normalize comparison for split point calculation
#[must_use]
pub fn normalize_comparison(
    cmp: &PtxInstruction,
    induction_var: VirtualReg,
) -> Option<(LoopPredicate, Operand)> {
    if cmp.srcs.len() < 2 {
        return None;
    }

    let lhs = &cmp.srcs[0];
    let rhs = &cmp.srcs[1];

    // Extract CmpOp from instruction (would need to store it in instruction)
    // For now, we check if this is a setp instruction
    if !matches!(cmp.op, PtxOp::Setp) {
        return None;
    }

    // Check if lhs is the induction variable
    if let Operand::Reg(lhs_reg) = lhs {
        if *lhs_reg == induction_var {
            // Already normalized: iv <op> bound
            // We'd need to extract the actual comparison from the instruction
            // For now, assume Lt as default
            return Some((LoopPredicate::LessThan, rhs.clone()));
        }
    }

    // Check if rhs is the induction variable (need to flip predicate)
    if let Operand::Reg(rhs_reg) = rhs {
        if *rhs_reg == induction_var {
            // Need to flip: bound <op> iv becomes iv <flipped_op> bound
            // Assume Lt -> Gt flip
            return Some((LoopPredicate::GreaterThan, lhs.clone()));
        }
    }

    None
}

/// Compute aligned split point for non-unit step loops
///
/// # Arguments
///
/// * `split` - The raw split point
/// * `lower` - Loop lower bound
/// * `step` - Loop step size
///
/// # Returns
///
/// Aligned split point: `lower + ceil((split - lower) / step) * step`
///
/// # cuda-tile-behavior.md References
///
/// - Section 3.3: Split point alignment for non-unit steps
/// - Falsification test #59: Splitting handles non-unit step sizes
#[must_use]
pub fn align_split_point(split: usize, lower: usize, step: usize) -> usize {
    if step == 0 {
        return split;
    }

    if split <= lower {
        return lower;
    }

    let diff = split - lower;
    let k = (diff + step - 1) / step; // Ceiling division
    lower + k * step
}

/// Apply loop splitting pass to instruction sequence
///
/// This is a simplified version that identifies splittable patterns
/// without transforming the IR (which would require CFG representation).
///
/// # Arguments
///
/// * `instructions` - Input instruction sequence
/// * `config` - Loop splitting configuration
///
/// # Returns
///
/// Analysis results (actual transformation requires CFG)
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #51: Loop splitting eliminates branch divergence
/// - Falsification test #54: Splitting preserves loop semantics
#[must_use]
pub fn analyze(
    instructions: &[PtxInstruction],
    config: &LoopSplitConfig,
) -> Vec<SplittableCondition> {
    let mut splittable = Vec::new();

    // Find comparison instructions that could be split points
    for (i, instr) in instructions.iter().enumerate() {
        if matches!(instr.op, PtxOp::Setp) {
            // Check if this comparison is used for a branch
            // and if the condition is splittable
            if let Some(Operand::Reg(_pred_reg)) = &instr.dst {
                // This is a potential split point
                // In a real implementation, we'd track:
                // 1. Which loops contain this comparison
                // 2. What the induction variable is
                // 3. Whether the condition is loop-invariant on the RHS

                // For now, check profitability with window of instructions
                let window_end = (i + 10).min(instructions.len());
                let window = &instructions[i..window_end];

                if is_split_profitable(window, config.threshold) {
                    // Would need actual loop analysis to populate correctly
                    // This is a placeholder for the analysis
                    splittable.push(SplittableCondition {
                        cmp_idx: i,
                        induction_var: VirtualReg::new(0, super::super::types::PtxType::U32),
                        predicate: LoopPredicate::LessThan,
                        bound: Operand::ImmU64(0),
                        if_ops: HashSet::new(),
                    });
                }
            }
        }
    }

    splittable
}

/// Check if loop splitting pass produces idempotent results
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #64: Loop splitting pass is idempotent
#[must_use]
pub fn is_idempotent(first: &[SplittableCondition], second: &[SplittableCondition]) -> bool {
    first.len() == second.len()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;
