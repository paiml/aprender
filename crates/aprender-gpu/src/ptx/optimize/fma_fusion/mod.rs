//! FMA Fusion Optimization Pass
//!
//! Detects `mul` + `add` patterns and fuses them to single `fma` instructions.
//!
//! ## Pattern Detection
//!
//! ```text
//! mul.f32 %f1, %a, %b    ; %f1 = a * b
//! add.f32 %f2, %f1, %c   ; %f2 = %f1 + c = a * b + c
//! ```
//!
//! Becomes:
//!
//! ```text
//! fma.rn.f32 %f2, %a, %b, %c  ; %f2 = a * b + c (single instruction)
//! ```
//!
//! ## Requirements for Fusion
//!
//! 1. `mul` result must have exactly one use (in the `add`)
//! 2. `mul` and `add` must have compatible rounding modes
//! 3. Both instructions must be f32 or f64 type
//!
//! ## Academic Foundation
//!
//! Based on Click & Paleczny (1995) SSA pattern matching for peephole optimization.
//! cuda-tile-behavior.md: Section 3.5, Falsification tests #16-30

use std::collections::HashMap;

use super::super::instructions::{Operand, PtxInstruction, PtxOp, RoundingMode};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;

/// Apply FMA fusion pass to instruction sequence.
///
/// # Arguments
///
/// * `instructions` - Input instruction sequence
///
/// # Returns
///
/// Optimized instruction sequence with mul+add fused to fma
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #16: FMA reduces instruction count by ~33%
/// - Falsification test #17: FMA improves numerical accuracy
/// - Falsification test #18: Single-use detection prevents incorrect fusion
#[must_use]
pub fn pass(instructions: Vec<PtxInstruction>) -> Vec<PtxInstruction> {
    if instructions.is_empty() {
        return instructions;
    }

    // Build use-def chains: for each virtual register, track which instruction defines it
    // and how many times it's used
    let use_counts = count_register_uses(&instructions);
    let definitions = build_def_map(&instructions);

    // Phase 1: Find all fusion pairs (mul_idx -> fma replacement for add_idx)
    let mut fused_muls: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut fma_replacements: HashMap<usize, PtxInstruction> = HashMap::new();

    for (add_idx, _) in instructions.iter().enumerate() {
        if let Some((fma, mul_idx)) =
            try_fuse_mul_add(add_idx, &instructions, &use_counts, &definitions)
        {
            fused_muls.insert(mul_idx);
            fma_replacements.insert(add_idx, fma);
        }
    }

    // Phase 2: Emit optimized instruction sequence
    let mut result = Vec::with_capacity(instructions.len() - fused_muls.len());

    for (i, instr) in instructions.iter().enumerate() {
        // Skip mul instructions that were fused into FMAs
        if fused_muls.contains(&i) {
            continue;
        }

        // Replace add instructions with their FMA equivalents
        if let Some(fma) = fma_replacements.get(&i) {
            result.push(fma.clone());
        } else {
            result.push(instr.clone());
        }
    }

    result
}

/// Count how many times each virtual register is used as a source operand.
fn count_register_uses(instructions: &[PtxInstruction]) -> HashMap<VirtualReg, usize> {
    let mut counts = HashMap::new();

    for instr in instructions {
        for src in &instr.srcs {
            if let Operand::Reg(reg) = src {
                *counts.entry(*reg).or_insert(0) += 1;
            }
        }
        if let Some(Operand::Reg(reg)) = &instr.predicate.as_ref().map(|p| Operand::Reg(p.reg)) {
            *counts.entry(*reg).or_insert(0) += 1;
        }
    }

    counts
}

/// Build a map from virtual register to the instruction index that defines it.
fn build_def_map(instructions: &[PtxInstruction]) -> HashMap<VirtualReg, usize> {
    let mut defs = HashMap::new();

    for (i, instr) in instructions.iter().enumerate() {
        if let Some(Operand::Reg(reg)) = &instr.dst {
            defs.insert(*reg, i);
        }
    }

    defs
}

/// Try to fuse an add instruction with its defining mul.
///
/// Returns the fused FMA instruction and the index of the mul definition if fusion is possible.
fn try_fuse_mul_add(
    add_idx: usize,
    instructions: &[PtxInstruction],
    use_counts: &HashMap<VirtualReg, usize>,
    definitions: &HashMap<VirtualReg, usize>,
) -> Option<(PtxInstruction, usize)> {
    let add_instr = &instructions[add_idx];

    // Only fuse floating-point add operations
    if !matches!(add_instr.op, PtxOp::Add) {
        return None;
    }
    if !matches!(add_instr.ty, PtxType::F32 | PtxType::F64) {
        return None;
    }

    // Check each source operand of the add to see if it's a fusable mul result
    for (src_idx, src) in add_instr.srcs.iter().enumerate() {
        let Operand::Reg(mul_result) = src else {
            continue;
        };

        if let Some(pair) = try_fuse_source(
            add_instr,
            src_idx,
            *mul_result,
            instructions,
            use_counts,
            definitions,
        ) {
            return Some(pair);
        }
    }

    None
}

/// Check if a specific source operand of an add can be fused with its defining mul.
fn try_fuse_source(
    add_instr: &PtxInstruction,
    src_idx: usize,
    mul_result: VirtualReg,
    instructions: &[PtxInstruction],
    use_counts: &HashMap<VirtualReg, usize>,
    definitions: &HashMap<VirtualReg, usize>,
) -> Option<(PtxInstruction, usize)> {
    // Register must have exactly one use (in this add)
    if use_counts.get(&mul_result) != Some(&1) {
        return None;
    }

    let &def_idx = definitions.get(&mul_result)?;
    let mul_instr = &instructions[def_idx];

    // Validate the defining instruction is a compatible mul
    if !is_fusable_mul(mul_instr, add_instr) {
        return None;
    }

    // Get the other operand of the add (the 'c' in a*b+c)
    let other_src = if src_idx == 0 {
        add_instr.srcs.get(1)?
    } else {
        add_instr.srcs.first()?
    };

    // Get mul operands (a and b) — need at least 2 sources
    let a = mul_instr.srcs.first()?;
    let b = mul_instr.srcs.get(1)?;

    // Create FMA instruction: dst = a * b + c
    let fma = PtxInstruction::new(PtxOp::Fma, add_instr.ty.clone())
        .dst(add_instr.dst.clone()?)
        .src(a.clone())
        .src(b.clone())
        .src(other_src.clone())
        .rounding(mul_instr.rounding.unwrap_or(RoundingMode::Rn));

    Some((fma, def_idx))
}

/// Check if a mul instruction is compatible for fusion with an add instruction.
fn is_fusable_mul(mul_instr: &PtxInstruction, add_instr: &PtxInstruction) -> bool {
    matches!(mul_instr.op, PtxOp::Mul)
        && mul_instr.ty == add_instr.ty
        && mul_instr.srcs.len() >= 2
        && rounding_modes_compatible(mul_instr.rounding.as_ref(), add_instr.rounding.as_ref())
}

/// Check if two rounding modes are compatible for fusion.
///
/// Fusion is allowed if:
/// - Both are None (default rounding)
/// - Both have the same explicit mode
/// - One is None and the other is Rn (default)
fn rounding_modes_compatible(a: Option<&RoundingMode>, b: Option<&RoundingMode>) -> bool {
    match (a, b) {
        (None | Some(RoundingMode::Rn), None) | (None, Some(RoundingMode::Rn)) => true,
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests;
