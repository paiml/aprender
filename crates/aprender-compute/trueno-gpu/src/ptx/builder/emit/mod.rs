//! PTX Instruction Emission
//!
//! This module contains all PTX code generation functions that convert
//! PtxInstruction structs into PTX assembly text.
//!
//! ## Architecture (PMAT-018 Shatter)
//!
//! The emission logic is split by operation category:
//! - `arithmetic.rs` - add, sub, mul, div, fma, transcendental ops
//! - `memory.rs` - ld, st, cvt, cvta, atomic ops
//! - `control.rs` - bra, ret, bar, membar, setp
//! - `warp.rs` - shfl, vote, bit manipulation
//! - `wmma.rs` - tensor core operations
//! - `operand.rs` - shared operand formatting
//!
//! ## Functions
//!
//! - `emit_instruction()` - Allocating version for single instruction
//! - `write_instruction()` - Zero-allocation version for bulk output

mod arithmetic;
mod control;
mod memory;
mod operand;
mod warp;
mod wmma;

use std::fmt::Write;

use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};
use crate::ptx::types::{PtxStateSpace, PtxType};

// Re-export operand functions for external use (wmma.rs, instruction_emission tests)
#[allow(unused_imports)]
pub(crate) use operand::{
    emit_global_mem_operand, emit_operand, emit_shared_mem_operand, write_mem_operand,
    write_operand,
};

/// Try to emit a label line, returning `Some(label_line)` if the instruction is a label.
fn try_emit_label(instr: &PtxInstruction) -> Option<String> {
    let label = instr.label.as_ref()?;
    if label.ends_with(':') {
        Some(format!("{}:\n", &label[..label.len() - 1]))
    } else {
        None
    }
}

/// Build the predicate prefix string for an instruction.
fn build_predicate_prefix(instr: &PtxInstruction) -> String {
    match &instr.predicate {
        Some(pred) => {
            let neg = if pred.negated { "!" } else { "" };
            format!("    @{}{} ", neg, pred.reg.to_ptx_string())
        }
        None => "    ".to_string(),
    }
}

/// Dispatch a WMMA op to the appropriate emitter.
fn emit_wmma_dispatch(s: String, instr: &PtxInstruction) -> String {
    match instr.op {
        PtxOp::WmmaLoadA => wmma::emit_wmma_load(s, instr, "a"),
        PtxOp::WmmaLoadB => wmma::emit_wmma_load(s, instr, "b"),
        PtxOp::WmmaLoadC => wmma::emit_wmma_load(s, instr, "c"),
        PtxOp::WmmaMma => wmma::emit_wmma_mma(s, instr),
        PtxOp::WmmaStoreD => wmma::emit_wmma_store(s, instr),
        PtxOp::MmaSync => wmma::emit_mma_sync(s, instr),
        PtxOp::LdMatrix => wmma::emit_ldmatrix(s, instr),
        PtxOp::LdMatrixTrans => wmma::emit_ldmatrix_trans(s, instr),
        _ => s,
    }
}

/// Emit opcode, type suffix, operands, and terminating semicolon.
fn emit_standard_body(instr: &PtxInstruction, out: &mut String) {
    emit_opcode(instr, out);

    if !should_skip_type_suffix(instr) {
        out.push_str(instr.ty.to_ptx_string());
    }

    out.push(' ');
    write_destinations(instr, out);
    write_sources(instr, out);
    out.push_str(";\n");
}

/// Emit a single instruction as PTX (allocating version)
pub(crate) fn emit_instruction(instr: &PtxInstruction) -> String {
    if let Some(label_line) = try_emit_label(instr) {
        return label_line;
    }

    let prefix = build_predicate_prefix(instr);
    let mut s = prefix.clone();

    if control::is_early_return_op(&instr.op) {
        if let Some(result) = control::emit_control_opcode(instr, &prefix) {
            return result;
        }
    }

    if wmma::is_wmma_op(&instr.op) {
        return emit_wmma_dispatch(s, instr);
    }

    // cp.async ops format themselves entirely (no type suffix or operand append)
    if matches!(instr.op, PtxOp::CpAsync | PtxOp::CpAsyncCommitGroup | PtxOp::CpAsyncWaitGroup) {
        memory::emit_memory_opcode(instr, &mut s);
        s.push_str(";\n");
        return s;
    }

    emit_standard_body(instr, &mut s);
    s
}

/// Check if type suffix should be skipped
fn should_skip_type_suffix(instr: &PtxInstruction) -> bool {
    // Wide mul from u32 sources
    let is_wide_mul_from_u32 = instr.op == PtxOp::Mul
        && (instr.ty == crate::ptx::types::PtxType::U64
            || instr.ty == crate::ptx::types::PtxType::S64)
        && !instr.srcs.first().is_some_and(|src| {
            matches!(src, Operand::Reg(vreg)
                if vreg.ty() == crate::ptx::types::PtxType::U64
                || vreg.ty() == crate::ptx::types::PtxType::S64)
        });

    // DP4A opcodes already include their type qualifiers (e.g., "dp4a.u32.s32")
    let is_dp4a = matches!(instr.op, PtxOp::Dp4a | PtxOp::Dp4aUS | PtxOp::Dp4aS32);

    memory::skip_type_for_memory_op(&instr.op)
        || warp::skip_type_for_warp_op(&instr.op)
        || wmma::is_wmma_op(&instr.op)
        || is_wide_mul_from_u32
        || is_dp4a
}

/// Emit the opcode portion of an instruction (shared by both emit and write paths)
fn emit_opcode(instr: &PtxInstruction, out: &mut String) {
    if arithmetic::is_arithmetic_op(&instr.op) {
        arithmetic::emit_arithmetic_opcode(instr, out);
    } else if memory::is_memory_op(&instr.op) {
        memory::emit_memory_opcode(instr, out);
    } else if instr.op == PtxOp::Setp {
        control::emit_setp_opcode(instr, out);
    } else if warp::is_warp_op(&instr.op) {
        warp::emit_warp_opcode(&instr.op, out);
    } else {
        let op_str = format!("{:?}", instr.op).to_lowercase();
        out.push_str(&op_str);
    }
}

/// Write destination operands directly to buffer
fn write_destinations(instr: &PtxInstruction, out: &mut String) {
    if !instr.dsts.is_empty() {
        out.push('{');
        for (i, dst) in instr.dsts.iter().enumerate() {
            write_operand(dst, out);
            if i < instr.dsts.len() - 1 {
                out.push_str(", ");
            }
        }
        out.push('}');
        if !instr.srcs.is_empty() {
            out.push_str(", ");
        }
    } else if let Some(dst) = &instr.dst {
        write_operand(dst, out);
        if !instr.srcs.is_empty() {
            out.push_str(", ");
        }
    }
}

/// Write source operands with proper memory addressing
fn write_sources(instr: &PtxInstruction, out: &mut String) {
    let is_memory_op =
        matches!(instr.op, PtxOp::Ld | PtxOp::LdVolatile | PtxOp::St | PtxOp::Prefetch);
    let is_atomic_op = matches!(
        instr.op,
        PtxOp::AtomAdd | PtxOp::AtomMin | PtxOp::AtomMax | PtxOp::AtomExch | PtxOp::AtomCas
    );
    let is_shared_mem = instr.state_space == Some(PtxStateSpace::Shared);
    let is_global_mem = instr.state_space == Some(PtxStateSpace::Global)
        || (is_memory_op && instr.state_space.is_none());

    // Vector stores (st.global.v2/v4) need braces around data operands:
    //   st.global.v2.f32 [addr], {val0, val1};
    let is_vector_store =
        instr.op == PtxOp::St && matches!(instr.ty, PtxType::V2F32 | PtxType::V4F32);

    for (i, src) in instr.srcs.iter().enumerate() {
        if i == 0 && (is_memory_op || is_atomic_op) {
            if is_shared_mem || is_global_mem || is_atomic_op {
                write_mem_operand(src, out);
            } else {
                write_operand(src, out);
            }
        } else {
            // For vector store: emit opening brace before first data operand
            if is_vector_store && i == 1 {
                out.push('{');
            }
            write_operand(src, out);
            // For vector store: emit closing brace after last data operand
            if is_vector_store && i == instr.srcs.len() - 1 {
                out.push('}');
            }
        }
        if i < instr.srcs.len() - 1 {
            out.push_str(", ");
        }
    }
}

/// Write the predicate prefix and return whether a prefix was written
fn write_predicate_prefix(instr: &PtxInstruction, out: &mut String) {
    if let Some(pred) = &instr.predicate {
        let neg = if pred.negated { "!" } else { "" };
        let _ = write!(out, "    @{}{} ", neg, pred.reg);
    } else {
        out.push_str("    ");
    }
}

/// Try to write a label line directly to the output buffer.
/// Returns `true` if a label was written (caller should return early).
fn try_write_label(instr: &PtxInstruction, out: &mut String) -> bool {
    if let Some(label) = &instr.label {
        if label.ends_with(':') {
            let _ = writeln!(out, "{}:", &label[..label.len() - 1]);
            return true;
        }
    }
    false
}

/// Try to handle an early-return control op in the write path.
/// Returns `true` if the op was fully handled (caller should return early).
fn try_write_control_op(instr: &PtxInstruction, out: &mut String) -> bool {
    if !control::is_early_return_op(&instr.op) {
        return false;
    }
    let prefix = build_predicate_prefix(instr);
    if let Some(result) = control::emit_control_opcode(instr, &prefix) {
        // Skip prefix portion — already written by write_predicate_prefix
        out.push_str(&result[prefix.len()..]);
        return true;
    }
    false
}

/// Write a single instruction directly to a String buffer (zero intermediate allocations)
pub(super) fn write_instruction(instr: &PtxInstruction, out: &mut String) {
    if try_write_label(instr, out) {
        return;
    }

    write_predicate_prefix(instr, out);

    if try_write_control_op(instr, out) {
        return;
    }

    if wmma::is_wmma_op(&instr.op) {
        out.push_str(&emit_instruction(instr));
        return;
    }

    // cp.async ops format themselves entirely
    if matches!(instr.op, PtxOp::CpAsync | PtxOp::CpAsyncCommitGroup | PtxOp::CpAsyncWaitGroup) {
        memory::emit_memory_opcode(instr, out);
        out.push_str(";\n");
        return;
    }

    emit_standard_body(instr, out);
}

#[cfg(test)]
mod tests;
