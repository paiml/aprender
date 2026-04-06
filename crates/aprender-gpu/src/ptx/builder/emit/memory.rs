//! Memory operation emission
//!
//! Handles: Ld, LdVolatile, LdParam, St, Cvt, Cvta, Atom* variants

use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};
use crate::ptx::types::PtxType;
use std::fmt::Write;

/// Emit memory opcode to the output string
pub(crate) fn emit_memory_opcode(instr: &PtxInstruction, s: &mut String) {
    match instr.op {
        PtxOp::Ld => {
            if let Some(ss) = instr.state_space {
                s.push_str("ld");
                s.push_str(ss.to_ptx_string());
            } else {
                s.push_str("ld");
            }
        }
        PtxOp::LdVolatile => {
            if let Some(ss) = instr.state_space {
                s.push_str("ld.volatile");
                s.push_str(ss.to_ptx_string());
            } else {
                s.push_str("ld.volatile");
            }
        }
        PtxOp::LdParam => s.push_str("ld.param"),
        PtxOp::St => {
            if let Some(ss) = instr.state_space {
                s.push_str("st");
                s.push_str(ss.to_ptx_string());
            } else {
                s.push_str("st");
            }
        }
        PtxOp::Cvt => emit_cvt_opcode(instr, s),
        PtxOp::Cvta => {
            let space = instr.state_space.map(|ss| ss.to_ptx_string()).unwrap_or(".shared");
            let ty = instr.ty.to_ptx_string();
            s.push_str("cvta");
            s.push_str(space);
            s.push_str(ty);
        }
        PtxOp::Prefetch => {
            s.push_str("prefetch.global.L2");
        }
        PtxOp::CpAsync => {
            // cp.async.ca.shared.global [dst], [src], size;
            // srcs[0] = shared dst addr, srcs[1] = global src addr, srcs[2] = size imm
            let size = instr
                .srcs
                .get(2)
                .and_then(|op| {
                    if let Operand::ImmU64(v) = op {
                        Some(*v as u32)
                    } else if let Operand::ImmI64(v) = op {
                        Some(*v as u32)
                    } else {
                        None
                    }
                })
                .unwrap_or(16);
            let dst = instr.srcs.first().map(super::operand::emit_operand).unwrap_or_default();
            let src = instr.srcs.get(1).map(super::operand::emit_operand).unwrap_or_default();
            let _ = write!(s, "cp.async.ca.shared.global [{dst}], [{src}], {size}");
        }
        PtxOp::CpAsyncCommitGroup => {
            s.push_str("cp.async.commit_group");
        }
        PtxOp::CpAsyncWaitGroup => {
            let n = instr
                .srcs
                .first()
                .and_then(|op| {
                    if let Operand::ImmU64(v) = op {
                        Some(*v as u32)
                    } else if let Operand::ImmI64(v) = op {
                        Some(*v as u32)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let _ = write!(s, "cp.async.wait_group {n}");
        }
        PtxOp::AtomAdd => emit_atomic_opcode(instr, s, "add"),
        PtxOp::AtomMin => emit_atomic_opcode(instr, s, "min"),
        PtxOp::AtomMax => emit_atomic_opcode(instr, s, "max"),
        PtxOp::AtomExch => emit_atomic_opcode(instr, s, "exch"),
        PtxOp::AtomCas => emit_atomic_opcode(instr, s, "cas"),
        _ => {}
    }
}

/// Emit cvt opcode with proper type conversion handling
fn emit_cvt_opcode(instr: &PtxInstruction, s: &mut String) {
    let dst_ty = instr.ty.to_ptx_string();
    let src_ty = if let Some(st) = instr.src_type {
        st.to_ptx_string()
    } else if let Some(Operand::Reg(vreg)) = instr.srcs.first() {
        vreg.ty().to_ptx_string()
    } else {
        ".u32"
    };

    let actual_src_type = instr.src_type.unwrap_or_else(|| {
        instr
            .srcs
            .first()
            .and_then(|src| if let Operand::Reg(vreg) = src { Some(vreg.ty()) } else { None })
            .unwrap_or(PtxType::U32)
    });

    let src_is_f16 = actual_src_type == PtxType::F16;
    let dst_is_f32 = instr.ty == PtxType::F32;
    let is_f16_to_f32 = src_is_f16 && dst_is_f32;

    // GH-561: f32→f64 promotion is exact — no rounding modifier needed.
    // Some CUDA drivers reject `cvt.rn.f64.f32` as invalid PTX.
    let src_is_f32 = actual_src_type == PtxType::F32;
    let dst_is_f64 = instr.ty == PtxType::F64;
    let is_f32_to_f64 = src_is_f32 && dst_is_f64;

    let needs_rounding = !is_f16_to_f32
        && !is_f32_to_f64
        && (instr.ty.is_float()
            || instr
                .srcs
                .first()
                .is_some_and(|src| matches!(src, Operand::Reg(vreg) if vreg.ty().is_float())));

    let round = if needs_rounding {
        instr.rounding.as_ref().map_or(".rn", |r| r.to_ptx_string())
    } else {
        ""
    };

    s.push_str("cvt");
    s.push_str(round);
    s.push_str(dst_ty);
    s.push_str(src_ty);
}

/// Emit atomic operation opcode
fn emit_atomic_opcode(instr: &PtxInstruction, s: &mut String, op: &str) {
    let space = instr.state_space.map(|ss| ss.to_ptx_string()).unwrap_or(".global");
    let _ = write!(s, "atom{}.{}", space, op);
}

/// Check if this is a memory operation
pub(crate) fn is_memory_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::Ld
            | PtxOp::LdVolatile
            | PtxOp::LdParam
            | PtxOp::St
            | PtxOp::Cvt
            | PtxOp::Cvta
            | PtxOp::AtomAdd
            | PtxOp::AtomMin
            | PtxOp::AtomMax
            | PtxOp::AtomExch
            | PtxOp::AtomCas
            | PtxOp::Prefetch
            | PtxOp::CpAsync
            | PtxOp::CpAsyncCommitGroup
            | PtxOp::CpAsyncWaitGroup
    )
}

/// Check if this op requires skipping the type suffix
pub(crate) fn skip_type_for_memory_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::Cvt
            | PtxOp::Cvta
            | PtxOp::Prefetch
            | PtxOp::CpAsync
            | PtxOp::CpAsyncCommitGroup
            | PtxOp::CpAsyncWaitGroup
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::instructions::{Operand, RoundingMode};
    use crate::ptx::registers::VirtualReg;
    use crate::ptx::types::{PtxStateSpace, PtxType};

    fn make_instr(op: PtxOp, ty: PtxType) -> PtxInstruction {
        PtxInstruction {
            op,
            ty,
            src_type: None,
            dst: None,
            dsts: vec![],
            srcs: vec![],
            label: None,
            predicate: None,
            state_space: None,
            rounding: None,
        }
    }

    // === emit_memory_opcode tests (exhaustive string matching) ===

    #[test]
    fn test_emit_ld_global() {
        let mut instr = make_instr(PtxOp::Ld, PtxType::F32);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "ld.global");
    }

    #[test]
    fn test_emit_ld_shared() {
        let mut instr = make_instr(PtxOp::Ld, PtxType::F32);
        instr.state_space = Some(PtxStateSpace::Shared);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "ld.shared");
    }

    #[test]
    fn test_emit_ld_no_space() {
        let instr = make_instr(PtxOp::Ld, PtxType::F32);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "ld");
    }

    #[test]
    fn test_emit_ld_volatile_global() {
        let mut instr = make_instr(PtxOp::LdVolatile, PtxType::F32);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "ld.volatile.global");
    }

    #[test]
    fn test_emit_ld_volatile_no_space() {
        let instr = make_instr(PtxOp::LdVolatile, PtxType::F32);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "ld.volatile");
    }

    #[test]
    fn test_emit_ld_param() {
        let instr = make_instr(PtxOp::LdParam, PtxType::U64);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "ld.param");
    }

    #[test]
    fn test_emit_st_global() {
        let mut instr = make_instr(PtxOp::St, PtxType::F32);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "st.global");
    }

    #[test]
    fn test_emit_st_shared() {
        let mut instr = make_instr(PtxOp::St, PtxType::F32);
        instr.state_space = Some(PtxStateSpace::Shared);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "st.shared");
    }

    #[test]
    fn test_emit_st_no_space() {
        let instr = make_instr(PtxOp::St, PtxType::F32);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "st");
    }

    #[test]
    fn test_emit_cvta_shared() {
        let mut instr = make_instr(PtxOp::Cvta, PtxType::U64);
        instr.state_space = Some(PtxStateSpace::Shared);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvta.shared.u64");
    }

    #[test]
    fn test_emit_cvta_global() {
        let mut instr = make_instr(PtxOp::Cvta, PtxType::U64);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvta.global.u64");
    }

    #[test]
    fn test_emit_cvta_default() {
        let instr = make_instr(PtxOp::Cvta, PtxType::U64);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvta.shared.u64"); // Default to .shared
    }

    #[test]
    fn test_emit_atom_add_global() {
        let mut instr = make_instr(PtxOp::AtomAdd, PtxType::U32);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "atom.global.add");
    }

    #[test]
    fn test_emit_atom_add_shared() {
        let mut instr = make_instr(PtxOp::AtomAdd, PtxType::U32);
        instr.state_space = Some(PtxStateSpace::Shared);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "atom.shared.add");
    }

    #[test]
    fn test_emit_atom_add_default() {
        let instr = make_instr(PtxOp::AtomAdd, PtxType::U32);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "atom.global.add"); // Default to .global
    }

    #[test]
    fn test_emit_atom_min() {
        let mut instr = make_instr(PtxOp::AtomMin, PtxType::S32);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "atom.global.min");
    }

    #[test]
    fn test_emit_atom_max() {
        let mut instr = make_instr(PtxOp::AtomMax, PtxType::S32);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "atom.global.max");
    }

    #[test]
    fn test_emit_atom_exch() {
        let mut instr = make_instr(PtxOp::AtomExch, PtxType::U32);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "atom.global.exch");
    }

    #[test]
    fn test_emit_atom_cas() {
        let mut instr = make_instr(PtxOp::AtomCas, PtxType::U32);
        instr.state_space = Some(PtxStateSpace::Global);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "atom.global.cas");
    }

    #[test]
    fn test_emit_non_memory_op() {
        let instr = make_instr(PtxOp::Add, PtxType::F32);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert!(s.is_empty()); // No output for non-memory
    }

    // === emit_cvt_opcode tests (all branches) ===

    #[test]
    fn test_emit_cvt_f32_to_u32() {
        let mut instr = make_instr(PtxOp::Cvt, PtxType::U32);
        instr.srcs = vec![Operand::Reg(VirtualReg::new(0, PtxType::F32))];
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvt.rn.u32.f32");
    }

    #[test]
    fn test_emit_cvt_u32_to_f32() {
        let mut instr = make_instr(PtxOp::Cvt, PtxType::F32);
        instr.srcs = vec![Operand::Reg(VirtualReg::new(0, PtxType::U32))];
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvt.rn.f32.u32");
    }

    #[test]
    fn test_emit_cvt_explicit_rounding() {
        let mut instr = make_instr(PtxOp::Cvt, PtxType::F32);
        instr.rounding = Some(RoundingMode::Rz);
        instr.srcs = vec![Operand::Reg(VirtualReg::new(0, PtxType::F64))];
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvt.rz.f32.f64");
    }

    #[test]
    fn test_emit_cvt_explicit_src_type() {
        let mut instr = make_instr(PtxOp::Cvt, PtxType::F32);
        instr.src_type = Some(PtxType::S32);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvt.rn.f32.s32");
    }

    #[test]
    fn test_emit_cvt_f16_to_f32_no_rounding() {
        // f16 -> f32 conversion doesn't need rounding
        let mut instr = make_instr(PtxOp::Cvt, PtxType::F32);
        instr.src_type = Some(PtxType::F16);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvt.f32.f16");
    }

    #[test]
    fn test_emit_cvt_no_sources() {
        // Edge case: no sources, should use default
        let instr = make_instr(PtxOp::Cvt, PtxType::F32);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvt.rn.f32.u32"); // Default to .u32
    }

    #[test]
    fn test_emit_cvt_imm_source() {
        // Non-register source (immediate)
        let mut instr = make_instr(PtxOp::Cvt, PtxType::F32);
        instr.srcs = vec![Operand::ImmU64(42)];
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cvt.rn.f32.u32"); // Default source type
    }

    // === is_memory_op tests ===

    #[test]
    fn test_is_memory_op_all_variants() {
        assert!(is_memory_op(&PtxOp::Ld));
        assert!(is_memory_op(&PtxOp::LdVolatile));
        assert!(is_memory_op(&PtxOp::LdParam));
        assert!(is_memory_op(&PtxOp::St));
        assert!(is_memory_op(&PtxOp::Cvt));
        assert!(is_memory_op(&PtxOp::Cvta));
        assert!(is_memory_op(&PtxOp::AtomAdd));
        assert!(is_memory_op(&PtxOp::AtomMin));
        assert!(is_memory_op(&PtxOp::AtomMax));
        assert!(is_memory_op(&PtxOp::AtomExch));
        assert!(is_memory_op(&PtxOp::AtomCas));
    }

    #[test]
    fn test_is_memory_op_non_memory() {
        assert!(!is_memory_op(&PtxOp::Add));
        assert!(!is_memory_op(&PtxOp::Mul));
        assert!(!is_memory_op(&PtxOp::Bra));
        assert!(!is_memory_op(&PtxOp::Ret));
        assert!(!is_memory_op(&PtxOp::ShflDown));
    }

    // === skip_type_for_memory_op tests ===

    #[test]
    fn test_skip_type_for_cvt() {
        assert!(skip_type_for_memory_op(&PtxOp::Cvt));
    }

    #[test]
    fn test_skip_type_for_cvta() {
        assert!(skip_type_for_memory_op(&PtxOp::Cvta));
    }

    #[test]
    fn test_no_skip_type_for_ld() {
        assert!(!skip_type_for_memory_op(&PtxOp::Ld));
    }

    #[test]
    fn test_no_skip_type_for_st() {
        assert!(!skip_type_for_memory_op(&PtxOp::St));
    }

    #[test]
    fn test_no_skip_type_for_atom() {
        assert!(!skip_type_for_memory_op(&PtxOp::AtomAdd));
    }

    #[test]
    fn test_cp_async_emission() {
        let dst = VirtualReg::new(0, PtxType::U32);
        let src = VirtualReg::new(1, PtxType::U64);
        let mut instr = make_instr(PtxOp::CpAsync, PtxType::U8);
        instr.srcs = vec![Operand::Reg(dst), Operand::Reg(src), Operand::ImmU64(16)];
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert!(s.contains("cp.async.ca.shared.global"), "got: {s}");
        assert!(s.contains("], ["), "must have [dst], [src] syntax, got: {s}");
        assert!(s.contains(", 16"), "must have size 16, got: {s}");
    }

    #[test]
    fn test_cp_async_commit_group_emission() {
        let instr = make_instr(PtxOp::CpAsyncCommitGroup, PtxType::U8);
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cp.async.commit_group");
    }

    #[test]
    fn test_cp_async_wait_group_emission() {
        let mut instr = make_instr(PtxOp::CpAsyncWaitGroup, PtxType::U8);
        instr.srcs = vec![Operand::ImmU64(1)];
        let mut s = String::new();
        emit_memory_opcode(&instr, &mut s);
        assert_eq!(s, "cp.async.wait_group 1");
    }

    #[test]
    fn test_cp_async_is_memory_op() {
        assert!(is_memory_op(&PtxOp::CpAsync));
        assert!(is_memory_op(&PtxOp::CpAsyncCommitGroup));
        assert!(is_memory_op(&PtxOp::CpAsyncWaitGroup));
    }

    #[test]
    fn test_cp_async_skip_type() {
        assert!(skip_type_for_memory_op(&PtxOp::CpAsync));
        assert!(skip_type_for_memory_op(&PtxOp::CpAsyncCommitGroup));
        assert!(skip_type_for_memory_op(&PtxOp::CpAsyncWaitGroup));
    }
}
