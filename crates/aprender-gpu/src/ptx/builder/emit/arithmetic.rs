//! Arithmetic operation emission
//!
//! Handles: Mov, Add, Sub, Mul, MadLo, Div, Fma, Neg, Ex2, Rsqrt, Rcp, Sqrt, Sin, Cos, Dp4a variants

use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};
use crate::ptx::types::PtxType;

/// Emit arithmetic opcode to the output string (allocating version)
pub(crate) fn emit_arithmetic_opcode(instr: &PtxInstruction, s: &mut String) {
    match instr.op {
        PtxOp::Mov => s.push_str("mov"),
        PtxOp::Add => s.push_str("add"),
        PtxOp::Sub => s.push_str("sub"),
        PtxOp::Mul => emit_mul_opcode(instr, s),
        PtxOp::MulWide => {
            // Explicitly emit mul.wide with the source type from instr.ty.
            // The instruction type IS the source type (e.g., U32 for u32 x u32 -> u64).
            s.push_str("mul.wide");
        }
        PtxOp::MadLo => s.push_str("mad.lo"),
        PtxOp::Div => {
            if instr.ty.is_float() {
                s.push_str("div.rn");
            } else {
                s.push_str("div");
            }
        }
        PtxOp::Fma => {
            let round = instr.rounding.as_ref().map_or(".rn", |r| r.to_ptx_string());
            s.push_str("fma");
            s.push_str(round);
        }
        PtxOp::Neg => s.push_str("neg"),
        PtxOp::Ex2 => s.push_str("ex2.approx"),
        PtxOp::Lg2 => s.push_str("lg2.approx"),
        PtxOp::Rsqrt => s.push_str("rsqrt.approx"),
        PtxOp::Rcp => s.push_str("rcp.approx"),
        PtxOp::Sqrt => {
            let round = instr.rounding.as_ref().map_or(".rn", |r| r.to_ptx_string());
            s.push_str("sqrt");
            s.push_str(round);
        }
        PtxOp::Sin => s.push_str("sin.approx"),
        PtxOp::Cos => s.push_str("cos.approx"),
        PtxOp::Dp4a => s.push_str("dp4a.u32.u32"),
        PtxOp::Dp4aUS => s.push_str("dp4a.u32.s32"),
        PtxOp::Dp4aS32 => s.push_str("dp4a.s32.s32"),
        _ => {}
    }
}

/// Handle complex mul opcode emission
fn emit_mul_opcode(instr: &PtxInstruction, s: &mut String) {
    let is_wide_output = instr.ty == PtxType::U64 || instr.ty == PtxType::S64;
    let has_u64_source = instr.srcs.first().is_some_and(|src| {
        matches!(src, Operand::Reg(vreg) if vreg.ty() == PtxType::U64 || vreg.ty() == PtxType::S64)
    });

    if is_wide_output && !has_u64_source {
        let src_ty = if instr.ty == PtxType::U64 { ".u32" } else { ".s32" };
        s.push_str("mul.wide");
        s.push_str(src_ty);
    } else if is_wide_output && has_u64_source {
        s.push_str("mul.lo");
    } else if instr.ty.is_float() {
        s.push_str("mul");
    } else {
        s.push_str("mul.lo");
    }
}

/// Check if this is an arithmetic operation
pub(crate) fn is_arithmetic_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::Mov
            | PtxOp::Add
            | PtxOp::Sub
            | PtxOp::Mul
            | PtxOp::MulWide
            | PtxOp::MadLo
            | PtxOp::Div
            | PtxOp::Fma
            | PtxOp::Neg
            | PtxOp::Ex2
            | PtxOp::Lg2
            | PtxOp::Rsqrt
            | PtxOp::Rcp
            | PtxOp::Sqrt
            | PtxOp::Sin
            | PtxOp::Cos
            | PtxOp::Dp4a
            | PtxOp::Dp4aUS
            | PtxOp::Dp4aS32
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::instructions::{Operand, RoundingMode};
    use crate::ptx::registers::VirtualReg;
    use crate::ptx::types::PtxType;

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

    // === emit_arithmetic_opcode tests (exhaustive string matching) ===

    #[test]
    fn test_emit_mov() {
        let instr = make_instr(PtxOp::Mov, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mov");
    }

    #[test]
    fn test_emit_add() {
        let instr = make_instr(PtxOp::Add, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "add");
    }

    #[test]
    fn test_emit_sub() {
        let instr = make_instr(PtxOp::Sub, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "sub");
    }

    #[test]
    fn test_emit_mad_lo() {
        let instr = make_instr(PtxOp::MadLo, PtxType::U32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mad.lo");
    }

    #[test]
    fn test_emit_div_float() {
        let instr = make_instr(PtxOp::Div, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "div.rn");
    }

    #[test]
    fn test_emit_div_integer() {
        let instr = make_instr(PtxOp::Div, PtxType::U32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "div");
    }

    #[test]
    fn test_emit_fma_default_rounding() {
        let instr = make_instr(PtxOp::Fma, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "fma.rn");
    }

    #[test]
    fn test_emit_fma_explicit_rounding() {
        let mut instr = make_instr(PtxOp::Fma, PtxType::F32);
        instr.rounding = Some(RoundingMode::Rz);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "fma.rz");
    }

    #[test]
    fn test_emit_neg() {
        let instr = make_instr(PtxOp::Neg, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "neg");
    }

    #[test]
    fn test_emit_ex2() {
        let instr = make_instr(PtxOp::Ex2, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "ex2.approx");
    }

    #[test]
    fn test_emit_rsqrt() {
        let instr = make_instr(PtxOp::Rsqrt, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "rsqrt.approx");
    }

    #[test]
    fn test_emit_rcp() {
        let instr = make_instr(PtxOp::Rcp, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "rcp.approx");
    }

    #[test]
    fn test_emit_sqrt_default() {
        let instr = make_instr(PtxOp::Sqrt, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "sqrt.rn");
    }

    #[test]
    fn test_emit_sqrt_explicit_rounding() {
        let mut instr = make_instr(PtxOp::Sqrt, PtxType::F32);
        instr.rounding = Some(RoundingMode::Rp);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "sqrt.rp");
    }

    #[test]
    fn test_emit_sin() {
        let instr = make_instr(PtxOp::Sin, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "sin.approx");
    }

    #[test]
    fn test_emit_cos() {
        let instr = make_instr(PtxOp::Cos, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "cos.approx");
    }

    #[test]
    fn test_emit_dp4a() {
        let instr = make_instr(PtxOp::Dp4a, PtxType::U32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "dp4a.u32.u32");
    }

    #[test]
    fn test_emit_dp4a_us() {
        let instr = make_instr(PtxOp::Dp4aUS, PtxType::U32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "dp4a.u32.s32");
    }

    #[test]
    fn test_emit_dp4a_s32() {
        let instr = make_instr(PtxOp::Dp4aS32, PtxType::S32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "dp4a.s32.s32");
    }

    #[test]
    fn test_emit_non_arithmetic_op() {
        let instr = make_instr(PtxOp::Ld, PtxType::F32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert!(s.is_empty()); // No output for non-arithmetic
    }

    // === emit_mul_opcode tests (all branches) ===

    #[test]
    fn test_mul_wide_u64_from_u32() {
        let mut instr = make_instr(PtxOp::Mul, PtxType::U64);
        instr.srcs = vec![Operand::Reg(VirtualReg::new(0, PtxType::U32))];
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mul.wide.u32");
    }

    #[test]
    fn test_mul_wide_s64_from_s32() {
        let mut instr = make_instr(PtxOp::Mul, PtxType::S64);
        instr.srcs = vec![Operand::Reg(VirtualReg::new(0, PtxType::S32))];
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mul.wide.s32");
    }

    #[test]
    fn test_mul_lo_u64_from_u64() {
        let mut instr = make_instr(PtxOp::Mul, PtxType::U64);
        instr.srcs = vec![Operand::Reg(VirtualReg::new(0, PtxType::U64))];
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mul.lo");
    }

    #[test]
    fn test_mul_float() {
        let mut instr = make_instr(PtxOp::Mul, PtxType::F32);
        instr.srcs = vec![Operand::Reg(VirtualReg::new(0, PtxType::F32))];
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mul");
    }

    #[test]
    fn test_mul_lo_integer() {
        let mut instr = make_instr(PtxOp::Mul, PtxType::U32);
        instr.srcs = vec![Operand::Reg(VirtualReg::new(0, PtxType::U32))];
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mul.lo");
    }

    // === PtxOp::MulWide tests ===

    #[test]
    fn test_mul_wide_op_u32() {
        let instr = make_instr(PtxOp::MulWide, PtxType::U32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mul.wide");
    }

    #[test]
    fn test_mul_wide_op_s32() {
        let instr = make_instr(PtxOp::MulWide, PtxType::S32);
        let mut s = String::new();
        emit_arithmetic_opcode(&instr, &mut s);
        assert_eq!(s, "mul.wide");
    }

    // === is_arithmetic_op tests ===

    #[test]
    fn test_is_arithmetic_op_all_variants() {
        assert!(is_arithmetic_op(&PtxOp::Mov));
        assert!(is_arithmetic_op(&PtxOp::Add));
        assert!(is_arithmetic_op(&PtxOp::Sub));
        assert!(is_arithmetic_op(&PtxOp::Mul));
        assert!(is_arithmetic_op(&PtxOp::MulWide));
        assert!(is_arithmetic_op(&PtxOp::MadLo));
        assert!(is_arithmetic_op(&PtxOp::Div));
        assert!(is_arithmetic_op(&PtxOp::Fma));
        assert!(is_arithmetic_op(&PtxOp::Neg));
        assert!(is_arithmetic_op(&PtxOp::Ex2));
        assert!(is_arithmetic_op(&PtxOp::Rsqrt));
        assert!(is_arithmetic_op(&PtxOp::Rcp));
        assert!(is_arithmetic_op(&PtxOp::Sqrt));
        assert!(is_arithmetic_op(&PtxOp::Sin));
        assert!(is_arithmetic_op(&PtxOp::Cos));
        assert!(is_arithmetic_op(&PtxOp::Dp4a));
        assert!(is_arithmetic_op(&PtxOp::Dp4aUS));
        assert!(is_arithmetic_op(&PtxOp::Dp4aS32));
    }

    #[test]
    fn test_is_arithmetic_op_non_arithmetic() {
        assert!(!is_arithmetic_op(&PtxOp::Ld));
        assert!(!is_arithmetic_op(&PtxOp::St));
        assert!(!is_arithmetic_op(&PtxOp::Bra));
        assert!(!is_arithmetic_op(&PtxOp::Ret));
        assert!(!is_arithmetic_op(&PtxOp::Bar));
        assert!(!is_arithmetic_op(&PtxOp::ShflDown));
    }
}
