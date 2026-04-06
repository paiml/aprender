use super::*;
use crate::ptx::instructions::{Operand, Predicate};
use crate::ptx::registers::VirtualReg;
use crate::ptx::types::PtxType;

fn make_instr(op: PtxOp) -> PtxInstruction {
    PtxInstruction {
        op,
        ty: PtxType::F32,
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

fn vreg(id: u32, ty: PtxType) -> VirtualReg {
    VirtualReg::new(id, ty)
}

// === emit_instruction tests ===

#[test]
fn test_emit_label() {
    let mut instr = make_instr(PtxOp::Add);
    instr.label = Some("loop_start:".to_string());
    let result = emit_instruction(&instr);
    assert_eq!(result, "loop_start:\n");
}

#[test]
fn test_emit_with_predicate() {
    let mut instr = make_instr(PtxOp::Add);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::F32)), Operand::Reg(vreg(2, PtxType::F32))];
    instr.predicate = Some(Predicate { reg: vreg(10, PtxType::Pred), negated: false });
    let result = emit_instruction(&instr);
    assert!(result.contains('@'));
}

#[test]
fn test_emit_with_negated_predicate() {
    let mut instr = make_instr(PtxOp::Add);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::F32)), Operand::Reg(vreg(2, PtxType::F32))];
    instr.predicate = Some(Predicate { reg: vreg(10, PtxType::Pred), negated: true });
    let result = emit_instruction(&instr);
    assert!(result.contains("@!") || result.contains("@%!"));
}

#[test]
fn test_emit_add_instruction() {
    let mut instr = make_instr(PtxOp::Add);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::F32)), Operand::Reg(vreg(2, PtxType::F32))];
    let result = emit_instruction(&instr);
    assert!(result.contains("add"));
    assert!(result.contains(".f32"));
}

#[test]
fn test_emit_mul_instruction() {
    let mut instr = make_instr(PtxOp::Mul);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::F32)), Operand::Reg(vreg(2, PtxType::F32))];
    let result = emit_instruction(&instr);
    assert!(result.contains("mul"));
}

#[test]
fn test_emit_ld_global() {
    let mut instr = make_instr(PtxOp::Ld);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64))];
    instr.state_space = Some(PtxStateSpace::Global);
    let result = emit_instruction(&instr);
    assert!(result.contains("ld.global"));
}

#[test]
fn test_emit_ld_shared() {
    let mut instr = make_instr(PtxOp::Ld);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64))];
    instr.state_space = Some(PtxStateSpace::Shared);
    let result = emit_instruction(&instr);
    assert!(result.contains("ld.shared"));
}

#[test]
fn test_emit_st_instruction() {
    let mut instr = make_instr(PtxOp::St);
    instr.srcs = vec![Operand::Reg(vreg(0, PtxType::U64)), Operand::Reg(vreg(1, PtxType::F32))];
    instr.state_space = Some(PtxStateSpace::Global);
    let result = emit_instruction(&instr);
    assert!(result.contains("st.global"));
}

#[test]
fn test_emit_setp_instruction() {
    let mut instr = make_instr(PtxOp::Setp);
    instr.ty = PtxType::F32;
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::Pred)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::F32)), Operand::Reg(vreg(2, PtxType::F32))];
    instr.label = Some("lt".to_string());
    let result = emit_instruction(&instr);
    assert!(result.contains("setp"));
}

#[test]
fn test_emit_shfl_down() {
    let mut instr = make_instr(PtxOp::ShflDown);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![
        Operand::ImmU64(0xFFFFFFFF),
        Operand::Reg(vreg(1, PtxType::F32)),
        Operand::ImmU64(1),
        Operand::ImmU64(0x1F),
    ];
    let result = emit_instruction(&instr);
    assert!(result.contains("shfl"));
}

#[test]
fn test_emit_atom_add() {
    let mut instr = make_instr(PtxOp::AtomAdd);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::U32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64)), Operand::ImmU64(1)];
    instr.state_space = Some(PtxStateSpace::Global);
    let result = emit_instruction(&instr);
    assert!(result.contains("atom"));
}

#[test]
fn test_emit_fallback_op() {
    // Use an op that isn't handled by any category
    let instr = make_instr(PtxOp::Exit);
    let result = emit_instruction(&instr);
    assert!(result.contains("exit"));
}

#[test]
fn test_emit_multiple_dsts() {
    let mut instr = make_instr(PtxOp::Ld);
    instr.dsts = vec![Operand::Reg(vreg(0, PtxType::F32)), Operand::Reg(vreg(1, PtxType::F32))];
    instr.srcs = vec![Operand::Reg(vreg(2, PtxType::U64))];
    let result = emit_instruction(&instr);
    assert!(result.contains('{'));
    assert!(result.contains('}'));
}

#[test]
fn test_emit_multiple_srcs() {
    let mut instr = make_instr(PtxOp::Fma);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![
        Operand::Reg(vreg(1, PtxType::F32)),
        Operand::Reg(vreg(2, PtxType::F32)),
        Operand::Reg(vreg(3, PtxType::F32)),
    ];
    let result = emit_instruction(&instr);
    assert!(result.contains("fma"));
    assert!(result.matches(',').count() >= 2);
}

// === write_instruction tests ===

#[test]
fn test_write_label() {
    let mut instr = make_instr(PtxOp::Add);
    instr.label = Some("done:".to_string());
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert_eq!(out, "done:\n");
}

#[test]
fn test_write_with_predicate() {
    let mut instr = make_instr(PtxOp::Add);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::F32)), Operand::Reg(vreg(2, PtxType::F32))];
    instr.predicate = Some(Predicate { reg: vreg(10, PtxType::Pred), negated: false });
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains('@'));
}

#[test]
fn test_write_add_instruction() {
    let mut instr = make_instr(PtxOp::Add);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::F32)), Operand::Reg(vreg(2, PtxType::F32))];
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains("add"));
    assert!(out.contains(".f32"));
}

#[test]
fn test_write_ld_global() {
    let mut instr = make_instr(PtxOp::Ld);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64))];
    instr.state_space = Some(PtxStateSpace::Global);
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains("ld.global"));
}

#[test]
fn test_write_ld_shared() {
    let mut instr = make_instr(PtxOp::Ld);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64))];
    instr.state_space = Some(PtxStateSpace::Shared);
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains("ld.shared"));
}

#[test]
fn test_write_setp() {
    let mut instr = make_instr(PtxOp::Setp);
    instr.ty = PtxType::F32;
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::Pred)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::F32)), Operand::Reg(vreg(2, PtxType::F32))];
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains("setp"));
}

#[test]
fn test_write_shfl() {
    let mut instr = make_instr(PtxOp::ShflDown);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
    instr.srcs = vec![
        Operand::ImmU64(0xFFFFFFFF),
        Operand::Reg(vreg(1, PtxType::F32)),
        Operand::ImmU64(1),
        Operand::ImmU64(0x1F),
    ];
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains("shfl"));
}

#[test]
fn test_write_fallback_op() {
    let instr = make_instr(PtxOp::Exit);
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains("exit"));
}

#[test]
fn test_write_multiple_dsts() {
    let mut instr = make_instr(PtxOp::Ld);
    instr.dsts = vec![Operand::Reg(vreg(0, PtxType::F32)), Operand::Reg(vreg(1, PtxType::F32))];
    instr.srcs = vec![Operand::Reg(vreg(2, PtxType::U64))];
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains('{'));
    assert!(out.contains('}'));
}

#[test]
fn test_write_atom_global() {
    let mut instr = make_instr(PtxOp::AtomAdd);
    instr.dst = Some(Operand::Reg(vreg(0, PtxType::U32)));
    instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64)), Operand::ImmU64(1)];
    instr.state_space = Some(PtxStateSpace::Global);
    let mut out = String::new();
    write_instruction(&instr, &mut out);
    assert!(out.contains("atom"));
}

// === should_skip_type_suffix tests ===

#[test]
fn test_skip_type_for_shfl() {
    let instr = make_instr(PtxOp::ShflDown);
    assert!(should_skip_type_suffix(&instr));
}

#[test]
fn test_skip_type_for_cvta() {
    let instr = make_instr(PtxOp::Cvta);
    assert!(should_skip_type_suffix(&instr));
}

#[test]
fn test_no_skip_type_for_add() {
    let instr = make_instr(PtxOp::Add);
    assert!(!should_skip_type_suffix(&instr));
}

#[test]
fn test_wide_mul_skips_type() {
    let mut instr = make_instr(PtxOp::Mul);
    instr.ty = PtxType::U64;
    instr.srcs = vec![Operand::Reg(vreg(0, PtxType::U32)), Operand::Reg(vreg(1, PtxType::U32))];
    assert!(should_skip_type_suffix(&instr));
}

#[test]
fn test_regular_mul_no_skip() {
    let mut instr = make_instr(PtxOp::Mul);
    instr.ty = PtxType::U64;
    instr.srcs = vec![Operand::Reg(vreg(0, PtxType::U64)), Operand::Reg(vreg(1, PtxType::U64))];
    assert!(!should_skip_type_suffix(&instr));
}

#[test]
fn test_mul_wide_op_no_skip_type() {
    // PtxOp::MulWide uses the source type (U32) as instr.ty,
    // so the type suffix .u32 must NOT be skipped.
    let mut instr = make_instr(PtxOp::MulWide);
    instr.ty = PtxType::U32;
    instr.srcs = vec![Operand::Reg(vreg(0, PtxType::U32)), Operand::Reg(vreg(1, PtxType::U32))];
    assert!(!should_skip_type_suffix(&instr));
}
