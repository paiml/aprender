use super::*;

#[test]
fn test_cmp_op_strings() {
    assert_eq!(CmpOp::Eq.to_ptx_string(), "eq");
    assert_eq!(CmpOp::Lt.to_ptx_string(), "lt");
    assert_eq!(CmpOp::Ge.to_ptx_string(), "ge");
}

#[test]
fn test_cmp_op_all_variants() {
    // Test all CmpOp variants for complete coverage
    assert_eq!(CmpOp::Ne.to_ptx_string(), "ne");
    assert_eq!(CmpOp::Le.to_ptx_string(), "le");
    assert_eq!(CmpOp::Gt.to_ptx_string(), "gt");
    assert_eq!(CmpOp::Lo.to_ptx_string(), "lo");
    assert_eq!(CmpOp::Ls.to_ptx_string(), "ls");
    assert_eq!(CmpOp::Hi.to_ptx_string(), "hi");
    assert_eq!(CmpOp::Hs.to_ptx_string(), "hs");
}

#[test]
fn test_rounding_mode_strings() {
    assert_eq!(RoundingMode::Rn.to_ptx_string(), ".rn");
    assert_eq!(RoundingMode::Rz.to_ptx_string(), ".rz");
}

#[test]
fn test_instruction_builder() {
    let instr = PtxInstruction::new(PtxOp::Add, PtxType::F32)
        .dst(Operand::ImmF32(0.0))
        .src(Operand::ImmF32(1.0))
        .src(Operand::ImmF32(2.0));

    assert_eq!(instr.op, PtxOp::Add);
    assert_eq!(instr.ty, PtxType::F32);
    assert!(instr.dst.is_some());
    assert_eq!(instr.srcs.len(), 2);
}

#[test]
fn test_instruction_predicated() {
    let pred_reg = VirtualReg::new(0, PtxType::Pred);
    let pred = Predicate { reg: pred_reg, negated: false };

    let instr = PtxInstruction::new(PtxOp::Bra, PtxType::B32).predicated(pred).label("exit");

    assert!(instr.predicate.is_some());
    assert!(instr.label.is_some());
}

#[test]
fn test_instruction_memory() {
    let instr = PtxInstruction::new(PtxOp::Ld, PtxType::F32).space(PtxStateSpace::Global);

    assert_eq!(instr.state_space, Some(PtxStateSpace::Global));
}

#[test]
fn test_wmma_layout_strings() {
    assert_eq!(WmmaLayout::RowMajor.to_ptx_string(), "row");
    assert_eq!(WmmaLayout::ColMajor.to_ptx_string(), "col");
}

#[test]
fn test_wmma_shape_strings() {
    assert_eq!(WmmaShape::M16N16K16.to_ptx_string(), "m16n16k16");
    assert_eq!(WmmaShape::M8N32K16.to_ptx_string(), "m8n32k16");
    assert_eq!(WmmaShape::M32N8K16.to_ptx_string(), "m32n8k16");
}

#[test]
fn test_wmma_shape_values() {
    let shape = WmmaShape::M16N16K16;
    assert_eq!(shape.m, 16);
    assert_eq!(shape.n, 16);
    assert_eq!(shape.k, 16);
}

#[test]
fn test_wmma_ops_exist() {
    // Verify WMMA ops are in the enum
    let _load_a = PtxOp::WmmaLoadA;
    let _load_b = PtxOp::WmmaLoadB;
    let _load_c = PtxOp::WmmaLoadC;
    let _mma = PtxOp::WmmaMma;
    let _store = PtxOp::WmmaStoreD;
}
