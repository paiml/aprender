//! Type compatibility, edge case, and fusion correctness tests for FMA fusion.

use super::*;

// Test F64 fusion (covers PtxType::F64 path)
#[test]
fn test_f64_fusion() {
    let r0 = make_vreg(0, PtxType::F64);
    let r1 = make_vreg(1, PtxType::F64);
    let r2 = make_vreg(2, PtxType::F64);
    let r3 = make_vreg(3, PtxType::F64);
    let r4 = make_vreg(4, PtxType::F64);

    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::F64)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::Reg(r1)),
        PtxInstruction::new(PtxOp::Add, PtxType::F64)
            .dst(Operand::Reg(r4))
            .src(Operand::Reg(r2))
            .src(Operand::Reg(r3)),
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 1, "F64 mul+add should fuse to FMA");
    assert!(matches!(result[0].op, PtxOp::Fma));
    assert_eq!(result[0].ty, PtxType::F64);
}

// Test fusion when mul result is second operand of add (c + a*b)
#[test]
fn test_fusion_mul_result_as_second_operand() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    // mul %r2, %r0, %r1  ; temp = a * b
    // add %r4, %r3, %r2  ; result = c + temp (mul result is second operand!)
    let instructions = vec![
        make_mul(r2, r0, r1),
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(r4))
            .src(Operand::Reg(r3)) // c is first
            .src(Operand::Reg(r2)), // mul result is second
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 1, "Should fuse even when mul result is second add operand");
    assert!(matches!(result[0].op, PtxOp::Fma));
}

// Test type mismatch prevents fusion (mul.f32, add.f64)
#[test]
fn test_type_mismatch_prevents_fusion() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F64);
    let r4 = make_vreg(4, PtxType::F64);

    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::F32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::Reg(r1)),
        // Add has F64 type while mul is F32 - should not fuse
        PtxInstruction::new(PtxOp::Add, PtxType::F64)
            .dst(Operand::Reg(r4))
            .src(Operand::Reg(r2))
            .src(Operand::Reg(r3)),
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 2, "Type mismatch between mul and add should prevent fusion");
}

// Test predicate counting in use-counts
#[test]
fn test_predicate_use_counts() {
    use super::super::super::super::instructions::Predicate;

    let pred = make_vreg(0, PtxType::Pred);
    let r0 = make_vreg(1, PtxType::F32);
    let r1 = make_vreg(2, PtxType::F32);
    let r2 = make_vreg(3, PtxType::F32);
    let r3 = make_vreg(4, PtxType::F32);
    let r4 = make_vreg(5, PtxType::F32);

    // mul with predicate guard - pred register is used in predicate
    let mut mul_instr = PtxInstruction::new(PtxOp::Mul, PtxType::F32)
        .dst(Operand::Reg(r2))
        .src(Operand::Reg(r0))
        .src(Operand::Reg(r1));
    mul_instr.predicate = Some(Predicate { reg: pred, negated: false });

    let instructions = vec![mul_instr, make_add(r4, r2, r3)];

    // Test should still fuse because predicate use doesn't affect r2's single use
    let result = pass(instructions);
    assert_eq!(result.len(), 1, "Predicate usage should not prevent fusion");
}

// Test defining instruction is not mul (e.g., mov)
#[test]
fn test_non_mul_definition_not_fused() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);

    // mov %r0, 1.0  ; r0 defined by mov, not mul
    // add %r2, %r0, %r1
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mov, PtxType::F32)
            .dst(Operand::Reg(r0))
            .src(Operand::ImmF32(1.0)),
        make_add(r2, r0, r1),
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 2, "Add with non-mul source definition should not fuse");
}

// Test mul with insufficient sources (edge case)
#[test]
fn test_mul_insufficient_sources() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);

    // Malformed mul with only 1 source (edge case)
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::F32).dst(Operand::Reg(r1)).src(Operand::Reg(r0)), // Only one source!
        make_add(r3, r1, r2),
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 2, "Mul with insufficient sources should not fuse");
}

// Test add with immediate (non-register) source
#[test]
fn test_add_with_immediate_source() {
    let r0 = make_vreg(0, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);

    // add %r2, %r0, 1.0  ; second source is immediate
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mov, PtxType::F32)
            .dst(Operand::Reg(r0))
            .src(Operand::ImmF32(1.0)),
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::ImmF32(2.0)), // Immediate source
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 2, "Add with immediate source should not fuse");
}

// Test register not in definition map (undefined register)
#[test]
fn test_undefined_register_source() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);

    // r0 is not defined anywhere but used in add
    let instructions = vec![make_add(r2, r0, r1)];

    let result = pass(instructions);
    assert_eq!(result.len(), 1, "Add with undefined register should be preserved");
}
