use super::*;

// Helper to create a simple mul instruction
fn make_mul(dst: VirtualReg, a: VirtualReg, b: VirtualReg) -> PtxInstruction {
    PtxInstruction::new(PtxOp::Mul, PtxType::F32)
        .dst(Operand::Reg(dst))
        .src(Operand::Reg(a))
        .src(Operand::Reg(b))
}

// Helper to create a simple add instruction
fn make_add(dst: VirtualReg, a: VirtualReg, b: VirtualReg) -> PtxInstruction {
    PtxInstruction::new(PtxOp::Add, PtxType::F32)
        .dst(Operand::Reg(dst))
        .src(Operand::Reg(a))
        .src(Operand::Reg(b))
}

fn make_vreg(id: u32, ty: PtxType) -> VirtualReg {
    VirtualReg::new(id, ty)
}

// cuda-tile-behavior.md: Falsification test #16
#[test]
fn test_fma_reduces_instruction_count() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    // mul %r2, %r0, %r1  ; temp = a * b
    // add %r4, %r2, %r3  ; result = temp + c
    let instructions = vec![make_mul(r2, r0, r1), make_add(r4, r2, r3)];

    let result = pass(instructions);

    // Should be fused to single FMA
    assert_eq!(result.len(), 1, "FMA fusion should reduce 2 instructions to 1");
    assert!(matches!(result[0].op, PtxOp::Fma), "Result should be FMA instruction");
}

// cuda-tile-behavior.md: Falsification test #18
#[test]
fn test_single_use_detection_prevents_incorrect_fusion() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);
    let r5 = make_vreg(5, PtxType::F32);

    // mul %r2, %r0, %r1  ; temp = a * b
    // add %r4, %r2, %r3  ; result1 = temp + c
    // add %r5, %r2, %r3  ; result2 = temp + c (uses temp again!)
    let instructions = vec![make_mul(r2, r0, r1), make_add(r4, r2, r3), make_add(r5, r2, r3)];

    let result = pass(instructions);

    // Should NOT fuse because r2 is used twice
    assert_eq!(result.len(), 3, "Should not fuse when mul result has multiple uses");
    assert!(!matches!(result[0].op, PtxOp::Fma), "First instruction should remain mul");
}

// cuda-tile-behavior.md: Falsification test #25
#[test]
fn test_fma_fusion_is_idempotent() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    let instructions = vec![make_mul(r2, r0, r1), make_add(r4, r2, r3)];

    let first_pass = pass(instructions);
    let second_pass = pass(first_pass.clone());

    assert_eq!(first_pass.len(), second_pass.len(), "FMA fusion should be idempotent");
}

// cuda-tile-behavior.md: Falsification test #30
#[test]
fn test_fma_pass_linear_complexity() {
    // Test with 1000 non-fusible instructions to verify O(n) complexity
    let mut instructions = Vec::with_capacity(1000);
    for i in 0..1000 {
        let r = make_vreg(i, PtxType::F32);
        instructions.push(PtxInstruction::new(PtxOp::Mov, PtxType::F32).dst(Operand::Reg(r)));
    }

    let start = std::time::Instant::now();
    let _result = pass(instructions);
    let elapsed = start.elapsed();

    // Should complete quickly (< 100ms for 1000 instructions)
    assert!(elapsed.as_millis() < 100, "FMA pass should have O(n) complexity, took {:?}", elapsed);
}

#[test]
fn test_fma_preserves_non_fusible() {
    // Instructions that can't be fused should be preserved
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);

    let instructions = vec![
        PtxInstruction::new(PtxOp::Mov, PtxType::F32)
            .dst(Operand::Reg(r0))
            .src(Operand::ImmF32(1.0)),
        make_add(r2, r0, r1), // No preceding mul
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 2, "Non-fusible instructions should be preserved");
}

#[test]
fn test_empty_input() {
    let result = pass(vec![]);
    assert!(result.is_empty());
}

#[test]
fn test_integer_ops_not_fused() {
    let r0 = make_vreg(0, PtxType::U32);
    let r1 = make_vreg(1, PtxType::U32);
    let r2 = make_vreg(2, PtxType::U32);
    let r3 = make_vreg(3, PtxType::U32);
    let r4 = make_vreg(4, PtxType::U32);

    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::U32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::Reg(r1)),
        PtxInstruction::new(PtxOp::Add, PtxType::U32)
            .dst(Operand::Reg(r4))
            .src(Operand::Reg(r2))
            .src(Operand::Reg(r3)),
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 2, "Integer ops should not be fused (no integer FMA)");
}

mod rounding_and_advanced;
mod type_and_edge_cases;
