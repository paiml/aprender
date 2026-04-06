//! Rounding mode compatibility, FMA instruction structure, use-count analysis,
//! def-map construction, multiple fusion, and special register tests.

use super::*;

// Test rounding mode compatibility - explicit Rn matches None
#[test]
fn test_rounding_mode_rn_matches_none() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    // mul with explicit Rn, add with None (implicit default)
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::F32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::Reg(r1))
            .rounding(RoundingMode::Rn),
        make_add(r4, r2, r3), // No explicit rounding mode
    ];

    let result = pass(instructions);
    assert_eq!(
        result.len(),
        1,
        "Explicit Rn should be compatible with None (default)"
    );
}

// Test rounding mode compatibility - None matches explicit Rn
#[test]
fn test_rounding_mode_none_matches_rn() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    // mul with None, add with explicit Rn
    let instructions = vec![
        make_mul(r2, r0, r1), // No explicit rounding mode
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(r4))
            .src(Operand::Reg(r2))
            .src(Operand::Reg(r3))
            .rounding(RoundingMode::Rn),
    ];

    let result = pass(instructions);
    assert_eq!(
        result.len(),
        1,
        "None should be compatible with explicit Rn"
    );
}

// Test rounding mode compatibility - same explicit non-Rn modes
#[test]
fn test_rounding_mode_same_explicit_mode() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    // Both with explicit Rz (round toward zero)
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::F32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::Reg(r1))
            .rounding(RoundingMode::Rz),
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(r4))
            .src(Operand::Reg(r2))
            .src(Operand::Reg(r3))
            .rounding(RoundingMode::Rz),
    ];

    let result = pass(instructions);
    assert_eq!(
        result.len(),
        1,
        "Same explicit rounding modes should be compatible"
    );
}

// Test incompatible rounding modes prevent fusion
#[test]
fn test_incompatible_rounding_modes_prevent_fusion() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    // mul with Rn, add with Rz - different modes should prevent fusion
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::F32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::Reg(r1))
            .rounding(RoundingMode::Rn),
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(r4))
            .src(Operand::Reg(r2))
            .src(Operand::Reg(r3))
            .rounding(RoundingMode::Rz),
    ];

    let result = pass(instructions);
    assert_eq!(
        result.len(),
        2,
        "Incompatible rounding modes should prevent fusion"
    );
}

// Test non-Rn mode with None is incompatible
#[test]
fn test_non_rn_with_none_incompatible() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    // mul with explicit Rz, add with None (implicit Rn)
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::F32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::Reg(r1))
            .rounding(RoundingMode::Rz),
        make_add(r4, r2, r3),
    ];

    let result = pass(instructions);
    assert_eq!(
        result.len(),
        2,
        "Non-Rn mode with implicit None should be incompatible"
    );
}

// Test rounding_modes_compatible directly
#[test]
fn test_rounding_modes_compatible_direct() {
    // Both None
    assert!(rounding_modes_compatible(None, None));

    // One None, one Rn
    assert!(rounding_modes_compatible(None, Some(&RoundingMode::Rn)));
    assert!(rounding_modes_compatible(Some(&RoundingMode::Rn), None));

    // Both Rn
    assert!(rounding_modes_compatible(
        Some(&RoundingMode::Rn),
        Some(&RoundingMode::Rn)
    ));

    // Same non-default modes
    assert!(rounding_modes_compatible(
        Some(&RoundingMode::Rz),
        Some(&RoundingMode::Rz)
    ));
    assert!(rounding_modes_compatible(
        Some(&RoundingMode::Rp),
        Some(&RoundingMode::Rp)
    ));
    assert!(rounding_modes_compatible(
        Some(&RoundingMode::Rm),
        Some(&RoundingMode::Rm)
    ));

    // Different explicit modes
    assert!(!rounding_modes_compatible(
        Some(&RoundingMode::Rn),
        Some(&RoundingMode::Rz)
    ));
    assert!(!rounding_modes_compatible(
        Some(&RoundingMode::Rp),
        Some(&RoundingMode::Rm)
    ));

    // Non-Rn with None is incompatible
    assert!(!rounding_modes_compatible(Some(&RoundingMode::Rz), None));
    assert!(!rounding_modes_compatible(None, Some(&RoundingMode::Rz)));
}

// Test FMA instruction structure
#[test]
fn test_fma_instruction_structure() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    let instructions = vec![make_mul(r2, r0, r1), make_add(r4, r2, r3)];

    let result = pass(instructions);
    assert_eq!(result.len(), 1);

    let fma = &result[0];
    assert!(matches!(fma.op, PtxOp::Fma));
    assert_eq!(fma.ty, PtxType::F32);

    // Check FMA has 3 sources: a, b, c
    assert_eq!(fma.srcs.len(), 3);

    // Check destination is r4
    match &fma.dst {
        Some(Operand::Reg(r)) => assert_eq!(*r, r4),
        _ => panic!("Expected register destination"),
    }

    // Check rounding mode is set
    assert!(fma.rounding.is_some());
}

// Test add with no destination (edge case)
#[test]
fn test_add_no_destination() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);

    let instructions = vec![
        make_mul(r2, r0, r1),
        // add without destination
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .src(Operand::Reg(r2))
            .src(Operand::Reg(r0)),
    ];

    let result = pass(instructions);
    // Should not fuse because add has no destination
    assert_eq!(result.len(), 2);
}

// Test count_register_uses with non-register sources
#[test]
fn test_count_uses_with_immediate_sources() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);

    // Instruction with immediate source - should not count as register use
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mov, PtxType::F32)
            .dst(Operand::Reg(r0))
            .src(Operand::ImmF32(1.0)), // Immediate, not register
        make_add(r1, r0, r0), // r0 is used twice
    ];

    let counts = count_register_uses(&instructions);

    // r0 should have exactly 2 uses (two reg sources in the add)
    assert_eq!(counts.get(&r0), Some(&2));
    // The immediate source should not create any register count entry
}

// Test build_def_map with non-register destination
#[test]
fn test_build_def_map_no_reg_dst() {
    let r0 = make_vreg(0, PtxType::F32);

    // Instruction with register destination
    let instr1 = PtxInstruction::new(PtxOp::Mov, PtxType::F32)
        .dst(Operand::Reg(r0))
        .src(Operand::ImmF32(1.0));

    // Instruction with no destination (e.g., branch)
    let instr2 = PtxInstruction::new(PtxOp::Bra, PtxType::Pred);

    let instructions = vec![instr1, instr2];

    let defs = build_def_map(&instructions);

    // r0 should be defined at index 0
    assert_eq!(defs.get(&r0), Some(&0));
    // Only one entry in the map
    assert_eq!(defs.len(), 1);
}

// Test add with only one source operand (edge case)
#[test]
fn test_add_single_source() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);

    let instructions = vec![
        make_mul(r1, r0, r0),
        // Malformed add with only 1 source
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r1)),
    ];

    let result = pass(instructions);
    // Should not fuse due to malformed add
    assert_eq!(result.len(), 2);
}

// Test multiple fusion opportunities in sequence
#[test]
fn test_multiple_fusions() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);
    let r5 = make_vreg(5, PtxType::F32);
    let r6 = make_vreg(6, PtxType::F32);
    let r7 = make_vreg(7, PtxType::F32);

    // Two independent mul+add pairs that should both fuse
    let instructions = vec![
        make_mul(r2, r0, r1), // First mul
        make_mul(r5, r3, r4), // Second mul
        make_add(r6, r2, r0), // First add (uses r2)
        make_add(r7, r5, r3), // Second add (uses r5)
    ];

    let result = pass(instructions);
    // Both should fuse, resulting in 2 FMAs
    assert_eq!(result.len(), 2);
    assert!(matches!(result[0].op, PtxOp::Fma));
    assert!(matches!(result[1].op, PtxOp::Fma));
}

// Test FMA preserves mul's rounding mode
#[test]
fn test_fma_inherits_mul_rounding() {
    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::F32);
    let r2 = make_vreg(2, PtxType::F32);
    let r3 = make_vreg(3, PtxType::F32);
    let r4 = make_vreg(4, PtxType::F32);

    // mul with explicit Rz rounding
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mul, PtxType::F32)
            .dst(Operand::Reg(r2))
            .src(Operand::Reg(r0))
            .src(Operand::Reg(r1))
            .rounding(RoundingMode::Rz),
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(r4))
            .src(Operand::Reg(r2))
            .src(Operand::Reg(r3))
            .rounding(RoundingMode::Rz),
    ];

    let result = pass(instructions);
    assert_eq!(result.len(), 1);

    // FMA should have Rz rounding (inherited from mul)
    assert_eq!(result[0].rounding, Some(RoundingMode::Rz));
}

// Test special register sources are not counted
#[test]
fn test_special_register_in_sources() {
    use super::super::super::super::registers::PtxReg;

    let r0 = make_vreg(0, PtxType::F32);
    let r1 = make_vreg(1, PtxType::U32);

    // Instruction using special register as source
    let instructions = vec![
        PtxInstruction::new(PtxOp::Mov, PtxType::U32)
            .dst(Operand::Reg(r1))
            .src(Operand::SpecialReg(PtxReg::TidX)),
        PtxInstruction::new(PtxOp::Mov, PtxType::F32)
            .dst(Operand::Reg(r0))
            .src(Operand::ImmF32(1.0)),
    ];

    let counts = count_register_uses(&instructions);

    // Special registers should not be counted
    // Only r1 is not used anywhere, r0 is not used anywhere either
    assert_eq!(counts.len(), 0);
}
