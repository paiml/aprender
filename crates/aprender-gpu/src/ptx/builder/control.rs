//! PTX Control Flow Extension Trait.
//!
//! Provides labels, branches, returns, and immediate value operations.

use super::super::instructions::{Operand, Predicate, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::core::KernelBuilderCore;

/// Extension trait for PTX control flow operations.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::ptx::builder::{KernelBuilder, PtxControl};
///
/// fn build_kernel(kb: &mut KernelBuilder) {
///     kb.label("loop_start");
///     // ... loop body ...
///     kb.branch("loop_start");
/// }
/// ```
pub trait PtxControl: KernelBuilderCore {
    // ===== Labels and Branches =====

    /// Create a label at the current position
    fn label(&mut self, name: &str) {
        self.labels_mut().push(name.to_string());
        // Labels are stored as a Mov instruction with label field
        let mut instr = PtxInstruction::new(PtxOp::Mov, PtxType::B32);
        instr.label = Some(format!("{}:", name));
        self.instructions_mut().push(instr);
    }

    /// Unconditional branch
    fn branch(&mut self, target: &str) {
        self.instructions_mut().push(PtxInstruction::new(PtxOp::Bra, PtxType::B32).label(target));
    }

    /// Conditional branch (if predicate is true)
    fn branch_if(&mut self, pred: VirtualReg, target: &str) {
        let predicate = Predicate { reg: pred, negated: false };
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Bra, PtxType::B32).predicated(predicate).label(target),
        );
    }

    /// Conditional branch (if predicate is false)
    fn branch_if_not(&mut self, pred: VirtualReg, target: &str) {
        let predicate = Predicate { reg: pred, negated: true };
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Bra, PtxType::B32).predicated(predicate).label(target),
        );
    }

    /// Return from kernel
    fn ret(&mut self) {
        self.instructions_mut().push(PtxInstruction::new(PtxOp::Ret, PtxType::Pred));
    }

    // ===== Immediate Moves =====

    /// Move u64 immediate into new register
    fn mov_u64_imm(&mut self, val: u64) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mov, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmU64(val)),
        );
        dst
    }

    /// Move u32 immediate into new register
    fn mov_u32_imm(&mut self, val: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mov, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmI64(val as i64)),
        );
        dst
    }

    /// Move f32 immediate into new register
    fn mov_f32_imm(&mut self, val: f32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mov, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmF32(val)),
        );
        dst
    }

    /// Move register to register (copy)
    fn mov_reg(&mut self, src: VirtualReg, ty: PtxType) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(ty);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mov, ty).dst(Operand::Reg(dst)).src(Operand::Reg(src)),
        );
        dst
    }
}

// Blanket implementation
impl<T: KernelBuilderCore> PtxControl for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::registers::RegisterAllocator;

    struct MockBuilder {
        registers: RegisterAllocator,
        instructions: Vec<PtxInstruction>,
        labels: Vec<String>,
    }

    impl MockBuilder {
        fn new() -> Self {
            Self {
                registers: RegisterAllocator::new(),
                instructions: Vec::new(),
                labels: Vec::new(),
            }
        }
    }

    impl KernelBuilderCore for MockBuilder {
        fn registers_mut(&mut self) -> &mut RegisterAllocator {
            &mut self.registers
        }
        fn instructions_mut(&mut self) -> &mut Vec<PtxInstruction> {
            &mut self.instructions
        }
        fn labels_mut(&mut self) -> &mut Vec<String> {
            &mut self.labels
        }
    }

    #[test]
    fn test_label_and_branch() {
        let mut builder = MockBuilder::new();

        builder.label("loop_start");
        builder.branch("loop_start");

        assert_eq!(builder.labels.len(), 1);
        assert_eq!(builder.labels[0], "loop_start");
        assert_eq!(builder.instructions.len(), 2);
    }

    #[test]
    fn test_conditional_branch() {
        let mut builder = MockBuilder::new();
        let pred = builder.registers.allocate_virtual(PtxType::Pred);

        builder.branch_if(pred, "target");
        builder.branch_if_not(pred, "other");

        assert_eq!(builder.instructions.len(), 2);
        assert!(builder.instructions[0].predicate.is_some());
        assert!(builder.instructions[1].predicate.is_some());
        // First one not negated
        assert!(!builder.instructions[0].predicate.as_ref().expect("test").negated);
        // Second one negated
        assert!(builder.instructions[1].predicate.as_ref().expect("test").negated);
    }

    #[test]
    fn test_mov_immediates() {
        let mut builder = MockBuilder::new();

        let _a = builder.mov_u32_imm(42);
        let _b = builder.mov_u64_imm(12345);
        let _c = builder.mov_f32_imm(3.125);

        assert_eq!(builder.instructions.len(), 3);
        for instr in &builder.instructions {
            assert_eq!(instr.op, PtxOp::Mov);
        }
    }

    #[test]
    fn test_ret() {
        let mut builder = MockBuilder::new();

        builder.ret();

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Ret);
    }

    #[test]
    fn test_mov_reg() {
        let mut builder = MockBuilder::new();

        // First create a source register with a value
        let src = builder.mov_u32_imm(42);

        // Then copy it to another register
        let dst = builder.mov_reg(src, PtxType::U32);

        // Should have 2 instructions: original mov_imm and the reg-to-reg mov
        assert_eq!(builder.instructions.len(), 2);

        // Both should be Mov operations
        assert_eq!(builder.instructions[0].op, PtxOp::Mov);
        assert_eq!(builder.instructions[1].op, PtxOp::Mov);

        // The source and destination should be different registers
        assert_ne!(src, dst);

        // Check that the second instruction has the src register as source operand
        assert!(!builder.instructions[1].srcs.is_empty());
        match &builder.instructions[1].srcs[0] {
            Operand::Reg(r) => assert_eq!(*r, src),
            _ => panic!("Expected register source operand"),
        }

        // Check that the second instruction has the dst register as destination operand
        match &builder.instructions[1].dst {
            Some(Operand::Reg(r)) => assert_eq!(*r, dst),
            _ => panic!("Expected register destination operand"),
        }
    }

    #[test]
    fn test_mov_reg_f32() {
        let mut builder = MockBuilder::new();

        // Create f32 source register
        let src = builder.mov_f32_imm(3.125);

        // Copy to another f32 register
        let dst = builder.mov_reg(src, PtxType::F32);

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[1].ty, PtxType::F32);
        assert_ne!(src, dst);
    }

    #[test]
    fn test_multiple_labels() {
        let mut builder = MockBuilder::new();

        builder.label("start");
        builder.label("middle");
        builder.label("end");

        assert_eq!(builder.labels.len(), 3);
        assert_eq!(builder.labels[0], "start");
        assert_eq!(builder.labels[1], "middle");
        assert_eq!(builder.labels[2], "end");

        // Verify label instructions are created
        assert_eq!(builder.instructions.len(), 3);
        for instr in &builder.instructions {
            assert!(instr.label.is_some());
        }
    }

    #[test]
    fn test_label_format() {
        let mut builder = MockBuilder::new();

        builder.label("my_loop");

        let instr = &builder.instructions[0];
        assert_eq!(instr.label, Some("my_loop:".to_string()));
    }

    #[test]
    fn test_branch_creates_label_target() {
        let mut builder = MockBuilder::new();

        builder.branch("exit");

        let instr = &builder.instructions[0];
        assert_eq!(instr.op, PtxOp::Bra);
        assert_eq!(instr.label, Some("exit".to_string()));
        assert!(instr.predicate.is_none());
    }

    #[test]
    fn test_branch_if_predicate_structure() {
        let mut builder = MockBuilder::new();
        let pred = builder.registers.allocate_virtual(PtxType::Pred);

        builder.branch_if(pred, "target_label");

        let instr = &builder.instructions[0];
        let predicate = instr.predicate.as_ref().expect("test");

        assert_eq!(predicate.reg, pred);
        assert!(!predicate.negated);
    }

    #[test]
    fn test_branch_if_not_predicate_structure() {
        let mut builder = MockBuilder::new();
        let pred = builder.registers.allocate_virtual(PtxType::Pred);

        builder.branch_if_not(pred, "other_label");

        let instr = &builder.instructions[0];
        let predicate = instr.predicate.as_ref().expect("test");

        assert_eq!(predicate.reg, pred);
        assert!(predicate.negated);
    }

    #[test]
    fn test_mov_u64_imm_value() {
        let mut builder = MockBuilder::new();

        let reg = builder.mov_u64_imm(0xDEADBEEFCAFEBABE);

        assert_eq!(builder.instructions.len(), 1);
        let instr = &builder.instructions[0];

        assert_eq!(instr.op, PtxOp::Mov);
        assert_eq!(instr.ty, PtxType::U64);

        // Verify destination is the returned register
        match &instr.dst {
            Some(Operand::Reg(r)) => assert_eq!(*r, reg),
            _ => panic!("Expected register destination"),
        }

        // Verify source is the immediate value
        assert!(!instr.srcs.is_empty());
        match &instr.srcs[0] {
            Operand::ImmU64(v) => assert_eq!(*v, 0xDEADBEEFCAFEBABE),
            _ => panic!("Expected ImmU64 source"),
        }
    }

    #[test]
    fn test_mov_u32_imm_value() {
        let mut builder = MockBuilder::new();

        let reg = builder.mov_u32_imm(0xCAFEBABE);

        let instr = &builder.instructions[0];
        assert_eq!(instr.ty, PtxType::U32);

        // Verify destination
        match &instr.dst {
            Some(Operand::Reg(r)) => assert_eq!(*r, reg),
            _ => panic!("Expected register destination"),
        }

        // Verify source (stored as i64)
        assert!(!instr.srcs.is_empty());
        match &instr.srcs[0] {
            Operand::ImmI64(v) => assert_eq!(*v, 0xCAFEBABE_i64),
            _ => panic!("Expected ImmI64 source"),
        }
    }

    #[test]
    fn test_mov_f32_imm_value() {
        let mut builder = MockBuilder::new();

        let reg = builder.mov_f32_imm(std::f32::consts::E);

        let instr = &builder.instructions[0];
        assert_eq!(instr.ty, PtxType::F32);

        // Verify destination
        match &instr.dst {
            Some(Operand::Reg(r)) => assert_eq!(*r, reg),
            _ => panic!("Expected register destination"),
        }

        // Verify source
        assert!(!instr.srcs.is_empty());
        match &instr.srcs[0] {
            Operand::ImmF32(v) => assert!((v - std::f32::consts::E).abs() < 1e-5),
            _ => panic!("Expected ImmF32 source"),
        }
    }

    #[test]
    fn test_ret_instruction_type() {
        let mut builder = MockBuilder::new();

        builder.ret();

        let instr = &builder.instructions[0];
        assert_eq!(instr.op, PtxOp::Ret);
        assert_eq!(instr.ty, PtxType::Pred);
    }
}
