//! PTX Comparison Operations Extension Trait.
//!
//! Provides predicate-setting comparison operations.

use super::super::instructions::{CmpOp, Operand, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::core::KernelBuilderCore;

/// Extension trait for PTX comparison operations.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::ptx::builder::{KernelBuilder, PtxComparison, PtxControl};
///
/// fn build_kernel(kb: &mut KernelBuilder) {
///     let a = kb.load_param_u32("a");
///     let b = kb.load_param_u32("b");
///     let pred = kb.setp_ge_u32(a, b);  // From PtxComparison trait
///     kb.branch_if(pred, "a_ge_b");     // From PtxControl trait
/// }
/// ```
pub trait PtxComparison: KernelBuilderCore {
    /// Set predicate if a >= b (unsigned)
    fn setp_ge_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::U32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b));
        instr.label = Some(CmpOp::Ge.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate if a == b (unsigned 32-bit)
    fn setp_eq_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::U32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b));
        instr.label = Some(CmpOp::Eq.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate if a != b (unsigned 32-bit)
    fn setp_ne_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::U32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b));
        instr.label = Some(CmpOp::Ne.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate if a < b (unsigned 32-bit)
    fn setp_lt_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::U32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b));
        instr.label = Some(CmpOp::Lt.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate if a > b (unsigned 32-bit)
    fn setp_gt_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::U32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b));
        instr.label = Some(CmpOp::Gt.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate if a <= b (unsigned 32-bit)
    fn setp_le_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::U32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b));
        instr.label = Some(CmpOp::Le.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate if a < b (f32)
    fn setp_lt_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::F32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b));
        instr.label = Some(CmpOp::Lt.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate if a > b (f32)
    fn setp_gt_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::F32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b));
        instr.label = Some(CmpOp::Gt.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate comparing u32 with immediate
    fn setp_ge_u32_imm(&mut self, a: VirtualReg, b: u32) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::U32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::ImmI64(b as i64));
        instr.label = Some(CmpOp::Ge.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }

    /// Set predicate comparing u32 for less than with immediate
    fn setp_lt_u32_imm(&mut self, a: VirtualReg, b: u32) -> VirtualReg {
        let pred = self.registers_mut().allocate_virtual(PtxType::Pred);
        let mut instr = PtxInstruction::new(PtxOp::Setp, PtxType::U32)
            .dst(Operand::Reg(pred))
            .src(Operand::Reg(a))
            .src(Operand::ImmI64(b as i64));
        instr.label = Some(CmpOp::Lt.to_ptx_string().to_string());
        self.instructions_mut().push(instr);
        pred
    }
}

// Blanket implementation
impl<T: KernelBuilderCore> PtxComparison for T {}

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
    fn test_setp_ge_u32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);

        let pred = builder.setp_ge_u32(a, b);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Setp);
        assert_eq!(builder.instructions[0].label.as_deref(), Some("ge"));
        // Verify predicate register was created (IDs start at 0)
        let _ = pred;
    }

    #[test]
    fn test_all_comparisons() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);

        let _ge = builder.setp_ge_u32(a, b);
        let _eq = builder.setp_eq_u32(a, b);
        let _ne = builder.setp_ne_u32(a, b);
        let _lt = builder.setp_lt_u32(a, b);
        let _gt = builder.setp_gt_u32(a, b);
        let _le = builder.setp_le_u32(a, b);

        assert_eq!(builder.instructions.len(), 6);

        let labels: Vec<_> =
            builder.instructions.iter().map(|i| i.label.as_deref().expect("test")).collect();
        assert_eq!(labels, vec!["ge", "eq", "ne", "lt", "gt", "le"]);
    }

    #[test]
    fn test_f32_comparisons() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::F32);
        let b = builder.registers.allocate_virtual(PtxType::F32);

        let _lt = builder.setp_lt_f32(a, b);
        let _gt = builder.setp_gt_f32(a, b);

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].ty, PtxType::F32);
        assert_eq!(builder.instructions[1].ty, PtxType::F32);
    }

    #[test]
    fn test_setp_ge_u32_imm() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);

        let pred = builder.setp_ge_u32_imm(a, 42);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Setp);
        assert_eq!(builder.instructions[0].ty, PtxType::U32);
        assert_eq!(builder.instructions[0].label.as_deref(), Some("ge"));

        // Verify the second source is an immediate
        assert_eq!(builder.instructions[0].srcs.len(), 2);
        match &builder.instructions[0].srcs[1] {
            Operand::ImmI64(val) => assert_eq!(*val, 42),
            _ => panic!("Expected ImmI64 operand"),
        }

        // Verify predicate register was created
        assert_eq!(pred.ty(), PtxType::Pred);
    }

    #[test]
    fn test_setp_lt_u32_imm() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);

        let pred = builder.setp_lt_u32_imm(a, 100);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Setp);
        assert_eq!(builder.instructions[0].ty, PtxType::U32);
        assert_eq!(builder.instructions[0].label.as_deref(), Some("lt"));

        // Verify the second source is an immediate
        assert_eq!(builder.instructions[0].srcs.len(), 2);
        match &builder.instructions[0].srcs[1] {
            Operand::ImmI64(val) => assert_eq!(*val, 100),
            _ => panic!("Expected ImmI64 operand"),
        }

        // Verify predicate register was created
        assert_eq!(pred.ty(), PtxType::Pred);
    }

    #[test]
    fn test_imm_comparisons_boundary_values() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);

        // Test with zero
        let _pred_zero = builder.setp_ge_u32_imm(a, 0);

        // Test with max u32
        let _pred_max = builder.setp_lt_u32_imm(a, u32::MAX);

        assert_eq!(builder.instructions.len(), 2);

        // Verify zero immediate
        match &builder.instructions[0].srcs[1] {
            Operand::ImmI64(val) => assert_eq!(*val, 0),
            _ => panic!("Expected ImmI64 operand"),
        }

        // Verify u32::MAX immediate
        match &builder.instructions[1].srcs[1] {
            Operand::ImmI64(val) => assert_eq!(*val, u32::MAX as i64),
            _ => panic!("Expected ImmI64 operand"),
        }
    }

    #[test]
    fn test_all_comparisons_generate_unique_predicates() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);
        let f32_a = builder.registers.allocate_virtual(PtxType::F32);
        let f32_b = builder.registers.allocate_virtual(PtxType::F32);

        // Call all comparison methods
        let pred1 = builder.setp_ge_u32(a, b);
        let pred2 = builder.setp_eq_u32(a, b);
        let pred3 = builder.setp_ne_u32(a, b);
        let pred4 = builder.setp_lt_u32(a, b);
        let pred5 = builder.setp_gt_u32(a, b);
        let pred6 = builder.setp_le_u32(a, b);
        let pred7 = builder.setp_lt_f32(f32_a, f32_b);
        let pred8 = builder.setp_gt_f32(f32_a, f32_b);
        let pred9 = builder.setp_ge_u32_imm(a, 10);
        let pred10 = builder.setp_lt_u32_imm(a, 20);

        // All predicates should be unique
        let preds = vec![pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10];
        for i in 0..preds.len() {
            for j in (i + 1)..preds.len() {
                assert_ne!(
                    preds[i].id(),
                    preds[j].id(),
                    "Predicate {} and {} should have different IDs",
                    i,
                    j
                );
            }
        }

        // All predicates should be of type Pred
        for (i, pred) in preds.iter().enumerate() {
            assert_eq!(pred.ty(), PtxType::Pred, "Predicate {} should be of type Pred", i);
        }
    }
}
