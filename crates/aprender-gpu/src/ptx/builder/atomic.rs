//! PTX Atomic Operations Extension Trait.
//!
//! Provides atomic memory operations for global and shared memory.

use super::super::instructions::{Operand, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;
use super::super::types::{PtxStateSpace, PtxType};
use super::core::KernelBuilderCore;

/// Extension trait for PTX atomic operations.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::ptx::builder::{KernelBuilder, PtxAtomic};
///
/// fn build_kernel(kb: &mut KernelBuilder) {
///     let addr = kb.load_param_u64("counter");
///     let one = kb.mov_u32_imm(1);
///     let old = kb.atom_add_global_u32(addr, one);  // atomic increment
/// }
/// ```
pub trait PtxAtomic: KernelBuilderCore {
    // ===== Global Memory Atomics =====

    /// Atomic add (global memory)
    fn atom_add_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomAdd, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic exchange (global memory)
    fn atom_exch_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomExch, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic min (global memory)
    fn atom_min_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomMin, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic max (global memory)
    fn atom_max_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomMax, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic CAS (compare-and-swap, global memory)
    fn atom_cas_global_u32(
        &mut self,
        addr: VirtualReg,
        expected: VirtualReg,
        new_val: VirtualReg,
    ) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomCas, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(expected))
                .src(Operand::Reg(new_val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic add f32 (global memory) — requires compute capability ≥ 2.0
    fn atom_add_global_f32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomAdd, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    // ===== Shared Memory Atomics =====

    /// Atomic add (shared memory)
    fn atom_add_shared_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomAdd, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Shared),
        );
        dst
    }

    /// Atomic exchange (shared memory)
    fn atom_exch_shared_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomExch, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Shared),
        );
        dst
    }

    /// Atomic min (shared memory)
    fn atom_min_shared_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomMin, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Shared),
        );
        dst
    }

    /// Atomic max (shared memory)
    fn atom_max_shared_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::AtomMax, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Shared),
        );
        dst
    }
}

// Blanket implementation
impl<T: KernelBuilderCore> PtxAtomic for T {}

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
    fn test_atom_add_global() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);
        let val = builder.registers.allocate_virtual(PtxType::U32);

        let old = builder.atom_add_global_u32(addr, val);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::AtomAdd);
        assert_eq!(builder.instructions[0].state_space, Some(PtxStateSpace::Global));
        assert!(old.id() > 0);
    }

    #[test]
    fn test_atom_cas() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);
        let expected = builder.registers.allocate_virtual(PtxType::U32);
        let new_val = builder.registers.allocate_virtual(PtxType::U32);

        let old = builder.atom_cas_global_u32(addr, expected, new_val);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::AtomCas);
        assert!(old.id() > 0);
    }

    #[test]
    fn test_shared_atomics() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U32);
        let val = builder.registers.allocate_virtual(PtxType::U32);

        let _add = builder.atom_add_shared_u32(addr, val);
        let _exch = builder.atom_exch_shared_u32(addr, val);
        let _min = builder.atom_min_shared_u32(addr, val);
        let _max = builder.atom_max_shared_u32(addr, val);

        assert_eq!(builder.instructions.len(), 4);
        for instr in &builder.instructions {
            assert_eq!(instr.state_space, Some(PtxStateSpace::Shared));
        }
    }

    #[test]
    fn test_atom_exch_global() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);
        let val = builder.registers.allocate_virtual(PtxType::U32);

        let old = builder.atom_exch_global_u32(addr, val);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::AtomExch);
        assert_eq!(builder.instructions[0].state_space, Some(PtxStateSpace::Global));
        assert!(old.id() > 0);
    }

    #[test]
    fn test_atom_min_global() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);
        let val = builder.registers.allocate_virtual(PtxType::U32);

        let old = builder.atom_min_global_u32(addr, val);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::AtomMin);
        assert_eq!(builder.instructions[0].state_space, Some(PtxStateSpace::Global));
        assert!(old.id() > 0);
    }

    #[test]
    fn test_atom_max_global() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);
        let val = builder.registers.allocate_virtual(PtxType::U32);

        let old = builder.atom_max_global_u32(addr, val);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::AtomMax);
        assert_eq!(builder.instructions[0].state_space, Some(PtxStateSpace::Global));
        assert!(old.id() > 0);
    }

    #[test]
    fn test_global_atomics_all_ops() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);
        let val = builder.registers.allocate_virtual(PtxType::U32);
        let expected = builder.registers.allocate_virtual(PtxType::U32);
        let new_val = builder.registers.allocate_virtual(PtxType::U32);

        // Test all global atomic operations
        let _add = builder.atom_add_global_u32(addr, val);
        let _exch = builder.atom_exch_global_u32(addr, val);
        let _min = builder.atom_min_global_u32(addr, val);
        let _max = builder.atom_max_global_u32(addr, val);
        let _cas = builder.atom_cas_global_u32(addr, expected, new_val);

        assert_eq!(builder.instructions.len(), 5);
        for instr in &builder.instructions {
            assert_eq!(instr.state_space, Some(PtxStateSpace::Global));
        }
    }
}
