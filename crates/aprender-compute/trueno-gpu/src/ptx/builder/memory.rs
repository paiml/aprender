//! PTX Memory Operations Extension Trait.
//!
//! Provides global and shared memory load/store operations.

use super::super::instructions::{Operand, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;
use super::super::types::{PtxStateSpace, PtxType};
use super::core::KernelBuilderCore;

/// Extension trait for PTX memory operations.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::ptx::builder::{KernelBuilder, PtxMemory};
///
/// fn build_kernel(kb: &mut KernelBuilder) {
///     let addr = kb.load_param_u64("ptr");
///     let val = kb.ld_global_f32(addr);  // From PtxMemory trait
///     kb.st_global_f32(addr, val);
/// }
/// ```
pub trait PtxMemory: KernelBuilderCore {
    // ===== Global Memory =====

    /// Load f32 from global memory
    fn ld_global_f32(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ld, PtxType::F32)
                .space(PtxStateSpace::Global)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    /// Store f32 to global memory
    fn st_global_f32(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::St, PtxType::F32)
                .space(PtxStateSpace::Global)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
        );
    }

    /// Load u32 from global memory
    fn ld_global_u32(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U32)
                .space(PtxStateSpace::Global)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    /// Store u32 to global memory
    fn st_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::St, PtxType::U32)
                .space(PtxStateSpace::Global)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
        );
    }

    /// Load u8 from global memory (zero-extended to u32)
    fn ld_global_u8(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U8)
                .space(PtxStateSpace::Global)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    /// Load u16 from global memory (zero-extended)
    fn ld_global_u16(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U16)
                .space(PtxStateSpace::Global)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    // ===== Shared Memory =====

    /// Load f32 from shared memory
    fn ld_shared_f32(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ld, PtxType::F32)
                .space(PtxStateSpace::Shared)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    /// Store f32 to shared memory
    fn st_shared_f32(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::St, PtxType::F32)
                .space(PtxStateSpace::Shared)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
        );
    }

    /// Load u32 from shared memory
    fn ld_shared_u32(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U32)
                .space(PtxStateSpace::Shared)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    /// Store u32 to shared memory
    fn st_shared_u32(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::St, PtxType::U32)
                .space(PtxStateSpace::Shared)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
        );
    }

    /// Load u32 from shared memory with volatile semantics
    ///
    /// Volatile loads ensure the value is always read from memory,
    /// not from registers or cache. Used for synchronization.
    fn ld_shared_u32_volatile(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::LdVolatile, PtxType::U32)
                .space(PtxStateSpace::Shared)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    /// Prefetch a cache line from global memory to L2 cache.
    ///
    /// This is a non-faulting hint — invalid addresses are silently ignored.
    /// Use to prefetch data for the next loop iteration while computing the current one.
    fn prefetch_global_l2(&mut self, addr: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Prefetch, PtxType::U8)
                .space(PtxStateSpace::Global)
                .src(Operand::Reg(addr)),
        );
    }

    /// Asynchronous copy from global to shared memory (SM 8.0+).
    ///
    /// Copies `size_bytes` (4, 8, or 16) from global `src_addr` to shared `dst_smem_offset`.
    /// Does NOT consume registers — data goes directly global→shared.
    /// Must be followed by `cp_async_commit_group` and `cp_async_wait_group`.
    fn cp_async_global_to_shared(
        &mut self,
        dst_smem_offset: VirtualReg,
        src_global_addr: VirtualReg,
        size_bytes: u32,
    ) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::CpAsync, PtxType::U8)
                .src(Operand::Reg(dst_smem_offset))
                .src(Operand::Reg(src_global_addr))
                .src(Operand::ImmU64(size_bytes as u64)),
        );
    }

    /// Commit all outstanding cp.async operations as a group.
    fn cp_async_commit_group(&mut self) {
        self.instructions_mut().push(PtxInstruction::new(PtxOp::CpAsyncCommitGroup, PtxType::U8));
    }

    /// Wait until at most `n` cp.async groups are outstanding.
    /// `n=0` waits for ALL groups to complete.
    fn cp_async_wait_group(&mut self, n: u32) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::CpAsyncWaitGroup, PtxType::U8)
                .src(Operand::ImmU64(n as u64)),
        );
    }

    /// Store f16 to shared memory (stored as b16)
    ///
    /// Half-precision floats are stored using b16 type in PTX.
    fn st_shared_f16(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::St, PtxType::B16)
                .space(PtxStateSpace::Shared)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
        );
    }

    /// Load f16 from shared memory (loaded as b16)
    ///
    /// Half-precision floats are loaded using b16 type in PTX.
    fn ld_shared_f16(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F16);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ld, PtxType::B16)
                .space(PtxStateSpace::Shared)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }
}

// Blanket implementation
impl<T: KernelBuilderCore> PtxMemory for T {}

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
    fn test_ld_st_global_f32() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);

        let val = builder.ld_global_f32(addr);
        builder.st_global_f32(addr, val);

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].op, PtxOp::Ld);
        assert_eq!(builder.instructions[1].op, PtxOp::St);
    }

    #[test]
    fn test_shared_memory_ops() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U32);

        let val = builder.ld_shared_f32(addr);
        builder.st_shared_f32(addr, val);

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].state_space, Some(PtxStateSpace::Shared));
        assert_eq!(builder.instructions[1].state_space, Some(PtxStateSpace::Shared));
    }

    #[test]
    fn test_global_u32_ops() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);

        let val = builder.ld_global_u32(addr);
        builder.st_global_u32(addr, val);

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].ty, PtxType::U32);
        assert_eq!(builder.instructions[1].ty, PtxType::U32);
    }

    #[test]
    fn test_global_u8_load() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);

        let val = builder.ld_global_u8(addr);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Ld);
        assert_eq!(builder.instructions[0].ty, PtxType::U8);
        assert_eq!(builder.instructions[0].state_space, Some(PtxStateSpace::Global));
        assert_eq!(val.ty(), PtxType::U32); // Zero-extended to u32
    }

    #[test]
    fn test_global_u16_load() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U64);

        let val = builder.ld_global_u16(addr);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Ld);
        assert_eq!(builder.instructions[0].ty, PtxType::U16);
        assert_eq!(builder.instructions[0].state_space, Some(PtxStateSpace::Global));
        assert_eq!(val.ty(), PtxType::U32); // Zero-extended
    }

    #[test]
    fn test_shared_u32_ops() {
        let mut builder = MockBuilder::new();
        let addr = builder.registers.allocate_virtual(PtxType::U32);

        let val = builder.ld_shared_u32(addr);
        builder.st_shared_u32(addr, val);

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].op, PtxOp::Ld);
        assert_eq!(builder.instructions[0].ty, PtxType::U32);
        assert_eq!(builder.instructions[0].state_space, Some(PtxStateSpace::Shared));
        assert_eq!(builder.instructions[1].op, PtxOp::St);
        assert_eq!(builder.instructions[1].state_space, Some(PtxStateSpace::Shared));
    }
}
