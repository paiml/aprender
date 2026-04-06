//! PTX Module and Kernel Builder
//!
//! Provides a fluent builder API for constructing PTX modules and kernels.
//!
//! ## Extension Traits
//!
//! The builder functionality is split into focused extension traits for maintainability:
//!
//! - [`PtxArithmetic`]: Add, sub, mul, fma, dp4a, transcendentals
//! - [`PtxComparison`]: setp operations for predicates
//! - [`PtxMemory`]: Global and shared memory load/store
//! - [`PtxControl`]: Labels, branches, returns, immediate moves
//! - [`PtxSync`]: Barriers, shuffles, warp votes, bit manipulation
//! - [`PtxAtomic`]: Atomic memory operations
//!
//! All traits are automatically implemented for `KernelBuilder` via blanket impls.

// Extension trait modules
mod arithmetic;
mod atomic;
mod comparison;
mod control;
mod core;
mod emit;
mod memory;
mod sync;

// Impl-block submodules (PMAT File Health split)
mod atomics_debug;
mod bitwise_ops;
mod conversions;
mod generic_mem;
mod global_mem;
mod inplace_ops;
mod misc_ops;
mod precise_math;
mod tensor_core;
mod warp_vote;

// Type definition modules (PMAT File Health split)
mod ptx_module;

// Re-export extension traits for easy use
pub use arithmetic::PtxArithmetic;
pub use atomic::PtxAtomic;
pub use comparison::PtxComparison;
pub use control::PtxControl;
pub use core::KernelBuilderCore;
pub use memory::PtxMemory;
pub use sync::PtxSync;

// Re-export types from ptx_module
pub use ptx_module::{KernelParam, PtxKernel, PtxModule};

use super::instructions::{Operand, Predicate, PtxInstruction, PtxOp};
use super::registers::{PtxReg, RegisterAllocator, VirtualReg};
use super::types::{PtxStateSpace, PtxType};

/// Macro for dp4a operations (in-place variant)
macro_rules! impl_dp4a_inplace {
    ($fn_name:ident, $op:ident, $ty:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $fn_name(&mut self, acc: VirtualReg, a: VirtualReg, b: VirtualReg) {
            self.instructions.push(
                PtxInstruction::new(PtxOp::$op, PtxType::$ty)
                    .dst(Operand::Reg(acc))
                    .src(Operand::Reg(a))
                    .src(Operand::Reg(b))
                    .src(Operand::Reg(acc)),
            );
        }
    };
}

/// Kernel builder context (passed to build closure)
pub struct KernelBuilder<'a> {
    /// Register allocator
    pub(crate) registers: &'a mut RegisterAllocator,
    /// Instructions
    pub(crate) instructions: Vec<PtxInstruction>,
    /// Labels
    pub(crate) labels: Vec<String>,
}

// Implement KernelBuilderCore to enable extension traits
impl<'a> core::KernelBuilderCore for KernelBuilder<'a> {
    fn registers_mut(&mut self) -> &mut RegisterAllocator {
        self.registers
    }

    fn instructions_mut(&mut self) -> &mut Vec<PtxInstruction> {
        &mut self.instructions
    }

    fn labels_mut(&mut self) -> &mut Vec<String> {
        &mut self.labels
    }
}

impl<'a> KernelBuilder<'a> {
    pub(crate) fn new(registers: &'a mut RegisterAllocator) -> Self {
        Self { registers, instructions: Vec::new(), labels: Vec::new() }
    }

    // ===== Special Registers =====

    /// Read a special register into a virtual register
    pub fn special_reg(&mut self, reg: PtxReg) -> VirtualReg {
        let vreg = self.registers.allocate_virtual(reg.data_type());
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, reg.data_type())
                .dst(Operand::Reg(vreg))
                .src(Operand::SpecialReg(reg)),
        );
        vreg
    }

    // ===== Parameter Loading =====

    /// Load a u32 parameter
    pub fn load_param_u32(&mut self, name: &str) -> VirtualReg {
        let vreg = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::LdParam, PtxType::U32)
                .dst(Operand::Reg(vreg))
                .src(Operand::Param(name.to_string())),
        );
        vreg
    }

    /// Load a u64 parameter
    pub fn load_param_u64(&mut self, name: &str) -> VirtualReg {
        let vreg = self.registers.allocate_virtual(PtxType::U64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::LdParam, PtxType::U64)
                .dst(Operand::Reg(vreg))
                .src(Operand::Param(name.to_string())),
        );
        vreg
    }

    /// Load an f32 parameter
    pub fn load_param_f32(&mut self, name: &str) -> VirtualReg {
        let vreg = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::LdParam, PtxType::F32)
                .dst(Operand::Reg(vreg))
                .src(Operand::Param(name.to_string())),
        );
        vreg
    }

    // ===== Register Reuse Operations (not in traits) =====

    /// Move u64 immediate into existing register (register reuse)
    pub fn mov_u64_into(&mut self, dst: VirtualReg, val: u64) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmU64(val)),
        );
    }

    /// Move u32 immediate into existing register (register reuse)
    pub fn mov_u32_into(&mut self, dst: VirtualReg, val: u32) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmI64(val as i64)),
        );
    }

    // ===== Immediate Arithmetic (not in traits - different signatures) =====

    /// Add u32 with immediate (different from trait version which takes two registers)
    pub fn add_u32(&mut self, a: VirtualReg, b: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::ImmU64(b as u64)),
        );
        dst
    }

    /// PAR-063: Dot product of 4 x u8 vectors with accumulate
    ///
    /// Computes: d = dot4(a, b) + c
    /// where a and b are u32 containing 4 x u8 values each
    ///
    /// This is the key SIMD instruction used by llama.cpp for Q4K inference.
    /// Each dp4a computes 4 multiply-adds in one instruction.
    ///
    /// # Example
    /// ```ignore
    /// let a = 0x01020304u32;  // bytes [4, 3, 2, 1]
    /// let b = 0x05060708u32;  // bytes [8, 7, 6, 5]
    /// let c = 0u32;           // accumulator
    /// // d = 4*8 + 3*7 + 2*6 + 1*5 = 32 + 21 + 12 + 5 = 70
    /// ```
    pub fn dp4a_u32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Dp4a, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
        dst
    }

    // ===== DP4A In-Place Operations (Macro-Generated) =====
    // Dot product of 4 x u8/s8 vectors with accumulate, generated via macro.
    // PAR-063: Key SIMD instruction for Q4K inference (llama.cpp pattern).

    impl_dp4a_inplace!(
        dp4a_u32_inplace,
        Dp4a,
        U32,
        "DP4A u32 in-place: acc += dot4(a, b) where a,b are packed u8x4"
    );
    impl_dp4a_inplace!(
        dp4a_u32_s32_inplace,
        Dp4aUS,
        S32,
        "DP4A u32\u{00d7}s32 in-place: acc += dot4(u8x4, s8x4)"
    );
    impl_dp4a_inplace!(
        dp4a_s32_inplace,
        Dp4aS32,
        S32,
        "DP4A s32 in-place: acc += dot4(s8x4, s8x4)"
    );

    /// Barrier synchronization (all threads in block must reach this point)
    pub fn bar_sync(&mut self, barrier_id: u32) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::Bar, PtxType::B32).label(format!("sync {}", barrier_id)),
        );
    }

    /// Memory fence at CTA (thread block) level
    ///
    /// Ensures all prior memory operations are visible to other threads in the block.
    /// PTX: membar.cta;
    pub fn membar_cta(&mut self) {
        self.instructions
            .push(PtxInstruction::new(PtxOp::MemBar, PtxType::B32).label("cta".to_string()));
    }

    /// Memory fence at GPU level
    ///
    /// Ensures all prior memory operations are visible to other threads on the GPU.
    /// PTX: membar.gl;
    pub fn membar_gl(&mut self) {
        self.instructions
            .push(PtxInstruction::new(PtxOp::MemBar, PtxType::B32).label("gl".to_string()));
    }

    // ===== Shared Memory Operations =====
    // These operations are provided by the PtxMemory extension trait in memory.rs.
    // KernelBuilder implements KernelBuilderCore, so it automatically gets all
    // PtxMemory methods via blanket impl. Available methods:
    //   - ld_shared_f32, st_shared_f32
    //   - ld_shared_u32, st_shared_u32
    //   - ld_shared_u32_volatile
    //   - st_shared_f16

    /// Store u16 to shared memory (for hash table positions)
    ///
    /// Format: st.shared.u16 [addr], val;
    ///
    /// Used for storing 16-bit positions in LZ4 hash table (2048 entries x 2 bytes).
    /// IMPORTANT: This is WRITE-ONLY usage - no ld.shared.u16 needed, avoiding F081!
    pub fn st_shared_u16(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U16)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Shared),
        );
    }

    // Additional operations split into submodules (PMAT File Health):
    // - conversions.rs: cvt_*, floor_f32, rcp_f32, mov_u32_inplace, s32 ops
    // - bitwise_ops.rs: shifts, AND, OR, selp, and_pred, setp_gt_f32, shared_ptr
    // - warp_vote.rs: shfl_down/idx, ballot, popc, bfind, clz, const_f32/u32
    // - inplace_ops.rs: in-place arithmetic, register copies, fma/max/mul/div inplace
    // - tensor_core.rs: WMMA load/store/mma, F16 conversions, F16 global memory
    // - generic_mem.rs: generic addressing, shared_base_addr, predicated loads
    // - global_mem.rs: ld_global_*/st_global_* typed memory ops
    // - precise_math.rs: sin_f32_precise, cos_f32_precise, ex2_f32_precise
    // - atomics_debug.rs: atom_*_global/shared, emit_debug_marker/value
    // - misc_ops.rs: min/max, div/rem, exp, trig, neg, abs, u64 ops, branch_if_not
}

// Tests (~3K lines extracted for TDG compliance)
#[cfg(test)]
mod tests;
