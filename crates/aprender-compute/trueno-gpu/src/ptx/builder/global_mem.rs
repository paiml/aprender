//! Global memory load/store operations for KernelBuilder.
//!
//! Extracted from mod.rs for PMAT File Health compliance.
//! Contains typed global memory access: f32, u8, u16, u32, u64, vectorized v4.

use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};
use crate::ptx::registers::VirtualReg;
use crate::ptx::types::{PtxStateSpace, PtxType};

use super::{KernelBuilder, PtxArithmetic};

impl<'a> KernelBuilder<'a> {
    // ===== Memory Operations (vectorized - not in traits) =====

    /// Load f32 from global memory (kept for compatibility - delegates to trait)
    pub fn ld_global_f32(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::F32)
                .space(PtxStateSpace::Global)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    /// Store f32 to global memory
    pub fn st_global_f32(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::F32)
                .space(PtxStateSpace::Global)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
        );
    }

    /// Store 2 consecutive f32 values to global memory (vectorized, 8-byte store).
    /// PTX: st.global.v2.f32 [addr], {val0, val1};
    /// Address should be 8-byte aligned for optimal performance.
    pub fn st_global_f32_v2(&mut self, addr: VirtualReg, val0: VirtualReg, val1: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::V2F32)
                .space(PtxStateSpace::Global)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val0))
                .src(Operand::Reg(val1)),
        );
    }

    /// Load 4 consecutive f32 values from global memory (vectorized, 16-byte load)
    ///
    /// Returns 4 registers containing the loaded values.
    /// Address must be 16-byte aligned for optimal performance.
    ///
    /// PTX: ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [addr];
    pub fn ld_global_f32_v4(&mut self, addr: VirtualReg) -> [VirtualReg; 4] {
        let r0 = self.registers.allocate_virtual(PtxType::F32);
        let r1 = self.registers.allocate_virtual(PtxType::F32);
        let r2 = self.registers.allocate_virtual(PtxType::F32);
        let r3 = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::V4F32)
                .space(PtxStateSpace::Global)
                .dst(Operand::Reg(r0))
                .dst(Operand::Reg(r1))
                .dst(Operand::Reg(r2))
                .dst(Operand::Reg(r3))
                .src(Operand::Reg(addr)),
        );
        [r0, r1, r2, r3]
    }

    /// Load u32 from global memory
    pub fn ld_global_u32(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Load u32 from global memory into existing register (register reuse)
    pub fn ld_global_u32_into(&mut self, dst: VirtualReg, addr: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .space(PtxStateSpace::Global),
        );
    }

    /// Store u32 to global memory
    pub fn st_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U32)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
    }

    /// Load u64 from global memory (PAR-118: for pointer arrays in batched attention)
    pub fn ld_global_u64(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Store u64 to global memory
    pub fn st_global_u64(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U64)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
    }

    /// Load u8 from global memory
    ///
    /// NOTE: PTX does not support .u8 register types (minimum is 16-bit).
    /// We allocate a U16 register and use ld.global.u8 which zero-extends
    /// the loaded byte into the 16-bit register.
    pub fn ld_global_u8(&mut self, addr: VirtualReg) -> VirtualReg {
        // CRITICAL: PTX requires registers to be at least 16-bit
        // ld.global.u8 zero-extends the byte into the U16 register
        let dst = self.registers.allocate_virtual(PtxType::U16);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U8)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Store u8 to global memory
    ///
    /// NOTE: PTX requires stores to come from at least a 16-bit register.
    /// The low 8 bits of the source register are stored to the address.
    pub fn st_global_u8(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U8)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
    }

    /// Store u16 to global memory
    pub fn st_global_u16(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U16)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
    }

    /// Load u32 from potentially unaligned global memory address.
    ///
    /// Uses 4 byte loads + `bfi.b32` to assemble a u32, avoiding
    /// `ld.global.u32` alignment requirements (4-byte aligned).
    /// Required for Q6K super-blocks (210 bytes each, not 4-byte aligned).
    ///
    /// sm_87 (Jetson Orin) faults on misaligned ld.global.u32 with
    /// CUDA_ERROR_MISALIGNED_ADDRESS (716).
    ///
    /// GH-131: Optimized from shl+or (9 instructions) to bfi.b32 (3 instructions)
    /// for the byte assembly step. Saves 6 instructions per call × 4 calls per
    /// Q6K super-block = 24 fewer instructions per super-block.
    pub fn ld_global_u32_unaligned(&mut self, addr: VirtualReg) -> VirtualReg {
        // Load 4 consecutive bytes
        let b0 = self.ld_global_u8(addr);
        let off1 = self.mov_u64_imm(1);
        let addr1 = self.add_u64(addr, off1);
        let b1 = self.ld_global_u8(addr1);
        let off2 = self.mov_u64_imm(2);
        let addr2 = self.add_u64(addr, off2);
        let b2 = self.ld_global_u8(addr2);
        let off3 = self.mov_u64_imm(3);
        let addr3 = self.add_u64(addr, off3);
        let b3 = self.ld_global_u8(addr3);

        // Convert u8 (in u16 registers) to u32
        let w0 = self.cvt_u32_u8(b0); // byte 0 → bits [7:0]
        let w1 = self.cvt_u32_u8(b1);
        let w2 = self.cvt_u32_u8(b2);
        let w3 = self.cvt_u32_u8(b3);

        // Assemble little-endian u32 using bfi.b32 (3 instructions vs 9 with shl+or)
        // bfi.b32 inserts `len` bits from `insert` into `base` at position `start`
        let t1 = self.bfi_b32(w1, w0, 8, 8); // insert byte 1 at bits [15:8]
        let t2 = self.bfi_b32(w2, t1, 16, 8); // insert byte 2 at bits [23:16]
        self.bfi_b32(w3, t2, 24, 8) // insert byte 3 at bits [31:24]
    }

    /// Load u16 from global memory (for f16 as raw bits)
    pub fn ld_global_u16(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U16);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U16)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .space(PtxStateSpace::Global),
        );
        dst
    }
}
