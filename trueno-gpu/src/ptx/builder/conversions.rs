//! Type conversion operations for KernelBuilder.
//!
//! Provides all cvt_* operations for converting between PTX types:
//! u8, u16, u32, u64, s32, f32, and sign extension helpers.

use super::super::instructions::{Operand, PtxInstruction, PtxOp, RoundingMode};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::comparison::PtxComparison;
use super::control::PtxControl;
use super::KernelBuilder;

impl<'a> KernelBuilder<'a> {
    /// Convert u32 to u64 (zero extend)
    pub fn cvt_u64_u32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert u32 to u64 into existing register (register reuse)
    pub fn cvt_u64_u32_into(&mut self, dst: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
    }

    /// Convert u64 to u32 (truncate)
    pub fn cvt_u32_u64(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert u32 to f32
    pub fn cvt_f32_u32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Convert signed int8 to signed int32: dst = sext(val)
    /// Used for Q8_0 dequantization (int8 quantized values)
    /// Note: Input is u8 (from ld.global.u8), we do manual sign extension
    pub fn cvt_s32_s8(&mut self, val: VirtualReg) -> VirtualReg {
        // First convert u8 -> u32 (zero extend)
        let u32_val = self.cvt_u32_u8(val);
        // For sign extension: if val >= 128, subtract 256
        // signed = unsigned - ((unsigned >= 128) ? 256 : 0)
        let const_128 = self.mov_u32_imm(128);
        let is_negative = self.setp_ge_u32(u32_val, const_128);
        let const_256 = self.mov_u32_imm(256);
        let zero = self.mov_u32_imm(0);
        // Select 256 if negative, else 0
        let adjust = self.selp_u32(is_negative, const_256, zero);
        // Compute signed value
        self.sub_u32_reg(u32_val, adjust)
    }

    /// Convert signed int32 to f32: dst = (f32)val
    /// Used for Q8_0 dequantization after s8->s32 conversion
    /// Emits cvt.rn.f32.s32 which interprets the source bits as signed
    pub fn cvt_f32_s32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .with_src_type(PtxType::S32) // Force .s32 source type
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Floor f32: dst = floor(val)
    /// Uses cvt.rmi.f32.f32 (round toward minus infinity)
    pub fn floor_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F32)
                .with_src_type(PtxType::F32)
                .rounding(RoundingMode::Rmi) // round-to-integer-toward-minus-infinity (floor)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert u8 to u32 (zero extend)
    pub fn cvt_u32_u8(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert u16 to u32 (zero extend)
    pub fn cvt_u32_u16(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert u32 to u16 (truncate)
    ///
    /// Takes the low 16 bits of the u32 value.
    /// Use this before storing u32 values to u16 memory locations.
    pub fn cvt_u16_u32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U16);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::U16)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert f32 to s32 with round-to-nearest-integer
    pub fn cvt_rni_s32_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::S32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .rounding(RoundingMode::Rni),
        );
        dst
    }

    /// Move immediate to s32 register
    pub fn mov_s32_imm(&mut self, val: i32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::S32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmI64(i64::from(val))),
        );
        dst
    }

    /// Reinterpret u32 bits as s32 (no instruction, just type change)
    pub fn mov_s32_from_u32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::S32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert s32 to u8 (truncate to low 8 bits)
    pub fn cvt_u8_s32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U8);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::U8)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert u8 to s32 with sign extension
    pub fn cvt_s32_u8_sx(&mut self, val: VirtualReg) -> VirtualReg {
        // For sign extension, we treat the u8 as s8 and extend to s32
        let dst = self.registers.allocate_virtual(PtxType::S32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Convert u32 to s32 (reinterpret bits)
    pub fn cvt_s32_u32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::S32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Reciprocal approximation (1/x)
    pub fn rcp_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Rcp, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Move u32 immediate into existing register
    pub fn mov_u32_inplace(&mut self, dst: VirtualReg, val: u32) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmU64(u64::from(val))),
        );
    }
}
