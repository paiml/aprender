//! In-place update operations for KernelBuilder.
//!
//! Provides in-place arithmetic, register copy, and accumulator operations
//! used in loops and reduction patterns where SSA would allocate new registers.

use super::super::instructions::{Operand, PtxInstruction, PtxOp, RoundingMode};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::KernelBuilder;

impl<'a> KernelBuilder<'a> {
    // ===== In-Place Updates (for loops) =====

    /// Add u32 immediate in-place: dst = dst + imm
    /// Used for loop counter updates where SSA would allocate a new register
    pub fn add_u32_inplace(&mut self, dst: VirtualReg, imm: u32) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(dst))
                .src(Operand::ImmU64(imm as u64)),
        );
    }

    /// Add f32 register in-place: dst = dst + src
    /// Used for accumulator updates in reduction loops
    /// Add register to u32 register in place (dst += src)
    pub fn add_u32_reg_inplace(&mut self, dst: VirtualReg, src: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.registers.extend_live_range(src);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(dst))
                .src(Operand::Reg(src)),
        );
    }

    /// Add f32 value in-place: dst = dst + src
    pub fn add_f32_inplace(&mut self, dst: VirtualReg, src: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Add, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(dst))
                .src(Operand::Reg(src))
                .rounding(RoundingMode::Rn),
        );
    }

    /// Shift right u32 in-place by immediate: dst = dst >> imm
    /// Used for stride halving in reduction loops
    ///
    /// NOTE: PTX requires .b32 (bitwise) type for shift ops, not .u32
    pub fn shr_u32_inplace(&mut self, dst: VirtualReg, imm: u32) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            // PTX requires .b32 for shift ops, not .u32
            PtxInstruction::new(PtxOp::Shr, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(dst))
                .src(Operand::ImmU64(imm as u64)),
        );
    }

    /// Fused multiply-add in-place: dst = a * b + dst
    /// Used for GEMM accumulation
    pub fn fma_f32_inplace(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Fma, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(dst))
                .rounding(RoundingMode::Rn),
        );
    }

    /// GH-561: Fused multiply-add in-place with f64 accumulator: dst = a * b + dst
    /// a and b are f32, promoted to f64 before FMA. dst is f64.
    /// Eliminates FP32 accumulation error that causes cosine=-0.005 on sm_121.
    pub fn fma_f64_acc_inplace(&mut self, dst: VirtualReg, a_f32: VirtualReg, b_f32: VirtualReg) {
        self.registers.extend_live_range(dst);
        // Promote f32 operands to f64
        let a_f64 = self.registers.allocate_virtual(PtxType::F64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F64)
                .dst(Operand::Reg(a_f64))
                .src(Operand::Reg(a_f32))
                .with_src_type(PtxType::F32),
        );
        let b_f64 = self.registers.allocate_virtual(PtxType::F64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F64)
                .dst(Operand::Reg(b_f64))
                .src(Operand::Reg(b_f32))
                .with_src_type(PtxType::F32),
        );
        // FMA in f64: dst = a_f64 * b_f64 + dst
        self.instructions.push(
            PtxInstruction::new(PtxOp::Fma, PtxType::F64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a_f64))
                .src(Operand::Reg(b_f64))
                .src(Operand::Reg(dst))
                .rounding(RoundingMode::Rn),
        );
    }

    /// GH-561: Initialize f64 accumulator to 0.0
    pub fn mov_f64_imm_zero(&mut self) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::F64)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmF64(0.0)),
        );
        dst
    }

    /// GH-561: Convert f64 accumulator to f32 result (round to nearest)
    pub fn cvt_f32_f64_rn(&mut self, src: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(src))
                .with_src_type(PtxType::F64)
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Max in-place: dst = max(dst, src)
    /// Used for online softmax running max
    pub fn max_f32_inplace(&mut self, dst: VirtualReg, src: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Max, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(dst))
                .src(Operand::Reg(src)),
        );
    }

    /// Copy f32 register: dst = src
    /// Used for accumulator state updates
    pub fn mov_f32_reg(&mut self, dst: VirtualReg, src: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(src)),
        );
    }

    /// Copy u32 register: dst = src
    /// Used for loop counter updates
    pub fn mov_u32_reg(&mut self, dst: VirtualReg, src: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(src)),
        );
    }

    /// Copy u64 register: dst = src
    pub fn mov_u64_reg(&mut self, dst: VirtualReg, src: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(src)),
        );
    }

    /// Multiply in-place: dst = dst * src
    /// Used for scaling operations
    pub fn mul_f32_inplace(&mut self, dst: VirtualReg, src: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mul, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(dst))
                .src(Operand::Reg(src))
                .rounding(RoundingMode::Rn),
        );
    }

    /// Divide in-place: dst = dst / src
    /// Used for normalization
    pub fn div_f32_inplace(&mut self, dst: VirtualReg, src: VirtualReg) {
        self.registers.extend_live_range(dst);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Div, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(dst))
                .src(Operand::Reg(src))
                .rounding(RoundingMode::Rn),
        );
    }
}
