//! PTX Arithmetic Operations Extension Trait.
//!
//! Provides arithmetic operations: add, sub, mul, fma, mad, dp4a, etc.

use super::super::instructions::{Operand, PtxInstruction, PtxOp, RoundingMode};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::core::KernelBuilderCore;

/// Extension trait for PTX arithmetic operations.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::ptx::builder::{KernelBuilder, PtxArithmetic};
///
/// fn build_kernel(kb: &mut KernelBuilder) {
///     let a = kb.load_param_f32("a");
///     let b = kb.load_param_f32("b");
///     let sum = kb.add_f32(a, b);  // From PtxArithmetic trait
/// }
/// ```
pub trait PtxArithmetic: KernelBuilderCore {
    // ===== Integer Arithmetic =====

    /// Multiply-add low: dst = a * b + c
    fn mad_lo_u32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::MadLo, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
        dst
    }

    /// Multiply wide (u32 * u32 -> u64)
    fn mul_wide_u32(&mut self, a: VirtualReg, b: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::MulWide, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::ImmU64(b as u64)),
        );
        dst
    }

    /// Multiply wide (u32 * u32 -> u64) with register operands
    fn mul_wide_u32_reg(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::MulWide, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Add u64
    fn add_u64(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Add u64 into existing register (register reuse)
    fn add_u64_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
    }

    /// Add u32
    fn add_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Add u32 into existing register (register reuse)
    fn add_u32_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
    }

    /// Subtract u32
    fn sub_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Sub, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Multiply u32
    fn mul_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    // ===== Floating Point Arithmetic =====

    /// Add f32
    fn add_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Subtract f32
    fn sub_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Sub, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Multiply f32
    fn mul_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mul, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Divide f32
    fn div_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Div, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Fused multiply-add: dst = a * b + c
    fn fma_f32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Fma, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Fused multiply-add into existing register
    fn fma_f32_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg, c: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Fma, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c))
                .rounding(RoundingMode::Rn),
        );
    }

    // ===== F64 Arithmetic (GH-561: FP64 accumulators for GEMM parity) =====

    /// Add f64
    fn add_f64(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::F64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Fused multiply-add f64: dst = a * b + c
    fn fma_f64(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Fma, PtxType::F64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Convert f32 → f64 (promotion, exact)
    fn cvt_f64_f32(&mut self, src: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(src))
                .with_src_type(PtxType::F32),
        );
        dst
    }

    /// Convert f64 → f32 (truncation, round to nearest)
    fn cvt_f32_f64(&mut self, src: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(src))
                .with_src_type(PtxType::F64)
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Negate f32
    fn neg_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Neg, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Absolute value f32
    fn abs_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Abs, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Square root f32
    fn sqrt_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Sqrt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Reciprocal square root (approximate)
    fn rsqrt_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Rsqrt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Reciprocal (approximate)
    fn rcp_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Rcp, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Exponential base 2 (approximate)
    fn ex2_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ex2, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Logarithm base 2 (approximate)
    fn lg2_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Lg2, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Sine (approximate)
    fn sin_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Sin, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Cosine (approximate)
    fn cos_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Cos, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Minimum f32
    fn min_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Min, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Maximum f32
    fn max_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Max, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    // ===== DP4A - Dot Product of 4 8-bit integers =====

    /// DP4A: signed dot product of 4 8-bit integers, accumulated into 32-bit
    /// dst = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c
    fn dp4a_s32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::S32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Dp4a, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
        dst
    }

    /// DP4A: unsigned dot product of 4 8-bit integers
    fn dp4a_u32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Dp4a, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
        dst
    }

    /// DP4A: signed dot product, accumulating into existing register
    fn dp4a_s32_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg, c: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Dp4a, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
    }
}

// Blanket implementation - any type implementing KernelBuilderCore gets PtxArithmetic
impl<T: KernelBuilderCore> PtxArithmetic for T {}

#[cfg(test)]
mod tests;
