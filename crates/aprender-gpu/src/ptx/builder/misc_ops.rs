//! Miscellaneous operations for KernelBuilder
//!
//! Contains remaining direct operations on KernelBuilder that don't fit into
//! the extension trait categories (arithmetic, comparison, control, memory, sync).
//! Includes: min/max, division/remainder, exponential, trig, negation, absolute value,
//! u64 operations, branching, and signed integer operations.

use super::{KernelBuilder, Operand, Predicate, PtxInstruction, PtxOp, PtxType, VirtualReg};

impl<'a> KernelBuilder<'a> {
    /// Min u32 of two values
    pub fn min_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Min, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Subtract u32 registers
    pub fn sub_u32_reg(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Sub, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Exp f32 (exponential)
    pub fn ex2_f32(&mut self, val: VirtualReg) -> VirtualReg {
        // PTX has ex2 (base 2), we scale input by log2(e) for natural exp
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ex2, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Multiply u32
    pub fn mul_u32(&mut self, a: VirtualReg, b: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::ImmU64(b as u64)),
        );
        dst
    }

    /// Multiply u32 (register * register)
    pub fn mul_u32_reg(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Add u32 (register + register)
    pub fn add_u32_reg(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Reciprocal square root f32: dst = 1/sqrt(val)
    pub fn rsqrt_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Rsqrt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Sine f32 (approximate): dst = sin(val)
    /// PAR-060: Used for RoPE (Rotary Position Embedding) kernel
    pub fn sin_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Sin, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Cosine f32 (approximate): dst = cos(val)
    /// PAR-060: Used for RoPE (Rotary Position Embedding) kernel
    pub fn cos_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cos, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Negate f32: dst = -val
    /// PAR-060: Used for RoPE (Rotary Position Embedding) kernel
    pub fn neg_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Neg, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Integer division u32 (immediate divisor)
    pub fn div_u32(&mut self, a: VirtualReg, b: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Div, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::ImmU64(b as u64)),
        );
        dst
    }

    /// Integer division u32 (register divisor)
    ///
    /// Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-001)
    /// Enables runtime dimension parameters instead of baked immediates.
    pub fn div_u32_reg(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Div, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Integer remainder (modulo) u32 (immediate divisor)
    pub fn rem_u32(&mut self, a: VirtualReg, b: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Rem, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::ImmU64(b as u64)),
        );
        dst
    }

    /// Integer remainder (modulo) u32 (register divisor)
    ///
    /// Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-001)
    /// Enables runtime dimension parameters instead of baked immediates.
    pub fn rem_u32_reg(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Rem, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Move immediate u64 value
    pub fn mov_u64_imm(&mut self, val: u64) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmU64(val)),
        );
        dst
    }

    /// Multiply u64 by immediate
    pub fn mul_u64(&mut self, a: VirtualReg, b: u64) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::ImmU64(b)),
        );
        dst
    }

    /// Multiply u64 (register * register)
    pub fn mul_u64_reg(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Branch if predicate is false (negated predicate)
    pub fn branch_if_not(&mut self, pred: VirtualReg, label: &str) {
        let predicate = Predicate { reg: pred, negated: true };
        self.instructions
            .push(PtxInstruction::new(PtxOp::Bra, PtxType::B32).predicated(predicate).label(label));
    }

    // =========================================================================
    // COALESCED GEMV SUPPORT - DECODER THROUGHPUT SPEC S5.3
    // =========================================================================

    /// Multiply low u32 (register * register -> u32)
    ///
    /// Unlike mul_wide, this keeps only the low 32 bits of the result.
    /// Used for computing block offsets: col_base = block_id * block_size
    pub fn mul_lo_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    // =========================================================================
    // PAR-063-V4: Additional ops for Q8 quantization kernels
    // =========================================================================

    /// Signed 32-bit multiply (low 32 bits of result)
    pub fn mul_lo_s32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::S32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mul, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Absolute value of f32
    pub fn abs_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Abs, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Minimum of two signed 32-bit integers
    pub fn min_s32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::S32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Min, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Maximum of two signed 32-bit integers
    pub fn max_s32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::S32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Max, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }
}
