//! Bitwise and select operations for KernelBuilder.
//!
//! Provides shift, AND, OR, XOR, and predicate-based select (selp) operations.

use super::super::instructions::{Operand, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::control::PtxControl;
use super::KernelBuilder;

impl<'a> KernelBuilder<'a> {
    /// Shift right u32 (logical shift)
    ///
    /// NOTE: PTX requires .b32 (bitwise) type for shift ops, not .u32
    pub fn shr_u32(&mut self, val: VirtualReg, shift: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            // PTX requires .b32 for shift ops, not .u32
            PtxInstruction::new(PtxOp::Shr, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::Reg(shift)),
        );
        dst
    }

    /// Shift right u32 by immediate (logical shift)
    ///
    /// Uses an immediate value for the shift amount, avoiding register clobbering issues.
    /// Use this in loops where the shift amount is constant to prevent SASS from
    /// reusing the shift register.
    pub fn shr_u32_imm(&mut self, val: VirtualReg, shift: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Shr, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmU64(shift as u64)),
        );
        dst
    }

    /// Bitwise AND u32 (register AND register)
    ///
    /// NOTE: PTX requires .b32 (bitwise) type for and/or/xor, not .u32
    pub fn and_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            // PTX requires .b32 for bitwise ops, not .u32
            PtxInstruction::new(PtxOp::And, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Bitwise OR u32 (register OR register)
    ///
    /// NOTE: PTX requires .b32 (bitwise) type for and/or/xor, not .u32
    pub fn or_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            // PTX requires .b32 for bitwise ops, not .u32
            PtxInstruction::new(PtxOp::Or, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Bitwise OR u32 into existing register (register reuse)
    pub fn or_u32_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::Or, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
    }

    /// Shift left u32 (register << register)
    ///
    /// NOTE: PTX requires .b32 (bitwise) type for shift ops, not .u32
    pub fn shl_u32(&mut self, val: VirtualReg, shift: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            // PTX requires .b32 for shift ops, not .u32
            PtxInstruction::new(PtxOp::Shl, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::Reg(shift)),
        );
        dst
    }

    /// Shift left u32 by immediate (register << immediate)
    pub fn shl_u32_imm(&mut self, val: VirtualReg, shift: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Shl, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmU64(shift as u64)),
        );
        dst
    }

    /// Select based on predicate: dst = pred ? true_val : false_val
    ///
    /// PTX format: selp.u32 d, a, b, p
    /// where d = destination, a = value if true, b = value if false, p = predicate
    pub fn selp_u32(
        &mut self,
        pred: VirtualReg,
        true_val: VirtualReg,
        false_val: VirtualReg,
    ) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Selp, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(true_val))
                .src(Operand::Reg(false_val))
                .src(Operand::Reg(pred)),
        );
        dst
    }

    /// Select f32 based on predicate: dst = pred ? true_val : false_val
    ///
    /// PTX format: selp.f32 d, a, b, p
    /// PAR-062: Used by ArgMax kernel for conditional max tracking
    pub fn selp_f32(
        &mut self,
        pred: VirtualReg,
        true_val: VirtualReg,
        false_val: VirtualReg,
    ) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Selp, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(true_val))
                .src(Operand::Reg(false_val))
                .src(Operand::Reg(pred)),
        );
        dst
    }

    /// Select u64 based on predicate: dst = pred ? true_val : false_val
    ///
    /// PTX format: selp.b64 d, a, b, p
    pub fn selp_u64(
        &mut self,
        pred: VirtualReg,
        true_val: VirtualReg,
        false_val: VirtualReg,
    ) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::B64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Selp, PtxType::B64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(true_val))
                .src(Operand::Reg(false_val))
                .src(Operand::Reg(pred)),
        );
        dst
    }

    // setp_gt_f32 is provided by PtxComparison trait (comparison.rs)

    /// AND two predicates: dst = a AND b
    /// Used for combining bounds checks (PARITY-114)
    pub fn and_pred(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::Pred);
        self.instructions.push(
            PtxInstruction::new(PtxOp::And, PtxType::Pred)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Get shared memory base pointer
    ///
    /// PAR-062: Returns base address of shared memory for this block
    pub fn shared_ptr(&mut self) -> VirtualReg {
        self.shared_base_addr()
    }

    /// Bitwise AND u32 with immediate
    ///
    /// PAR-062: Used for lane_id extraction (tid & 31)
    pub fn and_u32_imm(&mut self, a: VirtualReg, imm: u32) -> VirtualReg {
        let imm_reg = self.mov_u32_imm(imm);
        self.and_u32(a, imm_reg)
    }
}
