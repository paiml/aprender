//! Warp-level operations for KernelBuilder.
//!
//! Provides warp shuffle, ballot, population count, bit find,
//! and count leading zeros operations used in cooperative warp algorithms.

use super::super::instructions::{Operand, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::control::PtxControl;
use super::KernelBuilder;

impl<'a> KernelBuilder<'a> {
    /// Warp shuffle down (for reductions)
    /// Format: shfl.sync.down.b32 dst, src, delta, clamp, membermask
    pub fn shfl_down_f32(&mut self, val: VirtualReg, offset: u32, mask: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::ShflDown, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmU64(offset as u64))
                .src(Operand::ImmU64(31)) // clamp to warp size
                .src(Operand::ImmU64(mask as u64)), // membermask
        );
        dst
    }

    /// Warp shuffle indexed (for broadcasts - gets value from specific lane)
    ///
    /// Format: shfl.sync.idx.b32 dst, src, srcLane, c, membermask
    ///
    /// PTX ISA: c[4:0] = maxLane. Read succeeds when srcLane <= maxLane.
    /// Use c=31 for full-warp broadcasts (any lane 0-31 readable).
    pub fn shfl_idx_f32(&mut self, val: VirtualReg, src_lane: u32, mask: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::ShflIdx, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmU64(src_lane as u64))
                .src(Operand::ImmU64(31)) // maxLane=31: allow reads from any lane
                .src(Operand::ImmU64(mask as u64)), // membermask
        );
        dst
    }

    /// Warp shuffle indexed for u32 values (broadcasts, lane selection)
    ///
    /// Format: shfl.sync.idx.b32 dst, src, srcLane, c, membermask
    ///
    /// PTX ISA: c[4:0] = maxLane. Read succeeds when srcLane <= maxLane.
    /// Use c=31 for full-warp broadcasts (any lane 0-31 readable).
    pub fn shfl_idx_u32(&mut self, val: VirtualReg, src_lane: u32, mask: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::ShflIdx, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmU64(src_lane as u64))
                .src(Operand::ImmU64(31)) // maxLane=31: allow reads from any lane
                .src(Operand::ImmU64(mask as u64)), // membermask
        );
        dst
    }

    /// Warp shuffle indexed with dynamic lane (from register)
    ///
    /// Format: shfl.sync.idx.b32 dst, src, srcLane, c, membermask
    /// srcLane comes from a register instead of immediate.
    pub fn shfl_idx_u32_reg(
        &mut self,
        val: VirtualReg,
        src_lane_reg: VirtualReg,
        mask: u32,
    ) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::ShflIdx, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::Reg(src_lane_reg))
                .src(Operand::ImmU64(31)) // maxLane=31: allow reads from any lane
                .src(Operand::ImmU64(mask as u64)), // membermask
        );
        dst
    }

    // ===== KF-002: Warp Vote and Bit Manipulation =====

    /// Warp ballot - returns bitmask of lanes where predicate is true
    ///
    /// Format: vote.sync.ballot.b32 dst, pred, membermask;
    ///
    /// Returns a u32 where bit i is set if lane i has predicate true.
    /// Used for finding which lanes have matching hash values in LZ4 compression.
    pub fn ballot_sync(&mut self, pred: VirtualReg, mask: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::VoteBallot, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(pred))
                .src(Operand::ImmU64(mask as u64)),
        );
        dst
    }

    /// Population count - counts number of 1 bits in a u32
    ///
    /// Format: popc.b32 dst, src;
    ///
    /// Used for counting matches in ballot results.
    pub fn popc_u32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Popc, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Find first set bit (1-indexed, returns 0 if input is 0)
    ///
    /// Format: bfind.u32 dst, src;
    ///
    /// Returns position of most significant set bit (0 if src==0).
    /// To get lane number from ballot: use bfind or clz+subtract.
    pub fn bfind_u32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Bfind, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Count leading zeros
    ///
    /// Format: clz.b32 dst, src;
    ///
    /// Used with ballot to find first matching lane: lane = 31 - clz(ballot)
    pub fn clz_u32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Clz, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Warp shuffle down for u32: exchange with lane + offset
    ///
    /// PTX format: shfl.sync.down.b32 d, a, offset, clamp, mask
    /// PAR-062: Used by ArgMax kernel for warp-level index reduction
    pub fn shfl_down_u32(&mut self, val: VirtualReg, offset: u32, mask: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::ShflDown, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmU64(offset as u64))
                .src(Operand::ImmU64(31)) // clamp to warp size
                .src(Operand::ImmU64(mask as u64)),
        );
        dst
    }

    /// Bit field insert: insert `len` bits from `insert` into `base` at position `start`
    ///
    /// PTX: `bfi.b32 dst, insert, base, start, len;`
    /// dst = base with bits [start..start+len-1] replaced by insert[0..len-1]
    ///
    /// GH-131: Used to pack bytes into u32 for unaligned Q6K loads on sm_87.
    /// Replaces 3 instructions (mov+shl+or) with 1 instruction per byte insertion.
    pub fn bfi_b32(
        &mut self,
        insert: VirtualReg,
        base: VirtualReg,
        start: u32,
        len: u32,
    ) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Bfi, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(insert))
                .src(Operand::Reg(base))
                .src(Operand::ImmI64(start as i64))
                .src(Operand::ImmI64(len as i64)),
        );
        dst
    }

    /// Load f32 immediate constant
    ///
    /// PAR-062: Used for NEG_INFINITY initialization
    pub fn const_f32(&mut self, val: f32) -> VirtualReg {
        self.mov_f32_imm(val)
    }

    /// Load u32 immediate constant
    ///
    /// PAR-062: Used for index initialization
    pub fn const_u32(&mut self, val: u32) -> VirtualReg {
        self.mov_u32_imm(val)
    }
}
