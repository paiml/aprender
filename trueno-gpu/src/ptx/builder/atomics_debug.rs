//! Atomic operations and debug helpers for KernelBuilder.
//!
//! Extracted from mod.rs for PMAT File Health compliance.
//! Contains global/shared atomic operations and debug marker/value emission.

use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};
use crate::ptx::registers::VirtualReg;
use crate::ptx::types::{PtxStateSpace, PtxType};

use super::arithmetic::PtxArithmetic;
use super::control::PtxControl;
use super::KernelBuilder;

impl<'a> KernelBuilder<'a> {
    // ========================================================================
    // ATOMIC OPERATIONS - For debugging and synchronization
    // ========================================================================

    /// Atomic add to global memory, returns old value
    ///
    /// PTX: atom.global.add.u32 dst, [addr], val
    /// Atomically: old = *addr; *addr = old + val; return old
    pub fn atom_add_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::AtomAdd, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic exchange on global memory, returns old value
    ///
    /// PTX: atom.global.exch.u32 dst, [addr], val
    /// Atomically: old = *addr; *addr = val; return old
    pub fn atom_exch_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::AtomExch, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic min on global memory, returns old value
    ///
    /// PTX: atom.global.min.u32 dst, [addr], val
    pub fn atom_min_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::AtomMin, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic max on global memory, returns old value
    ///
    /// PTX: atom.global.max.u32 dst, [addr], val
    pub fn atom_max_global_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::AtomMax, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Atomic exchange on shared memory, returns old value
    ///
    /// PTX: atom.shared.exch.u32 dst, [addr], val
    /// Atomically: old = *addr; *addr = val; return old
    ///
    /// NOTE: This is a workaround for a ptxas bug where regular st.shared
    /// with computed addresses crashes the JIT compiler.
    pub fn atom_exch_shared_u32(&mut self, addr: VirtualReg, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::AtomExch, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Shared),
        );
        dst
    }

    // ========================================================================
    // DEBUG HELPERS - Printf-style debugging for PTX kernels
    // ========================================================================

    /// Emit a debug marker to a debug buffer
    ///
    /// This atomically increments a counter at debug_buf[0] and writes
    /// the marker value to debug_buf[old_counter + 1].
    ///
    /// Usage:
    /// - Pass a debug buffer with at least (max_markers + 1) u32 elements
    /// - debug_buf[0] = counter (starts at 0)
    /// - debug_buf[1..] = marker values written by emit_debug_marker
    ///
    /// Returns the slot index where the marker was written (for chaining)
    pub fn emit_debug_marker(&mut self, debug_buf_ptr: VirtualReg, marker: u32) -> VirtualReg {
        // Atomically get next slot: slot = atomicAdd(debug_buf[0], 1)
        let one = self.mov_u32_imm(1);
        let slot = self.atom_add_global_u32(debug_buf_ptr, one);

        // Compute address: addr = debug_buf_ptr + (slot + 1) * 4
        let slot_plus_1 = self.add_u32(slot, 1);
        let offset = self.mul_u32(slot_plus_1, 4);
        let offset_64 = self.cvt_u64_u32(offset);
        let addr = self.add_u64(debug_buf_ptr, offset_64);

        // Write marker value
        let marker_val = self.mov_u32_imm(marker);
        self.st_global_u32(addr, marker_val);

        slot
    }

    /// Emit a debug value to a debug buffer (for variables)
    ///
    /// Similar to emit_debug_marker but writes an arbitrary register value
    pub fn emit_debug_value(&mut self, debug_buf_ptr: VirtualReg, value: VirtualReg) -> VirtualReg {
        // Atomically get next slot
        let one = self.mov_u32_imm(1);
        let slot = self.atom_add_global_u32(debug_buf_ptr, one);

        // Compute address
        let slot_plus_1 = self.add_u32(slot, 1);
        let offset = self.mul_u32(slot_plus_1, 4);
        let offset_64 = self.cvt_u64_u32(offset);
        let addr = self.add_u64(debug_buf_ptr, offset_64);

        // Write value
        self.st_global_u32(addr, value);

        slot
    }
}
