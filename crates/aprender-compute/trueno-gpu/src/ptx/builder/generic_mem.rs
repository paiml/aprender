//! Generic address space memory operations and predicated loads for KernelBuilder.
//!
//! Provides generic (unified) address space load/store operations,
//! shared memory base address access, and predicated memory loads
//! for bounds-checked access patterns.

use super::super::instructions::{Operand, Predicate, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;
use super::super::types::{PtxStateSpace, PtxType};
use super::KernelBuilder;

impl<'a> KernelBuilder<'a> {
    /// Get base address of shared memory array 'smem' as generic address
    ///
    /// Returns a u64 pointer to the beginning of the shared memory region
    /// declared by `.shared .align 16 .b8 smem[N]`.
    ///
    /// NOTE: This returns a GENERIC address (via cvta.to.shared).
    /// Use with ld/st WITHOUT state space (generic addressing).
    /// For WMMA operations that require generic pointers.
    ///
    /// For shared-space ld.shared/st.shared, use `shared_base_addr_local()` instead.
    pub fn shared_base_addr(&mut self) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U64);
        // Use cvta.to.shared.u64 to get generic address from shared memory label
        // This is REQUIRED for WMMA operations which need generic pointers
        // Generates: cvta.to.shared.u64 %rd, smem;
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvta, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Label("smem".to_string()))
                .space(PtxStateSpace::Shared),
        );
        dst
    }

    /// Load u32 from generic address (unified address space)
    ///
    /// Use this after `shared_base_addr()` + offset computation for shared memory.
    /// Generic addressing allows the hardware to resolve the actual memory space.
    pub fn ld_generic_u32(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
            // No .space() means generic addressing
        );
        dst
    }

    /// Load u32 from generic address into existing register (register reuse)
    pub fn ld_generic_u32_into(&mut self, dst: VirtualReg, addr: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
    }

    /// Store u32 to generic address (unified address space)
    ///
    /// Use this after `shared_base_addr()` + offset computation for shared memory.
    /// Generic addressing allows the hardware to resolve the actual memory space.
    pub fn st_generic_u32(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U32)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
            // No .space() means generic addressing
        );
    }

    /// Load u64 from generic address (unified address space)
    pub fn ld_generic_u64(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U64);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );
        dst
    }

    /// Store u64 to generic address (unified address space)
    pub fn st_generic_u64(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U64)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
        );
    }

    /// Load u8 from generic address (unified address space)
    ///
    /// Use this for byte-level operations on shared memory.
    /// Returns value in a U16 register (PTX minimum register size).
    pub fn ld_generic_u8(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U16);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U8)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
            // No .space() means generic addressing
        );
        dst
    }

    /// Store u8 to generic address (unified address space)
    ///
    /// Use this for byte-level writes to shared memory.
    /// Source should be in a U16 or U32 register (low 8 bits stored).
    pub fn st_generic_u8(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U8)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
            // No .space() means generic addressing
        );
    }

    /// Load u16 from generic address (unified address space)
    ///
    /// Use this for 16-bit operations on shared memory (e.g., hash table entries).
    pub fn ld_generic_u16(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::U16);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::U16)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
            // No .space() means generic addressing
        );
        dst
    }

    /// Store u16 to generic address (unified address space)
    ///
    /// Use this for 16-bit writes to shared memory (e.g., hash table entries).
    pub fn st_generic_u16(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::U16)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
            // No .space() means generic addressing
        );
    }

    /// Load f32 from generic address (unified address space)
    ///
    /// Use this after `shared_base_addr()` + offset computation for shared memory.
    /// Generic addressing allows the hardware to resolve the actual memory space.
    pub fn ld_generic_f32(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
            // No .space() means generic addressing
        );
        dst
    }

    /// Store f32 to generic address (unified address space)
    ///
    /// Use this after `shared_base_addr()` + offset computation for shared memory.
    /// Generic addressing allows the hardware to resolve the actual memory space.
    pub fn st_generic_f32(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::F32)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val)),
            // No .space() means generic addressing
        );
    }

    /// Predicated load f32 from global memory with default value
    ///
    /// If predicate is true: loads value from addr
    /// If predicate is false: returns default_val (no memory access)
    ///
    /// Implementation:
    /// ```ptx
    /// mov.f32 %dst, default_val;     // Initialize with default
    /// @pred ld.global.f32 %dst, [addr];  // Conditional load
    /// ```
    ///
    /// Used for bounds-checked loads in GEMV:
    /// ```text
    /// let valid = setp_lt_u32(idx, n);
    /// let val = ld_global_f32_predicated(addr, valid, 0.0);
    /// ```
    pub fn ld_global_f32_predicated(
        &mut self,
        addr: VirtualReg,
        pred: VirtualReg,
        default_val: f32,
    ) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);

        // 1. Initialize with default value
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmF32(default_val)),
        );

        // 2. Predicated load - only executes if pred is true
        // If pred is false, dst keeps the default value
        let predicate = Predicate { reg: pred, negated: false };

        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::F32)
                .space(PtxStateSpace::Global)
                .predicated(predicate)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr)),
        );

        dst
    }

    /// PAR-028: Load F16 from global memory with predicate guard
    ///
    /// If predicate is true: loads from addr, converts to F32
    /// If predicate is false: returns 0.0 (no memory access)
    ///
    /// Implementation:
    /// ```ptx
    /// mov.f32 %dst, 0.0;                // Initialize with default
    /// @pred {
    ///     ld.global.b16 %tmp, [addr];   // Conditional load F16
    ///     cvt.f32.f16 %dst, %tmp;       // Convert to F32
    /// }
    /// ```
    ///
    /// Used for FP16 KV cache in attention kernels:
    /// ```text
    /// let valid = setp_lt_u32(idx, head_dim);
    /// let k_val = ld_global_f16_to_f32_predicated(addr, valid);
    /// ```
    pub fn ld_global_f16_to_f32_predicated(
        &mut self,
        addr: VirtualReg,
        pred: VirtualReg,
    ) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        let tmp = self.registers.allocate_virtual(PtxType::F16);

        // 1. Initialize with default value (0.0)
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmF32(0.0)),
        );

        // 2. Predicated F16 load - only executes if pred is true
        let predicate = Predicate { reg: pred, negated: false };

        // Load F16 (using .b16 as PTX requires)
        self.instructions.push(
            PtxInstruction::new(PtxOp::Ld, PtxType::B16)
                .space(PtxStateSpace::Global)
                .predicated(predicate.clone())
                .dst(Operand::Reg(tmp))
                .src(Operand::Reg(addr)),
        );

        // 3. Predicated convert F16 to F32
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F32)
                .predicated(predicate)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(tmp)),
        );

        dst
    }
}
