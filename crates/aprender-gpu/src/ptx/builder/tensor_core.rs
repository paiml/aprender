//! Tensor Core (WMMA) operations for KernelBuilder.
//!
//! Provides WMMA load/store/mma operations, F16/F32 conversions,
//! and F16 global memory access for tensor core pipelines.

use super::super::instructions::{Operand, PtxInstruction, PtxOp, RoundingMode, WmmaLayout};
use super::super::registers::VirtualReg;
use super::super::types::{PtxStateSpace, PtxType};
use super::KernelBuilder;

impl<'a> KernelBuilder<'a> {
    // ===== Tensor Core (WMMA) Operations =====
    // These require sm_70+ and generate WMMA PTX intrinsics

    /// Load F16 matrix fragment A for WMMA (16x16x16 tile)
    /// Returns fragment registers for use in wmma_mma
    pub fn wmma_load_a_f16(
        &mut self,
        addr: VirtualReg,
        stride: u32,
        layout: WmmaLayout,
    ) -> Vec<VirtualReg> {
        // WMMA 16x16x16 F16 requires 8 F16x2 registers (16 half values)
        let mut frag = Vec::with_capacity(8);
        for _ in 0..8 {
            frag.push(self.registers.allocate_virtual(PtxType::B32));
        }
        // Build instruction with all 8 destination registers
        let mut instr = PtxInstruction::new(PtxOp::WmmaLoadA, PtxType::F16).label(format!(
            "m16n16k16.{}.f16.stride.{}",
            layout.to_ptx_string(),
            stride
        ));
        // Add all fragment registers as destinations (use push_dst for vector dests)
        for reg in &frag {
            instr = instr.push_dst(Operand::Reg(*reg));
        }
        // Source is address and stride immediate
        instr = instr.src(Operand::Reg(addr));
        instr = instr.src(Operand::ImmI64(i64::from(stride)));
        self.instructions.push(instr);
        frag
    }

    /// Load F16 matrix fragment B for WMMA (16x16x16 tile)
    pub fn wmma_load_b_f16(
        &mut self,
        addr: VirtualReg,
        stride: u32,
        layout: WmmaLayout,
    ) -> Vec<VirtualReg> {
        let mut frag = Vec::with_capacity(8);
        for _ in 0..8 {
            frag.push(self.registers.allocate_virtual(PtxType::B32));
        }
        // Build instruction with all 8 destination registers
        let mut instr = PtxInstruction::new(PtxOp::WmmaLoadB, PtxType::F16).label(format!(
            "m16n16k16.{}.f16.stride.{}",
            layout.to_ptx_string(),
            stride
        ));
        for reg in &frag {
            instr = instr.push_dst(Operand::Reg(*reg));
        }
        instr = instr.src(Operand::Reg(addr));
        instr = instr.src(Operand::ImmI64(i64::from(stride)));
        self.instructions.push(instr);
        frag
    }

    /// Load F32 accumulator fragment C for WMMA (16x16x16 tile)
    pub fn wmma_load_c_f32(
        &mut self,
        addr: VirtualReg,
        stride: u32,
        layout: WmmaLayout,
    ) -> Vec<VirtualReg> {
        // Accumulator is 8 F32 values
        let mut frag = Vec::with_capacity(8);
        for _ in 0..8 {
            frag.push(self.registers.allocate_virtual(PtxType::F32));
        }
        // Build instruction with all 8 destination registers
        let mut instr = PtxInstruction::new(PtxOp::WmmaLoadC, PtxType::F32).label(format!(
            "m16n16k16.{}.f32.stride.{}",
            layout.to_ptx_string(),
            stride
        ));
        for reg in &frag {
            instr = instr.push_dst(Operand::Reg(*reg));
        }
        instr = instr.src(Operand::Reg(addr));
        instr = instr.src(Operand::ImmI64(i64::from(stride)));
        self.instructions.push(instr);
        frag
    }

    /// Initialize F32 accumulator fragment C to zero (WAPR-PERF-010)
    /// This avoids loading from memory address 0 which is invalid
    pub fn wmma_init_c_zero(&mut self) -> Vec<VirtualReg> {
        // Accumulator is 8 F32 values, initialize all to 0.0
        let mut frag = Vec::with_capacity(8);
        for _ in 0..8 {
            let reg = self.registers.allocate_virtual(PtxType::F32);
            self.instructions.push(
                PtxInstruction::new(PtxOp::Mov, PtxType::F32)
                    .dst(Operand::Reg(reg))
                    .src(Operand::ImmF32(0.0)),
            );
            frag.push(reg);
        }
        frag
    }

    /// WMMA matrix multiply-accumulate: D = A * B + C
    /// Takes A, B, C fragment registers and returns D fragment registers.
    /// Uses row.col layout (A=row-major, B=col-major). For row.row, use
    /// `wmma_mma_f16_f32_row_row`.
    #[allow(clippy::similar_names)]
    pub fn wmma_mma_f16_f32(
        &mut self,
        frag_a: &[VirtualReg],
        frag_b: &[VirtualReg],
        frag_c: &[VirtualReg],
    ) -> Vec<VirtualReg> {
        self.wmma_mma_f16_f32_layouts(
            frag_a,
            frag_b,
            frag_c,
            WmmaLayout::RowMajor,
            WmmaLayout::ColMajor,
        )
    }

    /// WMMA MMA with both A and B in row-major layout.
    /// Use when both wmma_load_a and wmma_load_b used `WmmaLayout::RowMajor`.
    #[allow(clippy::similar_names)]
    pub fn wmma_mma_f16_f32_row_row(
        &mut self,
        frag_a: &[VirtualReg],
        frag_b: &[VirtualReg],
        frag_c: &[VirtualReg],
    ) -> Vec<VirtualReg> {
        self.wmma_mma_f16_f32_layouts(
            frag_a,
            frag_b,
            frag_c,
            WmmaLayout::RowMajor,
            WmmaLayout::RowMajor,
        )
    }

    /// WMMA matrix multiply-accumulate with explicit layout specification.
    /// The layouts MUST match those used in the corresponding wmma_load instructions.
    #[allow(clippy::similar_names)]
    pub fn wmma_mma_f16_f32_layouts(
        &mut self,
        frag_a: &[VirtualReg],
        frag_b: &[VirtualReg],
        frag_c: &[VirtualReg],
        a_layout: WmmaLayout,
        b_layout: WmmaLayout,
    ) -> Vec<VirtualReg> {
        // Output accumulator D (8 F32 values)
        let mut frag_d = Vec::with_capacity(8);
        for _ in 0..8 {
            frag_d.push(self.registers.allocate_virtual(PtxType::F32));
        }

        // MMA instruction with all fragment registers
        // Format: wmma.mma.sync.aligned.m16n16k16.{alayout}.{blayout}.f32.f32
        let label =
            format!("m16n16k16.{}.{}.f32.f32", a_layout.to_ptx_string(), b_layout.to_ptx_string());
        let mut instr = PtxInstruction::new(PtxOp::WmmaMma, PtxType::F32).label(label);

        // Add all D registers as destinations (use push_dst for vector dests)
        for reg in &frag_d {
            instr = instr.push_dst(Operand::Reg(*reg));
        }

        // Add all A, B, C fragment registers as sources (in order)
        for reg in frag_a {
            instr = instr.src(Operand::Reg(*reg));
        }
        for reg in frag_b {
            instr = instr.src(Operand::Reg(*reg));
        }
        for reg in frag_c {
            instr = instr.src(Operand::Reg(*reg));
        }

        self.instructions.push(instr);
        frag_d
    }

    /// Store F32 accumulator fragment D to memory
    pub fn wmma_store_d_f32(
        &mut self,
        addr: VirtualReg,
        frag_d: &[VirtualReg],
        stride: u32,
        layout: WmmaLayout,
    ) {
        if frag_d.is_empty() {
            return;
        }
        // Format: wmma.store.d.sync.aligned.m16n16k16.row.f32 [addr], {d0-d7}, stride
        let mut instr = PtxInstruction::new(PtxOp::WmmaStoreD, PtxType::F32).label(format!(
            "m16n16k16.{}.f32.stride.{}",
            layout.to_ptx_string(),
            stride
        ));
        // Address is first source
        instr = instr.src(Operand::Reg(addr));
        // All fragment registers
        for reg in frag_d {
            instr = instr.src(Operand::Reg(*reg));
        }
        // Stride
        instr = instr.src(Operand::ImmI64(i64::from(stride)));
        self.instructions.push(instr);
    }

    /// Convert F32 values to F16 (for feeding tensor cores)
    pub fn cvt_f16_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F16);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F16)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Convert F16 value to F32 (for accumulation)
    pub fn cvt_f32_f16(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Load F16 from global memory
    ///
    /// NOTE: PTX uses `.b16` (binary 16-bit) for half-precision loads,
    /// not `.f16`. The loaded value is still interpreted as f16 for
    /// subsequent operations.
    pub fn ld_global_f16(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F16);
        self.instructions.push(
            // PTX requires ld.global.b16, not ld.global.f16
            PtxInstruction::new(PtxOp::Ld, PtxType::B16)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Store F16 to global memory
    ///
    /// PTX uses `.b16` (binary 16-bit) for half-precision stores, not `.f16`.
    /// The PTX ISA does not support `st.global.f16` — only `.b16` for 16-bit stores.
    /// This matches `ld_global_f16` which already uses `PtxType::B16`.
    pub fn st_global_f16(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::B16)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
    }

    // ===== B32 register helpers (for mma.sync/ldmatrix fragments) =====

    /// Allocate a .b32 register initialized to an immediate value.
    /// mma.sync A/B operands MUST be .b32 (not .u32) — ptxas enforces this.
    pub fn mov_b32_imm(&mut self, val: u32) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::B32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Mov, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::ImmI64(val as i64)),
        );
        dst
    }

    // ===== MMA.sync (SM 8.0+ — higher IPC than WMMA) =====

    /// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    ///
    /// Computes a 16×8 output fragment per warp. Per-thread register layout:
    ///   A: 4 B32 regs (8 FP16 values) — 16×16 fragment
    ///   B: 2 B32 regs (4 FP16 values) — 16×8 fragment
    ///   C: 4 F32 regs (4 FP32 accumulators) — 16×8 fragment
    ///   D: 4 F32 regs (4 FP32 results)
    ///
    /// To cover a full 16×16 output, issue TWO mma.sync with different B halves.
    ///
    /// Returns 4 destination registers (D fragment).
    pub fn mma_sync_m16n8k16(
        &mut self,
        a_regs: &[VirtualReg; 4],
        b_regs: &[VirtualReg; 2],
        c_regs: &[VirtualReg; 4],
    ) -> [VirtualReg; 4] {
        let d0 = self.registers.allocate_virtual(PtxType::F32);
        let d1 = self.registers.allocate_virtual(PtxType::F32);
        let d2 = self.registers.allocate_virtual(PtxType::F32);
        let d3 = self.registers.allocate_virtual(PtxType::F32);

        let mut instr = PtxInstruction::new(PtxOp::MmaSync, PtxType::F32);
        // Destinations: D[0..3]
        instr = instr.dst(Operand::Reg(d0));
        instr.dsts.push(Operand::Reg(d1));
        instr.dsts.push(Operand::Reg(d2));
        instr.dsts.push(Operand::Reg(d3));
        // Sources: A[0..3], B[0..1], C[0..3]
        for &a in a_regs {
            instr = instr.src(Operand::Reg(a));
        }
        for &b in b_regs {
            instr = instr.src(Operand::Reg(b));
        }
        for &c in c_regs {
            instr = instr.src(Operand::Reg(c));
        }
        self.instructions.push(instr);
        [d0, d1, d2, d3]
    }

    /// In-place mma.sync: D overwrites C registers (D = C, no extra mov needed).
    ///
    /// PTX allows D and C to alias when the instruction uses `.row.col` layout.
    /// This eliminates 4 mov instructions per call (8 per K-tile for left+right).
    pub fn mma_sync_m16n8k16_inplace(
        &mut self,
        a_regs: &[VirtualReg; 4],
        b_regs: &[VirtualReg; 2],
        c_regs: &[VirtualReg; 4],
    ) {
        // Extend live ranges so register allocator keeps C regs alive through the instruction
        for &c in c_regs {
            self.registers.extend_live_range(c);
        }

        let mut instr = PtxInstruction::new(PtxOp::MmaSync, PtxType::F32);
        // D = C (same registers)
        instr = instr.dst(Operand::Reg(c_regs[0]));
        instr.dsts.push(Operand::Reg(c_regs[1]));
        instr.dsts.push(Operand::Reg(c_regs[2]));
        instr.dsts.push(Operand::Reg(c_regs[3]));
        // Sources: A[0..3], B[0..1], C[0..3]
        for &a in a_regs {
            instr = instr.src(Operand::Reg(a));
        }
        for &b in b_regs {
            instr = instr.src(Operand::Reg(b));
        }
        for &c in c_regs {
            instr = instr.src(Operand::Reg(c));
        }
        self.instructions.push(instr);
    }

    /// ldmatrix.sync.aligned.m8n8.x4.shared.b16
    ///
    /// Loads 4 8×8 FP16 matrices from shared memory in one instruction.
    /// Each thread provides one source address (its row within the tile).
    /// Returns 4 B32 registers containing 4 matrix fragments.
    ///
    /// Replaces ~16 individual ld.shared instructions.
    pub fn ldmatrix_x4(&mut self, addr: VirtualReg) -> [VirtualReg; 4] {
        let d0 = self.registers.allocate_virtual(PtxType::B32);
        let d1 = self.registers.allocate_virtual(PtxType::B32);
        let d2 = self.registers.allocate_virtual(PtxType::B32);
        let d3 = self.registers.allocate_virtual(PtxType::B32);

        let mut instr = PtxInstruction::new(PtxOp::LdMatrix, PtxType::B16);
        instr = instr.dst(Operand::Reg(d0));
        instr.dsts.push(Operand::Reg(d1));
        instr.dsts.push(Operand::Reg(d2));
        instr.dsts.push(Operand::Reg(d3));
        instr = instr.src(Operand::Reg(addr));
        self.instructions.push(instr);
        [d0, d1, d2, d3]
    }

    /// ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
    ///
    /// Transposed variant: loads 2 8×8 FP16 matrices with implicit transpose.
    /// Used for B fragment of mma.sync.row.col — B stored row-major in smem,
    /// loaded as col-major into registers.
    /// Returns 2 B32 registers.
    pub fn ldmatrix_x2_trans(&mut self, addr: VirtualReg) -> [VirtualReg; 2] {
        let d0 = self.registers.allocate_virtual(PtxType::B32);
        let d1 = self.registers.allocate_virtual(PtxType::B32);

        let mut instr = PtxInstruction::new(PtxOp::LdMatrixTrans, PtxType::B16);
        instr = instr.dst(Operand::Reg(d0));
        instr.dsts.push(Operand::Reg(d1));
        instr = instr.src(Operand::Reg(addr));
        self.instructions.push(instr);
        [d0, d1]
    }
}
