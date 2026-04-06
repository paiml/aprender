//! NF4 Dequantization-Fused GEMM Kernel (trueno#108).
//!
//! Implements fused dequantization with matrix multiplication for NF4 (4-bit NormalFloat)
//! quantized weights, enabling QLoRA training with 8x memory compression.
//!
//! # NF4 Block Layout
//!
//! Scales and packed data are stored in separate GPU buffers (SoA layout) for coalescing:
//! - `b_scales`: `[f32; num_blocks]` — one scale per 64-value block, column-major block order
//! - `b_nf4`:    `[u8; num_blocks * 32]` — packed 4-bit indices (2 per byte), same order
//!
//! # Dequantization
//!
//! ```text
//! val = scale × NF4_LUT[nibble]
//! ```
//!
//! Where `NF4_LUT` is a fixed 16-entry codebook from normal distribution quantiles.
//!
//! # Contract: C-NF4-003 (GEMM Numerical Parity)
//!
//! `nf4_gemm(A, Q) ≈ naive_gemm(A, dequantize(Q))` within 1e-3 per-element.

#![allow(clippy::similar_names)]

use super::nf4_cpu::{NF4_BLOCK_SIZE, NF4_LUT};
use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// NF4 block size as u32 for PTX constants.
const NF4_BLOCK_SIZE_U32: u32 = NF4_BLOCK_SIZE as u32;

/// NF4 quantized GEMM kernel configuration.
///
/// Computes `C[M×N] = A[M×K] @ dequant(B_nf4[K×N])` where B is stored in NF4 format.
/// The kernel fuses dequantization with matmul to avoid materializing fp32 weights.
///
/// # Memory Layout (separate scale/data buffers)
///
/// - `A`: row-major f32 `[M × K]`
/// - `b_nf4`: packed nibbles `[N * (K/64) * 32]` bytes, column-major block order
/// - `b_scales`: `[N * (K/64)]` f32 values, column-major block order
/// - `C`: row-major f32 `[M × N]`
#[derive(Debug, Clone)]
pub struct Nf4GemmKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) — must be divisible by 64
    pub k: u32,
    /// Tile size for output (default: 32)
    pub tile_size: u32,
}

impl Nf4GemmKernel {
    /// Create a new NF4 quantized GEMM kernel.
    ///
    /// # Contract: C-NF4-002
    ///
    /// `k` must be divisible by 64 (NF4 block size).
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k, tile_size: 32 }
    }

    /// Set output tile size.
    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Get number of NF4 blocks per weight column (K / 64).
    #[must_use]
    pub const fn num_blocks_per_col(&self) -> u32 {
        self.k / NF4_BLOCK_SIZE_U32
    }
}

/// Perform register-based NF4 codebook lookup via 4-level binary selection tree.
///
/// Given a nibble value (0-15) in `nib` and 16 preloaded codebook registers in `lut`,
/// returns the f32 codebook value without any memory access.
///
/// Uses a binary tree of `selp.f32` operations (4 levels, 15 selp total):
/// - Level 0: pair adjacent LUT entries (8 selp)
/// - Level 1: pair level-0 results    (4 selp)
/// - Level 2: pair level-1 results    (2 selp)
/// - Level 3: final selection          (1 selp)
///
/// Total: 15 selp + 4 bit-extract = 19 instructions per lookup.
/// Compared to shared memory: 0 memory loads, 0 cache misses on unified memory.
pub(crate) fn nf4_register_lut_lookup(
    ctx: &mut crate::ptx::builder::KernelBuilder<'_>,
    nib: crate::ptx::VirtualReg,
    lut: &[crate::ptx::VirtualReg; 16],
) -> crate::ptx::VirtualReg {
    // Extract individual bits of the 4-bit nibble
    let bit0 = ctx.and_u32_imm(nib, 1); // nib & 1
    let bit1 = ctx.shr_u32_imm(nib, 1);
    let bit1 = ctx.and_u32_imm(bit1, 1); // (nib >> 1) & 1
    let bit2 = ctx.shr_u32_imm(nib, 2);
    let bit2 = ctx.and_u32_imm(bit2, 1); // (nib >> 2) & 1
    let bit3 = ctx.shr_u32_imm(nib, 3);
    let bit3 = ctx.and_u32_imm(bit3, 1); // (nib >> 3) & 1

    // Convert bit values to predicates (bit != 0)
    let zero = ctx.mov_u32_imm(0);
    let p0 = ctx.setp_ne_u32(bit0, zero);
    let p1 = ctx.setp_ne_u32(bit1, zero);
    let p2 = ctx.setp_ne_u32(bit2, zero);
    let p3 = ctx.setp_ne_u32(bit3, zero);

    // Level 0: select between adjacent pairs using bit 0
    // p0 ? lut[2i+1] : lut[2i] for i = 0..8
    let s0 = ctx.selp_f32(p0, lut[1], lut[0]);
    let s1 = ctx.selp_f32(p0, lut[3], lut[2]);
    let s2 = ctx.selp_f32(p0, lut[5], lut[4]);
    let s3 = ctx.selp_f32(p0, lut[7], lut[6]);
    let s4 = ctx.selp_f32(p0, lut[9], lut[8]);
    let s5 = ctx.selp_f32(p0, lut[11], lut[10]);
    let s6 = ctx.selp_f32(p0, lut[13], lut[12]);
    let s7 = ctx.selp_f32(p0, lut[15], lut[14]);

    // Level 1: select between pairs using bit 1
    let t0 = ctx.selp_f32(p1, s1, s0);
    let t1 = ctx.selp_f32(p1, s3, s2);
    let t2 = ctx.selp_f32(p1, s5, s4);
    let t3 = ctx.selp_f32(p1, s7, s6);

    // Level 2: select between pairs using bit 2
    let u0 = ctx.selp_f32(p2, t1, t0);
    let u1 = ctx.selp_f32(p2, t3, t2);

    // Level 3: final selection using bit 3
    ctx.selp_f32(p3, u1, u0)
}

impl Kernel for Nf4GemmKernel {
    fn name(&self) -> &str {
        "nf4_gemm_fused"
    }

    fn build_ptx(&self) -> PtxKernel {
        let tile_size = self.tile_size;

        // No shared memory needed — codebook lives in registers.
        // On GB10 unified memory, shared memory goes through the same DRAM,
        // so register-based LUT eliminates both the bar.sync and all LUT loads.
        let smem_size = 0;

        PtxKernel::new("nf4_gemm_fused")
            .param(PtxType::U64, "a_ptr") // Activations [M × K], f32
            .param(PtxType::U64, "b_nf4_ptr") // Packed nibbles
            .param(PtxType::U64, "b_scales_ptr") // Per-block scales, f32
            .param(PtxType::U64, "c_ptr") // Output [M × N], f32
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size)
            .build(|ctx| {
                // Thread and block indices
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                // Load parameters
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_nf4_ptr = ctx.load_param_u64("b_nf4_ptr");
                let b_scales_ptr = ctx.load_param_u64("b_scales_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // =========================================================
                // Load NF4 codebook into 16 f32 registers (no shared memory)
                // =========================================================
                // On unified memory (GB10), shared memory goes through the same
                // DRAM path as global. Register-based LUT avoids all memory
                // traffic for codebook access and eliminates the bar.sync that
                // was the #1 bottleneck (45% slowdown from barrier overhead).
                let lut_regs: [_; 16] = std::array::from_fn(|i| ctx.mov_f32_imm(NF4_LUT[i]));

                // =========================================================
                // Calculate output position (tile-based)
                // =========================================================
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let out_row = ctx.mul_u32_reg(ctaid_y, tile_size_reg);
                let out_col = ctx.mul_u32_reg(ctaid_x, tile_size_reg);

                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                // Bounds check predicates (store only for valid threads)
                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, n_param);

                // Clamp to valid range (ensures safe memory access for all threads)
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // GH-561: Initialize accumulator in f64 to eliminate FP32 accumulation
                // error that causes cosine=-0.005 on sm_121 Blackwell.
                // 3584 FMA ops in f32 → ~0.02% relative error per matmul × 196 matmuls = NaN.
                // f64 accumulator (53-bit mantissa) has 2^29× more precision headroom.
                let acc = ctx.mov_f64_imm_zero();

                // Number of NF4 blocks along K (K / 64)
                let num_k_blocks = ctx.div_u32(k_param, NF4_BLOCK_SIZE_U32);

                // =========================================================
                // Block loop: iterate over K dimension in chunks of 64
                // =========================================================
                let block_idx = ctx.mov_u32_imm(0);

                ctx.label("block_loop");
                let block_done = ctx.setp_ge_u32(block_idx, num_k_blocks);
                ctx.branch_if(block_done, "block_loop_done");

                // Scale layout: scales[col * num_k_blocks + block_idx]
                let col_block_offset = ctx.mul_u32_reg(clamped_col, num_k_blocks);
                let scale_idx = ctx.add_u32_reg(col_block_offset, block_idx);
                let scale_byte_offset = ctx.mul_wide_u32(scale_idx, 4);
                let scale_addr = ctx.add_u64(b_scales_ptr, scale_byte_offset);
                let scale = ctx.ld_global_f32(scale_addr);

                // Data layout: data[(col * num_k_blocks + block_idx) * 32 + byte]
                let data_block_byte_offset = ctx.mul_wide_u32(scale_idx, 32);
                let data_block_addr = ctx.add_u64(b_nf4_ptr, data_block_byte_offset);

                // Vectorized inner loop: process 8 nibbles (4 bytes) per iteration.
                // 64 values / 8 per iteration = 8 iterations (was 64).
                // Each iteration: load 4 bytes, extract 8 nibbles, 8 LUT lookups, 8 FMAs.
                let sixty_four = ctx.mov_u32_imm(NF4_BLOCK_SIZE_U32);
                let block_k_base = ctx.mul_u32_reg(block_idx, sixty_four);
                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, k_param);

                let chunk_idx = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);
                let mask_4bit = ctx.mov_u32_imm(0xF);

                ctx.label("chunk_loop");
                let chunk_done = ctx.setp_ge_u32(chunk_idx, eight);
                ctx.branch_if(chunk_done, "chunk_loop_done");

                // =========================================================
                // Load 4 packed bytes = 8 nibbles from B_nf4
                // =========================================================
                // Byte offset within block: chunk_idx * 4
                let byte_base = ctx.mul_u32(chunk_idx, 4);
                let byte_base_64 = ctx.cvt_u64_u32(byte_base);
                let chunk_addr = ctx.add_u64(data_block_addr, byte_base_64);

                // Load 4 individual bytes (each returns u16 register)
                let b0_raw = ctx.ld_global_u8(chunk_addr);
                let off1 = ctx.mov_u64_imm(1);
                let addr1 = ctx.add_u64(chunk_addr, off1);
                let b1_raw = ctx.ld_global_u8(addr1);
                let off2 = ctx.mov_u64_imm(2);
                let addr2 = ctx.add_u64(chunk_addr, off2);
                let b2_raw = ctx.ld_global_u8(addr2);
                let off3 = ctx.mov_u64_imm(3);
                let addr3 = ctx.add_u64(chunk_addr, off3);
                let b3_raw = ctx.ld_global_u8(addr3);

                // Convert u8 (in u16 registers) to u32 for bit manipulation
                let b0 = ctx.cvt_u32_u8(b0_raw);
                let b1 = ctx.cvt_u32_u8(b1_raw);
                let b2 = ctx.cvt_u32_u8(b2_raw);
                let b3 = ctx.cvt_u32_u8(b3_raw);

                // =========================================================
                // Extract 8 nibbles from 4 bytes
                // =========================================================
                // byte0: nibbles 0 (low) and 1 (high)
                let n0 = ctx.and_u32(b0, mask_4bit);
                let n1 = ctx.shr_u32_imm(b0, 4);
                let n1 = ctx.and_u32(n1, mask_4bit);
                // byte1: nibbles 2 (low) and 3 (high)
                let n2 = ctx.and_u32(b1, mask_4bit);
                let n3 = ctx.shr_u32_imm(b1, 4);
                let n3 = ctx.and_u32(n3, mask_4bit);
                // byte2: nibbles 4 (low) and 5 (high)
                let n4 = ctx.and_u32(b2, mask_4bit);
                let n5 = ctx.shr_u32_imm(b2, 4);
                let n5 = ctx.and_u32(n5, mask_4bit);
                // byte3: nibbles 6 (low) and 7 (high)
                let n6 = ctx.and_u32(b3, mask_4bit);
                let n7 = ctx.shr_u32_imm(b3, 4);
                let n7 = ctx.and_u32(n7, mask_4bit);

                // =========================================================
                // 8x register-based LUT lookups (no memory access)
                // =========================================================
                let v0 = nf4_register_lut_lookup(ctx, n0, &lut_regs);
                let v1 = nf4_register_lut_lookup(ctx, n1, &lut_regs);
                let v2 = nf4_register_lut_lookup(ctx, n2, &lut_regs);
                let v3 = nf4_register_lut_lookup(ctx, n3, &lut_regs);
                let v4 = nf4_register_lut_lookup(ctx, n4, &lut_regs);
                let v5 = nf4_register_lut_lookup(ctx, n5, &lut_regs);
                let v6 = nf4_register_lut_lookup(ctx, n6, &lut_regs);
                let v7 = nf4_register_lut_lookup(ctx, n7, &lut_regs);

                // =========================================================
                // 8x dequantize: val = scale * codebook_value
                // =========================================================
                let d0 = ctx.mul_f32(scale, v0);
                let d1 = ctx.mul_f32(scale, v1);
                let d2 = ctx.mul_f32(scale, v2);
                let d3 = ctx.mul_f32(scale, v3);
                let d4 = ctx.mul_f32(scale, v4);
                let d5 = ctx.mul_f32(scale, v5);
                let d6 = ctx.mul_f32(scale, v6);
                let d7 = ctx.mul_f32(scale, v7);

                // =========================================================
                // Load 8 activation values and accumulate via FMA
                // =========================================================
                // Element index within K: block_k_base + chunk_idx * 8 + i
                let elem_base_u32 = ctx.mul_u32_reg(chunk_idx, eight);
                let elem_base_k = ctx.add_u32_reg(block_k_base, elem_base_u32);

                // Compute base address for A[clamped_row, elem_base_k]
                let elem_base_k_64 = ctx.cvt_u64_u32(elem_base_k);
                let a_base_offset = ctx.add_u64(a_row_offset, elem_base_k_64);
                let a_base_bytes = ctx.mul_u64(a_base_offset, 4);
                let a_base_addr = ctx.add_u64(a_ptr, a_base_bytes);

                // Load 8 A values: A[row, k+0] through A[row, k+7]
                // First 4 via vectorized v4 load (16-byte aligned within row)
                let a_v4_0 = ctx.ld_global_f32_v4(a_base_addr);
                let sixteen = ctx.mov_u64_imm(16);
                let a_addr_4 = ctx.add_u64(a_base_addr, sixteen);
                let a_v4_1 = ctx.ld_global_f32_v4(a_addr_4);

                // GH-561: 8x FMA with f64 accumulator: acc(f64) += f32_to_f64(a) * f32_to_f64(d)
                ctx.fma_f64_acc_inplace(acc, a_v4_0[0], d0);
                ctx.fma_f64_acc_inplace(acc, a_v4_0[1], d1);
                ctx.fma_f64_acc_inplace(acc, a_v4_0[2], d2);
                ctx.fma_f64_acc_inplace(acc, a_v4_0[3], d3);
                ctx.fma_f64_acc_inplace(acc, a_v4_1[0], d4);
                ctx.fma_f64_acc_inplace(acc, a_v4_1[1], d5);
                ctx.fma_f64_acc_inplace(acc, a_v4_1[2], d6);
                ctx.fma_f64_acc_inplace(acc, a_v4_1[3], d7);

                ctx.add_u32_inplace(chunk_idx, 1);
                ctx.branch("chunk_loop");

                ctx.label("chunk_loop_done");

                ctx.add_u32_inplace(block_idx, 1);
                ctx.branch("block_loop");

                ctx.label("block_loop_done");

                // =========================================================
                // Store result (only for valid threads)
                // =========================================================
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                let c_row_offset = ctx.mul_wide_u32_reg(global_row, n_param);
                let global_col_64 = ctx.cvt_u64_u32(global_col);
                let c_elem_offset = ctx.add_u64(c_row_offset, global_col_64);
                let c_elem_bytes = ctx.mul_u64(c_elem_offset, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_bytes);

                // GH-561: Convert f64 accumulator to f32 for output store
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);
                ctx.st_global_f32(c_addr, acc_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// NF4 transposed GEMM kernel for backward pass (QLoRA gradient propagation).
///
/// Computes `C[M×K] = A[M×N] @ dequant(B_nf4[K×N])^T` where B is stored in NF4 format.
/// Equivalently: `C[i,j] = sum_n A[i,n] * B[j,n]` — reduces over B's column dimension.
///
/// Used in backward pass to propagate gradients through frozen NF4 projections:
/// `grad_input = grad_output @ W^T` where W is the NF4-quantized weight.
///
/// # Memory Layout
///
/// - `A`: row-major f32 `[M × N]` (grad_output)
/// - `b_nf4`: packed nibbles, column-major block order (same as forward)
/// - `b_scales`: per-block scales, column-major block order (same as forward)
/// - `C`: row-major f32 `[M × K]` (grad_input)
#[derive(Debug, Clone)]
pub struct Nf4GemmTransposeKernel {
    /// Output rows (M) — same as grad_output rows
    pub m: u32,
    /// Reduction dimension (N) — columns of B / cols of grad_output
    pub n: u32,
    /// Output columns (K) — rows of B / input hidden size
    pub k: u32,
    /// Tile size for output (default: 16, smaller than forward due to irregular access)
    pub tile_size: u32,
}

impl Nf4GemmTransposeKernel {
    /// Create a new NF4 transposed GEMM kernel.
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k, tile_size: 16 }
    }

    /// Number of NF4 blocks per column of B (K / 64).
    #[must_use]
    pub const fn num_blocks_per_col(&self) -> u32 {
        self.k / NF4_BLOCK_SIZE_U32
    }
}

impl Kernel for Nf4GemmTransposeKernel {
    fn name(&self) -> &str {
        "nf4_gemm_transpose"
    }

    fn build_ptx(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = 16 * 4; // NF4 codebook LUT

        PtxKernel::new("nf4_gemm_transpose")
            .param(PtxType::U64, "a_ptr") // grad_output [M × N], f32
            .param(PtxType::U64, "b_nf4_ptr") // NF4 weight data
            .param(PtxType::U64, "b_scales_ptr") // NF4 weight scales
            .param(PtxType::U64, "c_ptr") // grad_input [M × K], f32
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_nf4_ptr = ctx.load_param_u64("b_nf4_ptr");
                let b_scales_ptr = ctx.load_param_u64("b_scales_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Load NF4 codebook into shared memory
                let smem_base = ctx.shared_base_addr();
                for (i, &val) in NF4_LUT.iter().enumerate() {
                    let imm_i = ctx.mov_u32_imm(i as u32);
                    let is_i = ctx.setp_eq_u32(tid, imm_i);
                    ctx.branch_if_not(is_i, &format!("skip_lut_{i}"));
                    let val_reg = ctx.mov_f32_imm(val);
                    let offset = ctx.mov_u64_imm((i * 4) as u64);
                    let addr = ctx.add_u64(smem_base, offset);
                    ctx.st_generic_f32(addr, val_reg);
                    ctx.label(&format!("skip_lut_{i}"));
                }
                ctx.bar_sync(0);

                // Output position: C[global_row, global_col] where col is in K-dimension
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let out_row = ctx.mul_u32_reg(ctaid_y, tile_size_reg);
                let out_col = ctx.mul_u32_reg(ctaid_x, tile_size_reg);

                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, k_param);

                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let k_minus_1 = ctx.sub_u32_reg(k_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, k_minus_1); // col in K-dim

                // GH-561: f64 accumulator (same fix as forward kernel)
                let acc = ctx.mov_f64_imm_zero();

                // Number of K-blocks per column of B
                let num_k_blocks = ctx.div_u32(k_param, NF4_BLOCK_SIZE_U32);

                // clamped_col is in K-dimension. Find which block and position within block.
                let col_block_idx = ctx.div_u32(clamped_col, NF4_BLOCK_SIZE_U32);
                let col_elem_in_block = ctx.rem_u32(clamped_col, NF4_BLOCK_SIZE_U32);

                // NF4 byte and nibble for this K-position (fixed for all N iterations)
                let byte_in_block = ctx.div_u32(col_elem_in_block, 2);
                let nibble_idx = ctx.rem_u32(col_elem_in_block, 2);
                let four = ctx.mov_u32_imm(4);
                let nibble_shift = ctx.mul_u32_reg(nibble_idx, four);
                let mask_4bit = ctx.mov_u32_imm(0xF);

                // A row base: A[clamped_row, :] starts at a_ptr + clamped_row * N * 4
                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, n_param);

                // Loop over N (reduction dimension — columns of B)
                let n_idx = ctx.mov_u32_imm(0);

                ctx.label("n_loop");
                let n_done = ctx.setp_ge_u32(n_idx, n_param);
                ctx.branch_if(n_done, "n_loop_done");

                // B[clamped_col, n_idx]:
                // Block index = n_idx * num_k_blocks + col_block_idx
                let n_block_base = ctx.mul_u32_reg(n_idx, num_k_blocks);
                let block_idx = ctx.add_u32_reg(n_block_base, col_block_idx);

                // Load scale
                let scale_byte_off = ctx.mul_wide_u32(block_idx, 4);
                let scale_addr = ctx.add_u64(b_scales_ptr, scale_byte_off);
                let scale = ctx.ld_global_f32(scale_addr);

                // Load packed byte from NF4 data
                let data_block_off = ctx.mul_wide_u32(block_idx, 32);
                let data_block_addr = ctx.add_u64(b_nf4_ptr, data_block_off);
                let byte_off_64 = ctx.cvt_u64_u32(byte_in_block);
                let nibble_addr = ctx.add_u64(data_block_addr, byte_off_64);
                let packed_byte = ctx.ld_global_u8(nibble_addr);
                let packed_u32 = ctx.cvt_u32_u8(packed_byte);

                // Extract nibble
                let shifted = ctx.shr_u32(packed_u32, nibble_shift);
                let nf4_idx = ctx.and_u32(shifted, mask_4bit);

                // Codebook lookup
                let nf4_idx_64 = ctx.cvt_u64_u32(nf4_idx);
                let lut_byte_off = ctx.mul_u64(nf4_idx_64, 4);
                let lut_addr = ctx.add_u64(smem_base, lut_byte_off);
                let normalized_val = ctx.ld_generic_f32(lut_addr);

                // Dequantize
                let dequant = ctx.mul_f32(scale, normalized_val);

                // Load A[clamped_row, n_idx]
                let n_idx_64 = ctx.cvt_u64_u32(n_idx);
                let a_elem_off = ctx.add_u64(a_row_offset, n_idx_64);
                let a_elem_bytes = ctx.mul_u64(a_elem_off, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_bytes);
                let a_val = ctx.ld_global_f32(a_addr);

                // GH-561: acc(f64) += f32_to_f64(a_val) * f32_to_f64(dequant)
                ctx.fma_f64_acc_inplace(acc, a_val, dequant);

                ctx.add_u32_inplace(n_idx, 1);
                ctx.branch("n_loop");

                ctx.label("n_loop_done");

                // Store result
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                let c_row_off = ctx.mul_wide_u32_reg(global_row, k_param);
                let global_col_64 = ctx.cvt_u64_u32(global_col);
                let c_elem_off = ctx.add_u64(c_row_off, global_col_64);
                let c_elem_bytes = ctx.mul_u64(c_elem_off, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_bytes);
                // GH-561: Convert f64 accumulator to f32 for store
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);
                ctx.st_global_f32(c_addr, acc_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_gemm_kernel_name() {
        let kernel = Nf4GemmKernel::new(128, 896, 896);
        assert_eq!(kernel.name(), "nf4_gemm_fused");
    }

    #[test]
    fn test_nf4_gemm_num_blocks_per_col() {
        let kernel = Nf4GemmKernel::new(128, 896, 896);
        assert_eq!(kernel.num_blocks_per_col(), 896 / 64);
    }

    #[test]
    fn test_nf4_gemm_ptx_emits() {
        let kernel = Nf4GemmKernel::new(128, 896, 896);
        let ptx = kernel.emit_ptx();

        // Verify kernel name appears
        assert!(ptx.contains("nf4_gemm_fused"), "PTX missing kernel name");

        // Verify parameters declared
        assert!(ptx.contains("a_ptr"), "PTX missing a_ptr param");
        assert!(ptx.contains("b_nf4_ptr"), "PTX missing b_nf4_ptr param");
        assert!(ptx.contains("b_scales_ptr"), "PTX missing b_scales_ptr param");
        assert!(ptx.contains("c_ptr"), "PTX missing c_ptr param");

        // Register-based LUT: no shared memory, codebook is in registers.
        // Verify selp instruction present (binary tree LUT lookup)
        assert!(ptx.contains("selp"), "PTX missing selp (register LUT)");

        // Verify vectorized v4 load (8x fewer loop iterations)
        assert!(ptx.contains("v4"), "PTX missing v4 vectorized load");

        // Verify FMA instruction present (8x unrolled accumulation)
        assert!(ptx.contains("fma"), "PTX missing fma instruction");
    }

    #[test]
    fn test_nf4_gemm_ptx_targets() {
        let kernel = Nf4GemmKernel::new(64, 64, 64);

        let ptx_70 = kernel.emit_ptx_for_target("sm_70");
        assert!(ptx_70.contains("sm_70"));

        let ptx_89 = kernel.emit_ptx_for_target("sm_89");
        assert!(ptx_89.contains("sm_89"));
    }

    #[test]
    fn test_nf4_gemm_with_tile_size() {
        let kernel = Nf4GemmKernel::new(128, 128, 128).with_tile_size(16);
        assert_eq!(kernel.tile_size, 16);

        // Should still emit valid PTX
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("nf4_gemm_fused"));
    }

    #[test]
    fn test_nf4_gemm_transpose_kernel_name() {
        let kernel = Nf4GemmTransposeKernel::new(128, 896, 896);
        assert_eq!(kernel.name(), "nf4_gemm_transpose");
    }

    #[test]
    fn test_nf4_gemm_transpose_ptx_emits() {
        let kernel = Nf4GemmTransposeKernel::new(128, 896, 896);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("nf4_gemm_transpose"), "PTX missing kernel name");
        assert!(ptx.contains("a_ptr"), "PTX missing a_ptr param");
        assert!(ptx.contains("b_nf4_ptr"), "PTX missing b_nf4_ptr param");
        assert!(ptx.contains("c_ptr"), "PTX missing c_ptr param");
        assert!(ptx.contains("fma"), "PTX missing fma instruction");
    }

    #[test]
    fn test_nf4_gemm_transpose_num_blocks() {
        let kernel = Nf4GemmTransposeKernel::new(128, 2560, 2560);
        assert_eq!(kernel.num_blocks_per_col(), 40); // 2560/64
    }

    #[test]
    fn test_nf4_gemm_qwen3_4b_dimensions() {
        // Qwen3-4B: hidden=2560, intermediate=6912, heads=32, kv_heads=8, head_dim=80
        // seq_len=128 for training

        // Q/O projection: (128, 2560, 2560)
        let q_proj = Nf4GemmKernel::new(128, 2560, 2560);
        assert_eq!(q_proj.num_blocks_per_col(), 40); // 2560/64

        // K/V projection: (128, 640, 2560)  (kv_hidden = 8 * 80 = 640)
        let kv_proj = Nf4GemmKernel::new(128, 640, 2560);
        assert_eq!(kv_proj.num_blocks_per_col(), 40);

        // Gate/Up projection: (128, 6912, 2560)
        let gate_proj = Nf4GemmKernel::new(128, 6912, 2560);
        assert_eq!(gate_proj.num_blocks_per_col(), 40);

        // Down projection: (128, 2560, 6912)
        let down_proj = Nf4GemmKernel::new(128, 2560, 6912);
        assert_eq!(down_proj.num_blocks_per_col(), 108); // 6912/64
    }
}
