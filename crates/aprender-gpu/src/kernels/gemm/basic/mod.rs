//! Basic GEMM Kernel (naive, tiled, tensor core variants)
//!
//! Implements C = alpha * A @ B + beta * C for standard 2D matrix multiplication.

#![allow(clippy::similar_names)]

pub(crate) mod tensor_core;
mod tiled_unrolled;

#[cfg(test)]
mod tests;

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

/// GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// M dimension (rows of A and C)
    pub m: u32,
    /// N dimension (cols of B and C)
    pub n: u32,
    /// K dimension (cols of A, rows of B)
    pub k: u32,
    /// Tile size for shared memory
    pub tile_size: u32,
    /// Use Tensor Cores (requires FP16 and SM >= 70)
    pub use_tensor_cores: bool,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self { m: 1024, n: 1024, k: 1024, tile_size: 32, use_tensor_cores: false }
    }
}

/// GEMM kernel
#[derive(Debug, Clone)]
pub struct GemmKernel {
    /// Kernel configuration
    pub config: GemmConfig,
    variant: GemmVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmVariant {
    Naive,
    Tiled,
    /// Tiled with 4x unrolled inner loop (WAPR-PERF-009)
    TiledUnrolled,
    TensorCore,
    /// True WMMA using Tensor Core PTX intrinsics (sm_70+)
    WmmaFp16,
}

impl GemmKernel {
    /// Create naive GEMM kernel (for correctness testing)
    #[must_use]
    pub fn naive(m: u32, n: u32, k: u32) -> Self {
        Self { config: GemmConfig { m, n, k, ..Default::default() }, variant: GemmVariant::Naive }
    }

    /// Create tiled GEMM kernel (for performance)
    #[must_use]
    pub fn tiled(m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: GemmConfig { m, n, k, tile_size, ..Default::default() },
            variant: GemmVariant::Tiled,
        }
    }

    /// Create tiled GEMM kernel with 4x unrolled inner loop (WAPR-PERF-009)
    /// Reduces loop overhead from 12:1 to ~4:1 instructions per FMA
    #[must_use]
    pub fn tiled_unrolled(m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: GemmConfig { m, n, k, tile_size, ..Default::default() },
            variant: GemmVariant::TiledUnrolled,
        }
    }

    /// Create Tensor Core GEMM kernel (highest performance)
    #[must_use]
    pub fn tensor_core(m: u32, n: u32, k: u32) -> Self {
        Self {
            config: GemmConfig { m, n, k, use_tensor_cores: true, ..Default::default() },
            variant: GemmVariant::TensorCore,
        }
    }

    /// Create WMMA FP16 GEMM kernel using true Tensor Core PTX intrinsics
    /// Requires sm_70+ (Volta or later). Input is FP32, converted to FP16 internally.
    /// Output is FP32. Dimensions must be multiples of 16.
    #[must_use]
    pub fn wmma_fp16(m: u32, n: u32, k: u32) -> Self {
        Self {
            config: GemmConfig {
                m,
                n,
                k,
                tile_size: 16, // WMMA uses 16x16x16 tiles
                use_tensor_cores: true,
            },
            variant: GemmVariant::WmmaFp16,
        }
    }
}

impl Kernel for GemmKernel {
    fn name(&self) -> &str {
        match self.variant {
            GemmVariant::Naive => "gemm_naive",
            GemmVariant::Tiled => "gemm_tiled",
            GemmVariant::TiledUnrolled => "gemm_tiled_unrolled",
            GemmVariant::TensorCore => "gemm_tensor_core",
            GemmVariant::WmmaFp16 => "gemm_wmma_fp16",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match self.variant {
            GemmVariant::Naive => self.build_naive(),
            GemmVariant::Tiled => self.build_tiled(),
            GemmVariant::TiledUnrolled => self.build_tiled_unrolled(),
            GemmVariant::TensorCore => self.build_tensor_core(),
            GemmVariant::WmmaFp16 => self.build_wmma_fp16(),
        }
    }
}

impl GemmKernel {
    fn build_naive(&self) -> PtxKernel {
        // Naive GEMM: each thread computes one element of C
        // C[row, col] = sum(A[row, i] * B[i, col] for i in 0..K)
        let k_val = self.config.k;

        PtxKernel::new("gemm_naive")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                // Calculate row and column from thread/block IDs
                // row = ctaid.y * ntid.y + tid.y
                // col = ctaid.x * ntid.x + tid.x
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);
                let ntid_y = ctx.special_reg(crate::ptx::PtxReg::NtidY);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid_x = ctx.special_reg(crate::ptx::PtxReg::NtidX);
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);

                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Bounds check: if (row >= m || col >= n) return
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let pred_m = ctx.setp_ge_u32(row, m_param);
                ctx.branch_if(pred_m, "exit");
                let pred_n = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(pred_n, "exit");

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate base offset for A[row, 0] = a_ptr + row * K * 4
                let row_offset = ctx.mul_wide_u32(row, k_val * 4);
                let a_row_ptr = ctx.add_u64(a_ptr, row_offset);

                // Calculate base offset for B[0, col] = b_ptr + col * 4
                let col_offset = ctx.mul_wide_u32(col, 4);
                let b_col_base = ctx.add_u64(b_ptr, col_offset);

                // Loop over K dimension
                // For simplicity, unroll by 1 (production would unroll more)
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_k");

                // Check loop condition: if (i >= k) goto loop_end
                let pred_k = ctx.setp_ge_u32(i, k_param);
                ctx.branch_if(pred_k, "loop_end");

                // Load A[row, i] = a_row_ptr + i * 4
                let i_offset = ctx.mul_wide_u32(i, 4);
                let a_addr = ctx.add_u64(a_row_ptr, i_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                // Load B[i, col] = b_col_base + i * N * 4
                let b_row_offset = ctx.mul_wide_u32(i, self.config.n * 4);
                let b_addr = ctx.add_u64(b_col_base, b_row_offset);
                let b_val = ctx.ld_global_f32(b_addr);

                // acc += a_val * b_val (FMA) - IN-PLACE UPDATE
                ctx.fma_f32_inplace(acc, a_val, b_val);

                // i++ - IN-PLACE UPDATE
                ctx.add_u32_inplace(i, 1);

                // Branch back to loop
                ctx.branch("loop_k");

                ctx.label("loop_end");

                // Store result: C[row, col] = c_ptr + (row * N + col) * 4
                let c_row_offset = ctx.mul_wide_u32(row, self.config.n * 4);
                let c_row_ptr = ctx.add_u64(c_ptr, c_row_offset);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_addr = ctx.add_u64(c_row_ptr, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    fn build_tiled(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2; // A and B tiles, f32
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;

        PtxKernel::new("gemm_tiled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Tiled GEMM: Uses shared memory to reduce global memory traffic
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                // Global row and column
                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                // Load parameters (but DON'T exit early - all threads must participate in barriers)
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");

                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                let smem_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_a_offset = ctx.mul_u32(smem_idx, 4);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_b_offset = ctx.add_u32_reg(smem_b_base, smem_a_offset);

                // Load A tile
                let tile_k_offset = ctx.mul_u32(tile_idx, tile_size);
                let a_col = ctx.add_u32_reg(tile_k_offset, tid_x);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_a_offset, zero_a);

                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");
                let row_offset_a = ctx.mul_wide_u32(row, self.config.k * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_base = ctx.add_u64(a_ptr, row_offset_a);
                let a_addr = ctx.add_u64(a_row_base, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);
                ctx.label("skip_a_load");

                // Load B tile
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");
                let row_offset_b = ctx.mul_wide_u32(b_row, self.config.n * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_base = ctx.add_u64(b_ptr, row_offset_b);
                let b_addr = ctx.add_u64(b_row_base, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);
                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // Inner loop
                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("inner_k_loop");

                let inner_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                let as_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, inner_k);
                let as_addr = ctx.mul_u32(as_idx, 4);
                let a_shared = ctx.ld_shared_f32(as_addr);

                let bs_idx = ctx.mad_lo_u32(inner_k, tile_size_reg, tid_x);
                let bs_idx_bytes = ctx.mul_u32(bs_idx, 4);
                let bs_addr = ctx.add_u32_reg(smem_b_base, bs_idx_bytes);
                let b_shared = ctx.ld_shared_f32(bs_addr);

                ctx.fma_f32_inplace(acc, a_shared, b_shared);

                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                let c_row_offset = ctx.mul_wide_u32(row, self.config.n * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_base = ctx.add_u64(c_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_row_base, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
