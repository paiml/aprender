//! GEMM Backward Kernels (trueno#109)
//!
//! Backward (gradient) kernels for matrix multiplication with tiled variants.
//!
//! ## Mathematical Specification
//!
//! Forward: `C = A @ B` where A is (M, K) and B is (K, N)
//!
//! Backward:
//! - `∂L/∂A = ∂L/∂C @ B^T` (shape: M×N @ N×K = M×K)
//! - `∂L/∂B = A^T @ ∂L/∂C` (shape: K×M @ M×N = K×N)
//!
//! ## Variants
//!
//! - **Naive**: One thread per output element (correctness baseline)
//! - **Tiled**: Shared memory tiling (C-TILE-BWD-001/002)
//! - **TiledUnrolled**: 4x unrolled inner loop (C-TILE-BWD-005)
//!
//! ## Contract: gemm-backward-tiled-v1.yaml
//!
//! Tiled backward matches naive within ε < 1e-4 (C-TILE-BWD-003/004).
//! Arithmetic intensity: naive ~0.5 → tiled TILE/4 FLOP/byte.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmBackwardVariant {
    Naive,
    Tiled,
    TiledUnrolled,
}

// =============================================================================
// GemmBackwardAKernel: grad_a[M,K] = grad_c[M,N] @ B^T[N,K]
// =============================================================================

/// GEMM Backward Kernel (gradient w.r.t. A)
///
/// Computes `grad_a = grad_c @ B^T` for GEMM C = A @ B.
///
/// # Tiling Strategy (backward A)
///
/// Output is grad_a[M, K]. Reduction over N.
/// - Load grad_c tile: `smem_gc[ty][tx] = grad_c[row, tile_n + tx]` (coalesced)
/// - Load B tile transposed: `smem_b[tx][ty] = B[block_k + ty, tile_n + tx]` (coalesced load,
///   transposed store so inner loop reads `smem_b[inner_k][tx] = B[block_k + tx, tile_n + inner_k]`)
/// - Inner loop: `acc += smem_gc[ty][k] * smem_b[k][tx]`
#[derive(Debug, Clone)]
pub struct GemmBackwardAKernel {
    /// M dimension (rows of A)
    pub m: u32,
    /// N dimension (cols of B / reduction dim)
    pub n: u32,
    /// K dimension (cols of A / rows of B)
    pub k: u32,
    tile_size: u32,
    variant: GemmBackwardVariant,
}

impl GemmBackwardAKernel {
    /// Create a naive GEMM backward kernel (backward compatible)
    #[must_use]
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k, tile_size: 16, variant: GemmBackwardVariant::Naive }
    }

    /// Create a naive GEMM backward kernel (explicit)
    #[must_use]
    pub const fn naive(m: u32, n: u32, k: u32) -> Self {
        Self::new(m, n, k)
    }

    /// Create a tiled GEMM backward kernel with shared memory (C-TILE-BWD-001)
    #[must_use]
    pub const fn tiled(m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self { m, n, k, tile_size, variant: GemmBackwardVariant::Tiled }
    }

    /// Create a tiled GEMM backward kernel with 4x unrolled inner loop (C-TILE-BWD-005)
    #[must_use]
    pub const fn tiled_unrolled(m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self { m, n, k, tile_size, variant: GemmBackwardVariant::TiledUnrolled }
    }
}

impl Kernel for GemmBackwardAKernel {
    fn name(&self) -> &str {
        match self.variant {
            GemmBackwardVariant::Naive => "gemm_backward_a",
            GemmBackwardVariant::Tiled => "gemm_backward_a_tiled",
            GemmBackwardVariant::TiledUnrolled => "gemm_backward_a_tiled_unrolled",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match self.variant {
            GemmBackwardVariant::Naive => self.build_naive_a(),
            GemmBackwardVariant::Tiled => self.build_tiled_a(),
            GemmBackwardVariant::TiledUnrolled => self.build_tiled_unrolled_a(),
        }
    }
}

impl GemmBackwardAKernel {
    fn build_naive_a(&self) -> PtxKernel {
        PtxKernel::new("gemm_backward_a")
            .param(PtxType::U64, "grad_c_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "grad_a_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);
                let ntid_y = ctx.special_reg(PtxReg::NtidY);

                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                let m = ctx.load_param_u32("m");
                let n = ctx.load_param_u32("n");
                let k = ctx.load_param_u32("k");
                let grad_c_ptr = ctx.load_param_u64("grad_c_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let grad_a_ptr = ctx.load_param_u64("grad_a_ptr");

                let valid_row = ctx.setp_lt_u32(row, m);
                ctx.branch_if_not(valid_row, "exit");
                let valid_col = ctx.setp_lt_u32(col, k);
                ctx.branch_if_not(valid_col, "exit");

                // GH-561: f64 accumulator to prevent NaN on sm_121 (Blackwell)
                let acc = ctx.mov_f64_imm_zero();
                let four = ctx.mov_u32_imm(4);
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_start");
                let loop_cond = ctx.setp_lt_u32(i, n);
                ctx.branch_if_not(loop_cond, "loop_end");

                let grad_c_row_offset = ctx.mul_lo_u32(row, n);
                let grad_c_elem_idx = ctx.add_u32_reg(grad_c_row_offset, i);
                let grad_c_byte_offset = ctx.mul_wide_u32_reg(grad_c_elem_idx, four);
                let grad_c_addr = ctx.add_u64(grad_c_ptr, grad_c_byte_offset);
                let grad_c_val = ctx.ld_global_f32(grad_c_addr);

                let b_row_offset = ctx.mul_lo_u32(col, n);
                let b_elem_idx = ctx.add_u32_reg(b_row_offset, i);
                let b_byte_offset = ctx.mul_wide_u32_reg(b_elem_idx, four);
                let b_addr = ctx.add_u64(b_ptr, b_byte_offset);
                let b_val = ctx.ld_global_f32(b_addr);

                ctx.fma_f64_acc_inplace(acc, grad_c_val, b_val);

                ctx.add_u32_inplace(i, 1);
                ctx.branch("loop_start");

                ctx.label("loop_end");

                let grad_a_row_offset = ctx.mul_lo_u32(row, k);
                let grad_a_elem_idx = ctx.add_u32_reg(grad_a_row_offset, col);
                let grad_a_byte_offset = ctx.mul_wide_u32_reg(grad_a_elem_idx, four);
                let grad_a_addr = ctx.add_u64(grad_a_ptr, grad_a_byte_offset);
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);
                ctx.st_global_f32(grad_a_addr, acc_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Tiled backward A: shared memory tiles for grad_c and B^T.
    ///
    /// B tile is stored transposed in shared memory so the inner loop
    /// reads smem_b[inner_k][tx] = B[block_k + tx, tile_n + inner_k].
    #[allow(clippy::too_many_lines)]
    fn build_tiled_a(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2; // grad_c tile + B tile
        let n_tiles = (self.n + tile_size - 1) / tile_size;

        PtxKernel::new("gemm_backward_a_tiled")
            .param(PtxType::U64, "grad_c_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "grad_a_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                // Output position in grad_a[M, K]
                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Bounds predicates (don't exit early — all threads must hit barriers)
                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, k_param);

                let grad_c_ptr = ctx.load_param_u64("grad_c_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let grad_a_ptr = ctx.load_param_u64("grad_a_ptr");

                // GH-561: f64 accumulator to prevent NaN on sm_121 (Blackwell)
                let acc = ctx.mov_f64_imm_zero();

                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");
                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                // Shared memory layout:
                // smem_gc[ty][tx]: offset = (ty*TILE + tx) * 4
                // smem_b[tx][ty]: offset = TILE*TILE*4 + (tx*TILE + ty) * 4 (transposed store)
                let smem_gc_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_gc_offset = ctx.mul_u32(smem_gc_idx, 4);

                // B stored transposed: index = tx*TILE + ty
                let smem_bt_idx = ctx.mad_lo_u32(tid_x, tile_size_reg, tid_y);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_bt_bytes = ctx.mul_u32(smem_bt_idx, 4);
                let smem_b_offset = ctx.add_u32_reg(smem_b_base, smem_bt_bytes);

                // --- Load grad_c tile: grad_c[row, tile_n + tid_x] ---
                let tile_n_offset = ctx.mul_u32(tile_idx, tile_size);
                let gc_col = ctx.add_u32_reg(tile_n_offset, tid_x);
                let gc_col_valid = ctx.setp_lt_u32(gc_col, n_param);

                let zero = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_gc_offset, zero);

                ctx.branch_if_not(row_valid, "skip_gc_load");
                ctx.branch_if_not(gc_col_valid, "skip_gc_load");
                let gc_row_off = ctx.mul_wide_u32(row, self.n * 4);
                let gc_col_off = ctx.mul_wide_u32(gc_col, 4);
                let gc_row_base = ctx.add_u64(grad_c_ptr, gc_row_off);
                let gc_addr = ctx.add_u64(gc_row_base, gc_col_off);
                let gc_val = ctx.ld_global_f32(gc_addr);
                ctx.st_shared_f32(smem_gc_offset, gc_val);
                ctx.label("skip_gc_load");

                // --- Load B tile: B[block_k + tid_y, tile_n + tid_x] → store transposed ---
                // block_k = ctaid_x * TILE
                let block_k = ctx.mul_u32_reg(ctaid_x, tile_size_reg);
                let b_k_row = ctx.add_u32_reg(block_k, tid_y);
                let b_k_valid = ctx.setp_lt_u32(b_k_row, k_param);
                let b_n_col = gc_col; // same N-position as grad_c tile

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(b_k_valid, "skip_b_load");
                ctx.branch_if_not(gc_col_valid, "skip_b_load");
                // B is K×N row-major: offset = b_k_row * N + b_n_col
                let b_row_off = ctx.mul_wide_u32(b_k_row, self.n * 4);
                let b_col_off = ctx.mul_wide_u32(b_n_col, 4);
                let b_row_base = ctx.add_u64(b_ptr, b_row_off);
                let b_addr = ctx.add_u64(b_row_base, b_col_off);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);
                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // --- Inner loop: acc += smem_gc[ty][k] * smem_b[k][tx] ---
                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("inner_k_loop");
                let inner_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                // smem_gc[ty][inner_k] at offset (ty * TILE + inner_k) * 4
                let gc_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, inner_k);
                let gc_addr_s = ctx.mul_u32(gc_idx, 4);
                let gc_shared = ctx.ld_shared_f32(gc_addr_s);

                // smem_b[inner_k][tx] at offset TILE*TILE*4 + (inner_k * TILE + tx) * 4
                let b_idx = ctx.mad_lo_u32(inner_k, tile_size_reg, tid_x);
                let b_idx_bytes = ctx.mul_u32(b_idx, 4);
                let b_addr_s = ctx.add_u32_reg(smem_b_base, b_idx_bytes);
                let b_shared = ctx.ld_shared_f32(b_addr_s);

                ctx.fma_f64_acc_inplace(acc, gc_shared, b_shared);

                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                // Store result
                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                let ga_row_off = ctx.mul_wide_u32(row, self.k * 4);
                let ga_col_off = ctx.mul_wide_u32(col, 4);
                let ga_row_base = ctx.add_u64(grad_a_ptr, ga_row_off);
                let ga_addr = ctx.add_u64(ga_row_base, ga_col_off);
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);
                ctx.st_global_f32(ga_addr, acc_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Tiled + 4x unrolled inner loop for backward A.
    #[allow(clippy::too_many_lines)]
    fn build_tiled_unrolled_a(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2;
        let n_tiles = (self.n + tile_size - 1) / tile_size;
        let unroll_factor = 4u32;
        let unrolled_iters = tile_size / unroll_factor;

        PtxKernel::new("gemm_backward_a_tiled_unrolled")
            .param(PtxType::U64, "grad_c_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "grad_a_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, k_param);

                let grad_c_ptr = ctx.load_param_u64("grad_c_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let grad_a_ptr = ctx.load_param_u64("grad_a_ptr");

                // GH-561: f64 accumulator to prevent NaN on sm_121 (Blackwell)
                let acc = ctx.mov_f64_imm_zero();

                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");
                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                // Shared memory indices (same as tiled variant)
                let smem_gc_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_gc_offset = ctx.mul_u32(smem_gc_idx, 4);

                let smem_bt_idx = ctx.mad_lo_u32(tid_x, tile_size_reg, tid_y);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_bt_bytes = ctx.mul_u32(smem_bt_idx, 4);
                let smem_b_offset = ctx.add_u32_reg(smem_b_base, smem_bt_bytes);

                // Load grad_c tile
                let tile_n_offset = ctx.mul_u32(tile_idx, tile_size);
                let gc_col = ctx.add_u32_reg(tile_n_offset, tid_x);
                let gc_col_valid = ctx.setp_lt_u32(gc_col, n_param);

                let zero = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_gc_offset, zero);

                ctx.branch_if_not(row_valid, "skip_gc_load");
                ctx.branch_if_not(gc_col_valid, "skip_gc_load");
                let gc_row_off = ctx.mul_wide_u32(row, self.n * 4);
                let gc_col_off = ctx.mul_wide_u32(gc_col, 4);
                let gc_row_base = ctx.add_u64(grad_c_ptr, gc_row_off);
                let gc_addr = ctx.add_u64(gc_row_base, gc_col_off);
                let gc_val = ctx.ld_global_f32(gc_addr);
                ctx.st_shared_f32(smem_gc_offset, gc_val);
                ctx.label("skip_gc_load");

                // Load B tile (transposed store)
                let block_k = ctx.mul_u32_reg(ctaid_x, tile_size_reg);
                let b_k_row = ctx.add_u32_reg(block_k, tid_y);
                let b_k_valid = ctx.setp_lt_u32(b_k_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(b_k_valid, "skip_b_load");
                ctx.branch_if_not(gc_col_valid, "skip_b_load");
                let b_row_off = ctx.mul_wide_u32(b_k_row, self.n * 4);
                let b_col_off = ctx.mul_wide_u32(gc_col, 4);
                let b_row_base = ctx.add_u64(b_ptr, b_row_off);
                let b_addr = ctx.add_u64(b_row_base, b_col_off);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);
                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // 4x unrolled inner loop
                let inner_k = ctx.mov_u32_imm(0);
                let unrolled_iters_reg = ctx.mov_u32_imm(unrolled_iters);

                ctx.label("inner_k_loop");
                let inner_done = ctx.setp_ge_u32(inner_k, unrolled_iters_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                let k_base = ctx.mul_u32(inner_k, unroll_factor);

                // Iteration 0
                let k0 = k_base;
                let gc_idx0 = ctx.mad_lo_u32(tid_y, tile_size_reg, k0);
                let gc_addr0 = ctx.mul_u32(gc_idx0, 4);
                let gc_s0 = ctx.ld_shared_f32(gc_addr0);
                let b_idx0 = ctx.mad_lo_u32(k0, tile_size_reg, tid_x);
                let b_bytes0 = ctx.mul_u32(b_idx0, 4);
                let b_addr0 = ctx.add_u32_reg(smem_b_base, b_bytes0);
                let b_s0 = ctx.ld_shared_f32(b_addr0);
                ctx.fma_f64_acc_inplace(acc, gc_s0, b_s0);

                // Iteration 1
                let k1 = ctx.add_u32(k_base, 1);
                let gc_idx1 = ctx.mad_lo_u32(tid_y, tile_size_reg, k1);
                let gc_addr1 = ctx.mul_u32(gc_idx1, 4);
                let gc_s1 = ctx.ld_shared_f32(gc_addr1);
                let b_idx1 = ctx.mad_lo_u32(k1, tile_size_reg, tid_x);
                let b_bytes1 = ctx.mul_u32(b_idx1, 4);
                let b_addr1 = ctx.add_u32_reg(smem_b_base, b_bytes1);
                let b_s1 = ctx.ld_shared_f32(b_addr1);
                ctx.fma_f64_acc_inplace(acc, gc_s1, b_s1);

                // Iteration 2
                let k2 = ctx.add_u32(k_base, 2);
                let gc_idx2 = ctx.mad_lo_u32(tid_y, tile_size_reg, k2);
                let gc_addr2 = ctx.mul_u32(gc_idx2, 4);
                let gc_s2 = ctx.ld_shared_f32(gc_addr2);
                let b_idx2 = ctx.mad_lo_u32(k2, tile_size_reg, tid_x);
                let b_bytes2 = ctx.mul_u32(b_idx2, 4);
                let b_addr2 = ctx.add_u32_reg(smem_b_base, b_bytes2);
                let b_s2 = ctx.ld_shared_f32(b_addr2);
                ctx.fma_f64_acc_inplace(acc, gc_s2, b_s2);

                // Iteration 3
                let k3 = ctx.add_u32(k_base, 3);
                let gc_idx3 = ctx.mad_lo_u32(tid_y, tile_size_reg, k3);
                let gc_addr3 = ctx.mul_u32(gc_idx3, 4);
                let gc_s3 = ctx.ld_shared_f32(gc_addr3);
                let b_idx3 = ctx.mad_lo_u32(k3, tile_size_reg, tid_x);
                let b_bytes3 = ctx.mul_u32(b_idx3, 4);
                let b_addr3 = ctx.add_u32_reg(smem_b_base, b_bytes3);
                let b_s3 = ctx.ld_shared_f32(b_addr3);
                ctx.fma_f64_acc_inplace(acc, gc_s3, b_s3);

                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                let ga_row_off = ctx.mul_wide_u32(row, self.k * 4);
                let ga_col_off = ctx.mul_wide_u32(col, 4);
                let ga_row_base = ctx.add_u64(grad_a_ptr, ga_row_off);
                let ga_addr = ctx.add_u64(ga_row_base, ga_col_off);
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);
                ctx.st_global_f32(ga_addr, acc_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// =============================================================================
// GemmBackwardBKernel: grad_b[K,N] = A^T[K,M] @ grad_c[M,N]
// =============================================================================

/// GEMM Backward Kernel (gradient w.r.t. B)
///
/// Computes `grad_b = A^T @ grad_c` for GEMM C = A @ B.
///
/// # Tiling Strategy (backward B)
///
/// Output is grad_b[K, N]. Reduction over M.
/// - Load A tile transposed: `smem_a[tx][ty] = A[tile_m + ty, block_k + tx]` (coalesced load,
///   transposed store so inner loop reads `smem_a[ty][inner_k] = A[tile_m + inner_k, block_k + ty]`)
/// - Load grad_c tile: `smem_gc[ty][tx] = grad_c[tile_m + ty, block_n + tx]` (coalesced)
/// - Inner loop: `acc += smem_a[ty][k] * smem_gc[k][tx]`
#[derive(Debug, Clone)]
pub struct GemmBackwardBKernel {
    /// M dimension (rows of A)
    pub m: u32,
    /// N dimension (cols of B)
    pub n: u32,
    /// K dimension (cols of A / rows of B)
    pub k: u32,
    tile_size: u32,
    variant: GemmBackwardVariant,
}

impl GemmBackwardBKernel {
    /// Create a naive GEMM backward kernel (backward compatible)
    #[must_use]
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k, tile_size: 16, variant: GemmBackwardVariant::Naive }
    }

    /// Create a naive GEMM backward kernel (explicit)
    #[must_use]
    pub const fn naive(m: u32, n: u32, k: u32) -> Self {
        Self::new(m, n, k)
    }

    /// Create a tiled GEMM backward kernel with shared memory (C-TILE-BWD-002)
    #[must_use]
    pub const fn tiled(m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self { m, n, k, tile_size, variant: GemmBackwardVariant::Tiled }
    }

    /// Create a tiled GEMM backward kernel with 4x unrolled inner loop
    #[must_use]
    pub const fn tiled_unrolled(m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self { m, n, k, tile_size, variant: GemmBackwardVariant::TiledUnrolled }
    }
}

impl Kernel for GemmBackwardBKernel {
    fn name(&self) -> &str {
        match self.variant {
            GemmBackwardVariant::Naive => "gemm_backward_b",
            GemmBackwardVariant::Tiled => "gemm_backward_b_tiled",
            GemmBackwardVariant::TiledUnrolled => "gemm_backward_b_tiled_unrolled",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match self.variant {
            GemmBackwardVariant::Naive => self.build_naive_b(),
            GemmBackwardVariant::Tiled => self.build_tiled_b(),
            GemmBackwardVariant::TiledUnrolled => self.build_tiled_unrolled_b(),
        }
    }
}

impl GemmBackwardBKernel {
    fn build_naive_b(&self) -> PtxKernel {
        PtxKernel::new("gemm_backward_b")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "grad_c_ptr")
            .param(PtxType::U64, "grad_b_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);
                let ntid_y = ctx.special_reg(PtxReg::NtidY);

                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                let m = ctx.load_param_u32("m");
                let n = ctx.load_param_u32("n");
                let k = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let grad_c_ptr = ctx.load_param_u64("grad_c_ptr");
                let grad_b_ptr = ctx.load_param_u64("grad_b_ptr");

                let valid_row = ctx.setp_lt_u32(row, k);
                ctx.branch_if_not(valid_row, "exit");
                let valid_col = ctx.setp_lt_u32(col, n);
                ctx.branch_if_not(valid_col, "exit");

                // GH-561: f64 accumulator to prevent NaN on sm_121 (Blackwell)
                let acc = ctx.mov_f64_imm_zero();
                let four = ctx.mov_u32_imm(4);
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_start");
                let loop_cond = ctx.setp_lt_u32(i, m);
                ctx.branch_if_not(loop_cond, "loop_end");

                let a_row_offset = ctx.mul_lo_u32(i, k);
                let a_elem_idx = ctx.add_u32_reg(a_row_offset, row);
                let a_byte_offset = ctx.mul_wide_u32_reg(a_elem_idx, four);
                let a_addr = ctx.add_u64(a_ptr, a_byte_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                let grad_c_row_offset = ctx.mul_lo_u32(i, n);
                let grad_c_elem_idx = ctx.add_u32_reg(grad_c_row_offset, col);
                let grad_c_byte_offset = ctx.mul_wide_u32_reg(grad_c_elem_idx, four);
                let grad_c_addr = ctx.add_u64(grad_c_ptr, grad_c_byte_offset);
                let grad_c_val = ctx.ld_global_f32(grad_c_addr);

                ctx.fma_f64_acc_inplace(acc, a_val, grad_c_val);

                ctx.add_u32_inplace(i, 1);
                ctx.branch("loop_start");

                ctx.label("loop_end");

                let grad_b_row_offset = ctx.mul_lo_u32(row, n);
                let grad_b_elem_idx = ctx.add_u32_reg(grad_b_row_offset, col);
                let grad_b_byte_offset = ctx.mul_wide_u32_reg(grad_b_elem_idx, four);
                let grad_b_addr = ctx.add_u64(grad_b_ptr, grad_b_byte_offset);
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);
                ctx.st_global_f32(grad_b_addr, acc_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Tiled backward B: A tile stored transposed, grad_c tile normal.
    #[allow(clippy::too_many_lines)]
    fn build_tiled_b(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2;
        let m_tiles = (self.m + tile_size - 1) / tile_size;

        PtxKernel::new("gemm_backward_b_tiled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "grad_c_ptr")
            .param(PtxType::U64, "grad_b_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                // Output position in grad_b[K, N]
                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let row_valid = ctx.setp_lt_u32(row, k_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                let a_ptr = ctx.load_param_u64("a_ptr");
                let grad_c_ptr = ctx.load_param_u64("grad_c_ptr");
                let grad_b_ptr = ctx.load_param_u64("grad_b_ptr");

                // GH-561: f64 accumulator to prevent NaN on sm_121 (Blackwell)
                let acc = ctx.mov_f64_imm_zero();

                let tile_idx = ctx.mov_u32_imm(0);
                let m_tiles_reg = ctx.mov_u32_imm(m_tiles);

                ctx.label("tile_loop");
                let tile_done = ctx.setp_ge_u32(tile_idx, m_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                // Shared memory:
                // smem_a[tx][ty]: offset = (tx*TILE + ty) * 4  (transposed store)
                // smem_gc[ty][tx]: offset = TILE*TILE*4 + (ty*TILE + tx) * 4
                let smem_at_idx = ctx.mad_lo_u32(tid_x, tile_size_reg, tid_y);
                let smem_at_offset = ctx.mul_u32(smem_at_idx, 4);

                let smem_gc_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_gc_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_gc_bytes = ctx.mul_u32(smem_gc_idx, 4);
                let smem_gc_offset = ctx.add_u32_reg(smem_gc_base, smem_gc_bytes);

                let tile_m_offset = ctx.mul_u32(tile_idx, tile_size);

                // --- Load A tile: A[tile_m + tid_y, block_k + tid_x] → store transposed ---
                // block_k = ctaid_y * TILE (output row = K dimension)
                let a_m_row = ctx.add_u32_reg(tile_m_offset, tid_y);
                let block_k = ctx.mul_u32_reg(ctaid_y, tile_size_reg);
                let a_k_col = ctx.add_u32_reg(block_k, tid_x);

                let a_m_valid = ctx.setp_lt_u32(a_m_row, m_param);
                let a_k_valid = ctx.setp_lt_u32(a_k_col, k_param);

                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_at_offset, zero_a);

                ctx.branch_if_not(a_m_valid, "skip_a_load");
                ctx.branch_if_not(a_k_valid, "skip_a_load");
                // A is M×K row-major
                let a_row_off = ctx.mul_wide_u32(a_m_row, self.k * 4);
                let a_col_off = ctx.mul_wide_u32(a_k_col, 4);
                let a_row_base = ctx.add_u64(a_ptr, a_row_off);
                let a_addr = ctx.add_u64(a_row_base, a_col_off);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_at_offset, a_val);
                ctx.label("skip_a_load");

                // --- Load grad_c tile: grad_c[tile_m + tid_y, block_n + tid_x] ---
                let gc_m_row = a_m_row; // same M position
                let gc_n_col = col; // block_n + tid_x

                let zero_gc = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_gc_offset, zero_gc);

                ctx.branch_if_not(a_m_valid, "skip_gc_load");
                ctx.branch_if_not(col_valid, "skip_gc_load");
                // grad_c is M×N row-major
                let gc_row_off = ctx.mul_wide_u32(gc_m_row, self.n * 4);
                let gc_col_off = ctx.mul_wide_u32(gc_n_col, 4);
                let gc_row_base = ctx.add_u64(grad_c_ptr, gc_row_off);
                let gc_addr = ctx.add_u64(gc_row_base, gc_col_off);
                let gc_val = ctx.ld_global_f32(gc_addr);
                ctx.st_shared_f32(smem_gc_offset, gc_val);
                ctx.label("skip_gc_load");

                ctx.bar_sync(0);

                // Inner loop: acc += smem_a[ty][k] * smem_gc[k][tx]
                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("inner_k_loop");
                let inner_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                // smem_a[ty][inner_k] at offset (ty * TILE + inner_k) * 4
                let a_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, inner_k);
                let a_addr_s = ctx.mul_u32(a_idx, 4);
                let a_shared = ctx.ld_shared_f32(a_addr_s);

                // smem_gc[inner_k][tx] at offset TILE*TILE*4 + (inner_k * TILE + tx) * 4
                let gc_idx = ctx.mad_lo_u32(inner_k, tile_size_reg, tid_x);
                let gc_idx_bytes = ctx.mul_u32(gc_idx, 4);
                let gc_addr_s = ctx.add_u32_reg(smem_gc_base, gc_idx_bytes);
                let gc_shared = ctx.ld_shared_f32(gc_addr_s);

                ctx.fma_f64_acc_inplace(acc, a_shared, gc_shared);

                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                let gb_row_off = ctx.mul_wide_u32(row, self.n * 4);
                let gb_col_off = ctx.mul_wide_u32(col, 4);
                let gb_row_base = ctx.add_u64(grad_b_ptr, gb_row_off);
                let gb_addr = ctx.add_u64(gb_row_base, gb_col_off);
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);
                ctx.st_global_f32(gb_addr, acc_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Tiled + 4x unrolled inner loop for backward B.
    #[allow(clippy::too_many_lines)]
    fn build_tiled_unrolled_b(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2;
        let m_tiles = (self.m + tile_size - 1) / tile_size;
        let unroll_factor = 4u32;
        let unrolled_iters = tile_size / unroll_factor;

        PtxKernel::new("gemm_backward_b_tiled_unrolled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "grad_c_ptr")
            .param(PtxType::U64, "grad_b_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let row_valid = ctx.setp_lt_u32(row, k_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                let a_ptr = ctx.load_param_u64("a_ptr");
                let grad_c_ptr = ctx.load_param_u64("grad_c_ptr");
                let grad_b_ptr = ctx.load_param_u64("grad_b_ptr");

                // GH-561: f64 accumulator to prevent NaN on sm_121 (Blackwell)
                let acc = ctx.mov_f64_imm_zero();

                let tile_idx = ctx.mov_u32_imm(0);
                let m_tiles_reg = ctx.mov_u32_imm(m_tiles);

                ctx.label("tile_loop");
                let tile_done = ctx.setp_ge_u32(tile_idx, m_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                let smem_at_idx = ctx.mad_lo_u32(tid_x, tile_size_reg, tid_y);
                let smem_at_offset = ctx.mul_u32(smem_at_idx, 4);

                let smem_gc_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_gc_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_gc_bytes = ctx.mul_u32(smem_gc_idx, 4);
                let smem_gc_offset = ctx.add_u32_reg(smem_gc_base, smem_gc_bytes);

                let tile_m_offset = ctx.mul_u32(tile_idx, tile_size);

                // Load A tile (transposed store)
                let a_m_row = ctx.add_u32_reg(tile_m_offset, tid_y);
                let block_k = ctx.mul_u32_reg(ctaid_y, tile_size_reg);
                let a_k_col = ctx.add_u32_reg(block_k, tid_x);

                let a_m_valid = ctx.setp_lt_u32(a_m_row, m_param);
                let a_k_valid = ctx.setp_lt_u32(a_k_col, k_param);

                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_at_offset, zero_a);

                ctx.branch_if_not(a_m_valid, "skip_a_load");
                ctx.branch_if_not(a_k_valid, "skip_a_load");
                let a_row_off = ctx.mul_wide_u32(a_m_row, self.k * 4);
                let a_col_off = ctx.mul_wide_u32(a_k_col, 4);
                let a_row_base = ctx.add_u64(a_ptr, a_row_off);
                let a_addr = ctx.add_u64(a_row_base, a_col_off);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_at_offset, a_val);
                ctx.label("skip_a_load");

                // Load grad_c tile
                let gc_m_row = a_m_row;
                let gc_n_col = col;

                let zero_gc = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_gc_offset, zero_gc);

                ctx.branch_if_not(a_m_valid, "skip_gc_load");
                ctx.branch_if_not(col_valid, "skip_gc_load");
                let gc_row_off = ctx.mul_wide_u32(gc_m_row, self.n * 4);
                let gc_col_off = ctx.mul_wide_u32(gc_n_col, 4);
                let gc_row_base = ctx.add_u64(grad_c_ptr, gc_row_off);
                let gc_addr = ctx.add_u64(gc_row_base, gc_col_off);
                let gc_val = ctx.ld_global_f32(gc_addr);
                ctx.st_shared_f32(smem_gc_offset, gc_val);
                ctx.label("skip_gc_load");

                ctx.bar_sync(0);

                // 4x unrolled inner loop
                let inner_k = ctx.mov_u32_imm(0);
                let unrolled_iters_reg = ctx.mov_u32_imm(unrolled_iters);

                ctx.label("inner_k_loop");
                let inner_done = ctx.setp_ge_u32(inner_k, unrolled_iters_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                let k_base = ctx.mul_u32(inner_k, unroll_factor);

                // Iteration 0
                let k0 = k_base;
                let a_idx0 = ctx.mad_lo_u32(tid_y, tile_size_reg, k0);
                let a_addr0 = ctx.mul_u32(a_idx0, 4);
                let a_s0 = ctx.ld_shared_f32(a_addr0);
                let gc_idx0 = ctx.mad_lo_u32(k0, tile_size_reg, tid_x);
                let gc_bytes0 = ctx.mul_u32(gc_idx0, 4);
                let gc_addr0 = ctx.add_u32_reg(smem_gc_base, gc_bytes0);
                let gc_s0 = ctx.ld_shared_f32(gc_addr0);
                ctx.fma_f64_acc_inplace(acc, a_s0, gc_s0);

                // Iteration 1
                let k1 = ctx.add_u32(k_base, 1);
                let a_idx1 = ctx.mad_lo_u32(tid_y, tile_size_reg, k1);
                let a_addr1 = ctx.mul_u32(a_idx1, 4);
                let a_s1 = ctx.ld_shared_f32(a_addr1);
                let gc_idx1 = ctx.mad_lo_u32(k1, tile_size_reg, tid_x);
                let gc_bytes1 = ctx.mul_u32(gc_idx1, 4);
                let gc_addr1 = ctx.add_u32_reg(smem_gc_base, gc_bytes1);
                let gc_s1 = ctx.ld_shared_f32(gc_addr1);
                ctx.fma_f64_acc_inplace(acc, a_s1, gc_s1);

                // Iteration 2
                let k2 = ctx.add_u32(k_base, 2);
                let a_idx2 = ctx.mad_lo_u32(tid_y, tile_size_reg, k2);
                let a_addr2 = ctx.mul_u32(a_idx2, 4);
                let a_s2 = ctx.ld_shared_f32(a_addr2);
                let gc_idx2 = ctx.mad_lo_u32(k2, tile_size_reg, tid_x);
                let gc_bytes2 = ctx.mul_u32(gc_idx2, 4);
                let gc_addr2 = ctx.add_u32_reg(smem_gc_base, gc_bytes2);
                let gc_s2 = ctx.ld_shared_f32(gc_addr2);
                ctx.fma_f64_acc_inplace(acc, a_s2, gc_s2);

                // Iteration 3
                let k3 = ctx.add_u32(k_base, 3);
                let a_idx3 = ctx.mad_lo_u32(tid_y, tile_size_reg, k3);
                let a_addr3 = ctx.mul_u32(a_idx3, 4);
                let a_s3 = ctx.ld_shared_f32(a_addr3);
                let gc_idx3 = ctx.mad_lo_u32(k3, tile_size_reg, tid_x);
                let gc_bytes3 = ctx.mul_u32(gc_idx3, 4);
                let gc_addr3 = ctx.add_u32_reg(smem_gc_base, gc_bytes3);
                let gc_s3 = ctx.ld_shared_f32(gc_addr3);
                ctx.fma_f64_acc_inplace(acc, a_s3, gc_s3);

                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                let gb_row_off = ctx.mul_wide_u32(row, self.n * 4);
                let gb_col_off = ctx.mul_wide_u32(col, 4);
                let gb_row_base = ctx.add_u64(grad_b_ptr, gb_row_off);
                let gb_addr = ctx.add_u64(gb_row_base, gb_col_off);
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);
                ctx.st_global_f32(gb_addr, acc_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Backward A tests ===

    #[test]
    fn test_gemm_backward_a_name() {
        let kernel = GemmBackwardAKernel::new(64, 64, 64);
        assert_eq!(kernel.name(), "gemm_backward_a");
    }

    #[test]
    fn test_gemm_backward_a_ptx_generation() {
        let kernel = GemmBackwardAKernel::new(64, 64, 64);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gemm_backward_a"));
        assert!(ptx.contains(".param .u64 grad_c_ptr"));
        assert!(ptx.contains(".param .u64 b_ptr"));
        assert!(ptx.contains(".param .u64 grad_a_ptr"));
        assert!(ptx.contains("loop_start"));
        assert!(ptx.contains("loop_end"));
    }

    #[test]
    fn test_gemm_backward_a_barrier_safety() {
        let kernel = GemmBackwardAKernel::new(32, 32, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "GEMM backward A should be barrier-safe: {:?}", result.violations);
    }

    // === Backward A tiled tests (C-TILE-BWD-001) ===

    #[test]
    fn test_gemm_backward_a_tiled_name() {
        let kernel = GemmBackwardAKernel::tiled(128, 2560, 2560, 32);
        assert_eq!(kernel.name(), "gemm_backward_a_tiled");
    }

    #[test]
    fn test_gemm_backward_a_tiled_ptx_has_shared_memory() {
        let kernel = GemmBackwardAKernel::tiled(128, 2560, 2560, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".shared"), "C-TILE-BWD-001: must use shared memory");
        assert!(ptx.contains(".entry gemm_backward_a_tiled"));
        assert!(ptx.contains("bar.sync"), "tiled kernel must have barriers");
    }

    #[test]
    fn test_gemm_backward_a_tiled_ptx_has_fma() {
        let kernel = GemmBackwardAKernel::tiled(128, 2560, 2560, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fma.rn.f64"), "C-TILE-BWD-005: tiled must use f64 FMA (GH-561)");
    }

    #[test]
    fn test_gemm_backward_a_tiled_barrier_safety() {
        let kernel = GemmBackwardAKernel::tiled(128, 2560, 2560, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "C-TILE-BWD-006: barrier safety violated: {:?}", result.violations);
    }

    // === Backward A tiled unrolled tests ===

    #[test]
    fn test_gemm_backward_a_tiled_unrolled_name() {
        let kernel = GemmBackwardAKernel::tiled_unrolled(128, 2560, 2560, 32);
        assert_eq!(kernel.name(), "gemm_backward_a_tiled_unrolled");
    }

    #[test]
    fn test_gemm_backward_a_tiled_unrolled_ptx() {
        let kernel = GemmBackwardAKernel::tiled_unrolled(128, 2560, 2560, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".shared"), "must use shared memory");
        assert!(ptx.contains("fma.rn.f64"), "must use f64 FMA (GH-561)");
        assert!(ptx.contains("bar.sync"), "must have barriers");
    }

    #[test]
    fn test_gemm_backward_a_tiled_unrolled_barrier_safety() {
        let kernel = GemmBackwardAKernel::tiled_unrolled(128, 2560, 2560, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "C-TILE-BWD-006: barrier safety violated: {:?}", result.violations);
    }

    // === Backward B tests ===

    #[test]
    fn test_gemm_backward_b_name() {
        let kernel = GemmBackwardBKernel::new(64, 64, 64);
        assert_eq!(kernel.name(), "gemm_backward_b");
    }

    #[test]
    fn test_gemm_backward_b_ptx_generation() {
        let kernel = GemmBackwardBKernel::new(64, 64, 64);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gemm_backward_b"));
        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 grad_c_ptr"));
        assert!(ptx.contains(".param .u64 grad_b_ptr"));
        assert!(ptx.contains("loop_start"));
    }

    #[test]
    fn test_gemm_backward_b_barrier_safety() {
        let kernel = GemmBackwardBKernel::new(32, 32, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "GEMM backward B should be barrier-safe: {:?}", result.violations);
    }

    // === Backward B tiled tests (C-TILE-BWD-002) ===

    #[test]
    fn test_gemm_backward_b_tiled_name() {
        let kernel = GemmBackwardBKernel::tiled(128, 2560, 2560, 32);
        assert_eq!(kernel.name(), "gemm_backward_b_tiled");
    }

    #[test]
    fn test_gemm_backward_b_tiled_ptx_has_shared_memory() {
        let kernel = GemmBackwardBKernel::tiled(128, 2560, 2560, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".shared"), "C-TILE-BWD-002: must use shared memory");
        assert!(ptx.contains(".entry gemm_backward_b_tiled"));
        assert!(ptx.contains("bar.sync"), "tiled kernel must have barriers");
    }

    #[test]
    fn test_gemm_backward_b_tiled_ptx_has_fma() {
        let kernel = GemmBackwardBKernel::tiled(128, 2560, 2560, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fma.rn.f64"), "tiled must use f64 FMA (GH-561)");
    }

    #[test]
    fn test_gemm_backward_b_tiled_barrier_safety() {
        let kernel = GemmBackwardBKernel::tiled(128, 2560, 2560, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "C-TILE-BWD-006: barrier safety violated: {:?}", result.violations);
    }

    // === Backward B tiled unrolled tests ===

    #[test]
    fn test_gemm_backward_b_tiled_unrolled_ptx() {
        let kernel = GemmBackwardBKernel::tiled_unrolled(128, 2560, 2560, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".shared"), "must use shared memory");
        assert!(ptx.contains("fma.rn.f64"), "must use f64 FMA (GH-561)");
        assert!(ptx.contains("bar.sync"), "must have barriers");
        assert_eq!(kernel.name(), "gemm_backward_b_tiled_unrolled");
    }

    #[test]
    fn test_gemm_backward_b_tiled_unrolled_barrier_safety() {
        let kernel = GemmBackwardBKernel::tiled_unrolled(128, 2560, 2560, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "barrier safety violated: {:?}", result.violations);
    }

    // === Qwen3-4B training shape tests ===

    #[test]
    fn test_backward_a_qwen3_4b_shapes() {
        // Q projection backward: grad_q[128,2560] @ W_q^T[2560,2560]
        let q = GemmBackwardAKernel::tiled_unrolled(128, 2560, 2560, 32);
        let ptx = q.emit_ptx();
        assert!(ptx.contains("gemm_backward_a_tiled_unrolled"));

        // Down projection backward: grad[128,2560] @ W_down^T[2560,6912]
        let down = GemmBackwardAKernel::tiled_unrolled(128, 2560, 6912, 32);
        let ptx = down.emit_ptx();
        assert!(ptx.contains("gemm_backward_a_tiled_unrolled"));
    }

    #[test]
    fn test_backward_b_qwen3_4b_shapes() {
        // LoRA grad_B: lora_inter^T[16,128] @ grad_q[128,2560] → [16,2560]
        let lora = GemmBackwardBKernel::tiled_unrolled(128, 2560, 16, 16);
        let ptx = lora.emit_ptx();
        assert!(ptx.contains("gemm_backward_b_tiled_unrolled"));
    }

    // === PTX target tests ===

    #[test]
    fn test_tiled_backward_ptx_targets() {
        let kernel = GemmBackwardAKernel::tiled(64, 64, 64, 16);
        let ptx_70 = kernel.emit_ptx_for_target("sm_70");
        assert!(ptx_70.contains("sm_70"));
        let ptx_89 = kernel.emit_ptx_for_target("sm_89");
        assert!(ptx_89.contains("sm_89"));
    }
}
