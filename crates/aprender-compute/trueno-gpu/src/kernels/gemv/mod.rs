//! GEMV (General Matrix-Vector Multiply) Kernel
//!
//! Optimized for M=1 matmuls: y = A * x where A is (K×N), x is (K), y is (N)
//!
//! This is the critical path for LLM token generation where each new token
//! requires M=1 matmuls through all layers.
//!
//! # Performance Target
//! - Ollama: ~228 tok/s with cuBLAS GEMV
//! - Goal: Match cuBLAS performance without external dependencies
//!
//! # Strategy
//! - One warp (32 threads) per output element
//! - Each warp computes one dot product using warp shuffle reduce
//! - Coalesced memory access for weight matrix

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

/// GEMV kernel configuration
#[derive(Debug, Clone)]
pub struct GemvKernel {
    /// K dimension (input/reduction dimension)
    k: u32,
    /// N dimension (output dimension)
    n: u32,
}

impl GemvKernel {
    /// Create a new GEMV kernel for y = A * x
    ///
    /// # Arguments
    /// * `k` - Input vector length / matrix rows
    /// * `n` - Output vector length / matrix columns
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}

impl super::Kernel for GemvKernel {
    fn name(&self) -> &str {
        "gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let _k_val = self.k; // Used for documentation, kernel uses runtime k_dim
        let n_val = self.n;

        // Strategy: One warp (32 threads) per output element
        // Each thread loads K/32 elements, does partial sum, then warp shuffle reduce

        PtxKernel::new("gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "a_ptr") // Weight matrix (K × N), row-major: A[i,j] at i*N+j
            .param(PtxType::U64, "x_ptr") // Input vector (K)
            .param(PtxType::U32, "k_dim") // K dimension (for bounds check)
            .param(PtxType::U32, "n_dim") // N dimension (for bounds check)
            .build(move |ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output element y[block_id] = sum_k(A[k, block_id] * x[k])

                // Get output index (which column of A we're computing)
                let block_id = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(crate::ptx::PtxReg::TidX);

                // Bounds check: if block_id >= n_dim, return
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                // Load parameters
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                // Initialize partial sum
                let partial_sum = ctx.mov_f32_imm(0.0);

                // For row-major A[K×N]: A[i,j] is at offset i*N + j
                // We want y[j] = sum_i(A[i,j] * x[i]) for j=block_id
                // So we need A[thread_id, block_id], A[thread_id+32, block_id], etc.

                // Compute base address for this output column
                // A[0, block_id] = a_ptr + block_id * 4
                let col_offset = ctx.mul_wide_u32(block_id, 4);
                let a_col_base = ctx.add_u64(a_ptr, col_offset);

                // Row stride = N * 4 bytes (baked in for efficiency)
                let row_stride = n_val * 4;

                // Each thread processes elements: thread_id, thread_id+32, thread_id+64, ...
                // Unroll for common K values to avoid loop overhead
                // For K=4096, that's 128 iterations per thread

                // Simple loop: i = thread_id; while i < k_dim: process; i += 32
                // Start index (i = 0 + thread_id = thread_id)
                let zero_u32 = ctx.mov_u32_imm(0);
                let i = ctx.add_u32_reg(zero_u32, thread_id);

                ctx.label("loop_start");

                // Check if i < k_dim
                let done = ctx.setp_ge_u32(i, k_dim);
                ctx.branch_if(done, "loop_end");

                // Load x[i]
                let four = ctx.mov_u32_imm(4);
                let x_offset = ctx.mul_wide_u32_reg(i, four);
                let x_addr = ctx.add_u64(x_ptr, x_offset);
                let x_val = ctx.ld_global_f32(x_addr);

                // Load A[i, block_id] = a_ptr + i * N * 4 + block_id * 4
                //                     = a_col_base + i * row_stride
                let stride_val = ctx.mov_u32_imm(row_stride);
                let row_offset = ctx.mul_wide_u32_reg(i, stride_val);
                let a_addr = ctx.add_u64(a_col_base, row_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                // partial_sum += x[i] * A[i, block_id]
                ctx.fma_f32_inplace(partial_sum, x_val, a_val);

                // i += 32
                ctx.add_u32_inplace(i, 32);
                ctx.branch("loop_start");

                ctx.label("loop_end");

                // Warp shuffle reduce: sum all 32 partial sums
                // shfl.sync.down.f32 with decreasing offsets: 16, 8, 4, 2, 1

                // offset = 16
                let tmp16 = ctx.shfl_down_f32(partial_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(partial_sum, tmp16);

                // offset = 8
                let tmp8 = ctx.shfl_down_f32(partial_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(partial_sum, tmp8);

                // offset = 4
                let tmp4 = ctx.shfl_down_f32(partial_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(partial_sum, tmp4);

                // offset = 2
                let tmp2 = ctx.shfl_down_f32(partial_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(partial_sum, tmp2);

                // offset = 1
                let tmp1 = ctx.shfl_down_f32(partial_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(partial_sum, tmp1);

                // Only thread 0 writes the result (thread_id < 1 means thread_id == 0)
                let one = ctx.mov_u32_imm(1);
                let is_thread0 = ctx.setp_lt_u32(thread_id, one);
                ctx.branch_if_not(is_thread0, "exit");

                // Store y[block_id] = partial_sum
                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, partial_sum);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Coalesced GEMV kernel for decode throughput (DECODER-THROUGHPUT-SPEC §5.1)
///
/// Computes y = A × x where:
/// - A is K×N (row-major)
/// - x is K×1
/// - y is N×1
///
/// # Memory Access Pattern
///
/// Unlike the warp-reduce kernel, this uses coalesced memory access:
/// - 256 threads per block, each computing one output element
/// - Consecutive threads read consecutive memory addresses (stride = 4 bytes)
/// - Shared memory caches the input vector x
///
/// # Performance Target
///
/// - Current (non-coalesced): 4.41ms per 1×4096×4096
/// - Target (coalesced): <0.1ms per 1×4096×4096
/// - Memory bandwidth utilization: >90% (vs 1.4% current)
#[derive(Debug, Clone)]
pub struct CoalescedGemvKernel {
    /// K dimension (input/reduction dimension)
    k: u32,
    /// N dimension (output dimension)
    n: u32,
}

impl CoalescedGemvKernel {
    /// Create a new coalesced GEMV kernel for y = A * x
    ///
    /// # Arguments
    /// * `k` - Input vector length / matrix rows
    /// * `n` - Output vector length / matrix columns
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}

impl super::Kernel for CoalescedGemvKernel {
    fn name(&self) -> &str {
        "gemv_coalesced"
    }

    fn build_ptx(&self) -> PtxKernel {
        use crate::ptx::PtxReg;

        // Tile size for shared memory caching
        // Process K in tiles of 256 elements (matches block size)
        const TILE_SIZE: u32 = 256;
        // Unroll factor: process 4 elements per iteration = 4x less branch overhead
        const UNROLL: u32 = 4;

        PtxKernel::new("gemv_coalesced")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory((TILE_SIZE * 4) as usize) // Cache tile of x vector
            .build(move |ctx| {
                // Block config: 256 threads, grid = ceil(N/256) blocks
                // Each thread computes one output element y[col]
                // CRITICAL: ALL threads must participate in shared memory loads

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                let block_size = ctx.mov_u32_imm(TILE_SIZE);
                let col_base = ctx.mul_lo_u32(block_id, block_size);
                let col = ctx.add_u32_reg(col_base, thread_id);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let col_valid = ctx.setp_lt_u32(col, n_dim);

                // Initialize accumulator
                let sum = ctx.mov_f32_imm(0.0);

                let smem_base = ctx.shared_base_addr();
                let row = ctx.mov_u32_imm(0);
                let four = ctx.mov_u32_imm(4);

                // Pre-compute col*4 and col as u64 (used repeatedly in inner loop)
                let col_64 = ctx.cvt_u64_u32(col);

                ctx.label("row_loop");
                let row_done = ctx.setp_ge_u32(row, k_dim);
                ctx.branch_if(row_done, "row_loop_end");

                // === Cooperative load x into shared memory ===
                let x_idx = ctx.add_u32_reg(row, thread_id);
                let x_valid = ctx.setp_lt_u32(x_idx, k_dim);
                let x_offset = ctx.mul_wide_u32_reg(x_idx, four);
                let x_addr = ctx.add_u64(x_ptr, x_offset);
                let x_val = ctx.ld_global_f32_predicated(x_addr, x_valid, 0.0);

                let smem_thread_offset = ctx.mul_u32(thread_id, 4);
                let smem_thread_offset_64 = ctx.cvt_u64_u32(smem_thread_offset);
                let smem_addr = ctx.add_u64(smem_base, smem_thread_offset_64);
                ctx.st_shared_f32(smem_addr, x_val);

                ctx.bar_sync(0);

                ctx.branch_if_not(col_valid, "skip_compute");

                // Calculate tile bounds
                let remaining = ctx.sub_u32_reg(k_dim, row);
                let tile_end = ctx.min_u32(block_size, remaining);

                // Unrolled loop: process 4 elements per iteration
                let tile_idx = ctx.mov_u32_imm(0);
                // unroll_end = tile_end & ~3 (round down to multiple of 4)
                let mask = ctx.mov_u32_imm(0xFFFF_FFFC);
                let unroll_end = ctx.and_u32(tile_end, mask);

                ctx.label("unroll_loop");
                let unroll_done = ctx.setp_ge_u32(tile_idx, unroll_end);
                ctx.branch_if(unroll_done, "unroll_loop_end");

                // Process 4 elements: tile_idx, tile_idx+1, tile_idx+2, tile_idx+3
                // Element 0
                let smem_off0 = ctx.mul_u32(tile_idx, 4);
                let smem_off0_64 = ctx.cvt_u64_u32(smem_off0);
                let smem_addr0 = ctx.add_u64(smem_base, smem_off0_64);
                let x0 = ctx.ld_shared_f32(smem_addr0);

                let a_row0 = ctx.add_u32_reg(row, tile_idx);
                let a_row0_times_n = ctx.mul_wide_u32_reg(a_row0, n_dim);
                let a_off0 = ctx.add_u64(a_row0_times_n, col_64);
                let a_byte0 = ctx.mul_u64(a_off0, 4);
                let a_addr0 = ctx.add_u64(a_ptr, a_byte0);
                let a0 = ctx.ld_global_f32(a_addr0);
                ctx.fma_f32_inplace(sum, x0, a0);

                // Element 1
                let tile_idx_1 = ctx.add_u32(tile_idx, 1);
                let smem_off1 = ctx.mul_u32(tile_idx_1, 4);
                let smem_off1_64 = ctx.cvt_u64_u32(smem_off1);
                let smem_addr1 = ctx.add_u64(smem_base, smem_off1_64);
                let x1 = ctx.ld_shared_f32(smem_addr1);

                let a_row1 = ctx.add_u32_reg(row, tile_idx_1);
                let a_row1_times_n = ctx.mul_wide_u32_reg(a_row1, n_dim);
                let a_off1 = ctx.add_u64(a_row1_times_n, col_64);
                let a_byte1 = ctx.mul_u64(a_off1, 4);
                let a_addr1 = ctx.add_u64(a_ptr, a_byte1);
                let a1 = ctx.ld_global_f32(a_addr1);
                ctx.fma_f32_inplace(sum, x1, a1);

                // Element 2
                let tile_idx_2 = ctx.add_u32(tile_idx, 2);
                let smem_off2 = ctx.mul_u32(tile_idx_2, 4);
                let smem_off2_64 = ctx.cvt_u64_u32(smem_off2);
                let smem_addr2 = ctx.add_u64(smem_base, smem_off2_64);
                let x2 = ctx.ld_shared_f32(smem_addr2);

                let a_row2 = ctx.add_u32_reg(row, tile_idx_2);
                let a_row2_times_n = ctx.mul_wide_u32_reg(a_row2, n_dim);
                let a_off2 = ctx.add_u64(a_row2_times_n, col_64);
                let a_byte2 = ctx.mul_u64(a_off2, 4);
                let a_addr2 = ctx.add_u64(a_ptr, a_byte2);
                let a2 = ctx.ld_global_f32(a_addr2);
                ctx.fma_f32_inplace(sum, x2, a2);

                // Element 3
                let tile_idx_3 = ctx.add_u32(tile_idx, 3);
                let smem_off3 = ctx.mul_u32(tile_idx_3, 4);
                let smem_off3_64 = ctx.cvt_u64_u32(smem_off3);
                let smem_addr3 = ctx.add_u64(smem_base, smem_off3_64);
                let x3 = ctx.ld_shared_f32(smem_addr3);

                let a_row3 = ctx.add_u32_reg(row, tile_idx_3);
                let a_row3_times_n = ctx.mul_wide_u32_reg(a_row3, n_dim);
                let a_off3 = ctx.add_u64(a_row3_times_n, col_64);
                let a_byte3 = ctx.mul_u64(a_off3, 4);
                let a_addr3 = ctx.add_u64(a_ptr, a_byte3);
                let a3 = ctx.ld_global_f32(a_addr3);
                ctx.fma_f32_inplace(sum, x3, a3);

                // tile_idx += 4
                ctx.add_u32_inplace(tile_idx, UNROLL);
                ctx.branch("unroll_loop");

                ctx.label("unroll_loop_end");

                // Handle remaining 0-3 elements
                ctx.label("remainder_loop");
                let rem_done = ctx.setp_ge_u32(tile_idx, tile_end);
                ctx.branch_if(rem_done, "remainder_loop_end");

                let smem_off_r = ctx.mul_u32(tile_idx, 4);
                let smem_off_r_64 = ctx.cvt_u64_u32(smem_off_r);
                let smem_addr_r = ctx.add_u64(smem_base, smem_off_r_64);
                let x_r = ctx.ld_shared_f32(smem_addr_r);

                let a_row_r = ctx.add_u32_reg(row, tile_idx);
                let a_row_r_times_n = ctx.mul_wide_u32_reg(a_row_r, n_dim);
                let a_off_r = ctx.add_u64(a_row_r_times_n, col_64);
                let a_byte_r = ctx.mul_u64(a_off_r, 4);
                let a_addr_r = ctx.add_u64(a_ptr, a_byte_r);
                let a_r = ctx.ld_global_f32(a_addr_r);
                ctx.fma_f32_inplace(sum, x_r, a_r);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("remainder_loop");

                ctx.label("remainder_loop_end");
                ctx.label("skip_compute");

                ctx.bar_sync(0);

                ctx.add_u32_inplace(row, TILE_SIZE);
                ctx.branch("row_loop");

                ctx.label("row_loop_end");

                ctx.branch_if_not(col_valid, "exit");
                let y_offset = ctx.mul_wide_u32(col, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, sum);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;
