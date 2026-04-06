//! Naive batched GEMM variant (one thread per output element, no shared memory).

#![allow(clippy::similar_names)]

use super::BatchedGemmKernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxType};

impl BatchedGemmKernel {
    pub(super) fn build_naive(&self) -> PtxKernel {
        // Naive Batched GEMM: each thread computes one element of C[batch, row, col]
        // Grid: (n, m, batch) - z-dimension indexes batch
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;

        PtxKernel::new("batched_gemm_naive")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                // Get batch index from ctaid.z
                let batch_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);

                // Calculate row and column from thread/block IDs
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);
                let ntid_y = ctx.special_reg(crate::ptx::PtxReg::NtidY);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid_x = ctx.special_reg(crate::ptx::PtxReg::NtidX);
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);

                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Bounds check
                let batch_param = ctx.load_param_u32("batch");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let pred_batch = ctx.setp_ge_u32(batch_idx, batch_param);
                ctx.branch_if(pred_batch, "exit");
                let pred_m = ctx.setp_ge_u32(row, m_param);
                ctx.branch_if(pred_m, "exit");
                let pred_n = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(pred_n, "exit");

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate batch offsets using immediate values
                // A batch offset = batch_idx * m * k * 4
                // B batch offset = batch_idx * k * n * 4
                // C batch offset = batch_idx * m * n * 4
                let a_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * k_val * 4);
                let b_batch_offset = ctx.mul_wide_u32(batch_idx, k_val * n_val * 4);
                let c_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * n_val * 4);

                let a_batch_ptr = ctx.add_u64(a_ptr, a_batch_offset);
                let b_batch_ptr = ctx.add_u64(b_ptr, b_batch_offset);
                let c_batch_ptr = ctx.add_u64(c_ptr, c_batch_offset);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate base offset for A[row, 0]
                let row_offset = ctx.mul_wide_u32(row, k_val * 4);
                let a_row_ptr = ctx.add_u64(a_batch_ptr, row_offset);

                // Calculate base offset for B[0, col]
                let col_offset = ctx.mul_wide_u32(col, 4);
                let b_col_base = ctx.add_u64(b_batch_ptr, col_offset);

                // Loop over K dimension
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_k");

                let pred_k = ctx.setp_ge_u32(i, k_param);
                ctx.branch_if(pred_k, "loop_end");

                // Load A[row, i]
                let i_offset = ctx.mul_wide_u32(i, 4);
                let a_addr = ctx.add_u64(a_row_ptr, i_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                // Load B[i, col]
                let b_row_offset = ctx.mul_wide_u32(i, n_val * 4);
                let b_addr = ctx.add_u64(b_col_base, b_row_offset);
                let b_val = ctx.ld_global_f32(b_addr);

                // acc += a_val * b_val
                ctx.fma_f32_inplace(acc, a_val, b_val);

                ctx.add_u32_inplace(i, 1);
                ctx.branch("loop_k");

                ctx.label("loop_end");

                // Store result: C[batch, row, col]
                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_row_ptr = ctx.add_u64(c_batch_ptr, c_row_offset);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_addr = ctx.add_u64(c_row_ptr, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
