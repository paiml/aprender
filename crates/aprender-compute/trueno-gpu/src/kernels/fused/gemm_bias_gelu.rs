//! Fused GEMM + Bias + GELU kernel (WAPR-PERF-007)

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

use super::super::Kernel;

/// Fused GEMM + Bias + GELU kernel (WAPR-PERF-007)
///
/// Computes: output = GELU(A @ B + bias) in a single kernel launch.
/// Eliminates 3 kernel launches (GEMM, Bias, GELU) into 1.
///
/// For FFN in Whisper encoder:
/// - First linear: [seq, hidden] @ [hidden, intermediate] + bias -> GELU -> [seq, intermediate]
/// - This kernel handles that in ONE launch instead of THREE.
///
/// Grid: (N / block_size, M / block_size, 1)
/// Block: (block_size, block_size, 1) - typically 16x16
///
/// # Citations
///
/// - [Dao2022] FlashAttention: Fast and Memory-Efficient Exact Attention
/// - [Kwon2023] PagedAttention: Efficient LLM Serving
#[derive(Debug, Clone)]
pub struct FusedGemmBiasGeluKernel {
    /// M: Number of rows in A and C
    pub m: u32,
    /// N: Number of columns in B and C
    pub n: u32,
    /// K: Shared dimension (columns of A, rows of B)
    pub k: u32,
}

impl FusedGemmBiasGeluKernel {
    /// Create a new fused GEMM+Bias+GELU kernel
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

impl Kernel for FusedGemmBiasGeluKernel {
    fn name(&self) -> &str {
        "fused_gemm_bias_gelu"
    }

    fn build_ptx(&self) -> PtxKernel {
        let k_val = self.k;
        let n_val = self.n;

        PtxKernel::new("fused_gemm_bias_gelu")
            .param(PtxType::U64, "a_ptr") // Input matrix A [M, K]
            .param(PtxType::U64, "b_ptr") // Weight matrix B [K, N]
            .param(PtxType::U64, "bias_ptr") // Bias vector [N]
            .param(PtxType::U64, "c_ptr") // Output matrix C [M, N]
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(move |ctx| {
                // Calculate row and column from thread/block IDs
                // row = ctaid.y * ntid.y + tid.y
                // col = ctaid.x * ntid.x + tid.x
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_y = ctx.special_reg(PtxReg::NtidY);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);
                let tid_x = ctx.special_reg(PtxReg::TidX);

                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Load params
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Bounds check
                let pred_m = ctx.setp_ge_u32(row, m_param);
                ctx.branch_if(pred_m, "exit");
                let pred_n = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(pred_n, "exit");

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let bias_ptr = ctx.load_param_u64("bias_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate base offset for A[row, 0]
                let row_offset = ctx.mul_wide_u32(row, k_val * 4);
                let a_row_ptr = ctx.add_u64(a_ptr, row_offset);

                // Calculate base offset for B[0, col]
                let col_offset = ctx.mul_wide_u32(col, 4);
                let b_col_base = ctx.add_u64(b_ptr, col_offset);

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

                // i++
                ctx.add_u32_inplace(i, 1);
                ctx.branch("loop_k");

                ctx.label("loop_end");

                // Load and add bias[col]
                let bias_offset = ctx.mul_wide_u32(col, 4);
                let bias_addr = ctx.add_u64(bias_ptr, bias_offset);
                let bias_val = ctx.ld_global_f32(bias_addr);
                let acc_biased = ctx.add_f32(acc, bias_val);

                // ============================================
                // GELU approximation (fused in same kernel)
                // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                // ============================================
                let x = acc_biased;

                // Constants
                let sqrt_2_pi = ctx.mov_f32_imm(0.797_884_6); // sqrt(2/π)
                let c = ctx.mov_f32_imm(0.044_715);
                let half = ctx.mov_f32_imm(0.5);
                let one = ctx.mov_f32_imm(1.0);
                let two = ctx.mov_f32_imm(2.0);
                let zero = ctx.mov_f32_imm(0.0);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                // x³
                let x2 = ctx.mul_f32(x, x);
                let x3 = ctx.mul_f32(x2, x);

                // 0.044715 * x³
                let cx3 = ctx.mul_f32(c, x3);

                // x + 0.044715 * x³
                let inner = ctx.add_f32(x, cx3);

                // sqrt(2/π) * (x + 0.044715 * x³)
                let scaled = ctx.mul_f32(sqrt_2_pi, inner);

                // tanh approximation: tanh(x) = 2*sigmoid(2x) - 1
                let two_x = ctx.mul_f32(two, scaled);
                let neg_two_x = ctx.sub_f32(zero, two_x);
                let scaled_exp = ctx.mul_f32(neg_two_x, log2_e);
                let exp_neg = ctx.ex2_f32(scaled_exp);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                let two_sigmoid = ctx.mul_f32(two, sigmoid);
                let tanh = ctx.sub_f32(two_sigmoid, one);

                // 1 + tanh(...)
                let one_plus_tanh = ctx.add_f32(one, tanh);

                // 0.5 * x
                let half_x = ctx.mul_f32(half, x);

                // result = 0.5 * x * (1 + tanh(...))
                let result = ctx.mul_f32(half_x, one_plus_tanh);

                // Store result: C[row, col]
                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_row_ptr = ctx.add_u64(c_ptr, c_row_offset);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_addr = ctx.add_u64(c_row_ptr, c_col_offset);
                ctx.st_global_f32(c_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
