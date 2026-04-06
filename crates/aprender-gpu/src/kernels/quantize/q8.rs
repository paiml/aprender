//! Q8 Quantization Kernel (Activation Quantization)
//!
//! PAR-063-V4: Converts f32 activations to Q8_1 format for use with DP4A dot products.

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q8_1 Quantization kernel for activations (PAR-063-V4)
///
/// Converts f32 activations to Q8_1 format for use with DP4A dot products.
/// This is the key optimization used by llama.cpp to enable true DP4A SIMD.
///
/// # Q8_1 Format
///
/// Each block of 32 values is stored as:
/// - qs[32]: 32 x int8 quantized values
/// - d: f16 scale factor
/// - s: f16 sum of values (for min contribution in Q4K dot product)
///
/// Total: 34 bytes per 32 values = 8.5 bits per value
///
/// # Quantization Formula
///
/// ```text
/// max_abs = max(|x_0|, |x_1|, ..., |x_31|)
/// scale = max_abs / 127
/// q_i = round(x_i / scale)  // clamped to [-127, 127]
/// ```
///
/// # Performance Impact
///
/// By pre-quantizing activations:
/// - GEMV can use pure integer DP4A (4 MADs per instruction)
/// - Eliminates f32 activation loads in inner loop
/// - Expected 2-4x instruction reduction
///
/// # References
///
/// - llama.cpp: ggml_quantize_q8_1 in ggml-quants.c
/// - NVIDIA: dp4a.u32.s32 for unsigned weights × signed activations
#[derive(Debug, Clone)]
pub struct Q8QuantizeKernel {
    /// Number of elements to quantize (must be multiple of 32)
    pub n: u32,
}

impl Q8QuantizeKernel {
    /// Create a new Q8 quantization kernel
    #[must_use]
    pub fn new(n: u32) -> Self {
        Self { n }
    }

    /// Get number of Q8 blocks (32 values each)
    #[must_use]
    pub const fn num_blocks(&self) -> u32 {
        (self.n + 31) / 32
    }
}

impl Kernel for Q8QuantizeKernel {
    fn name(&self) -> &str {
        "q8_quantize"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Grid: one block per Q8 block (32 values)
        // Each warp (32 threads) processes one Q8 block cooperatively
        PtxKernel::new("q8_quantize")
            .param(PtxType::U64, "out_ptr") // Q8 output: [num_blocks * 34] bytes
            .param(PtxType::U64, "in_ptr") // f32 input: [n] floats
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);

                let n_dim = ctx.load_param_u32("n_dim");
                let num_blocks = ctx.add_u32(n_dim, 31);
                let num_blocks = ctx.div_u32(num_blocks, 32);

                // Bounds check
                let oob = ctx.setp_ge_u32(block_id, num_blocks);
                ctx.branch_if(oob, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let in_ptr = ctx.load_param_u64("in_ptr");

                // Each thread loads 1 value (32 threads = 32 values = 1 Q8 block)
                let block_start = ctx.mul_u32(block_id, 32);
                let idx = ctx.add_u32_reg(block_start, lane_id);

                // Load f32 value
                let idx_64 = ctx.cvt_u64_u32(idx);
                let idx_bytes = ctx.mul_u64(idx_64, 4);
                let in_addr = ctx.add_u64(in_ptr, idx_bytes);
                let val = ctx.ld_global_f32(in_addr);

                // Compute absolute value
                let abs_val = ctx.abs_f32(val);

                // Find max absolute value across warp using shuffle reduction
                let max_abs = abs_val;
                let tmp16 = ctx.shfl_down_f32(max_abs, 16, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp16);
                let tmp8 = ctx.shfl_down_f32(max_abs, 8, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp8);
                let tmp4 = ctx.shfl_down_f32(max_abs, 4, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp4);
                let tmp2 = ctx.shfl_down_f32(max_abs, 2, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp2);
                let tmp1 = ctx.shfl_down_f32(max_abs, 1, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp1);

                // Broadcast max to all lanes
                let max_abs = ctx.shfl_idx_f32(max_abs, 0, 0xFFFF_FFFF);

                // Compute scale: d = max_abs / 127
                let inv_127 = ctx.mov_f32_imm(1.0 / 127.0);
                let scale = ctx.mul_f32(max_abs, inv_127);

                // Compute inverse scale for quantization
                let eps = ctx.mov_f32_imm(1e-10);
                let scale_eps = ctx.add_f32(scale, eps);
                let inv_scale = ctx.rcp_f32(scale_eps);

                // Quantize: q = round(val * inv_scale) clamped to [-127, 127]
                let scaled = ctx.mul_f32(val, inv_scale);
                let rounded = ctx.cvt_rni_s32_f32(scaled);

                // Clamp to [-127, 127]
                let min_val = ctx.mov_u32_imm(0xFFFF_FF81); // -127 as u32
                let min_s32 = ctx.mov_s32_from_u32(min_val);
                let max_val = ctx.mov_s32_imm(127);
                let clamped = ctx.max_s32(rounded, min_s32);
                let clamped = ctx.min_s32(clamped, max_val);

                // Convert to u8 (as signed byte stored in unsigned format)
                let q8_val = ctx.cvt_u8_s32(clamped);

                // Store quantized value
                // Q8_1 layout: [32 bytes qs] [2 bytes d] [2 bytes s]
                // Output offset for this block: block_id * 36 bytes
                let block_bytes = ctx.mov_u32_imm(36);
                let block_offset = ctx.mul_wide_u32_reg(block_id, block_bytes);
                let block_base = ctx.add_u64(out_ptr, block_offset);

                // Store qs[lane_id]
                let lane_64 = ctx.cvt_u64_u32(lane_id);
                let qs_addr = ctx.add_u64(block_base, lane_64);
                ctx.st_global_u8(qs_addr, q8_val);

                // Only lane 0 stores scale (d) and sum (s)
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);
                ctx.branch_if_not(is_lane0, "exit");

                // Store scale at offset 32
                let thirty_two_64 = ctx.mov_u64_imm(32);
                let d_addr = ctx.add_u64(block_base, thirty_two_64);
                let scale_f16 = ctx.cvt_f16_f32(scale);
                ctx.st_global_f16(d_addr, scale_f16);

                // Compute sum of values for min contribution (warp reduction)
                // Note: sum is already computed from original values
                let sum = val;
                let sum_tmp16 = ctx.shfl_down_f32(sum, 16, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp16);
                let sum_tmp8 = ctx.shfl_down_f32(sum, 8, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp8);
                let sum_tmp4 = ctx.shfl_down_f32(sum, 4, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp4);
                let sum_tmp2 = ctx.shfl_down_f32(sum, 2, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp2);
                let sum_tmp1 = ctx.shfl_down_f32(sum, 1, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp1);

                // Store sum at offset 34
                let thirty_four_64 = ctx.mov_u64_imm(34);
                let s_addr = ctx.add_u64(block_base, thirty_four_64);
                let sum_f16 = ctx.cvt_f16_f32(sum);
                ctx.st_global_f16(s_addr, sum_f16);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
