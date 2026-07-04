//! Q8K Quantization Kernel (Activation Quantization)
//!
//! Converts f32 activations to Q8K format (256-element super-blocks).

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

#[derive(Debug, Clone)]
pub struct Q8KQuantizeKernel {
    pub n: u32,
}

impl Q8KQuantizeKernel {
    #[must_use]
    pub fn new(n: u32) -> Self {
        Self { n }
    }

    #[must_use]
    pub const fn num_blocks(&self) -> u32 {
        (self.n + 255) / 256
    }
}

impl Kernel for Q8KQuantizeKernel {
    fn name(&self) -> &str {
        "q8k_quantize"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q8k_quantize")
            .shared_memory(128) // 32 * 4 bytes
            .param(PtxType::U64, "scales_ptr") // f32
            .param(PtxType::U64, "quants_ptr") // i8
            .param(PtxType::U64, "in_ptr")     // f32
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);

                let n_dim = ctx.load_param_u32("n_dim");
                let num_blocks = ctx.add_u32(n_dim, 255);
                let num_blocks = ctx.div_u32(num_blocks, 256);

                let oob = ctx.setp_ge_u32(block_id, num_blocks);
                ctx.branch_if(oob, "exit");

                let in_ptr = ctx.load_param_u64("in_ptr");
                let scales_ptr = ctx.load_param_u64("scales_ptr");
                let quants_ptr = ctx.load_param_u64("quants_ptr");

                let block_start = ctx.mul_u32(block_id, 256);
                let idx = ctx.add_u32_reg(block_start, thread_id);
                let is_idx_valid = ctx.setp_lt_u32(idx, n_dim);

                let idx_64 = ctx.cvt_u64_u32(idx);
                let offset_bytes = ctx.mul_u64(idx_64, 4);
                let in_addr = ctx.add_u64(in_ptr, offset_bytes);

                let val = ctx.ld_global_f32_predicated(in_addr, is_idx_valid, 0.0);

                let abs_val = ctx.abs_f32(val);
                let mut max_abs = abs_val;

                let tmp16 = ctx.shfl_down_f32(max_abs, 16, 0xFFFF_FFFF);
                max_abs = ctx.max_f32(max_abs, tmp16);
                let tmp8 = ctx.shfl_down_f32(max_abs, 8, 0xFFFF_FFFF);
                max_abs = ctx.max_f32(max_abs, tmp8);
                let tmp4 = ctx.shfl_down_f32(max_abs, 4, 0xFFFF_FFFF);
                max_abs = ctx.max_f32(max_abs, tmp4);
                let tmp2 = ctx.shfl_down_f32(max_abs, 2, 0xFFFF_FFFF);
                max_abs = ctx.max_f32(max_abs, tmp2);
                let tmp1 = ctx.shfl_down_f32(max_abs, 1, 0xFFFF_FFFF);
                max_abs = ctx.max_f32(max_abs, tmp1);

                let smem_base = ctx.shared_base_addr();
                let zero = ctx.mov_u32_imm(0);

                let is_warp0 = ctx.setp_eq_u32(warp_id, zero);
                ctx.branch_if_not(is_warp0, "skip_init");
                let offset = ctx.mul_u32(lane_id, 4);
                let offset_64 = ctx.cvt_u64_u32(offset);
                let smem_addr = ctx.add_u64(smem_base, offset_64);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                ctx.st_generic_f32(smem_addr, zero_f32);
                ctx.label("skip_init");
                ctx.bar_sync(0);

                let is_lane0 = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane0, "skip_write");
                let offset = ctx.mul_u32(warp_id, 4);
                let offset_64 = ctx.cvt_u64_u32(offset);
                let smem_addr = ctx.add_u64(smem_base, offset_64);
                ctx.st_generic_f32(smem_addr, max_abs);
                ctx.label("skip_write");
                ctx.bar_sync(0);

                ctx.branch_if_not(is_warp0, "skip_read");
                let offset = ctx.mul_u32(lane_id, 4);
                let offset_64 = ctx.cvt_u64_u32(offset);
                let smem_addr = ctx.add_u64(smem_base, offset_64);
                let mut max_block_val = ctx.ld_generic_f32(smem_addr);

                let max_block_tmp16 = ctx.shfl_down_f32(max_block_val, 16, 0xFFFF_FFFF);
                max_block_val = ctx.max_f32(max_block_val, max_block_tmp16);
                let max_block_tmp8 = ctx.shfl_down_f32(max_block_val, 8, 0xFFFF_FFFF);
                max_block_val = ctx.max_f32(max_block_val, max_block_tmp8);
                let max_block_tmp4 = ctx.shfl_down_f32(max_block_val, 4, 0xFFFF_FFFF);
                max_block_val = ctx.max_f32(max_block_val, max_block_tmp4);
                let max_block_tmp2 = ctx.shfl_down_f32(max_block_val, 2, 0xFFFF_FFFF);
                max_block_val = ctx.max_f32(max_block_val, max_block_tmp2);
                let max_block_tmp1 = ctx.shfl_down_f32(max_block_val, 1, 0xFFFF_FFFF);
                max_block_val = ctx.max_f32(max_block_val, max_block_tmp1);

                ctx.branch_if_not(is_lane0, "skip_bcast_write");
                ctx.st_generic_f32(smem_base, max_block_val);
                ctx.label("skip_bcast_write");
                ctx.label("skip_read");
                ctx.bar_sync(0);

                let max_block = ctx.ld_generic_f32(smem_base);

                // scale = max_abs > 1e-10 ? max_abs / 127.0 : 1.0 / 127.0
                let eps = ctx.mov_f32_imm(1e-10);
                let inv_127 = ctx.mov_f32_imm(1.0 / 127.0);
                let max_gt_eps = ctx.setp_gt_f32(max_block, eps);
                let mut scale = ctx.mov_f32_imm(1.0 / 127.0);
                ctx.branch_if_not(max_gt_eps, "skip_scale");
                let scaled_max = ctx.mul_f32(max_block, inv_127);
                ctx.mov_f32_reg(scale, scaled_max);
                ctx.label("skip_scale");

                let inv_scale = ctx.rcp_f32(scale);

                ctx.branch_if_not(is_idx_valid, "skip_quant_store");
                let scaled = ctx.mul_f32(val, inv_scale);
                let rounded = ctx.cvt_rni_s32_f32(scaled);

                let min_val = ctx.mov_u32_imm(0xFFFF_FF80); // -128 as u32
                let min_s32 = ctx.mov_s32_from_u32(min_val);
                let max_val = ctx.mov_s32_imm(127);
                let clamped = ctx.max_s32(rounded, min_s32);
                let clamped = ctx.min_s32(clamped, max_val);

                let q8_val = ctx.cvt_u8_s32(clamped);
                let quants_addr = ctx.add_u64(quants_ptr, idx_64);
                ctx.st_global_u8(quants_addr, q8_val);
                ctx.label("skip_quant_store");

                let is_thread0 = ctx.setp_eq_u32(thread_id, zero);
                ctx.branch_if_not(is_thread0, "skip_scale_store");
                let block_id_64 = ctx.cvt_u64_u32(block_id);
                let offset_scale = ctx.mul_u64(block_id_64, 4);
                let scale_addr = ctx.add_u64(scales_ptr, offset_scale);
                ctx.st_global_f32(scale_addr, scale);
                ctx.label("skip_scale_store");

                ctx.label("exit");
                ctx.ret();
            })
    }
}
