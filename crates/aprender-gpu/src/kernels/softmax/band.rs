//! Band-masked row softmax for bidirectional, optionally windowed, attention scores.
//!
//! APR-EMBED-001 EG-4 (#4903): an encoder attends in both directions, and its local
//! layers see only the keys within `half_window` positions of the query. The causal
//! mask kernel masks only the keys after the query, so it cannot express either.

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// `half_window` for a global layer: every key of a pass is inside the band.
pub const BAND_UNBOUNDED: u32 = 0x7FFF_FFFF;

/// In-place softmax of each score row over the keys inside its band.
///
/// Scores are `[heads][m][total]`, row-major: row `t` of head `h` is query `base + t`,
/// relative to the first key of the pass. Key `j` is inside the band when
/// `|base + t - j| <= half_window`; a key outside it gets probability exactly `0`
/// (its score becomes `-inf` before the exponential). The query's own key is always
/// inside, so no row is empty.
///
/// Grid `(heads, m, 1)`, block `(32, 1, 1)`: one warp per row. Every shape value is a
/// launch parameter, so one module serves every pass.
///
/// Params: `scores`, `m`, `total`, `base`, `half_window` ([`BAND_UNBOUNDED`] for none).
#[derive(Debug, Clone, Copy, Default)]
pub struct BandSoftmaxKernel;

impl Kernel for BandSoftmaxKernel {
    fn name(&self) -> &str {
        "band_softmax"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("band_softmax")
            .param(PtxType::U64, "scores")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "total")
            .param(PtxType::U32, "base")
            .param(PtxType::U32, "half_window")
            .build(|ctx| {
                let head = ctx.special_reg(PtxReg::CtaIdX);
                let t = ctx.special_reg(PtxReg::CtaIdY);
                let tid = ctx.special_reg(PtxReg::TidX);

                let scores = ctx.load_param_u64("scores");
                let m = ctx.load_param_u32("m");
                let total = ctx.load_param_u32("total");
                let base = ctx.load_param_u32("base");
                let w = ctx.load_param_u32("half_window");

                // Row (head * m + t), `total` floats wide.
                let row = ctx.mad_lo_u32(head, m, t);
                let four = ctx.mov_u32_imm(4);
                let row_bytes = ctx.mul_lo_u32(total, four);
                let row_off = ctx.mul_wide_u32_reg(row, row_bytes);
                let row_ptr = ctx.add_u64(scores, row_off);

                // Key j is inside when j + w >= pos and j < pos + w + 1.
                let pos = ctx.add_u32_reg(base, t);
                let pos_w = ctx.add_u32_reg(pos, w);
                let hi = ctx.add_u32(pos_w, 1);
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                // Pass 1: the row max over the band.
                let local_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let j = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(j, tid);
                ctx.label("max_loop");
                let done = ctx.setp_ge_u32(j, total);
                ctx.branch_if(done, "max_done");
                let off = ctx.mul_wide_u32_reg(j, four);
                let addr = ctx.add_u64(row_ptr, off);
                let raw = ctx.ld_global_f32(addr);
                let jw = ctx.add_u32_reg(j, w);
                let lo_ok = ctx.setp_ge_u32(jw, pos);
                let hi_ok = ctx.setp_lt_u32(j, hi);
                let inside = ctx.and_pred(lo_ok, hi_ok);
                let val = ctx.selp_f32(inside, raw, neg_inf);
                ctx.max_f32_inplace(local_max, val);
                ctx.add_u32_inplace(j, 32);
                ctx.branch("max_loop");
                ctx.label("max_done");
                for lane in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(local_max, lane, 0xFFFF_FFFF);
                    ctx.max_f32_inplace(local_max, other);
                }
                let row_max = ctx.shfl_idx_f32(local_max, 0, 0xFFFF_FFFF);

                // Pass 2: write exp(x - max) (0 outside the band) and sum it.
                let local_sum = ctx.mov_f32_imm(0.0);
                let j2 = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(j2, tid);
                ctx.label("exp_loop");
                let done2 = ctx.setp_ge_u32(j2, total);
                ctx.branch_if(done2, "exp_done");
                let off2 = ctx.mul_wide_u32_reg(j2, four);
                let addr2 = ctx.add_u64(row_ptr, off2);
                let raw2 = ctx.ld_global_f32(addr2);
                let jw2 = ctx.add_u32_reg(j2, w);
                let lo_ok2 = ctx.setp_ge_u32(jw2, pos);
                let hi_ok2 = ctx.setp_lt_u32(j2, hi);
                let inside2 = ctx.and_pred(lo_ok2, hi_ok2);
                let val2 = ctx.selp_f32(inside2, raw2, neg_inf);
                let diff = ctx.sub_f32(val2, row_max);
                let arg = ctx.mul_f32(diff, log2e);
                let e = ctx.ex2_f32(arg);
                ctx.st_global_f32(addr2, e);
                ctx.add_f32_inplace(local_sum, e);
                ctx.add_u32_inplace(j2, 32);
                ctx.branch("exp_loop");
                ctx.label("exp_done");
                for lane in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(local_sum, lane, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(local_sum, other);
                }
                let row_sum = ctx.shfl_idx_f32(local_sum, 0, 0xFFFF_FFFF);

                // Pass 3: normalize.
                let j3 = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(j3, tid);
                ctx.label("norm_loop");
                let done3 = ctx.setp_ge_u32(j3, total);
                ctx.branch_if(done3, "exit");
                let off3 = ctx.mul_wide_u32_reg(j3, four);
                let addr3 = ctx.add_u64(row_ptr, off3);
                let e3 = ctx.ld_global_f32(addr3);
                let p = ctx.div_f32(e3, row_sum);
                ctx.st_global_f32(addr3, p);
                ctx.add_u32_inplace(j3, 32);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}
