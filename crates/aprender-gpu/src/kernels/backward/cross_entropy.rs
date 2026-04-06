//! Fused Cross-Entropy Loss + Softmax Backward Kernel (KAIZEN-050)
//!
//! Computes cross-entropy loss and softmax backward gradient in a single fused kernel,
//! eliminating the need to:
//! 1. Download logits from GPU (77.8MB for Qwen3-4B)
//! 2. Compute softmax on CPU (40ms for 19M elements)
//! 3. Re-upload gradient to GPU (77.8MB)
//!
//! ## Algorithm
//!
//! One block (256 threads) per sequence position. Each block processes one row
//! of `vocab_size` elements using grid-stride loops:
//!
//! 1. **Phase 1**: Find max(logits[pos]) via multi-warp reduction
//! 2. **Phase 2**: Compute sum(exp(logits[i] - max)) via multi-warp reduction
//! 3. **Phase 3**: Write gradient and loss:
//!    - `grad[i] = (exp(logits[i] - max) / sum - one_hot(target)) * scale`
//!    - `loss[pos] = -log(exp(logits[target] - max) / sum)`
//!
//! ## Transfer Budget
//!
//! - Input: logits (GPU-resident), targets (seq_len × u32 = 512 bytes for seq_len=128)
//! - Output: grad_logits (GPU-resident), loss_partials (seq_len × f32 = 512 bytes)
//! - Total D2H: ~512 bytes (vs 77.8MB before)

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused cross-entropy loss + softmax backward kernel.
///
/// Each block processes one sequence position (one row of vocab_size elements).
/// Writes gradient **in-place** to the logits buffer (KAIZEN-052) and per-position loss partials.
///
/// # Contract (C-XENT-001)
///
/// - **Precondition**: `seq_len > 0`, `vocab_size > 0`, targets in `[0, vocab_size)`
/// - **Postcondition**: `grad[pos][i] = (softmax(logits[pos])[i] - one_hot(target[pos])[i]) * scale`
/// - **Postcondition**: `loss[pos] = -log(softmax(logits[pos])[target[pos]])`
/// - **Invariant**: No logits D2H or gradient H2D — everything stays on GPU
/// - **Numerical**: Uses `exp2(x * log2(e))` for fast exp, `-log2(prob) / log2(e)` for log
#[derive(Debug, Clone)]
pub struct FusedCrossEntropyKernel {
    /// Vocabulary size (elements per row)
    pub vocab_size: u32,
}

impl FusedCrossEntropyKernel {
    /// Create a new fused cross-entropy kernel
    #[must_use]
    pub const fn new(vocab_size: u32) -> Self {
        Self { vocab_size }
    }

    /// Block size (threads per block) — one block per sequence position
    #[must_use]
    pub const fn block_size(&self) -> u32 {
        256
    }
}

impl Kernel for FusedCrossEntropyKernel {
    fn name(&self) -> &str {
        "fused_cross_entropy"
    }

    fn build_ptx(&self) -> PtxKernel {
        let block_size = 256_u32;
        let n_warps = block_size / 32;
        // Shared memory layout (same as LongRowSoftmaxKernel):
        // [0..32)   = 8 warp maxes (32 bytes)
        // [32..36)  = global max (4 bytes)
        // [36..68)  = 8 warp sums (32 bytes)
        // [68..72)  = global sum (4 bytes)
        let smem_size = (n_warps * 2 + 2) * 4;

        // KAIZEN-052: In-place operation — gradient written back to logits buffer.
        // Phase 3 reads each logit once and writes gradient to the same address.
        // Phases 1-2 complete all logit reads before phase 3 writes (barrier-protected).
        PtxKernel::new("fused_cross_entropy")
            .param(PtxType::U64, "logits_grad_ptr") // [seq_len, vocab_size] f32 — logits in, grad out
            .param(PtxType::U64, "targets_ptr") // [seq_len] u32
            .param(PtxType::U64, "loss_ptr") // [seq_len] f32 (output)
            .param(PtxType::U32, "vocab_size")
            .param(PtxType::F32, "scale") // 1.0 / seq_len (or 1/(seq_len * accum_steps))
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let pos = ctx.special_reg(PtxReg::CtaIdX); // block = position
                let ntid = ctx.special_reg(PtxReg::NtidX);

                let lane_mask = ctx.mov_u32_imm(31);
                let lane_id = ctx.and_u32(tid, lane_mask);
                let warp_id = ctx.shr_u32_imm(tid, 5);

                let vocab_size = ctx.load_param_u32("vocab_size");
                let logits_grad_ptr = ctx.load_param_u64("logits_grad_ptr");
                let targets_ptr = ctx.load_param_u64("targets_ptr");
                let loss_ptr = ctx.load_param_u64("loss_ptr");
                let scale = ctx.load_param_f32("scale");

                // Load target for this position: targets[pos]
                let target_byte_off = ctx.mul_wide_u32(pos, 4);
                let target_addr = ctx.add_u64(targets_ptr, target_byte_off);
                let target_id = ctx.ld_global_u32(target_addr);

                // Row base pointer: logits_grad_ptr + pos * vocab_size * 4
                // KAIZEN-052: Same buffer for logits input and gradient output (in-place).
                let row_elem_off = ctx.mul_lo_u32(pos, vocab_size);
                let row_byte_off = ctx.mul_wide_u32(row_elem_off, 4);
                let row_logits = ctx.add_u64(logits_grad_ptr, row_byte_off);

                let zero = ctx.mov_u32_imm(0);
                let is_lane_0 = ctx.setp_eq_u32(lane_id, zero);
                let is_warp_0 = ctx.setp_eq_u32(warp_id, zero);

                // =============================================================
                // Phase 1: Find max(logits) — grid-stride + multi-warp reduce
                // =============================================================
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let local_max = neg_inf;
                let idx = ctx.add_u32(tid, 0);

                ctx.label("max_loop");
                let done_max = ctx.setp_ge_u32(idx, vocab_size);
                ctx.branch_if(done_max, "max_done");

                let byte_off = ctx.mul_wide_u32(idx, 4);
                let addr = ctx.add_u64(row_logits, byte_off);
                let val = ctx.ld_global_f32(addr);
                ctx.max_f32_inplace(local_max, val);

                ctx.add_u32_reg_inplace(idx, ntid);
                ctx.branch("max_loop");
                ctx.label("max_done");

                // Warp-level max reduction
                let s16 = ctx.shfl_down_f32(local_max, 16, 0xFFFF_FFFF);
                let wm1 = ctx.max_f32(local_max, s16);
                let s8 = ctx.shfl_down_f32(wm1, 8, 0xFFFF_FFFF);
                let wm2 = ctx.max_f32(wm1, s8);
                let s4 = ctx.shfl_down_f32(wm2, 4, 0xFFFF_FFFF);
                let wm3 = ctx.max_f32(wm2, s4);
                let s2 = ctx.shfl_down_f32(wm3, 2, 0xFFFF_FFFF);
                let wm4 = ctx.max_f32(wm3, s2);
                let s1 = ctx.shfl_down_f32(wm4, 1, 0xFFFF_FFFF);
                let warp_max = ctx.max_f32(wm4, s1);

                // Lane 0 stores warp max to shared memory
                ctx.branch_if_not(is_lane_0, "skip_store_wmax");
                let smem_off = ctx.mul_u32(warp_id, 4);
                let smem_off_64 = ctx.cvt_u64_u32(smem_off);
                ctx.st_shared_f32(smem_off_64, warp_max);
                ctx.label("skip_store_wmax");

                ctx.bar_sync(0);

                // Warp 0 reduces across 8 warp maxes
                ctx.branch_if_not(is_warp_0, "skip_inter_max");
                let seven = ctx.mov_u32_imm(7);
                let clamped = ctx.and_u32(lane_id, seven);
                let lane_off = ctx.mul_u32(clamped, 4);
                let lane_off_64 = ctx.cvt_u64_u32(lane_off);
                let loaded_max = ctx.ld_shared_f32(lane_off_64);

                let im4 = ctx.shfl_down_f32(loaded_max, 4, 0xFFFF_FFFF);
                let im1 = ctx.max_f32(loaded_max, im4);
                let im2r = ctx.shfl_down_f32(im1, 2, 0xFFFF_FFFF);
                let im2 = ctx.max_f32(im1, im2r);
                let im1r = ctx.shfl_down_f32(im2, 1, 0xFFFF_FFFF);
                let global_max = ctx.max_f32(im2, im1r);

                let is_l0 = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_l0, "skip_store_gmax");
                let gmax_off = ctx.mov_u32_imm(32);
                let gmax_off_64 = ctx.cvt_u64_u32(gmax_off);
                ctx.st_shared_f32(gmax_off_64, global_max);
                ctx.label("skip_store_gmax");

                ctx.label("skip_inter_max");
                ctx.bar_sync(1);

                // All threads read global max
                let gmax_read = ctx.mov_u32_imm(32);
                let gmax_read_64 = ctx.cvt_u64_u32(gmax_read);
                let global_max_val = ctx.ld_shared_f32(gmax_read_64);

                // =============================================================
                // Phase 2: Compute sum(exp(x - max)) — grid-stride + multi-warp
                // =============================================================
                let local_sum = ctx.mov_f32_imm(0.0);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let idx2 = ctx.add_u32(tid, 0);

                ctx.label("sum_loop");
                let done_sum = ctx.setp_ge_u32(idx2, vocab_size);
                ctx.branch_if(done_sum, "sum_done");

                let byte_off2 = ctx.mul_wide_u32(idx2, 4);
                let addr2 = ctx.add_u64(row_logits, byte_off2);
                let val2 = ctx.ld_global_f32(addr2);
                let shifted = ctx.sub_f32(val2, global_max_val);
                let scaled_v = ctx.mul_f32(shifted, log2_e);
                let exp_val = ctx.ex2_f32(scaled_v);
                ctx.add_f32_inplace(local_sum, exp_val);

                ctx.add_u32_reg_inplace(idx2, ntid);
                ctx.branch("sum_loop");
                ctx.label("sum_done");

                // Warp-level sum reduction
                let ss16 = ctx.shfl_down_f32(local_sum, 16, 0xFFFF_FFFF);
                let ws1 = ctx.add_f32(local_sum, ss16);
                let ss8 = ctx.shfl_down_f32(ws1, 8, 0xFFFF_FFFF);
                let ws2 = ctx.add_f32(ws1, ss8);
                let ss4 = ctx.shfl_down_f32(ws2, 4, 0xFFFF_FFFF);
                let ws3 = ctx.add_f32(ws2, ss4);
                let ss2 = ctx.shfl_down_f32(ws3, 2, 0xFFFF_FFFF);
                let ws4 = ctx.add_f32(ws3, ss2);
                let ss1 = ctx.shfl_down_f32(ws4, 1, 0xFFFF_FFFF);
                let warp_sum = ctx.add_f32(ws4, ss1);

                // Lane 0 stores warp sum at shared[36..]
                ctx.branch_if_not(is_lane_0, "skip_store_wsum");
                let four = ctx.mov_u32_imm(4);
                let sum_base = ctx.mov_u32_imm(36);
                let sum_off = ctx.mad_lo_u32(warp_id, four, sum_base);
                let sum_off_64 = ctx.cvt_u64_u32(sum_off);
                ctx.st_shared_f32(sum_off_64, warp_sum);
                ctx.label("skip_store_wsum");

                ctx.bar_sync(2);

                // Warp 0 reduces across 8 warp sums
                ctx.branch_if_not(is_warp_0, "skip_inter_sum");
                let seven2 = ctx.mov_u32_imm(7);
                let clamped2 = ctx.and_u32(lane_id, seven2);
                let sum_base2 = ctx.mov_u32_imm(36);
                let four2 = ctx.mov_u32_imm(4);
                let sum_lane_off = ctx.mad_lo_u32(clamped2, four2, sum_base2);
                let sum_lane_64 = ctx.cvt_u64_u32(sum_lane_off);
                let loaded_sum = ctx.ld_shared_f32(sum_lane_64);

                let is4 = ctx.shfl_down_f32(loaded_sum, 4, 0xFFFF_FFFF);
                let is1v = ctx.add_f32(loaded_sum, is4);
                let is2r = ctx.shfl_down_f32(is1v, 2, 0xFFFF_FFFF);
                let is2v = ctx.add_f32(is1v, is2r);
                let is1r = ctx.shfl_down_f32(is2v, 1, 0xFFFF_FFFF);
                let global_sum = ctx.add_f32(is2v, is1r);

                let is_l0s = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_l0s, "skip_store_gsum");
                let gsum_off = ctx.mov_u32_imm(68);
                let gsum_off_64 = ctx.cvt_u64_u32(gsum_off);
                ctx.st_shared_f32(gsum_off_64, global_sum);
                ctx.label("skip_store_gsum");

                ctx.label("skip_inter_sum");
                ctx.bar_sync(3);

                // All threads read global sum
                let gsum_read = ctx.mov_u32_imm(68);
                let gsum_read_64 = ctx.cvt_u64_u32(gsum_read);
                let global_sum_val = ctx.ld_shared_f32(gsum_read_64);

                // =============================================================
                // Phase 3: Write gradient + compute loss
                // =============================================================
                // grad[i] = (exp(logits[i] - max) / sum - one_hot(target)) * scale
                // loss = -log(exp(logits[target] - max) / sum)
                //       = -(logits[target] - max - log(sum))
                //       = -logits[target] + max + log(sum)
                //
                // For log(sum): log(x) = log2(x) / log2(e)

                // Thread 0 computes and stores loss for this position
                let is_tid0 = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_tid0, "skip_loss");

                // loss = max + log(sum) - logits[target]
                // log(sum) = log2(sum) / log2(e) = log2(sum) * (1/log2(e)) = log2(sum) * ln(2)
                let log2_sum = ctx.lg2_f32(global_sum_val);
                let ln2 = ctx.mov_f32_imm(std::f32::consts::LN_2);
                let log_sum = ctx.mul_f32(log2_sum, ln2);

                // Load logits[target]
                let target_byte = ctx.mul_wide_u32(target_id, 4);
                let target_logit_addr = ctx.add_u64(row_logits, target_byte);
                let target_logit = ctx.ld_global_f32(target_logit_addr);

                // loss = max + log_sum - target_logit
                let loss_val = ctx.add_f32(global_max_val, log_sum);
                let loss_val = ctx.sub_f32(loss_val, target_logit);

                // Store loss[pos]
                let loss_byte_off = ctx.mul_wide_u32(pos, 4);
                let loss_addr = ctx.add_u64(loss_ptr, loss_byte_off);
                ctx.st_global_f32(loss_addr, loss_val);

                ctx.label("skip_loss");

                // KAIZEN-052: Barrier ensures thread 0 has read logits[target] for loss
                // before any thread overwrites logits with gradients (in-place).
                ctx.bar_sync(4);

                // All threads write gradient in-place via grid-stride loop.
                // Safe because: each thread processes disjoint indices, and phases 1-2
                // completed all reads before this point (barriers 0-3).
                let one_f32 = ctx.mov_f32_imm(1.0);
                let idx3 = ctx.add_u32(tid, 0);

                ctx.label("grad_loop");
                let done_grad = ctx.setp_ge_u32(idx3, vocab_size);
                ctx.branch_if(done_grad, "grad_done");

                let byte_off3 = ctx.mul_wide_u32(idx3, 4);
                let addr3 = ctx.add_u64(row_logits, byte_off3);
                let val3 = ctx.ld_global_f32(addr3);

                // softmax_val = exp(val - max) / sum
                let shifted3 = ctx.sub_f32(val3, global_max_val);
                let scaled3 = ctx.mul_f32(shifted3, log2_e);
                let exp3 = ctx.ex2_f32(scaled3);
                let softmax_val = ctx.div_f32(exp3, global_sum_val);

                // grad = (softmax - one_hot) * scale
                // one_hot is 1.0 at target_id, 0.0 elsewhere
                let is_target = ctx.setp_eq_u32(idx3, target_id);
                // Start with softmax_val * scale (for non-target elements)
                let grad_nontarget = ctx.mul_f32(softmax_val, scale);
                // For target: (softmax - 1.0) * scale
                let sm_minus_one = ctx.sub_f32(softmax_val, one_f32);
                let grad_target = ctx.mul_f32(sm_minus_one, scale);
                // Select based on predicate: is_target ? grad_target : grad_nontarget
                let grad_val = ctx.selp_f32(is_target, grad_target, grad_nontarget);

                // KAIZEN-052: Write gradient to same address as logits (in-place)
                ctx.st_global_f32(addr3, grad_val);

                ctx.add_u32_reg_inplace(idx3, ntid);
                ctx.branch("grad_loop");

                ctx.label("grad_done");
                ctx.ret();
            })
    }
}

/// Fused causal cross-entropy loss + softmax backward kernel (KAIZEN-064).
///
/// Extends `FusedCrossEntropyKernel` with `loss_start`/`loss_end` parameters
/// for causal language model training. Positions outside `[loss_start, loss_end)`
/// produce zero gradient and zero loss — prompt tokens are masked.
///
/// # Contract (C-CAUSALXENT-001)
///
/// - **Precondition**: `loss_start < loss_end <= seq_len`, `vocab_size > 0`
/// - **Postcondition**: For `loss_start <= pos < loss_end`:
///   `grad[pos][i] = (softmax(logits[pos])[i] - one_hot(target[pos])[i]) * scale`
/// - **Postcondition**: For `pos < loss_start || pos >= loss_end`:
///   `grad[pos][i] = 0.0`, `loss[pos] = 0.0`
/// - **Invariant**: No logits D2H or gradient H2D — everything stays on GPU
#[derive(Debug, Clone)]
pub struct FusedCausalCrossEntropyKernel {
    /// Vocabulary size (elements per row)
    pub vocab_size: u32,
}

impl FusedCausalCrossEntropyKernel {
    /// Create a new fused causal cross-entropy kernel
    #[must_use]
    pub const fn new(vocab_size: u32) -> Self {
        Self { vocab_size }
    }

    /// Block size (threads per block) — one block per sequence position
    #[must_use]
    pub const fn block_size(&self) -> u32 {
        256
    }
}

impl Kernel for FusedCausalCrossEntropyKernel {
    fn name(&self) -> &str {
        "fused_causal_cross_entropy"
    }

    fn build_ptx(&self) -> PtxKernel {
        let block_size = 256_u32;
        let n_warps = block_size / 32;
        let smem_size = (n_warps * 2 + 2) * 4;

        PtxKernel::new("fused_causal_cross_entropy")
            .param(PtxType::U64, "logits_grad_ptr")
            .param(PtxType::U64, "targets_ptr")
            .param(PtxType::U64, "loss_ptr")
            .param(PtxType::U32, "vocab_size")
            .param(PtxType::F32, "scale")
            .param(PtxType::U32, "loss_start") // KAIZEN-064: first loss position
            .param(PtxType::U32, "loss_end") // KAIZEN-064: one-past-last loss position
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let pos = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);

                let lane_mask = ctx.mov_u32_imm(31);
                let lane_id = ctx.and_u32(tid, lane_mask);
                let warp_id = ctx.shr_u32_imm(tid, 5);

                let vocab_size = ctx.load_param_u32("vocab_size");
                let logits_grad_ptr = ctx.load_param_u64("logits_grad_ptr");
                let targets_ptr = ctx.load_param_u64("targets_ptr");
                let loss_ptr = ctx.load_param_u64("loss_ptr");
                let scale = ctx.load_param_f32("scale");
                let loss_start = ctx.load_param_u32("loss_start");
                let loss_end = ctx.load_param_u32("loss_end");

                let zero = ctx.mov_u32_imm(0);
                let zero_f32 = ctx.mov_f32_imm(0.0);

                // Row base pointer: logits_grad_ptr + pos * vocab_size * 4
                let row_elem_off = ctx.mul_lo_u32(pos, vocab_size);
                let row_byte_off = ctx.mul_wide_u32(row_elem_off, 4);
                let row_logits = ctx.add_u64(logits_grad_ptr, row_byte_off);

                // =====================================================================
                // KAIZEN-064: Early exit for masked positions (prompt tokens).
                // Zero the gradient row and write loss=0.
                // Two-branch check: pos < loss_start OR pos >= loss_end → masked.
                // =====================================================================
                let is_in_range = ctx.setp_ge_u32(pos, loss_start);
                ctx.branch_if_not(is_in_range, "masked_position");
                let is_before_end = ctx.setp_lt_u32(pos, loss_end);
                ctx.branch_if(is_before_end, "loss_position");
                ctx.label("masked_position");

                // Masked position: zero gradient row via grid-stride loop
                let mask_idx = ctx.add_u32(tid, 0);
                ctx.label("mask_zero_loop");
                let done_mask = ctx.setp_ge_u32(mask_idx, vocab_size);
                ctx.branch_if(done_mask, "mask_zero_done");

                let mask_off = ctx.mul_wide_u32(mask_idx, 4);
                let mask_addr = ctx.add_u64(row_logits, mask_off);
                ctx.st_global_f32(mask_addr, zero_f32);

                ctx.add_u32_reg_inplace(mask_idx, ntid);
                ctx.branch("mask_zero_loop");
                ctx.label("mask_zero_done");

                // Write loss[pos] = 0.0 (thread 0 only)
                let is_tid0_mask = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_tid0_mask, "mask_exit");
                let loss_byte_off_mask = ctx.mul_wide_u32(pos, 4);
                let loss_addr_mask = ctx.add_u64(loss_ptr, loss_byte_off_mask);
                ctx.st_global_f32(loss_addr_mask, zero_f32);
                ctx.label("mask_exit");
                ctx.ret();

                // =====================================================================
                // Loss position: standard fused cross-entropy (identical to base kernel)
                // =====================================================================
                ctx.label("loss_position");

                let is_lane_0 = ctx.setp_eq_u32(lane_id, zero);
                let is_warp_0 = ctx.setp_eq_u32(warp_id, zero);

                // Load target for this position: targets[pos]
                let target_byte_off = ctx.mul_wide_u32(pos, 4);
                let target_addr = ctx.add_u64(targets_ptr, target_byte_off);
                let target_id = ctx.ld_global_u32(target_addr);

                // Phase 1: Find max(logits) — grid-stride + multi-warp reduce
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let local_max = neg_inf;
                let idx = ctx.add_u32(tid, 0);

                ctx.label("max_loop");
                let done_max = ctx.setp_ge_u32(idx, vocab_size);
                ctx.branch_if(done_max, "max_done");

                let byte_off = ctx.mul_wide_u32(idx, 4);
                let addr = ctx.add_u64(row_logits, byte_off);
                let val = ctx.ld_global_f32(addr);
                ctx.max_f32_inplace(local_max, val);

                ctx.add_u32_reg_inplace(idx, ntid);
                ctx.branch("max_loop");
                ctx.label("max_done");

                // Warp-level max reduction
                let s16 = ctx.shfl_down_f32(local_max, 16, 0xFFFF_FFFF);
                let wm1 = ctx.max_f32(local_max, s16);
                let s8 = ctx.shfl_down_f32(wm1, 8, 0xFFFF_FFFF);
                let wm2 = ctx.max_f32(wm1, s8);
                let s4 = ctx.shfl_down_f32(wm2, 4, 0xFFFF_FFFF);
                let wm3 = ctx.max_f32(wm2, s4);
                let s2 = ctx.shfl_down_f32(wm3, 2, 0xFFFF_FFFF);
                let wm4 = ctx.max_f32(wm3, s2);
                let s1 = ctx.shfl_down_f32(wm4, 1, 0xFFFF_FFFF);
                let warp_max = ctx.max_f32(wm4, s1);

                ctx.branch_if_not(is_lane_0, "skip_store_wmax");
                let smem_off = ctx.mul_u32(warp_id, 4);
                let smem_off_64 = ctx.cvt_u64_u32(smem_off);
                ctx.st_shared_f32(smem_off_64, warp_max);
                ctx.label("skip_store_wmax");

                ctx.bar_sync(0);

                ctx.branch_if_not(is_warp_0, "skip_inter_max");
                let seven = ctx.mov_u32_imm(7);
                let clamped = ctx.and_u32(lane_id, seven);
                let lane_off = ctx.mul_u32(clamped, 4);
                let lane_off_64 = ctx.cvt_u64_u32(lane_off);
                let loaded_max = ctx.ld_shared_f32(lane_off_64);

                let im4 = ctx.shfl_down_f32(loaded_max, 4, 0xFFFF_FFFF);
                let im1 = ctx.max_f32(loaded_max, im4);
                let im2r = ctx.shfl_down_f32(im1, 2, 0xFFFF_FFFF);
                let im2 = ctx.max_f32(im1, im2r);
                let im1r = ctx.shfl_down_f32(im2, 1, 0xFFFF_FFFF);
                let global_max = ctx.max_f32(im2, im1r);

                let is_l0 = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_l0, "skip_store_gmax");
                let gmax_off = ctx.mov_u32_imm(32);
                let gmax_off_64 = ctx.cvt_u64_u32(gmax_off);
                ctx.st_shared_f32(gmax_off_64, global_max);
                ctx.label("skip_store_gmax");

                ctx.label("skip_inter_max");
                ctx.bar_sync(1);

                let gmax_read = ctx.mov_u32_imm(32);
                let gmax_read_64 = ctx.cvt_u64_u32(gmax_read);
                let global_max_val = ctx.ld_shared_f32(gmax_read_64);

                // Phase 2: Compute sum(exp(x - max))
                let local_sum = ctx.mov_f32_imm(0.0);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let idx2 = ctx.add_u32(tid, 0);

                ctx.label("sum_loop");
                let done_sum = ctx.setp_ge_u32(idx2, vocab_size);
                ctx.branch_if(done_sum, "sum_done");

                let byte_off2 = ctx.mul_wide_u32(idx2, 4);
                let addr2 = ctx.add_u64(row_logits, byte_off2);
                let val2 = ctx.ld_global_f32(addr2);
                let shifted = ctx.sub_f32(val2, global_max_val);
                let scaled_v = ctx.mul_f32(shifted, log2_e);
                let exp_val = ctx.ex2_f32(scaled_v);
                ctx.add_f32_inplace(local_sum, exp_val);

                ctx.add_u32_reg_inplace(idx2, ntid);
                ctx.branch("sum_loop");
                ctx.label("sum_done");

                // Warp-level sum reduction
                let ss16 = ctx.shfl_down_f32(local_sum, 16, 0xFFFF_FFFF);
                let ws1 = ctx.add_f32(local_sum, ss16);
                let ss8 = ctx.shfl_down_f32(ws1, 8, 0xFFFF_FFFF);
                let ws2 = ctx.add_f32(ws1, ss8);
                let ss4 = ctx.shfl_down_f32(ws2, 4, 0xFFFF_FFFF);
                let ws3 = ctx.add_f32(ws2, ss4);
                let ss2 = ctx.shfl_down_f32(ws3, 2, 0xFFFF_FFFF);
                let ws4 = ctx.add_f32(ws3, ss2);
                let ss1 = ctx.shfl_down_f32(ws4, 1, 0xFFFF_FFFF);
                let warp_sum = ctx.add_f32(ws4, ss1);

                ctx.branch_if_not(is_lane_0, "skip_store_wsum");
                let four = ctx.mov_u32_imm(4);
                let sum_base = ctx.mov_u32_imm(36);
                let sum_off = ctx.mad_lo_u32(warp_id, four, sum_base);
                let sum_off_64 = ctx.cvt_u64_u32(sum_off);
                ctx.st_shared_f32(sum_off_64, warp_sum);
                ctx.label("skip_store_wsum");

                ctx.bar_sync(2);

                ctx.branch_if_not(is_warp_0, "skip_inter_sum");
                let seven2 = ctx.mov_u32_imm(7);
                let clamped2 = ctx.and_u32(lane_id, seven2);
                let sum_base2 = ctx.mov_u32_imm(36);
                let four2 = ctx.mov_u32_imm(4);
                let sum_lane_off = ctx.mad_lo_u32(clamped2, four2, sum_base2);
                let sum_lane_64 = ctx.cvt_u64_u32(sum_lane_off);
                let loaded_sum = ctx.ld_shared_f32(sum_lane_64);

                let is4 = ctx.shfl_down_f32(loaded_sum, 4, 0xFFFF_FFFF);
                let is1v = ctx.add_f32(loaded_sum, is4);
                let is2r = ctx.shfl_down_f32(is1v, 2, 0xFFFF_FFFF);
                let is2v = ctx.add_f32(is1v, is2r);
                let is1r = ctx.shfl_down_f32(is2v, 1, 0xFFFF_FFFF);
                let global_sum = ctx.add_f32(is2v, is1r);

                let is_l0s = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_l0s, "skip_store_gsum");
                let gsum_off = ctx.mov_u32_imm(68);
                let gsum_off_64 = ctx.cvt_u64_u32(gsum_off);
                ctx.st_shared_f32(gsum_off_64, global_sum);
                ctx.label("skip_store_gsum");

                ctx.label("skip_inter_sum");
                ctx.bar_sync(3);

                let gsum_read = ctx.mov_u32_imm(68);
                let gsum_read_64 = ctx.cvt_u64_u32(gsum_read);
                let global_sum_val = ctx.ld_shared_f32(gsum_read_64);

                // Phase 3: Write gradient + compute loss
                let is_tid0 = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_tid0, "skip_loss");

                let log2_sum = ctx.lg2_f32(global_sum_val);
                let ln2 = ctx.mov_f32_imm(std::f32::consts::LN_2);
                let log_sum = ctx.mul_f32(log2_sum, ln2);

                let target_byte = ctx.mul_wide_u32(target_id, 4);
                let target_logit_addr = ctx.add_u64(row_logits, target_byte);
                let target_logit = ctx.ld_global_f32(target_logit_addr);

                let loss_val = ctx.add_f32(global_max_val, log_sum);
                let loss_val = ctx.sub_f32(loss_val, target_logit);

                let loss_byte_off = ctx.mul_wide_u32(pos, 4);
                let loss_addr = ctx.add_u64(loss_ptr, loss_byte_off);
                ctx.st_global_f32(loss_addr, loss_val);

                ctx.label("skip_loss");

                ctx.bar_sync(4);

                // Write gradient in-place via grid-stride loop
                let one_f32 = ctx.mov_f32_imm(1.0);
                let idx3 = ctx.add_u32(tid, 0);

                ctx.label("grad_loop");
                let done_grad = ctx.setp_ge_u32(idx3, vocab_size);
                ctx.branch_if(done_grad, "grad_done");

                let byte_off3 = ctx.mul_wide_u32(idx3, 4);
                let addr3 = ctx.add_u64(row_logits, byte_off3);
                let val3 = ctx.ld_global_f32(addr3);

                let shifted3 = ctx.sub_f32(val3, global_max_val);
                let scaled3 = ctx.mul_f32(shifted3, log2_e);
                let exp3 = ctx.ex2_f32(scaled3);
                let softmax_val = ctx.div_f32(exp3, global_sum_val);

                let is_target = ctx.setp_eq_u32(idx3, target_id);
                let grad_nontarget = ctx.mul_f32(softmax_val, scale);
                let sm_minus_one = ctx.sub_f32(softmax_val, one_f32);
                let grad_target = ctx.mul_f32(sm_minus_one, scale);
                // Select based on predicate: is_target ? grad_target : grad_nontarget
                let grad_val = ctx.selp_f32(is_target, grad_target, grad_nontarget);

                ctx.st_global_f32(addr3, grad_val);

                ctx.add_u32_reg_inplace(idx3, ntid);
                ctx.branch("grad_loop");

                ctx.label("grad_done");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_cross_entropy_name() {
        let kernel = FusedCrossEntropyKernel::new(32000);
        assert_eq!(kernel.name(), "fused_cross_entropy");
    }

    #[test]
    fn test_fused_cross_entropy_block_size() {
        let kernel = FusedCrossEntropyKernel::new(32000);
        assert_eq!(kernel.block_size(), 256);
    }

    #[test]
    fn test_fused_cross_entropy_ptx_generation() {
        let kernel = FusedCrossEntropyKernel::new(32000);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry fused_cross_entropy"));
        assert!(ptx.contains(".param .u64 logits_grad_ptr"));
        assert!(ptx.contains(".param .u64 targets_ptr"));
        assert!(ptx.contains(".param .u64 loss_ptr"));
        assert!(ptx.contains(".param .u32 vocab_size"));
        assert!(ptx.contains(".param .f32 scale"));
        // KAIZEN-052: No separate grad_ptr — in-place operation
        assert!(!ptx.contains(".param .u64 grad_ptr"));
        // Verify warp shuffle
        assert!(ptx.contains("shfl.sync.down"));
        // Verify shared memory
        assert!(ptx.contains(".shared"));
        // Verify exp2 (fast exp)
        assert!(ptx.contains("ex2.approx.f32"));
        // Verify log2 (for loss computation)
        assert!(ptx.contains("lg2.approx.f32"));
        // Verify selp (predicated select for one_hot)
        assert!(ptx.contains("selp.f32"));
    }

    #[test]
    fn test_fused_cross_entropy_barrier_safety() {
        let kernel = FusedCrossEntropyKernel::new(32000);
        let _ptx = kernel.emit_ptx_validated();
    }

    #[test]
    fn test_fused_cross_entropy_small_vocab() {
        // Edge case: tiny vocab
        let kernel = FusedCrossEntropyKernel::new(2);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry fused_cross_entropy"));
    }

    #[test]
    fn test_fused_cross_entropy_large_vocab() {
        // Qwen3-4B vocab size
        let kernel = FusedCrossEntropyKernel::new(151936);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry fused_cross_entropy"));
    }

    // --- FusedCausalCrossEntropyKernel tests (KAIZEN-064) ---

    #[test]
    fn test_fused_causal_cross_entropy_name() {
        let kernel = FusedCausalCrossEntropyKernel::new(32000);
        assert_eq!(kernel.name(), "fused_causal_cross_entropy");
    }

    #[test]
    fn test_fused_causal_cross_entropy_block_size() {
        let kernel = FusedCausalCrossEntropyKernel::new(32000);
        assert_eq!(kernel.block_size(), 256);
    }

    #[test]
    fn test_fused_causal_cross_entropy_ptx_generation() {
        let kernel = FusedCausalCrossEntropyKernel::new(32000);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry fused_causal_cross_entropy"));
        assert!(ptx.contains(".param .u64 logits_grad_ptr"));
        assert!(ptx.contains(".param .u64 targets_ptr"));
        assert!(ptx.contains(".param .u64 loss_ptr"));
        assert!(ptx.contains(".param .u32 vocab_size"));
        assert!(ptx.contains(".param .f32 scale"));
        // KAIZEN-064: causal masking params
        assert!(ptx.contains(".param .u32 loss_start"));
        assert!(ptx.contains(".param .u32 loss_end"));
        // Verify core operations
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("lg2.approx.f32"));
        assert!(ptx.contains("selp.f32"));
    }

    #[test]
    fn test_fused_causal_cross_entropy_barrier_safety() {
        let kernel = FusedCausalCrossEntropyKernel::new(32000);
        let _ptx = kernel.emit_ptx_validated();
    }

    #[test]
    fn test_fused_causal_cross_entropy_large_vocab() {
        let kernel = FusedCausalCrossEntropyKernel::new(151936);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry fused_causal_cross_entropy"));
    }

    #[test]
    fn test_fused_causal_cross_entropy_small_vocab() {
        let kernel = FusedCausalCrossEntropyKernel::new(2);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry fused_causal_cross_entropy"));
    }
}
