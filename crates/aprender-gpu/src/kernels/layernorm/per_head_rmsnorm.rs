//! GH-280: Per-Head RMSNorm kernel for QK normalization (Qwen3)
//!
//! Applies RMSNorm independently to each attention head:
//!
//! ```text
//! For each head h in 0..num_heads:
//!     slice = input[h*head_dim .. (h+1)*head_dim]
//!     rms = sqrt(mean(slice^2) + eps)
//!     output[h*head_dim..(h+1)*head_dim] = slice / rms * gamma
//! ```
//!
//! Gamma weights have shape `[head_dim]` and are shared across all heads.
//! Grid: (num_heads, 1, 1), Block: (32, 1, 1) — one warp per head.
//!
//! #3413 B: the batched variant (`with_batch(m)`, m > 1) normalizes `m` packed
//! sequences in one launch. Grid: (num_heads, m, 1); the head base offset becomes
//! `(seq_idx * num_heads + head_idx) * head_dim` with `seq_idx = blockIdx.y`.
//! Gamma is still indexed by lane within `head_dim` — it has no batch stride.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Per-head RMSNorm kernel for QK normalization (Qwen3).
///
/// Each CUDA block (one warp = 32 threads) processes one attention head.
/// `blockIdx.x` selects the head, threads stride over `head_dim` elements.
///
/// For Qwen3-8B: head_dim=128, num_heads=32 (Q) or 8 (K), eps=1e-6.
#[derive(Debug, Clone)]
pub struct PerHeadRmsNormKernel {
    /// Elements per head (128 for Qwen3)
    pub head_dim: u32,
    /// Number of heads (32 for Q, 8 for K)
    pub num_heads: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// #3413 B: number of packed sequences (1 = single-sequence decode kernel).
    /// `> 1` selects the batched prefill variant indexed by `blockIdx.y`.
    pub batch: u32,
}

impl PerHeadRmsNormKernel {
    /// Create a new per-head RMSNorm kernel (single sequence)
    #[must_use]
    pub fn new(head_dim: u32, num_heads: u32) -> Self {
        Self {
            head_dim,
            num_heads,
            epsilon: 1e-6,
            batch: 1,
        }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// #3413 B: select the batched variant for `batch` packed sequences.
    ///
    /// `batch == 1` keeps the single-sequence kernel (byte-identical PTX).
    #[must_use]
    pub const fn with_batch(mut self, batch: u32) -> Self {
        self.batch = batch;
        self
    }

    /// True when the batched (grid.y indexed) variant is selected.
    #[must_use]
    pub const fn is_batched(&self) -> bool {
        self.batch > 1
    }
}

impl Kernel for PerHeadRmsNormKernel {
    fn name(&self) -> &str {
        if self.is_batched() {
            "batched_per_head_rmsnorm"
        } else {
            "per_head_rmsnorm"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let epsilon = self.epsilon;
        let batched = self.is_batched();

        // Per-head RMSNorm using warp shuffle (same pattern as RmsNormKernel)
        // Grid: (num_heads, 1, 1) — one block per head
        //   batched: (num_heads, batch, 1) — one block per (sequence, head)
        // Block: (32, 1, 1) — one warp
        // Each thread handles head_dim/32 elements within its head
        PtxKernel::new(self.name())
            .param(PtxType::U64, "input_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "output_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "gamma_ptr") // [head_dim] shared across heads
            .shared_memory(0) // Warp shuffle, no shared memory
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);

                // Load parameters
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);

                // #3413 B: batched variant folds the sequence index into the row:
                // row = seq_idx * num_heads + head_idx, seq_idx = blockIdx.y.
                // Single-sequence variant emits nothing here, so its PTX is unchanged.
                let row_idx = if batched {
                    let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                    let seq_idx = ctx.special_reg(PtxReg::CtaIdY);
                    ctx.mad_lo_u32(seq_idx, num_heads_u32, head_idx)
                } else {
                    head_idx
                };

                let four = ctx.mov_u32_imm(4);

                // Compute base offset for this row: row_idx * head_dim * 4 bytes
                let head_elem_offset = ctx.mul_u32_reg(row_idx, head_dim_u32);
                let head_byte_offset = ctx.mul_wide_u32_reg(head_elem_offset, four);
                let head_input_base = ctx.add_u64(input_ptr, head_byte_offset);
                let head_output_base = ctx.add_u64(output_ptr, head_byte_offset);

                // Pass 1: Accumulate sum of squares within this head
                // Each thread processes elements: tid, tid+32, tid+64, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, head_dim_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load input[head_offset + idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(head_input_base, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                // sq_sum += val * val
                ctx.fma_f32_inplace(sq_sum, val, val);

                // idx += 32 (stride by warp size)
                ctx.add_u32_inplace(idx, 32);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp reduce sq_sum
                let shfl16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl16);
                let shfl8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl8);
                let shfl4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl4);
                let shfl2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl2);
                let shfl1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl1);

                // Broadcast final sum to all threads
                let total_sq_sum = ctx.shfl_idx_f32(sq_sum, 0, 0xFFFF_FFFF);

                // Compute RMS = sqrt(mean(x^2) + epsilon) over head_dim
                let head_dim_f32 = ctx.cvt_f32_u32(head_dim_u32);
                let mean_sq = ctx.div_f32(total_sq_sum, head_dim_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Pass 2: Normalize and scale
                // output[head_offset+i] = input[head_offset+i] * rms_inv * gamma[i]
                // Note: gamma is indexed by position within head (no head offset)
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, head_dim_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(head_input_base, elem_offset2);
                // gamma is [head_dim], shared across heads — no head offset
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(head_output_base, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = input * rms_inv * gamma
                let normalized = ctx.mul_f32(inp, rms_inv);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod batched_ptx_tests {
    use super::*;

    /// Byte-exact snapshot of the single-sequence PTX, captured from
    /// `PerHeadRmsNormKernel::new(64, 4).with_epsilon(1e-6)` BEFORE the batched
    /// variant existed (#3413 B). The decode path still launches the m=1 kernel
    /// (it was just fixed for #3413 A), so the batched grid must not perturb it.
    const M1_PTX_SNAPSHOT: &str = r#"// Generated by trueno-gpu
// Pure Rust PTX generation - no external dependencies

.version 8.0
.target sm_70
.address_size 64

.visible .entry per_head_rmsnorm(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 gamma_ptr
) {
    .reg .f32  %f<17>;
    .reg .pred  %p<2>;
    .reg .u32  %r<9>;
    .reg .u64  %rd<12>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    ld.param.u64 %rd0, [input_ptr];
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [gamma_ptr];
    mov.u32 %r2, 64;
    mov.u32 %r3, 4;
    mul.lo.u32 %r4, %r1, %r2;
    mul.wide.u32 %rd3, %r4, %r3;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    mov.f32 %f0, 0F00000000;
    mov.u32 %r5, 0;
sum_loop:
    add.u32 %r6, %r5, %r0;
    setp.lt.u32 %p0, %r6, %r2;
    @!%p0 bra sum_loop_end;
    mul.wide.u32 %rd6, %r6, %r3;
    add.u64 %rd7, %rd4, %rd6;
    ld.global.f32 %f1, [%rd7];
    fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r5, %r5, 32;
    bra sum_loop;
sum_loop_end:
    shfl.sync.down.b32 %f2, %f0, 16, 31, 4294967295;
    add.f32 %f0, %f0, %f2;
    shfl.sync.down.b32 %f3, %f0, 8, 31, 4294967295;
    add.f32 %f0, %f0, %f3;
    shfl.sync.down.b32 %f4, %f0, 4, 31, 4294967295;
    add.f32 %f0, %f0, %f4;
    shfl.sync.down.b32 %f5, %f0, 2, 31, 4294967295;
    add.f32 %f0, %f0, %f5;
    shfl.sync.down.b32 %f6, %f0, 1, 31, 4294967295;
    add.f32 %f0, %f0, %f6;
    shfl.sync.idx.b32 %f7, %f0, 0, 31, 4294967295;
    cvt.rn.f32.u32 %f8, %r2;
    div.rn.f32 %f9, %f7, %f8;
    mov.f32 %f10, 0F358637BD;
    add.f32 %f11, %f9, %f10;
    rsqrt.approx.f32 %f12, %f11;
    mov.u32 %r7, 0;
norm_loop:
    add.u32 %r8, %r7, %r0;
    setp.lt.u32 %p1, %r8, %r2;
    @!%p1 bra exit;
    mul.wide.u32 %rd8, %r8, %r3;
    add.u64 %rd9, %rd4, %rd8;
    add.u64 %rd10, %rd2, %rd8;
    add.u64 %rd11, %rd5, %rd8;
    ld.global.f32 %f13, [%rd9];
    ld.global.f32 %f14, [%rd10];
    mul.f32 %f15, %f13, %f12;
    mul.f32 %f16, %f15, %f14;
    st.global.f32 [%rd11], %f16;
    add.u32 %r7, %r7, 32;
    bra norm_loop;
exit:
    ret;
}

"#;

    #[test]
    fn test_per_head_rmsnorm_m1_ptx_byte_identical() {
        let kernel = PerHeadRmsNormKernel::new(64, 4).with_epsilon(1e-6);
        assert_eq!(
            kernel.emit_ptx(),
            M1_PTX_SNAPSHOT,
            "single-sequence PTX changed; the decode path depends on it byte-for-byte"
        );
        assert_eq!(kernel.name(), "per_head_rmsnorm");
    }

    #[test]
    fn test_per_head_rmsnorm_m1_ptx_has_no_ctaid_y() {
        let ptx = PerHeadRmsNormKernel::new(64, 4)
            .with_epsilon(1e-6)
            .emit_ptx();
        assert!(
            !ptx.contains("ctaid.y"),
            "m=1 kernel must stay one-block-per-head (grid.y unused)"
        );
    }

    #[test]
    fn test_batched_per_head_rmsnorm_ptx_uses_ctaid_y() {
        let kernel = PerHeadRmsNormKernel::new(64, 7)
            .with_epsilon(1e-6)
            .with_batch(3);
        let ptx = kernel.emit_ptx();

        assert_eq!(kernel.name(), "batched_per_head_rmsnorm");
        assert!(
            ptx.contains(".entry batched_per_head_rmsnorm"),
            "batched variant needs its own entry point"
        );
        assert!(
            ptx.contains("%ctaid.y"),
            "batched variant must read the sequence index from grid.y"
        );
        assert!(
            ptx.contains("%ctaid.x"),
            "batched variant still reads the head index from grid.x"
        );
        // row = seq_idx * num_heads + head_idx — num_heads must appear as an immediate
        assert!(
            ptx.contains(", 7;"),
            "batched offset must fold num_heads (7) into the row stride"
        );
    }

    #[test]
    fn test_batched_per_head_rmsnorm_keeps_single_gamma_row() {
        // gamma is [head_dim], shared across heads AND sequences: the batched PTX
        // must issue exactly the same global loads/stores as the m=1 PTX.
        let batched = PerHeadRmsNormKernel::new(64, 7)
            .with_epsilon(1e-6)
            .with_batch(3)
            .emit_ptx();
        let single = PerHeadRmsNormKernel::new(64, 7)
            .with_epsilon(1e-6)
            .emit_ptx();
        assert_eq!(
            batched.matches("ld.global.f32").count(),
            single.matches("ld.global.f32").count(),
            "batched variant must not add a per-sequence gamma load"
        );
        assert_eq!(
            batched.matches("st.global.f32").count(),
            single.matches("st.global.f32").count()
        );
    }

    #[test]
    fn test_batched_per_head_rmsnorm_defaults() {
        let kernel = PerHeadRmsNormKernel::new(128, 32);
        assert_eq!(kernel.batch, 1, "default is the single-sequence kernel");
        let batched = kernel.with_batch(5);
        assert_eq!(batched.batch, 5);
    }
}
