//! PMAT-3596: row-batched twins of three per-token Gated `DeltaNet` / attention kernels.
//!
//! A prefill chunk holds `T` tokens as `T` rows. Most per-token kernels batch as they
//! are, because their heads are contiguous and a chunk is just `T * heads` heads
//! (gated RMSNorm, the sigmoid gate, the q|gate split, per-head RMSNorm). These three
//! cannot:
//!
//! | kernel | why the per-token one cannot serve a chunk |
//! |--------|--------------------------------------------|
//! | [`PerHeadL2NormRowsKernel`] | `q` and `k` sit INSIDE each `[conv_dim]` conv-output row, `conv_dim` apart — not contiguous |
//! | [`GdnGatesRowsKernel`] | `dt_bias` and `a` are per head and must be re-read for every row, not indexed by the row |
//! | [`PartialNeoxRopeRowsKernel`] | every row has its OWN position (`pos0 + row`) |
//!
//! Each is its per-token twin with the row index added to the addressing (and, for
//! RoPE, to the position) and nothing else: the arithmetic is the twin's, op for op,
//! so a launch over `T` rows equals `T` per-token launches exactly. The device tests
//! assert equality, not a tolerance.

use crate::kernels::gdn::partial_rope::emit_sin_cos;
use crate::kernels::gdn::{emit_sigmoid_f32, emit_softplus_f32, ELEMENTWISE_BLOCK};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Per-head L2 normalisation of `num_heads` heads in each of `rows` strided rows.
///
/// Grid `(num_heads, rows, 1)`, block `(32, 1, 1)` — the twin's one warp per head.
#[derive(Debug, Clone, Copy)]
pub struct PerHeadL2NormRowsKernel {
    /// Elements per head (`head_k_dim`).
    pub head_dim: u32,
    /// Heads per row (`num_k_heads`).
    pub num_heads: u32,
    /// Epsilon, added to the sum of squares.
    pub eps: f32,
    /// Floats between row `t` and row `t + 1`.
    pub row_stride: u32,
}

impl PerHeadL2NormRowsKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(head_dim: u32, num_heads: u32, eps: f32, row_stride: u32) -> Self {
        Self {
            head_dim,
            num_heads,
            eps,
            row_stride,
        }
    }

    /// Launch grid for `rows` rows.
    #[must_use]
    pub const fn grid(&self, rows: u32) -> (u32, u32, u32) {
        (self.num_heads, rows, 1)
    }

    /// Launch block — one warp per head.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32, 1, 1)
    }
}

impl Kernel for PerHeadL2NormRowsKernel {
    fn name(&self) -> &str {
        "gdn_per_head_l2_norm_rows"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let eps = self.eps;
        let row_stride = self.row_stride;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "x_ptr") // [rows][row_stride], heads normalised in place
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let row = ctx.special_reg(PtxReg::CtaIdY);
                let x_ptr = ctx.load_param_u64("x_ptr");

                let head_dim_r = ctx.mov_u32_imm(head_dim);
                let four = ctx.mov_u32_imm(4);
                let row_elems = ctx.mul_u32(row, row_stride);
                let head_in_row = ctx.mul_u32_reg(head_idx, head_dim_r);
                let head_elems = ctx.add_u32_reg(row_elems, head_in_row);
                let head_bytes = ctx.mul_wide_u32_reg(head_elems, four);
                let head_base = ctx.add_u64(x_ptr, head_bytes);

                // From here on: PerHeadL2NormKernel's body, verbatim.
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);
                ctx.label("gdn_l2r_sum_loop");
                let i = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(i, head_dim_r);
                ctx.branch_if_not(in_bounds, "gdn_l2r_sum_end");
                let off = ctx.mul_wide_u32_reg(i, four);
                let addr = ctx.add_u64(head_base, off);
                let val = ctx.ld_global_f32(addr);
                ctx.fma_f32_inplace(sq_sum, val, val);
                ctx.add_u32_inplace(idx, 32);
                ctx.branch("gdn_l2r_sum_loop");
                ctx.label("gdn_l2r_sum_end");

                let s16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s16);
                let s8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s8);
                let s4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s4);
                let s2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s2);
                let s1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s1);
                let total = ctx.shfl_idx_f32(sq_sum, 0, 0xFFFF_FFFF);

                let eps_r = ctx.mov_f32_imm(eps);
                let denom = ctx.add_f32(total, eps_r);
                let scale = ctx.rsqrt_f32(denom);

                let idx2 = ctx.mov_u32_imm(0);
                ctx.label("gdn_l2r_scale_loop");
                let i2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(i2, head_dim_r);
                ctx.branch_if_not(in_bounds2, "gdn_l2r_exit");
                let off2 = ctx.mul_wide_u32_reg(i2, four);
                let addr2 = ctx.add_u64(head_base, off2);
                let v = ctx.ld_global_f32(addr2);
                let scaled = ctx.mul_f32(v, scale);
                ctx.st_global_f32(addr2, scaled);
                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("gdn_l2r_scale_loop");

                ctx.label("gdn_l2r_exit");
                ctx.ret();
            })
    }
}

/// The `dt`/`beta` gates for `rows` rows of `num_heads` value heads.
///
/// `alpha`, `beta_raw`, `dt` and `beta` are `[rows][num_heads]`; `dt_bias` and `a`
/// are `[num_heads]`, shared by every row. Grid `(ceil(rows * num_heads / 256), 1, 1)`,
/// block `(256, 1, 1)` — one thread per (row, head).
#[derive(Debug, Clone, Copy)]
pub struct GdnGatesRowsKernel {
    /// Value heads per row (`num_v_heads`).
    pub num_heads: u32,
}

impl GdnGatesRowsKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(num_heads: u32) -> Self {
        Self { num_heads }
    }

    /// Launch grid for `rows` rows.
    #[must_use]
    pub const fn grid(&self, rows: u32) -> (u32, u32, u32) {
        ((rows * self.num_heads).div_ceil(ELEMENTWISE_BLOCK), 1, 1)
    }

    /// Launch block.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (ELEMENTWISE_BLOCK, 1, 1)
    }
}

impl Kernel for GdnGatesRowsKernel {
    fn name(&self) -> &str {
        "gdn_gates_rows"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_heads = self.num_heads;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "alpha_ptr") // [rows][num_heads]
            .param(PtxType::U64, "dt_bias_ptr") // [num_heads]
            .param(PtxType::U64, "a_ptr") // [num_heads]
            .param(PtxType::U64, "beta_raw_ptr") // [rows][num_heads]
            .param(PtxType::U64, "dt_ptr") // [rows][num_heads] out
            .param(PtxType::U64, "beta_ptr") // [rows][num_heads] out
            .param(PtxType::U32, "count") // rows * num_heads
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let cta = ctx.special_reg(PtxReg::CtaIdX);
                let block = ctx.mov_u32_imm(ELEMENTWISE_BLOCK);
                let idx = ctx.mad_lo_u32(cta, block, tid);

                let count = ctx.load_param_u32("count");
                let in_bounds = ctx.setp_lt_u32(idx, count);
                ctx.branch_if_not(in_bounds, "gdn_gates_rows_exit");

                let alpha_ptr = ctx.load_param_u64("alpha_ptr");
                let dt_bias_ptr = ctx.load_param_u64("dt_bias_ptr");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let beta_raw_ptr = ctx.load_param_u64("beta_raw_ptr");
                let dt_ptr = ctx.load_param_u64("dt_ptr");
                let beta_ptr = ctx.load_param_u64("beta_ptr");

                let four = ctx.mov_u32_imm(4);
                let off = ctx.mul_wide_u32_reg(idx, four);
                let h = ctx.rem_u32(idx, num_heads);
                let head_off = ctx.mul_wide_u32_reg(h, four);

                // dt = softplus(alpha + dt_bias[h]) * a[h] — GdnGatesKernel's ops.
                let alpha_addr = ctx.add_u64(alpha_ptr, off);
                let bias_addr = ctx.add_u64(dt_bias_ptr, head_off);
                let a_addr = ctx.add_u64(a_ptr, head_off);
                let alpha = ctx.ld_global_f32(alpha_addr);
                let bias = ctx.ld_global_f32(bias_addr);
                let a = ctx.ld_global_f32(a_addr);
                let pre = ctx.add_f32(alpha, bias);
                let sp = emit_softplus_f32(ctx, pre, "gdn_gates_rows_dt");
                let dt = ctx.mul_f32(sp, a);
                let dt_addr = ctx.add_u64(dt_ptr, off);
                ctx.st_global_f32(dt_addr, dt);

                // beta = sigmoid(beta_raw)
                let beta_raw_addr = ctx.add_u64(beta_raw_ptr, off);
                let beta_raw = ctx.ld_global_f32(beta_raw_addr);
                let beta = emit_sigmoid_f32(ctx, beta_raw);
                let beta_addr = ctx.add_u64(beta_ptr, off);
                ctx.st_global_f32(beta_addr, beta);

                ctx.label("gdn_gates_rows_exit");
                ctx.ret();
            })
    }
}

/// Partial NEOX RoPE over `rows` rows of `num_heads` heads, row `t` at position
/// `pos0 + t`.
///
/// Grid `(num_heads, rows, 1)`, block `(n_rot / 2, 1, 1)`. `theta_scale` must be
/// [`PartialNeoxRopeKernel::theta_scale`](super::PartialNeoxRopeKernel::theta_scale)'s
/// value, for the reason that kernel documents.
#[derive(Debug, Clone, Copy)]
pub struct PartialNeoxRopeRowsKernel {
    /// Heads per row.
    pub num_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// Rotated prefix of each head.
    pub n_rot: u32,
    /// Floats between row `t` and row `t + 1`.
    pub row_stride: u32,
}

impl PartialNeoxRopeRowsKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(num_heads: u32, head_dim: u32, n_rot: u32, row_stride: u32) -> Self {
        Self {
            num_heads,
            head_dim,
            n_rot,
            row_stride,
        }
    }

    /// Rotated pairs per head.
    #[must_use]
    pub const fn half(&self) -> u32 {
        self.n_rot / 2
    }

    /// Launch grid for `rows` rows.
    #[must_use]
    pub const fn grid(&self, rows: u32) -> (u32, u32, u32) {
        (self.num_heads, rows, 1)
    }

    /// Launch block — one thread per rotated pair.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (self.half(), 1, 1)
    }
}

impl Kernel for PartialNeoxRopeRowsKernel {
    fn name(&self) -> &str {
        "gdn_partial_neox_rope_rows"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let half = self.half();
        let row_stride = self.row_stride;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "x_ptr") // [rows][row_stride], rotated in place
            .param(PtxType::U32, "pos0") // position of row 0
            .param(PtxType::F32, "theta_scale")
            .shared_memory(0)
            .build(|ctx| {
                let j = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);
                let row = ctx.special_reg(PtxReg::CtaIdY);

                let half_r = ctx.mov_u32_imm(half);
                let in_bounds = ctx.setp_lt_u32(j, half_r);
                ctx.branch_if_not(in_bounds, "gdn_roper_exit");

                let x_ptr = ctx.load_param_u64("x_ptr");
                let pos0 = ctx.load_param_u32("pos0");
                let theta_scale = ctx.load_param_f32("theta_scale");
                let position = ctx.add_u32_reg(pos0, row);

                // PartialNeoxRopeKernel's body from here, at this row's position.
                let theta = ctx.cvt_f32_u32(position);
                let step = ctx.mov_u32_imm(0);
                ctx.label("gdn_roper_theta_loop");
                let more = ctx.setp_lt_u32(step, j);
                ctx.branch_if_not(more, "gdn_roper_theta_end");
                ctx.mul_f32_inplace(theta, theta_scale);
                ctx.add_u32_inplace(step, 1);
                ctx.branch("gdn_roper_theta_loop");
                ctx.label("gdn_roper_theta_end");

                let (sin, cos) = emit_sin_cos(ctx, theta);

                let row_elems = ctx.mul_u32(row, row_stride);
                let row_off = ctx.mul_wide_u32(row_elems, 4);
                let row_base = ctx.add_u64(x_ptr, row_off);
                let head_off = ctx.mul_wide_u32(h, head_dim * 4);
                let head_base = ctx.add_u64(row_base, head_off);
                let a_off = ctx.mul_wide_u32(j, 4);
                let a_addr = ctx.add_u64(head_base, a_off);
                let b_delta = ctx.mov_u64_imm(u64::from(half) * 4);
                let b_addr = ctx.add_u64(a_addr, b_delta);
                let a = ctx.ld_global_f32(a_addr);
                let b = ctx.ld_global_f32(b_addr);

                let a_cos = ctx.mul_f32(a, cos);
                let b_sin = ctx.mul_f32(b, sin);
                let rot_a = ctx.sub_f32(a_cos, b_sin);
                let a_sin = ctx.mul_f32(a, sin);
                let b_cos = ctx.mul_f32(b, cos);
                let rot_b = ctx.add_f32(a_sin, b_cos);
                ctx.st_global_f32(a_addr, rot_a);
                ctx.st_global_f32(b_addr, rot_b);

                ctx.label("gdn_roper_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_rows_kernels_emit_their_entries_and_no_fma_beyond_the_twins() {
        let l2 = PerHeadL2NormRowsKernel::new(128, 16, 1e-6, 8192).emit_ptx();
        assert!(l2.contains(".entry gdn_per_head_l2_norm_rows"), "{l2}");
        assert!(
            l2.contains("%ctaid.y"),
            "the row index must come from ctaid.y: {l2}"
        );
        let gates = GdnGatesRowsKernel::new(32).emit_ptx();
        assert!(gates.contains(".entry gdn_gates_rows"), "{gates}");
        assert!(
            gates.contains("rem.u32"),
            "dt_bias/a are indexed by idx % num_heads: {gates}"
        );
        let rope = PartialNeoxRopeRowsKernel::new(16, 256, 64, 4096).emit_ptx();
        assert!(rope.contains(".entry gdn_partial_neox_rope_rows"), "{rope}");
        assert!(rope.contains("%ctaid.y"), "{rope}");
        assert_eq!(GdnGatesRowsKernel::new(32).grid(512), (64, 1, 1));
        assert_eq!(
            PartialNeoxRopeRowsKernel::new(16, 256, 64, 4096).block(),
            (32, 1, 1)
        );
    }
}

/// Device proof: each rows kernel over `T` rows equals `T` launches of its twin.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_rows_device_tests {
    use super::{GdnGatesRowsKernel, PartialNeoxRopeRowsKernel, PerHeadL2NormRowsKernel};
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::gdn::test_support::{run_kernel, Lcg};
    use crate::kernels::gdn::{GdnGatesKernel, PartialNeoxRopeKernel, PerHeadL2NormKernel};
    use crate::kernels::Kernel;

    /// Launch with raw 64-bit argument slots (pointers and scalars alike).
    fn launch<K: Kernel>(
        ctx: &CudaContext,
        stream: &CudaStream,
        kernel: &K,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &mut [u64],
    ) {
        let mut module = CudaModule::from_ptx(ctx, &kernel.emit_ptx()).expect("module");
        let config = LaunchConfig {
            grid,
            block,
            shared_mem: 0,
        };
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast())
            .collect();
        // SAFETY: live device allocations sized for the launch; scalars in the low
        // half of their slots, as the driver reads a declared u32/f32 param.
        unsafe {
            stream
                .launch_kernel(&mut module, kernel.name(), &config, &mut raw)
                .expect("launch");
        }
        stream.synchronize().expect("sync");
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn gdn_l2_norm_rows_is_bitwise_per_row() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn rows: no CUDA device — SKIPPED");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let (head_dim, heads, rows, stride, eps) = (128usize, 16usize, 9usize, 8192usize, 1e-6);
        let data = Lcg::new(0x3596_0003).vec(rows * stride, 1.0);

        // Twin: per row, the contiguous head block at the row's start.
        let twin = PerHeadL2NormKernel::new(head_dim as u32, heads as u32, eps);
        let mut want = data.clone();
        for t in 0..rows {
            let span = t * stride..t * stride + heads * head_dim;
            let buf = GpuBuffer::from_host(&ctx, &data[span.clone()]).expect("row");
            let mut args = [buf.as_ptr()];
            run_kernel(&ctx, &stream, &twin, twin.grid(), twin.block(), &mut args);
            buf.copy_to_host(&mut want[span]).expect("row");
        }

        let k = PerHeadL2NormRowsKernel::new(head_dim as u32, heads as u32, eps, stride as u32);
        let buf = GpuBuffer::from_host(&ctx, &data).expect("all");
        let mut args = [buf.as_ptr()];
        launch(&ctx, &stream, &k, k.grid(rows as u32), k.block(), &mut args);
        let mut got = vec![0.0f32; data.len()];
        buf.copy_to_host(&mut got).expect("all");

        assert_ne!(bits(&want), bits(&data), "the twin normalised nothing");
        assert_eq!(
            bits(&got),
            bits(&want),
            "l2 rows differs from per-row launches"
        );
    }

    #[test]
    fn gdn_gates_rows_is_bitwise_per_row() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn rows: no CUDA device — SKIPPED");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let (nv, rows) = (32usize, 37usize);
        let mut rng = Lcg::new(0x3596_0004);
        let alpha = rng.vec(rows * nv, 2.0);
        let beta_raw = rng.vec(rows * nv, 2.0);
        let dt_bias = rng.vec(nv, 1.0);
        let a: Vec<f32> = rng.vec(nv, 1.0).iter().map(|x| -x.abs() - 0.1).collect();
        let dt_bias_buf = GpuBuffer::from_host(&ctx, &dt_bias).expect("bias");
        let a_buf = GpuBuffer::from_host(&ctx, &a).expect("a");

        let twin = GdnGatesKernel::new(nv as u32);
        let (mut want_dt, mut want_beta) = (vec![0.0f32; rows * nv], vec![0.0f32; rows * nv]);
        for t in 0..rows {
            let s = t * nv..(t + 1) * nv;
            let al = GpuBuffer::from_host(&ctx, &alpha[s.clone()]).expect("alpha");
            let br = GpuBuffer::from_host(&ctx, &beta_raw[s.clone()]).expect("beta_raw");
            let dt = GpuBuffer::<f32>::new(&ctx, nv).expect("dt");
            let be = GpuBuffer::<f32>::new(&ctx, nv).expect("beta");
            let mut args = [
                al.as_ptr(),
                dt_bias_buf.as_ptr(),
                a_buf.as_ptr(),
                br.as_ptr(),
                dt.as_ptr(),
                be.as_ptr(),
            ];
            run_kernel(&ctx, &stream, &twin, twin.grid(), twin.block(), &mut args);
            dt.copy_to_host(&mut want_dt[s.clone()]).expect("dt");
            be.copy_to_host(&mut want_beta[s]).expect("beta");
        }

        let k = GdnGatesRowsKernel::new(nv as u32);
        let al = GpuBuffer::from_host(&ctx, &alpha).expect("alpha");
        let br = GpuBuffer::from_host(&ctx, &beta_raw).expect("beta_raw");
        let dt = GpuBuffer::<f32>::new(&ctx, rows * nv).expect("dt");
        let be = GpuBuffer::<f32>::new(&ctx, rows * nv).expect("beta");
        let mut args = [
            al.as_ptr(),
            dt_bias_buf.as_ptr(),
            a_buf.as_ptr(),
            br.as_ptr(),
            dt.as_ptr(),
            be.as_ptr(),
            (rows * nv) as u64,
        ];
        launch(&ctx, &stream, &k, k.grid(rows as u32), k.block(), &mut args);
        let (mut got_dt, mut got_beta) = (vec![0.0f32; rows * nv], vec![0.0f32; rows * nv]);
        dt.copy_to_host(&mut got_dt).expect("dt");
        be.copy_to_host(&mut got_beta).expect("beta");

        assert!(want_dt.iter().any(|v| v.abs() > 1e-3), "twin dt is ~zero");
        assert_eq!(bits(&got_dt), bits(&want_dt), "gates rows dt differs");
        assert_eq!(bits(&got_beta), bits(&want_beta), "gates rows beta differs");
    }

    #[test]
    fn gdn_rope_rows_is_bitwise_per_row_at_each_rows_position() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn rows: no CUDA device — SKIPPED");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        // Qwen3.5-9B's full-attention q: 16 heads x 256, n_rot 64. A pos0 far from 0
        // so theta is large and the Cody-Waite reduction is exercised.
        let (heads, head_dim, n_rot, rows, pos0) = (16usize, 256usize, 64u32, 11usize, 20_000u32);
        let stride = heads * head_dim;
        let data = Lcg::new(0x3596_0005).vec(rows * stride, 1.0);
        let theta_scale =
            PartialNeoxRopeKernel::new(heads as u32, head_dim as u32, n_rot).theta_scale(1.0e7);

        let twin = PartialNeoxRopeKernel::new(heads as u32, head_dim as u32, n_rot);
        let mut want = data.clone();
        for t in 0..rows {
            let s = t * stride..(t + 1) * stride;
            let buf = GpuBuffer::from_host(&ctx, &data[s.clone()]).expect("row");
            let mut module = CudaModule::from_ptx(&ctx, &twin.emit_ptx()).expect("twin");
            let config = LaunchConfig {
                grid: twin.grid(),
                block: twin.block(),
                shared_mem: 0,
            };
            let mut args = [
                buf.as_ptr(),
                u64::from(pos0 + t as u32),
                u64::from(theta_scale.to_bits()),
            ];
            let mut raw: Vec<*mut std::ffi::c_void> = args
                .iter_mut()
                .map(|a| std::ptr::from_mut(a).cast())
                .collect();
            // SAFETY: one live row buffer; the u32 and f32 scalars are in the low half
            // of their slots, as the twin declares them.
            unsafe {
                stream
                    .launch_kernel(&mut module, twin.name(), &config, &mut raw)
                    .expect("twin launch");
            }
            stream.synchronize().expect("sync");
            buf.copy_to_host(&mut want[s]).expect("row");
        }

        let k = PartialNeoxRopeRowsKernel::new(heads as u32, head_dim as u32, n_rot, stride as u32);
        let buf = GpuBuffer::from_host(&ctx, &data).expect("all");
        let mut args = [
            buf.as_ptr(),
            u64::from(pos0),
            u64::from(theta_scale.to_bits()),
        ];
        launch(&ctx, &stream, &k, k.grid(rows as u32), k.block(), &mut args);
        let mut got = vec![0.0f32; data.len()];
        buf.copy_to_host(&mut got).expect("all");

        assert_ne!(bits(&want), bits(&data), "the twin rotated nothing");
        assert_eq!(
            bits(&got),
            bits(&want),
            "rope rows differs from per-row launches"
        );
    }
}
