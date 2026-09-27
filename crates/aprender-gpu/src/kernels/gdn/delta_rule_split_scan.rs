//! #4376: the gated delta-rule chunk scan with each state row split across lanes.
//!
//! [`DeltaRuleChunkScanKernel`](super::DeltaRuleChunkScanKernel) runs one block per
//! value head and one thread per state row. On Qwen3.5-4B that is 32 blocks of 128
//! threads on a 128-SM part, and each thread walks two dependent `Dk`-long add chains per
//! token: measured 450 µs a layer at pp1006 on an RTX 4090, 20% of the whole prefill
//! (nsys, car @219367a391). The chains, not bandwidth, are the cost.
//!
//! This kernel cuts both. Each state row `j` is held by [`LANES`] adjacent lanes, lane
//! `l` owning the entries `i = m·LANES + l` (interleaved so the lanes of a row read
//! distinct shared-memory banks). Each lane keeps two partial sums (even and odd `m`),
//! and the lanes combine through warp shuffles. The rows of a head are also spread over
//! `Dv / COLS` blocks, so the grid is `(num_v_heads, Dv / COLS)`.
//!
//! **Not bitwise-identical to the per-token kernel.** The dot products are summed in a
//! different order, so outputs and the final state match within f32 rounding, not bit for
//! bit. The device test bounds the difference against `T` per-token launches. The
//! bitwise kernel is kept as the reference and as the `APR_QWEN35_SCAN=bitwise`
//! fallback.
//!
//! Per token, lane `l` of row `j` (all `LANES` lanes reach every barrier and shuffle):
//!
//! ```text
//! 1+2.  for m:  s[m] *= exp(gate[h]);  part += s[m] * k[m·L + l]
//!       sum = Σ_lanes part;  delta = (v[j] - sum) * beta[h]
//! 3+4.  for m:  s[m] += k[m·L + l] * delta;  opart += s[m] * q[m·L + l]
//!       lane 0:  o[j] = (Σ_lanes opart) * Dk^-0.5
//! ```

use crate::kernels::gdn::emit_exp_f32;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Lanes that share one state row.
pub const LANES: u32 = 4;
/// State rows per block.
pub const COLS: u32 = 32;

/// Gated delta-rule recurrence over `t_count` tokens, each state row split over
/// [`LANES`] lanes. Shapes and buffers as
/// [`DeltaRuleChunkScanKernel`](super::DeltaRuleChunkScanKernel).
#[derive(Debug, Clone, Copy)]
pub struct DeltaRuleSplitScanKernel {
    /// Number of key/query heads.
    pub num_k_heads: u32,
    /// Key/query head width `Dk`.
    pub head_k_dim: u32,
    /// Number of value heads.
    pub num_v_heads: u32,
    /// Value head width `Dv` — the number of state rows per head.
    pub head_v_dim: u32,
    /// Floats between token `t` and token `t + 1` in the `q`, `k` and `v` buffers.
    pub qkv_row_stride: u32,
    /// Floats between token `t` and token `t + 1` in the output buffer.
    pub out_row_stride: u32,
}

impl DeltaRuleSplitScanKernel {
    /// Create the kernel.
    ///
    /// # Panics
    /// If `num_v_heads` is not a positive multiple of `num_k_heads`, `head_k_dim` is not
    /// a positive multiple of `2 · LANES`, or `head_v_dim` not a positive multiple of
    /// [`COLS`].
    #[must_use]
    pub fn new(
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
        qkv_row_stride: u32,
        out_row_stride: u32,
    ) -> Self {
        assert!(
            num_k_heads > 0 && num_v_heads % num_k_heads == 0,
            "num_v_heads {num_v_heads} must be a positive multiple of num_k_heads {num_k_heads}"
        );
        assert!(
            head_k_dim > 0 && head_k_dim % (2 * LANES) == 0,
            "head_k_dim {head_k_dim} must be a positive multiple of {}",
            2 * LANES
        );
        assert!(
            head_v_dim > 0 && head_v_dim % COLS == 0,
            "head_v_dim {head_v_dim} must be a positive multiple of {COLS}"
        );
        Self {
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            qkv_row_stride,
            out_row_stride,
        }
    }

    /// Launch grid: value heads × row groups of [`COLS`].
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.num_v_heads, self.head_v_dim / COLS, 1)
    }

    /// Launch block: [`COLS`] rows × [`LANES`] lanes. Every thread reaches every barrier
    /// and shuffle, so there is no early exit.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (COLS * LANES, 1, 1)
    }

    /// Shared memory: one token's `k` head and `q` head.
    #[must_use]
    pub const fn shared_bytes(&self) -> usize {
        (self.head_k_dim * 2 * 4) as usize
    }
}

impl Kernel for DeltaRuleSplitScanKernel {
    fn name(&self) -> &str {
        "gdn_delta_rule_split_scan"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let dk = self.head_k_dim;
        let dv = self.head_v_dim;
        let nk = self.num_k_heads;
        let nv = self.num_v_heads;
        let qkv_stride = self.qkv_row_stride;
        let out_stride = self.out_row_stride;
        let per_lane = dk / LANES;
        let threads = COLS * LANES;
        let scale = 1.0 / (dk as f32).sqrt();
        // Shared layout: k head at [0, Dk), q head at [Dk, 2 Dk).
        let q_shared_base = u64::from(dk) * 4;
        let full = 0xFFFF_FFFF;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "beta_ptr")
            .param(PtxType::U64, "gate_ptr")
            .param(PtxType::U64, "state_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "t_count")
            .shared_memory(self.shared_bytes())
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);
                let grp = ctx.special_reg(PtxReg::CtaIdY);

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let beta_ptr = ctx.load_param_u64("beta_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let state_ptr = ctx.load_param_u64("state_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let t_count = ctx.load_param_u32("t_count");

                let four = ctx.mov_u32_imm(4);

                // tid = col · LANES + l; the row is j = grp · COLS + col.
                let lane = ctx.and_u32_imm(tid, LANES - 1);
                let col = ctx.shr_u32_imm(tid, LANES.trailing_zeros());
                let grp_base = ctx.mul_u32(grp, COLS);
                let j = ctx.add_u32_reg(grp_base, col);
                // The warp lane of this row's lane 0 — the reduction's broadcast source.
                let warp_lane = ctx.and_u32_imm(tid, 31);
                let row_lane0 = ctx.and_u32_imm(warp_lane, !(LANES - 1));
                let is_lane0 = {
                    let zero = ctx.mov_u32_imm(0);
                    ctx.setp_eq_u32(lane, zero)
                };

                let kh = if nk == nv { h } else { ctx.rem_u32(h, nk) };
                let kh_elems = ctx.mul_u32(kh, dk);
                let h_v_elems = ctx.mul_u32(h, dv);
                let v_col = ctx.add_u32_reg(h_v_elems, j); // h · Dv + j

                // This lane's shared k/q byte offset, lane · 4; entry m adds m · LANES · 4.
                let lane_bytes32 = ctx.mul_u32(lane, 4);
                let lane_bytes = ctx.cvt_u64_u32(lane_bytes32);

                // State entries s_h[j · Dk + m · LANES + l] for m in 0..Dk/LANES.
                let state_head_off = ctx.mul_wide_u32(h, dv * dk * 4);
                let state_head = ctx.add_u64(state_ptr, state_head_off);
                let row_off = ctx.mul_wide_u32(j, dk * 4);
                let s_row0 = ctx.add_u64(state_head, row_off);
                let s_row = ctx.add_u64(s_row0, lane_bytes);
                let s_addrs: Vec<_> = (0..per_lane)
                    .map(|m| {
                        let off = ctx.mov_u64_imm(u64::from(m * LANES) * 4);
                        ctx.add_u64(s_row, off)
                    })
                    .collect();
                let s: Vec<_> = s_addrs.iter().map(|&a| ctx.ld_global_f32(a)).collect();

                let t = ctx.mov_u32_imm(0);
                ctx.label("gdn_split_token_loop");
                let more = ctx.setp_lt_u32(t, t_count);
                ctx.branch_if_not(more, "gdn_split_token_end");

                ctx.bar_sync(0);

                // Stage this token's k and q heads: thread tid copies tid, tid + threads, …
                let qkv_row = ctx.mul_u32(t, qkv_stride);
                let qk_head_row = ctx.add_u32_reg(qkv_row, kh_elems);
                for base in (0..dk).step_by(threads as usize) {
                    let base_r = ctx.mov_u32_imm(base);
                    let idx = ctx.add_u32_reg(base_r, tid);
                    let skip = format!("gdn_split_stage_skip_{base}");
                    let dk_r = ctx.mov_u32_imm(dk);
                    let in_head = ctx.setp_lt_u32(idx, dk_r);
                    ctx.branch_if_not(in_head, &skip);
                    let elem = ctx.add_u32_reg(qk_head_row, idx);
                    let elem_off = ctx.mul_wide_u32_reg(elem, four);
                    let k_addr = ctx.add_u64(k_ptr, elem_off);
                    let q_addr = ctx.add_u64(q_ptr, elem_off);
                    let k_val = ctx.ld_global_f32(k_addr);
                    let q_val = ctx.ld_global_f32(q_addr);
                    let slot = ctx.mul_u32(idx, 4);
                    let k_slot = ctx.cvt_u64_u32(slot);
                    ctx.st_shared_f32(k_slot, k_val);
                    let q_off = ctx.mov_u64_imm(q_shared_base);
                    let q_slot = ctx.add_u64(k_slot, q_off);
                    ctx.st_shared_f32(q_slot, q_val);
                    ctx.label(&skip);
                }
                ctx.bar_sync(0);

                let nv_row = ctx.mul_u32(t, nv);
                let scalar_idx = ctx.add_u32_reg(nv_row, h);
                let scalar_off = ctx.mul_wide_u32_reg(scalar_idx, four);
                let beta_addr = ctx.add_u64(beta_ptr, scalar_off);
                let gate_addr = ctx.add_u64(gate_ptr, scalar_off);
                let beta = ctx.ld_global_f32(beta_addr);
                let gate = ctx.ld_global_f32(gate_addr);
                let exp_gate = emit_exp_f32(ctx, gate);

                // Steps 1 + 2: decay this lane's entries and dot them with k, two chains.
                let part = [ctx.mov_f32_imm(0.0), ctx.mov_f32_imm(0.0)];
                for (m, &s_m) in s.iter().enumerate() {
                    ctx.mul_f32_inplace(s_m, exp_gate);
                    let off = ctx.mov_u64_imm(m as u64 * u64::from(LANES) * 4);
                    let k_slot = ctx.add_u64(lane_bytes, off);
                    let k_val = ctx.ld_shared_f32(k_slot);
                    let prod = ctx.mul_f32(s_m, k_val);
                    ctx.add_f32_inplace(part[m % 2], prod);
                }
                let lane_sum = ctx.add_f32(part[0], part[1]);
                let sum = reduce_row(ctx, lane_sum, full);
                let sum = ctx.shfl_idx_f32_reg(sum, row_lane0, full);

                let v_elem = ctx.add_u32_reg(qkv_row, v_col);
                let v_off = ctx.mul_wide_u32_reg(v_elem, four);
                let v_addr = ctx.add_u64(v_ptr, v_off);
                let v_j = ctx.ld_global_f32(v_addr);
                let diff = ctx.sub_f32(v_j, sum);
                let delta = ctx.mul_f32(diff, beta);

                // Steps 3 + 4: update this lane's entries and dot them with q, two chains.
                let opart = [ctx.mov_f32_imm(0.0), ctx.mov_f32_imm(0.0)];
                for (m, &s_m) in s.iter().enumerate() {
                    let off = ctx.mov_u64_imm(m as u64 * u64::from(LANES) * 4);
                    let k_slot = ctx.add_u64(lane_bytes, off);
                    let k_val = ctx.ld_shared_f32(k_slot);
                    let upd = ctx.mul_f32(k_val, delta);
                    ctx.add_f32_inplace(s_m, upd);
                    let q_off = ctx.mov_u64_imm(q_shared_base + m as u64 * u64::from(LANES) * 4);
                    let q_slot = ctx.add_u64(lane_bytes, q_off);
                    let q_val = ctx.ld_shared_f32(q_slot);
                    let prod = ctx.mul_f32(s_m, q_val);
                    ctx.add_f32_inplace(opart[m % 2], prod);
                }
                let lane_out = ctx.add_f32(opart[0], opart[1]);
                let out_sum = reduce_row(ctx, lane_out, full);

                // Lane 0 of the row holds the full dot product; it alone writes.
                ctx.branch_if_not(is_lane0, "gdn_split_store_skip");
                let scale_r = ctx.mov_f32_imm(scale);
                let result = ctx.mul_f32(out_sum, scale_r);
                let out_row = ctx.mul_u32(t, out_stride);
                let out_elem = ctx.add_u32_reg(out_row, v_col);
                let out_off = ctx.mul_wide_u32_reg(out_elem, four);
                let out_addr = ctx.add_u64(output_ptr, out_off);
                ctx.st_global_f32(out_addr, result);
                ctx.label("gdn_split_store_skip");

                ctx.add_u32_inplace(t, 1);
                ctx.branch("gdn_split_token_loop");
                ctx.label("gdn_split_token_end");

                for (&addr, &s_m) in s_addrs.iter().zip(&s) {
                    ctx.st_global_f32(addr, s_m);
                }
                ctx.ret();
            })
    }
}

/// Sum a value over the [`LANES`] lanes of one row into the row's lane 0:
/// `(p0 + p2) + (p1 + p3)` for four lanes. Other lanes hold partial garbage.
fn reduce_row(
    ctx: &mut crate::ptx::builder::KernelBuilder<'_>,
    val: crate::ptx::VirtualReg,
    mask: u32,
) -> crate::ptx::VirtualReg {
    let mut acc = val;
    let mut off = LANES / 2;
    while off > 0 {
        let other = ctx.shfl_down_f32(acc, off, mask);
        acc = ctx.add_f32(acc, other);
        off /= 2;
    }
    acc
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_split_scan_ptx_shape() {
        // Qwen3.5-4B / 9B: 16 key heads, 32 value heads, 128-wide.
        let kernel = DeltaRuleSplitScanKernel::new(16, 128, 32, 128, 8192, 4096);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_delta_rule_split_scan"), "{ptx}");
        // 32 state entries per lane stored once, plus the one output store per token.
        assert_eq!(ptx.matches("st.global.f32").count(), 32 + 1, "{ptx}");
        // Two 2-step reductions and one broadcast per token.
        assert_eq!(ptx.matches("shfl.sync.down.b32").count(), 4, "{ptx}");
        assert_eq!(ptx.matches("shfl.sync.idx.b32").count(), 1, "{ptx}");
        assert!(ptx.contains("rem.u32"), "{ptx}");
        assert_eq!(kernel.grid(), (32, 4, 1));
        assert_eq!(kernel.block(), (128, 1, 1));
        assert_eq!(kernel.shared_bytes(), 1024);
    }

    #[test]
    fn gdn_split_scan_symmetric_heads_emit_no_modulo() {
        let ptx = DeltaRuleSplitScanKernel::new(16, 128, 16, 128, 6144, 2048).emit_ptx();
        assert!(!ptx.contains("rem.u32"), "{ptx}");
    }

    #[test]
    #[should_panic(expected = "multiple of 32")]
    fn gdn_split_scan_refuses_ragged_rows() {
        let _ = DeltaRuleSplitScanKernel::new(16, 128, 16, 96 + 1, 6144, 2048);
    }
}

/// Device proof: ONE split-scan launch over `T` tokens matches `T` launches of the
/// per-token kernel within f32 summation-order rounding.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_split_scan_device_tests {
    use super::DeltaRuleSplitScanKernel;
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::gdn::test_support::{run_kernel, Lcg};
    use crate::kernels::gdn::DeltaRuleRecurrenceKernel;
    use crate::kernels::Kernel;

    fn unit_heads(rng: &mut Lcg, heads: usize, dim: usize) -> Vec<f32> {
        let mut v = rng.vec(heads * dim, 1.0);
        for h in v.chunks_mut(dim) {
            let n = h.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            h.iter_mut().for_each(|x| *x /= n);
        }
        v
    }

    /// Largest |got − want| over the largest |want|.
    fn rel_linf(got: &[f32], want: &[f32]) -> f32 {
        let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let err = got
            .iter()
            .zip(want)
            .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
        err / scale.max(1e-12)
    }

    fn split_matches_per_token(nk: usize, nv: usize, dk: usize, dv: usize, tokens: usize) {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_split_scan: no CUDA device — SKIPPED (the PTX tests still run)");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let k_dim = nk * dk;
        let v_dim = nv * dv;
        let stride = 2 * k_dim + v_dim;

        let mut rng = Lcg::new(0x4376_0002 ^ (tokens as u32));
        let mut rows = vec![0.0f32; tokens * stride];
        for t in 0..tokens {
            let q = unit_heads(&mut rng, nk, dk);
            let k = unit_heads(&mut rng, nk, dk);
            let v = rng.vec(v_dim, 1.0);
            let row = &mut rows[t * stride..(t + 1) * stride];
            row[..k_dim].copy_from_slice(&q);
            row[k_dim..2 * k_dim].copy_from_slice(&k);
            row[2 * k_dim..].copy_from_slice(&v);
        }
        let beta: Vec<f32> = (0..tokens * nv).map(|_| 0.5 + 0.49 * rng.next()).collect();
        let gate: Vec<f32> = (0..tokens * nv)
            .map(|_| -0.05 + 0.04 * rng.next())
            .collect();
        let state0 = rng.vec(nv * dv * dk, 0.1);

        let per_token = DeltaRuleRecurrenceKernel::new(nk as u32, dk as u32, nv as u32, dv as u32);
        let state_a = GpuBuffer::from_host(&ctx, &state0).expect("state a");
        let mut want_out = vec![0.0f32; tokens * v_dim];
        for t in 0..tokens {
            let row = &rows[t * stride..(t + 1) * stride];
            let q = GpuBuffer::from_host(&ctx, &row[..k_dim]).expect("q");
            let k = GpuBuffer::from_host(&ctx, &row[k_dim..2 * k_dim]).expect("k");
            let v = GpuBuffer::from_host(&ctx, &row[2 * k_dim..]).expect("v");
            let b = GpuBuffer::from_host(&ctx, &beta[t * nv..(t + 1) * nv]).expect("beta");
            let g = GpuBuffer::from_host(&ctx, &gate[t * nv..(t + 1) * nv]).expect("gate");
            let o = GpuBuffer::<f32>::new(&ctx, v_dim).expect("out");
            let mut args = [
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                b.as_ptr(),
                g.as_ptr(),
                state_a.as_ptr(),
                o.as_ptr(),
            ];
            run_kernel(
                &ctx,
                &stream,
                &per_token,
                per_token.grid(),
                per_token.block(),
                &mut args,
            );
            o.copy_to_host(&mut want_out[t * v_dim..(t + 1) * v_dim])
                .expect("download");
        }
        let mut want_state = vec![0.0f32; state0.len()];
        state_a.copy_to_host(&mut want_state).expect("state a");

        let scan = DeltaRuleSplitScanKernel::new(
            nk as u32,
            dk as u32,
            nv as u32,
            dv as u32,
            stride as u32,
            v_dim as u32,
        );
        let rows_buf = GpuBuffer::from_host(&ctx, &rows).expect("rows");
        let beta_buf = GpuBuffer::from_host(&ctx, &beta).expect("beta");
        let gate_buf = GpuBuffer::from_host(&ctx, &gate).expect("gate");
        let state_b = GpuBuffer::from_host(&ctx, &state0).expect("state b");
        let out_buf = GpuBuffer::<f32>::new(&ctx, tokens * v_dim).expect("out");
        let base = rows_buf.as_ptr();
        let mut args = [
            base,
            base + (k_dim * 4) as u64,
            base + (2 * k_dim * 4) as u64,
            beta_buf.as_ptr(),
            gate_buf.as_ptr(),
            state_b.as_ptr(),
            out_buf.as_ptr(),
            tokens as u64,
        ];
        let ptx = scan.emit_ptx();
        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("split module");
        let config = LaunchConfig {
            grid: scan.grid(),
            block: scan.block(),
            shared_mem: 0,
        };
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast())
            .collect();
        // SAFETY: every argument is a live device allocation sized for T tokens, the
        // scalar is a u32 in the low half of its slot, and grid/block are the kernel's.
        unsafe {
            stream
                .launch_kernel(&mut module, scan.name(), &config, &mut raw)
                .expect("split launch");
        }
        stream.synchronize().expect("sync");
        let mut got_out = vec![0.0f32; tokens * v_dim];
        out_buf.copy_to_host(&mut got_out).expect("out");
        let mut got_state = vec![0.0f32; state0.len()];
        state_b.copy_to_host(&mut got_state).expect("state b");

        assert!(
            want_out.iter().any(|v| v.abs() > 1e-4),
            "per-token output is ~zero"
        );
        assert_ne!(
            want_state, state0,
            "the per-token path never moved the state"
        );
        let out_err = rel_linf(&got_out, &want_out);
        let state_err = rel_linf(&got_state, &want_state);
        assert!(
            out_err <= 1e-5,
            "split scan output vs {tokens} per-token launches: rel L∞ {out_err:e} > 1e-5"
        );
        assert!(
            state_err <= 1e-5,
            "split scan final state vs per-token: rel L∞ {state_err:e} > 1e-5"
        );
    }

    #[test]
    fn gdn_split_scan_matches_per_token_0_8b_shape() {
        split_matches_per_token(16, 16, 128, 128, 37);
    }

    #[test]
    fn gdn_split_scan_matches_per_token_4b_grouped_shape() {
        split_matches_per_token(16, 32, 128, 128, 257);
    }

    #[test]
    fn gdn_split_scan_matches_per_token_27b_grouped_shape() {
        split_matches_per_token(16, 48, 128, 128, 9);
    }

    #[test]
    fn gdn_split_scan_one_token() {
        split_matches_per_token(16, 32, 128, 128, 1);
    }
}
