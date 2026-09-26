//! PMAT-3596: the gated delta-rule recurrence over a whole prefill chunk.
//!
//! [`DeltaRuleRecurrenceKernel`](super::DeltaRuleRecurrenceKernel) advances the state by
//! ONE token per launch and reads and writes the whole state from global memory each
//! time. A prompt of `T` tokens is therefore `T` launches and `2T` full-state round
//! trips per layer. This kernel advances it by `T` tokens in ONE launch, with each
//! thread's state row held in registers for the whole chunk — the "chunk-resident scan"
//! the #3596 ruling defines as the chunked form.
//!
//! ## Why it is bitwise-identical to `T` launches of the per-token kernel
//!
//! Both kernels give each state row `j` to one WARP, and lane `l` owns the row elements
//! `i = l, l + 32, …`. Per token, per value head `h`, each lane runs the per-token
//! kernel's exact sequence on its elements, `m` ascending:
//!
//! ```text
//! 1+2.  sum += (s[i] * exp(gate[h])) * k[kh*Dk + i]        (s[i] NOT overwritten)
//!       sum  = warp_sum_broadcast(sum)
//!       delta = (v[j] - sum) * beta[h]
//! 3+4.  s[i] = s[i] * exp(gate[h]) + k[kh*Dk + i] * delta;  out += s[i] * q[kh*Dk + i]
//!       out  = warp_sum_broadcast(out);  o[j] = out * Dk^-0.5   (lane 0)
//! ```
//!
//! The f32 operations are the per-token kernel's, in its order: the same `mul`/`add`
//! dependency chains, the same `emit_exp_f32`, the same shuffle tree
//! (`delta_rule::warp_sum_broadcast`). Only WHERE the state lives changes: `Dk / 32`
//! registers per lane for the whole chunk instead of a global round trip per token.
//!
//! **Equality is measured, not constructed.** The builder records `.rn` on these ops
//! but the emitter does not print a rounding modifier (the PTX says `mul.f32`,
//! `add.f32`), so ptxas may contract a `mul` feeding an `add` into an `fma` — in this
//! kernel and in the per-token one alike. The two present ptxas with the same chains,
//! so they contract the same way; the device test asserts the consequence exactly —
//! equality of every output and of the final state, not a tolerance — and has to be
//! run on each architecture the path ships on (sm_89 and sm_121), because the
//! contraction decision is ptxas's, per target.
//!
//! ## Layout
//!
//! Token `t` of the chunk reads `q`, `k` and `v` at row `t` of buffers whose rows are
//! `qkv_row_stride` floats apart — the `[T][conv_dim]` conv output, with the three
//! pointers already offset to their sections. `beta` and `gate` are `[T][num_v_heads]`,
//! and the output is `[T][out_row_stride]`. The state is the per-token kernel's layout
//! exactly (`s_h[j * Dk + i] == S[i][j]`), read once at the start and written once at
//! the end.
//!
//! Grid and block are the per-token kernel's: `(ceil(Dv / ROWS_PER_BLOCK), num_v_heads,
//! 1)` and `(32 * ROWS_PER_BLOCK, 1, 1)`. Warps never communicate, so there is no
//! barrier and a warp past the last row exits at once.

use crate::kernels::gdn::delta_rule::{warp_sum_broadcast, ROWS_PER_BLOCK};
use crate::kernels::gdn::emit_exp_f32;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Gated delta-rule recurrence over `t_count` consecutive tokens, state in registers.
///
/// Shapes as [`DeltaRuleRecurrenceKernel`](super::DeltaRuleRecurrenceKernel), plus the
/// two row strides of the chunk buffers.
#[derive(Debug, Clone, Copy)]
pub struct DeltaRuleChunkScanKernel {
    /// Number of key/query heads.
    pub num_k_heads: u32,
    /// Key/query head width `Dk` — each lane holds `ceil(Dk / 32)` state floats.
    pub head_k_dim: u32,
    /// Number of value heads — grid `y`.
    pub num_v_heads: u32,
    /// Value head width `Dv` — one warp per state row.
    pub head_v_dim: u32,
    /// Floats between token `t` and token `t + 1` in the `q`, `k` and `v` buffers.
    pub qkv_row_stride: u32,
    /// Floats between token `t` and token `t + 1` in the output buffer.
    pub out_row_stride: u32,
}

impl DeltaRuleChunkScanKernel {
    /// Create the kernel.
    ///
    /// # Panics
    /// If `num_v_heads` is not a positive multiple of `num_k_heads` (the tiled
    /// `h % num_k_heads` mapping the per-token kernel uses), or a head width is zero.
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
            head_k_dim > 0 && head_v_dim > 0,
            "head widths must be non-zero"
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

    /// Launch grid — the per-token kernel's: row tiles in `x`, value heads in `y`.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (
            self.head_v_dim.div_ceil(ROWS_PER_BLOCK),
            self.num_v_heads,
            1,
        )
    }

    /// Launch block — one warp per state row, as the per-token kernel.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32 * ROWS_PER_BLOCK, 1, 1)
    }

    /// Shared memory: none — `k` and `q` are read straight from global memory, where
    /// the four warps of a block share them through L1.
    #[must_use]
    pub const fn shared_bytes(&self) -> usize {
        0
    }
}

impl Kernel for DeltaRuleChunkScanKernel {
    fn name(&self) -> &str {
        "gdn_delta_rule_chunk_scan"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let dk = self.head_k_dim;
        let dv = self.head_v_dim;
        let nk = self.num_k_heads;
        let nv = self.num_v_heads;
        let qkv_stride = self.qkv_row_stride;
        let out_stride = self.out_row_stride;
        // The per-token kernel's immediate, computed the same way.
        let scale = 1.0 / (dk as f32).sqrt();
        // Lane l holds elements l + 32 m for m in 0..per_lane; only the last can be
        // past the row end, and only when Dk is not a multiple of 32.
        let per_lane = dk.div_ceil(32);
        let ragged = dk % 32 != 0;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // [T][qkv_row_stride], offset to the q section
            .param(PtxType::U64, "k_ptr") // [T][qkv_row_stride], offset to the k section
            .param(PtxType::U64, "v_ptr") // [T][qkv_row_stride], offset to the v section
            .param(PtxType::U64, "beta_ptr") // [T][num_v_heads]
            .param(PtxType::U64, "gate_ptr") // [T][num_v_heads] (dt)
            .param(PtxType::U64, "state_ptr") // [num_v_heads * Dv * Dk], updated in place
            .param(PtxType::U64, "output_ptr") // [T][out_row_stride]
            .param(PtxType::U32, "t_count")
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let lane = ctx.and_u32_imm(tid, 31);
                let warp = ctx.shr_u32_imm(tid, 5);
                let tile = ctx.special_reg(PtxReg::CtaIdX);
                let h = ctx.special_reg(PtxReg::CtaIdY);
                let rows_r = ctx.mov_u32_imm(ROWS_PER_BLOCK);
                let tile_base = ctx.mul_u32_reg(tile, rows_r);
                let j = ctx.add_u32_reg(tile_base, warp);

                // Warp-uniform exit; no barrier follows.
                let dv_r = ctx.mov_u32_imm(dv);
                let row_in_bounds = ctx.setp_lt_u32(j, dv_r);
                ctx.branch_if_not(row_in_bounds, "gdn_scan_exit");

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let beta_ptr = ctx.load_param_u64("beta_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let state_ptr = ctx.load_param_u64("state_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let t_count = ctx.load_param_u32("t_count");

                let four = ctx.mov_u32_imm(4);

                // The key/query head this value head reads — the per-token kernel's
                // `h % num_k_heads`, emitted only when the heads are grouped.
                let kh = if nk == nv { h } else { ctx.rem_u32(h, nk) };
                let kh_elems = ctx.mul_u32(kh, dk);
                let kh_lane = ctx.add_u32_reg(kh_elems, lane);
                let h_v_elems = ctx.mul_u32(h, dv);
                let v_col = ctx.add_u32_reg(h_v_elems, j); // h * Dv + j

                // Last element's guard, computed once: lane + 32 (per_lane - 1) < Dk.
                let tail_ok = if ragged {
                    let tail_lane_limit = ctx.mov_u32_imm(dk - 32 * (per_lane - 1));
                    Some(ctx.setp_lt_u32(lane, tail_lane_limit))
                } else {
                    None
                };
                let guard = |m: u32| (m + 1 == per_lane).then_some(()).and(tail_ok);

                // This lane's slice of the state row, into registers.
                let state_head_off = ctx.mul_wide_u32(h, dv * dk * 4);
                let state_head = ctx.add_u64(state_ptr, state_head_off);
                let row_off = ctx.mul_wide_u32(j, dk * 4);
                let s_row = ctx.add_u64(state_head, row_off);
                let lane_off = ctx.mul_wide_u32_reg(lane, four);
                let s_lane = ctx.add_u64(s_row, lane_off);
                let s_addrs: Vec<_> = (0..per_lane)
                    .map(|m| {
                        let off = ctx.mov_u64_imm(u64::from(m) * 128);
                        ctx.add_u64(s_lane, off)
                    })
                    .collect();
                let s: Vec<_> = (0..per_lane)
                    .map(|m| {
                        let s_m = ctx.mov_f32_imm(0.0);
                        let skip = format!("gdn_scan_load_skip_{m}");
                        if let Some(ok) = guard(m) {
                            ctx.branch_if_not(ok, &skip);
                        }
                        let loaded = ctx.ld_global_f32(s_addrs[m as usize]);
                        ctx.mov_f32_reg(s_m, loaded);
                        if guard(m).is_some() {
                            ctx.label(&skip);
                        }
                        s_m
                    })
                    .collect();

                let t = ctx.mov_u32_imm(0);
                ctx.label("gdn_scan_token_loop");
                let more = ctx.setp_lt_u32(t, t_count);
                ctx.branch_if_not(more, "gdn_scan_token_end");

                // This token's k/q base for this lane: row t, key head kh, element lane.
                let qkv_row = ctx.mul_u32(t, qkv_stride);
                let qk_elem = ctx.add_u32_reg(qkv_row, kh_lane);
                let qk_off = ctx.mul_wide_u32_reg(qk_elem, four);
                let k_lane = ctx.add_u64(k_ptr, qk_off);
                let q_lane = ctx.add_u64(q_ptr, qk_off);

                // Per-(token, head) scalars.
                let nv_row = ctx.mul_u32(t, nv);
                let scalar_idx = ctx.add_u32_reg(nv_row, h);
                let scalar_off = ctx.mul_wide_u32_reg(scalar_idx, four);
                let beta_addr = ctx.add_u64(beta_ptr, scalar_off);
                let gate_addr = ctx.add_u64(gate_ptr, scalar_off);
                let beta = ctx.ld_global_f32(beta_addr);
                let gate = ctx.ld_global_f32(gate_addr);
                let exp_gate = emit_exp_f32(ctx, gate);

                // Pass 1 (steps 1 + 2): the decayed row dotted with k; s is NOT written.
                let sum = ctx.mov_f32_imm(0.0);
                for (m, &s_m) in s.iter().enumerate() {
                    let skip = format!("gdn_scan_p1_skip_{m}");
                    if let Some(ok) = guard(m as u32) {
                        ctx.branch_if_not(ok, &skip);
                    }
                    let off = ctx.mov_u64_imm(m as u64 * 128);
                    let k_addr = ctx.add_u64(k_lane, off);
                    let s_scaled = ctx.mul_f32(s_m, exp_gate);
                    let k_val = ctx.ld_global_f32(k_addr);
                    let prod = ctx.mul_f32(s_scaled, k_val);
                    ctx.add_f32_inplace(sum, prod);
                    if guard(m as u32).is_some() {
                        ctx.label(&skip);
                    }
                }
                let sum = warp_sum_broadcast(ctx, sum);

                // delta = (v[j] - sum) * beta
                let v_elem = ctx.add_u32_reg(qkv_row, v_col);
                let v_off = ctx.mul_wide_u32_reg(v_elem, four);
                let v_addr = ctx.add_u64(v_ptr, v_off);
                let v_j = ctx.ld_global_f32(v_addr);
                let diff = ctx.sub_f32(v_j, sum);
                let delta = ctx.mul_f32(diff, beta);

                // Pass 2 (steps 1 + 3 + 4): s = s * exp_gate + k * delta, dotted with q.
                let out_sum = ctx.mov_f32_imm(0.0);
                for (m, &s_m) in s.iter().enumerate() {
                    let skip = format!("gdn_scan_p2_skip_{m}");
                    if let Some(ok) = guard(m as u32) {
                        ctx.branch_if_not(ok, &skip);
                    }
                    let off = ctx.mov_u64_imm(m as u64 * 128);
                    let k_addr = ctx.add_u64(k_lane, off);
                    let q_addr = ctx.add_u64(q_lane, off);
                    let s_decayed = ctx.mul_f32(s_m, exp_gate);
                    let k_val = ctx.ld_global_f32(k_addr);
                    let upd = ctx.mul_f32(k_val, delta);
                    let s_new = ctx.add_f32(s_decayed, upd);
                    ctx.mov_f32_reg(s_m, s_new);
                    let q_val = ctx.ld_global_f32(q_addr);
                    let prod = ctx.mul_f32(s_new, q_val);
                    ctx.add_f32_inplace(out_sum, prod);
                    if guard(m as u32).is_some() {
                        ctx.label(&skip);
                    }
                }
                let out_sum = warp_sum_broadcast(ctx, out_sum);

                let zero = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane, zero);
                ctx.branch_if_not(is_lane0, "gdn_scan_out_skip");
                let scale_r = ctx.mov_f32_imm(scale);
                let result = ctx.mul_f32(out_sum, scale_r);
                let out_row = ctx.mul_u32(t, out_stride);
                let out_elem = ctx.add_u32_reg(out_row, v_col);
                let out_off = ctx.mul_wide_u32_reg(out_elem, four);
                let out_addr = ctx.add_u64(output_ptr, out_off);
                ctx.st_global_f32(out_addr, result);
                ctx.label("gdn_scan_out_skip");

                ctx.add_u32_inplace(t, 1);
                ctx.branch("gdn_scan_token_loop");
                ctx.label("gdn_scan_token_end");

                // This lane's slice of the state row, back to global memory once.
                for (m, (&addr, &s_m)) in s_addrs.iter().zip(&s).enumerate() {
                    let skip = format!("gdn_scan_store_skip_{m}");
                    if let Some(ok) = guard(m as u32) {
                        ctx.branch_if_not(ok, &skip);
                    }
                    ctx.st_global_f32(addr, s_m);
                    if guard(m as u32).is_some() {
                        ctx.label(&skip);
                    }
                }
                ctx.label("gdn_scan_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_chunk_scan_ptx_shape() {
        // Qwen3.5-9B: 16 key heads, 32 value heads, 128-wide, conv_dim 8192.
        let kernel = DeltaRuleChunkScanKernel::new(16, 128, 32, 128, 8192, 4096);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_delta_rule_chunk_scan"), "{ptx}");
        // Each lane's slice of the state row (Dk / 32 = 4 floats) is stored once, plus
        // the one output store per token (lane 0).
        assert_eq!(
            ptx.matches("st.global.f32").count(),
            4 + 1,
            "state + output stores"
        );
        // No explicit fma: the per-token kernel emits none, and one here would be a
        // rounding the twin does not have (ptxas's own contraction is the same for
        // both; the device test measures it).
        assert!(
            !ptx.contains("fma."),
            "an explicit fma the per-token kernel lacks: {ptx}"
        );
        // Per token and element: pass 1 decays + dots (2 muls), pass 2 re-decays,
        // updates and dots (3 muls) — the per-token kernel's five.
        assert!(ptx.matches("mul.f32").count() >= 5 * 4, "{ptx}");
        // The per-token kernel's two shuffle trees, no barrier, no shared staging.
        assert_eq!(ptx.matches("shfl.sync.down").count(), 10, "{ptx}");
        assert!(!ptx.contains("bar.sync"), "{ptx}");
        // Grouped heads: the tiled modulo is emitted.
        assert!(ptx.contains("rem.u32"), "{ptx}");
        assert_eq!(kernel.grid(), (32, 32, 1));
        assert_eq!(kernel.block(), (128, 1, 1));
        assert_eq!(kernel.shared_bytes(), 0);
    }

    #[test]
    fn gdn_chunk_scan_symmetric_heads_emit_no_modulo() {
        let ptx = DeltaRuleChunkScanKernel::new(16, 128, 16, 128, 6144, 2048).emit_ptx();
        assert!(!ptx.contains("rem.u32"), "{ptx}");
    }

    #[test]
    #[should_panic(expected = "positive multiple")]
    fn gdn_chunk_scan_refuses_ungroupable_heads() {
        let _ = DeltaRuleChunkScanKernel::new(16, 128, 24, 128, 8192, 3072);
    }
}

/// Device proof: ONE chunk-scan launch over `T` tokens equals `T` launches of the
/// per-token kernel, bit for bit — the definition of "chunked form" ruled on #3596.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_chunk_scan_device_tests {
    use super::DeltaRuleChunkScanKernel;
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::gdn::test_support::{run_kernel, Lcg};
    use crate::kernels::gdn::DeltaRuleRecurrenceKernel;
    use crate::kernels::Kernel;

    /// Launch the scan, which needs its declared shared memory.
    fn run_scan(
        ctx: &CudaContext,
        stream: &CudaStream,
        kernel: &DeltaRuleChunkScanKernel,
        args: &mut [u64],
    ) {
        let ptx = kernel.emit_ptx();
        let mut module = CudaModule::from_ptx(ctx, &ptx).expect("scan module");
        let config = LaunchConfig {
            grid: kernel.grid(),
            block: kernel.block(),
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
                .launch_kernel(&mut module, kernel.name(), &config, &mut raw)
                .expect("scan launch");
        }
        stream.synchronize().expect("sync");
    }

    /// Unit-norm rows, as the L2 norm upstream of the recurrence produces them —
    /// arbitrary-magnitude k would blow the state up and hide nothing but test noise.
    fn unit_heads(rng: &mut Lcg, heads: usize, dim: usize) -> Vec<f32> {
        let mut v = rng.vec(heads * dim, 1.0);
        for h in v.chunks_mut(dim) {
            let n = h.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            h.iter_mut().for_each(|x| *x /= n);
        }
        v
    }

    fn scan_equals_per_token(nk: usize, nv: usize, dk: usize, dv: usize, tokens: usize) {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_chunk_scan: no CUDA device — SKIPPED (the PTX tests still run)");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let k_dim = nk * dk;
        let v_dim = nv * dv;
        let stride = 2 * k_dim + v_dim; // the conv output row: [q | k | v]

        let mut rng = Lcg::new(0x3596_0001 ^ (tokens as u32));
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
        // beta in (0, 1), gate (dt) negative so exp(gate) decays, as the gates emit.
        let beta: Vec<f32> = (0..tokens * nv).map(|_| 0.5 + 0.49 * rng.next()).collect();
        let gate: Vec<f32> = (0..tokens * nv)
            .map(|_| -0.05 + 0.04 * rng.next())
            .collect();
        let state0 = rng.vec(nv * dv * dk, 0.1);

        // --- per-token: T launches, each on one row ---
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

        // --- chunk scan: one launch over all T rows ---
        let scan = DeltaRuleChunkScanKernel::new(
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
        run_scan(&ctx, &stream, &scan, &mut args);
        let mut got_out = vec![0.0f32; tokens * v_dim];
        out_buf.copy_to_host(&mut got_out).expect("out");
        let mut got_state = vec![0.0f32; state0.len()];
        state_b.copy_to_host(&mut got_state).expect("state b");

        // Not vacuous: the recurrence moved the state and produced non-zero output.
        assert!(
            want_out.iter().any(|v| v.abs() > 1e-4),
            "per-token output is ~zero"
        );
        assert_ne!(
            want_state, state0,
            "the per-token path never moved the state"
        );

        let first_out = got_out
            .iter()
            .zip(&want_out)
            .position(|(g, w)| g.to_bits() != w.to_bits());
        assert!(
            first_out.is_none(),
            "chunk scan output differs from {tokens} per-token launches at flat index {i} \
             (token {t}): scan {g:e} vs per-token {w:e} — the scan must be bitwise-identical",
            i = first_out.unwrap_or(0),
            t = first_out.unwrap_or(0) / v_dim,
            g = got_out[first_out.unwrap_or(0)],
            w = want_out[first_out.unwrap_or(0)],
        );
        let first_state = got_state
            .iter()
            .zip(&want_state)
            .position(|(g, w)| g.to_bits() != w.to_bits());
        assert!(
            first_state.is_none(),
            "chunk scan final state differs from the per-token state at index {}",
            first_state.unwrap_or(0)
        );
    }

    #[test]
    fn gdn_chunk_scan_is_bitwise_per_token_0_8b_shape() {
        // 0.8B / 2B: 16 key heads, 16 value heads.
        scan_equals_per_token(16, 16, 128, 128, 37);
    }

    #[test]
    fn gdn_chunk_scan_is_bitwise_per_token_9b_grouped_shape() {
        // 4B / 9B: 16 key heads, 32 value heads (tiled h % 16).
        scan_equals_per_token(16, 32, 128, 128, 64);
    }

    #[test]
    fn gdn_chunk_scan_is_bitwise_per_token_27b_grouped_shape() {
        // 27B: 16 key heads, 48 value heads.
        scan_equals_per_token(16, 48, 128, 128, 9);
    }

    #[test]
    fn gdn_chunk_scan_one_token_is_one_per_token_step() {
        scan_equals_per_token(16, 32, 128, 128, 1);
    }
}
