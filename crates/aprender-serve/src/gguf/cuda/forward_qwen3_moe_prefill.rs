//! #3714 gap 2: batched (chunked) prefill for [`Qwen3MoeCudaModel`].
//!
//! The per-token prompt loop ran `prompt_len × num_layers` router round trips and
//! `prompt_len × num_layers × k × 3` expert GEMVs. This forwards a chunk of `T` rows
//! per layer instead, with the kernels the hybrid prefill already uses (#3596):
//!
//! - every projection (q, k, v, o, router, each expert's gate/up/down) is ONE
//!   `qwen35_project_rows` GEMM over the rows it applies to;
//! - norms and RoPE are the batched kernels, row `t` at position `pos + t`;
//! - attention is the cuBLAS `QKᵀ → causal softmax → PV` pass over the KV cache,
//!   after this chunk's K/V rows are written into it (head_dim 128: the flash
//!   kernel is head_dim 256 only).
//!
//! ## Routing and the expert gather stay on the host (v1)
//!
//! Per layer the router logits and the normed rows come back once (the decode
//! path already pays one sync per layer per token; this pays one per layer per
//! chunk). [`route_top_k`] — the CPU's own rule — picks each row's experts, and
//! the rows are gathered expert-major so each expert runs one GEMM over the rows
//! routed to it. The weight `w` is folded into the SwiGLU activation before the
//! down GEMM, as in the decode path, and each row's `k` expert outputs are added
//! into its residual in slot order — the decode path's order, so a row's sum is
//! the same f32 arithmetic. A gather/scatter kernel would remove the transfers;
//! it is not needed for the long-context rungs and is not here.

use super::{gpu_err, Qwen3MoeCudaModel, Qwen3MoeCudaState, RealizarError, Result};
use crate::gguf::qwen3_moe_load::route_top_k;
use trueno_gpu::driver::GpuBuffer;

/// Rows per prefill chunk — every projection is ONE GEMM over this many rows.
pub const MOE_PREFILL_MAX_CHUNK_ROWS: usize = 512;

/// Bytes the attention scores of one pass of one KV group may take, as in the
/// hybrid prefill (`heads_per_kv × pass_rows × L × 4`).
const MOE_PREFILL_SCORES_BUDGET_BYTES: usize = 1 << 30;

/// Query rows per attention pass at the longest `L` of the call.
fn attention_rows_for(hpk: usize, total_positions: usize, rows: usize) -> usize {
    (MOE_PREFILL_SCORES_BUDGET_BYTES / (4 * hpk.max(1) * total_positions.max(1))).clamp(1, rows)
}

/// One expert's slice of the expert-major gather: rows `start..start + rows` of
/// the gathered buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpertRun {
    expert: usize,
    start: usize,
    rows: usize,
}

/// The expert-major layout of one layer's routing over `rows` rows.
#[derive(Debug, Clone, PartialEq)]
struct Gather {
    /// The experts that got at least one row, ascending, with their slices.
    runs: Vec<ExpertRun>,
    /// For gathered row `g`: the source row.
    src_row: Vec<usize>,
    /// For gathered row `g`: its routing weight.
    weight: Vec<f32>,
    /// `slot_row[t * k + s]`: the gathered row holding row `t`'s slot-`s` expert.
    slot_row: Vec<usize>,
}

/// Lay `routes` (row `t`'s `k` `(expert, weight)` pairs, rank order) out
/// expert-major. Pure, so the layout is tested without a device.
fn gather_by_expert(routes: &[Vec<(usize, f32)>], k: usize, num_experts: usize) -> Gather {
    let mut count = vec![0usize; num_experts];
    for r in routes {
        for &(e, _) in r {
            count[e] += 1;
        }
    }
    let mut start = vec![0usize; num_experts];
    let mut runs = Vec::new();
    let mut acc = 0;
    for e in 0..num_experts {
        start[e] = acc;
        if count[e] > 0 {
            runs.push(ExpertRun {
                expert: e,
                start: acc,
                rows: count[e],
            });
        }
        acc += count[e];
    }
    let mut fill = start;
    let mut src_row = vec![0usize; acc];
    let mut weight = vec![0f32; acc];
    let mut slot_row = vec![0usize; routes.len() * k];
    for (t, r) in routes.iter().enumerate() {
        for (s, &(e, w)) in r.iter().enumerate() {
            let g = fill[e];
            fill[e] += 1;
            src_row[g] = t;
            weight[g] = w;
            slot_row[t * k + s] = g;
        }
    }
    Gather {
        runs,
        src_row,
        weight,
        slot_row,
    }
}

/// Device and host buffers for one chunk of up to `rows` rows, allocated once per
/// prefill call.
struct MoePrefillBuffers {
    rows: usize,
    attn_rows: usize,
    /// `[rows][hidden]` — the residual stream.
    x: GpuBuffer<f32>,
    normed: GpuBuffer<f32>,
    proj: GpuBuffer<f32>,
    q: GpuBuffer<f32>,
    q_normed: GpuBuffer<f32>,
    k_raw: GpuBuffer<f32>,
    attn_out_in: GpuBuffer<f32>,
    scores: GpuBuffer<f32>,
    /// `[rows][num_experts]`.
    router_logits: GpuBuffer<f32>,
    /// `[rows * k][hidden]` — the normed rows, gathered expert-major.
    xg: GpuBuffer<f32>,
    /// `[rows * k][expert_dim]`.
    e_gate: GpuBuffer<f32>,
    e_up: GpuBuffer<f32>,
    e_act: GpuBuffer<f32>,
    e_act_w: GpuBuffer<f32>,
    /// `[rows * k][expert_dim]`: gathered row `g`'s weight, repeated across its row.
    route_w: GpuBuffer<f32>,
    /// `[rows * k][hidden]` — each gathered row's weighted expert output.
    yg: GpuBuffer<f32>,
    host_logits: Vec<f32>,
    host_normed: Vec<f32>,
    host_x: Vec<f32>,
    host_xg: Vec<f32>,
    host_w: Vec<f32>,
    host_yg: Vec<f32>,
}

impl Qwen3MoeCudaModel<'_> {
    fn alloc_prefill(&self, rows: usize, total_positions: usize) -> Result<MoePrefillBuffers> {
        let d = self.dims;
        let z = |n: usize| Self::zeros(&self.executor, n.max(1));
        let hidden = d.hidden_dim as usize;
        let q_dim = (d.num_heads * d.head_dim) as usize;
        let kv_dim = (d.num_kv_heads * d.head_dim) as usize;
        let hpk = (d.num_heads / d.num_kv_heads.max(1)).max(1) as usize;
        let routed = rows * d.num_experts_per_tok as usize;
        let e_dim = d.expert_dim as usize;
        let attn_rows = attention_rows_for(hpk, total_positions, rows);
        Ok(MoePrefillBuffers {
            rows,
            attn_rows,
            x: z(rows * hidden)?,
            normed: z(rows * hidden)?,
            proj: z(rows * hidden)?,
            q: z(rows * q_dim)?,
            q_normed: z(rows * q_dim)?,
            k_raw: z(rows * kv_dim)?,
            attn_out_in: z(rows * q_dim)?,
            scores: z(hpk * attn_rows * total_positions)?,
            router_logits: z(rows * d.num_experts as usize)?,
            xg: z(routed * hidden)?,
            e_gate: z(routed * e_dim)?,
            e_up: z(routed * e_dim)?,
            e_act: z(routed * e_dim)?,
            e_act_w: z(routed * e_dim)?,
            route_w: z(routed * e_dim)?,
            yg: z(routed * hidden)?,
            host_logits: vec![0.0; rows * d.num_experts as usize],
            host_normed: vec![0.0; rows * hidden],
            host_x: vec![0.0; rows * hidden],
            host_xg: vec![0.0; routed * hidden],
            host_w: vec![0.0; routed * e_dim],
            host_yg: vec![0.0; routed * hidden],
        })
    }

    /// Rows per prefill chunk for a prompt ending at `total_positions`.
    #[must_use]
    pub fn prefill_chunk_rows(total_positions: usize) -> usize {
        MOE_PREFILL_MAX_CHUNK_ROWS.min(total_positions.max(1))
    }

    /// Prefill `tokens` at positions `pos0..pos0 + tokens.len()` and return the
    /// logits of the LAST position — what `tokens.len()` calls of
    /// [`Self::forward_single`] return from their last call, with `state` advanced
    /// to the same place.
    ///
    /// # Errors
    /// An empty prompt, a `pos0` other than the state's `kv_len`, a position past
    /// the KV cache, a token outside the vocabulary, a projection whose
    /// quantization has no dequant kernel, or any device failure. On `Err` the
    /// state must be discarded: some layers' KV rows may be written.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        state: &mut Qwen3MoeCudaState,
        pos0: usize,
    ) -> Result<Vec<f32>> {
        let end = pos0 + tokens.len();
        Ok(self
            .prefill_logits_at(tokens, state, pos0, &[end.saturating_sub(1)])?
            .pop()
            .unwrap_or_default())
    }

    /// [`Self::prefill`], returning the logits at each position in `positions`
    /// (absolute, ascending) — for the F2 guard, which compares the GPU against
    /// the CPU at every probe position. Only the requested rows pay an `lm_head`.
    ///
    /// # Errors
    /// As [`Self::prefill`], and a `positions` that is not ascending or falls
    /// outside `pos0..pos0 + tokens.len()`.
    pub fn prefill_logits_at(
        &mut self,
        tokens: &[u32],
        state: &mut Qwen3MoeCudaState,
        pos0: usize,
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.prefill_check(tokens, state, pos0)?;
        let end = pos0 + tokens.len();
        if positions.windows(2).any(|w| w[0] >= w[1])
            || positions.iter().any(|&p| p < pos0 || p >= end)
        {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen3moe_cuda prefill: requested positions must ascend within {pos0}..{end}"
                ),
            });
        }
        let rows = Self::prefill_chunk_rows(end);
        let mut bufs = self.alloc_prefill(rows, end)?;
        let mut pos = pos0;
        let mut want = positions.iter().copied().peekable();
        let mut out = Vec::with_capacity(positions.len());
        for chunk in tokens.chunks(rows) {
            self.prefill_chunk(&mut bufs, chunk, state, pos)?;
            while let Some(&p) = want.peek() {
                if p >= pos + chunk.len() {
                    break;
                }
                out.push(self.prefill_tail(&bufs, p - pos)?);
                want.next();
            }
            pos += chunk.len();
        }
        Ok(out)
    }

    fn prefill_check(&self, tokens: &[u32], state: &Qwen3MoeCudaState, pos0: usize) -> Result<()> {
        let refuse = |reason: String| Err(RealizarError::InvalidShape { reason });
        if tokens.is_empty() {
            return refuse("qwen3moe_cuda prefill: the prompt is empty".to_string());
        }
        if state.kv.len() != self.layers.len() {
            return refuse("qwen3moe_cuda prefill: the state was built for another model".into());
        }
        if pos0 != state.kv_len {
            return refuse(format!(
                "qwen3moe_cuda prefill: pos0 {pos0} is not where the state stands ({} KV rows \
                 written) — a prefill continues the state, it cannot skip or rewind it",
                state.kv_len
            ));
        }
        let end = pos0 + tokens.len();
        if end > state.max_seq_len {
            return refuse(format!(
                "qwen3moe_cuda prefill: positions {pos0}..{end} are past the KV cache ({} rows)",
                state.max_seq_len
            ));
        }
        if let Some(bad) = tokens.iter().find(|&&t| t >= self.dims.vocab_size) {
            return refuse(format!(
                "qwen3moe_cuda prefill: token {bad} is outside the {}-token vocabulary",
                self.dims.vocab_size
            ));
        }
        Ok(())
    }

    /// Output norm + `lm_head` on row `row` of the chunk `bufs.x` holds.
    fn prefill_tail(&mut self, bufs: &MoePrefillBuffers, row: usize) -> Result<Vec<f32>> {
        let d = self.dims;
        let hidden = d.hidden_dim as usize;
        let err = |e: trueno_gpu::GpuError| gpu_err("qwen3moe_prefill_tail", &e);
        let last = Self::view(&bufs.x, row * hidden, hidden);
        let ex = &mut self.executor;
        let r = ex
            .rmsnorm_into(
                &last,
                &self.output_norm,
                &self.out_normed,
                d.hidden_dim,
                d.eps,
            )
            .and_then(|()| {
                ex.gemv_dispatch(
                    self.lm_head.qtype,
                    self.lm_head.ptr,
                    &self.out_normed,
                    &self.logits_buf,
                    self.lm_head.n,
                    self.lm_head.k,
                )
            })
            .and_then(|()| ex.sync_stream());
        std::mem::forget(last);
        r.map_err(err)?;
        let mut logits = vec![0.0f32; d.vocab_size as usize];
        self.logits_buf.copy_to_host(&mut logits).map_err(err)?;
        Ok(logits)
    }

    /// One chunk of `chunk.len()` rows at positions `pos..pos + chunk.len()`.
    fn prefill_chunk(
        &mut self,
        b: &mut MoePrefillBuffers,
        chunk: &[u32],
        state: &mut Qwen3MoeCudaState,
        pos: usize,
    ) -> Result<()> {
        let n = chunk.len();
        debug_assert!(n <= b.rows);
        let err = |e: trueno_gpu::GpuError| gpu_err("qwen3moe_prefill", &e);
        // The previous chunk's kernels read `x` until its last layer; the host copy
        // below is not ordered with the model's stream, so wait.
        self.executor.sync_stream().map_err(err)?;
        b.x.copy_from_host_at(&self.model.embed(chunk), 0)
            .map_err(err)?;
        for il in 0..self.layers.len() {
            self.prefill_attention(b, state, il, n, pos)
                .map_err(|e| gpu_err("qwen3moe_prefill_attention", &e))?;
            let routes = self
                .prefill_moe(b, il, n)
                .map_err(|e| gpu_err("qwen3moe_prefill_moe", &e))?;
            if let Some(last) = routes.into_iter().last() {
                self.last_routes[il] = last;
            }
        }
        state.kv_len = state.kv_len.max(pos + n);
        Ok(())
    }

    /// `attention_layer` over `n` rows.
    fn prefill_attention(
        &mut self,
        b: &MoePrefillBuffers,
        state: &Qwen3MoeCudaState,
        il: usize,
        n: usize,
        pos: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let w = &self.layers[il];
        let ex = &mut self.executor;
        let rows = n as u32;
        let q_dim = d.num_heads * d.head_dim;
        let kv_dim = d.num_kv_heads * d.head_dim;
        let (k_cache, v_cache) = &state.kv[il];
        // This chunk's rows of the cache: [pos, pos + n) x kv_dim, contiguous.
        let k_rows = Self::view(k_cache, pos * kv_dim as usize, n * kv_dim as usize);
        let v_rows_ptr = v_cache.as_ptr() + (pos as u64) * u64::from(kv_dim) * 4;

        let r = (|| {
            ex.batched_rmsnorm_into(&b.x, &w.attn_norm, &b.normed, d.hidden_dim, rows, d.eps)?;
            for (p, y, ldc) in [
                (&w.attn_q, b.q.as_ptr(), w.attn_q.n),
                (&w.attn_k, b.k_raw.as_ptr(), w.attn_k.n),
                (&w.attn_v, v_rows_ptr, kv_dim),
            ] {
                ex.qwen35_project_rows(p.qtype, p.ptr, b.normed.as_ptr(), y, rows, p.n, p.k, ldc)?;
            }
            ex.per_head_rmsnorm_into(
                &b.q,
                &w.attn_q_norm,
                &b.q_normed,
                d.head_dim,
                rows * d.num_heads,
                d.eps,
            )?;
            ex.per_head_rmsnorm_into(
                &b.k_raw,
                &w.attn_k_norm,
                &k_rows,
                d.head_dim,
                rows * d.num_kv_heads,
                d.eps,
            )?;
            let pos32 = u32::try_from(pos).unwrap_or(u32::MAX);
            ex.qwen35_rope_rows(
                b.q_normed.as_ptr(),
                d.num_heads,
                d.head_dim,
                d.head_dim,
                q_dim,
                rows,
                pos32,
                d.theta_scale,
            )?;
            ex.qwen35_rope_rows(
                k_rows.as_ptr(),
                d.num_kv_heads,
                d.head_dim,
                d.head_dim,
                kv_dim,
                rows,
                pos32,
                d.theta_scale,
            )?;
            ex.qwen35_prefill_attention(
                b.q_normed.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                b.attn_out_in.as_ptr(),
                b.scores.as_ptr(),
                rows,
                b.attn_rows as u32,
                pos32,
                d.num_heads,
                d.num_kv_heads,
                d.head_dim,
            )?;
            let p = &w.attn_output;
            ex.qwen35_project_rows(
                p.qtype,
                p.ptr,
                b.attn_out_in.as_ptr(),
                b.proj.as_ptr(),
                rows,
                p.n,
                p.k,
                p.n,
            )?;
            ex.residual_add_into(&b.x, &b.proj, &b.x, rows * d.hidden_dim)
        })();
        std::mem::forget(k_rows);
        r
    }

    /// `moe_layer` over `n` rows. Returns each row's routing.
    #[allow(clippy::too_many_lines)]
    fn prefill_moe(
        &mut self,
        b: &mut MoePrefillBuffers,
        il: usize,
        n: usize,
    ) -> std::result::Result<Vec<Vec<(usize, f32)>>, trueno_gpu::GpuError> {
        let d = self.dims;
        let w = &self.layers[il];
        let ex = &mut self.executor;
        let rows = n as u32;
        let hidden = d.hidden_dim as usize;
        let e_dim = d.expert_dim as usize;
        let k = d.num_experts_per_tok as usize;
        let ne = d.num_experts as usize;

        ex.batched_rmsnorm_into(&b.x, &w.ffn_norm, &b.normed, d.hidden_dim, rows, d.eps)?;
        let p = &w.router;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.normed.as_ptr(),
            b.router_logits.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;

        // The one host round trip of the layer: logits and normed rows come back,
        // the shared routing rule picks each row's experts, the rows go down
        // gathered expert-major.
        ex.sync_stream()?;
        b.router_logits.copy_to_host(&mut b.host_logits[..n * ne])?;
        b.normed.copy_to_host(&mut b.host_normed[..n * hidden])?;
        let routes: Vec<Vec<(usize, f32)>> = b.host_logits[..n * ne]
            .chunks_exact(ne)
            .map(|l| {
                let r = route_top_k(l, k);
                #[cfg(test)]
                let r = super::tests::inject(self.fault, r, ne);
                r
            })
            .collect();
        let g = gather_by_expert(&routes, k, ne);
        let routed = g.src_row.len();
        for (row, (&t, &wt)) in g.src_row.iter().zip(&g.weight).enumerate() {
            b.host_xg[row * hidden..(row + 1) * hidden]
                .copy_from_slice(&b.host_normed[t * hidden..(t + 1) * hidden]);
            b.host_w[row * e_dim..(row + 1) * e_dim].fill(wt);
        }
        b.xg.copy_from_host_at(&b.host_xg[..routed * hidden], 0)?;
        b.route_w
            .copy_from_host_at(&b.host_w[..routed * e_dim], 0)?;

        let f = 4u64;
        for run in &g.runs {
            let r = run.rows as u32;
            let x_in = b.xg.as_ptr() + (run.start * hidden) as u64 * f;
            for (stack, y) in [(&w.gate, &b.e_gate), (&w.up, &b.e_up)] {
                ex.qwen35_project_rows(
                    stack.qtype,
                    stack.expert(run.expert),
                    x_in,
                    y.as_ptr() + (run.start * e_dim) as u64 * f,
                    r,
                    stack.n,
                    stack.k,
                    stack.n,
                )?;
            }
        }
        let act_len = (routed * e_dim) as u32;
        ex.fused_swiglu_into(&b.e_gate, &b.e_up, &b.e_act, act_len)?;
        ex.elementwise_mul_into(&b.e_act, &b.route_w, &b.e_act_w, act_len)?;
        for run in &g.runs {
            ex.qwen35_project_rows(
                w.down.qtype,
                w.down.expert(run.expert),
                b.e_act_w.as_ptr() + (run.start * e_dim) as u64 * f,
                b.yg.as_ptr() + (run.start * hidden) as u64 * f,
                run.rows as u32,
                w.down.n,
                w.down.k,
                w.down.n,
            )?;
        }

        // Each row's k weighted expert outputs into its residual, in slot order.
        ex.sync_stream()?;
        b.yg.copy_to_host(&mut b.host_yg[..routed * hidden])?;
        b.x.copy_to_host(&mut b.host_x[..n * hidden])?;
        for t in 0..n {
            let xr = &mut b.host_x[t * hidden..(t + 1) * hidden];
            for s in 0..k {
                let gr = g.slot_row[t * k + s];
                for (a, &y) in xr
                    .iter_mut()
                    .zip(&b.host_yg[gr * hidden..(gr + 1) * hidden])
                {
                    *a += y;
                }
            }
        }
        b.x.copy_from_host_at(&b.host_x[..n * hidden], 0)?;
        Ok(routes)
    }
}

#[cfg(test)]
mod gather_tests {
    use super::{attention_rows_for, gather_by_expert, ExpertRun};

    /// The layout the expert GEMMs and the residual sum read: experts ascending,
    /// each run contiguous, every (row, slot) mapped to exactly one gathered row
    /// that carries that row's input and weight.
    #[test]
    fn gather_is_expert_major_and_every_slot_round_trips() {
        let routes = vec![
            vec![(3, 0.6), (0, 0.4)],
            vec![(0, 0.7), (2, 0.3)],
            vec![(3, 0.5), (2, 0.5)],
        ];
        let g = gather_by_expert(&routes, 2, 4);
        assert_eq!(
            g.runs,
            vec![
                ExpertRun {
                    expert: 0,
                    start: 0,
                    rows: 2
                },
                ExpertRun {
                    expert: 2,
                    start: 2,
                    rows: 2
                },
                ExpertRun {
                    expert: 3,
                    start: 4,
                    rows: 2
                },
            ]
        );
        assert_eq!(g.src_row, vec![0, 1, 1, 2, 0, 2]);
        for (t, r) in routes.iter().enumerate() {
            for (s, &(e, w)) in r.iter().enumerate() {
                let gr = g.slot_row[t * 2 + s];
                assert_eq!(g.src_row[gr], t, "row {t} slot {s}");
                assert_eq!(g.weight[gr], w, "row {t} slot {s}");
                let run = g
                    .runs
                    .iter()
                    .find(|x| x.expert == e)
                    .expect("expert has a run");
                assert!(
                    (run.start..run.start + run.rows).contains(&gr),
                    "row {t} slot {s}"
                );
            }
        }
        let mut seen = g.slot_row.clone();
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..6).collect::<Vec<_>>(),
            "each gathered row used once"
        );
    }

    /// Experts no row picked get no run, so no GEMM launches over zero rows.
    #[test]
    fn unrouted_experts_get_no_run() {
        let g = gather_by_expert(&[vec![(5, 1.0)]], 1, 8);
        assert_eq!(
            g.runs,
            vec![ExpertRun {
                expert: 5,
                start: 0,
                rows: 1
            }]
        );
    }

    /// The scores budget bounds the attention pass; it never drops below one row
    /// or exceeds the chunk.
    #[test]
    fn attention_rows_stay_within_the_chunk_and_the_budget() {
        assert_eq!(attention_rows_for(8, 512, 512), 512);
        assert_eq!(attention_rows_for(8, 131_072, 512), 256);
        assert_eq!(attention_rows_for(8, usize::MAX / 64, 512), 1);
    }
}
