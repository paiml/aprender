//! aprender#5279 P4c: the token mixers of one batched decode step.
//!
//! Every projection that reads the normed rows, and each mixer's output
//! projection, runs once for all m slots, so its weights are read once per
//! step instead of m times. The ops that read or write one slot's own state
//! (the conv window, the recurrent state, the KV cache) run per slot on row
//! views, exactly as [`Qwen35CudaModel::deltanet_mixer_inner`] and
//! [`Qwen35CudaModel::attention_mixer_inner`] run them, so each row is
//! bitwise the single-row mixer's.

use super::super::{CudaQuantWeight, Qwen35CudaDims};
use super::CudaLayer;
use super::{row_view, Qwen35BatchBufs, Qwen35CudaModel, Qwen35CudaState, Result};
use crate::cuda::CudaExecutor;
use trueno_gpu::driver::GpuBuffer;
use trueno_gpu::GpuError;

/// The m-row buffers the mixers write, each `[rows × width]`.
pub(super) struct Qwen35MixerBufs {
    /// `hidden_dim` wide: `rms_norm(hidden, attn_norm)` per row.
    normed: GpuBuffer<f32>,
    /// `hidden_dim` wide: the output projection of either mixer.
    out: GpuBuffer<f32>,
    /// `conv_dim` wide: `attn_qkv . normed`.
    conv_in: GpuBuffer<f32>,
    /// `num_v_heads` wide: `ssm_alpha . normed`.
    alpha_raw: GpuBuffer<f32>,
    /// `num_v_heads` wide: `ssm_beta . normed`.
    beta_raw: GpuBuffer<f32>,
    /// `conv_dim` wide: the conv1d + SiLU output, `[q | k | v]` per row.
    conv_out: GpuBuffer<f32>,
    /// `num_v_heads` wide: the decay gate.
    dt: GpuBuffer<f32>,
    /// `num_v_heads` wide: `sigmoid(beta_raw)`.
    beta: GpuBuffer<f32>,
    /// `v_dim` wide: the delta rule's output.
    out_h: GpuBuffer<f32>,
    /// `v_dim` wide: `attn_gate . normed`.
    gate: GpuBuffer<f32>,
    /// `v_dim` wide: the gated norm, the `ssm_out` input.
    ssm_out_in: GpuBuffer<f32>,
    /// `2 × q_dim` wide: `attn_q . normed`, `[q | gate]` per head.
    q_full: GpuBuffer<f32>,
    /// `kv_dim` wide: `attn_k . normed`.
    k_raw: GpuBuffer<f32>,
    /// `kv_dim` wide: `attn_v . normed`, copied into each slot's cache row.
    v_raw: GpuBuffer<f32>,
    /// `q_dim` wide: the query half of `q_full`.
    attn_q: GpuBuffer<f32>,
    /// `q_dim` wide: the gate half of `q_full`.
    attn_gate: GpuBuffer<f32>,
    /// `q_dim` wide: `attn_q` normed per head, then roped in place.
    q_normed: GpuBuffer<f32>,
    /// `kv_dim` wide: `k_raw` normed per head, copied into each slot's cache row.
    k_normed: GpuBuffer<f32>,
    /// `q_dim` wide: the gated attention output, the `attn_output` input.
    attn_out_in: GpuBuffer<f32>,
}

impl Qwen35MixerBufs {
    pub(super) fn new(executor: &CudaExecutor, d: Qwen35CudaDims, rows: usize) -> Result<Self> {
        let q_dim = (d.num_heads * d.attn_head_dim) as usize;
        let kv_dim = (d.num_kv_heads * d.attn_head_dim) as usize;
        let zeros = |width: usize| Qwen35CudaModel::zeros(executor, rows * width);
        Ok(Self {
            normed: zeros(d.hidden_dim as usize)?,
            out: zeros(d.hidden_dim as usize)?,
            conv_in: zeros(d.conv_dim as usize)?,
            alpha_raw: zeros(d.num_v_heads as usize)?,
            beta_raw: zeros(d.num_v_heads as usize)?,
            conv_out: zeros(d.conv_dim as usize)?,
            dt: zeros(d.num_v_heads as usize)?,
            beta: zeros(d.num_v_heads as usize)?,
            out_h: zeros(d.v_dim as usize)?,
            gate: zeros(d.v_dim as usize)?,
            ssm_out_in: zeros(d.v_dim as usize)?,
            q_full: zeros(q_dim * 2)?,
            k_raw: zeros(kv_dim)?,
            v_raw: zeros(kv_dim)?,
            attn_q: zeros(q_dim)?,
            attn_gate: zeros(q_dim)?,
            q_normed: zeros(q_dim)?,
            k_normed: zeros(kv_dim)?,
            attn_out_in: zeros(q_dim)?,
        })
    }
}

/// `w . x` for m rows of `x` into m rows of `y`, the weights read once.
fn project(
    ex: &mut CudaExecutor,
    w: &CudaQuantWeight,
    x: &GpuBuffer<f32>,
    y: &GpuBuffer<f32>,
    m: usize,
) -> std::result::Result<(), GpuError> {
    ex.batched_gemv_dispatch(w.qtype, w.ptr, x, y, m as u32, w.n, w.k)
}

/// Run `f` on row `row` of each of the `N` buffers, `widths[j]` wide.
fn with_rows<const N: usize, R>(
    bufs: [&GpuBuffer<f32>; N],
    row: usize,
    widths: [u32; N],
    f: impl FnOnce(&[GpuBuffer<f32>; N]) -> R,
) -> R {
    let views: [GpuBuffer<f32>; N] = std::array::from_fn(|j| row_view(bufs[j], row, widths[j]));
    let out = f(&views);
    for view in views {
        std::mem::forget(view);
    }
    out
}

/// `rms_norm(hidden, norm)` for each of the m rows of `b.hidden`, in one launch.
fn norm_rows(
    ex: &mut CudaExecutor,
    b: &Qwen35BatchBufs,
    norm: &GpuBuffer<f32>,
    d: Qwen35CudaDims,
    m: usize,
) -> std::result::Result<(), GpuError> {
    ex.rmsnorm_rows_into(
        &b.hidden,
        norm,
        &b.mixer.normed,
        d.hidden_dim,
        m as u32,
        d.eps,
    )
}

impl Qwen35CudaModel<'_> {
    /// [`Self::deltanet_mixer_inner`] for every slot of `b.hidden`. The conv
    /// window and the recurrent state are the slot's own, so the conv1d and
    /// the delta rule run per slot; the L2 norms, the gates and the gated
    /// norm hold no state and run once over the m rows, each with the row
    /// twin of its single-row kernel, which matches it op for op.
    pub(super) fn batched_deltanet_mixer(
        &mut self,
        b: &Qwen35BatchBufs,
        il: usize,
        states: &[Qwen35CudaState],
    ) -> std::result::Result<(), GpuError> {
        let d = self.dims;
        let m = states.len();
        let rows = m as u32;
        let CudaLayer::DeltaNet(w) = &self.layers[il] else {
            unreachable!("the caller matched a DeltaNet layer")
        };
        let x = &b.mixer;
        let ex = &mut self.executor;
        norm_rows(ex, b, &w.attn_norm, d, m)?;
        for (proj, out) in [
            (&w.attn_qkv, &x.conv_in),
            (&w.ssm_alpha, &x.alpha_raw),
            (&w.ssm_beta, &x.beta_raw),
            (&w.attn_gate, &x.gate),
        ] {
            project(ex, proj, &x.normed, out, m)?;
        }
        // Slot i's conv window and recurrent state at this layer, from the table
        // the step uploaded.
        let table = |kind: usize| b.state_table.as_ptr() + ((il * 2 + kind) * m * 8) as u64;
        ex.qwen35_conv1d_slots(
            x.conv_in.as_ptr(),
            table(0),
            w.conv1d_weight.as_ptr(),
            x.conv_out.as_ptr(),
            d.conv_dim,
            d.conv_kernel,
            d.conv_dim,
            rows,
        )?;
        // q then k of every row: each `num_k_heads` heads, `conv_dim` apart.
        for offset in [0, d.k_dim] {
            ex.qwen35_l2_norm_rows(
                x.conv_out.as_ptr() + u64::from(offset) * 4,
                d.head_k_dim,
                d.num_k_heads,
                d.eps,
                d.conv_dim,
                rows,
            )?;
        }
        ex.qwen35_gates_rows(
            x.alpha_raw.as_ptr(),
            w.ssm_dt_bias.as_ptr(),
            w.ssm_a.as_ptr(),
            x.beta_raw.as_ptr(),
            x.dt.as_ptr(),
            x.beta.as_ptr(),
            d.num_v_heads,
            rows,
        )?;
        let qkv = x.conv_out.as_ptr();
        ex.qwen35_delta_rule_slots(
            [
                qkv,
                qkv + u64::from(d.k_dim) * 4,
                qkv + u64::from(d.k_dim * 2) * 4,
            ],
            x.beta.as_ptr(),
            x.dt.as_ptr(),
            table(1),
            x.out_h.as_ptr(),
            [d.num_k_heads, d.head_k_dim, d.num_v_heads, d.head_v_dim],
            d.conv_dim,
            rows,
        )?;
        // Every head of every row: the norm weight is shared across heads.
        ex.gdn_gated_rmsnorm_into(
            &x.out_h,
            &x.gate,
            &w.ssm_norm_weight,
            &x.ssm_out_in,
            d.head_v_dim,
            rows * d.num_v_heads,
            d.eps,
        )?;
        project(ex, &w.ssm_out, &x.ssm_out_in, &x.out, m)?;
        ex.residual_add_into(&b.hidden, &x.out, &b.hidden, rows * d.hidden_dim)
    }

    /// [`Self::attention_mixer_inner`] (ungraphed) for every slot of
    /// `b.hidden`, slot `i` at `positions[i]`.
    pub(super) fn batched_attention_mixer(
        &mut self,
        b: &Qwen35BatchBufs,
        il: usize,
        states: &[Qwen35CudaState],
        positions: &[usize],
    ) -> std::result::Result<(), GpuError> {
        let d = self.dims;
        let m = states.len();
        let CudaLayer::Attention(w) = &self.layers[il] else {
            unreachable!("the caller matched an attention layer")
        };
        let x = &b.mixer;
        let ex = &mut self.executor;
        let q_dim = d.num_heads * d.attn_head_dim;
        let kv_dim = d.num_kv_heads * d.attn_head_dim;
        norm_rows(ex, b, &w.attn_norm, d, m)?;
        for (proj, out) in [
            (&w.attn_q, &x.q_full),
            (&w.attn_k, &x.k_raw),
            (&w.attn_v, &x.v_raw),
        ] {
            project(ex, proj, &x.normed, out, m)?;
        }
        // The split and both per-head norms treat each head alone, so the m
        // rows run as m times the heads in one launch each.
        let rows = m as u32;
        ex.gdn_split_interleaved_into(
            &x.q_full,
            &x.attn_q,
            &x.attn_gate,
            rows * d.num_heads,
            d.attn_head_dim,
        )?;
        for (input, norm, output, heads) in [
            (&x.attn_q, &w.attn_q_norm, &x.q_normed, d.num_heads),
            (&x.k_raw, &w.attn_k_norm, &x.k_normed, d.num_kv_heads),
        ] {
            ex.per_head_rmsnorm_into(input, norm, output, d.attn_head_dim, rows * heads, d.eps)?;
        }
        // Every row at its own position, before K goes into the caches.
        for (buf, heads) in [(&x.q_normed, d.num_heads), (&x.k_normed, d.num_kv_heads)] {
            ex.gdn_partial_neox_rope_slots_into(
                buf,
                &b.positions,
                heads,
                d.attn_head_dim,
                d.n_rot,
                d.theta_scale,
                rows,
            )?;
        }
        if CudaExecutor::gdn_decode_attention_has_slots(d.attn_head_dim) {
            // Every slot's K/V row goes into its cache, and all m attend, in
            // two pairs of launches through this layer's entries of the state
            // table.
            let max_pos = positions.iter().copied().max().unwrap_or(0);
            let entry = |col: usize| b.state_table.as_ptr() + ((il * 2 + col) * m * 8) as u64;
            ex.gdn_kv_row_scatter_slots_into(&x.k_normed, entry(0), &b.positions, kv_dim, rows)?;
            ex.gdn_kv_row_scatter_slots_into(&x.v_raw, entry(1), &b.positions, kv_dim, rows)?;
            ex.gdn_decode_attention_slots_into(
                &x.q_normed,
                entry(0),
                entry(1),
                &x.attn_out_in,
                &b.positions,
                u32::try_from(max_pos).unwrap_or(u32::MAX),
                rows,
                (d.num_heads, d.num_kv_heads, d.attn_head_dim),
            )?;
        } else {
            Self::attend_each_slot(ex, x, states, positions, il, (q_dim, kv_dim), d)?;
        }
        // Elementwise, so one launch for all rows.
        ex.gdn_sigmoid_gate_into(&x.attn_out_in, &x.attn_gate, rows * q_dim)?;
        project(ex, &w.attn_output, &x.attn_out_in, &x.out, m)?;
        ex.residual_add_into(&b.hidden, &x.out, &b.hidden, m as u32 * d.hidden_dim)
    }

    /// The attention of [`Self::batched_attention_mixer`] one slot at a time,
    /// for a head dim the slots kernels do not take: each slot's K/V row is
    /// copied into its cache, then it attends alone.
    fn attend_each_slot(
        ex: &mut CudaExecutor,
        x: &Qwen35MixerBufs,
        states: &[Qwen35CudaState],
        positions: &[usize],
        il: usize,
        (q_dim, kv_dim): (u32, u32),
        d: Qwen35CudaDims,
    ) -> std::result::Result<(), GpuError> {
        let bufs = [&x.q_normed, &x.k_normed, &x.v_raw, &x.attn_out_in];
        let widths = [q_dim, kv_dim, kv_dim, q_dim];
        for (i, (state, &position)) in states.iter().zip(positions).enumerate() {
            let (k_cache, v_cache) = state.kv[il]
                .as_ref()
                .expect("an attention layer owns a KV cache");
            let row_off = u32::try_from(position).unwrap_or(0) * kv_dim;
            let pos32 = u32::try_from(position).unwrap_or(u32::MAX);
            let mut k_row = Self::view(k_cache, row_off, kv_dim);
            let mut v_row = Self::view(v_cache, row_off, kv_dim);
            let run = with_rows(
                bufs,
                i,
                widths,
                |[q_normed, k_normed, v_raw, attn_out_in]| {
                    // The single-row mixer writes the normed K and the `attn_v`
                    // GEMV straight into the cache row; here the batch rows are
                    // copied in, on the stream the kernels around them are
                    // enqueued on.
                    // SAFETY: all four are live device buffers of `kv_dim`
                    // elements (the copy checks the lengths), and none is freed
                    // before the stream runs the copies: `k_normed` and `v_raw`
                    // are batch buffers, `k_row` and `v_row` the slot's KV cache.
                    unsafe {
                        k_row.copy_from_buffer_async(k_normed, ex.exec_stream())?;
                        v_row.copy_from_buffer_async(v_raw, ex.exec_stream())
                    }?;
                    ex.gdn_decode_attention_into(
                        q_normed,
                        k_cache,
                        v_cache,
                        attn_out_in,
                        d.num_heads,
                        d.num_kv_heads,
                        d.attn_head_dim,
                        pos32 + 1,
                    )
                },
            );
            std::mem::forget(k_row);
            std::mem::forget(v_row);
            run?;
        }
        Ok(())
    }
}
