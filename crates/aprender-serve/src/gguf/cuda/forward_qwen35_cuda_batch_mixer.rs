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
            gate: zeros(d.v_dim as usize)?,
            ssm_out_in: zeros(d.v_dim as usize)?,
            q_full: zeros(q_dim * 2)?,
            k_raw: zeros(kv_dim)?,
            v_raw: zeros(kv_dim)?,
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

/// `rms_norm(hidden, norm)` for each of the m rows of `b.hidden`.
fn norm_rows(
    ex: &mut CudaExecutor,
    b: &Qwen35BatchBufs,
    norm: &GpuBuffer<f32>,
    d: Qwen35CudaDims,
    m: usize,
) -> std::result::Result<(), GpuError> {
    for i in 0..m {
        with_rows(
            [&b.hidden, &b.mixer.normed],
            i,
            [d.hidden_dim; 2],
            |[h, n]| ex.rmsnorm_into(h, norm, n, d.hidden_dim, d.eps),
        )?;
    }
    Ok(())
}

impl Qwen35CudaModel<'_> {
    /// [`Self::deltanet_mixer_inner`] for every slot of `b.hidden`.
    pub(super) fn batched_deltanet_mixer(
        &mut self,
        b: &Qwen35BatchBufs,
        il: usize,
        states: &[Qwen35CudaState],
    ) -> std::result::Result<(), GpuError> {
        let d = self.dims;
        let m = states.len();
        let CudaLayer::DeltaNet(w) = &self.layers[il] else {
            unreachable!("the caller matched a DeltaNet layer")
        };
        let s = &self.scratch;
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
        let bufs = [
            &x.conv_in,
            &x.alpha_raw,
            &x.beta_raw,
            &x.gate,
            &x.ssm_out_in,
        ];
        let widths = [d.conv_dim, d.num_v_heads, d.num_v_heads, d.v_dim, d.v_dim];
        for (i, state) in states.iter().enumerate() {
            with_rows(
                bufs,
                i,
                widths,
                |[conv_in, alpha_raw, beta_raw, gate, ssm_out_in]| {
                    ex.gdn_causal_conv1d_silu_into(
                        conv_in,
                        &state.conv[il],
                        &w.conv1d_weight,
                        &s.conv_out,
                        d.conv_dim,
                        d.conv_kernel,
                    )?;
                    let q_view = Self::view(&s.conv_out, 0, d.k_dim);
                    let k_view = Self::view(&s.conv_out, d.k_dim, d.k_dim);
                    let v_view = Self::view(&s.conv_out, d.k_dim * 2, d.v_dim);
                    let run = (|| {
                        ex.gdn_per_head_l2_norm_into(&q_view, d.head_k_dim, d.num_k_heads, d.eps)?;
                        ex.gdn_per_head_l2_norm_into(&k_view, d.head_k_dim, d.num_k_heads, d.eps)?;
                        ex.gdn_gates_into(
                            alpha_raw,
                            &w.ssm_dt_bias,
                            &w.ssm_a,
                            beta_raw,
                            &s.dt,
                            &s.beta,
                            d.num_v_heads,
                        )?;
                        ex.gdn_delta_rule_into(
                            &q_view,
                            &k_view,
                            &v_view,
                            &s.beta,
                            &s.dt,
                            &state.ssm[il],
                            &s.out_h,
                            d.num_k_heads,
                            d.head_k_dim,
                            d.num_v_heads,
                            d.head_v_dim,
                        )
                    })();
                    std::mem::forget(q_view);
                    std::mem::forget(k_view);
                    std::mem::forget(v_view);
                    run?;
                    ex.gdn_gated_rmsnorm_into(
                        &s.out_h,
                        gate,
                        &w.ssm_norm_weight,
                        ssm_out_in,
                        d.head_v_dim,
                        d.num_v_heads,
                        d.eps,
                    )
                },
            )?;
        }
        project(ex, &w.ssm_out, &x.ssm_out_in, &x.out, m)?;
        ex.residual_add_into(&b.hidden, &x.out, &b.hidden, m as u32 * d.hidden_dim)
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
        let a = &self.attn_scratch;
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
        let bufs = [&x.q_full, &x.k_raw, &x.v_raw, &x.attn_out_in];
        let widths = [q_dim * 2, kv_dim, kv_dim, q_dim];
        for (i, (state, &position)) in states.iter().zip(positions).enumerate() {
            let (k_cache, v_cache) = state.kv[il]
                .as_ref()
                .expect("an attention layer owns a KV cache");
            let row_off = u32::try_from(position).unwrap_or(0) * kv_dim;
            let pos32 = u32::try_from(position).unwrap_or(u32::MAX);
            let k_row = Self::view(k_cache, row_off, kv_dim);
            let mut v_row = Self::view(v_cache, row_off, kv_dim);
            let run = with_rows(bufs, i, widths, |[q_full, k_raw, v_raw, attn_out_in]| {
                // The single-row mixer's `attn_v` GEMV writes this row in
                // place; here the batched GEMV's row is copied in, on the
                // stream the kernels around it are enqueued on.
                // SAFETY: both are live device buffers of `kv_dim` elements
                // (the copy checks the lengths), and neither is freed before
                // the stream runs the copy: `v_raw` is the batch buffer and
                // `v_row` is the slot's KV cache.
                unsafe { v_row.copy_from_buffer_async(v_raw, ex.exec_stream()) }?;
                ex.gdn_split_interleaved_into(q_full, &a.q, &a.gate, d.num_heads, d.attn_head_dim)?;
                ex.per_head_rmsnorm_into(
                    &a.q,
                    &w.attn_q_norm,
                    &a.q_normed,
                    d.attn_head_dim,
                    d.num_heads,
                    d.eps,
                )?;
                ex.per_head_rmsnorm_into(
                    k_raw,
                    &w.attn_k_norm,
                    &k_row,
                    d.attn_head_dim,
                    d.num_kv_heads,
                    d.eps,
                )?;
                for (rows, heads) in [(&a.q_normed, d.num_heads), (&k_row, d.num_kv_heads)] {
                    ex.gdn_partial_neox_rope_into(
                        rows,
                        heads,
                        d.attn_head_dim,
                        d.n_rot,
                        pos32,
                        d.theta_scale,
                    )?;
                }
                ex.gdn_decode_attention_into(
                    &a.q_normed,
                    k_cache,
                    v_cache,
                    attn_out_in,
                    d.num_heads,
                    d.num_kv_heads,
                    d.attn_head_dim,
                    pos32 + 1,
                )?;
                ex.gdn_sigmoid_gate_into(attn_out_in, &a.gate, q_dim)
            });
            std::mem::forget(k_row);
            std::mem::forget(v_row);
            run?;
        }
        project(ex, &w.attn_output, &x.attn_out_in, &x.out, m)?;
        ex.residual_add_into(&b.hidden, &x.out, &b.hidden, m as u32 * d.hidden_dim)
    }
}
