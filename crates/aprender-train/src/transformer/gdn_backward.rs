//! T1-R3 step 1 (#4000, contract `qwen35-train-gdn-v1`, `FALSIFY-QTG-003/004`): the
//! reverse-mode gradient of [`gated_delta_scan`](super::gated_delta_scan).
//!
//! The backward walks the sequence from the last position to the first, reading the
//! state *before* each step from the forward's `history` (or `s0` at `t = 0`). For one
//! value head and one step, with `A = e^g · S_prev`:
//!
//! ```text
//! pred = A k          δ = β (v − pred)          S = A + δ kᵀ          o = scale · S q
//! ```
//!
//! (`S` is `[hv × hk]`, row `j` = value channel.) Each line is reversed in turn, and
//! `dS_prev = e^g · dA` carries the state gradient to the step before. A value head
//! reads key/query head `h % num_k_heads`, so `dq`/`dk` sum over every value head that
//! shares a key head.

use super::gdn::{
    causal_conv1d_seq, gated_delta_scan, l2_norm_heads, project, sigmoid, silu, softplus, GdnDims,
    GdnFloat, GdnWeights,
};

/// The gradients [`gated_delta_scan_backward`] returns, shaped like the scan's inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct GdnScanGrads<T = f32> {
    /// `∂L/∂q`, `[seq_len × k_dim]`.
    pub dq: Vec<T>,
    /// `∂L/∂k`, `[seq_len × k_dim]`.
    pub dk: Vec<T>,
    /// `∂L/∂v`, `[seq_len × v_dim]`.
    pub dv: Vec<T>,
    /// `∂L/∂β`, `[seq_len × num_v_heads]`.
    pub dbeta: Vec<T>,
    /// `∂L/∂g`, `[seq_len × num_v_heads]`.
    pub dg: Vec<T>,
    /// `∂L/∂S₀`, `[state_len]` — the gradient chunked training hands the previous chunk.
    pub ds0: Vec<T>,
}

/// The inputs of one step of one value head, and the gradients it writes.
struct HeadStep<'a, T> {
    s_prev: &'a [T],
    s_new: &'a [T],
    q: &'a [T],
    k: &'a [T],
    v: &'a [T],
    beta: T,
    decay: T,
    scale: T,
}

/// Reverse one step of one value head. `ds` holds `∂L/∂S` after the step on entry and
/// `∂L/∂S_prev` on return; `dq`/`dk` are accumulated into, the rest are written.
#[allow(clippy::too_many_arguments)]
fn step_head_backward<T: GdnFloat>(
    st: &HeadStep<'_, T>,
    d_o: &[T],
    ds: &mut [T],
    dq: &mut [T],
    dk: &mut [T],
    dv: &mut [T],
    dbeta: &mut T,
    dg: &mut T,
) {
    let hk = st.k.len();
    let (mut db, mut dgs) = (T::ZERO, T::ZERO);
    for j in 0..st.v.len() {
        let rows = j * hk..(j + 1) * hk;
        let (prev, new, dsr) = (&st.s_prev[rows.clone()], &st.s_new[rows.clone()], &mut ds[rows]);
        // o_j = scale · Σ_i S[j,i] q_i
        let go = d_o[j] * st.scale;
        for i in 0..hk {
            dq[i] += go * new[i];
            dsr[i] += go * st.q[i];
        }
        // Recompute this row's forward: A = decay · S_prev, pred = A k, δ = β (v − pred).
        let pred: T = prev.iter().zip(st.k).map(|(&a, &b)| st.decay * a * b).sum();
        let err = st.v[j] - pred;
        let delta = st.beta * err;
        // S = A + δ kᵀ  →  dA = dS,  dk += dS δ,  dδ = dS · k
        let mut d_delta = T::ZERO;
        for i in 0..hk {
            dk[i] += dsr[i] * delta;
            d_delta += dsr[i] * st.k[i];
        }
        // δ = β (v − pred)
        dv[j] = st.beta * d_delta;
        db += d_delta * err;
        let d_pred = T::ZERO - st.beta * d_delta;
        // pred = A k  →  dA += d_pred kᵀ,  dk += d_pred · A
        for i in 0..hk {
            let a = st.decay * prev[i];
            dk[i] += d_pred * a;
            let da = dsr[i] + d_pred * st.k[i];
            // A = e^g · S_prev  →  dg += Σ A·dA,  dS_prev = e^g · dA
            dgs += a * da;
            dsr[i] = st.decay * da;
        }
    }
    *dbeta = db;
    *dg = dgs;
}

/// Reverse-mode gradient of [`gated_delta_scan`](super::gated_delta_scan).
///
/// `history` is the forward's `keep_history` output and `s0` the state it started from
/// (zero if `None`). `d_out` is `∂L/∂out`; `d_final` is `∂L/∂final_state` (zero if
/// `None` — a loss that reads only the outputs).
///
/// # Panics
/// If a length disagrees with `dims`, or `history` is not `seq_len × state_len`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_scan_backward<T: GdnFloat>(
    q: &[T],
    k: &[T],
    v: &[T],
    beta: &[T],
    g: &[T],
    dims: &GdnDims,
    s0: Option<&[T]>,
    history: &[T],
    d_out: &[T],
    d_final: Option<&[T]>,
) -> GdnScanGrads<T> {
    let (nk, hk, nv, hv) = (dims.num_k_heads, dims.head_k_dim, dims.num_v_heads, dims.head_v_dim);
    let (kd, vd, sl) = (dims.k_dim(), dims.v_dim(), dims.state_len());
    let seq_len = v.len() / vd;
    assert_eq!(v.len(), seq_len * vd);
    assert_eq!((q.len(), k.len(), d_out.len()), (seq_len * kd, seq_len * kd, seq_len * vd));
    assert_eq!((beta.len(), g.len()), (seq_len * nv, seq_len * nv));
    assert_eq!(history.len(), seq_len * sl, "history must be the forward's keep_history");
    let zero = vec![T::ZERO; sl];
    let s0 = s0.unwrap_or(&zero);
    assert_eq!(s0.len(), sl);
    let mut ds = d_final.map_or_else(|| zero.clone(), <[T]>::to_vec);
    assert_eq!(ds.len(), sl);
    let scale = T::inv_sqrt(hk);
    let mut gr = GdnScanGrads {
        dq: vec![T::ZERO; q.len()],
        dk: vec![T::ZERO; k.len()],
        dv: vec![T::ZERO; v.len()],
        dbeta: vec![T::ZERO; beta.len()],
        dg: vec![T::ZERO; g.len()],
        ds0: Vec::new(),
    };
    for t in (0..seq_len).rev() {
        let s_new = &history[t * sl..(t + 1) * sl];
        let s_prev = if t == 0 { s0 } else { &history[(t - 1) * sl..t * sl] };
        for h in 0..nv {
            let kh = h % nk;
            let qk = t * kd + kh * hk..t * kd + (kh + 1) * hk;
            let vs = t * vd + h * hv..t * vd + (h + 1) * hv;
            let blk = h * hv * hk..(h + 1) * hv * hk;
            let st = HeadStep {
                s_prev: &s_prev[blk.clone()],
                s_new: &s_new[blk.clone()],
                q: &q[qk.clone()],
                k: &k[qk.clone()],
                v: &v[vs.clone()],
                beta: beta[t * nv + h],
                decay: g[t * nv + h].exp(),
                scale,
            };
            let (mut db, mut dgh) = (T::ZERO, T::ZERO);
            step_head_backward(
                &st,
                &d_out[vs.clone()],
                &mut ds[blk],
                &mut gr.dq[qk.clone()],
                &mut gr.dk[qk],
                &mut gr.dv[vs],
                &mut db,
                &mut dgh,
            );
            gr.dbeta[t * nv + h] = db;
            gr.dg[t * nv + h] = dgh;
        }
    }
    gr.ds0 = ds;
    gr
}

/// `∂L/∂W` for every weight of a GDN mixer, shaped like [`GdnWeights`].
#[derive(Debug, Clone, PartialEq)]
pub struct GdnWeightGrads<T = f32> {
    /// `∂L/∂attn_qkv`.
    pub qkv: Vec<T>,
    /// `∂L/∂attn_gate`.
    pub gate: Vec<T>,
    /// `∂L/∂ssm_alpha`.
    pub alpha: Vec<T>,
    /// `∂L/∂ssm_beta`.
    pub beta: Vec<T>,
    /// `∂L/∂ssm_a`.
    pub a: Vec<T>,
    /// `∂L/∂ssm_dt_bias`.
    pub dt_bias: Vec<T>,
    /// `∂L/∂ssm_conv1d`.
    pub conv: Vec<T>,
    /// `∂L/∂ssm_norm`.
    pub norm: Vec<T>,
    /// `∂L/∂ssm_out`.
    pub out: Vec<T>,
}

/// Reverse `y = x · Wᵀ`: adds `∂L/∂x` into `dx` and returns `∂L/∂W`.
pub(super) fn project_backward<T: GdnFloat>(
    x: &[T],
    w: &[T],
    dy: &[T],
    d_in: usize,
    d_out: usize,
    dx: &mut [T],
) -> Vec<T> {
    let mut dw = vec![T::ZERO; w.len()];
    for ((xr, dyr), dxr) in
        x.chunks_exact(d_in).zip(dy.chunks_exact(d_out)).zip(dx.chunks_exact_mut(d_in))
    {
        for ((&g, wr), dwr) in dyr.iter().zip(w.chunks_exact(d_in)).zip(dw.chunks_exact_mut(d_in)) {
            for i in 0..d_in {
                dxr[i] += g * wr[i];
                dwr[i] += g * xr[i];
            }
        }
    }
    dw
}

/// `d silu(x) / dx = σ(x) (1 + x (1 − σ(x)))`.
pub(super) fn silu_grad<T: GdnFloat>(x: T) -> T {
    let s = sigmoid(x);
    s * (T::ONE + x * (T::ONE - s))
}

/// Reverse [`l2_norm_heads`] in place: `dy` (w.r.t. the normalised heads of `x`)
/// becomes `∂L/∂x`. `y = x·s`, `s = (Σx² + ε)^-½` ⇒ `dx = s·dy − s³·x·(x·dy)`.
fn l2_norm_heads_backward<T: GdnFloat>(x: &[T], dy: &mut [T], head_dim: usize, eps: T) {
    for (xh, dh) in x.chunks_exact(head_dim).zip(dy.chunks_exact_mut(head_dim)) {
        let s = T::ONE / (xh.iter().map(|&v| v * v).sum::<T>() + eps).sqrt();
        let xdy: T = xh.iter().zip(dh.iter()).map(|(&a, &b)| a * b).sum();
        for (d, &xi) in dh.iter_mut().zip(xh) {
            *d = s * *d - s * s * s * xi * xdy;
        }
    }
}

/// Reverse the gated `RMSNorm` `y = x·r·w·silu(z)` per `hv`-wide head. Returns
/// `(∂L/∂x, ∂L/∂z)` and adds `∂L/∂w` into `dw`.
fn gated_rmsnorm_backward<T: GdnFloat>(
    x: &[T],
    z: &[T],
    w: &[T],
    dy: &[T],
    eps: T,
    dw: &mut [T],
) -> (Vec<T>, Vec<T>) {
    let hv = w.len();
    let (mut dx, mut dz) = (vec![T::ZERO; x.len()], vec![T::ZERO; z.len()]);
    let chunks = x.chunks_exact(hv).zip(z.chunks_exact(hv)).zip(dy.chunks_exact(hv));
    for (((xh, zh), dh), (dxh, dzh)) in
        chunks.zip(dx.chunks_exact_mut(hv).zip(dz.chunks_exact_mut(hv)))
    {
        let r = T::ONE / (xh.iter().map(|&v| v * v).sum::<T>() / T::from_usize(hv) + eps).sqrt();
        // u = ∂L/∂(x·r) = dy·w·silu(z)
        let mut ux = T::ZERO;
        for i in 0..hv {
            let (xi, zi, di) = (xh[i], zh[i], dh[i]);
            dw[i] += di * xi * r * silu(zi);
            dzh[i] = di * xi * r * w[i] * silu_grad(zi);
            let u = di * w[i] * silu(zi);
            dxh[i] = r * u;
            ux += u * xi;
        }
        // r = (mean x² + ε)^-½ ⇒ ∂r/∂x_i = −r³ x_i / hv
        let c = r * r * r * ux / T::from_usize(hv);
        for i in 0..hv {
            dxh[i] = dxh[i] - c * xh[i];
        }
    }
    (dx, dz)
}

/// Reverse causal depthwise conv1d: `y[t,c] = Σ_lag w[c, K−1−lag] · x[t−lag, c]`.
/// Returns `(∂L/∂x, ∂L/∂w)`.
fn causal_conv1d_backward<T: GdnFloat>(
    x: &[T],
    w: &[T],
    dy: &[T],
    channels: usize,
    kernel: usize,
) -> (Vec<T>, Vec<T>) {
    let seq_len = x.len() / channels;
    let (mut dx, mut dw) = (vec![T::ZERO; x.len()], vec![T::ZERO; w.len()]);
    for t in 0..seq_len {
        for c in 0..channels {
            let g = dy[t * channels + c];
            for lag in 0..kernel.min(t + 1) {
                let (tap, src) = (c * kernel + kernel - 1 - lag, (t - lag) * channels + c);
                dx[src] += w[tap] * g;
                dw[tap] += x[src] * g;
            }
        }
    }
    (dx, dw)
}

/// Reverse-mode gradient of [`gdn_mixer_forward`](super::gdn_mixer_forward) from an
/// empty state. The forward is recomputed here (nothing is cached between the calls),
/// keeping the scan's history. Returns `(∂L/∂normed, ∂L/∂W)` for `d_y = ∂L/∂output`.
///
/// # Panics
/// If a weight's length disagrees with `dims`, or `d_y` is not `normed`'s shape.
#[must_use]
pub fn gdn_mixer_backward<T: GdnFloat>(
    normed: &[T],
    w: &GdnWeights<'_, T>,
    dims: &GdnDims,
    d_y: &[T],
) -> (Vec<T>, GdnWeightGrads<T>) {
    let (hidden, nv, kd, vd, cd, hk) = (
        dims.hidden_dim,
        dims.num_v_heads,
        dims.k_dim(),
        dims.v_dim(),
        dims.conv_dim(),
        dims.head_k_dim,
    );
    assert_eq!(d_y.len(), normed.len());
    let eps = T::from_f32(dims.eps);
    // ── forward, keeping what the backward reads ──
    let conv_in = project(normed, w.qkv, hidden, cd);
    let pre = causal_conv1d_seq(&conv_in, w.conv, cd, dims.conv_kernel);
    let seq_len = pre.len() / cd;
    let (mut q_raw, mut k_raw, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for row in pre.chunks_exact(cd) {
        q_raw.extend(row[..kd].iter().map(|&x| silu(x)));
        k_raw.extend(row[kd..2 * kd].iter().map(|&x| silu(x)));
        v.extend(row[2 * kd..].iter().map(|&x| silu(x)));
    }
    let (mut q, mut k) = (q_raw.clone(), k_raw.clone());
    l2_norm_heads(&mut q, hk, eps);
    l2_norm_heads(&mut k, hk, eps);
    let a_pre: Vec<T> = project(normed, w.alpha, hidden, nv)
        .iter()
        .enumerate()
        .map(|(i, &x)| x + w.dt_bias[i % nv])
        .collect();
    let sp: Vec<T> = a_pre.iter().map(|&x| softplus(x)).collect();
    let g: Vec<T> = sp.iter().enumerate().map(|(i, &s)| s * w.a[i % nv]).collect();
    let beta: Vec<T> = project(normed, w.beta, hidden, nv).iter().map(|&x| sigmoid(x)).collect();
    let z = project(normed, w.gate, hidden, vd);
    let scan = gated_delta_scan(&q, &k, &v, &beta, &g, dims, None, true);
    let (mut yn, mut dnorm) = (Vec::with_capacity(scan.out.len()), vec![T::ZERO; w.norm.len()]);
    {
        // y = gated_rmsnorm(scan.out, z) — recomputed only as ssm_out's input.
        let hv = w.norm.len();
        for (xh, zh) in scan.out.chunks_exact(hv).zip(z.chunks_exact(hv)) {
            let r =
                T::ONE / (xh.iter().map(|&v| v * v).sum::<T>() / T::from_usize(hv) + eps).sqrt();
            yn.extend((0..hv).map(|i| xh[i] * r * w.norm[i] * silu(zh[i])));
        }
    }
    // ── backward ──
    let mut d_normed = vec![T::ZERO; normed.len()];
    let mut d_yn = vec![T::ZERO; yn.len()];
    let d_out_w = project_backward(&yn, w.out, d_y, vd, hidden, &mut d_yn);
    let (d_o, dz) = gated_rmsnorm_backward(&scan.out, &z, w.norm, &d_yn, eps, &mut dnorm);
    let sg =
        gated_delta_scan_backward(&q, &k, &v, &beta, &g, dims, None, &scan.history, &d_o, None);
    let d_gate = project_backward(normed, w.gate, &dz, hidden, vd, &mut d_normed);
    // beta = σ(b_pre)
    let d_bpre: Vec<T> = sg.dbeta.iter().zip(&beta).map(|(&d, &b)| d * b * (T::ONE - b)).collect();
    let d_beta_w = project_backward(normed, w.beta, &d_bpre, hidden, nv, &mut d_normed);
    // g = softplus(a_pre) · a, a_pre = alpha·x + dt_bias; softplus' = σ below the cut-over.
    let (mut d_a, mut d_dt) = (vec![T::ZERO; nv], vec![T::ZERO; nv]);
    let mut d_apre = vec![T::ZERO; g.len()];
    for i in 0..g.len() {
        let h = i % nv;
        d_a[h] += sg.dg[i] * sp[i];
        let slope = if a_pre[i] > T::from_f32(20.0) { T::ONE } else { sigmoid(a_pre[i]) };
        d_apre[i] = sg.dg[i] * w.a[h] * slope;
        d_dt[h] += d_apre[i];
    }
    let d_alpha = project_backward(normed, w.alpha, &d_apre, hidden, nv, &mut d_normed);
    // q, k = l2norm(silu(pre)); v = silu(pre)
    let (mut dq, mut dk) = (sg.dq, sg.dk);
    l2_norm_heads_backward(&q_raw, &mut dq, hk, eps);
    l2_norm_heads_backward(&k_raw, &mut dk, hk, eps);
    let mut d_pre = Vec::with_capacity(pre.len());
    for t in 0..seq_len {
        d_pre.extend_from_slice(&dq[t * kd..(t + 1) * kd]);
        d_pre.extend_from_slice(&dk[t * kd..(t + 1) * kd]);
        d_pre.extend_from_slice(&sg.dv[t * vd..(t + 1) * vd]);
    }
    for (d, &p) in d_pre.iter_mut().zip(&pre) {
        *d *= silu_grad(p);
    }
    let (d_conv_in, d_conv) =
        causal_conv1d_backward(&conv_in, w.conv, &d_pre, cd, dims.conv_kernel);
    let d_qkv = project_backward(normed, w.qkv, &d_conv_in, hidden, cd, &mut d_normed);
    let grads = GdnWeightGrads {
        qkv: d_qkv,
        gate: d_gate,
        alpha: d_alpha,
        beta: d_beta_w,
        a: d_a,
        dt_bias: d_dt,
        conv: d_conv,
        norm: dnorm,
        out: d_out_w,
    };
    (d_normed, grads)
}

#[cfg(test)]
#[path = "gdn_backward_tests.rs"]
mod tests;
