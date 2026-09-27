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

use super::gdn::{GdnDims, GdnFloat};

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

#[cfg(test)]
#[path = "gdn_backward_tests.rs"]
mod tests;
