//! T1-R3 step 3 (#4000, contract `qwen35-train-gdn-v1`, `FALSIFY-QTG-003`): the
//! reverse-mode gradients of the gated full-attention mixer and of the whole Qwen3.5
//! block (both mixer kinds), completing the layer backward begun in [`super::gdn_backward`].
//!
//! Each backward recomputes its forward (nothing is cached across calls) and returns
//! `(∂L/∂input, ∂L/∂W)`. Every function is generic over [`GdnFloat`] so the gradcheck
//! differentiates the same code training runs, in f64.

use super::gdn::{gdn_mixer_forward, project, sigmoid, silu, GdnFloat};
use super::gdn_backward::{gdn_mixer_backward, project_backward, silu_grad, GdnWeightGrads};
use super::qwen35_layer::{
    gated_attn_forward, rms_norm_chunks, rope_seq, GatedAttnDims, GatedAttnWeights, Qwen35Mixer,
    SwiGluWeights,
};

/// `∂L/∂W` for a gated full-attention mixer, shaped like [`GatedAttnWeights`].
#[derive(Debug, Clone, PartialEq)]
pub struct GatedAttnGrads<T = f32> {
    /// `∂L/∂attn_q` (per head `[q_h | gate_h]` rows, as the weight).
    pub q: Vec<T>,
    /// `∂L/∂attn_k`.
    pub k: Vec<T>,
    /// `∂L/∂attn_v`.
    pub v: Vec<T>,
    /// `∂L/∂attn_q_norm`.
    pub q_norm: Vec<T>,
    /// `∂L/∂attn_k_norm`.
    pub k_norm: Vec<T>,
    /// `∂L/∂attn_output`.
    pub out: Vec<T>,
}

/// The mixer part of [`Qwen35BlockGrads`].
#[derive(Debug, Clone, PartialEq)]
pub enum Qwen35MixerGrads<T = f32> {
    /// Gradients of a Gated `DeltaNet` mixer.
    Gdn(GdnWeightGrads<T>),
    /// Gradients of a gated full-attention mixer.
    Attention(GatedAttnGrads<T>),
}

/// `∂L/∂W` for one Qwen3.5 block.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35BlockGrads<T = f32> {
    /// `∂L/∂attn_norm`.
    pub attn_norm: Vec<T>,
    /// The mixer's weights.
    pub mixer: Qwen35MixerGrads<T>,
    /// `∂L/∂post_attention_norm`.
    pub post_norm: Vec<T>,
    /// `∂L/∂ffn_gate`.
    pub ffn_gate: Vec<T>,
    /// `∂L/∂ffn_up`.
    pub ffn_up: Vec<T>,
    /// `∂L/∂ffn_down`.
    pub ffn_down: Vec<T>,
}

/// Reverse [`rms_norm_chunks`]: `y = x·r·w` per `w.len()`-wide chunk, `r = (mean x² + ε)^-½`.
/// Returns `∂L/∂x` and adds `∂L/∂w` into `dw`.
pub(super) fn rms_norm_chunks_backward<T: GdnFloat>(
    x: &[T],
    w: &[T],
    dy: &[T],
    eps: T,
    dw: &mut [T],
) -> Vec<T> {
    let n = w.len();
    let mut dx = vec![T::ZERO; x.len()];
    for ((xc, dc), dxc) in x.chunks_exact(n).zip(dy.chunks_exact(n)).zip(dx.chunks_exact_mut(n)) {
        let r = T::ONE / (xc.iter().map(|&v| v * v).sum::<T>() / T::from_usize(n) + eps).sqrt();
        let mut ux = T::ZERO;
        for i in 0..n {
            dw[i] += dc[i] * xc[i] * r;
            let u = dc[i] * w[i];
            dxc[i] = r * u;
            ux += u * xc[i];
        }
        let c = r * r * r * ux / T::from_usize(n);
        for i in 0..n {
            dxc[i] = dxc[i] - c * xc[i];
        }
    }
    dx
}

/// Reverse-mode gradient of [`gated_attn_forward`] (causal, from position 0).
/// Returns `(∂L/∂normed, ∂L/∂W)` for `d_y = ∂L/∂output`.
///
/// # Panics
/// If a weight's length disagrees with `dims`, or `d_y` is not `normed`'s shape.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn gated_attn_backward<T: GdnFloat>(
    normed: &[T],
    w: &GatedAttnWeights<'_, T>,
    dims: &GatedAttnDims,
    d_y: &[T],
) -> (Vec<T>, GatedAttnGrads<T>) {
    let (hidden, hd, nh, nkv) = (dims.hidden_dim, dims.head_dim, dims.num_heads, dims.num_kv_heads);
    assert!(nkv > 0 && nh % nkv == 0, "{nh} query heads over {nkv} kv heads");
    assert_eq!(d_y.len(), normed.len());
    let (qd, kvd, group) = (dims.q_dim(), dims.kv_dim(), nh / nkv);
    let (eps, base) = (T::from_f32(dims.eps), T::from_f32(dims.rope_theta));
    let seq_len = normed.len() / hidden;
    // ── forward, keeping what the backward reads ──
    let q_full = project(normed, w.q, hidden, 2 * qd);
    let (mut q_pre, mut gate) =
        (Vec::with_capacity(seq_len * qd), Vec::with_capacity(seq_len * qd));
    for head in q_full.chunks_exact(2 * hd) {
        q_pre.extend_from_slice(&head[..hd]);
        gate.extend_from_slice(&head[hd..]);
    }
    let k_pre = project(normed, w.k, hidden, kvd);
    let v = project(normed, w.v, hidden, kvd);
    let (mut q, mut k) = (q_pre.clone(), k_pre.clone());
    rms_norm_chunks(&mut q, w.q_norm, eps);
    rms_norm_chunks(&mut k, w.k_norm, eps);
    rope_seq(&mut q, nh, hd, dims.n_rot, base, false);
    rope_seq(&mut k, nkv, hd, dims.n_rot, base, false);
    let scale = T::inv_sqrt(hd);
    // probs[t][h] = softmax over p ≤ t
    let mut probs: Vec<Vec<T>> = Vec::with_capacity(seq_len * nh);
    let mut attn = vec![T::ZERO; seq_len * qd];
    for t in 0..seq_len {
        for h in 0..nh {
            let kv_off = (h / group) * hd;
            let q_h = &q[t * qd + h * hd..t * qd + (h + 1) * hd];
            let mut s: Vec<T> = (0..=t)
                .map(|p| {
                    let k_p = &k[p * kvd + kv_off..p * kvd + kv_off + hd];
                    q_h.iter().zip(k_p).map(|(&a, &b)| a * b).sum::<T>() * scale
                })
                .collect();
            let max = s.iter().copied().fold(T::NEG_INFINITY, T::max);
            let mut sum = T::ZERO;
            for x in &mut s {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in &mut s {
                *x = *x / sum;
            }
            let out_h = &mut attn[t * qd + h * hd..t * qd + (h + 1) * hd];
            for (p, &pr) in s.iter().enumerate() {
                let v_p = &v[p * kvd + kv_off..p * kvd + kv_off + hd];
                for (o, &vi) in out_h.iter_mut().zip(v_p) {
                    *o += pr * vi;
                }
            }
            probs.push(s);
        }
    }
    let gated: Vec<T> = attn.iter().zip(&gate).map(|(&a, &g)| a * sigmoid(g)).collect();
    // ── backward ──
    let mut d_gated = vec![T::ZERO; gated.len()];
    let d_out_w = project_backward(&gated, w.out, d_y, qd, hidden, &mut d_gated);
    let mut d_attn = vec![T::ZERO; attn.len()];
    let mut d_gate = vec![T::ZERO; gate.len()];
    for i in 0..attn.len() {
        let sg = sigmoid(gate[i]);
        d_attn[i] = d_gated[i] * sg;
        d_gate[i] = d_gated[i] * attn[i] * sg * (T::ONE - sg);
    }
    let (mut dq, mut dk, mut dv) =
        (vec![T::ZERO; q.len()], vec![T::ZERO; k.len()], vec![T::ZERO; v.len()]);
    for t in 0..seq_len {
        for h in 0..nh {
            let kv_off = (h / group) * hd;
            let pr = &probs[t * nh + h];
            let d_o = &d_attn[t * qd + h * hd..t * qd + (h + 1) * hd];
            // o = Σ_p P_p v_p  →  dP_p = d_o·v_p,  dv_p += P_p d_o
            let d_p: Vec<T> = (0..=t)
                .map(|p| {
                    let v_p = &v[p * kvd + kv_off..p * kvd + kv_off + hd];
                    d_o.iter().zip(v_p).map(|(&a, &b)| a * b).sum::<T>()
                })
                .collect();
            let pdp: T = pr.iter().zip(&d_p).map(|(&a, &b)| a * b).sum();
            for p in 0..=t {
                let (kr, qr) = (p * kvd + kv_off, t * qd + h * hd);
                for i in 0..hd {
                    dv[kr + i] += pr[p] * d_o[i];
                }
                // softmax: dS_p = P_p (dP_p − Σ P dP); S_p = scale · q·k_p
                let ds = pr[p] * (d_p[p] - pdp) * scale;
                for i in 0..hd {
                    dq[qr + i] += ds * k[kr + i];
                    dk[kr + i] += ds * q[qr + i];
                }
            }
        }
    }
    rope_seq(&mut dq, nh, hd, dims.n_rot, base, true);
    rope_seq(&mut dk, nkv, hd, dims.n_rot, base, true);
    let (mut d_qn, mut d_kn) = (vec![T::ZERO; hd], vec![T::ZERO; hd]);
    let d_qpre = rms_norm_chunks_backward(&q_pre, w.q_norm, &dq, eps, &mut d_qn);
    let d_kpre = rms_norm_chunks_backward(&k_pre, w.k_norm, &dk, eps, &mut d_kn);
    let mut d_qfull = Vec::with_capacity(q_full.len());
    for (dqh, dgh) in d_qpre.chunks_exact(hd).zip(d_gate.chunks_exact(hd)) {
        d_qfull.extend_from_slice(dqh);
        d_qfull.extend_from_slice(dgh);
    }
    let mut d_normed = vec![T::ZERO; normed.len()];
    let grads = GatedAttnGrads {
        q: project_backward(normed, w.q, &d_qfull, hidden, 2 * qd, &mut d_normed),
        k: project_backward(normed, w.k, &d_kpre, hidden, kvd, &mut d_normed),
        v: project_backward(normed, w.v, &dv, hidden, kvd, &mut d_normed),
        q_norm: d_qn,
        k_norm: d_kn,
        out: d_out_w,
    };
    (d_normed, grads)
}

/// Reverse-mode gradient of [`qwen35_block_forward`](super::qwen35_block_forward).
/// Returns `(∂L/∂hidden, ∂L/∂W)` for `d_out = ∂L/∂output`.
///
/// # Panics
/// If a weight's length disagrees with the layer's dims, or `d_out` is not `hidden`'s shape.
#[must_use]
pub fn qwen35_block_backward<T: GdnFloat>(
    hidden: &[T],
    attn_norm: &[T],
    mixer: &Qwen35Mixer<'_, T>,
    post_norm: &[T],
    ffn: &SwiGluWeights<'_, T>,
    eps: T,
    d_out: &[T],
) -> (Vec<T>, Qwen35BlockGrads<T>) {
    let d = attn_norm.len();
    assert_eq!(d_out.len(), hidden.len());
    // ── forward ──
    let mut normed = hidden.to_vec();
    rms_norm_chunks(&mut normed, attn_norm, eps);
    let mixed = match mixer {
        Qwen35Mixer::Gdn(w, dims) => gdn_mixer_forward(&normed, w, dims),
        Qwen35Mixer::Attention(w, dims) => gated_attn_forward(&normed, w, dims),
    };
    let h1: Vec<T> = hidden.iter().zip(&mixed).map(|(&a, &b)| a + b).collect();
    let mut post = h1.clone();
    rms_norm_chunks(&mut post, post_norm, eps);
    let inter = ffn.gate.len() / d;
    let gate = project(&post, ffn.gate, d, inter);
    let up = project(&post, ffn.up, d, inter);
    let act: Vec<T> = up.iter().zip(&gate).map(|(&u, &g)| u * silu(g)).collect();
    // ── backward: out = h1 + down(act) ──
    let mut d_act = vec![T::ZERO; act.len()];
    let ffn_down = project_backward(&act, ffn.down, d_out, inter, d, &mut d_act);
    let d_up: Vec<T> = d_act.iter().zip(&gate).map(|(&da, &g)| da * silu(g)).collect();
    let d_gate: Vec<T> =
        d_act.iter().zip(&up).zip(&gate).map(|((&da, &u), &g)| da * u * silu_grad(g)).collect();
    let mut d_post = vec![T::ZERO; post.len()];
    let ffn_gate = project_backward(&post, ffn.gate, &d_gate, d, inter, &mut d_post);
    let ffn_up = project_backward(&post, ffn.up, &d_up, d, inter, &mut d_post);
    let mut d_pn = vec![T::ZERO; d];
    let d_h1_ffn = rms_norm_chunks_backward(&h1, post_norm, &d_post, eps, &mut d_pn);
    let d_h1: Vec<T> = d_out.iter().zip(&d_h1_ffn).map(|(&a, &b)| a + b).collect();
    // h1 = hidden + mixer(rms(hidden))
    let (d_normed, mixer_grads) = match mixer {
        Qwen35Mixer::Gdn(w, dims) => {
            let (dn, g) = gdn_mixer_backward(&normed, w, dims, &d_h1);
            (dn, Qwen35MixerGrads::Gdn(g))
        }
        Qwen35Mixer::Attention(w, dims) => {
            let (dn, g) = gated_attn_backward(&normed, w, dims, &d_h1);
            (dn, Qwen35MixerGrads::Attention(g))
        }
    };
    let mut d_an = vec![T::ZERO; d];
    let d_hidden_mix = rms_norm_chunks_backward(hidden, attn_norm, &d_normed, eps, &mut d_an);
    let d_hidden = d_h1.iter().zip(&d_hidden_mix).map(|(&a, &b)| a + b).collect();
    let grads = Qwen35BlockGrads {
        attn_norm: d_an,
        mixer: mixer_grads,
        post_norm: d_pn,
        ffn_gate,
        ffn_up,
        ffn_down,
    };
    (d_hidden, grads)
}

#[cfg(test)]
#[path = "qwen35_layer_backward_tests.rs"]
mod tests;
