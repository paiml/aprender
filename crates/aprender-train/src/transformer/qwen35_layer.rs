//! T1-R2 step 2a (#4000, contract `qwen35-train-gdn-v1`): the rest of a Qwen3.5 layer
//! for training — the gated full-attention mixer, and the block (norm, mixer, residual,
//! norm, `SwiGLU`, residual) that both layer kinds share.
//!
//! Every fourth Qwen3.5 layer is full attention, and it is not Qwen3's: the Q
//! projection is twice as wide and carries a per-head output gate, Q and K get a
//! per-head RMS norm, the head is 256 wide, and `RoPE` rotates only the first `n_rot`
//! dims of each head (partial NEOX). This is the sequence form of serve's
//! `Qwen35Model::forward_attention`, which is the oracle (`serve_parity` in the tests).
//!
//! Layouts match serve, as in [`super::gdn`]:
//! - a projection `W` is row-major `[out × in]`
//! - `attn_q` rows are per head `[q_h | gate_h]`, each `head_dim` wide
//! - query head `h` reads key/value head `h / (num_heads / num_kv_heads)` (contiguous groups)

use super::gdn::{gdn_mixer_forward, project, sigmoid, silu, GdnDims, GdnFloat, GdnWeights};

/// The shape of one gated full-attention layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatedAttnDims {
    /// Model width.
    pub hidden_dim: usize,
    /// Query heads.
    pub num_heads: usize,
    /// Key/value heads; divides `num_heads`.
    pub num_kv_heads: usize,
    /// Width of one head (256 on every Qwen3.5 size, not `hidden / heads`).
    pub head_dim: usize,
    /// Rotated dims per head, `2 · Σ rope.dimension_sections` (64 on Qwen3.5).
    pub n_rot: usize,
    /// `RoPE` frequency base.
    pub rope_theta: f32,
    /// RMS norm epsilon.
    pub eps: f32,
}

impl GatedAttnDims {
    /// Width of all query heads together.
    #[must_use]
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Width of all key (or value) heads together.
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// One gated full-attention layer's weights, named as in the GGUF (`blk.N.*`).
#[derive(Debug, Clone, Copy)]
pub struct GatedAttnWeights<'a, T = f32> {
    /// `attn_q`: `[2·q_dim × hidden]`, per head `[q_h | gate_h]`.
    pub q: &'a [T],
    /// `attn_k`: `[kv_dim × hidden]`.
    pub k: &'a [T],
    /// `attn_v`: `[kv_dim × hidden]`.
    pub v: &'a [T],
    /// `attn_q_norm`: `[head_dim]`, shared by every query head.
    pub q_norm: &'a [T],
    /// `attn_k_norm`: `[head_dim]`, shared by every key head.
    pub k_norm: &'a [T],
    /// `attn_output`: `[hidden × q_dim]`.
    pub out: &'a [T],
}

/// `RMSNorm(x) · weight` over each `weight.len()`-wide chunk of `x`, in place.
pub(super) fn rms_norm_chunks<T: GdnFloat>(x: &mut [T], weight: &[T], eps: T) {
    let n = weight.len();
    for c in x.chunks_exact_mut(n) {
        let inv_rms =
            T::ONE / (c.iter().map(|&v| v * v).sum::<T>() / T::from_usize(n) + eps).sqrt();
        for (v, &w) in c.iter_mut().zip(weight) {
            *v = *v * inv_rms * w;
        }
    }
}

/// Partial NEOX `RoPE` over a sequence, in place: row `t` of `x` (`[seq_len × heads·head_dim]`)
/// is at position `t`. Pair `(j, j + n_rot/2)` of each head rotates by
/// `t · base^(-2j/n_rot)`, computed iteratively as ggml (and serve) do; dims
/// `n_rot..head_dim` pass through.
///
/// # Panics
/// If `n_rot` is odd or wider than `head_dim`.
pub fn partial_neox_rope_seq<T: GdnFloat>(
    x: &mut [T],
    heads: usize,
    head_dim: usize,
    n_rot: usize,
    base: T,
) {
    rope_seq(x, heads, head_dim, n_rot, base, false);
}

/// [`partial_neox_rope_seq`], or with `inverse` its transpose (rotation by `−θ`), which
/// is also its backward: a rotation's Jacobian is the rotation.
pub(super) fn rope_seq<T: GdnFloat>(
    x: &mut [T],
    heads: usize,
    head_dim: usize,
    n_rot: usize,
    base: T,
    inverse: bool,
) {
    assert!(n_rot % 2 == 0 && n_rot <= head_dim, "n_rot {n_rot} vs head_dim {head_dim}");
    let half = n_rot / 2;
    let theta_scale = base.powf(T::from_f32(-2.0) / T::from_usize(n_rot));
    for (pos, row) in x.chunks_exact_mut(heads * head_dim).enumerate() {
        for head in row.chunks_exact_mut(head_dim) {
            let mut theta = T::from_usize(pos);
            for j in 0..half {
                let (sin, cos) = theta.sin_cos();
                let sin = if inverse { T::ZERO - sin } else { sin };
                let (a, b) = (head[j], head[j + half]);
                head[j] = a * cos - b * sin;
                head[j + half] = a * sin + b * cos;
                theta *= theta_scale;
            }
        }
    }
}

/// The gated full-attention mixer over a sequence: `normed` is the attention-normed
/// input `[seq_len × hidden]`; the result is `[seq_len × hidden]` before the residual
/// add — serve's `forward_attention` from `attn_q` through `attn_output`, causal.
///
/// # Panics
/// If a weight's length disagrees with `dims`, or `num_kv_heads` does not divide `num_heads`.
#[must_use]
pub fn gated_attn_forward<T: GdnFloat>(
    normed: &[T],
    w: &GatedAttnWeights<'_, T>,
    dims: &GatedAttnDims,
) -> Vec<T> {
    let (hidden, hd, nh, nkv) = (dims.hidden_dim, dims.head_dim, dims.num_heads, dims.num_kv_heads);
    assert!(nkv > 0 && nh % nkv == 0, "{nh} query heads over {nkv} kv heads");
    assert_eq!((w.q_norm.len(), w.k_norm.len()), (hd, hd));
    let (qd, kvd, group) = (dims.q_dim(), dims.kv_dim(), nh / nkv);
    let q_full = project(normed, w.q, hidden, 2 * qd);
    let mut q = Vec::with_capacity(q_full.len() / 2);
    let mut gate = Vec::with_capacity(q_full.len() / 2);
    for head in q_full.chunks_exact(2 * hd) {
        q.extend_from_slice(&head[..hd]);
        gate.extend_from_slice(&head[hd..]);
    }
    let mut k = project(normed, w.k, hidden, kvd);
    let v = project(normed, w.v, hidden, kvd);
    let (eps, theta) = (T::from_f32(dims.eps), T::from_f32(dims.rope_theta));
    rms_norm_chunks(&mut q, w.q_norm, eps);
    rms_norm_chunks(&mut k, w.k_norm, eps);
    partial_neox_rope_seq(&mut q, nh, hd, dims.n_rot, theta);
    partial_neox_rope_seq(&mut k, nkv, hd, dims.n_rot, theta);

    let seq_len = normed.len() / hidden;
    let scale = T::inv_sqrt(hd);
    let mut attn = vec![T::ZERO; seq_len * qd];
    for t in 0..seq_len {
        for h in 0..nh {
            let kv_off = (h / group) * hd;
            let q_h = &q[t * qd + h * hd..t * qd + (h + 1) * hd];
            let mut scores: Vec<T> = (0..=t)
                .map(|p| {
                    let k_p = &k[p * kvd + kv_off..p * kvd + kv_off + hd];
                    q_h.iter().zip(k_p).map(|(&a, &b)| a * b).sum::<T>() * scale
                })
                .collect();
            let max = scores.iter().copied().fold(T::NEG_INFINITY, T::max);
            let mut sum = T::ZERO;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            let out_h = &mut attn[t * qd + h * hd..t * qd + (h + 1) * hd];
            for (p, &s) in scores.iter().enumerate() {
                let v_p = &v[p * kvd + kv_off..p * kvd + kv_off + hd];
                for (o, &vi) in out_h.iter_mut().zip(v_p) {
                    *o += s / sum * vi;
                }
            }
        }
    }
    for (o, &g) in attn.iter_mut().zip(&gate) {
        *o *= sigmoid(g);
    }
    project(&attn, w.out, qd, hidden)
}

/// The token mixer of one Qwen3.5 layer: GDN on three layers of every four, gated
/// full attention on the fourth (`full_attention_interval`).
#[derive(Debug, Clone, Copy)]
pub enum Qwen35Mixer<'a, T = f32> {
    /// A Gated `DeltaNet` layer.
    Gdn(GdnWeights<'a, T>, GdnDims),
    /// A gated full-attention layer.
    Attention(GatedAttnWeights<'a, T>, GatedAttnDims),
}

/// The `SwiGLU` FFN of a layer: `down(silu(gate·x) ⊙ up·x)`.
#[derive(Debug, Clone, Copy)]
pub struct SwiGluWeights<'a, T = f32> {
    /// `ffn_gate`: `[intermediate × hidden]`.
    pub gate: &'a [T],
    /// `ffn_up`: `[intermediate × hidden]`.
    pub up: &'a [T],
    /// `ffn_down`: `[hidden × intermediate]`.
    pub down: &'a [T],
}

/// One Qwen3.5 layer over a sequence, both kinds: `h += mixer(rms(h)); h += ffn(rms(h))`.
/// `hidden` is `[seq_len × hidden_dim]`; `attn_norm` and `post_norm` are `[hidden_dim]`.
///
/// # Panics
/// If a weight's length disagrees with the layer's dims.
#[must_use]
pub fn qwen35_block_forward<T: GdnFloat>(
    hidden: &[T],
    attn_norm: &[T],
    mixer: &Qwen35Mixer<'_, T>,
    post_norm: &[T],
    ffn: &SwiGluWeights<'_, T>,
    eps: T,
) -> Vec<T> {
    let d = attn_norm.len();
    let mut normed = hidden.to_vec();
    rms_norm_chunks(&mut normed, attn_norm, eps);
    let mixed = match mixer {
        Qwen35Mixer::Gdn(w, dims) => gdn_mixer_forward(&normed, w, dims),
        Qwen35Mixer::Attention(w, dims) => gated_attn_forward(&normed, w, dims),
    };
    let mut h: Vec<T> = hidden.iter().zip(&mixed).map(|(&a, &b)| a + b).collect();
    let mut post = h.clone();
    rms_norm_chunks(&mut post, post_norm, eps);
    let inter = ffn.gate.len() / d;
    let gate = project(&post, ffn.gate, d, inter);
    let mut up = project(&post, ffn.up, d, inter);
    for (u, &g) in up.iter_mut().zip(&gate) {
        *u *= silu(g);
    }
    for (x, y) in h.iter_mut().zip(project(&up, ffn.down, inter, d)) {
        *x += y;
    }
    h
}

#[cfg(test)]
#[path = "qwen35_layer_tests.rs"]
mod tests;
