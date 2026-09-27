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

use super::gdn::{gdn_mixer_forward, project, sigmoid, silu, GdnDims, GdnWeights};

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
pub struct GatedAttnWeights<'a> {
    /// `attn_q`: `[2·q_dim × hidden]`, per head `[q_h | gate_h]`.
    pub q: &'a [f32],
    /// `attn_k`: `[kv_dim × hidden]`.
    pub k: &'a [f32],
    /// `attn_v`: `[kv_dim × hidden]`.
    pub v: &'a [f32],
    /// `attn_q_norm`: `[head_dim]`, shared by every query head.
    pub q_norm: &'a [f32],
    /// `attn_k_norm`: `[head_dim]`, shared by every key head.
    pub k_norm: &'a [f32],
    /// `attn_output`: `[hidden × q_dim]`.
    pub out: &'a [f32],
}

/// `RMSNorm(x) · weight` over each `weight.len()`-wide chunk of `x`, in place.
pub(super) fn rms_norm_chunks(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = weight.len();
    for c in x.chunks_exact_mut(n) {
        let inv_rms = 1.0 / (c.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        for (v, w) in c.iter_mut().zip(weight) {
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
pub fn partial_neox_rope_seq(
    x: &mut [f32],
    heads: usize,
    head_dim: usize,
    n_rot: usize,
    base: f32,
) {
    assert!(n_rot % 2 == 0 && n_rot <= head_dim, "n_rot {n_rot} vs head_dim {head_dim}");
    let half = n_rot / 2;
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    for (pos, row) in x.chunks_exact_mut(heads * head_dim).enumerate() {
        for head in row.chunks_exact_mut(head_dim) {
            let mut theta = pos as f32;
            for j in 0..half {
                let (sin, cos) = theta.sin_cos();
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
pub fn gated_attn_forward(
    normed: &[f32],
    w: &GatedAttnWeights<'_>,
    dims: &GatedAttnDims,
) -> Vec<f32> {
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
    rms_norm_chunks(&mut q, w.q_norm, dims.eps);
    rms_norm_chunks(&mut k, w.k_norm, dims.eps);
    partial_neox_rope_seq(&mut q, nh, hd, dims.n_rot, dims.rope_theta);
    partial_neox_rope_seq(&mut k, nkv, hd, dims.n_rot, dims.rope_theta);

    let seq_len = normed.len() / hidden;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut attn = vec![0.0; seq_len * qd];
    for t in 0..seq_len {
        for h in 0..nh {
            let kv_off = (h / group) * hd;
            let q_h = &q[t * qd + h * hd..t * qd + (h + 1) * hd];
            let mut scores: Vec<f32> = (0..=t)
                .map(|p| {
                    let k_p = &k[p * kvd + kv_off..p * kvd + kv_off + hd];
                    q_h.iter().zip(k_p).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            let out_h = &mut attn[t * qd + h * hd..t * qd + (h + 1) * hd];
            for (p, s) in scores.iter().enumerate() {
                let v_p = &v[p * kvd + kv_off..p * kvd + kv_off + hd];
                for (o, vi) in out_h.iter_mut().zip(v_p) {
                    *o += s / sum * vi;
                }
            }
        }
    }
    for (o, g) in attn.iter_mut().zip(&gate) {
        *o *= sigmoid(*g);
    }
    project(&attn, w.out, qd, hidden)
}

/// The token mixer of one Qwen3.5 layer: GDN on three layers of every four, gated
/// full attention on the fourth (`full_attention_interval`).
#[derive(Debug, Clone, Copy)]
pub enum Qwen35Mixer<'a> {
    /// A Gated `DeltaNet` layer.
    Gdn(GdnWeights<'a>, GdnDims),
    /// A gated full-attention layer.
    Attention(GatedAttnWeights<'a>, GatedAttnDims),
}

/// The `SwiGLU` FFN of a layer: `down(silu(gate·x) ⊙ up·x)`.
#[derive(Debug, Clone, Copy)]
pub struct SwiGluWeights<'a> {
    /// `ffn_gate`: `[intermediate × hidden]`.
    pub gate: &'a [f32],
    /// `ffn_up`: `[intermediate × hidden]`.
    pub up: &'a [f32],
    /// `ffn_down`: `[hidden × intermediate]`.
    pub down: &'a [f32],
}

/// One Qwen3.5 layer over a sequence, both kinds: `h += mixer(rms(h)); h += ffn(rms(h))`.
/// `hidden` is `[seq_len × hidden_dim]`; `attn_norm` and `post_norm` are `[hidden_dim]`.
///
/// # Panics
/// If a weight's length disagrees with the layer's dims.
#[must_use]
pub fn qwen35_block_forward(
    hidden: &[f32],
    attn_norm: &[f32],
    mixer: &Qwen35Mixer<'_>,
    post_norm: &[f32],
    ffn: &SwiGluWeights<'_>,
    eps: f32,
) -> Vec<f32> {
    let d = attn_norm.len();
    let mut normed = hidden.to_vec();
    rms_norm_chunks(&mut normed, attn_norm, eps);
    let mixed = match mixer {
        Qwen35Mixer::Gdn(w, dims) => gdn_mixer_forward(&normed, w, dims),
        Qwen35Mixer::Attention(w, dims) => gated_attn_forward(&normed, w, dims),
    };
    let mut h: Vec<f32> = hidden.iter().zip(&mixed).map(|(a, b)| a + b).collect();
    let mut post = h.clone();
    rms_norm_chunks(&mut post, post_norm, eps);
    let inter = ffn.gate.len() / d;
    let gate = project(&post, ffn.gate, d, inter);
    let mut up = project(&post, ffn.up, d, inter);
    for (u, g) in up.iter_mut().zip(&gate) {
        *u *= silu(*g);
    }
    for (x, y) in h.iter_mut().zip(project(&up, ffn.down, inter, d)) {
        *x += y;
    }
    h
}

#[cfg(test)]
#[path = "qwen35_layer_tests.rs"]
mod tests;
