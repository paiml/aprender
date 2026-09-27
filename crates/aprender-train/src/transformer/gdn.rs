//! T1-R2 (#4000, contract `qwen35-train-gdn-v1`): the Gated `DeltaNet` (GDN) token
//! mixer of Qwen3.5, computed over a whole sequence at once, for training.
//!
//! aprender-serve steps GDN one token at a time and updates its state in place
//! (`realizar::gguf::forward_qwen35`, `Qwen35Model::forward_deltanet`), keeping nothing
//! for a backward pass. Training needs the whole sequence in one call and, for the
//! backward (R3), the recurrent state at every step. This module is that sequence form.
//! It is a separate implementation on purpose (spec §R2, spike S-R2): serve is the
//! oracle, and `serve_parity` below compares the two on the same weights.
//!
//! Every layout matches serve, so weights loaded for one are valid for the other:
//! - a projection `W` is row-major `[out × in]`: `y[o] = Σ_i W[o·in + i] · x[i]`
//! - the conv weight is `[channels × kernel]`; tap `kernel - 1` multiplies the current token
//! - the state of value head `h` is `S[i][j] = state[h·hv·hk + j·hk + i]` (`i` key, `j` value)
//! - value head `h` reads key/query head `h % num_k_heads` (the GGUF tiled order)

/// The shape of one GDN layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnDims {
    /// Model width (input and output of the mixer).
    pub hidden_dim: usize,
    /// Key/query heads.
    pub num_k_heads: usize,
    /// Width of one key/query head.
    pub head_k_dim: usize,
    /// Value heads; a positive multiple of `num_k_heads`.
    pub num_v_heads: usize,
    /// Width of one value head.
    pub head_v_dim: usize,
    /// Causal conv1d taps (4 on every Qwen3.5 size).
    pub conv_kernel: usize,
    /// RMS / L2 norm epsilon.
    pub eps: f32,
}

impl GdnDims {
    /// Width of all key (or query) heads together.
    #[must_use]
    pub fn k_dim(&self) -> usize {
        self.num_k_heads * self.head_k_dim
    }

    /// Width of all value heads together.
    #[must_use]
    pub fn v_dim(&self) -> usize {
        self.num_v_heads * self.head_v_dim
    }

    /// Channels of the fused q|k|v projection and of the conv.
    #[must_use]
    pub fn conv_dim(&self) -> usize {
        2 * self.k_dim() + self.v_dim()
    }

    /// Floats in one layer's recurrent state (all value heads).
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.num_v_heads * self.head_v_dim * self.head_k_dim
    }
}

/// One GDN layer's weights, named as in the GGUF (`blk.N.*`), in serve's layouts.
#[derive(Debug, Clone, Copy)]
pub struct GdnWeights<'a> {
    /// `attn_qkv`: `[conv_dim × hidden]`.
    pub qkv: &'a [f32],
    /// `attn_gate` (z): `[v_dim × hidden]`.
    pub gate: &'a [f32],
    /// `ssm_alpha` (dt projection): `[num_v_heads × hidden]`.
    pub alpha: &'a [f32],
    /// `ssm_beta`: `[num_v_heads × hidden]`.
    pub beta: &'a [f32],
    /// `ssm_a` (= `-exp(A_log)`): `[num_v_heads]`.
    pub a: &'a [f32],
    /// `ssm_dt_bias`: `[num_v_heads]`.
    pub dt_bias: &'a [f32],
    /// `ssm_conv1d`: `[conv_dim × conv_kernel]`.
    pub conv: &'a [f32],
    /// `ssm_norm`: `[head_v_dim]`, shared by every value head.
    pub norm: &'a [f32],
    /// `ssm_out`: `[hidden × v_dim]`.
    pub out: &'a [f32],
}

/// The result of [`gated_delta_scan`].
#[derive(Debug, Clone, PartialEq)]
pub struct GdnScan {
    /// Read-out `o_t` for every position: `[seq_len × v_dim]`.
    pub out: Vec<f32>,
    /// State after the last position: `[state_len]`. Passing it as the next call's
    /// `s0` continues the sequence exactly (chunked training carries it across chunks).
    pub final_state: Vec<f32>,
    /// With `keep_history`, the state after every position, `[seq_len × state_len]`
    /// (what the backward reads); empty otherwise.
    pub history: Vec<f32>,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Serve's softplus, including its linear cut-over above 20.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `y = x · Wᵀ` for every position: `x` is `[seq_len × d_in]`, `w` is `[d_out × d_in]`.
fn project(x: &[f32], w: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    assert_eq!(w.len(), d_out * d_in, "projection weight is not [{d_out} x {d_in}]");
    let mut y = Vec::with_capacity(x.len() / d_in * d_out);
    for row in x.chunks_exact(d_in) {
        y.extend(
            w.chunks_exact(d_in).map(|w_o| w_o.iter().zip(row).map(|(a, b)| a * b).sum::<f32>()),
        );
    }
    y
}

/// Depthwise causal conv1d over a whole sequence starting from an all-zero history,
/// which is serve's state after `Qwen35State::reset`.
///
/// # Panics
/// If `x` is not `seq_len × channels` or `w` is not `channels × kernel`.
#[must_use]
pub fn causal_conv1d_seq(x: &[f32], w: &[f32], channels: usize, kernel: usize) -> Vec<f32> {
    assert_eq!(x.len() % channels, 0);
    assert_eq!(w.len(), channels * kernel);
    let seq_len = x.len() / channels;
    let mut y = vec![0.0; x.len()];
    for t in 0..seq_len {
        for c in 0..channels {
            let taps = &w[c * kernel..(c + 1) * kernel];
            // Tap `kernel - 1 - lag` multiplies token `t - lag`; tokens before 0 are zero.
            y[t * channels + c] = (0..kernel.min(t + 1))
                .map(|lag| taps[kernel - 1 - lag] * x[(t - lag) * channels + c])
                .sum();
        }
    }
    y
}

/// Scale every `head_dim`-wide head of `x` to unit L2 norm on its own.
fn l2_norm_heads(x: &mut [f32], head_dim: usize, eps: f32) {
    for head in x.chunks_exact_mut(head_dim) {
        let scale = 1.0 / (head.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
        for v in head.iter_mut() {
            *v *= scale;
        }
    }
}

/// One step of one value head: decay, delta-rule write, read-out. `s` is that head's
/// `[hv × hk]` state block; returns `o = scale · Sᵀ q` into `o`.
#[allow(clippy::too_many_arguments)]
fn step_head(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    g: f32,
    scale: f32,
    o: &mut [f32],
) {
    let hk = k.len();
    let decay = g.exp();
    for x in s.iter_mut() {
        *x *= decay;
    }
    for (j, row) in s.chunks_exact_mut(hk).enumerate() {
        let pred: f32 = row.iter().zip(k).map(|(a, b)| a * b).sum();
        let delta = (v[j] - pred) * beta;
        row.iter_mut().zip(k).for_each(|(x, ki)| *x += ki * delta);
        o[j] = row.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() * scale;
    }
}

/// The gated delta rule over a sequence: for each position `t` and value head `h`,
/// `S ← exp(g)·S`, `S ← S + k (β (v − Sᵀk))ᵀ`, `o = Sᵀq / √hk`. `q`/`k` are
/// `[seq_len × k_dim]` (already L2-normalised), `v` is `[seq_len × v_dim]`, `beta` and
/// `g` are `[seq_len × num_v_heads]`. `s0` is the carried-in state (zero if `None`).
///
/// # Panics
/// If a length disagrees with `dims`, or `num_v_heads` is not a multiple of `num_k_heads`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_scan(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    dims: &GdnDims,
    s0: Option<&[f32]>,
    keep_history: bool,
) -> GdnScan {
    let (nk, hk, nv, hv) = (dims.num_k_heads, dims.head_k_dim, dims.num_v_heads, dims.head_v_dim);
    assert!(
        nk > 0 && nv % nk == 0,
        "num_v_heads ({nv}) must be a positive multiple of num_k_heads ({nk})"
    );
    let seq_len = v.len() / dims.v_dim();
    assert_eq!(v.len(), seq_len * dims.v_dim());
    assert_eq!((q.len(), k.len()), (seq_len * dims.k_dim(), seq_len * dims.k_dim()));
    assert_eq!((beta.len(), g.len()), (seq_len * nv, seq_len * nv));
    let mut state = s0.map_or_else(|| vec![0.0; dims.state_len()], <[f32]>::to_vec);
    assert_eq!(state.len(), dims.state_len());
    let scale = 1.0 / (hk as f32).sqrt();
    let mut out = vec![0.0; v.len()];
    let mut history = Vec::with_capacity(if keep_history { seq_len * state.len() } else { 0 });
    for t in 0..seq_len {
        for h in 0..nv {
            let kh = h % nk;
            let qk = t * dims.k_dim() + kh * hk..t * dims.k_dim() + (kh + 1) * hk;
            let vs = t * dims.v_dim() + h * hv..t * dims.v_dim() + (h + 1) * hv;
            step_head(
                &mut state[h * hv * hk..(h + 1) * hv * hk],
                &q[qk.clone()],
                &k[qk],
                &v[vs.clone()],
                beta[t * nv + h],
                g[t * nv + h],
                scale,
                &mut out[vs],
            );
        }
        if keep_history {
            history.extend_from_slice(&state);
        }
    }
    GdnScan { out, final_state: state, history }
}

/// `RMSNorm(x) · weight · silu(z)` over each `head_v_dim`-wide head.
fn gated_rmsnorm(x: &[f32], z: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let hv = weight.len();
    let mut y = Vec::with_capacity(x.len());
    for (xh, zh) in x.chunks_exact(hv).zip(z.chunks_exact(hv)) {
        let inv_rms = 1.0 / (xh.iter().map(|v| v * v).sum::<f32>() / hv as f32 + eps).sqrt();
        y.extend(xh.iter().zip(zh).zip(weight).map(|((xi, zi), wi)| xi * inv_rms * wi * silu(*zi)));
    }
    y
}

/// The GDN token mixer over a sequence: `normed` is the attention-normed input
/// `[seq_len × hidden]`; the result is the mixer output `[seq_len × hidden]` before the
/// residual add — serve's `forward_deltanet` from `attn_qkv` through `ssm_out`.
///
/// # Panics
/// If a weight's length disagrees with `dims`.
#[must_use]
pub fn gdn_mixer_forward(normed: &[f32], w: &GdnWeights<'_>, dims: &GdnDims) -> Vec<f32> {
    let (hidden, nv, kd, vd, cd) =
        (dims.hidden_dim, dims.num_v_heads, dims.k_dim(), dims.v_dim(), dims.conv_dim());
    let conv_in = project(normed, w.qkv, hidden, cd);
    let mut conv = causal_conv1d_seq(&conv_in, w.conv, cd, dims.conv_kernel);
    for x in conv.iter_mut() {
        *x = silu(*x);
    }
    let mut q = Vec::with_capacity(conv.len() / cd * kd);
    let mut k = Vec::with_capacity(conv.len() / cd * kd);
    let mut v = Vec::with_capacity(conv.len() / cd * vd);
    for row in conv.chunks_exact(cd) {
        q.extend_from_slice(&row[..kd]);
        k.extend_from_slice(&row[kd..2 * kd]);
        v.extend_from_slice(&row[2 * kd..]);
    }
    l2_norm_heads(&mut q, dims.head_k_dim, dims.eps);
    l2_norm_heads(&mut k, dims.head_k_dim, dims.eps);
    assert_eq!((w.a.len(), w.dt_bias.len()), (nv, nv));
    let mut g = project(normed, w.alpha, hidden, nv);
    for (i, x) in g.iter_mut().enumerate() {
        *x = softplus(*x + w.dt_bias[i % nv]) * w.a[i % nv];
    }
    let mut beta = project(normed, w.beta, hidden, nv);
    for x in beta.iter_mut() {
        *x = sigmoid(*x);
    }
    let z = project(normed, w.gate, hidden, vd);
    let scan = gated_delta_scan(&q, &k, &v, &beta, &g, dims, None, false);
    let y = gated_rmsnorm(&scan.out, &z, w.norm, dims.eps);
    project(&y, w.out, vd, hidden)
}

#[cfg(test)]
#[path = "gdn_tests.rs"]
mod tests;
