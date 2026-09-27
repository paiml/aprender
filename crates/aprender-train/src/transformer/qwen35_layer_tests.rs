use super::*;

/// Small deterministic generator (the same one `gdn_tests` uses).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

/// Grouped (4 query : 2 kv heads), head wider than hidden/heads, partial rotation —
/// every way Qwen3.5 attention differs from Qwen3's, at toy size.
const DIMS: GatedAttnDims = GatedAttnDims {
    hidden_dim: 12,
    num_heads: 4,
    num_kv_heads: 2,
    head_dim: 8,
    n_rot: 4,
    rope_theta: 10_000.0,
    eps: 1e-6,
};
const INTER: usize = 20;

struct Owned {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    out: Vec<f32>,
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

impl Owned {
    fn random(d: &GatedAttnDims, seed: u64) -> Self {
        let mut r = Lcg(seed);
        let h = d.hidden_dim;
        let mut norm = |n| r.vec(n, 1.0).iter().map(|x| 1.0 + 0.3 * x).collect::<Vec<f32>>();
        let (q_norm, k_norm, attn_norm, post_norm) =
            (norm(d.head_dim), norm(d.head_dim), norm(h), norm(h));
        let mut r = Lcg(seed + 7);
        Self {
            q: r.vec(2 * d.q_dim() * h, 0.5),
            k: r.vec(d.kv_dim() * h, 0.5),
            v: r.vec(d.kv_dim() * h, 0.5),
            q_norm,
            k_norm,
            out: r.vec(h * d.q_dim(), 0.4),
            attn_norm,
            post_norm,
            ffn_gate: r.vec(INTER * h, 0.4),
            ffn_up: r.vec(INTER * h, 0.4),
            ffn_down: r.vec(h * INTER, 0.4),
        }
    }

    fn attn(&self) -> GatedAttnWeights<'_> {
        GatedAttnWeights {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            out: &self.out,
        }
    }

    fn ffn(&self) -> SwiGluWeights<'_> {
        SwiGluWeights { gate: &self.ffn_gate, up: &self.ffn_up, down: &self.ffn_down }
    }
}

#[test]
fn rope_is_identity_at_position_zero_and_leaves_the_tail() {
    let x = Lcg(1).vec(3 * 2 * 8, 1.0);
    let mut y = x.clone();
    partial_neox_rope_seq(&mut y, 2, 8, 4, 10_000.0);
    assert_eq!(y[..16], x[..16], "position 0 must not rotate");
    for (row_y, row_x) in y.chunks_exact(16).zip(x.chunks_exact(16)) {
        for (hy, hx) in row_y.chunks_exact(8).zip(row_x.chunks_exact(8)) {
            assert_eq!(hy[4..], hx[4..], "dims n_rot.. must pass through");
            let (ny, nx) =
                (hy.iter().map(|v| v * v).sum::<f32>(), hx.iter().map(|v| v * v).sum::<f32>());
            assert!((ny - nx).abs() < 1e-5, "a rotation keeps the norm");
        }
    }
    assert_ne!(y[16..32], x[16..32], "position 1 must rotate");
}

#[test]
fn attention_is_causal() {
    let (t, w) = (6, Owned::random(&DIMS, 11));
    let x = Lcg(5).vec(t * DIMS.hidden_dim, 1.0);
    let base = gated_attn_forward(&x, &w.attn(), &DIMS);
    for pos in 0..t {
        let mut x2 = x.clone();
        x2[pos * DIMS.hidden_dim] += 0.75;
        let y = gated_attn_forward(&x2, &w.attn(), &DIMS);
        let cut = pos * DIMS.hidden_dim;
        assert_eq!(y[..cut], base[..cut], "token {pos} leaked backwards");
        assert_ne!(y[cut..cut + DIMS.hidden_dim], base[cut..cut + DIMS.hidden_dim]);
    }
}

/// QTG-001 at layer scale for the attention layers: the sequence forms equal serve's
/// own per-token arithmetic (`realizar::gguf::{ops, forward_qwen35}`, as
/// `Qwen35Model::forward_attention` composes it) on the same weights.
mod serve_parity {
    use super::*;
    use realizar::gguf::{forward_qwen35 as serve, ops};

    fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
        w.chunks_exact(x.len()).map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
    }

    /// Serve's `forward_attention` mixer, one token at a time with a growing KV cache.
    fn serve_attn(normed: &[f32], w: &Owned, d: &GatedAttnDims) -> Vec<f32> {
        let (hd, nh, nkv, kvd) = (d.head_dim, d.num_heads, d.num_kv_heads, d.kv_dim());
        let (mut k_cache, mut v_cache, mut out) = (Vec::new(), Vec::new(), Vec::new());
        for (pos, xt) in normed.chunks_exact(d.hidden_dim).enumerate() {
            let q_full = matvec(&w.q, xt);
            let (mut k, v) = (matvec(&w.k, xt), matvec(&w.v, xt));
            let (mut q, mut gate) = (Vec::new(), Vec::new());
            for h in 0..nh {
                q.extend_from_slice(&q_full[h * 2 * hd..h * 2 * hd + hd]);
                gate.extend_from_slice(&q_full[h * 2 * hd + hd..(h + 1) * 2 * hd]);
            }
            ops::apply_per_head_rms_norm(&mut q, &w.q_norm, nh, d.eps);
            ops::apply_per_head_rms_norm(&mut k, &w.k_norm, nkv, d.eps);
            serve::apply_partial_neox_rope(&mut q, nh, hd, d.n_rot, pos, d.rope_theta);
            serve::apply_partial_neox_rope(&mut k, nkv, hd, d.n_rot, pos, d.rope_theta);
            k_cache.extend(k);
            v_cache.extend(v);
            let mut attn = vec![0.0; nh * hd];
            for h in 0..nh {
                let kv_h = h / (nh / nkv);
                let q_h = &q[h * hd..(h + 1) * hd];
                let mut scores: Vec<f32> = (0..=pos)
                    .map(|p| {
                        let k_p = &k_cache[p * kvd + kv_h * hd..p * kvd + (kv_h + 1) * hd];
                        q_h.iter().zip(k_p).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
                    })
                    .collect();
                ops::softmax(&mut scores);
                for (p, s) in scores.iter().enumerate() {
                    let v_p = &v_cache[p * kvd + kv_h * hd..p * kvd + (kv_h + 1) * hd];
                    for (o, vi) in attn[h * hd..(h + 1) * hd].iter_mut().zip(v_p) {
                        *o += s * vi;
                    }
                }
            }
            serve::apply_sigmoid_gate(&mut attn, &gate);
            out.extend(matvec(&w.out, &attn));
        }
        out
    }

    /// Serve's attention layer in full (norm, mixer, residual, norm, `SwiGLU`, residual).
    fn serve_block(x: &[f32], w: &Owned, d: &GatedAttnDims) -> Vec<f32> {
        let hdim = d.hidden_dim;
        let mut normed = vec![0.0; x.len()];
        for (xt, nt) in x.chunks_exact(hdim).zip(normed.chunks_exact_mut(hdim)) {
            ops::rms_norm_into(xt, &w.attn_norm, d.eps, nt);
        }
        let mixed = serve_attn(&normed, w, d);
        let mut out = Vec::new();
        for (xt, mt) in x.chunks_exact(hdim).zip(mixed.chunks_exact(hdim)) {
            let h: Vec<f32> = xt.iter().zip(mt).map(|(a, b)| a + b).collect();
            let mut post = vec![0.0; hdim];
            ops::rms_norm_into(&h, &w.post_norm, d.eps, &mut post);
            let gate = matvec(&w.ffn_gate, &post);
            let up: Vec<f32> = matvec(&w.ffn_up, &post)
                .iter()
                .zip(&gate)
                .map(|(u, g)| u * serve::silu(*g))
                .collect();
            out.extend(h.iter().zip(matvec(&w.ffn_down, &up)).map(|(a, b)| a + b));
        }
        out
    }

    fn assert_close(ours: &[f32], theirs: &[f32], what: &str) {
        assert_eq!(ours.len(), theirs.len(), "{what}: length");
        let max_err = ours.iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let scale = theirs.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(scale > 1e-3, "{what}: degenerate output, the comparison would be vacuous");
        assert!(
            max_err <= 1e-5 * scale.max(1.0),
            "{what}: max |train - serve| = {max_err} (scale {scale})"
        );
    }

    #[test]
    fn falsify_qtg_001_gated_attention_equals_serve_per_token() {
        for seed in [1_u64, 2, 3] {
            let (t, w) = (9, Owned::random(&DIMS, seed));
            let x = Lcg(seed + 100).vec(t * DIMS.hidden_dim, 1.0);
            let ours = gated_attn_forward(&x, &w.attn(), &DIMS);
            assert_close(&ours, &serve_attn(&x, &w, &DIMS), &format!("seed {seed} mixer"));
            let block = qwen35_block_forward(
                &x,
                &w.attn_norm,
                &Qwen35Mixer::Attention(w.attn(), DIMS),
                &w.post_norm,
                &w.ffn(),
                DIMS.eps,
            );
            assert_close(&block, &serve_block(&x, &w, &DIMS), &format!("seed {seed} block"));
        }
    }
}
