use super::*;

/// Small deterministic generator (no `rand` dependency in the test's arithmetic).
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

/// A grouped layer (2 key heads shared by 4 value heads), like 4B/9B's 16:32.
const DIMS: GdnDims = GdnDims {
    hidden_dim: 12,
    num_k_heads: 2,
    head_k_dim: 4,
    num_v_heads: 4,
    head_v_dim: 4,
    conv_kernel: 4,
    eps: 1e-6,
};

/// QTC-001's shape: 3 key heads of width 8 shared by 6 value heads of width 4. With
/// d_k = d_v, code that uses one width where the other belongs computes the same
/// numbers, and with 2 : 4 heads the key-head count equals the head ratio, so either
/// can stand for the other. R21's CUDA kernel is compared against this forward at this
/// shape (qwen35-train-cuda-v1 QTC-001).
const QTC: GdnDims = GdnDims {
    hidden_dim: 12,
    num_k_heads: 3,
    head_k_dim: 8,
    num_v_heads: 6,
    head_v_dim: 4,
    conv_kernel: 4,
    eps: 1e-6,
};

struct Owned {
    qkv: Vec<f32>,
    gate: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    conv: Vec<f32>,
    norm: Vec<f32>,
    out: Vec<f32>,
}

impl Owned {
    fn random(d: &GdnDims, seed: u64) -> Self {
        let mut r = Lcg(seed);
        let h = d.hidden_dim;
        Self {
            qkv: r.vec(d.conv_dim() * h, 0.4),
            gate: r.vec(d.v_dim() * h, 0.4),
            alpha: r.vec(d.num_v_heads * h, 0.4),
            beta: r.vec(d.num_v_heads * h, 0.4),
            // ssm_a = -exp(A_log) is negative, so exp(g) is a decay in (0, 1).
            a: r.vec(d.num_v_heads, 1.0).iter().map(|x| -(x.exp())).collect(),
            dt_bias: r.vec(d.num_v_heads, 0.5),
            conv: r.vec(d.conv_dim() * d.conv_kernel, 0.5),
            norm: r.vec(d.head_v_dim, 1.0).iter().map(|x| 1.0 + 0.2 * x).collect(),
            out: r.vec(h * d.v_dim(), 0.4),
        }
    }

    fn view(&self) -> GdnWeights<'_> {
        GdnWeights {
            qkv: &self.qkv,
            gate: &self.gate,
            alpha: &self.alpha,
            beta: &self.beta,
            a: &self.a,
            dt_bias: &self.dt_bias,
            conv: &self.conv,
            norm: &self.norm,
            out: &self.out,
        }
    }
}

/// The hand-worked single step from serve's `test_delta_rule_recurrence`
/// (S₀ = I, k = [½, ½], v = [1, −1], β = ½, g = 0, q = [1, 2]).
#[test]
fn scan_reproduces_the_hand_worked_step() {
    let d = GdnDims {
        hidden_dim: 2,
        num_k_heads: 1,
        head_k_dim: 2,
        num_v_heads: 1,
        head_v_dim: 2,
        conv_kernel: 4,
        eps: 1e-6,
    };
    let s0 = [1.0, 0.0, 0.0, 1.0];
    let scan = gated_delta_scan(
        &[1.0, 2.0],
        &[0.5, 0.5],
        &[1.0, -1.0],
        &[0.5],
        &[0.0],
        &d,
        Some(&s0),
        true,
    );
    let s2 = 2.0_f32.sqrt();
    assert!((scan.out[0] - 1.375 / s2).abs() < 1e-6, "{:?}", scan.out);
    assert!((scan.out[1] - 0.875 / s2).abs() < 1e-6, "{:?}", scan.out);
    for (got, want) in scan.final_state.iter().zip([1.125, 0.125, -0.375, 0.625]) {
        assert!((got - want).abs() < 1e-6, "{:?}", scan.final_state);
    }
    assert_eq!(scan.history, scan.final_state);
}

/// Chunked training carries the state across chunks: two half-length scans joined by
/// `final_state` equal one full-length scan, bit for bit.
#[test]
fn scan_split_at_any_point_equals_one_scan() {
    let (t, mut r) = (7, Lcg(3));
    let q = r.vec(t * DIMS.k_dim(), 0.5);
    let k = r.vec(t * DIMS.k_dim(), 0.5);
    let v = r.vec(t * DIMS.v_dim(), 1.0);
    let beta: Vec<f32> = r.vec(t * DIMS.num_v_heads, 1.0).iter().map(|x| sigmoid(*x)).collect();
    let g: Vec<f32> = r.vec(t * DIMS.num_v_heads, 1.0).iter().map(|x| -x.abs()).collect();
    let whole = gated_delta_scan(&q, &k, &v, &beta, &g, &DIMS, None, true);
    for cut in 1..t {
        let (ck, cv, cg) = (cut * DIMS.k_dim(), cut * DIMS.v_dim(), cut * DIMS.num_v_heads);
        let a = gated_delta_scan(
            &q[..ck],
            &k[..ck],
            &v[..cv],
            &beta[..cg],
            &g[..cg],
            &DIMS,
            None,
            false,
        );
        let b = gated_delta_scan(
            &q[ck..],
            &k[ck..],
            &v[cv..],
            &beta[cg..],
            &g[cg..],
            &DIMS,
            Some(&a.final_state),
            false,
        );
        assert_eq!([a.out, b.out].concat(), whole.out, "cut at {cut}");
        assert_eq!(b.final_state, whole.final_state, "cut at {cut}");
    }
    assert_eq!(whole.history.len(), t * DIMS.state_len());
    assert_eq!(&whole.history[(t - 1) * DIMS.state_len()..], &whole.final_state[..]);
}

/// The mixer is causal: changing token `t` leaves every output before `t` unchanged,
/// and changes the output at `t`.
#[test]
fn mixer_is_causal() {
    let (t, w) = (6, Owned::random(&DIMS, 11));
    let x = Lcg(5).vec(t * DIMS.hidden_dim, 1.0);
    let base = gdn_mixer_forward(&x, &w.view(), &DIMS);
    for pos in 0..t {
        let mut x2 = x.clone();
        x2[pos * DIMS.hidden_dim] += 0.75;
        let y = gdn_mixer_forward(&x2, &w.view(), &DIMS);
        let cut = pos * DIMS.hidden_dim;
        assert_eq!(y[..cut], base[..cut], "token {pos} leaked backwards");
        assert_ne!(
            y[cut..cut + DIMS.hidden_dim],
            base[cut..cut + DIMS.hidden_dim],
            "token {pos} had no effect"
        );
    }
}

/// QTG-001 at layer scale: the sequence mixer equals serve's own per-token GDN
/// arithmetic (`realizar::gguf::forward_qwen35`, as `forward_deltanet` composes it)
/// on the same weights, on a grouped (2 key : 4 value heads) layer and at QTC-001's shape.
/// `realizar` is a dev-dependency, so this compiles in every `cargo test` run, CI included.
mod serve_parity {
    use super::*;
    use realizar::gguf::forward_qwen35 as serve;

    fn matvec(w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        w.chunks_exact(x.len())
            .take(d_out)
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Serve's `forward_deltanet`, mixer part, one token at a time (no residual, no FFN).
    fn serve_mixer(x: &[f32], w: &Owned, d: &GdnDims) -> Vec<f32> {
        let (cd, kd, vd, nv) = (d.conv_dim(), d.k_dim(), d.v_dim(), d.num_v_heads);
        let mut conv_state = vec![0.0; (d.conv_kernel - 1) * cd];
        let mut ssm = vec![0.0; d.state_len()];
        let mut out = Vec::new();
        for xt in x.chunks_exact(d.hidden_dim) {
            let conv_in = matvec(&w.qkv, xt, cd);
            let mut c = vec![0.0; cd];
            serve::causal_conv1d(&conv_in, &mut conv_state, &w.conv, d.conv_kernel, cd, &mut c);
            for v in c.iter_mut() {
                *v = serve::silu(*v);
            }
            let (mut q, mut k, v) =
                (c[..kd].to_vec(), c[kd..2 * kd].to_vec(), c[2 * kd..].to_vec());
            serve::l2_norm_per_head(&mut q, d.head_k_dim, d.eps);
            serve::l2_norm_per_head(&mut k, d.head_k_dim, d.eps);
            let dt: Vec<f32> = matvec(&w.alpha, xt, nv)
                .iter()
                .enumerate()
                .map(|(i, a)| serve::softplus(a + w.dt_bias[i]) * w.a[i])
                .collect();
            let beta: Vec<f32> =
                matvec(&w.beta, xt, nv).iter().map(|b| 1.0 / (1.0 + (-b).exp())).collect();
            let z = matvec(&w.gate, xt, vd);
            let mut o = vec![0.0; vd];
            serve::delta_rule_recurrence_gqa(
                &q,
                &k,
                &v,
                &beta,
                &dt,
                &mut ssm,
                &mut o,
                d.num_k_heads,
                d.head_k_dim,
                nv,
                d.head_v_dim,
            );
            let mut y = vec![0.0; vd];
            serve::gated_rmsnorm(&o, &z, &w.norm, d.eps, d.head_v_dim, &mut y);
            out.extend(matvec(&w.out, &y, d.hidden_dim));
        }
        out
    }

    /// `gdn_mixer_forward` equals `serve_mixer` on `d`'s layer, for three seeds.
    fn assert_mixer_equals_serve(d: &GdnDims, at: &str) {
        for seed in [1_u64, 2, 3] {
            let (t, w) = (9, Owned::random(d, seed));
            let x = Lcg(seed + 100).vec(t * d.hidden_dim, 1.0);
            let ours = gdn_mixer_forward(&x, &w.view(), d);
            let theirs = serve_mixer(&x, &w, d);
            let max_err =
                ours.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            let scale = theirs.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            assert!(
                scale > 1e-3,
                "{at}, seed {seed}: degenerate output, the comparison would be vacuous"
            );
            assert!(
                max_err <= 1e-5 * scale.max(1.0),
                "{at}, seed {seed}: max |train - serve| = {max_err} (scale {scale})"
            );
        }
    }

    #[test]
    fn falsify_qtg_001_sequence_mixer_equals_serve_per_token() {
        assert_mixer_equals_serve(&DIMS, "2 : 4 heads");
    }

    /// The same at QTC-001's shape (d_k = 8, d_v = 4, 3 key : 6 value heads).
    #[test]
    fn falsify_qtg_001_sequence_mixer_equals_serve_per_token_unequal_widths() {
        assert_mixer_equals_serve(&QTC, "QTC shape");
    }
}
