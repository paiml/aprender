use super::super::gdn::{gated_delta_scan, gdn_mixer_forward};
use super::*;

/// Small deterministic generator in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n).map(|_| lo + (hi - lo) * (self.next() + 1.0) / 2.0).collect()
    }
}

/// The contract's tiny layer (`QTG-003`: d = 8, T = 16), grouped 2 key : 4 value heads
/// so the shared-key-head accumulation is exercised.
const DIMS: GdnDims = GdnDims {
    hidden_dim: 8,
    num_k_heads: 2,
    head_k_dim: 8,
    num_v_heads: 4,
    head_v_dim: 8,
    conv_kernel: 4,
    eps: 1e-6,
};
const T: usize = 16;

/// QTC-001's shape: 3 key heads of width 8 shared by 6 value heads of width 4. With
/// d_k = d_v, code that uses one width where the other belongs computes the same
/// numbers, and with 2 : 4 heads the key-head count equals the head ratio, so either
/// can stand for the other. R21's CUDA kernel is compared against this oracle at this
/// shape (qwen35-train-cuda-v1 QTC-001).
const QTC: GdnDims = GdnDims {
    hidden_dim: 8,
    num_k_heads: 3,
    head_k_dim: 8,
    num_v_heads: 6,
    head_v_dim: 4,
    conv_kernel: 4,
    eps: 1e-6,
};

/// Every scan input, plus the fixed cotangents that define the scalar loss
/// `L = Σ w_out · out + Σ w_final · final_state`.
#[derive(Clone)]
struct Case {
    dims: &'static GdnDims,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    beta: Vec<f64>,
    g: Vec<f64>,
    s0: Vec<f64>,
    w_out: Vec<f64>,
    w_final: Vec<f64>,
}

impl Case {
    fn random(seed: u64) -> Self {
        Self::random_in(&DIMS, seed)
    }

    fn random_in(d: &'static GdnDims, seed: u64) -> Self {
        let mut r = Lcg(seed);
        let (kd, vd, nv, sl) = (d.k_dim(), d.v_dim(), d.num_v_heads, d.state_len());
        Self {
            dims: d,
            // Unit-scale keys (the forward L2-normalises them before the scan).
            q: r.vec(T * kd, -0.35, 0.35),
            k: r.vec(T * kd, -0.35, 0.35),
            v: r.vec(T * vd, -1.0, 1.0),
            beta: r.vec(T * nv, 0.05, 0.95),
            // log-decay: e^g in (0.37, 0.99).
            g: r.vec(T * nv, -1.0, -0.01),
            s0: r.vec(sl, -0.5, 0.5),
            w_out: r.vec(T * vd, -1.0, 1.0),
            w_final: r.vec(sl, -1.0, 1.0),
        }
    }

    fn loss(&self) -> f64 {
        let s = gated_delta_scan(
            &self.q,
            &self.k,
            &self.v,
            &self.beta,
            &self.g,
            self.dims,
            Some(&self.s0),
            false,
        );
        dot(&s.out, &self.w_out) + dot(&s.final_state, &self.w_final)
    }

    fn analytic(&self) -> GdnScanGrads<f64> {
        let s = gated_delta_scan(
            &self.q,
            &self.k,
            &self.v,
            &self.beta,
            &self.g,
            self.dims,
            Some(&self.s0),
            true,
        );
        gated_delta_scan_backward(
            &self.q,
            &self.k,
            &self.v,
            &self.beta,
            &self.g,
            self.dims,
            Some(&self.s0),
            &s.history,
            &self.w_out,
            Some(&self.w_final),
        )
    }

    fn field(&mut self, name: &str) -> &mut Vec<f64> {
        match name {
            "q" => &mut self.q,
            "k" => &mut self.k,
            "v" => &mut self.v,
            "beta" => &mut self.beta,
            "g" => &mut self.g,
            "s0" => &mut self.s0,
            _ => unreachable!("{name}"),
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Central-difference gradient of the case's loss w.r.t. one input, in f64.
fn numeric(case: &Case, name: &str) -> Vec<f64> {
    const H: f64 = 1e-6;
    let mut c = case.clone();
    (0..c.field(name).len())
        .map(|i| {
            let x = c.field(name)[i];
            c.field(name)[i] = x + H;
            let up = c.loss();
            c.field(name)[i] = x - H;
            let down = c.loss();
            c.field(name)[i] = x;
            (up - down) / (2.0 * H)
        })
        .collect()
}

/// `‖analytic − numeric‖ / ‖numeric‖` for each input, in input order.
fn gradcheck(case: &Case) -> Vec<(&'static str, f64)> {
    let a = case.analytic();
    [("q", &a.dq), ("k", &a.dk), ("v", &a.dv), ("beta", &a.dbeta), ("g", &a.dg), ("s0", &a.ds0)]
        .into_iter()
        .map(|(name, got)| {
            let want = numeric(case, name);
            assert!(norm(&want) > 1e-3, "{name}: numeric gradient ~0, the check would be vacuous");
            let diff: Vec<f64> = got.iter().zip(&want).map(|(x, y)| x - y).collect();
            (name, norm(&diff) / norm(&want))
        })
        .collect()
}

/// FALSIFY-QTG-003: every scan input and `S₀` passes central-difference gradcheck at
/// rel err ≤ 1e-3 in f64, with a loss that reads both the outputs and the final state.
#[test]
fn falsify_qtg_003_scan_gradcheck_f64() {
    for seed in [1, 7, 42] {
        for (name, rel) in gradcheck(&Case::random(seed)) {
            assert!(rel <= 1e-3, "seed {seed}: d{name} rel err {rel:e} > 1e-3");
        }
    }
}

/// FALSIFY-QTG-003 at QTC-001's shape (d_k = 8, d_v = 4, 3 key : 6 value heads).
#[test]
fn falsify_qtg_003_scan_gradcheck_unequal_widths() {
    for seed in [1, 7, 42] {
        for (name, rel) in gradcheck(&Case::random_in(&QTC, seed)) {
            assert!(rel <= 1e-3, "QTC shape, seed {seed}: d{name} rel err {rel:e} > 1e-3");
        }
    }
}

/// The f32 instantiation training runs is the same backward: it agrees with f64 to
/// f32 precision on the same inputs.
#[test]
fn f32_backward_equals_f64_backward() {
    let c = Case::random(3);
    let want = c.analytic();
    let f = |x: &[f64]| x.iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let (q, k, v, beta, g, s0) = (f(&c.q), f(&c.k), f(&c.v), f(&c.beta), f(&c.g), f(&c.s0));
    let s = gated_delta_scan(&q, &k, &v, &beta, &g, &DIMS, Some(&s0), true);
    let got = gated_delta_scan_backward(
        &q,
        &k,
        &v,
        &beta,
        &g,
        &DIMS,
        Some(&s0),
        &s.history,
        &f(&c.w_out),
        Some(&f(&c.w_final)),
    );
    for (name, a, b) in [
        ("q", &got.dq, &want.dq),
        ("k", &got.dk, &want.dk),
        ("v", &got.dv, &want.dv),
        ("beta", &got.dbeta, &want.dbeta),
        ("g", &got.dg, &want.dg),
        ("s0", &got.ds0, &want.ds0),
    ] {
        let diff: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| f64::from(x) - y).collect();
        let rel = norm(&diff) / norm(b);
        assert!(rel <= 1e-5, "d{name}: f32 vs f64 rel {rel:e}");
    }
}

/// Without `d_final` the backward is the gradient of the outputs alone, and a scan
/// started from no `s0` has `ds0` = the gradient w.r.t. a zero state (not zero).
#[test]
fn backward_without_final_cotangent_or_s0() {
    let mut c = Case::random(9);
    c.w_final.iter_mut().for_each(|w| *w = 0.0);
    c.s0.iter_mut().for_each(|s| *s = 0.0);
    let with = c.analytic();
    let s = gated_delta_scan(&c.q, &c.k, &c.v, &c.beta, &c.g, &DIMS, None, true);
    let without = gated_delta_scan_backward(
        &c.q, &c.k, &c.v, &c.beta, &c.g, &DIMS, None, &s.history, &c.w_out, None,
    );
    assert_eq!(with, without);
    assert!(norm(&without.ds0) > 0.0);
}

/// A whole mixer at gradcheck size: hidden 8, 2 key heads : 4 value heads of width 4,
/// kernel 4, over 8 positions (past the kernel, so every conv tap is live).
const MIX: GdnDims = GdnDims {
    hidden_dim: 8,
    num_k_heads: 2,
    head_k_dim: 4,
    num_v_heads: 4,
    head_v_dim: 4,
    conv_kernel: 4,
    eps: 1e-6,
};
const MIX_T: usize = 8;
const MIX_FIELDS: [&str; 10] =
    ["normed", "qkv", "gate", "alpha", "beta", "a", "dt_bias", "conv", "norm", "out"];

/// The mixer's input, every weight, and the cotangent `w_y` of `L = Σ w_y · y`.
#[derive(Clone)]
struct MixCase {
    dims: &'static GdnDims,
    f: Vec<Vec<f64>>,
    w_y: Vec<f64>,
}

impl MixCase {
    fn random(seed: u64) -> Self {
        Self::random_in(&MIX, seed)
    }

    fn random_in(d: &'static GdnDims, seed: u64) -> Self {
        let mut r = Lcg(seed);
        let (h, cd, vd, nv) = (d.hidden_dim, d.conv_dim(), d.v_dim(), d.num_v_heads);
        let f = vec![
            r.vec(MIX_T * h, -1.0, 1.0),          // normed
            r.vec(cd * h, -0.5, 0.5),             // qkv
            r.vec(vd * h, -0.5, 0.5),             // gate
            r.vec(nv * h, -0.5, 0.5),             // alpha
            r.vec(nv * h, -0.5, 0.5),             // beta
            r.vec(nv, -1.5, -0.2),                // a = -exp(A_log) < 0
            r.vec(nv, -0.5, 0.5),                 // dt_bias
            r.vec(cd * d.conv_kernel, -0.6, 0.6), // conv
            r.vec(d.head_v_dim, 0.5, 1.5),        // norm
            r.vec(h * vd, -0.5, 0.5),             // out
        ];
        Self { dims: d, f, w_y: r.vec(MIX_T * h, -1.0, 1.0) }
    }

    fn weights(&self) -> GdnWeights<'_, f64> {
        let f = &self.f;
        GdnWeights {
            qkv: &f[1],
            gate: &f[2],
            alpha: &f[3],
            beta: &f[4],
            a: &f[5],
            dt_bias: &f[6],
            conv: &f[7],
            norm: &f[8],
            out: &f[9],
        }
    }

    fn loss(&self) -> f64 {
        dot(&gdn_mixer_forward(&self.f[0], &self.weights(), self.dims), &self.w_y)
    }
}

/// FALSIFY-QTG-003 at mixer scale: `∂L/∂normed` and every GDN weight pass central-
/// difference gradcheck at rel err ≤ 1e-3 in f64 — through `ssm_out`, the gated
/// `RMSNorm`, the scan, both gates, the L2 norms, SiLU, the causal conv and `attn_qkv`.
#[test]
fn falsify_qtg_003_mixer_gradcheck_f64() {
    for seed in [2, 11, 23] {
        mixer_gradcheck(&MixCase::random(seed), &format!("seed {seed}"));
    }
}

/// FALSIFY-QTG-003 at mixer scale and QTC-001's shape (d_k = 8, d_v = 4, 3 : 6 heads).
#[test]
fn falsify_qtg_003_mixer_gradcheck_unequal_widths() {
    for seed in [2, 11, 23] {
        mixer_gradcheck(&MixCase::random_in(&QTC, seed), &format!("QTC shape, seed {seed}"));
    }
}

/// Gradcheck `normed` and every weight of `case`'s mixer at rel err ≤ 1e-3.
fn mixer_gradcheck(case: &MixCase, at: &str) {
    const H: f64 = 1e-6;
    let (d_normed, g) = gdn_mixer_backward(&case.f[0], &case.weights(), case.dims, &case.w_y);
    let got =
        [&d_normed, &g.qkv, &g.gate, &g.alpha, &g.beta, &g.a, &g.dt_bias, &g.conv, &g.norm, &g.out];
    for (fi, name) in MIX_FIELDS.iter().enumerate() {
        let mut c = case.clone();
        let want: Vec<f64> = (0..c.f[fi].len())
            .map(|i| {
                let x = c.f[fi][i];
                c.f[fi][i] = x + H;
                let up = c.loss();
                c.f[fi][i] = x - H;
                let down = c.loss();
                c.f[fi][i] = x;
                (up - down) / (2.0 * H)
            })
            .collect();
        assert!(norm(&want) > 1e-4, "{at}: d{name} ~0, the check would be vacuous");
        assert_eq!(got[fi].len(), want.len(), "d{name} shape");
        let diff: Vec<f64> = got[fi].iter().zip(&want).map(|(x, y)| x - y).collect();
        let rel = norm(&diff) / norm(&want);
        assert!(rel <= 1e-3, "{at}: d{name} rel err {rel:e} > 1e-3");
    }
}

/// The softplus cut-over (serve: identity above 20) is differentiated too: with
/// `dt_bias` ≈ 25 every head's `alpha·x + dt_bias` is past 20, and a small `|a|` keeps
/// the decay `e^(a·softplus)` live so `d_a`/`d_dt_bias`/`d_alpha` are not vanishing.
/// (Review 6b, survivor B3: the other gradchecks never reach this branch.)
#[test]
fn falsify_qtg_003_mixer_gradcheck_past_softplus_cutover() {
    for seed in [4, 17] {
        let mut c = MixCase::random(seed);
        c.f[6].iter_mut().for_each(|b| *b += 25.0); // dt_bias
        c.f[5].iter_mut().for_each(|a| *a *= 0.02); // a ∈ (-0.03, -0.004)
        let a_pre = super::super::gdn::project(&c.f[0], &c.f[3], MIX.hidden_dim, MIX.num_v_heads);
        let min = (0..a_pre.len())
            .map(|i| a_pre[i] + c.f[6][i % MIX.num_v_heads])
            .fold(f64::INFINITY, f64::min);
        assert!(min > 20.5, "seed {seed}: a_pre {min} does not clear the cut-over");
        mixer_gradcheck(&c, &format!("cut-over seed {seed}"));
    }
}

/// The f64 mixer forward the gradcheck differentiates is the f32 forward serve parity
/// covers: same inputs, same outputs to f32 precision.
#[test]
fn mixer_forward_f64_equals_f32() {
    let c = MixCase::random(5);
    let f: Vec<Vec<f32>> = c.f.iter().map(|v| v.iter().map(|&x| x as f32).collect()).collect();
    let w = GdnWeights {
        qkv: &f[1],
        gate: &f[2],
        alpha: &f[3],
        beta: &f[4],
        a: &f[5],
        dt_bias: &f[6],
        conv: &f[7],
        norm: &f[8],
        out: &f[9],
    };
    let got = gdn_mixer_forward(&f[0], &w, &MIX);
    let want = gdn_mixer_forward(&c.f[0], &c.weights(), &MIX);
    let diff: Vec<f64> = got.iter().zip(&want).map(|(&a, &b)| f64::from(a) - b).collect();
    assert!(norm(&diff) / norm(&want) <= 1e-5);
}
