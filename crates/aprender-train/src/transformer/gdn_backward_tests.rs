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

/// Every scan input, plus the fixed cotangents that define the scalar loss
/// `L = Σ w_out · out + Σ w_final · final_state`.
#[derive(Clone)]
struct Case {
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
        let mut r = Lcg(seed);
        let (kd, vd, nv, sl) = (DIMS.k_dim(), DIMS.v_dim(), DIMS.num_v_heads, DIMS.state_len());
        Self {
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
            &DIMS,
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
            &DIMS,
            Some(&self.s0),
            true,
        );
        gated_delta_scan_backward(
            &self.q,
            &self.k,
            &self.v,
            &self.beta,
            &self.g,
            &DIMS,
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
    f: Vec<Vec<f64>>,
    w_y: Vec<f64>,
}

impl MixCase {
    fn random(seed: u64) -> Self {
        let mut r = Lcg(seed);
        let (h, cd, vd, nv) = (MIX.hidden_dim, MIX.conv_dim(), MIX.v_dim(), MIX.num_v_heads);
        let f = vec![
            r.vec(MIX_T * h, -1.0, 1.0),            // normed
            r.vec(cd * h, -0.5, 0.5),               // qkv
            r.vec(vd * h, -0.5, 0.5),               // gate
            r.vec(nv * h, -0.5, 0.5),               // alpha
            r.vec(nv * h, -0.5, 0.5),               // beta
            r.vec(nv, -1.5, -0.2),                  // a = -exp(A_log) < 0
            r.vec(nv, -0.5, 0.5),                   // dt_bias
            r.vec(cd * MIX.conv_kernel, -0.6, 0.6), // conv
            r.vec(MIX.head_v_dim, 0.5, 1.5),        // norm
            r.vec(h * vd, -0.5, 0.5),               // out
        ];
        Self { f, w_y: r.vec(MIX_T * h, -1.0, 1.0) }
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
        dot(&gdn_mixer_forward(&self.f[0], &self.weights(), &MIX), &self.w_y)
    }
}

/// FALSIFY-QTG-003 at mixer scale: `∂L/∂normed` and every GDN weight pass central-
/// difference gradcheck at rel err ≤ 1e-3 in f64 — through `ssm_out`, the gated
/// `RMSNorm`, the scan, both gates, the L2 norms, SiLU, the causal conv and `attn_qkv`.
#[test]
fn falsify_qtg_003_mixer_gradcheck_f64() {
    const H: f64 = 1e-6;
    for seed in [2, 11, 23] {
        let case = MixCase::random(seed);
        let (d_normed, g) = gdn_mixer_backward(&case.f[0], &case.weights(), &MIX, &case.w_y);
        let got = [
            &d_normed, &g.qkv, &g.gate, &g.alpha, &g.beta, &g.a, &g.dt_bias, &g.conv, &g.norm,
            &g.out,
        ];
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
            assert!(norm(&want) > 1e-4, "seed {seed}: d{name} ~0, the check would be vacuous");
            assert_eq!(got[fi].len(), want.len(), "d{name} shape");
            let diff: Vec<f64> = got[fi].iter().zip(&want).map(|(x, y)| x - y).collect();
            let rel = norm(&diff) / norm(&want);
            assert!(rel <= 1e-3, "seed {seed}: d{name} rel err {rel:e} > 1e-3");
        }
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
