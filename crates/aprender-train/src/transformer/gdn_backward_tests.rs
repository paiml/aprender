use super::super::gdn::gated_delta_scan;
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
