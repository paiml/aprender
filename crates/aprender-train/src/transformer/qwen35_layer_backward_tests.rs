use super::super::gdn::GdnWeights;
use super::super::qwen35_layer::qwen35_block_forward;
use super::super::GdnDims;
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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// 4 query heads over 2 kv heads (so the kv-group fan-in is exercised), partial rope.
const ATT: GatedAttnDims = GatedAttnDims {
    hidden_dim: 8,
    num_heads: 4,
    num_kv_heads: 2,
    head_dim: 8,
    n_rot: 4,
    rope_theta: 10_000.0,
    eps: 1e-6,
};
const GDN: GdnDims = GdnDims {
    hidden_dim: 8,
    num_k_heads: 2,
    head_k_dim: 4,
    num_v_heads: 4,
    head_v_dim: 4,
    conv_kernel: 4,
    eps: 1e-6,
};
const SEQ: usize = 6;
const INTER: usize = 12;
const EPS: f64 = 1e-6;

/// Every input of a loss `L = Σ w_y · f(inputs)`, in a fixed order, plus `w_y`.
#[derive(Clone)]
struct Case {
    f: Vec<Vec<f64>>,
    w_y: Vec<f64>,
}

fn attn_fields(r: &mut Lcg) -> Vec<Vec<f64>> {
    let (h, qd, kvd, hd) = (ATT.hidden_dim, ATT.q_dim(), ATT.kv_dim(), ATT.head_dim);
    vec![
        r.vec(2 * qd * h, -0.5, 0.5), // q (+ gate)
        r.vec(kvd * h, -0.5, 0.5),    // k
        r.vec(kvd * h, -0.5, 0.5),    // v
        r.vec(hd, 0.5, 1.5),          // q_norm
        r.vec(hd, 0.5, 1.5),          // k_norm
        r.vec(h * qd, -0.5, 0.5),     // out
    ]
}

fn gdn_fields(r: &mut Lcg) -> Vec<Vec<f64>> {
    let (h, cd, vd, nv) = (GDN.hidden_dim, GDN.conv_dim(), GDN.v_dim(), GDN.num_v_heads);
    vec![
        r.vec(cd * h, -0.5, 0.5),
        r.vec(vd * h, -0.5, 0.5),
        r.vec(nv * h, -0.5, 0.5),
        r.vec(nv * h, -0.5, 0.5),
        r.vec(nv, -1.5, -0.2),
        r.vec(nv, -0.5, 0.5),
        r.vec(cd * GDN.conv_kernel, -0.6, 0.6),
        r.vec(GDN.head_v_dim, 0.5, 1.5),
        r.vec(h * vd, -0.5, 0.5),
    ]
}

fn attn_weights(f: &[Vec<f64>]) -> GatedAttnWeights<'_, f64> {
    GatedAttnWeights { q: &f[0], k: &f[1], v: &f[2], q_norm: &f[3], k_norm: &f[4], out: &f[5] }
}

fn gdn_weights(f: &[Vec<f64>]) -> GdnWeights<'_, f64> {
    GdnWeights {
        qkv: &f[0],
        gate: &f[1],
        alpha: &f[2],
        beta: &f[3],
        a: &f[4],
        dt_bias: &f[5],
        conv: &f[6],
        norm: &f[7],
        out: &f[8],
    }
}

/// Central-difference gradcheck of every input of `case` against `got` (same order),
/// rel err ≤ 1e-3 each, and never against a ~0 numeric gradient.
fn gradcheck(case: &Case, loss: impl Fn(&Case) -> f64, got: &[Vec<f64>], at: &str) {
    const H: f64 = 1e-6;
    assert_eq!(got.len(), case.f.len(), "{at}: one gradient per input");
    for fi in 0..case.f.len() {
        let mut c = case.clone();
        let want: Vec<f64> = (0..c.f[fi].len())
            .map(|i| {
                let x = c.f[fi][i];
                c.f[fi][i] = x + H;
                let up = loss(&c);
                c.f[fi][i] = x - H;
                let down = loss(&c);
                c.f[fi][i] = x;
                (up - down) / (2.0 * H)
            })
            .collect();
        assert!(norm(&want) > 1e-4, "{at}: input {fi} gradient ~0, the check would be vacuous");
        assert_eq!(got[fi].len(), want.len(), "{at}: input {fi} shape");
        let diff: Vec<f64> = got[fi].iter().zip(&want).map(|(x, y)| x - y).collect();
        let rel = norm(&diff) / norm(&want);
        assert!(rel <= 1e-3, "{at}: input {fi} rel err {rel:e} > 1e-3");
    }
}

fn attn_case(seed: u64) -> Case {
    let mut r = Lcg(seed);
    let mut f = vec![r.vec(SEQ * ATT.hidden_dim, -1.0, 1.0)];
    f.extend(attn_fields(&mut r));
    Case { f, w_y: r.vec(SEQ * ATT.hidden_dim, -1.0, 1.0) }
}

/// FALSIFY-QTG-003 for the gated full-attention mixer: `∂L/∂normed` and all six weights
/// pass f64 gradcheck — output projection, sigmoid gate, causal softmax, GQA fan-in,
/// partial NEOX rope, per-head q/k `RMSNorm`, and the interleaved `[q_h | gate_h]` rows.
#[test]
fn falsify_qtg_003_gated_attn_gradcheck_f64() {
    for seed in [3, 19, 31] {
        let case = attn_case(seed);
        let loss =
            |c: &Case| dot(&gated_attn_forward(&c.f[0], &attn_weights(&c.f[1..]), &ATT), &c.w_y);
        let (dn, g) = gated_attn_backward(&case.f[0], &attn_weights(&case.f[1..]), &ATT, &case.w_y);
        let got = [dn, g.q, g.k, g.v, g.q_norm, g.k_norm, g.out];
        gradcheck(&case, loss, &got, &format!("attn seed {seed}"));
    }
}

/// Block inputs: `hidden, attn_norm, <mixer weights…>, post_norm, ffn_gate, ffn_up, ffn_down`.
fn block_case(seed: u64, mixer: fn(&mut Lcg) -> Vec<Vec<f64>>) -> Case {
    let mut r = Lcg(seed);
    let d = 8;
    let mut f = vec![r.vec(SEQ * d, -1.0, 1.0), r.vec(d, 0.5, 1.5)];
    f.extend(mixer(&mut r));
    f.extend([
        r.vec(d, 0.5, 1.5),
        r.vec(INTER * d, -0.5, 0.5),
        r.vec(INTER * d, -0.5, 0.5),
        r.vec(d * INTER, -0.5, 0.5),
    ]);
    Case { f, w_y: r.vec(SEQ * d, -1.0, 1.0) }
}

fn block_mixer(f: &[Vec<f64>], attention: bool) -> Qwen35Mixer<'_, f64> {
    if attention {
        Qwen35Mixer::Attention(attn_weights(&f[2..8]), ATT)
    } else {
        Qwen35Mixer::Gdn(gdn_weights(&f[2..11]), GDN)
    }
}

fn block_loss(c: &Case, attention: bool) -> f64 {
    let f = &c.f;
    let n = f.len();
    let ffn = SwiGluWeights { gate: &f[n - 3], up: &f[n - 2], down: &f[n - 1] };
    let mixer = block_mixer(f, attention);
    dot(&qwen35_block_forward(&f[0], &f[1], &mixer, &f[n - 4], &ffn, EPS), &c.w_y)
}

fn block_grads(c: &Case, attention: bool) -> Vec<Vec<f64>> {
    let f = &c.f;
    let n = f.len();
    let ffn = SwiGluWeights { gate: &f[n - 3], up: &f[n - 2], down: &f[n - 1] };
    let mixer = block_mixer(f, attention);
    let (dh, g) = qwen35_block_backward(&f[0], &f[1], &mixer, &f[n - 4], &ffn, EPS, &c.w_y);
    let mut got = vec![dh, g.attn_norm];
    match g.mixer {
        Qwen35MixerGrads::Attention(a) => got.extend([a.q, a.k, a.v, a.q_norm, a.k_norm, a.out]),
        Qwen35MixerGrads::Gdn(m) => {
            got.extend([m.qkv, m.gate, m.alpha, m.beta, m.a, m.dt_bias, m.conv, m.norm, m.out])
        }
    }
    got.extend([g.post_norm, g.ffn_gate, g.ffn_up, g.ffn_down]);
    got
}

/// FALSIFY-QTG-003 for a whole Qwen3.5 block, both mixer kinds: `∂L/∂hidden` and every
/// weight — both residuals, both `RMSNorm`s, `SwiGLU`, and the mixer — pass f64 gradcheck.
#[test]
fn falsify_qtg_003_block_gradcheck_f64() {
    for (attention, kind) in [(true, "attention"), (false, "gdn")] {
        for seed in [5, 29] {
            let mixer: fn(&mut Lcg) -> Vec<Vec<f64>> =
                if attention { attn_fields } else { gdn_fields };
            let case = block_case(seed, mixer);
            let got = block_grads(&case, attention);
            gradcheck(
                &case,
                |c| block_loss(c, attention),
                &got,
                &format!("{kind} block seed {seed}"),
            );
        }
    }
}

/// The f32 instantiation (what serve parity is proven on) and the f64 one (what the
/// gradcheck differentiates) are the same function: the attention forward agrees to f32
/// rounding.
#[test]
fn gated_attn_forward_f64_equals_f32() {
    let case = attn_case(7);
    let y64 = gated_attn_forward(&case.f[0], &attn_weights(&case.f[1..]), &ATT);
    let f32s: Vec<Vec<f32>> =
        case.f.iter().map(|v| v.iter().map(|&x| x as f32).collect()).collect();
    let w = GatedAttnWeights {
        q: &f32s[1],
        k: &f32s[2],
        v: &f32s[3],
        q_norm: &f32s[4],
        k_norm: &f32s[5],
        out: &f32s[6],
    };
    let y32 = gated_attn_forward(&f32s[0], &w, &ATT);
    for (a, b) in y64.iter().zip(&y32) {
        assert!((a - f64::from(*b)).abs() <= 1e-4 * (1.0 + a.abs()), "{a} vs {b}");
    }
}
