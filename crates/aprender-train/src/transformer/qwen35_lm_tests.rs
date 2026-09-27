use super::super::gdn::GdnWeights;
use super::super::qwen35_layer::GatedAttnWeights;
use super::super::qwen35_layer_backward::Qwen35MixerGrads;
use super::super::{GatedAttnDims, GdnDims};
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

fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

const D: usize = 8;
const VOCAB: usize = 11;
const INTER: usize = 12;
const ATT: GatedAttnDims = GatedAttnDims {
    hidden_dim: D,
    num_heads: 4,
    num_kv_heads: 2,
    head_dim: 8,
    n_rot: 4,
    rope_theta: 10_000.0,
    eps: 1e-6,
};
const GDN: GdnDims = GdnDims {
    hidden_dim: D,
    num_k_heads: 2,
    head_k_dim: 4,
    num_v_heads: 4,
    head_v_dim: 4,
    conv_kernel: 4,
    eps: 1e-6,
};
/// Repeated ids (1) so the embedding gradient must scatter-ADD.
const TOKENS: [u32; 7] = [3, 1, 4, 1, 5, 9, 2];
const TARGETS: [u32; 7] = [1, 4, 1, 5, 9, 2, 6];
/// Layer 0 GDN (14 tensors), layer 1 attention (11), after `embed`.
const L1: usize = 15;
const FINAL: usize = 26;

/// Every weight, flat: `embed, [attn_norm, gdn×9, post, gate, up, down],
/// [attn_norm, attn×6, post, gate, up, down], final_norm, (lm_head)`.
#[derive(Clone)]
struct Lm {
    f: Vec<Vec<f64>>,
}

fn ffn_and_norms(r: &mut Lcg, f: &mut Vec<Vec<f64>>) {
    f.extend([r.vec(D, 0.5, 1.5), r.vec(INTER * D, -0.5, 0.5)]);
    f.extend([r.vec(INTER * D, -0.5, 0.5), r.vec(D * INTER, -0.5, 0.5)]);
}

impl Lm {
    fn random(seed: u64, tied: bool) -> Self {
        let mut r = Lcg(seed);
        let (cd, vd, nv) = (GDN.conv_dim(), GDN.v_dim(), GDN.num_v_heads);
        let (qd, kvd) = (ATT.q_dim(), ATT.kv_dim());
        let mut f = vec![r.vec(VOCAB * D, -1.0, 1.0), r.vec(D, 0.5, 1.5)];
        f.extend([
            r.vec(cd * D, -0.5, 0.5),
            r.vec(vd * D, -0.5, 0.5),
            r.vec(nv * D, -0.5, 0.5),
            r.vec(nv * D, -0.5, 0.5),
            r.vec(nv, -1.5, -0.2),
            r.vec(nv, -0.5, 0.5),
            r.vec(cd * GDN.conv_kernel, -0.6, 0.6),
            r.vec(GDN.head_v_dim, 0.5, 1.5),
            r.vec(D * vd, -0.5, 0.5),
        ]);
        ffn_and_norms(&mut r, &mut f);
        f.push(r.vec(D, 0.5, 1.5));
        f.extend([
            r.vec(2 * qd * D, -0.5, 0.5),
            r.vec(kvd * D, -0.5, 0.5),
            r.vec(kvd * D, -0.5, 0.5),
            r.vec(ATT.head_dim, 0.5, 1.5),
            r.vec(ATT.head_dim, 0.5, 1.5),
            r.vec(D * qd, -0.5, 0.5),
        ]);
        ffn_and_norms(&mut r, &mut f);
        f.push(r.vec(D, 0.5, 1.5));
        if !tied {
            f.push(r.vec(VOCAB * D, -1.0, 1.0));
        }
        assert_eq!(f.len(), FINAL + 1 + usize::from(!tied));
        Self { f }
    }

    fn lm(&self) -> Qwen35LmRef<'_, f64> {
        let f = &self.f;
        let ffn = |i: usize| SwiGluWeights { gate: &f[i], up: &f[i + 1], down: &f[i + 2] };
        let gdn = Qwen35LayerRef {
            attn_norm: &f[1],
            mixer: Qwen35Mixer::Gdn(
                GdnWeights {
                    qkv: &f[2],
                    gate: &f[3],
                    alpha: &f[4],
                    beta: &f[5],
                    a: &f[6],
                    dt_bias: &f[7],
                    conv: &f[8],
                    norm: &f[9],
                    out: &f[10],
                },
                GDN,
            ),
            post_norm: &f[11],
            ffn: ffn(12),
        };
        let attn = Qwen35LayerRef {
            attn_norm: &f[L1],
            mixer: Qwen35Mixer::Attention(
                GatedAttnWeights {
                    q: &f[L1 + 1],
                    k: &f[L1 + 2],
                    v: &f[L1 + 3],
                    q_norm: &f[L1 + 4],
                    k_norm: &f[L1 + 5],
                    out: &f[L1 + 6],
                },
                ATT,
            ),
            post_norm: &f[L1 + 7],
            ffn: ffn(L1 + 8),
        };
        Qwen35LmRef {
            embed: &f[0],
            layers: vec![gdn, attn],
            final_norm: &f[FINAL],
            lm_head: f.get(FINAL + 1).map(Vec::as_slice),
            eps: 1e-6,
        }
    }

    fn loss(&self) -> f64 {
        self.lm().loss(&TOKENS, &TARGETS)
    }
}

/// Flatten grads in [`Lm`]'s field order.
fn flat(g: Qwen35LmGrads<f64>) -> Vec<Vec<f64>> {
    let mut out = vec![g.embed];
    for l in g.layers {
        out.push(l.attn_norm);
        match l.mixer {
            Qwen35MixerGrads::Gdn(m) => {
                out.extend([m.qkv, m.gate, m.alpha, m.beta, m.a, m.dt_bias, m.conv, m.norm, m.out])
            }
            Qwen35MixerGrads::Attention(a) => {
                out.extend([a.q, a.k, a.v, a.q_norm, a.k_norm, a.out]);
            }
        }
        out.extend([l.post_norm, l.ffn_gate, l.ffn_up, l.ffn_down]);
    }
    out.push(g.final_norm);
    out.extend(g.lm_head);
    out
}

fn gradcheck(m: &Lm, at: &str) {
    const H: f64 = 1e-6;
    let (loss, g) = m.lm().loss_and_grads(&TOKENS, &TARGETS);
    assert!((loss - m.loss()).abs() <= 1e-12, "{at}: loss_and_grads loss ≠ loss");
    let got = flat(g);
    assert_eq!(got.len(), m.f.len(), "{at}: one gradient per weight");
    for fi in 0..m.f.len() {
        let mut c = m.clone();
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
        assert!(norm(&want) > 1e-5, "{at}: weight {fi} gradient ~0, the check would be vacuous");
        assert_eq!(got[fi].len(), want.len(), "{at}: weight {fi} shape");
        let diff: Vec<f64> = got[fi].iter().zip(&want).map(|(x, y)| x - y).collect();
        let rel = norm(&diff) / norm(&want);
        assert!(rel <= 1e-3, "{at}: weight {fi} rel err {rel:e} > 1e-3");
    }
}

/// FALSIFY-QTG-003 at model scale: mean next-token cross-entropy through embedding,
/// a GDN layer, an attention layer, final norm and `lm_head` — every weight passes f64
/// gradcheck, with a separate `lm_head` and tied to the embedding.
#[test]
fn falsify_qtg_003_lm_gradcheck_f64() {
    for tied in [false, true] {
        for seed in [8, 21] {
            gradcheck(&Lm::random(seed, tied), &format!("tied={tied} seed {seed}"));
        }
    }
}

/// The gradient is a descent direction: one small step against it lowers the loss.
#[test]
fn sgd_step_lowers_loss() {
    let mut m = Lm::random(13, true);
    let before = m.loss();
    let g = flat(m.lm().loss_and_grads(&TOKENS, &TARGETS).1);
    for (w, gw) in m.f.iter_mut().zip(&g) {
        for (x, d) in w.iter_mut().zip(gw) {
            *x -= 0.05 * d;
        }
    }
    let after = m.loss();
    assert!(after < before, "loss {before} → {after} after a step down the gradient");
}

/// A real-vocabulary-sized row whose tail sits below half an f32 ulp of the head: a
/// plain f32 softmax denominator drops the whole tail (loss off by ~3.8e-3); the
/// compensated sum keeps f32 cross-entropy within 1e-6 of f64.
#[test]
fn f32_cross_entropy_keeps_a_large_vocab_tail() {
    let vocab = 250_000;
    let mut row = vec![-18.0_f32; vocab];
    row[0] = 0.0;
    let ours = cross_entropy(&row, &[0], vocab, None);
    let row64: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
    let want = cross_entropy(&row64, &[0], vocab, None);
    assert!(want > 3e-3, "the tail carries {want} of loss, else the check is vacuous");
    assert!((f64::from(ours) - want).abs() <= 1e-6, "f32 {ours} vs f64 {want}");
}
