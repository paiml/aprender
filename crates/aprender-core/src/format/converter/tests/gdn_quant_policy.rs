//! R8 (la-0.72): the GDN quantize policy — `contracts/gdn-quantize-policy-v1.yaml`.
//!
//! Three parts: the name table (FALSIFY-GQP-001), every converter rule obeys
//! it (FALSIFY-GQP-002), and a cosine gate on the dequantized output of a GDN
//! layer, with a planted falsifier per protected tensor (FALSIFY-GQP-003/004).

use super::super::*;

// ============================================================================
// FALSIFY-GQP-001: the name table
// ============================================================================

const KEEP: &[&str] = &[
    "model.layers.0.linear_attn.conv1d.weight",
    "model.language_model.layers.3.linear_attn.conv1d.weight",
    "model.layers.0.linear_attn.A_log",
    "model.layers.0.linear_attn.dt_bias",
    "blk.0.ssm_conv1d.weight",
    "blk.12.ssm_a",
    "blk.0.ssm_dt.bias",
    "blk.0.ssm_dt_bias",
];

const QUANTIZE: &[&str] = &[
    "blk.0.ssm_alpha.weight", // a projection; `ssm_a` must not match it
    "blk.0.ssm_beta.weight",
    "blk.0.ssm_out.weight",
    "blk.0.ssm_dt.weight", // Mamba's dt projection, not the bias
    "model.layers.0.linear_attn.in_proj_qkvz.weight",
    "model.layers.0.linear_attn.in_proj_ba.weight",
    "model.layers.0.linear_attn.out_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "blk.3.attn_q.weight",
];

#[test]
fn falsify_gqp_001_name_table() {
    for n in KEEP {
        assert!(gdn_keeps_full_precision(n), "must keep F32: {n}");
    }
    for n in QUANTIZE {
        assert!(!gdn_keeps_full_precision(n), "must stay quantizable: {n}");
    }
}

// ============================================================================
// FALSIFY-GQP-002: every converter rule honours the table
// ============================================================================

/// A large 2-D shape, so size and rank never excuse a tensor: only the
/// name decides.
const BIG: [usize; 2] = [64, 4096];

#[test]
fn falsify_gqp_002_every_rule_keeps_the_gdn_tensors() {
    let len = BIG[0] * BIG[1];
    for n in KEEP {
        assert!(
            should_skip_quantization(n, len),
            "import/convert rule quantizes {n}"
        );
        assert!(
            !should_quantize_tensor(n, &BIG, len),
            "Q4K rule quantizes {n}"
        );
    }
    for n in QUANTIZE {
        assert!(
            !should_skip_quantization(n, len),
            "import/convert rule skips {n}"
        );
        assert!(should_quantize_tensor(n, &BIG, len), "Q4K rule skips {n}");
    }
}

#[test]
fn falsify_gqp_002_export_keeps_the_gdn_tensors_bit_exact() {
    let data: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.37).sin() * 0.3).collect();
    let mut m = BTreeMap::new();
    for n in KEEP.iter().chain(QUANTIZE) {
        m.insert((*n).to_string(), (data.clone(), vec![4, 256]));
    }
    let out = quantize_tensors(&NativeF32Tensors::new(m), &QuantizationType::Q4K).expect("q4k");
    for n in KEEP {
        assert_eq!(out.as_ref()[*n].0, data, "export fake-quantized {n}");
    }
    for n in QUANTIZE {
        assert_ne!(out.as_ref()[*n].0, data, "export left {n} unquantized");
    }
}

// ============================================================================
// FALSIFY-GQP-003/004: cosine gate on the dequantized GDN layer output
// ============================================================================

/// Deterministic uniform in [lo, hi).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        lo + (hi - lo) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(lo, hi)).collect()
    }
}

const T: usize = 256; // tokens: decay errors compound along the sequence
const D: usize = 256; // hidden
const H: usize = 16; // value heads
const HD: usize = 8; // channels per head
const C: usize = H * HD; // conv channels
const K: usize = 4; // conv taps

/// One GDN layer's weights, HF names, in the value ranges Qwen3.5 ships:
/// `A = exp(A_log)` in [1, 16], `dt = softplus(dt_bias)` in [0.001, 0.1].
fn layer() -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let mut r = Lcg(0x5eed_0072_0008);
    let a_log = r.vec(H, 0.0, 16f32.ln());
    let dt_bias = r
        .vec(H, 0.001, 0.1)
        .into_iter()
        .map(|dt| dt.exp_m1().ln())
        .collect();
    vec![
        (
            "linear_attn.in_proj_qkvz.weight",
            vec![C, D],
            r.vec(C * D, -0.06, 0.06),
        ),
        (
            "linear_attn.in_proj_ba.weight",
            vec![H, D],
            r.vec(H * D, -0.06, 0.06),
        ),
        (
            "linear_attn.conv1d.weight",
            vec![C, 1, K],
            r.vec(C * K, -0.5, 0.5),
        ),
        ("linear_attn.A_log", vec![H], a_log),
        ("linear_attn.dt_bias", vec![H], dt_bias),
    ]
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// The GDN layer's gated recurrence over T tokens, all states concatenated:
/// u = W_qkvz x, a = W_ba x, c = silu(causal depthwise conv1d(u)),
/// g = -exp(A_log) * softplus(a + dt_bias), s_t = exp(g_t) * s_{t-1} + c_t.
fn forward(w: &BTreeMap<&str, Vec<f32>>, x: &[f32]) -> Vec<f32> {
    let (wu, wa) = (
        &w["linear_attn.in_proj_qkvz.weight"],
        &w["linear_attn.in_proj_ba.weight"],
    );
    let (conv, a_log, dt_bias) = (
        &w["linear_attn.conv1d.weight"],
        &w["linear_attn.A_log"],
        &w["linear_attn.dt_bias"],
    );
    let matvec = |m: &[f32], rows: usize, xt: &[f32]| -> Vec<f32> {
        (0..rows)
            .map(|i| (0..D).map(|j| m[i * D + j] * xt[j]).sum())
            .collect()
    };
    let u: Vec<Vec<f32>> = (0..T)
        .map(|t| matvec(wu, C, &x[t * D..(t + 1) * D]))
        .collect();
    let mut s = vec![0f32; C];
    let mut out = Vec::with_capacity(T * C);
    for t in 0..T {
        let a = matvec(wa, H, &x[t * D..(t + 1) * D]);
        for ch in 0..C {
            let mut acc = 0f32;
            for k in 0..K {
                if let Some(src) = (t + k + 1).checked_sub(K) {
                    acc += conv[ch * K + k] * u[src][ch];
                }
            }
            let c = acc / (1.0 + (-acc).exp());
            let h = ch / HD;
            let g = -a_log[h].exp() * softplus(a[h] + dt_bias[h]);
            s[ch] = g.exp() * s[ch] + c;
            out.push(s[ch]);
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// Cosine of the layer output with each tensor Q4K round-tripped where
/// `quantize(name, shape, len)` says so, against the all-F32 output.
fn gate_cosine(quantize: impl Fn(&str, &[usize], usize) -> bool) -> f64 {
    let mut r = Lcg(0xfeed_0072);
    let x = r.vec(T * D, -1.0, 1.0);
    let f32w: BTreeMap<&str, Vec<f32>> = layer().into_iter().map(|(n, _, d)| (n, d)).collect();
    let qw: BTreeMap<&str, Vec<f32>> = layer()
        .into_iter()
        .map(|(n, shape, d)| {
            let d = if quantize(n, &shape, d.len()) {
                dequantize_q4_k_to_f32(&quantize_q4_k(&d), d.len())
            } else {
                d
            };
            (n, d)
        })
        .collect();
    cosine(&forward(&f32w, &x), &forward(&qw, &x))
}

/// The gate: the policy-quantized layer tracks the F32 layer.
const GATE: f64 = 0.998;

/// The planted falsifier's bar, fixed before measuring: Q4K on ONE protected
/// tensor (under 1% of the layer's bytes) must add at least half again the
/// damage `1 - cos` that Q4K on every projection does.
///
/// Measured (T=256): conv1d x2.48, dt_bias x3.02, A_log x1.21. A_log misses
/// this bar, so it is kept on the second ground instead: Q4K makes it
/// LARGER (a 16-float vector still costs one 144-byte super-block, 64 -> 144
/// bytes) while still adding damage. Quantizing it loses on both axes.
const PLANTED_DAMAGE_RATIO: f64 = 1.5;

/// The converter's Q4K rule, applied to the layer.
fn q4k_rule(n: &str, shape: &[usize], len: usize) -> bool {
    should_quantize_tensor(n, shape, len)
}

#[test]
fn falsify_gqp_003_policy_layer_clears_the_cosine_gate() {
    // The projections really are quantized, so the gate is not vacuous.
    let quantized: Vec<_> = layer()
        .into_iter()
        .filter(|(n, s, d)| q4k_rule(n, s, d.len()))
        .map(|(n, _, _)| n)
        .collect();
    assert_eq!(
        quantized,
        [
            "linear_attn.in_proj_qkvz.weight",
            "linear_attn.in_proj_ba.weight"
        ]
    );
    let cos = gate_cosine(q4k_rule);
    eprintln!("GQP-003 policy cosine = {cos:.6}");
    assert!(cos >= GATE, "policy layer cosine {cos:.6} < {GATE}");
}

/// The planted falsifier: override the policy for ONE protected tensor and
/// measure it alone, so each keep-list entry is justified by its own
/// measurement. Every one must add damage; each must either add at least
/// `PLANTED_DAMAGE_RATIO` of it or save no bytes by being quantized.
#[test]
fn falsify_gqp_004_quantizing_any_protected_tensor_turns_the_gate_red() {
    let base = 1.0 - gate_cosine(q4k_rule);
    let rows: Vec<_> = layer()
        .into_iter()
        .filter(|(n, _, _)| gdn_keeps_full_precision(n))
        .map(|(planted, _, d)| {
            let cos = gate_cosine(|n, s, len| n == planted || q4k_rule(n, s, len));
            let ratio = (1.0 - cos) / base;
            let (f32_b, q4k_b) = (d.len() * 4, quantize_q4_k(&d).len());
            eprintln!(
                "GQP-004 planted {planted}: cosine {cos:.6}, damage x{ratio:.2}, \
                 bytes f32 {f32_b} -> q4k {q4k_b}"
            );
            (planted, ratio, q4k_b >= f32_b)
        })
        .collect();
    assert_eq!(
        rows.len(),
        3,
        "the layer must carry all three protected tensors"
    );
    for (planted, ratio, no_saving) in rows {
        assert!(
            ratio > 1.0,
            "quantizing {planted} added no damage (x{ratio:.2})"
        );
        assert!(
            ratio >= PLANTED_DAMAGE_RATIO || no_saving,
            "quantizing {planted} cost only x{ratio:.2} the policy's damage and saves bytes"
        );
    }
}
