//! Proof-obligation bodies for `contracts/qwen35-hybrid-forward-v1.yaml` (#3091).
//!
//! Every test drives the real Qwen3.5 sublayers (`Qwen35Model::forward_attention`,
//! `Qwen35Model::forward_deltanet`, `Qwen35Model::forward_single_qwen35`) on a tiny seeded
//! F32 model built in memory: no GGUF file. Obligation ids follow PR #3316.
//!
//! Isolating one sublayer: `forward_attention` and `forward_deltanet` each run the
//! mixer sublayer AND the FFN sublayer, with no entry point for either alone. A sublayer is
//! isolated by zeroing the OTHER sublayer's output projection (`ffn_down`, `ssm_out`,
//! `attn_output`), which makes that sublayer add exactly `0.0` to every coordinate.

use super::{
    Qwen35Model, Qwen35OwnedAttentionLayer, Qwen35OwnedDeltaNetLayer, Qwen35OwnedLayer, Qwen35State,
};
use crate::error::Result;
use crate::gguf::{GGUFConfig, OwnedQuantizedModel, OwnedQuantizedTensor};

/// QHF-CON-007 `tolerance: 1.0e-06`, the only tolerance the contract states.
const CONTRACT_TOL: f32 = 1.0e-6;

/// Shapes of one synthetic model. Every inner width differs from `hidden`, so a projection
/// that fails to restore `d_model` cannot pass by coincidence.
#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    /// Attention head width (`attn_q_norm.len()`), deliberately != hidden / num_heads.
    attn_head_dim: usize,
    intermediate: usize,
    /// Gated `DeltaNet` head width (k and v share it, as the loader does).
    gdn_head_dim: usize,
    /// Gated `DeltaNet` head count (k and v, so `q`, `k` and `v` have equal widths).
    gdn_heads: usize,
    vocab: usize,
}

const DIMS: [Dims; 2] = [
    Dims {
        hidden: 8,
        num_heads: 2,
        num_kv_heads: 1,
        attn_head_dim: 6,
        intermediate: 12,
        gdn_head_dim: 4,
        gdn_heads: 2,
        vocab: 5,
    },
    Dims {
        hidden: 12,
        num_heads: 4,
        num_kv_heads: 2,
        attn_head_dim: 4,
        intermediate: 20,
        gdn_head_dim: 2,
        gdn_heads: 3,
        vocab: 7,
    },
];

/// `Qwen35State` allocates `3 * conv_dim` of conv window, i.e. kernel 4 is baked in.
const CONV_KERNEL: usize = 4;
const SEQ_LEN: usize = 3;
const INPUT_SEEDS: [u64; 3] = [11, 23, 57];

impl Dims {
    fn gdn_width(self) -> usize {
        self.gdn_head_dim * self.gdn_heads
    }
    fn conv_dim(self) -> usize {
        self.gdn_width() * 3
    }
}

/// Deterministic xorshift stream of floats in [-0.5, 0.5).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // 24 high bits -> [0, 1), exactly representable in f32.
        ((self.0 >> 40) as f32) / 16_777_216.0 - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
    /// Strictly positive norm weights in [0.5, 1.5).
    fn norm_weight(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() + 1.0).collect()
    }
}

fn f32_tensor(rng: &mut Rng, in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    OwnedQuantizedTensor {
        data: rng
            .vec(in_dim * out_dim)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        in_dim,
        out_dim,
        qtype: crate::gguf::types::GGUF_TYPE_F32,
    }
}

fn zero_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    OwnedQuantizedTensor {
        data: vec![0u8; in_dim * out_dim * 4],
        in_dim,
        out_dim,
        qtype: crate::gguf::types::GGUF_TYPE_F32,
    }
}

fn base_model(dims: Dims) -> OwnedQuantizedModel {
    let mut rng = Rng::new(1);
    let config = GGUFConfig {
        architecture: "qwen35".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("qwen35"),
        hidden_dim: dims.hidden,
        num_layers: 0,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        vocab_size: dims.vocab,
        intermediate_dim: dims.intermediate,
        context_length: 32,
        rope_theta: 10_000.0,
        // RMSNorm is exactly scale-invariant only as eps -> 0; 1e-12 vanishes against the
        // O(0.1) mean squares these tests feed it.
        eps: 1.0e-12,
        rope_type: 2,
        explicit_head_dim: Some(dims.attn_head_dim),
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    OwnedQuantizedModel {
        config,
        token_embedding: rng.vec(dims.vocab * dims.hidden),
        position_embedding: None,
        layers: vec![],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: rng.norm_weight(dims.hidden),
        output_norm_bias: None,
        lm_head_weight: f32_tensor(&mut rng, dims.hidden, dims.vocab),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

fn attention_layer(dims: Dims, seed: u64) -> Qwen35OwnedAttentionLayer {
    let mut r = Rng::new(seed);
    let (h, hd) = (dims.hidden, dims.attn_head_dim);
    Qwen35OwnedAttentionLayer {
        attn_norm: r.norm_weight(h),
        attn_q: f32_tensor(&mut r, h, dims.num_heads * hd * 2),
        attn_k: f32_tensor(&mut r, h, dims.num_kv_heads * hd),
        attn_v: f32_tensor(&mut r, h, dims.num_kv_heads * hd),
        attn_q_norm: r.norm_weight(hd),
        attn_k_norm: r.norm_weight(hd),
        attn_output: f32_tensor(&mut r, dims.num_heads * hd, h),
        post_attention_norm: r.norm_weight(h),
        ffn_gate: f32_tensor(&mut r, h, dims.intermediate),
        ffn_up: f32_tensor(&mut r, h, dims.intermediate),
        ffn_down: f32_tensor(&mut r, dims.intermediate, h),
    }
}

fn deltanet_layer(dims: Dims, seed: u64) -> Qwen35OwnedDeltaNetLayer {
    let mut r = Rng::new(seed);
    let h = dims.hidden;
    let heads = dims.gdn_heads;
    Qwen35OwnedDeltaNetLayer {
        attn_norm: r.norm_weight(h),
        attn_qkv: f32_tensor(&mut r, h, dims.conv_dim()),
        attn_gate: f32_tensor(&mut r, h, dims.gdn_width()),
        ssm_alpha: f32_tensor(&mut r, h, heads),
        ssm_beta: f32_tensor(&mut r, h, heads),
        // A = -exp(A_log) < 0, so the state decays.
        ssm_a: r.vec(heads).iter().map(|v| -(v + 0.6)).collect(),
        ssm_dt_bias: r.vec(heads),
        ssm_conv1d_weight: r.vec(CONV_KERNEL * dims.conv_dim()),
        ssm_norm_weight: r.norm_weight(dims.gdn_head_dim),
        ssm_out: f32_tensor(&mut r, dims.gdn_width(), h),
        post_attention_norm: r.norm_weight(h),
        ffn_gate: f32_tensor(&mut r, h, dims.intermediate),
        ffn_up: f32_tensor(&mut r, h, dims.intermediate),
        ffn_down: f32_tensor(&mut r, dims.intermediate, h),
    }
}

fn model(base: &OwnedQuantizedModel, dims: Dims, layers: Vec<Qwen35OwnedLayer>) -> Qwen35Model<'_> {
    Qwen35Model {
        base,
        layers,
        head_dim: dims.attn_head_dim,
        num_kv_heads: dims.num_kv_heads,
        num_v_heads: dims.gdn_heads,
        head_v_dim: dims.gdn_head_dim,
        num_k_heads: dims.gdn_heads,
        head_k_dim: dims.gdn_head_dim,
        conv_kernel: CONV_KERNEL,
        // n_rot = 2 * 2 = 4 <= every attn_head_dim above.
        rope_sections: [1, 1, 0, 0],
    }
}

/// One whole block (mixer sublayer then FFN sublayer) of layer `il` at `position`.
fn run_block(
    m: &Qwen35Model<'_>,
    il: usize,
    h: &[f32],
    state: &mut Qwen35State,
    position: usize,
) -> Result<Vec<f32>> {
    let hd = m.base.config.hidden_dim;
    let mut out = h.to_vec();
    let mut normed = vec![0.0; hd];
    let mut post = vec![0.0; hd];
    match &m.layers[il] {
        Qwen35OwnedLayer::DeltaNet(d) => {
            m.forward_deltanet(d, &mut out, state, il, position, &mut normed, &mut post)?;
        },
        Qwen35OwnedLayer::Attention(a) => {
            m.forward_attention(a, &mut out, state, il, position, &mut normed, &mut post)?;
        },
    }
    Ok(out)
}

/// Run `inputs` (one per position) through the single layer of `m` from a fresh state and
/// return each position's residual delta `h_out - h_in`.
fn deltas_over_sequence(m: &Qwen35Model<'_>, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    let mut state = m.new_state(inputs.len() + 1);
    let mut deltas = Vec::with_capacity(inputs.len());
    for (pos, h) in inputs.iter().enumerate() {
        let out = run_block(m, 0, h, &mut state, pos)?;
        assert_eq!(out.len(), h.len(), "block output width at position {pos}");
        deltas.push(out.iter().zip(h).map(|(o, i)| o - i).collect());
        state.kv_cache.advance();
    }
    Ok(deltas)
}

fn inputs(dims: Dims, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng::new(seed);
    (0..SEQ_LEN).map(|_| r.vec(dims.hidden)).collect()
}

/// The sublayer wrote a finite value into EVERY one of the `d_model` coordinates. A
/// sublayer whose output is narrower than `d_model` leaves a coordinate's delta at exactly
/// zero (or panics on the width mismatch); seeded dense weights make an honest exact zero
/// vanishingly unlikely.
fn assert_fills_d_model(deltas: &[Vec<f32>], dims: Dims, what: &str) {
    for (pos, d) in deltas.iter().enumerate() {
        assert_eq!(d.len(), dims.hidden, "{what}: width at position {pos}");
        for (i, v) in d.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{what}: coordinate {i} at position {pos} is {v}"
            );
            assert!(
                *v != 0.0,
                "{what}: coordinate {i} of d_model={} never received the sublayer output \
                 (position {pos}, deltas {d:?})",
                dims.hidden
            );
        }
    }
}

// --- independent reference math (QHF-CON-007 oracle) --------------------------------------

fn weights(t: &OwnedQuantizedTensor) -> Vec<f32> {
    t.data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn matvec(t: &OwnedQuantizedTensor, x: &[f32]) -> Vec<f32> {
    let w = weights(t);
    (0..t.out_dim)
        .map(|r| (0..t.in_dim).map(|c| w[r * t.in_dim + c] * x[c]).sum())
        .collect()
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `swiglu(rmsnorm(x))`: down(silu(gate(n)) * up(n)).
fn ref_ffn(
    h: &[f32],
    norm: &[f32],
    gate: &OwnedQuantizedTensor,
    up: &OwnedQuantizedTensor,
    down: &OwnedQuantizedTensor,
    eps: f32,
) -> Vec<f32> {
    let n = rmsnorm(h, norm, eps);
    let g = matvec(gate, &n);
    let u = matvec(up, &n);
    let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| u * g * sigmoid(*g)).collect();
    matvec(down, &act)
}

/// Gated attention sublayer at position 0 of a fresh cache: one key, so the softmax weight
/// is 1 and each head's output is its KV group's value, times sigmoid(gate), projected.
fn ref_attention_pos0(dims: Dims, a: &Qwen35OwnedAttentionLayer, h: &[f32], eps: f32) -> Vec<f32> {
    let hd = dims.attn_head_dim;
    let group = dims.num_heads / dims.num_kv_heads;
    let n = rmsnorm(h, &a.attn_norm, eps);
    let q_full = matvec(&a.attn_q, &n);
    let v = matvec(&a.attn_v, &n);
    let mut out = vec![0.0; dims.num_heads * hd];
    for head in 0..dims.num_heads {
        let kv = head / group;
        for i in 0..hd {
            out[head * hd + i] = v[kv * hd + i] * sigmoid(q_full[head * hd * 2 + hd + i]);
        }
    }
    matvec(&a.attn_output, &out)
}

/// Gated `DeltaNet` sublayer at position 0 of a fresh state: the conv window and the
/// recurrent state are zero, so conv = last tap * input and `S' = k (beta v)^T`.
fn ref_deltanet_pos0(dims: Dims, d: &Qwen35OwnedDeltaNetLayer, h: &[f32], eps: f32) -> Vec<f32> {
    let (hd, w) = (dims.gdn_head_dim, dims.gdn_width());
    let n = rmsnorm(h, &d.attn_norm, eps);
    let conv_in = matvec(&d.attn_qkv, &n);
    let conv: Vec<f32> = conv_in
        .iter()
        .enumerate()
        .map(|(c, x)| {
            let y = x * d.ssm_conv1d_weight[c * CONV_KERNEL + CONV_KERNEL - 1];
            y * sigmoid(y)
        })
        .collect();
    let l2 = |x: &[f32]| -> Vec<f32> {
        x.chunks_exact(hd)
            .flat_map(|c| {
                let inv = 1.0 / (c.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
                c.iter().map(move |v| v * inv)
            })
            .collect()
    };
    let (q, k, v) = (l2(&conv[..w]), l2(&conv[w..2 * w]), &conv[2 * w..]);
    let beta: Vec<f32> = matvec(&d.ssm_beta, &n).into_iter().map(sigmoid).collect();
    let gate = matvec(&d.attn_gate, &n);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut normed = vec![0.0; w];
    for head in 0..dims.gdn_heads {
        let r = head * hd..(head + 1) * hd;
        let kq: f32 = k[r.clone()]
            .iter()
            .zip(&q[r.clone()])
            .map(|(a, b)| a * b)
            .sum();
        let out: Vec<f32> = v[r.clone()]
            .iter()
            .map(|vj| vj * beta[head] * kq * scale)
            .collect();
        let inv = 1.0 / (out.iter().map(|x| x * x).sum::<f32>() / hd as f32 + eps).sqrt();
        for (j, o) in out.iter().enumerate() {
            let g = gate[head * hd + j];
            normed[head * hd + j] = o * inv * d.ssm_norm_weight[j] * g * sigmoid(g);
        }
    }
    matvec(&d.ssm_out, &normed)
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: width");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= CONTRACT_TOL,
            "{what}: coordinate {i}: h_(l+1) - h_l = {g}, sublayer(norm(h_l)) = {w} \
             (|diff| {} > {CONTRACT_TOL})\n got {got:?}\nwant {want:?}",
            (g - w).abs()
        );
    }
}

// --- QHF-INV-001 / 002 / 003: shape preservation ------------------------------------------

/// QHF-INV-001 `∀x: shape(attention_sublayer(x)) = shape(x)` (FALSIFY-QHF-001), over two
/// `d_model`s, three seeded inputs and a three-token sequence (so the KV cache grows). The
/// attention inner width (`num_heads * attn_head_dim`) differs from `d_model` in both
/// configs; the FFN is zeroed so the delta is the attention sublayer alone.
#[test]
fn qhf_inv_001_attention_sublayer_preserves_d_model() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let mut layer = attention_layer(dims, 101);
        layer.ffn_down = zero_tensor(dims.intermediate, dims.hidden);
        let m = model(&base, dims, vec![Qwen35OwnedLayer::Attention(layer)]);
        for seed in INPUT_SEEDS {
            let deltas = deltas_over_sequence(&m, &inputs(dims, seed))?;
            assert_fills_d_model(&deltas, dims, "QHF-INV-001 attention sublayer");
        }
    }
    Ok(())
}

/// QHF-INV-002 `∀x: shape(gdn_sublayer(x)) = shape(x)` (FALSIFY-QHF-002). The Gated
/// `DeltaNet` value width (`gdn_heads * gdn_head_dim`) differs from `d_model` in the second
/// config; the FFN is zeroed so the delta is the GDN sublayer alone.
#[test]
fn qhf_inv_002_gdn_sublayer_preserves_d_model() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let mut layer = deltanet_layer(dims, 202);
        layer.ffn_down = zero_tensor(dims.intermediate, dims.hidden);
        let m = model(&base, dims, vec![Qwen35OwnedLayer::DeltaNet(layer)]);
        for seed in INPUT_SEEDS {
            let deltas = deltas_over_sequence(&m, &inputs(dims, seed))?;
            assert_fills_d_model(&deltas, dims, "QHF-INV-002 GDN sublayer");
        }
    }
    Ok(())
}

/// QHF-INV-003 `∀x: shape(ffn_sublayer(x)) = shape(x)` (FALSIFY-QHF-003), in BOTH layer
/// types (the contract's `ffn_sublayer` is shared). `intermediate != d_model`; the mixer's
/// output projection is zeroed so the delta is the FFN sublayer alone.
#[test]
fn qhf_inv_003_ffn_sublayer_preserves_d_model() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let mut attn = attention_layer(dims, 303);
        attn.attn_output = zero_tensor(dims.num_heads * dims.attn_head_dim, dims.hidden);
        let mut gdn = deltanet_layer(dims, 304);
        gdn.ssm_out = zero_tensor(dims.gdn_width(), dims.hidden);
        for (layer, what) in [
            (
                Qwen35OwnedLayer::Attention(attn),
                "QHF-INV-003 FFN sublayer (attention layer)",
            ),
            (
                Qwen35OwnedLayer::DeltaNet(gdn),
                "QHF-INV-003 FFN sublayer (GDN layer)",
            ),
        ] {
            let m = model(&base, dims, vec![layer]);
            for seed in INPUT_SEEDS {
                let deltas = deltas_over_sequence(&m, &inputs(dims, seed))?;
                assert_fills_d_model(&deltas, dims, what);
            }
        }
    }
    Ok(())
}

// --- QHF-INV-004: exactly one mixer per layer ---------------------------------------------

/// QHF-INV-004 `∀l: is_attention(l) XOR is_gdn(l)` (FALSIFY-QHF-004), observed by EFFECT
/// through the full `forward_single_qwen35`: after a two-token run, a layer that ran
/// attention has written its KV cache, a layer that ran Gated `DeltaNet` has written its
/// conv window / recurrent state, and no layer has written both or neither. The schedule
/// is Qwen3.5's (every 4th layer is attention) over two periods.
#[test]
fn qhf_inv_004_each_layer_runs_exactly_one_mixer() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let layers: Vec<Qwen35OwnedLayer> = (0..8u64)
            .map(|l| {
                if (l + 1) % 4 == 0 {
                    Qwen35OwnedLayer::Attention(attention_layer(dims, 400 + l))
                } else {
                    Qwen35OwnedLayer::DeltaNet(deltanet_layer(dims, 400 + l))
                }
            })
            .collect();
        let m = model(&base, dims, layers);
        let mut state = m.new_state(4);
        for (pos, token) in [1u32, 3].into_iter().enumerate() {
            let logits = m.forward_single_qwen35(token, &mut state, pos)?;
            assert_eq!(logits.len(), dims.vocab);
            assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
        }
        for (l, layer) in m.layers.iter().enumerate() {
            let ran_attention = !state.kv_cache.get_k(l).is_empty();
            let ran_gdn = state.conv_states[l].iter().any(|v| *v != 0.0)
                || state.ssm_states[l].iter().any(|v| *v != 0.0);
            assert!(
                ran_attention ^ ran_gdn,
                "QHF-INV-004 layer {l}: ran attention = {ran_attention}, ran GDN = {ran_gdn}"
            );
            let is_attention = matches!(layer, Qwen35OwnedLayer::Attention(_));
            assert_eq!(
                ran_attention, is_attention,
                "QHF-INV-004 layer {l}: the mixer that ran is not the layer's type"
            );
        }
    }
    Ok(())
}

// --- QHF-INV-006: pre-norm order ----------------------------------------------------------

/// QHF-INV-006 check for one single-layer model: the residual delta for `c * h` equals the
/// delta for `h` at every position and coordinate, for c = 4 and c = 1/4.
fn assert_delta_scale_invariant(m: &Qwen35Model<'_>, xs: &[Vec<f32>], what: &str) -> Result<()> {
    let reference = deltas_over_sequence(m, xs)?;
    for c in [4.0f32, 0.25] {
        let scaled: Vec<Vec<f32>> = xs
            .iter()
            .map(|x| x.iter().map(|v| v * c).collect())
            .collect();
        let got = deltas_over_sequence(m, &scaled)?;
        for (pos, (g, r)) in got.iter().zip(&reference).enumerate() {
            for (i, (a, b)) in g.iter().zip(r).enumerate() {
                assert!(
                    (a - b).abs() <= CONTRACT_TOL,
                    "QHF-INV-006 {what}: delta for {c} * h differs from delta for h at position \
                     {pos}, coordinate {i}: {a} vs {b}; the sublayer does not read RMSNorm(h)"
                );
            }
        }
    }
    Ok(())
}

/// QHF-INV-006 `pre-norm architecture: norm before attention/GDN and before FFN`
/// (FALSIFY-QHF-006), by the EFFECT of the order: RMSNorm is invariant to a positive scale
/// of its input, so a pre-normed sublayer adds the same delta for `c * h` as for `h`. A
/// sublayer that reads the raw residual (norm missing, or moved after it) adds a delta
/// that changes with `c`. Checked for each sublayer on its own (the other zeroed), in both
/// layer types, over a three-token sequence. c = 4 and 1/4 are exact in binary floating
/// point, so the norm of the scaled input is bit-identical; the bound is CONTRACT_TOL.
#[test]
fn qhf_inv_006_rmsnorm_precedes_each_sublayer() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let hd_attn = dims.num_heads * dims.attn_head_dim;
        let mut attn_only = attention_layer(dims, 601);
        attn_only.ffn_down = zero_tensor(dims.intermediate, dims.hidden);
        let mut gdn_only = deltanet_layer(dims, 602);
        gdn_only.ffn_down = zero_tensor(dims.intermediate, dims.hidden);
        let mut ffn_in_attn = attention_layer(dims, 603);
        ffn_in_attn.attn_output = zero_tensor(hd_attn, dims.hidden);
        let mut ffn_in_gdn = deltanet_layer(dims, 604);
        ffn_in_gdn.ssm_out = zero_tensor(dims.gdn_width(), dims.hidden);
        for (layer, what) in [
            (Qwen35OwnedLayer::Attention(attn_only), "attention sublayer"),
            (Qwen35OwnedLayer::DeltaNet(gdn_only), "GDN sublayer"),
            (
                Qwen35OwnedLayer::Attention(ffn_in_attn),
                "FFN sublayer (attention layer)",
            ),
            (
                Qwen35OwnedLayer::DeltaNet(ffn_in_gdn),
                "FFN sublayer (GDN layer)",
            ),
        ] {
            let m = model(&base, dims, vec![layer]);
            for seed in INPUT_SEEDS {
                assert_delta_scale_invariant(&m, &inputs(dims, seed), what)?;
            }
        }
    }
    Ok(())
}

// --- QHF-CON-007: residual identity -------------------------------------------------------

/// QHF-CON-007 `h_{l+1} - h_l = sublayer(norm(h_l))`, `tolerance: 1.0e-06`
/// (FALSIFY-QHF-007). The residual delta the code produces is compared with an
/// independent evaluation of `sublayer(norm(h_l))` from the contract's equations
/// (`attention_sublayer`, `gdn_sublayer`, `ffn_sublayer`): each sublayer isolated, then the
/// whole block `h + mixer(norm(h)) + ffn(norm(h + mixer(norm(h))))`. Position 0 of a fresh
/// state, where the attention softmax and the delta-rule state have closed forms.
#[test]
fn qhf_con_007_residual_identity() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let eps = base.config.eps;
        for seed in INPUT_SEEDS {
            let h = &inputs(dims, seed)[..1];
            let h0 = &h[0];

            // Attention layer.
            let attn = attention_layer(dims, 701);
            let mixer = ref_attention_pos0(dims, &attn, h0, eps);
            let mid: Vec<f32> = h0.iter().zip(&mixer).map(|(a, b)| a + b).collect();
            let ffn = ref_ffn(
                &mid,
                &attn.post_attention_norm,
                &attn.ffn_gate,
                &attn.ffn_up,
                &attn.ffn_down,
                eps,
            );
            let ffn_alone = ref_ffn(
                h0,
                &attn.post_attention_norm,
                &attn.ffn_gate,
                &attn.ffn_up,
                &attn.ffn_down,
                eps,
            );
            let block: Vec<f32> = mixer.iter().zip(&ffn).map(|(a, b)| a + b).collect();

            let mut only_mixer = attention_layer(dims, 701);
            only_mixer.ffn_down = zero_tensor(dims.intermediate, dims.hidden);
            let mut only_ffn = attention_layer(dims, 701);
            only_ffn.attn_output = zero_tensor(dims.num_heads * dims.attn_head_dim, dims.hidden);
            for (layer, want, what) in [
                (only_mixer, &mixer, "attention sublayer"),
                (only_ffn, &ffn_alone, "FFN sublayer (attention layer)"),
                (attn, &block, "attention block"),
            ] {
                let m = model(&base, dims, vec![Qwen35OwnedLayer::Attention(layer)]);
                let got = deltas_over_sequence(&m, h)?;
                assert_close(&got[0], want, &format!("QHF-CON-007 {what}"));
            }

            // Gated DeltaNet layer.
            let gdn = deltanet_layer(dims, 702);
            let mixer = ref_deltanet_pos0(dims, &gdn, h0, eps);
            let mid: Vec<f32> = h0.iter().zip(&mixer).map(|(a, b)| a + b).collect();
            let ffn = ref_ffn(
                &mid,
                &gdn.post_attention_norm,
                &gdn.ffn_gate,
                &gdn.ffn_up,
                &gdn.ffn_down,
                eps,
            );
            let ffn_alone = ref_ffn(
                h0,
                &gdn.post_attention_norm,
                &gdn.ffn_gate,
                &gdn.ffn_up,
                &gdn.ffn_down,
                eps,
            );
            let block: Vec<f32> = mixer.iter().zip(&ffn).map(|(a, b)| a + b).collect();

            let mut only_mixer = deltanet_layer(dims, 702);
            only_mixer.ffn_down = zero_tensor(dims.intermediate, dims.hidden);
            let mut only_ffn = deltanet_layer(dims, 702);
            only_ffn.ssm_out = zero_tensor(dims.gdn_width(), dims.hidden);
            for (layer, want, what) in [
                (only_mixer, &mixer, "GDN sublayer"),
                (only_ffn, &ffn_alone, "FFN sublayer (GDN layer)"),
                (gdn, &block, "GDN block"),
            ] {
                let m = model(&base, dims, vec![Qwen35OwnedLayer::DeltaNet(layer)]);
                let got = deltas_over_sequence(&m, h)?;
                assert_close(&got[0], want, &format!("QHF-CON-007 {what}"));
            }
        }
    }
    Ok(())
}
