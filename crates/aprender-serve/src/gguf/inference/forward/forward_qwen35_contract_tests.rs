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

/// Qwen3.5's layer schedule (every 4th layer is full attention) over two periods, each layer
/// seeded `seed + l`.
fn hybrid_layers(dims: Dims, seed: u64) -> Vec<Qwen35OwnedLayer> {
    hybrid_schedule(dims, seed, 8)
}

/// Qwen3.5's layer schedule over `n` layers: every 4th is full attention, each seeded `seed + l`.
fn hybrid_schedule(dims: Dims, seed: u64, n: u64) -> Vec<Qwen35OwnedLayer> {
    (0..n)
        .map(|l| {
            if (l + 1) % 4 == 0 {
                Qwen35OwnedLayer::Attention(attention_layer(dims, seed + l))
            } else {
                Qwen35OwnedLayer::DeltaNet(deltanet_layer(dims, seed + l))
            }
        })
        .collect()
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
        let m = model(&base, dims, hybrid_layers(dims, 400));
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

// === contracts/qwen35-e2e-verification-v1.yaml (QE2E) ======================================
//
// Three of the seven obligations are unit-testable against this code and are discharged
// below: QE2E-ORD-003, QE2E-INV-006, QE2E-CON-007. QE2E-INV-001 (a 9B checkpoint's
// parameter count), QE2E-BND-002 (an O() bound with no constant), QE2E-MON-004 (a
// throughput claim) and QE2E-BND-005 (a coverage meta-claim) are not; see #3091.

/// Tokens that visit the first and the last embedding row.
fn token_sequence(dims: Dims, seq_len: usize) -> Vec<u32> {
    (0..seq_len)
        .map(|p| u32::try_from((p * 3 + 1) % dims.vocab).unwrap_or(0))
        .collect()
}

// --- QE2E-ORD-003: quantization memory ordering -------------------------------------------

/// QE2E-ORD-003 `M(Q4K) < M(Q6K) < M(F16) < M(F32)` (FALSIFY-QE2E-003, "Quantization byte
/// formula wrong"). The four byte counts are MEASURED, not read from block constants: the
/// same seeded tensor is encoded by the real encoders (`quantize_q4_k`, `quantize_q6_k`,
/// `f32_to_f16`, `f32::to_le_bytes`), the ordering is asserted on what they wrote, and then
/// each encoding is written into a GGUF and the loader's `get_tensor_ref` must reserve
/// exactly that many bytes, so the ordering holds for the bytes the serving path maps.
///
/// Domain: whole 256-element super-blocks, the unit ggml's K-quant encoders are defined on.
/// The contract's formal states no domain; below one super-block the padded K-quant blocks
/// are LARGER than F16 (see #3091), so the formal is false for tiny tensors as written.
#[test]
fn qe2e_ord_003_quantization_memory_ordering() -> Result<()> {
    use crate::gguf::test_factory::GGUFBuilder;
    use crate::gguf::{GGUFModel, QuantizedGGUFTransformer};

    for (i, dims) in [vec![256u64], vec![1024], vec![256, 512]]
        .into_iter()
        .enumerate()
    {
        let n: usize = dims
            .iter()
            .map(|&d| usize::try_from(d).unwrap_or(0))
            .product();
        let data = Rng::new(31 + i as u64).vec(n);
        let q4k = trueno_quant::quantize_q4_k(&data);
        let q6k = trueno_quant::quantize_q6_k(&data);
        let f16: Vec<u8> = data
            .iter()
            .flat_map(|v| trueno_quant::f32_to_f16(*v).to_le_bytes())
            .collect();
        let f32_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let measured = [
            ("Q4K", q4k.len()),
            ("Q6K", q6k.len()),
            ("F16", f16.len()),
            ("F32", f32_bytes.len()),
        ];
        for pair in measured.windows(2) {
            assert!(
                pair[0].1 < pair[1].1,
                "QE2E-ORD-003 dims {dims:?}: M({}) = {} bytes is not < M({}) = {} bytes",
                pair[0].0,
                pair[0].1,
                pair[1].0,
                pair[1].1
            );
        }

        let file = GGUFBuilder::new()
            .add_q4_k_tensor("q4k", &dims, &q4k)
            .add_q6_k_tensor("q6k", &dims, &q6k)
            .add_f16_tensor("f16", &dims, &f16)
            .add_f32_tensor("f32", &dims, &data)
            .build();
        let gguf = GGUFModel::from_bytes(&file)?;
        for (name, (label, written)) in ["q4k", "q6k", "f16", "f32"].into_iter().zip(measured) {
            let reserved = QuantizedGGUFTransformer::get_tensor_ref(&gguf, &file, name)?.byte_size;
            assert_eq!(
                reserved, written,
                "QE2E-ORD-003 dims {dims:?}: the loader maps {reserved} bytes for the {label} \
                 tensor, the encoder wrote {written}"
            );
        }
    }
    Ok(())
}

// --- QE2E-INV-006: every block preserves shape ---------------------------------------------

/// QE2E-INV-006 `∀l: shape(block_l(x)) = shape(x)` (FALSIFY-QE2E-006). The residual stream
/// of a real token is carried through the eight-layer hybrid stack ONE BLOCK AT A TIME: at
/// every layer `l` the block's output must be `d_model` wide, finite, and must have written
/// every one of the `d_model` coordinates (a block whose output is narrower than `x` leaves
/// a coordinate's delta at exactly zero). The hidden buffer is a slice, so its LENGTH cannot
/// change; the width a block actually writes is the observable shape.
///
/// So that the `x` fed to each block is the model's real `h_l` and not a test artefact, the
/// block-by-block chain must reproduce `forward_single_qwen35`'s logits bit for bit (same
/// code, same inputs; no tolerance is involved).
#[test]
fn qe2e_inv_006_every_block_preserves_d_model() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let m = model(&base, dims, hybrid_layers(dims, 800));
        let tokens = token_sequence(dims, SEQ_LEN);
        let mut by_block = m.new_state(SEQ_LEN);
        let mut whole = m.new_state(SEQ_LEN);
        for (pos, &token) in tokens.iter().enumerate() {
            let t = token as usize;
            let mut h = base.token_embedding[t * dims.hidden..(t + 1) * dims.hidden].to_vec();
            for l in 0..m.layers.len() {
                let out = run_block(&m, l, &h, &mut by_block, pos)?;
                assert_eq!(out.len(), dims.hidden, "QE2E-INV-006 block {l} width");
                for (i, (o, x)) in out.iter().zip(&h).enumerate() {
                    assert!(
                        o.is_finite() && o != x,
                        "QE2E-INV-006 block {l} (position {pos}) did not write coordinate {i} \
                         of d_model={}: x = {x}, block(x) = {o}",
                        dims.hidden
                    );
                }
                h = out;
            }
            by_block.kv_cache.advance();

            let mut normed = vec![0.0; dims.hidden];
            crate::gguf::ops::rms_norm_into(
                &h,
                &base.output_norm_weight,
                base.config.eps,
                &mut normed,
            );
            let mut chained = vec![0.0; dims.vocab];
            base.fused_matmul_into(&normed, &base.lm_head_weight, &mut chained)?;
            let logits = m.forward_single_qwen35(token, &mut whole, pos)?;
            assert_eq!(
                chained, logits,
                "QE2E-INV-006 position {pos}: the block-by-block chain is not the model"
            );
        }
    }
    Ok(())
}

// --- QE2E-CON-007: tokens in, [seq_len, V] logits out --------------------------------------

/// QE2E-CON-007 `shape(model(tokens)) = [seq_len, V]`, `tolerance: 0.0` (FALSIFY-QE2E-007).
/// A token sequence is run through the full hybrid stack (`forward_single_qwen35` per
/// position, as `run_qwen35_generate` prefills) from a state sized for exactly `seq_len`
/// positions. The result must be `seq_len` rows of exactly `V` logits, each one finite and
/// written (a logit buffer wider than `V` leaves coordinates at exactly zero; a narrower one
/// fails the width). Sequence lengths 1, 3 and 6 over two vocabularies, every token id in
/// range including `V - 1`.
#[test]
fn qe2e_con_007_tokens_in_logits_out_shape() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let m = model(&base, dims, hybrid_layers(dims, 900));
        for seq_len in [1usize, 3, 6] {
            let tokens = token_sequence(dims, seq_len);
            let mut state = m.new_state(seq_len);
            let logits = tokens
                .iter()
                .enumerate()
                .map(|(pos, &t)| m.forward_single_qwen35(t, &mut state, pos))
                .collect::<Result<Vec<Vec<f32>>>>()?;
            assert_eq!(logits.len(), seq_len, "QE2E-CON-007 rows for {tokens:?}");
            for (pos, row) in logits.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    dims.vocab,
                    "QE2E-CON-007 position {pos}: logits width, V = {}",
                    dims.vocab
                );
                for (v, x) in row.iter().enumerate() {
                    assert!(
                        x.is_finite() && *x != 0.0,
                        "QE2E-CON-007 position {pos}: logit {v} is {x} (never written or not \
                         finite); row {row:?}"
                    );
                }
            }
        }
    }
    Ok(())
}

// --- QHF-BND-005: activation stability -------------------------------------------------------

/// FALSIFY-QHF-005 "No NaN/Inf after 48 layers": the depth the prediction names.
const BND_005_LAYERS: u64 = 48;
/// One more position than the conv window holds, so every tap of the causal conv state and
/// the `DeltaNet` recurrent state is overwritten at least once, and the KV cache grows.
const BND_005_POSITIONS: usize = CONV_KERNEL + 1;

/// One adversarial-but-finite input sequence: `positions` hidden vectors of width `hidden`.
type SeqGen = fn(usize, usize) -> Vec<Vec<f32>>;

fn signed(rng: &mut Rng, magnitude: f32) -> f32 {
    if rng.next_f32() < 0.0 {
        -magnitude
    } else {
        magnitude
    }
}

fn seq_constant(value: f32) -> impl Fn(usize, usize) -> Vec<Vec<f32>> {
    move |hidden, positions| vec![vec![value; hidden]; positions]
}

fn seq_signed(magnitude: f32, seed: u64) -> impl Fn(usize, usize) -> Vec<Vec<f32>> {
    move |hidden, positions| {
        let mut r = Rng::new(seed);
        (0..positions)
            .map(|_| (0..hidden).map(|_| signed(&mut r, magnitude)).collect())
            .collect()
    }
}

/// Every coordinate of one vector cycles through huge, underflowing, zero and O(1) values of
/// both signs, so the RMS is dominated by a few coordinates and the rest are ~0 after the norm.
fn seq_mixed(hidden: usize, positions: usize) -> Vec<Vec<f32>> {
    let mut r = Rng::new(5);
    (0..positions)
        .map(|p| {
            (0..hidden)
                .map(|i| match (i + p) % 6 {
                    0 => 1.0e4,
                    1 => -1.0e-30,
                    2 => 0.0,
                    3 => -1.0e4,
                    4 => 1.0e-30,
                    _ => r.next_f32(),
                })
                .collect()
        })
        .collect()
}

/// Magnitude switches between positions: the recurrent and conv states built from a huge
/// token are then read by a zero token and an underflowing one.
fn seq_switching(hidden: usize, positions: usize) -> Vec<Vec<f32>> {
    let mut r = Rng::new(9);
    (0..positions)
        .map(|p| match p % 4 {
            0 => (0..hidden).map(|_| signed(&mut r, 1.0e4)).collect(),
            1 => vec![0.0; hidden],
            2 => (0..hidden).map(|_| signed(&mut r, 1.0e-30)).collect(),
            _ => vec![-1.0e4; hidden],
        })
        .collect()
}

fn assert_all_finite(h: &[f32], case: &str, hidden: usize, layer: usize, kind: &str, pos: usize) {
    for (i, v) in h.iter().enumerate() {
        assert!(
            v.is_finite(),
            "QHF-BND-005 violated: input `{case}` (d_model={hidden}), after layer {layer} \
             ({kind}), position {pos}, coordinate {i} = {v}"
        );
    }
}

/// QHF-BND-005 `∀l ∈ [0, L], ∀i: is_finite(h_l[i])` (FALSIFY-QHF-005). Drives the real
/// `forward_deltanet` / `forward_attention` blocks through a 48-layer Qwen3.5 schedule
/// (36 Gated `DeltaNet` + 12 attention), token by token over `BND_005_POSITIONS` positions
/// sharing one state, exactly as `forward_single_qwen35` chains them, and checks EVERY
/// coordinate of `h_0` and of `h_l` after each layer.
///
/// Inputs are adversarial but finite. The all-zero vector is a fixed point of the whole
/// stack (no biases), so every RMSNorm, per-head L2 norm and gated RMSNorm of all 48 layers
/// sees `sum(x^2) = 0` and only its `eps` stands between `0 * (1/sqrt(0))` and NaN. The
/// `1e-30` input squares to `0` in f32 (underflow), reaching the same `eps` with nonzero
/// values (`x * inf = inf`). `±1e30` squares to `+inf`, so `inv_rms = 0`.
#[test]
fn qhf_bnd_005_hidden_states_stay_finite() -> Result<()> {
    let cases: [(&str, &dyn Fn(usize, usize) -> Vec<Vec<f32>>); 9] = [
        ("all +1e4", &seq_constant(1.0e4)),
        ("all -1e4", &seq_constant(-1.0e4)),
        ("mixed sign ±1e4", &seq_signed(1.0e4, 3)),
        (
            "mixed sign ±1e-30 (squares underflow)",
            &seq_signed(1.0e-30, 4),
        ),
        ("all-zero", &seq_constant(0.0)),
        (
            "mixed magnitude and sign within a vector",
            &(seq_mixed as SeqGen),
        ),
        (
            "magnitude switching across positions",
            &(seq_switching as SeqGen),
        ),
        (
            "mixed sign ±1e30 (squares overflow)",
            &seq_signed(1.0e30, 6),
        ),
        ("seeded O(1) control", &seq_signed(0.25, 7)),
    ];
    for dims in DIMS {
        let base = base_model(dims);
        let m = model(
            &base,
            dims,
            hybrid_schedule(dims, dims.hidden as u64 * 1000, BND_005_LAYERS),
        );
        assert_eq!(m.layers.len(), BND_005_LAYERS as usize, "stack depth");
        for (case, gen) in cases {
            let seq = gen(dims.hidden, BND_005_POSITIONS);
            let mut state = m.new_state(BND_005_POSITIONS + 1);
            for (pos, h0) in seq.iter().enumerate() {
                assert_all_finite(h0, case, dims.hidden, 0, "input h_0", pos);
                let mut h = h0.clone();
                for il in 0..m.layers.len() {
                    h = run_block(&m, il, &h, &mut state, pos)?;
                    assert_eq!(h.len(), dims.hidden, "{case}: width after layer {}", il + 1);
                    let kind = match &m.layers[il] {
                        Qwen35OwnedLayer::DeltaNet(_) => "Gated DeltaNet",
                        Qwen35OwnedLayer::Attention(_) => "attention",
                    };
                    assert_all_finite(&h, case, dims.hidden, il + 1, kind, pos);
                }
                state.kv_cache.advance();
            }
        }
    }
    Ok(())
}
