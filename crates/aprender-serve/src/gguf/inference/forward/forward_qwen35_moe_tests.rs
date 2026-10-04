//! #4665: the Qwen3.5-MoE FFN against an independent reference, and the batched prefill
//! against one `forward_single_qwen35` per token, on the seeded F32 fixtures of
//! `forward_qwen35_contract_tests.rs` (no GGUF file).

use super::super::qhf_contract_tests::{
    assert_close, attention_layer, base_model, deltanet_layer, deltas_over_sequence, f32_tensor,
    hybrid_schedule, inputs, matvec, model, ref_ffn, rmsnorm, sigmoid, token_sequence, zero_tensor,
    Dims, Rng, DIMS, INPUT_SEEDS,
};
use super::{split_experts, Qwen35MoeFfn};
use crate::error::Result;
use crate::gguf::forward_qwen35::Qwen35OwnedLayer;
use crate::gguf::quantized::QuantizedTensorRef;

const EXPERTS: usize = 6;
const TOP_K: usize = 2;
/// Deliberately != hidden and != the dense intermediate width of either `DIMS` entry.
const EXPERT_DIM: usize = 5;

fn moe_ffn(dims: Dims, seed: u64) -> Box<Qwen35MoeFfn> {
    let mut r = Rng::new(seed);
    let h = dims.hidden;
    Box::new(Qwen35MoeFfn {
        // ×4 spreads the logits so the top-k is not a near-tie.
        router: r.vec(EXPERTS * h).iter().map(|v| v * 4.0).collect(),
        shared_gate: r.vec(h),
        gate_exps: (0..EXPERTS)
            .map(|_| f32_tensor(&mut r, h, EXPERT_DIM))
            .collect(),
        up_exps: (0..EXPERTS)
            .map(|_| f32_tensor(&mut r, h, EXPERT_DIM))
            .collect(),
        down_exps: (0..EXPERTS)
            .map(|_| f32_tensor(&mut r, EXPERT_DIM, h))
            .collect(),
        top_k: TOP_K,
    })
}

/// llama.cpp `qwen35moe.cpp` `build_layer_ffn`, written out without `route_top_k`:
/// softmax over all experts, the top-k by probability, renormalized, plus the shared
/// expert scaled by `sigmoid(shared_gate · x)`.
fn ref_moe_ffn(
    h: &[f32],
    norm: &[f32],
    shared: (
        &crate::gguf::OwnedQuantizedTensor,
        &crate::gguf::OwnedQuantizedTensor,
        &crate::gguf::OwnedQuantizedTensor,
    ),
    moe: &Qwen35MoeFfn,
    eps: f32,
) -> Vec<f32> {
    let x = rmsnorm(h, norm, eps);
    let logits: Vec<f32> = moe
        .router
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(&x).map(|(w, v)| w * v).sum())
        .collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let z: f32 = exp.iter().sum();
    let mut order: Vec<usize> = (0..exp.len()).collect();
    order.sort_by(|&a, &b| exp[b].total_cmp(&exp[a]));
    let chosen = &order[..moe.top_k];
    let mass: f32 = chosen.iter().map(|&e| exp[e] / z).sum();

    let g: f32 = moe.shared_gate.iter().zip(&x).map(|(w, v)| w * v).sum();
    let mut out: Vec<f32> = ref_ffn(h, norm, shared.0, shared.1, shared.2, eps)
        .iter()
        .map(|v| v * sigmoid(g))
        .collect();
    for &e in chosen {
        let gate = matvec(&moe.gate_exps[e], &x);
        let up = matvec(&moe.up_exps[e], &x);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| u * g * sigmoid(*g))
            .collect();
        let down = matvec(&moe.down_exps[e], &act);
        for (o, d) in out.iter_mut().zip(&down) {
            *o += exp[e] / z / mass * d;
        }
    }
    out
}

/// The FFN sublayer of a MoE layer (mixer zeroed) equals the reference, at every position,
/// for both layer kinds — so neither the routed sum nor the shared-expert gate is dropped.
#[test]
fn qwen35moe_ffn_sublayer_matches_the_reference() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let eps = base.config.eps;
        let mut attn = attention_layer(dims, 4665);
        attn.attn_output = zero_tensor(dims.num_heads * dims.attn_head_dim, dims.hidden);
        attn.moe = Some(moe_ffn(dims, 1));
        let mut gdn = deltanet_layer(dims, 4666);
        gdn.ssm_out = zero_tensor(dims.gdn_width(), dims.hidden);
        gdn.moe = Some(moe_ffn(dims, 2));
        for layer in [
            Qwen35OwnedLayer::Attention(attn),
            Qwen35OwnedLayer::DeltaNet(gdn),
        ] {
            let what = match &layer {
                Qwen35OwnedLayer::Attention(_) => "MoE FFN (attention layer)",
                Qwen35OwnedLayer::DeltaNet(_) => "MoE FFN (GDN layer)",
            };
            let m = model(&base, dims, vec![layer]);
            let (norm, gate, up, down, moe) = match &m.layers[0] {
                Qwen35OwnedLayer::Attention(a) => (
                    &a.post_attention_norm,
                    &a.ffn_gate,
                    &a.ffn_up,
                    &a.ffn_down,
                    a.moe.as_deref(),
                ),
                Qwen35OwnedLayer::DeltaNet(d) => (
                    &d.post_attention_norm,
                    &d.ffn_gate,
                    &d.ffn_up,
                    &d.ffn_down,
                    d.moe.as_deref(),
                ),
            };
            let moe = moe.expect("the layer was given experts");
            for seed in INPUT_SEEDS {
                let xs = inputs(dims, seed);
                let deltas = deltas_over_sequence(&m, &xs)?;
                for (pos, (delta, h)) in deltas.iter().zip(&xs).enumerate() {
                    let want = ref_moe_ffn(h, norm, (gate, up, down), moe, eps);
                    assert_close(delta, &want, &format!("{what}, position {pos}"));
                }
            }
        }
    }
    Ok(())
}

/// The batched prefill runs the MoE combine per row: its logits are bit-identical to one
/// `forward_single_qwen35` per token over a MoE hybrid stack.
#[test]
fn qwen35moe_prefill_is_bit_identical_to_per_token() -> Result<()> {
    for dims in DIMS {
        let base = base_model(dims);
        let mut layers = hybrid_schedule(dims, 4700, 4);
        for (l, layer) in layers.iter_mut().enumerate() {
            let moe = Some(moe_ffn(dims, 4800 + l as u64));
            match layer {
                Qwen35OwnedLayer::Attention(a) => a.moe = moe,
                Qwen35OwnedLayer::DeltaNet(d) => d.moe = moe,
            }
        }
        let m = model(&base, dims, layers);
        let tokens = token_sequence(dims, 6);
        let mut per_token = m.new_state(tokens.len());
        let mut want = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            want = m.forward_single_qwen35(t, &mut per_token, pos)?;
        }
        let mut batched = m.new_state(tokens.len());
        let got = m.forward_prefill_qwen35(&tokens, &mut batched, 0)?;
        assert_eq!(got, want, "prefill logits differ from per-token");
    }
    Ok(())
}

fn stacked_ref(bytes: usize, elems: usize) -> QuantizedTensorRef {
    QuantizedTensorRef {
        offset: 0,
        byte_size: bytes,
        num_elements: elems,
        qtype: crate::gguf::types::GGUF_TYPE_F32,
    }
}

/// Expert `e` of a stacked `[E × out × in]` tensor is its `e`-th contiguous block.
#[test]
fn split_experts_takes_expert_e_as_the_eth_block() -> Result<()> {
    let (e, out, inn) = (3usize, 2usize, 4usize);
    let values: Vec<f32> = (0..e * out * inn).map(|v| v as f32).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (experts, out_dim) =
        split_experts("t", &stacked_ref(data.len(), values.len()), &data, e, inn)?;
    assert_eq!(out_dim, out);
    assert_eq!(experts.len(), e);
    for (i, x) in experts.iter().enumerate() {
        assert_eq!((x.in_dim, x.out_dim), (inn, out));
        let first = f32::from_le_bytes([x.data[0], x.data[1], x.data[2], x.data[3]]);
        assert_eq!(
            first,
            (i * out * inn) as f32,
            "expert {i} starts at its own block"
        );
    }
    Ok(())
}

/// A stack that is not `E` whole experts of whole rows is refused, never sliced.
#[test]
fn split_experts_refuses_a_ragged_stack() {
    let data = vec![0u8; 4 * 25];
    assert!(
        split_experts("t", &stacked_ref(100, 25), &data, 3, 4).is_err(),
        "25 elems / 3"
    );
    assert!(
        split_experts("t", &stacked_ref(96, 24), &data, 3, 5).is_err(),
        "8-wide / 5"
    );
    assert!(
        split_experts("t", &stacked_ref(400, 25), &data, 5, 5).is_err(),
        "past the file"
    );
}
