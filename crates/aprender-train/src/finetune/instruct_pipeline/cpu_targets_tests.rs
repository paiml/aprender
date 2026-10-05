//! FALSIFY-LORA_TARGET_SELECTION_V1_008: the CPU forward adds every selected
//! adapter's delta to its own projection, reading each adapter from its own
//! slot (0.72 R15a C4b; contract `lora-target-selection-v1`, cpu_forward).

use crate::lora::{LoRALayer, LoraTarget, LoraTargets};
use crate::transformer::{Transformer, TransformerConfig};
use crate::Tensor;

const RANK: usize = 2;
const ALPHA: f32 = 4.0;
const TOKENS: [u32; 5] = [3, 17, 256, 9, 42];

/// A two-layer model without QK-norm in which q_proj, o_proj and down_proj are
/// not square: one KV head and a head dim of 16 make q_proj `[32, 64]`, k_proj
/// and v_proj `[16, 64]`, o_proj `[64, 32]`, with hidden 64 and intermediate 256.
fn model() -> (TransformerConfig, Transformer) {
    let config = TransformerConfig {
        num_kv_heads: 1,
        head_dim_override: Some(16),
        ..TransformerConfig::tiny()
    };
    let model = Transformer::new(&config);
    (config, model)
}

/// `len` values in `[-amp/2, amp/2)`, distinct for each `seed`.
fn spread(len: usize, seed: usize, amp: f32) -> Vec<f32> {
    (0..len).map(|i| (((i * 7919 + seed * 104_729) % 1000) as f32 / 1000.0 - 0.5) * amp).collect()
}

fn all_linear() -> LoraTargets {
    let names: Vec<&str> = LoraTarget::ALL.iter().map(|t| t.module_name()).collect();
    LoraTargets::parse(&names).expect("every target name must parse")
}

/// The adapters of every layer, laid out by `targets`. Every A is nonzero;
/// B is nonzero for the slots `nonzero_b` accepts and zero elsewhere.
fn adapters(
    config: &TransformerConfig,
    targets: &LoraTargets,
    nonzero_b: impl Fn(usize) -> bool,
) -> Vec<LoRALayer> {
    let mut out = Vec::new();
    for _layer in 0..config.num_hidden_layers {
        for &target in targets.as_slice() {
            let slot = out.len();
            let (d_out, d_in) = target.dims(config);
            let mut lora =
                LoRALayer::new(Tensor::zeros(d_out * d_in, false), d_out, d_in, RANK, ALPHA);
            *lora.lora_a_mut() = Tensor::from_vec(spread(RANK * d_in, 2 * slot + 1, 0.4), true);
            let b = if nonzero_b(slot) {
                spread(d_out * RANK, 2 * slot + 2, 0.4)
            } else {
                vec![0.0; d_out * RANK]
            };
            *lora.lora_b_mut() = Tensor::from_vec(b, true);
            out.push(lora);
        }
    }
    out
}

fn weight_mut(model: &mut Transformer, layer: usize, target: LoraTarget) -> &mut Tensor {
    let block = &mut model.layers[layer];
    match target {
        LoraTarget::Q => &mut block.self_attn.w_q,
        LoraTarget::K => &mut block.self_attn.w_k,
        LoraTarget::V => &mut block.self_attn.w_v,
        LoraTarget::O => &mut block.self_attn.w_o,
        LoraTarget::Gate => &mut block.ffn.w_gate,
        LoraTarget::Up => &mut block.ffn.w_up,
        LoraTarget::Down => &mut block.ffn.w_down,
    }
}

/// `W + scale·B·A`, with W `[d_out, d_in]`, B `(d_out, rank)`, A `(rank, d_in)`.
fn merged(base: &Tensor, lora: &LoRALayer, d_out: usize, d_in: usize) -> Tensor {
    let (w, a, b) = (base.data(), lora.lora_a().data(), lora.lora_b().data());
    let mut out = w.to_vec();
    for o in 0..d_out {
        for i in 0..d_in {
            let ba: f32 = (0..RANK).map(|r| b[o * RANK + r] * a[r * d_in + i]).sum();
            out[o * d_in + i] += lora.scale() * ba;
        }
    }
    Tensor::from_vec(out, false)
}

fn values(t: &Tensor) -> Vec<f32> {
    t.data().to_vec()
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "outputs must have the same length");
    x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)
}

#[test]
fn falsify_lora_target_selection_v1_008_each_adapter_equals_its_merged_weight() {
    let (config, mut model) = model();
    let targets = all_linear();
    let per_layer = targets.per_layer();
    let plain = values(&model.forward_hidden(&TOKENS));
    for slot in 0..per_layer * config.num_hidden_layers {
        let (layer, target) = (slot / per_layer, targets.as_slice()[slot % per_layer]);
        let lora = adapters(&config, &targets, |s| s == slot);
        let with_lora = values(&model.forward_hidden_with_targets(&TOKENS, &lora, &targets));
        assert!(
            max_abs_diff(&with_lora, &plain) > 1e-3,
            "slot {slot} ({target:?}, layer {layer}) must change the output"
        );

        let (d_out, d_in) = target.dims(&config);
        let base = weight_mut(&mut model, layer, target).clone();
        *weight_mut(&mut model, layer, target) = merged(&base, &lora[slot], d_out, d_in);
        let with_merged = values(&model.forward_hidden(&TOKENS));
        *weight_mut(&mut model, layer, target) = base;

        let diff = max_abs_diff(&with_lora, &with_merged);
        assert!(
            diff <= 1e-4,
            "slot {slot} ({target:?}, layer {layer}): LoRA forward differs from the merged \
             weight's plain forward by {diff}"
        );
    }
}

#[test]
fn falsify_lora_target_selection_v1_008_default_targets_are_the_old_forward() {
    let (config, model) = model();
    let targets = LoraTargets::default();
    assert_eq!(targets.as_slice(), &[LoraTarget::Q, LoraTarget::V]);
    let lora = adapters(&config, &targets, |_| true);
    let new = values(&model.forward_hidden_with_targets(&TOKENS, &lora, &targets));
    let old = values(&model.forward_hidden_with_lora(&TOKENS, &lora));
    assert_eq!(new, old, "under q_proj, v_proj the forward must be bit-identical");
}

#[test]
fn falsify_lora_target_selection_v1_008_backward_reaches_every_adapter() {
    let (config, model) = model();
    let targets = all_linear();
    let lora = adapters(&config, &targets, |_| true);
    let hidden = model.forward_hidden_with_targets(&TOKENS, &lora, &targets);
    let weights: Vec<f32> = (0..hidden.len()).map(|i| ((i % 7) as f32 - 3.0) / 3.0).collect();
    hidden.set_grad(ndarray::Array1::from(weights));
    let op = hidden.backward_op().expect("the LoRA forward must record a backward op");
    op.backward();
    for (slot, adapter) in lora.iter().enumerate() {
        for (name, t) in [("A", adapter.lora_a()), ("B", adapter.lora_b())] {
            let grad = t.grad().unwrap_or_else(|| panic!("slot {slot} {name}: no gradient"));
            assert!(grad.iter().any(|g| g.abs() > 0.0), "slot {slot} {name}: gradient is all zero");
        }
    }
}
