//! FALSIFY-LORA_TARGET_SELECTION_V1_009: TransformerTrainer lays its LoRA
//! layers out by its targets and its forward reads each adapter from that slot
//! (0.72 R15a C4c; contract `lora-target-selection-v1`, trainer_forward).

use super::batch::LMBatch;
use super::config::TransformerTrainConfig;
use super::trainer::TransformerTrainer;
use crate::lora::{LoRALayer, LoraTarget, LoraTargets};
use crate::transformer::{Transformer, TransformerConfig};
use crate::Tensor;

const RANK: usize = 2;
const ALPHA: f32 = 4.0;
const TOKENS: [u32; 5] = [3, 17, 256, 9, 42];
const NEXT: [u32; 5] = [17, 256, 9, 42, 7];

/// A two-layer model in which q_proj, k_proj, o_proj and down_proj are not
/// square: one KV head and a head dim of 16, with hidden 64.
fn model_config() -> TransformerConfig {
    TransformerConfig { num_kv_heads: 1, head_dim_override: Some(16), ..TransformerConfig::tiny() }
}

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|n| (*n).to_string()).collect()
}

fn trainer(modules: &[&str]) -> TransformerTrainer {
    let config = TransformerTrainConfig::new(model_config()).with_lora(RANK, ALPHA, names(modules));
    TransformerTrainer::new(config)
}

/// `len` values in `[-amp/2, amp/2)`, distinct for each `seed`.
fn spread(len: usize, seed: usize, amp: f32) -> Vec<f32> {
    (0..len).map(|i| (((i * 7919 + seed * 104_729) % 1000) as f32 / 1000.0 - 0.5) * amp).collect()
}

/// Every A nonzero; B nonzero for the slots `nonzero_b` accepts, zero elsewhere.
fn fill(trainer: &mut TransformerTrainer, nonzero_b: impl Fn(usize) -> bool) {
    let layers = trainer.lora_layers_mut().expect("LoRA is enabled");
    for (slot, lora) in layers.iter_mut().enumerate() {
        let (d_out, d_in) = (lora.d_out(), lora.d_in());
        *lora.lora_a_mut() = Tensor::from_vec(spread(RANK * d_in, 2 * slot + 1, 0.4), true);
        let b = if nonzero_b(slot) {
            spread(d_out * RANK, 2 * slot + 2, 0.4)
        } else {
            vec![0.0; d_out * RANK]
        };
        *lora.lora_b_mut() = Tensor::from_vec(b, true);
    }
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
fn merged(base: &Tensor, lora: &LoRALayer) -> Tensor {
    let (d_out, d_in) = (lora.d_out(), lora.d_in());
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

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "outputs must have the same length");
    x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)
}

/// The targets, slot count and slot shapes `build` made for `modules`.
fn check_layout(trainer: &TransformerTrainer, modules: &[&str], config: &TransformerConfig) {
    let expected = LoraTargets::parse(modules).expect("the test's names parse");
    assert_eq!(trainer.lora_targets(), &expected, "{modules:?}: the trainer's targets");
    let layers = trainer.lora_layers().expect("LoRA is enabled");
    let per_layer = expected.per_layer();
    assert_eq!(layers.len(), per_layer * config.num_hidden_layers, "{modules:?}: |T|·L slots");
    for (slot, lora) in layers.iter().enumerate() {
        let target = expected.as_slice()[slot % per_layer];
        assert_eq!(
            (lora.d_out(), lora.d_in()),
            target.dims(config),
            "{modules:?}: slot {slot} must have {target:?}'s shape"
        );
    }
}

#[test]
fn falsify_lora_target_selection_v1_009_forward_reads_the_slots_build_made() {
    let config = model_config();
    let sets: [&[&str]; 3] = [&["all_linear"], &["q_proj"], &["down_proj", "k_proj"]];
    for modules in sets {
        let mut trainer = trainer(modules);
        check_layout(&trainer, modules, &config);
        let targets = trainer.lora_targets().clone();
        let per_layer = targets.per_layer();
        let plain = trainer.model().forward(&TOKENS).data().to_vec();
        for slot in 0..per_layer * config.num_hidden_layers {
            let (layer, target) = (slot / per_layer, targets.as_slice()[slot % per_layer]);
            fill(&mut trainer, |s| s == slot);
            let (_, _, logits) = trainer.forward_single(&TOKENS, &NEXT);
            let with_lora = logits.data().to_vec();
            assert!(
                max_abs_diff(&with_lora, &plain) > 1e-3,
                "{modules:?}: slot {slot} ({target:?}, layer {layer}) must change the logits"
            );

            let base = weight_mut(trainer.model_mut(), layer, target).clone();
            let merged_weight =
                merged(&base, &trainer.lora_layers().expect("LoRA is enabled")[slot]);
            *weight_mut(trainer.model_mut(), layer, target) = merged_weight;
            let with_merged = trainer.model().forward(&TOKENS).data().to_vec();
            *weight_mut(trainer.model_mut(), layer, target) = base;

            let diff = max_abs_diff(&with_lora, &with_merged);
            assert!(
                diff <= 1e-4,
                "{modules:?}: slot {slot} ({target:?}, layer {layer}): forward_single differs \
                 from the merged weight's plain forward by {diff}"
            );
        }
    }
}

#[test]
fn falsify_lora_target_selection_v1_009_default_targets_are_the_old_forward() {
    let config = model_config();
    let mut trainer = trainer(&["q_proj", "v_proj"]);
    check_layout(&trainer, &["q_proj", "v_proj"], &config);
    fill(&mut trainer, |_| true);
    let (_, _, logits) = trainer.forward_single(&TOKENS, &NEXT);
    let lora = trainer.lora_layers().expect("LoRA is enabled");
    let old = trainer.model().forward_with_lora(&TOKENS, lora);
    assert_eq!(
        logits.data().to_vec(),
        old.data().to_vec(),
        "under q_proj, v_proj forward_single must equal forward_with_lora bit for bit"
    );
}

#[test]
fn falsify_lora_target_selection_v1_009_backward_and_step_reach_every_adapter() {
    let mut trainer = {
        let config = TransformerTrainConfig::new(model_config())
            .with_lora(RANK, ALPHA, names(&["all_linear"]))
            .with_lr(0.01);
        TransformerTrainer::new(config)
    };
    fill(&mut trainer, |_| true);

    let (_, loss, _) = trainer.forward_single(&TOKENS, &NEXT);
    loss.backward_op().expect("the LoRA forward must record a backward op").backward();
    let layers = trainer.lora_layers().expect("LoRA is enabled");
    assert_eq!(layers.len(), 14, "all_linear on two layers is 14 adapters");
    for (slot, adapter) in layers.iter().enumerate() {
        for (name, t) in [("A", adapter.lora_a()), ("B", adapter.lora_b())] {
            let grad = t.grad().unwrap_or_else(|| panic!("slot {slot} {name}: no gradient"));
            assert!(grad.iter().any(|g| g.abs() > 0.0), "slot {slot} {name}: gradient is all zero");
        }
    }

    let b_before: Vec<Vec<f32>> = layers.iter().map(|l| l.lora_b().data().to_vec()).collect();
    trainer.train_batch(&LMBatch::single(TOKENS.to_vec(), NEXT.to_vec()));
    let layers = trainer.lora_layers().expect("LoRA is enabled");
    for (slot, (adapter, before)) in layers.iter().zip(&b_before).enumerate() {
        let moved = max_abs_diff(adapter.lora_b().data().as_slice().expect("contiguous"), before);
        assert!(moved > 0.0, "slot {slot}: one train_batch must change B");
    }
}
