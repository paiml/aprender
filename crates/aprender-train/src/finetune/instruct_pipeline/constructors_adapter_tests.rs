//! FALSIFY-LORA_TARGET_SELECTION_V1_003: a PEFT adapter is routed into the
//! q_proj and v_proj LoRA layers by tensor name, or the load is refused and
//! no LoRA layer changes (0.72 R15a C2a).

use super::*;

const PROJECTIONS: [&str; 2] = ["q_proj", "v_proj"];
const KINDS: [&str; 2] = ["lora_A", "lora_B"];

/// The Q and V LoRA layers of a two-layer GQA model at rank 4, as
/// `from_pretrained` builds them before it loads an adapter. One KV head
/// makes V's B half the length of Q's B, so A and B lengths differ.
fn tiny_layers() -> (usize, Vec<LoRALayer>) {
    let model_config = TransformerConfig { num_kv_heads: 1, ..TransformerConfig::tiny() };
    let model = Transformer::new(&model_config);
    let config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let layers = InstructPipeline::build_lora_layers(&model, &model_config, &config);
    assert_eq!(layers.len(), model_config.num_hidden_layers * PROJECTIONS.len());
    assert_ne!(layers[1].lora_a().len(), layers[1].lora_b().len());
    (model_config.num_hidden_layers, layers)
}

/// A tensor name as `peft_export` writes it.
fn peft_name(layer: usize, proj: &str, kind: &str) -> String {
    format!("base_model.model.model.layers.{layer}.self_attn.{proj}.{kind}.weight")
}

/// Values no other tensor and no initial A or B holds: `tag` in the integer
/// part, the element index in the fraction.
pub(super) fn filled(len: usize, tag: usize) -> Vec<f32> {
    (0..len).map(|i| tag as f32 + i as f32 * 1e-4).collect()
}

fn len_of(layers: &[LoRALayer], slot: usize, kind: usize) -> usize {
    if kind == 0 {
        layers[slot].lora_a().len()
    } else {
        layers[slot].lora_b().len()
    }
}

/// Every A and B, in slot order.
fn snapshot(layers: &[LoRALayer]) -> Vec<Vec<f32>> {
    layers.iter().flat_map(|l| [l.lora_a().data().to_vec(), l.lora_b().data().to_vec()]).collect()
}

/// A complete adapter for `num_layers` layers: slot `2·layer + proj` gets
/// `filled(len, 1 + 2·slot + kind)` as its A (kind 0) and B (kind 1).
fn full_adapter(num_layers: usize, layers: &[LoRALayer]) -> Vec<(String, Vec<f32>)> {
    let mut weights = Vec::new();
    for layer in 0..num_layers {
        for (proj, proj_name) in PROJECTIONS.iter().enumerate() {
            let slot = layer * PROJECTIONS.len() + proj;
            for (kind, kind_name) in KINDS.iter().enumerate() {
                let data = filled(len_of(layers, slot, kind), 1 + 2 * slot + kind);
                weights.push((peft_name(layer, proj_name, kind_name), data));
            }
        }
    }
    weights
}

#[test]
fn falsify_lora_target_selection_v1_003_routes_by_name_in_any_order() {
    let (num_layers, fresh) = tiny_layers();
    let adapter = full_adapter(num_layers, &fresh);
    let mut reversed = adapter.clone();
    reversed.reverse();

    for weights in [&adapter, &reversed] {
        let (_, mut layers) = tiny_layers();
        InstructPipeline::inject_adapter_weights(&mut layers, weights, &LoraTargets::default())
            .expect("a q_proj/v_proj adapter of the right rank must load");
        for (slot, layer) in layers.iter().enumerate() {
            let a = filled(len_of(&fresh, slot, 0), 1 + 2 * slot);
            let b = filled(len_of(&fresh, slot, 1), 2 + 2 * slot);
            assert_eq!(layer.lora_a().data().to_vec(), a, "slot {slot} A");
            assert_eq!(layer.lora_b().data().to_vec(), b, "slot {slot} B");
        }
    }
}

#[test]
fn falsify_lora_target_selection_v1_003_unplaceable_tensor_refuses_and_changes_nothing() {
    let (num_layers, fresh) = tiny_layers();
    let (q_a, v_a, v_b) = (len_of(&fresh, 0, 0), len_of(&fresh, 1, 0), len_of(&fresh, 1, 1));
    let v_proj_a = peft_name(0, "v_proj", "lora_A");
    let v_proj_b = peft_name(0, "v_proj", "lora_B");
    // (bad tensor, its length, the adapter tensor left out). Each bad tensor
    // has the length of the slot the 316dee2cd4 rule sent it to (layer 0's V
    // adapter for every name without q_proj; B for every name without
    // lora_A), and that slot is left empty, so only the name can refuse it.
    let cases: [(String, usize, Option<&str>); 7] = [
        (peft_name(0, "k_proj", "lora_A"), v_a, Some(&v_proj_a)),
        (peft_name(0, "o_proj", "lora_A"), v_a, Some(&v_proj_a)),
        (
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight".into(),
            v_a,
            Some(&v_proj_a),
        ),
        (peft_name(0, "qkv_proj", "lora_A"), v_a, Some(&v_proj_a)),
        (peft_name(0, "v_proj", "lora_a"), v_b, Some(&v_proj_b)),
        (peft_name(num_layers, "q_proj", "lora_A"), q_a, None),
        (peft_name(0, "q_proj", "lora_A"), q_a, None),
    ];
    for (name, len, left_out) in cases {
        let (_, mut layers) = tiny_layers();
        let before = snapshot(&layers);
        // The bad tensor comes last, after tensors that load on their own, so
        // a load that writes as it goes changes the layers.
        let mut weights = full_adapter(num_layers, &fresh);
        weights.retain(|(n, _)| Some(n.as_str()) != left_out);
        weights.push((name.clone(), filled(len, 99)));

        let Err(err) = InstructPipeline::inject_adapter_weights(
            &mut layers,
            &weights,
            &LoraTargets::default(),
        ) else {
            panic!("{name} must refuse the load");
        };
        assert!(err.to_string().contains(&name), "the error names {name}: {err}");
        assert_eq!(snapshot(&layers), before, "{name}: no LoRA layer may change");
    }
}

#[test]
fn falsify_lora_target_selection_v1_003_wrong_rank_refuses() {
    let (num_layers, fresh) = tiny_layers();
    let q_a = len_of(&fresh, 0, 0);
    // One more rank row of A, and the f32 length of bf16 bytes read as f32.
    for len in [q_a + fresh[0].d_in(), q_a / 2] {
        let (_, mut layers) = tiny_layers();
        let before = snapshot(&layers);
        let mut weights = full_adapter(num_layers, &fresh);
        weights[0].1 = filled(len, 99);

        let err = InstructPipeline::inject_adapter_weights(
            &mut layers,
            &weights,
            &LoraTargets::default(),
        )
        .expect_err("an A of the wrong length must refuse the load");
        assert!(err.to_string().contains(&weights[0].0), "the error names the tensor: {err}");
        assert_eq!(snapshot(&layers), before, "len {len}: no LoRA layer may change");
    }
}
