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

/// FALSIFY-LORA_TARGET_SELECTION_V1_014 (R15a C8): an adapter that leaves a LoRA
/// place (slot, A or B) without a tensor refuses the whole load, names each place
/// it left, and no LoRA layer changes. Before C8 the place kept its fresh init.
#[test]
fn falsify_lora_target_selection_v1_014_partial_adapter_refuses_and_changes_nothing() {
    let (num_layers, fresh) = tiny_layers();
    let adapter = full_adapter(num_layers, &fresh);
    let without = |pred: &dyn Fn(&str) -> bool| -> Vec<(String, Vec<f32>)> {
        adapter.iter().filter(|(n, _)| !pred(n)).cloned().collect()
    };
    let cases: [(&str, Vec<(String, Vec<f32>)>, &[&str]); 3] = [
        (
            "no v_proj",
            without(&|n| n.contains(".v_proj.")),
            &["layers.0.v_proj.lora_A", "layers.0.v_proj.lora_B", "layers.1.v_proj.lora_B"],
        ),
        (
            "no layer 1",
            without(&|n| n.contains(".layers.1.")),
            &["layers.1.q_proj.lora_A", "layers.1.v_proj.lora_B"],
        ),
        (
            "one B missing",
            without(&|n| *n == peft_name(0, "q_proj", "lora_B")),
            &["layers.0.q_proj.lora_B"],
        ),
    ];
    for (case, weights, places) in cases {
        let (_, mut layers) = tiny_layers();
        let before = snapshot(&layers);
        let Err(err) = InstructPipeline::inject_adapter_weights(
            &mut layers,
            &weights,
            &LoraTargets::default(),
        ) else {
            panic!("{case}: a partial adapter must refuse the load");
        };
        for place in places {
            assert!(err.to_string().contains(place), "{case}: the error names {place}: {err}");
        }
        assert_eq!(snapshot(&layers), before, "{case}: no LoRA layer may change");
    }
}

/// Only the places an adapter leaves are named: one missing B names that B alone,
/// and an empty adapter names all 2·slots places.
#[test]
fn falsify_lora_target_selection_v1_014_names_exactly_the_unfilled_places() {
    let (num_layers, fresh) = tiny_layers();
    let mut weights = full_adapter(num_layers, &fresh);
    weights.retain(|(n, _)| *n != peft_name(1, "v_proj", "lora_B"));
    let (_, mut layers) = tiny_layers();
    let err =
        InstructPipeline::inject_adapter_weights(&mut layers, &weights, &LoraTargets::default())
            .expect_err("one B missing must refuse");
    let msg = err.to_string();
    assert!(msg.contains("1 of 8 LoRA places have no tensor"), "{msg}");
    assert!(msg.contains("layers.1.v_proj.lora_B"), "{msg}");
    assert!(!msg.contains("layers.1.v_proj.lora_A"), "a filled place is not named: {msg}");
    assert!(!msg.contains("fit no LoRA layer"), "no tensor was unplaced: {msg}");

    let err = InstructPipeline::inject_adapter_weights(&mut layers, &[], &LoraTargets::default())
        .expect_err("an empty adapter must refuse");
    assert!(err.to_string().contains("8 of 8 LoRA places have no tensor"), "{err}");
}

/// Write `full_adapter`'s tensors to `dir` as `PeftAdapterBundle::save_peft` does,
/// with `declared` as the `target_modules` of `adapter_config.json`.
fn save_adapter(dir: &Path, declared: &[&str], num_layers: usize, layers: &[LoRALayer]) {
    let mut bundle = crate::lora::PeftAdapterBundle::new(
        crate::lora::LoRAConfig::new(4, 8.0).target_modules(declared),
    );
    for layer in 0..num_layers {
        for (proj, proj_name) in PROJECTIONS.iter().enumerate() {
            let slot = layer * PROJECTIONS.len() + proj;
            let l = &layers[slot];
            bundle.add_raw_adapter(
                format!("model.layers.{layer}.self_attn.{proj_name}"),
                filled(len_of(layers, slot, 0), 1 + 2 * slot),
                filled(len_of(layers, slot, 1), 2 + 2 * slot),
                l.rank(),
                l.d_in(),
                l.d_out(),
            );
        }
    }
    bundle.save_peft(dir).expect("the test adapter must save");
}

/// `load_trained_adapter` into fresh layers must refuse, with every one of
/// `needles` in the message, and change no layer.
fn assert_refuses(dir: &Path, needles: &[&str]) {
    let (_, mut layers) = tiny_layers();
    let before = snapshot(&layers);
    let err = InstructPipeline::load_trained_adapter(dir, &mut layers, &LoraTargets::default())
        .expect_err("the adapter must refuse")
        .to_string();
    for needle in needles {
        assert!(err.contains(needle), "{needle:?} not in: {err}");
    }
    assert_eq!(snapshot(&layers), before, "a refused load changed a layer: {err}");
}

#[test]
fn falsify_lora_target_selection_v1_015_declared_targets_must_match() {
    let (num_layers, fresh) = tiny_layers();
    // Each adapter holds complete q_proj/v_proj tensors, so only the declaration differs.
    for (declared, named) in [
        (&["q_proj", "k_proj", "v_proj"][..], "q_proj, k_proj, v_proj"),
        (&["q_proj"][..], "trained on targets q_proj,"),
        (&["k_proj", "o_proj"][..], "k_proj, o_proj"),
        (&["all_linear"][..], "gate_proj"),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        save_adapter(dir.path(), declared, num_layers, &fresh);
        assert_refuses(dir.path(), &[named, "selects q_proj, v_proj"]);
    }
}

#[test]
fn falsify_lora_target_selection_v1_015_unreadable_adapter_refuses() {
    let (num_layers, fresh) = tiny_layers();
    let garbage = tempfile::tempdir().expect("tempdir");
    save_adapter(garbage.path(), &["q_proj", "v_proj"], num_layers, &fresh);
    std::fs::write(garbage.path().join("adapter_model.safetensors"), b"not safetensors")
        .expect("write");
    assert_refuses(garbage.path(), &["adapter_model.safetensors", "failed to load"]);

    let no_config = tempfile::tempdir().expect("tempdir");
    save_adapter(no_config.path(), &["q_proj", "v_proj"], num_layers, &fresh);
    std::fs::remove_file(no_config.path().join("adapter_config.json")).expect("remove");
    assert_refuses(no_config.path(), &["failed to load"]);

    let unknown = tempfile::tempdir().expect("tempdir");
    save_adapter(unknown.path(), &["q_proj", "w_proj"], num_layers, &fresh);
    assert_refuses(unknown.path(), &["target_modules", "\"w_proj\""]);
}

#[test]
fn falsify_lora_target_selection_v1_015_matching_or_undeclared_adapter_loads() {
    let (num_layers, fresh) = tiny_layers();
    for declared in [&["v_proj", "q_proj"][..], &["qv"][..], &[][..]] {
        let dir = tempfile::tempdir().expect("tempdir");
        save_adapter(dir.path(), declared, num_layers, &fresh);
        let (_, mut layers) = tiny_layers();
        let loaded = InstructPipeline::load_trained_adapter(
            dir.path(),
            &mut layers,
            &LoraTargets::default(),
        )
        .expect("a matching or undeclared adapter must load");
        assert_eq!(loaded, Some(num_layers * PROJECTIONS.len() * KINDS.len()), "{declared:?}");
        for (slot, layer) in layers.iter().enumerate() {
            assert_eq!(
                layer.lora_a().data().to_vec(),
                filled(len_of(&fresh, slot, 0), 1 + 2 * slot)
            );
            assert_eq!(
                layer.lora_b().data().to_vec(),
                filled(len_of(&fresh, slot, 1), 2 + 2 * slot)
            );
        }
    }
}

#[test]
fn falsify_lora_target_selection_v1_015_no_adapter_keeps_init() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, mut layers) = tiny_layers();
    let before = snapshot(&layers);
    let loaded =
        InstructPipeline::load_trained_adapter(dir.path(), &mut layers, &LoraTargets::default())
            .expect("no adapter is not an error");
    assert_eq!(loaded, None);
    assert_eq!(snapshot(&layers), before);
}

#[test]
fn falsify_lora_target_selection_v1_015_compares_against_the_selected_targets() {
    let qkv = LoraTargets::parse(&["v_proj", "k_proj", "q_proj"]).expect("qkv");
    let declared = |names: &[&str]| names.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert!(check_adapter_targets(&declared(&["q_proj", "k_proj", "v_proj"]), &qkv).is_ok());
    let err = check_adapter_targets(&declared(&["q_proj", "v_proj"]), &qkv)
        .expect_err("q/v against q/k/v must refuse")
        .to_string();
    assert!(err.contains("trained on targets q_proj, v_proj"), "{err}");
    assert!(err.contains("selects q_proj, k_proj, v_proj"), "{err}");
}
