//! FALSIFY-LORA_TARGET_SELECTION_V1_004: `InstructConfig.lora_targets` selects
//! the adapters `build_lora_layers` builds and the slots `inject_adapter_weights`
//! routes into, and every constructor refuses a target set the forward does not
//! train (0.72 R15a C2b).

use super::adapter_tests::filled;
use super::*;
use crate::finetune::multi_adapter_pipeline::{
    AdapterConfig, AdapterSchedule, MultiAdapterPipeline,
};

/// How `all_linear` is refused: every target outside `q_proj`, `v_proj`.
const UNTRAINED_ALL: &str = "would not train: k_proj, o_proj, gate_proj, up_proj, down_proj";

fn targets(names: &[&str]) -> LoraTargets {
    LoraTargets::parse(names).expect("known targets must parse")
}

fn with_targets(names: &[&str]) -> InstructConfig {
    InstructConfig { lora_rank: 4, lora_targets: targets(names), ..InstructConfig::default() }
}

/// A two-layer model in which q_proj, o_proj and down_proj are not square, so
/// a swapped `d_out` and `d_in` shows: one KV head and a head dim of 16 make
/// q_proj `[32, 64]`, k_proj and v_proj `[16, 64]` and o_proj `[64, 32]`, with
/// hidden 64 and intermediate 256. Projection `p` of layer `l`
/// (in [`LoraTarget::ALL`] order) holds `filled(len, 1 + 7·l + p)`, so a base
/// weight copied from the wrong projection or layer shows.
fn tagged_model() -> (TransformerConfig, Transformer) {
    let config = TransformerConfig {
        num_kv_heads: 1,
        head_dim_override: Some(16),
        ..TransformerConfig::tiny()
    };
    let mut model = Transformer::new(&config);
    for (layer, block) in model.layers.iter_mut().enumerate() {
        let (attn, ffn) = (&mut block.self_attn, &mut block.ffn);
        let projections = [
            &mut attn.w_q,
            &mut attn.w_k,
            &mut attn.w_v,
            &mut attn.w_o,
            &mut ffn.w_gate,
            &mut ffn.w_up,
            &mut ffn.w_down,
        ];
        for (position, weight) in projections.into_iter().enumerate() {
            *weight = Tensor::from_vec(filled(weight.len(), tag(layer, position)), false);
        }
    }
    (config, model)
}

fn tag(layer: usize, position: usize) -> usize {
    1 + LoraTarget::ALL.len() * layer + position
}

/// `(d_out, d_in)` of `target` in [`tagged_model`], from the projection layouts.
fn shape(target: LoraTarget) -> (usize, usize) {
    let (h, q, kv, i) = (64, 32, 16, 256);
    match target {
        LoraTarget::Q => (q, h),
        LoraTarget::K | LoraTarget::V => (kv, h),
        LoraTarget::O => (h, q),
        LoraTarget::Gate | LoraTarget::Up => (i, h),
        LoraTarget::Down => (h, i),
    }
}

/// A tensor name as PEFT writes it: attention projections under `self_attn`,
/// MLP projections under `mlp`.
fn peft_tensor(layer: usize, target: LoraTarget, kind: &str) -> String {
    let block = if target < LoraTarget::Gate { "self_attn" } else { "mlp" };
    format!("base_model.model.model.layers.{layer}.{block}.{target}.{kind}.weight")
}

#[test]
fn falsify_lora_target_selection_v1_004_builds_one_adapter_per_layer_and_target() {
    let (model_config, model) = tagged_model();
    let num_layers = model_config.num_hidden_layers;
    for names in [&["all_linear"][..], &["qv"], &["mlp"], &["o_proj", "k_proj"]] {
        let selected = targets(names);
        let layers =
            InstructPipeline::build_lora_layers(&model, &model_config, &with_targets(names));
        assert_eq!(layers.len(), selected.per_layer() * num_layers, "{names:?}");
        for layer in 0..num_layers {
            for &target in selected.as_slice() {
                let slot = selected.slot(layer, target).expect("a selected target has a slot");
                let lora = &layers[slot];
                let (d_out, d_in) = shape(target);
                let position = LoraTarget::ALL.iter().position(|&t| t == target).expect("in ALL");
                assert_eq!(
                    (lora.d_out(), lora.d_in()),
                    (d_out, d_in),
                    "{names:?} {layer} {target}"
                );
                assert_eq!(
                    lora.base_weight().data().to_vec(),
                    filled(d_out * d_in, tag(layer, position)),
                    "{names:?}: slot {slot} must wrap layer {layer}'s {target}"
                );
            }
        }
    }
    // The default is the pre-0.72 layout: Q then V, per layer.
    let layers =
        InstructPipeline::build_lora_layers(&model, &model_config, &InstructConfig::default());
    assert_eq!(layers.len(), 2 * num_layers);
    for (slot, lora) in layers.iter().enumerate() {
        let target = if slot % 2 == 0 { LoraTarget::Q } else { LoraTarget::V };
        assert_eq!((lora.d_out(), lora.d_in()), shape(target), "default slot {slot}");
    }
}

#[test]
fn falsify_lora_target_selection_v1_004_routes_every_selected_target_by_name() {
    let (model_config, model) = tagged_model();
    let num_layers = model_config.num_hidden_layers;
    let all = targets(&["all_linear"]);
    let fresh =
        InstructPipeline::build_lora_layers(&model, &model_config, &with_targets(&["all_linear"]));
    // Slot s gets filled(len, 1000 + 2·s) as its A and 1001 + 2·s as its B,
    // the tensors listed in reverse slot order.
    let mut weights = Vec::new();
    for layer in (0..num_layers).rev() {
        for &target in all.as_slice().iter().rev() {
            let slot = all.slot(layer, target).expect("selected");
            let (a, b) = (fresh[slot].lora_a().len(), fresh[slot].lora_b().len());
            weights.push((peft_tensor(layer, target, "lora_A"), filled(a, 1000 + 2 * slot)));
            weights.push((peft_tensor(layer, target, "lora_B"), filled(b, 1001 + 2 * slot)));
        }
    }
    let mut layers = fresh.clone();
    InstructPipeline::inject_adapter_weights(&mut layers, &weights, &all)
        .expect("an all_linear adapter must load into all_linear layers");
    for (slot, lora) in layers.iter().enumerate() {
        assert_eq!(lora.lora_a().data().to_vec(), filled(lora.lora_a().len(), 1000 + 2 * slot));
        assert_eq!(lora.lora_b().data().to_vec(), filled(lora.lora_b().len(), 1001 + 2 * slot));
    }

    // The same k_proj and down_proj tensors fit no slot of the default layers.
    let mut layers =
        InstructPipeline::build_lora_layers(&model, &model_config, &with_targets(&["qv"]));
    for target in [LoraTarget::K, LoraTarget::Down] {
        let name = peft_tensor(0, target, "lora_A");
        let tensor = weights.iter().find(|(n, _)| *n == name).expect("in the adapter").clone();
        let err = InstructPipeline::inject_adapter_weights(
            &mut layers,
            &[tensor],
            &LoraTargets::default(),
        )
        .expect_err("a tensor of an unselected target must refuse the load");
        assert!(err.to_string().contains(&name), "the error names {name}: {err}");
    }
}

/// Each constructor that returns a `Result` refuses before it loads anything:
/// a model path that does not load yields the target error, not a load error.
fn assert_refuses_before_loading(load: &dyn Fn(InstructConfig) -> crate::Result<InstructPipeline>) {
    let cases: [(&[&str], &str); 3] = [
        (&["all_linear"], UNTRAINED_ALL),
        (&["q_proj"], "missing: v_proj"),
        (&["k_proj", "v_proj"], "would not train: k_proj; missing: q_proj"),
    ];
    for (names, want) in cases {
        let Err(err) = load(with_targets(names)) else {
            panic!("{names:?} must be refused");
        };
        let err = err.to_string();
        assert!(err.contains("LoRA targets") && err.contains(want), "{names:?}: {err}");
    }
    // Control: the default targets get past the check and fail on the model.
    let Err(err) = load(InstructConfig { lora_rank: 4, ..InstructConfig::default() }) else {
        panic!("a model path that does not load must fail");
    };
    assert!(!err.to_string().contains("LoRA targets"), "the default must pass the check: {err}");
}

#[test]
fn falsify_lora_target_selection_v1_004_from_pretrained_refuses_before_loading() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_config = TransformerConfig::tiny();
    assert_refuses_before_loading(&|config| {
        InstructPipeline::from_pretrained(dir.path(), &model_config, config)
    });
}

#[test]
fn falsify_lora_target_selection_v1_004_from_apr_refuses_before_loading() {
    // `from_apr` requires the path to exist; its bytes are not an APR file.
    let file = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(file.path(), b"not an APR file\n".repeat(16)).expect("write");
    let model_config = TransformerConfig::tiny();
    assert_refuses_before_loading(&|config| {
        InstructPipeline::from_apr(file.path(), &model_config, config)
    });
}

#[test]
#[should_panic(expected = "would not train: k_proj, o_proj, gate_proj, up_proj, down_proj")]
fn falsify_lora_target_selection_v1_004_new_refuses() {
    let _ = InstructPipeline::new(&TransformerConfig::tiny(), with_targets(&["all_linear"]));
}

#[test]
#[should_panic(expected = "would not train: k_proj, o_proj, gate_proj, up_proj, down_proj")]
fn falsify_lora_target_selection_v1_004_add_adapter_refuses() {
    let base = InstructPipeline::new(&TransformerConfig::tiny(), InstructConfig::default());
    let mut pipeline = MultiAdapterPipeline::new(base, AdapterSchedule::RoundRobin);
    let adapter = |instruct_config| AdapterConfig {
        data_path: "data.jsonl".into(),
        checkpoint_dir: "ckpt".into(),
        instruct_config,
    };
    // Control: an adapter with the default targets is added.
    pipeline.add_adapter(adapter(InstructConfig::default()), Vec::new(), Vec::new());
    assert_eq!(pipeline.adapters.len(), 1);
    pipeline.add_adapter(adapter(with_targets(&["all_linear"])), Vec::new(), Vec::new());
}
