//! FALSIFY-LORA_TARGET_SELECTION_V1_017: the instruct trainer's checkpoint names
//! each adapter by its own layer and target, and records the targets
//! (0.72 R15a C11a; contract `lora-target-selection-v1`, instruct_checkpoint_names).

use std::collections::BTreeSet;

use super::{InstructEpochMetrics, InstructTrainer, InstructTrainingConfig};
use crate::finetune::instruct_corpus::InstructSample;
use crate::finetune::instruct_pipeline::{InstructConfig, InstructPipeline};
use crate::lora::{LoRALayer, LoraTarget, LoraTargets};
use crate::transformer::TransformerConfig;
use crate::Tensor;

const RANK: usize = 2;
const ALPHA: f32 = 4.0;

/// A two-layer model in which q_proj, o_proj and down_proj are not square, so a
/// tensor saved under another target's name has another shape.
fn config() -> TransformerConfig {
    TransformerConfig { num_kv_heads: 1, head_dim_override: Some(16), ..TransformerConfig::tiny() }
}

fn all_linear() -> LoraTargets {
    let names: Vec<&str> = LoraTarget::ALL.iter().map(|t| t.module_name()).collect();
    LoraTargets::parse(&names).expect("every target name must parse")
}

/// `len` values distinct for each `seed`.
fn spread(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7919 + seed * 104_729) % 1000) as f32 / 1000.0 - 0.5).collect()
}

/// A trainer whose pipeline holds `targets` and one adapter per layer and target
/// in slot order, every A and B distinct.
fn trainer(targets: &LoraTargets) -> InstructTrainer {
    let config = config();
    let instruct = InstructConfig { lora_rank: RANK, max_seq_len: 32, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&config, instruct);
    let mut layers = Vec::new();
    for _layer in 0..config.num_hidden_layers {
        for &target in targets.as_slice() {
            let slot = layers.len();
            let (d_out, d_in) = target.dims(&config);
            let mut lora =
                LoRALayer::new(Tensor::zeros(d_out * d_in, false), d_out, d_in, RANK, ALPHA);
            *lora.lora_a_mut() = Tensor::from_vec(spread(RANK * d_in, 2 * slot + 1), true);
            *lora.lora_b_mut() = Tensor::from_vec(spread(d_out * RANK, 2 * slot + 2), true);
            layers.push(lora);
        }
    }
    pipeline.config.lora_targets = targets.clone();
    pipeline.lora_layers = layers;
    let corpus = (0..10)
        .map(|i| InstructSample {
            instruction: format!("Write function {i}"),
            response: format!("def func_{i}():\n    return {i}"),
            system: None,
            metadata: None,
        })
        .collect();
    InstructTrainer::new(pipeline, corpus, InstructTrainingConfig::default())
        .expect("a ten-sample corpus must build a trainer")
}

fn metrics() -> InstructEpochMetrics {
    InstructEpochMetrics {
        epoch: 0,
        train_loss: 1.0,
        train_perplexity: 2.7,
        val_loss: 1.0,
        val_perplexity: 2.7,
        learning_rate: 1e-4,
        epoch_time_ms: 1,
        samples_per_sec: 1.0,
    }
}

fn values(t: &Tensor) -> Vec<f32> {
    t.data().as_slice().expect("contiguous").to_vec()
}

/// Save a checkpoint of `trainer` and check that it holds exactly the adapters
/// of `targets`, each under its own layer and target with its slot's shape and
/// values, and that metadata.json lists `targets` in slot order.
fn check_checkpoint(targets: &LoraTargets) {
    let mut trainer = trainer(targets);
    let dir = tempfile::tempdir().expect("temp dir");
    trainer.save_checkpoint(dir.path(), 0, &metrics()).expect("save_checkpoint");

    let bytes = std::fs::read(dir.path().join("model.safetensors")).expect("model.safetensors");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("a readable safetensors file");
    let held: BTreeSet<String> = st.names().into_iter().cloned().collect();

    let t = targets.as_slice();
    let layers = &trainer.pipeline.lora_layers;
    let mut expected = BTreeSet::new();
    for (slot, lora) in layers.iter().enumerate() {
        let (layer, target) = (slot / t.len(), t[slot % t.len()]);
        let module = target.module_name();
        let (name_a, name_b) =
            (format!("lora.{layer}.{module}.lora_a"), format!("lora.{layer}.{module}.lora_b"));
        for (name, shape, want) in [
            (&name_a, vec![RANK, lora.d_in()], values(lora.lora_a())),
            (&name_b, vec![lora.d_out(), RANK], values(lora.lora_b())),
        ] {
            let view = st.tensor(name).unwrap_or_else(|e| panic!("slot {slot}: no {name}: {e}"));
            assert_eq!(view.shape(), shape.as_slice(), "slot {slot}: shape of {name}");
            let got: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got, want, "slot {slot}: values of {name}");
        }
        expected.insert(name_a);
        expected.insert(name_b);
    }
    assert_eq!(held, expected, "the checkpoint holds exactly one A and one B per slot");
    assert_eq!(held.len(), 2 * config().num_hidden_layers * t.len());

    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.path().join("metadata.json")).expect("metadata.json"),
    )
    .expect("metadata.json is JSON");
    let listed: Vec<&str> = t.iter().map(|t| t.module_name()).collect();
    assert_eq!(meta["lora_targets"], serde_json::json!(listed), "metadata.json lora_targets");
}

#[test]
fn falsify_lora_target_selection_v1_017_all_linear_names_every_adapter() {
    check_checkpoint(&all_linear());
}

#[test]
fn falsify_lora_target_selection_v1_017_default_names_unchanged() {
    let targets = LoraTargets::default();
    check_checkpoint(&targets);
    // The names the instruct checkpoint has always used for q_proj, v_proj.
    let mut trainer = trainer(&targets);
    let dir = tempfile::tempdir().expect("temp dir");
    trainer.save_checkpoint(dir.path(), 0, &metrics()).expect("save_checkpoint");
    let bytes = std::fs::read(dir.path().join("model.safetensors")).expect("model.safetensors");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("a readable safetensors file");
    let held: BTreeSet<&str> = st.names().into_iter().map(String::as_str).collect();
    let old: BTreeSet<&str> = [
        "lora.0.q_proj.lora_a",
        "lora.0.q_proj.lora_b",
        "lora.0.v_proj.lora_a",
        "lora.0.v_proj.lora_b",
        "lora.1.q_proj.lora_a",
        "lora.1.q_proj.lora_b",
        "lora.1.v_proj.lora_a",
        "lora.1.v_proj.lora_b",
    ]
    .into_iter()
    .collect();
    assert_eq!(held, old);
}
