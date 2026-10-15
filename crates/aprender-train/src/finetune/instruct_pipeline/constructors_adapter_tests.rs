//! ENT-269 trained-adapter loading tests for `InstructPipeline` (#4698).

use super::*;

/// Tiny Q/V LoRA layers plus the layer count `load_trained_adapter` is given.
fn tiny_layers() -> (Vec<LoRALayer>, usize) {
    let model_config = TransformerConfig { num_kv_heads: 1, ..TransformerConfig::tiny() };
    let model = Transformer::new(&model_config);
    let config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let layers = InstructPipeline::build_lora_layers(&model, &model_config, &config);
    (layers, model_config.num_hidden_layers)
}

/// Every A and B weight, in order, so a test can tell whether anything moved.
fn snapshot(layers: &[LoRALayer]) -> Vec<Vec<f32>> {
    layers.iter().flat_map(|l| [l.lora_a().data().to_vec(), l.lora_b().data().to_vec()]).collect()
}

/// #4698 falsifier: an `adapter_model.safetensors` that cannot be read refuses the
/// load and names the file; it never falls through to fresh adapters. No file at
/// all is still fresh adapters, untouched.
#[test]
fn unreadable_trained_adapter_refuses_the_load_4698() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (mut layers, num_layers) = tiny_layers();
    assert!(!layers.is_empty(), "rank 4 must build LoRA layers");
    let before = snapshot(&layers);

    InstructPipeline::load_trained_adapter(dir.path(), &mut layers, num_layers)
        .expect("no adapter file means fresh adapters");
    assert_eq!(snapshot(&layers), before, "no adapter file must leave the layers untouched");

    // A valid adapter_config.json beside the junk weights, so the refusal comes from the
    // safetensors parser and not from a missing or bad config.
    let config = r#"{"peft_type":"LORA","r":4,"lora_alpha":8.0,"target_modules":["q_proj","v_proj"],"lora_dropout":0.0,"bias":"none"}"#;
    crate::lora::PeftAdapterConfig::from_json(config).expect("fixture config must parse");
    std::fs::write(dir.path().join("adapter_config.json"), config).expect("write config");
    std::fs::write(dir.path().join("adapter_model.safetensors"), b"not a safetensors file")
        .expect("write adapter");
    let err = InstructPipeline::load_trained_adapter(dir.path(), &mut layers, num_layers)
        .expect_err("an unreadable trained adapter must refuse the load");
    let msg = err.to_string();
    assert!(
        msg.contains("Failed to load adapter_model.safetensors"),
        "the refusal must come from the safetensors parser, got: {msg}"
    );
    assert!(
        msg.contains(&dir.path().join("adapter_model.safetensors").display().to_string()),
        "the refusal must name the file, got: {msg}"
    );
    assert_eq!(snapshot(&layers), before, "a refused load must leave the layers untouched");
}
