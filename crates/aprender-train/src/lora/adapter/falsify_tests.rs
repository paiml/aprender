//! Falsification tests for LoRA checkpoint & serialization (Layer 4)
//! From spec: entrenar/docs/lora-qlora-enhancement.md Section 6
//!
//! F-CKPT-001: Adapter save/load roundtrip preserves forward output
//! F-CKPT-002: PEFT export has valid JSON schema
//! F-CKPT-003: Merged model has no LoRA tensors
//! F-CKPT-004: Merged model is same size as base model
//! F-CKPT-005: Resume from adapter continues training

#![allow(clippy::unwrap_used)]

use super::*;
use crate::lora::layer::LoRALayer;
use crate::lora::LoRAConfig;
use crate::Tensor;
use approx::assert_abs_diff_eq;
use tempfile::TempDir;

// ========================================================================
// Helpers
// ========================================================================

/// Create a LoRA layer with non-zero weights for meaningful forward output.
fn make_layer_with_weights(d_out: usize, d_in: usize, rank: usize, alpha: f32) -> LoRALayer {
    let base_weight = Tensor::from_vec(vec![0.5; d_out * d_in], false);
    let mut layer = LoRALayer::new(base_weight, d_out, d_in, rank, alpha);

    // Set non-trivial LoRA weights so forward produces non-zero adaptation
    let a_data: Vec<f32> = (0..rank * d_in).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let b_data: Vec<f32> = (0..d_out * rank).map(|i| 0.05 * (i as f32 + 1.0)).collect();
    *layer.lora_a_mut().data_mut() = ndarray::Array1::from_vec(a_data);
    *layer.lora_b_mut().data_mut() = ndarray::Array1::from_vec(b_data);

    layer
}

// ========================================================================
// F-CKPT-001: Adapter save/load roundtrip preserves forward output
// ========================================================================

/// Save adapter, create fresh model, load adapter, assert forward(x) identical
/// to 6 decimal places.
#[test]
fn test_falsify_f_ckpt_001_roundtrip_preserves_output() {
    let d_out = 8;
    let d_in = 4;
    let rank = 2;
    let alpha = 4.0;

    let layer = make_layer_with_weights(d_out, d_in, rank, alpha);

    // Forward with original layer
    let x = Tensor::from_vec(vec![1.0, 0.5, -0.3, 0.7], true);
    let original_output = layer.forward(&x);

    // Save adapter
    let tmp = TempDir::new().unwrap();
    let adapter_path = tmp.path().join("adapter.json");
    let adapter = LoRAAdapter::from_layer(&layer, rank, alpha);
    adapter.save(&adapter_path).unwrap();

    // Load into a FRESH base weight (same values, new allocation)
    let fresh_base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
    let loaded_adapter = LoRAAdapter::load(&adapter_path).unwrap();
    let loaded_layer = loaded_adapter.to_layer(fresh_base).unwrap();

    // Forward with loaded layer
    let loaded_output = loaded_layer.forward(&x);

    // Assert identical to 6 decimal places
    assert_eq!(original_output.len(), loaded_output.len());
    for i in 0..original_output.len() {
        assert_abs_diff_eq!(original_output.data()[i], loaded_output.data()[i], epsilon = 1e-6);
    }
}

/// Roundtrip with larger dimensions to stress floating-point precision.
#[test]
fn test_falsify_f_ckpt_001_roundtrip_larger_dimensions() {
    let d_out = 32;
    let d_in = 16;
    let rank = 4;
    let alpha = 8.0;

    let layer = make_layer_with_weights(d_out, d_in, rank, alpha);

    let x = Tensor::from_vec((0..d_in).map(|i| (i as f32) * 0.1 - 0.5).collect(), true);
    let original_output = layer.forward(&x);

    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("adapter_large.json");
    save_adapter(&layer, rank, alpha, &path).unwrap();

    let fresh_base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
    let loaded_layer = load_adapter(fresh_base, &path).unwrap();
    let loaded_output = loaded_layer.forward(&x);

    for i in 0..original_output.len() {
        assert_abs_diff_eq!(original_output.data()[i], loaded_output.data()[i], epsilon = 1e-6);
    }
}

// ========================================================================
// F-CKPT-002: PEFT export has valid JSON schema
// ========================================================================

/// Export adapter_config.json + adapter_model.safetensors, validate JSON has
/// correct fields (r, lora_alpha, target_modules, peft_type="LORA").
#[test]
fn test_falsify_f_ckpt_002_peft_export_valid_schema() {
    let config = LoRAConfig::new(8, 16.0).target_qv_projections();
    let layer = make_layer_with_weights(16, 16, 8, 16.0);

    let mut bundle = PeftAdapterBundle::new(config.clone());
    bundle = bundle.with_base_model("test/model-7b");
    bundle.add_adapter("model.layers.0.self_attn.q_proj", &layer);

    let tmp = TempDir::new().unwrap();
    bundle.save_peft(tmp.path()).unwrap();

    // Verify both files exist
    assert!(tmp.path().join("adapter_config.json").exists());
    assert!(tmp.path().join("adapter_model.safetensors").exists());

    // Parse and validate adapter_config.json
    let json_str = std::fs::read_to_string(tmp.path().join("adapter_config.json")).unwrap();
    let parsed: PeftAdapterConfig = serde_json::from_str(&json_str).unwrap();

    assert_eq!(parsed.peft_type, "LORA");
    assert_eq!(parsed.r, 8);
    assert_eq!(parsed.lora_alpha, 16.0);
    assert!(parsed.target_modules.contains(&"q_proj".to_string()));
    assert!(parsed.target_modules.contains(&"v_proj".to_string()));
    assert_eq!(parsed.base_model_name_or_path, Some("test/model-7b".to_string()));
}

/// Validate that PEFT JSON roundtrips through from_json/to_json.
#[test]
fn test_falsify_f_ckpt_002_peft_json_roundtrip() {
    let config = LoRAConfig::new(4, 8.0).target_attention_projections();
    let peft = PeftAdapterConfig::from_lora_config(&config, Some("base/model"));

    let json = peft.to_json().unwrap();
    let roundtripped = PeftAdapterConfig::from_json(&json).unwrap();

    assert_eq!(peft, roundtripped);
}

// ========================================================================
// F-CKPT-003: Merged model has no LoRA tensors
// ========================================================================

/// After merge, inspect safetensors keys — assert NO key contains "lora_a"
/// or "lora_b".
#[test]
fn test_falsify_f_ckpt_003_merged_model_no_lora_tensors() {
    let layer1 = make_layer_with_weights(8, 16, 4, 8.0);
    let layer2 = make_layer_with_weights(8, 16, 4, 8.0);

    let layers: Vec<(&str, &LoRALayer)> =
        vec![("model.layers.0.q_proj.weight", &layer1), ("model.layers.0.v_proj.weight", &layer2)];

    let merged = merge_and_collect(&layers);

    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("merged.safetensors");
    merged.save_safetensors(&path).unwrap();

    // Load and inspect tensor names
    let data = std::fs::read(&path).unwrap();
    let loaded = safetensors::SafeTensors::deserialize(&data).unwrap();

    for name in loaded.names() {
        assert!(
            !name.contains("lora_a"),
            "Merged model must not contain lora_a tensors, found: {name}"
        );
        assert!(
            !name.contains("lora_b"),
            "Merged model must not contain lora_b tensors, found: {name}"
        );
        assert!(
            !name.contains("lora_A"),
            "Merged model must not contain lora_A tensors, found: {name}"
        );
        assert!(
            !name.contains("lora_B"),
            "Merged model must not contain lora_B tensors, found: {name}"
        );
    }
}

/// Merged model should contain only the base weight tensor names.
#[test]
fn test_falsify_f_ckpt_003_merged_keys_match_layer_names() {
    let layer = make_layer_with_weights(8, 8, 2, 4.0);

    let layers: Vec<(&str, &LoRALayer)> = vec![("attn.weight", &layer)];
    let merged = merge_and_collect(&layers);

    assert!(merged.tensors.contains_key("attn.weight"));
    assert_eq!(merged.tensors.len(), 1);
}

// ========================================================================
// F-CKPT-004: Merged model is same size as base model
// ========================================================================

/// size(merged) vs size(base), within 1%.
#[test]
fn test_falsify_f_ckpt_004_merged_model_same_size_as_base() {
    let d_out = 16;
    let d_in = 32;
    let rank = 4;

    let layer = make_layer_with_weights(d_out, d_in, rank, 8.0);

    // Base model param count: just the base weight
    let base_param_count = (d_out * d_in) as u64;

    let layers: Vec<(&str, &LoRALayer)> = vec![("weight", &layer)];
    let merged = merge_and_collect(&layers);

    let merged_param_count = merged.param_count();

    // Merged model should have exactly the same number of parameters as base
    assert_eq!(
        merged_param_count, base_param_count,
        "Merged param count ({merged_param_count}) must equal base param count ({base_param_count})"
    );

    // Also verify the 1% tolerance contract (trivially satisfied when equal)
    let ratio = merged_param_count as f64 / base_param_count as f64;
    assert!((ratio - 1.0).abs() < 0.01, "Merged/base size ratio {ratio} exceeds 1% tolerance");
}

/// Multi-layer merged model size matches sum of base weights.
#[test]
fn test_falsify_f_ckpt_004_multi_layer_size_matches_base() {
    let layer1 = make_layer_with_weights(8, 16, 4, 8.0);
    let layer2 = make_layer_with_weights(16, 8, 2, 4.0);

    let base_params = (8 * 16 + 16 * 8) as u64;

    let layers: Vec<(&str, &LoRALayer)> = vec![("w1", &layer1), ("w2", &layer2)];
    let merged = merge_and_collect(&layers);

    assert_eq!(merged.param_count(), base_params);
}

// ========================================================================
// F-CKPT-005: Resume from adapter continues training
// ========================================================================

/// Save at step N, load, continue training, loss trajectory continuous.
/// We simulate training by manually updating LoRA weights and verifying that
/// the forward output after load+update is consistent with continued training.
#[test]
fn test_falsify_f_ckpt_005_resume_continues_training() {
    let d_out = 4;
    let d_in = 4;
    let rank = 2;
    let alpha = 2.0;
    let lr = 0.01;

    let mut layer = make_layer_with_weights(d_out, d_in, rank, alpha);

    let x = Tensor::from_vec(vec![1.0, 0.5, -0.3, 0.7], true);
    let target = vec![1.0, 0.0, 0.5, -0.5];

    // Simulate N steps of "training" (simple gradient-free weight perturbation)
    for step in 0..5 {
        let output = layer.forward(&x);
        // Compute simple MSE-like gradient direction on lora_b
        let grad: Vec<f32> = (0..d_out * rank)
            .map(|i| {
                let o_idx = i / rank;
                let diff = output.data()[o_idx] - target[o_idx];
                diff * lr * (step as f32 + 1.0) * 0.01
            })
            .collect();
        let b_data = layer.lora_b().data().to_vec();
        let updated: Vec<f32> = b_data.iter().zip(&grad).map(|(w, g)| w - g).collect();
        *layer.lora_b_mut().data_mut() = ndarray::Array1::from_vec(updated);
    }

    // Capture output at step N
    let output_at_n = layer.forward(&x);

    // Save adapter at step N
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("checkpoint_step5.json");
    save_adapter(&layer, rank, alpha, &path).unwrap();

    // Load into fresh base
    let fresh_base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
    let mut resumed_layer = load_adapter(fresh_base, &path).unwrap();

    // Verify output matches before continuing
    let resumed_output = resumed_layer.forward(&x);
    for i in 0..d_out {
        assert_abs_diff_eq!(output_at_n.data()[i], resumed_output.data()[i], epsilon = 1e-6);
    }

    // Continue training for 5 more steps on the resumed layer
    for step in 5..10 {
        let output = resumed_layer.forward(&x);
        let grad: Vec<f32> = (0..d_out * rank)
            .map(|i| {
                let o_idx = i / rank;
                let diff = output.data()[o_idx] - target[o_idx];
                diff * lr * (step as f32 + 1.0) * 0.01
            })
            .collect();
        let b_data = resumed_layer.lora_b().data().to_vec();
        let updated: Vec<f32> = b_data.iter().zip(&grad).map(|(w, g)| w - g).collect();
        *resumed_layer.lora_b_mut().data_mut() = ndarray::Array1::from_vec(updated);
    }

    // Also continue training the original layer for 5 more steps
    for step in 5..10 {
        let output = layer.forward(&x);
        let grad: Vec<f32> = (0..d_out * rank)
            .map(|i| {
                let o_idx = i / rank;
                let diff = output.data()[o_idx] - target[o_idx];
                diff * lr * (step as f32 + 1.0) * 0.01
            })
            .collect();
        let b_data = layer.lora_b().data().to_vec();
        let updated: Vec<f32> = b_data.iter().zip(&grad).map(|(w, g)| w - g).collect();
        *layer.lora_b_mut().data_mut() = ndarray::Array1::from_vec(updated);
    }

    // Loss trajectory must be continuous: original and resumed must converge identically
    let final_original = layer.forward(&x);
    let final_resumed = resumed_layer.forward(&x);

    for i in 0..d_out {
        assert_abs_diff_eq!(final_original.data()[i], final_resumed.data()[i], epsilon = 1e-6);
    }
}

/// Verify that resumed adapter's LoRA weights are not reset to init values.
#[test]
fn test_falsify_f_ckpt_005_resumed_weights_not_reinitialized() {
    let d_out = 4;
    let d_in = 4;
    let rank = 2;
    let alpha = 4.0;

    let mut layer = make_layer_with_weights(d_out, d_in, rank, alpha);

    // Modify weights to simulate training progress
    let trained_b: Vec<f32> = (0..d_out * rank).map(|i| 0.5 + i as f32 * 0.1).collect();
    *layer.lora_b_mut().data_mut() = ndarray::Array1::from_vec(trained_b.clone());

    // Save and reload
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("trained.json");
    save_adapter(&layer, rank, alpha, &path).unwrap();

    let fresh_base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
    let loaded = load_adapter(fresh_base, &path).unwrap();

    // Loaded weights must match the trained weights, not the init values
    let loaded_b = loaded.lora_b().data().to_vec();
    for (i, (&expected, &actual)) in trained_b.iter().zip(loaded_b.iter()).enumerate() {
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6,);
        // Verify they are NOT the default zero-init values
        assert!(
            actual.abs() > 1e-8,
            "Weight at index {i} is near zero ({actual}), suggesting reinitialization"
        );
    }
}
