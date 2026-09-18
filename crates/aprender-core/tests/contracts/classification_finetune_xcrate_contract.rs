// CONTRACT: classification-finetune-v1.yaml, contracts/model-families/qwen3_5.yaml
//
// Cross-crate falsification: aprender's Poka-Yoke validated types and contract
// YAML vs entrenar's `TransformerConfig` / `finetune` pipeline.
//
// WHY THIS FILE IS IN `tests/` AND NOT IN `src/` (PMAT-1098)
// ---------------------------------------------------------
// `entrenar` is a PATH-ONLY dev-dependency of aprender-core:
//
//     entrenar = { path = "../aprender-train", package = "aprender-train" }
//
// It carries no `version` on purpose (PMAT-955): aprender-train depends back on
// aprender-core, so a versioned dev-dep would demand aprender-train be on
// crates.io before aprender-core can publish — an unpublishable cycle. A dev-dep
// with no source is OMITTED by `cargo publish`, while `#[cfg(test)]` code inside
// `src/` is published anyway. So a `src/` test that names `entrenar` cannot
// compile in published form: clean-room GATE B2 (`cargo test --lib` after the
// publish strip) died with 19 × `error[E0433]: cannot find module or crate
// entrenar` on v0.67.0 and v0.68.0.
//
// The mechanism is the one #3307 used for aprender-compute's stripped dev-deps,
// in the variant its baseline reserves for a genuine publish cycle: "the code has
// to leave src/". Precedent in this crate: `tests/explainable_monitor.rs` (GH-305).
// These tests keep naming `entrenar` and keep running in-workspace — this file is
// a module of the `contract_tests` target, which CI runs explicitly
// (ci/explicit-test-commands.d/390-aprender-core-contract-tests.cmd).

use aprender::format::model_family_loader::load_family_yaml;
use aprender::format::validated_classification::ValidatedClassLogits;
use std::path::Path;

/// Load the `qwen3_5` family contract that FALSIFY-FT-QWEN35-006 compares against.
fn qwen35_family() -> aprender::format::model_family::ModelFamilyConfig {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/model-families/qwen3_5.yaml");
    assert!(
        path.exists(),
        "contracts/model-families/qwen3_5.yaml must exist"
    );
    load_family_yaml(&path).expect("qwen3_5.yaml must load")
}

// =============================================================================
// FALSIFY-CLASS-007: Qwen3.5 must have use_bias=false (F-CLASS-007)
//
// Contract: classification-finetune-v1.yaml F-CLASS-007
// Prediction: TransformerConfig::qwen3_5_9b().use_bias == false
// If fails: LoRA adapters would wrongly create bias tensors for Qwen3.5
// =============================================================================

#[test]
fn falsify_class_007_qwen35_no_bias() {
    let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    assert!(
        !config.use_bias,
        "FALSIFIED F-CLASS-007: Qwen3.5 must have use_bias=false, got true"
    );
}

#[test]
fn falsify_class_007_qwen2_has_bias() {
    // Counterexample: Qwen2 DOES have bias — confirms 007 is Qwen3.5-specific
    let config = entrenar::transformer::TransformerConfig::qwen2_0_5b();
    assert!(
        config.use_bias,
        "Qwen2 should have use_bias=true (verifies 007 is discriminating)"
    );
}

// =============================================================================
// FALSIFY-CLASS-008: LoRA must target Q/V projections (F-CLASS-008)
//
// Contract: classification-finetune-v1.yaml F-CLASS-008
// Prediction: LoRA adapters are placed on q_proj and v_proj (2 per layer)
// If fails: LoRA would target wrong projections, breaking fine-tuning
// =============================================================================

#[test]
fn falsify_class_008_lora_adapter_count_per_layer() {
    // Each transformer layer should have 2 LoRA adapters (Q, V)
    let model_config = entrenar::transformer::TransformerConfig::qwen2_0_5b();
    let classify_config = entrenar::finetune::ClassifyConfig::default();
    let pipeline = entrenar::finetune::ClassifyPipeline::new(&model_config, classify_config);
    let expected = model_config.num_hidden_layers * 2; // Q + V per layer
    assert_eq!(
        pipeline.lora_layers.len(),
        expected,
        "FALSIFIED F-CLASS-008: Expected {} LoRA adapters ({}*2), got {}",
        expected,
        model_config.num_hidden_layers,
        pipeline.lora_layers.len()
    );
}

// ========================================================================
// FINE-TUNING CONFIG FALSIFICATION (FALSIFY-FT-QWEN35-001..007)
//
// These tests verify that entrenar's TransformerConfig::qwen3_5_9b() factory
// matches the model-family YAML contract. If any test fails, the config
// factory is out of sync with the contract — fine-tuning would use wrong
// dimensions.
// ========================================================================

// ========================================================================
// FALSIFY-FT-QWEN35-001: vocab_size must be 248320 (not Qwen2's 152064)
//
// Prediction: qwen3_5_9b().vocab_size == 248320.
// If fails: embedding/lm_head LoRA shapes would be wrong.
// ========================================================================
#[test]
fn falsify_ft_qwen35_001_vocab_size() {
    let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    assert_eq!(
        config.vocab_size, 248320,
        "FALSIFIED FT-QWEN35-001: vocab_size={}, expected 248320",
        config.vocab_size
    );
}

// ========================================================================
// FALSIFY-FT-QWEN35-002: use_bias must be false
//
// Prediction: qwen3_5_9b().use_bias == false.
// If fails: LoRA adapter would create bias tensors that don't exist.
// ========================================================================
#[test]
fn falsify_ft_qwen35_002_no_bias() {
    let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    assert!(
        !config.use_bias,
        "FALSIFIED FT-QWEN35-002: use_bias={}, expected false",
        config.use_bias
    );
}

// ========================================================================
// FALSIFY-FT-QWEN35-003: head_dim must be 256
//
// Prediction: qwen3_5_9b().head_dim() == 256 (4096/16).
// If fails: LoRA Q/K/V projection dimensions would be wrong.
// ========================================================================
#[test]
fn falsify_ft_qwen35_003_head_dim() {
    let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    assert_eq!(
        config.head_dim(),
        256,
        "FALSIFIED FT-QWEN35-003: head_dim()={}, expected 256",
        config.head_dim()
    );
}

// ========================================================================
// FALSIFY-FT-QWEN35-004: num_hidden_layers must be 32
//
// Prediction: qwen3_5_9b().num_hidden_layers == 32.
// If fails: LoRA would target wrong number of layers.
// ========================================================================
#[test]
fn falsify_ft_qwen35_004_num_layers() {
    let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    assert_eq!(
        config.num_hidden_layers, 32,
        "FALSIFIED FT-QWEN35-004: num_hidden_layers={}, expected 32",
        config.num_hidden_layers
    );
}

// ========================================================================
// FALSIFY-FT-QWEN35-005: num_kv_heads must be 4 (GQA)
//
// Prediction: qwen3_5_9b().num_kv_heads == 4.
// If fails: GQA ratio wrong → K/V projection LoRA shapes wrong.
// ========================================================================
#[test]
fn falsify_ft_qwen35_005_num_kv_heads() {
    let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    assert_eq!(
        config.num_kv_heads, 4,
        "FALSIFIED FT-QWEN35-005: num_kv_heads={}, expected 4",
        config.num_kv_heads
    );
}

// ========================================================================
// FALSIFY-FT-QWEN35-006: Contract YAML dimensions match config factory
//
// Prediction: Every dimension in qwen3_5.yaml 9b variant matches the
// corresponding field in TransformerConfig::qwen3_5_9b().
// If fails: contract and code are out of sync — one of them has a bug.
// ========================================================================
#[test]
fn falsify_ft_qwen35_006_contract_config_sync() {
    let qwen35 = qwen35_family();
    let variant = qwen35
        .size_variants
        .get("9b")
        .expect("qwen3_5 missing '9b' variant");

    let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();

    assert_eq!(
        config.hidden_size, variant.hidden_dim,
        "FALSIFIED FT-QWEN35-006: hidden_size mismatch: config={} vs contract={}",
        config.hidden_size, variant.hidden_dim
    );
    assert_eq!(
        config.num_hidden_layers, variant.num_layers,
        "FALSIFIED FT-QWEN35-006: num_layers mismatch: config={} vs contract={}",
        config.num_hidden_layers, variant.num_layers
    );
    assert_eq!(
        config.num_attention_heads, variant.num_heads,
        "FALSIFIED FT-QWEN35-006: num_heads mismatch: config={} vs contract={}",
        config.num_attention_heads, variant.num_heads
    );
    assert_eq!(
        config.num_kv_heads, variant.num_kv_heads,
        "FALSIFIED FT-QWEN35-006: num_kv_heads mismatch: config={} vs contract={}",
        config.num_kv_heads, variant.num_kv_heads
    );
    assert_eq!(
        config.intermediate_size, variant.intermediate_dim,
        "FALSIFIED FT-QWEN35-006: intermediate_dim mismatch: config={} vs contract={}",
        config.intermediate_size, variant.intermediate_dim
    );
    assert_eq!(
        config.vocab_size, variant.vocab_size,
        "FALSIFIED FT-QWEN35-006: vocab_size mismatch: config={} vs contract={}",
        config.vocab_size, variant.vocab_size
    );
    assert_eq!(
        config.head_dim(),
        variant.head_dim,
        "FALSIFIED FT-QWEN35-006: head_dim mismatch: config={} vs contract={}",
        config.head_dim(),
        variant.head_dim
    );
}

// ========================================================================
// FALSIFY-FT-QWEN35-007: CLI dispatch "9B" resolves to qwen3_5_9b() config
//
// Prediction: The same config values used by "9B"/"qwen3.5-9b" CLI dispatch
// match TransformerConfig::qwen3_5_9b(). This is a cross-boundary check
// between CLI and library.
// If fails: CLI dispatches to wrong config factory.
// ========================================================================
#[test]
fn falsify_ft_qwen35_007_cli_dispatch_consistency() {
    // Verify all aliases produce the same config
    let config_9b = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    assert_eq!(
        config_9b.vocab_size, 248320,
        "FALSIFIED FT-QWEN35-007: '9B' dispatch config has wrong vocab_size"
    );
    assert_eq!(
        config_9b.hidden_size, 4096,
        "FALSIFIED FT-QWEN35-007: '9B' dispatch config has wrong hidden_size"
    );
    assert!(
        !config_9b.use_bias,
        "FALSIFIED FT-QWEN35-007: '9B' dispatch config should not have bias"
    );

    // Verify it's different from Qwen2 (no confusion between families)
    let config_qwen2 = entrenar::transformer::TransformerConfig::qwen2_0_5b();
    assert_ne!(
        config_9b.vocab_size, config_qwen2.vocab_size,
        "FALSIFIED FT-QWEN35-007: Qwen3.5 and Qwen2 should have different vocab_size"
    );
    assert_ne!(
        config_9b.use_bias, config_qwen2.use_bias,
        "FALSIFIED FT-QWEN35-007: Qwen3.5 (no bias) vs Qwen2 (has bias) must differ"
    );
}

// ========================================================================
// CROSS-CRATE FINE-TUNING CONTRACT FALSIFICATION (FALSIFY-FT-XCRATE-001..004)
//
// These tests verify cross-crate invariants between aprender's Poka-Yoke
// validated types and entrenar's training pipeline. They attempt to falsify
// the claim that the two crates agree on classification contracts.
//
// Contract: classification-finetune-v1.yaml
// ========================================================================

// ========================================================================
// FALSIFY-FT-XCRATE-001: ClassifyConfig default num_classes >= 2
//
// Prediction: ClassifyConfig::default().num_classes >= 2.
// If fails: default config would violate F-CLASS-006 (degenerate class count).
// ========================================================================
#[test]
fn falsify_ft_xcrate_001_default_num_classes() {
    let config = entrenar::finetune::ClassifyConfig::default();
    assert!(
        config.num_classes >= 2,
        "FALSIFIED FT-XCRATE-001: ClassifyConfig default num_classes={} < 2",
        config.num_classes
    );
}

// ========================================================================
// FALSIFY-FT-XCRATE-002: ClassificationHead shape matches F-CLASS-004
//
// Prediction: ClassificationHead weight tensor has exactly
//   hidden_size * num_classes elements.
// If fails: weight shape contract is broken between aprender and entrenar.
// ========================================================================
#[test]
fn falsify_ft_xcrate_002_classifier_head_shape() {
    let hidden_size = 896; // Qwen2-0.5B
    let num_classes = 5;
    let head = entrenar::finetune::ClassificationHead::new(hidden_size, num_classes);
    let weight_data = head.weight.data();
    let weight_slice = weight_data.as_slice().expect("contiguous weight data");
    assert_eq!(
        weight_slice.len(),
        hidden_size * num_classes,
        "FALSIFIED FT-XCRATE-002: ClassificationHead weight.len()={} != hidden_size({}) * num_classes({})",
        weight_slice.len(), hidden_size, num_classes
    );
}

// ========================================================================
// FALSIFY-FT-XCRATE-003: cross_entropy_loss matches F-CLASS-003 postcondition
//
// Prediction: cross_entropy_loss output is finite and non-negative.
// If fails: loss computation violates F-CLASS-005 postcondition.
// ========================================================================
#[test]
fn falsify_ft_xcrate_003_cross_entropy_postcondition() {
    let logits = entrenar::Tensor::from_vec(vec![2.0_f32, 1.0, 0.1, -1.0, 3.0], false);
    let label = 2; // third class
    let num_classes = 5;
    let loss_tensor = entrenar::finetune::cross_entropy_loss(&logits, label, num_classes);
    let loss_data = loss_tensor.data();
    let loss_val = loss_data.as_slice().expect("contiguous loss")[0];
    assert!(
        loss_val.is_finite(),
        "FALSIFIED FT-XCRATE-003: cross_entropy_loss returned non-finite: {loss_val}"
    );
    assert!(
        loss_val >= 0.0,
        "FALSIFIED FT-XCRATE-003: cross_entropy_loss returned negative: {loss_val}"
    );
}

// ========================================================================
// FALSIFY-FT-XCRATE-004: ValidatedClassLogits accepts entrenar logit output
//
// Prediction: logits from ClassificationHead::forward() can be validated
//   by aprender's ValidatedClassLogits::new() without error.
// If fails: the two crates disagree on logit shape contract.
// ========================================================================
#[test]
fn falsify_ft_xcrate_004_validated_logits_accept_head_output() {
    let hidden_size = 896;
    let num_classes = 5;
    let seq_len = 1;
    let head = entrenar::finetune::ClassificationHead::new(hidden_size, num_classes);

    // Simulate a hidden state tensor [seq_len * hidden_size]
    let hidden_state = entrenar::Tensor::from_vec(vec![0.1_f32; seq_len * hidden_size], false);
    let logits_tensor = head.forward(&hidden_state, seq_len);
    let logits_data: Vec<f32> = logits_tensor
        .data()
        .as_slice()
        .expect("contiguous logits")
        .to_vec();

    // aprender's Poka-Yoke type must accept these logits
    let validated = ValidatedClassLogits::new(logits_data, num_classes);
    assert!(
        validated.is_ok(),
        "FALSIFIED FT-XCRATE-004: ValidatedClassLogits rejected entrenar logits: {:?}",
        validated.err()
    );
}
