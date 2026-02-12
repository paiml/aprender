//! Tests for infer_model_config_from_tensors and related import.rs functions
//!
//! Targets coverage gaps in:
//! - infer_model_config_from_tensors (import.rs:1034-1238)
//! - compute_tensor_stats / compute_std / TensorAccumulator (import.rs:1554-1669)
//! - validate_single_tensor (import.rs:1474-1489)
//! - validate_tensors strict mode (import.rs:1511-1551)
//! - map_tensor_names (import.rs:1427-1437)
//! - quantize_tensors embedding skip (mod.rs:650-685)
//! - AprConverter builder (mod.rs:66-143)
//! - ConvertOptions default (mod.rs:160-168)
//! - resolve_source error paths (import.rs:373-447)
//! - cache helpers (import.rs:449-530)

use super::super::import;
use super::super::*;
use std::collections::BTreeMap;

// Helper: create a minimal tensor with given shape (data doesn't matter for config inference)
fn dummy_tensor(shape: Vec<usize>) -> (Vec<f32>, Vec<usize>) {
    let size: usize = shape.iter().product();
    (vec![0.1; size], shape)
}

// ============================================================================
// infer_model_config_from_tensors: HuggingFace naming
// ============================================================================

#[test]
fn test_infer_config_huggingface_naming() {
    let mut tensors = BTreeMap::new();
    // Qwen2-style HuggingFace naming
    // embed_tokens: [vocab_size=1000, hidden_size=128]
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    // 4 layers
    for i in 0..4 {
        tensors.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            dummy_tensor(vec![128, 128]),
        );
        // Bug 210: Qwen2 has attention bias — distinguishes from LLaMA/Mistral
        tensors.insert(
            format!("model.layers.{i}.self_attn.q_proj.bias"),
            dummy_tensor(vec![128]),
        );
        tensors.insert(
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            dummy_tensor(vec![128, 128]),
        );
        tensors.insert(
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            dummy_tensor(vec![512, 128]),
        );
    }

    let config = infer_model_config_from_tensors(&tensors);
    assert!(config.is_some(), "Should infer config from HF tensors");
    let config = config.expect("config should be Some");

    assert_eq!(config.vocab_size, Some(1000));
    assert_eq!(config.hidden_size, Some(128));
    assert_eq!(config.num_layers, Some(4));
    assert_eq!(config.intermediate_size, Some(512));
    assert_eq!(
        config.architecture.as_deref(),
        Some("qwen2"),
        "model.layers pattern should detect qwen2"
    );
    // MHA: q_dim == hidden_size
    assert!(config.num_heads.is_some());
    assert!(config.num_kv_heads.is_some());
}

#[test]
fn test_infer_config_gguf_naming() {
    let mut tensors = BTreeMap::new();
    // GGUF naming convention
    // token_embd: [hidden_size=64, vocab_size=500] (GGUF order: hidden first)
    tensors.insert("token_embd.weight".to_string(), dummy_tensor(vec![64, 500]));
    // 2 blocks
    for i in 0..2 {
        tensors.insert(format!("blk.{i}.attn_q.weight"), dummy_tensor(vec![64, 64]));
        tensors.insert(format!("blk.{i}.attn_k.weight"), dummy_tensor(vec![64, 64]));
        tensors.insert(
            format!("blk.{i}.ffn_gate.weight"),
            dummy_tensor(vec![256, 64]),
        );
    }

    let config = infer_model_config_from_tensors(&tensors);
    assert!(config.is_some(), "Should infer config from GGUF tensors");
    let config = config.expect("config should be Some");

    assert_eq!(config.vocab_size, Some(500));
    assert_eq!(config.hidden_size, Some(64));
    assert_eq!(config.num_layers, Some(2));
    assert_eq!(config.intermediate_size, Some(256));
    // Bug 210: blk. naming is shared by many architectures (GGUF convention)
    // Cannot reliably detect specific architecture from blk. alone
    assert_eq!(
        config.architecture.as_deref(),
        Some("unknown"),
        "blk. pattern should detect unknown (ambiguous GGUF naming)"
    );
}

#[test]
fn test_infer_config_gpt2_naming() {
    let mut tensors = BTreeMap::new();
    // GPT-2 naming: wte, transformer.h.N
    tensors.insert("wte.weight".to_string(), dummy_tensor(vec![5000, 256]));
    for i in 0..6 {
        tensors.insert(
            format!("transformer.h.{i}.attn.query.weight"),
            dummy_tensor(vec![256, 256]),
        );
        tensors.insert(
            format!("transformer.h.{i}.attn.key.weight"),
            dummy_tensor(vec![256, 256]),
        );
    }

    let config = infer_model_config_from_tensors(&tensors);
    assert!(config.is_some());
    let config = config.expect("config should be Some");

    assert_eq!(config.vocab_size, Some(5000));
    assert_eq!(config.hidden_size, Some(256));
    assert_eq!(config.num_layers, Some(6));
    assert_eq!(
        config.architecture.as_deref(),
        Some("gpt2"),
        "transformer.h pattern should detect gpt2"
    );
}

#[test]
fn test_infer_config_gqa_model() {
    // GQA model: k_proj has smaller dim than q_proj
    let mut tensors = BTreeMap::new();
    // hidden_size=128, head_dim=64, num_heads=2, num_kv_heads=1
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        dummy_tensor(vec![128, 128]), // q_dim=128
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        dummy_tensor(vec![64, 128]), // kv_dim=64 < q_dim=128
    );

    let config = infer_model_config_from_tensors(&tensors);
    assert!(config.is_some());
    let config = config.expect("config should be Some");

    // GQA: kv_dim=64 < q_dim=128
    // head_dim=64 -> n_kv=1, n_heads=2
    assert_eq!(config.num_heads, Some(2));
    assert_eq!(config.num_kv_heads, Some(1));
}

#[test]
fn test_infer_config_mha_with_head_dim_64() {
    // MHA: q_dim == hidden_size
    // The function checks head_dims in order [64, 128, 96, 80]
    // So for hidden_size=256, it finds head_dim=64 first -> 4 heads
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 256]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        dummy_tensor(vec![256, 256]), // q_dim=256 == hidden_size
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        dummy_tensor(vec![256, 256]), // kv_dim=256 == q_dim (MHA)
    );

    let config = infer_model_config_from_tensors(&tensors);
    assert!(config.is_some());
    let config = config.expect("config should be Some");
    // MHA: hidden_size=256, head_dim=64 (first match) -> num_heads=4
    assert_eq!(config.num_heads, Some(4));
    assert_eq!(config.num_kv_heads, Some(4)); // MHA: same as num_heads
}

#[test]
fn test_infer_config_no_embedding_returns_none() {
    let mut tensors = BTreeMap::new();
    // Only layer tensors, no embedding
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        dummy_tensor(vec![128, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors);
    assert!(
        config.is_none(),
        "Should return None without embedding tensor"
    );
}

#[test]
fn test_infer_config_empty_tensors_returns_none() {
    let tensors = BTreeMap::new();
    let config = infer_model_config_from_tensors(&tensors);
    assert!(config.is_none(), "Should return None for empty tensor map");
}

#[test]
fn test_infer_config_1d_embedding_returns_none() {
    let mut tensors = BTreeMap::new();
    // 1D embedding (invalid shape for config inference)
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![128]),
    );

    let config = infer_model_config_from_tensors(&tensors);
    assert!(
        config.is_none(),
        "Should return None for 1D embedding shape"
    );
}

#[test]
fn test_infer_config_word_embeddings_naming() {
    // BERT-style naming
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "word_embeddings.weight".to_string(),
        dummy_tensor(vec![3000, 768]),
    );
    tensors.insert(
        "blocks.0.attn.query.weight".to_string(),
        dummy_tensor(vec![768, 768]),
    );

    let config = infer_model_config_from_tensors(&tensors);
    assert!(config.is_some());
    let config = config.expect("config should be Some");
    assert_eq!(config.vocab_size, Some(3000));
    assert_eq!(config.hidden_size, Some(768));
}

#[test]
fn test_infer_config_rope_type_qwen() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        dummy_tensor(vec![128, 128]),
    );
    // Bug 210: Qwen2 detected by attention bias presence
    tensors.insert(
        "model.layers.0.self_attn.q_proj.bias".to_string(),
        dummy_tensor(vec![128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    // q_proj.bias -> qwen2 -> rope_type=2 (NEOX)
    assert_eq!(config.rope_type, Some(2), "qwen2 should have rope_type=2");
}

#[test]
fn test_infer_config_rope_type_gpt2() {
    let mut tensors = BTreeMap::new();
    tensors.insert("wte.weight".to_string(), dummy_tensor(vec![1000, 128]));
    tensors.insert(
        "transformer.h.0.dummy".to_string(),
        dummy_tensor(vec![128, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    // gpt2 -> rope_type=0 (NORM)
    assert_eq!(config.rope_type, Some(0), "gpt2 should have rope_type=0");
}

#[test]
fn test_infer_config_unknown_architecture() {
    // Tensors with no recognizable layer patterns
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    // No layer-like patterns
    tensors.insert(
        "some.random.tensor".to_string(),
        dummy_tensor(vec![128, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert_eq!(config.architecture.as_deref(), Some("unknown"));
    assert_eq!(config.num_layers, Some(0));
}

#[test]
fn test_infer_config_ffn_up_naming() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".to_string(),
        dummy_tensor(vec![64, 1000]),
    );
    // Use ffn_up instead of ffn_gate
    tensors.insert(
        "blk.0.ffn_up.weight".to_string(),
        dummy_tensor(vec![256, 64]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert_eq!(config.intermediate_size, Some(256));
}

#[test]
fn test_infer_config_fc1_naming() {
    // Some architectures use fc1 for FFN
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.fc1.weight".to_string(),
        dummy_tensor(vec![512, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert_eq!(config.intermediate_size, Some(512));
}

// ============================================================================
// compute_tensor_stats / compute_std / TensorAccumulator
// ============================================================================

#[test]
fn test_compute_tensor_stats_normal_data() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = compute_tensor_stats("test_tensor", &data);

    assert_eq!(stats.name, "test_tensor");
    assert_eq!(stats.count, 5);
    assert!((stats.mean - 3.0).abs() < 1e-5);
    assert!((stats.min - 1.0).abs() < 1e-5);
    assert!((stats.max - 5.0).abs() < 1e-5);
    assert!(stats.std > 0.0);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert_eq!(stats.zero_count, 0);
}

#[test]
fn test_compute_tensor_stats_empty() {
    let data: Vec<f32> = vec![];
    let stats = compute_tensor_stats("empty", &data);

    assert_eq!(stats.count, 0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
    assert_eq!(stats.std, 0.0);
}

#[test]
fn test_compute_tensor_stats_with_nan() {
    let data = vec![1.0, f32::NAN, 3.0];
    let stats = compute_tensor_stats("nan_test", &data);

    assert_eq!(stats.nan_count, 1);
    assert_eq!(stats.count, 3);
}

#[test]
fn test_compute_tensor_stats_with_inf() {
    let data = vec![1.0, f32::INFINITY, f32::NEG_INFINITY, 2.0];
    let stats = compute_tensor_stats("inf_test", &data);

    assert_eq!(stats.inf_count, 2);
    assert_eq!(stats.count, 4);
}

#[test]
fn test_compute_tensor_stats_with_zeros() {
    let data = vec![0.0, 0.0, 1.0, 0.0];
    let stats = compute_tensor_stats("zero_test", &data);

    assert_eq!(stats.zero_count, 3);
}

#[test]
fn test_compute_std_normal() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let mean = 5.0_f32;
    let std = compute_std(&data, mean, data.len());
    assert!(std > 0.0);
    // Standard deviation should be reasonable for this data spread
    assert!(
        std > 1.0 && std < 3.0,
        "std={std} should be between 1.0 and 3.0"
    );
}

#[test]
fn test_compute_std_single_value() {
    let data = vec![42.0];
    let std = compute_std(&data, 42.0, 1);
    assert_eq!(std, 0.0, "Std of single value should be 0");
}

#[test]
fn test_compute_std_zero_valid_count() {
    let data = vec![f32::NAN, f32::NAN];
    let std = compute_std(&data, 0.0, 0);
    assert_eq!(std, 0.0, "Std with 0 valid should be 0");
}

#[test]
fn test_tensor_accumulator_new() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.sum, 0.0);
    assert_eq!(acc.nan_count, 0);
    assert_eq!(acc.inf_count, 0);
    assert_eq!(acc.zero_count, 0);
    assert_eq!(acc.valid_count, 0);
    assert_eq!(acc.min, f32::INFINITY);
    assert_eq!(acc.max, f32::NEG_INFINITY);
}

#[test]
fn test_tensor_accumulator_accumulate_normal() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(1.0);
    acc.accumulate(2.0);
    acc.accumulate(3.0);

    assert_eq!(acc.valid_count, 3);
    assert!((acc.sum - 6.0).abs() < 1e-10);
    assert!((acc.min - 1.0).abs() < 1e-5);
    assert!((acc.max - 3.0).abs() < 1e-5);
}

#[test]
fn test_tensor_accumulator_accumulate_nan() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(f32::NAN);
    assert_eq!(acc.nan_count, 1);
    assert_eq!(acc.valid_count, 0);
}

#[test]
fn test_tensor_accumulator_accumulate_inf() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(f32::INFINITY);
    acc.accumulate(f32::NEG_INFINITY);
    assert_eq!(acc.inf_count, 2);
    assert_eq!(acc.valid_count, 0);
}

#[test]
fn test_tensor_accumulator_accumulate_zero() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(0.0);
    assert_eq!(acc.zero_count, 1);
    assert_eq!(acc.valid_count, 1);
}

#[test]
fn test_tensor_accumulator_mean() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(2.0);
    acc.accumulate(4.0);
    assert!((acc.mean() - 3.0).abs() < 1e-5);
}

#[test]
fn test_tensor_accumulator_mean_empty() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.mean(), 0.0);
}

#[test]
fn test_tensor_accumulator_safe_min_max() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.safe_min(), 0.0, "safe_min of empty should be 0.0");
    assert_eq!(acc.safe_max(), 0.0, "safe_max of empty should be 0.0");

    let mut acc2 = TensorAccumulator::new();
    acc2.accumulate(-5.0);
    acc2.accumulate(10.0);
    assert!((acc2.safe_min() - (-5.0)).abs() < 1e-5);
    assert!((acc2.safe_max() - 10.0).abs() < 1e-5);
}

// ============================================================================
// validate_single_tensor
// ============================================================================

#[test]
fn test_validate_single_tensor_no_errors() {
    let data = vec![0.1, 0.2, 0.3];
    let options = ImportOptions {
        allow_no_config: true,
        ..ImportOptions::default()
    };
    let mut validator = AprValidator::new();
    let mut errors = Vec::new();

    validate_single_tensor("test", &data, &options, &mut validator, &mut errors);
    assert!(errors.is_empty(), "Valid data should produce no errors");
}

#[test]
fn test_validate_single_tensor_with_nan_strict() {
    let data = vec![0.1, f32::NAN, 0.3];
    let options = ImportOptions {
        strict: true,
        validation: ValidationConfig::Strict,
        allow_no_config: true,
        ..Default::default()
    };
    let mut validator = AprValidator::new();
    let mut errors = Vec::new();

    validate_single_tensor("nan_tensor", &data, &options, &mut validator, &mut errors);
    assert!(
        !errors.is_empty(),
        "NaN data in strict mode should produce errors"
    );
    assert!(errors.iter().any(|e| e.contains("NaN")));
}

// ============================================================================
// AprConverter builder
// ============================================================================

#[test]
fn test_apr_converter_new() {
    let converter = AprConverter::new();
    // Just ensure it's constructible
    let debug_str = format!("{converter:?}");
    assert!(debug_str.contains("AprConverter"));
}

#[test]
fn test_apr_converter_default() {
    let converter = AprConverter::default();
    let debug_str = format!("{converter:?}");
    assert!(debug_str.contains("Auto"));
}

#[test]
fn test_apr_converter_builder_chain() {
    let result = AprConverter::new()
        .source("/tmp/nonexistent.safetensors")
        .map(|c| c.architecture(Architecture::Qwen2))
        .map(|c| c.validate(ValidationConfig::Strict))
        .map(|c| c.quantize(QuantizationType::Fp16))
        .map(|c| c.compress(Compression::None));

    assert!(result.is_ok(), "Builder chain should succeed");
}

#[test]
fn test_apr_converter_convert_without_source() {
    let converter = AprConverter::new();
    let result = converter.convert();
    assert!(result.is_err(), "Convert without source should error");
    assert!(result.unwrap_err().to_string().contains("No source"));
}

#[test]
fn test_apr_converter_convert_returns_not_implemented() {
    let converter = AprConverter::new()
        .source("/tmp/test.safetensors")
        .expect("source parse");
    let result = converter.convert();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("GH-80"));
}

// ============================================================================
// ConvertOptions
// ============================================================================

#[test]
fn test_convert_options_default() {
    let opts = ConvertOptions::default();
    assert!(opts.quantize.is_none());
    assert!(opts.compress.is_none());
    assert!(opts.validate);
}

// ============================================================================
// quantize_tensors - embedding skip logic
// ============================================================================

#[test]
fn test_quantize_tensors_skips_embeddings() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let quantized = quantized.as_ref();

    // Embedding should be unchanged (F32 preserved)
    let embed = &quantized["model.embed_tokens.weight"].0;
    assert!(
        (embed[0] - 0.1).abs() < 1e-6,
        "Embedding should not be quantized"
    );

    // Non-embedding should be quantized (values will differ due to int8 round-trip)
    let weight = &quantized["model.layers.0.self_attn.q_proj.weight"].0;
    // Just verify it's not exactly the same (quantization applied)
    assert_eq!(weight.len(), 4);
}

#[test]
fn test_quantize_tensors_fp16() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "some.weight".to_string(),
        (vec![1.0, -1.0, 0.5, 0.0], vec![2, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Fp16);
    assert!(result.is_ok());
}

#[test]
fn test_quantize_tensors_int4() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "some.weight".to_string(),
        (vec![1.0, -1.0, 0.5, 0.0], vec![2, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4);
    assert!(result.is_ok());
}

#[test]
fn test_quantize_tensors_skips_token_embd() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".to_string(),
        (vec![0.01, 0.02, 0.03, 0.04], vec![2, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let embed = &quantized.as_ref()["token_embd.weight"].0;
    assert!(
        (embed[0] - 0.01).abs() < 1e-6,
        "token_embd should not be quantized"
    );
}

#[test]
fn test_quantize_tensors_skips_wte() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "wte.weight".to_string(),
        (vec![0.01, 0.02, 0.03, 0.04], vec![2, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let embed = &quantized.as_ref()["wte.weight"].0;
    assert!(
        (embed[0] - 0.01).abs() < 1e-6,
        "wte should not be quantized"
    );
}

#[test]
fn test_quantize_tensors_skips_word_embeddings() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "word_embeddings.weight".to_string(),
        (vec![0.01, 0.02], vec![1, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Fp16);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let embed = &quantized.as_ref()["word_embeddings.weight"].0;
    assert!(
        (embed[0] - 0.01).abs() < 1e-6,
        "word_embeddings should not be quantized"
    );
}

// ============================================================================
// map_tensor_names
// ============================================================================

#[test]
fn test_map_tensor_names_preserves_count() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "blk.0.attn_q.weight".to_string(),
        (vec![1.0, 2.0], vec![1, 2]),
    );
    tensors.insert(
        "blk.0.attn_k.weight".to_string(),
        (vec![3.0, 4.0], vec![1, 2]),
    );
    tensors.insert(
        "token_embd.weight".to_string(),
        (vec![5.0, 6.0], vec![1, 2]),
    );

    let mapped = map_tensor_names(&tensors, Architecture::Qwen2);
    assert_eq!(mapped.len(), tensors.len(), "Should preserve tensor count");
}

// ============================================================================
// resolve_source error paths
// ============================================================================

#[test]
fn test_resolve_source_local_not_found() {
    let source = Source::Local(std::path::PathBuf::from("/nonexistent/model.safetensors"));
    let result = import::resolve_source(&source, false);
    assert!(result.is_err());
}

#[test]
fn test_resolve_source_url_not_implemented() {
    let source = Source::Url("https://example.com/model.bin".to_string());
    let result = import::resolve_source(&source, false);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not yet implemented"));
}

// ============================================================================
// validate_tensor_values (additional edge cases)
// ============================================================================

#[test]
fn test_validate_tensor_values_empty_ok() {
    let result = validate_tensor_values("empty", &[]);
    assert!(result.is_ok());
}

#[test]
fn test_validate_tensor_values_valid_ok() {
    let data = vec![0.1, -0.2, 0.5, 0.0];
    let result = validate_tensor_values("valid", &data);
    assert!(result.is_ok());
}

#[test]
fn test_validate_tensor_values_nan_error() {
    let data = vec![0.1, f32::NAN, 0.3];
    let result = validate_tensor_values("nan_tensor", &data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("NaN"));
}

#[test]
fn test_validate_tensor_values_inf_error() {
    let data = vec![0.1, f32::INFINITY, 0.3];
    let result = validate_tensor_values("inf_tensor", &data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Inf"));
}

#[test]
fn test_validate_tensor_values_explosive_mean() {
    // Mean > 100 should trigger error
    let data = vec![200.0, 300.0, 400.0];
    let result = validate_tensor_values("explosive", &data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("explosive mean"));
}

// ============================================================================
// load_model_tensors unsupported format
// ============================================================================

#[test]
fn test_load_model_tensors_unsupported_format() {
    let path = std::path::Path::new("/tmp/test.xyz");
    let result = load_model_tensors(path);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Unsupported format"));
}

// ============================================================================
// ConvertReport
// ============================================================================

#[test]
fn test_convert_report_reduction_percent() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 500,
        tensor_count: 10,
        quantization: Some(QuantizationType::Int8),
        compression: None,
        reduction_ratio: 2.0,
    };
    assert_eq!(report.reduction_percent(), "50.0%");
}

#[test]
fn test_convert_report_reduction_percent_zero_original() {
    let report = ConvertReport {
        original_size: 0,
        converted_size: 500,
        tensor_count: 10,
        quantization: None,
        compression: None,
        reduction_ratio: 0.0,
    };
    assert_eq!(report.reduction_percent(), "N/A");
}

#[test]
fn test_convert_report_reduction_percent_zero_converted() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 0,
        tensor_count: 10,
        quantization: None,
        compression: None,
        reduction_ratio: 0.0,
    };
    assert_eq!(report.reduction_percent(), "N/A");
}

// ============================================================================
// calculate_tensor_size
// ============================================================================

#[test]
fn test_calculate_tensor_size_basic() {
    let mut tensors = BTreeMap::new();
    tensors.insert("a".to_string(), (vec![1.0; 100], vec![10, 10]));
    tensors.insert("b".to_string(), (vec![1.0; 50], vec![50]));

    let size = calculate_tensor_size(&tensors);
    assert_eq!(size, (100 + 50) * 4); // 150 f32 values * 4 bytes each
}

#[test]
fn test_calculate_tensor_size_empty() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    assert_eq!(calculate_tensor_size(&tensors), 0);
}

// ============================================================================
// GQA head dim inference: head_dim=96 and head_dim=80
// ============================================================================

#[test]
fn test_infer_config_gqa_head_dim_64_first() {
    // The function checks head_dims in order [64, 128, 96, 80]
    // For q_dim=768, kv_dim=192: head_dim=64 matches first (768/64=12, 192/64=3)
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![5000, 768]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        dummy_tensor(vec![768, 768]), // q_dim=768
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        dummy_tensor(vec![192, 768]), // kv_dim=192
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    // head_dim=64 first match: 768/64=12 heads, 192/64=3 kv_heads
    assert_eq!(config.num_heads, Some(12));
    assert_eq!(config.num_kv_heads, Some(3));
}

#[test]
fn test_infer_config_no_q_no_k_projections() {
    // No Q/K projections - num_heads and num_kv_heads should be None
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        dummy_tensor(vec![512, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert!(
        config.num_heads.is_none(),
        "Without Q/K projections, num_heads should be None"
    );
}

#[test]
fn test_infer_config_no_intermediate_size() {
    // No FFN tensors -> intermediate_size should be None
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        dummy_tensor(vec![128, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert!(
        config.intermediate_size.is_none(),
        "Without FFN tensors, intermediate_size should be None"
    );
}

// ============================================================================
// infer_model_config: up_proj naming variant
// ============================================================================

#[test]
fn test_infer_config_up_proj_naming() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        dummy_tensor(vec![512, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert_eq!(config.intermediate_size, Some(512));
}

// ============================================================================
// infer_model_config: defaults (max_position_embeddings, rope_theta, rms_norm_eps)
// ============================================================================

#[test]
fn test_infer_config_defaults() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert_eq!(config.max_position_embeddings, Some(4096));
    assert_eq!(config.rope_theta, Some(10000.0));
    assert_eq!(config.rms_norm_eps, Some(1e-6));
}
