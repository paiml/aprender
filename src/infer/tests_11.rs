//! T-COV-95 Phase 60: Extended coverage for infer/mod.rs uncovered lines
//!
//! Targets lines not exercised by tests_part_01 through tests_part_10:
//! - prepare_tokens: SafeTensors and Apr format variants (input_tokens path)
//! - prepare_tokens: no-prompt path for SafeTensors and Apr formats
//! - tok_per_sec: negative milliseconds edge case
//! - validate_model_path: ".." in filename (not path traversal)
//! - validate_model_path: case-insensitive extension matching (e.g., .GGUF)
//! - InferenceConfig: trace_verbose and trace_steps fields
//! - safetensors_arch_to_template_hint: mixed-case, partial matches
//! - apr_arch_to_template_hint: edge cases for substring matching
//! - clean_model_output: partial marker substrings
//! - is_legacy_gguf_quant: boundary values around legacy range
//! - prefault_mmap: multi-page data with non-zero bytes
//! - InferenceResult: Debug format contains all field names

use super::*;
use std::path::PathBuf;

// ============================================================================
// prepare_tokens: SafeTensors format with raw input_tokens
// ============================================================================

#[test]
fn test_prepare_tokens_safetensors_with_raw_tokens() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.safetensors").with_input_tokens(vec![10, 20, 30]);
    let prepared = prepare_tokens(&config, &ModelFormat::SafeTensors)
        .expect("should prepare tokens from raw input");
    assert_eq!(prepared.tokens(), &[10, 20, 30]);
    assert_eq!(prepared.input_count(), 3);
}

#[test]
fn test_prepare_tokens_apr_with_raw_tokens() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.apr").with_input_tokens(vec![5, 15, 25, 35]);
    let prepared =
        prepare_tokens(&config, &ModelFormat::Apr).expect("should prepare tokens from raw input");
    assert_eq!(prepared.tokens(), &[5, 15, 25, 35]);
    assert_eq!(prepared.input_count(), 4);
}

#[test]
fn test_prepare_tokens_safetensors_no_prompt() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.safetensors");
    let prepared =
        prepare_tokens(&config, &ModelFormat::SafeTensors).expect("should return BOS token");
    assert_eq!(prepared.tokens(), &[1u32]);
    assert_eq!(prepared.input_count(), 1);
}

#[test]
fn test_prepare_tokens_apr_no_prompt() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.apr");
    let prepared = prepare_tokens(&config, &ModelFormat::Apr).expect("should return BOS token");
    assert_eq!(prepared.tokens(), &[1u32]);
    assert_eq!(prepared.input_count(), 1);
}

#[test]
fn test_prepare_tokens_empty_input_tokens() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![]);
    let prepared =
        prepare_tokens(&config, &ModelFormat::Gguf).expect("should handle empty token list");
    assert_eq!(prepared.tokens(), &[] as &[u32]);
    assert_eq!(prepared.input_count(), 0);
}

#[test]
fn test_prepare_tokens_input_tokens_takes_precedence_over_prompt() {
    use crate::format::ModelFormat;

    // When both prompt and input_tokens are set, input_tokens should win
    let config = InferenceConfig::new("/model.gguf")
        .with_prompt("This should be ignored")
        .with_input_tokens(vec![42, 43, 44]);
    let prepared =
        prepare_tokens(&config, &ModelFormat::Gguf).expect("input_tokens should take precedence");
    assert_eq!(prepared.tokens(), &[42, 43, 44]);
    assert_eq!(prepared.input_count(), 3);
}

// ============================================================================
// tok_per_sec: edge cases
// ============================================================================

#[test]
fn test_tok_per_sec_negative_ms() {
    // Negative ms fails the `ms > 0.0` guard, so returns 0.0
    let tps = tok_per_sec(10, -100.0);
    assert_eq!(tps, 0.0, "negative ms should return 0.0 (guard clause)");
}

#[test]
fn test_tok_per_sec_very_small_ms() {
    let tps = tok_per_sec(1, 0.001);
    // 1 token / 0.000001 seconds = 1_000_000 tok/s
    assert!(
        tps > 100_000.0,
        "very small ms should produce very high tps"
    );
}

#[test]
fn test_tok_per_sec_large_count() {
    let tps = tok_per_sec(1_000_000, 1000.0);
    // 1M tokens / 1 second = 1M tok/s
    assert!((tps - 1_000_000.0).abs() < 1.0);
}

// ============================================================================
// validate_model_path: additional edge cases
// ============================================================================

#[test]
fn test_validate_model_path_double_dot_in_filename_not_extension() {
    // A filename like "model..gguf" contains ".." but in the name, not as traversal
    let tmp = std::env::temp_dir().join("model..gguf");
    // This has ".." in path string, so it should be caught as traversal
    let result = validate_model_path(&tmp);
    // The validator checks for ".." anywhere in the path string
    assert!(
        result.is_err(),
        "'..' in filename should be caught as potential traversal"
    );
}

#[test]
fn test_validate_model_path_uppercase_extension() {
    // Extension check should be case-insensitive
    let tmp = std::env::temp_dir().join("test_validate_upper.GGUF");
    std::fs::write(&tmp, "dummy").expect("write test file");

    let result = validate_model_path(&tmp);
    // Should NOT fail on extension check (may fail on other checks)
    if let Err(e) = &result {
        let err = e.to_string();
        assert!(
            !err.contains("Invalid model file extension"),
            "Uppercase .GGUF should be accepted, got: {}",
            err
        );
    }

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_validate_model_path_mixed_case_extension() {
    let tmp = std::env::temp_dir().join("test_validate_mixed.SafeTensors");
    std::fs::write(&tmp, "dummy").expect("write test file");

    let result = validate_model_path(&tmp);
    if let Err(e) = &result {
        let err = e.to_string();
        assert!(
            !err.contains("Invalid model file extension"),
            "Mixed-case .SafeTensors should be accepted, got: {}",
            err
        );
    }

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_validate_model_path_json_extension_allowed() {
    // JSON is in the valid extensions list (for sharded SafeTensors index.json)
    let tmp = std::env::temp_dir().join(format!("test_validate_{}.json", std::process::id()));
    std::fs::write(&tmp, "{}").expect("write test file");

    let result = validate_model_path(&tmp);
    assert!(
        result.is_ok(),
        "JSON extension should be valid: {:?}",
        result
    );

    std::fs::remove_file(&tmp).ok();
}

// ============================================================================
// InferenceConfig: trace_verbose and trace_steps fields
// ============================================================================

#[test]
fn test_inference_config_trace_verbose_default_false() {
    let config = InferenceConfig::new("model.gguf");
    assert!(
        !config.trace_verbose,
        "trace_verbose should default to false"
    );
}

#[test]
fn test_inference_config_trace_steps_default_none() {
    let config = InferenceConfig::new("model.gguf");
    assert!(
        config.trace_steps.is_none(),
        "trace_steps should default to None"
    );
}

#[test]
fn test_inference_config_trace_verbose_can_be_set() {
    let mut config = InferenceConfig::new("model.gguf");
    config.trace_verbose = true;
    assert!(config.trace_verbose);
}

#[test]
fn test_inference_config_trace_steps_can_be_set() {
    let mut config = InferenceConfig::new("model.gguf");
    config.trace_steps = Some(vec!["Tokenize".to_string(), "Embed".to_string()]);
    assert_eq!(
        config.trace_steps,
        Some(vec!["Tokenize".to_string(), "Embed".to_string()])
    );
}

// ============================================================================
// safetensors_arch_to_template_hint: additional edge cases
// ============================================================================

// GH-317/318: normalize_architecture does exact matching from contract.
// Mixed-case or non-contract names all default to "llama".
#[test]
fn test_safetensors_arch_exact_match_qwen() {
    // Exact HF class name → matches
    assert_eq!(
        safetensors_arch_to_template_hint("Qwen2ForCausalLM", "model"),
        "qwen2"
    );
    // Wrong case → not in contract → defaults to "llama"
    assert_eq!(
        safetensors_arch_to_template_hint("QWEN2ForCausalLM", "model"),
        "llama"
    );
}

#[test]
fn test_safetensors_arch_exact_match_llama() {
    // Exact matches
    assert_eq!(
        safetensors_arch_to_template_hint("LlamaForCausalLM", "model"),
        "llama"
    );
    // Non-contract variant → "llama" (still llama, but by default path)
    assert_eq!(
        safetensors_arch_to_template_hint("LLAMA_Model", "model"),
        "llama"
    );
}

#[test]
fn test_safetensors_arch_exact_match_mistral() {
    assert_eq!(
        safetensors_arch_to_template_hint("MistralForCausalLM", "model"),
        "mistral"
    );
    // Non-contract variant → "llama" default
    assert_eq!(
        safetensors_arch_to_template_hint("MISTRAL_v2", "model"),
        "llama"
    );
}

#[test]
fn test_safetensors_arch_exact_match_phi() {
    // Phi-3+ → "phi"
    assert_eq!(safetensors_arch_to_template_hint("Phi3ForCausalLM", "model"), "phi");
    // Phi-1.5/Phi-2 → "phi2"
    assert_eq!(safetensors_arch_to_template_hint("PhiForCausalLM", "model"), "phi2");
    // Non-contract variant → "llama" default
    assert_eq!(safetensors_arch_to_template_hint("PHI_3", "model"), "llama");
}

#[test]
fn test_safetensors_arch_empty_string() {
    // GH-317/318: Empty architecture defaults to "llama" via normalize_architecture
    let result = safetensors_arch_to_template_hint("", "fallback-model");
    assert_eq!(result, "llama");
}

#[test]
fn test_safetensors_arch_non_contract_names_default_to_llama() {
    // GH-317/318: Non-contract names all default to "llama"
    assert_eq!(
        safetensors_arch_to_template_hint("QwenLlamaHybrid", "model"),
        "llama"
    );
}

// ============================================================================
// apr_arch_to_template_hint: additional edge cases
// ============================================================================

#[test]
fn test_apr_arch_empty_string_defaults_to_llama() {
    // GH-318: No fallback — contract always returns canonical key
    assert_eq!(apr_arch_to_template_hint("", "my-fallback"), "llama");
}

#[test]
fn test_apr_arch_unknown_substring_defaults_to_llama() {
    // GH-318: Non-contract strings → "llama" default (no fallback to model_name)
    assert_eq!(apr_arch_to_template_hint("superqwen", "model"), "llama");
}

#[test]
fn test_apr_arch_codellama_defaults_to_llama() {
    // GH-318: "codellama" not in contract → "llama"
    assert_eq!(apr_arch_to_template_hint("codellama", "model"), "llama");
}

#[test]
fn test_apr_arch_mixtral_defaults_to_llama() {
    // GH-318: "mixtral-mistral-v2" not in contract → "llama"
    assert_eq!(
        apr_arch_to_template_hint("mixtral-mistral-v2", "model"),
        "llama"
    );
}

#[test]
fn test_apr_arch_microsoft_phi_defaults_to_llama() {
    // GH-318: "microsoft-phi2" not in contract → "llama"
    assert_eq!(apr_arch_to_template_hint("microsoft-phi2", "model"), "llama");
}

#[test]
fn test_apr_arch_no_match_defaults_to_llama() {
    // GH-318: Unknown architectures always default to "llama"
    assert_eq!(apr_arch_to_template_hint("gpt-neo", "some-filename.gguf"), "llama");
}

// ============================================================================
// is_legacy_gguf_quant: boundary values
// ============================================================================

#[test]
fn test_is_legacy_quant_boundary_below() {
    assert!(!is_legacy_gguf_quant(1)); // F16, not legacy
}

#[test]
fn test_is_legacy_quant_boundary_between_4_and_5() {
    // 4 and 5 are NOT in the legacy set
    assert!(!is_legacy_gguf_quant(4));
    assert!(!is_legacy_gguf_quant(5));
}

#[test]
fn test_is_legacy_quant_boundary_above() {
    assert!(!is_legacy_gguf_quant(8)); // Q8_0, not legacy
}

#[test]
fn test_is_legacy_quant_u32_max() {
    assert!(!is_legacy_gguf_quant(u32::MAX));
}

// ============================================================================
// prefault_mmap: non-zero data
// ============================================================================

#[test]
fn test_prefault_mmap_nonzero_data() {
    let data: Vec<u8> = (0..8192u32).map(|i| (i % 256) as u8).collect();
    // Should not panic with any data pattern
    prefault_mmap(&data);
}

#[test]
fn test_prefault_mmap_all_ones() {
    let data = vec![0xFFu8; 4096 * 2 + 1];
    prefault_mmap(&data);
}

#[test]
fn test_prefault_mmap_single_byte() {
    prefault_mmap(&[42u8]);
}

// ============================================================================
// InferenceResult: Debug output completeness
// ============================================================================

#[test]
fn test_inference_result_debug_contains_all_fields() {
    let result = InferenceResult {
        text: "generated_text_here".to_string(),
        tokens: vec![1, 2, 3],
        input_token_count: 1,
        generated_token_count: 2,
        inference_ms: 123.456,
        tok_per_sec: 789.012,
        load_ms: 45.678,
        format: "TestFormat".to_string(),
        used_gpu: true,
    };
    let debug = format!("{:?}", result);
    assert!(
        debug.contains("generated_text_here"),
        "Debug should contain text field"
    );
    assert!(
        debug.contains("input_token_count"),
        "Debug should contain input_token_count"
    );
    assert!(
        debug.contains("generated_token_count"),
        "Debug should contain generated_token_count"
    );
    assert!(
        debug.contains("inference_ms"),
        "Debug should contain inference_ms"
    );
    assert!(
        debug.contains("tok_per_sec"),
        "Debug should contain tok_per_sec"
    );
    assert!(debug.contains("load_ms"), "Debug should contain load_ms");
    assert!(
        debug.contains("TestFormat"),
        "Debug should contain format value"
    );
    assert!(debug.contains("used_gpu"), "Debug should contain used_gpu");
}

#[test]
fn test_inference_config_debug_contains_all_fields() {
    let config = InferenceConfig::new("debug_test.gguf")
        .with_prompt("debug prompt")
        .with_max_tokens(99)
        .with_temperature(0.42)
        .with_top_k(50)
        .without_gpu()
        .with_verbose(true)
        .with_trace(true)
        .with_trace_output("trace.json")
        .with_mock_backend();

    let debug = format!("{:?}", config);
    assert!(
        debug.contains("debug_test.gguf"),
        "Debug should contain model_path"
    );
    assert!(
        debug.contains("debug prompt"),
        "Debug should contain prompt"
    );
    assert!(
        debug.contains("99"),
        "Debug should contain max_tokens value"
    );
    assert!(
        debug.contains("0.42"),
        "Debug should contain temperature value"
    );
    assert!(debug.contains("50"), "Debug should contain top_k value");
    assert!(
        debug.contains("true"),
        "Debug should contain boolean values"
    );
    assert!(
        debug.contains("trace.json"),
        "Debug should contain trace_output"
    );
}

include!("tests_clean_model_02.rs");
