//! T-COV-95 Phase 50: Deep coverage for infer/mod.rs pure functions
//!
//! Covers:
//! - qtype_to_dtype_str: all quantization type mappings
//! - InferenceConfig builder chain completeness
//! - InferenceResult struct construction
//! - validate_model_path: path traversal, extension, existence checks
//! - tok_per_sec: timing calculation
//! - prefault_mmap: page touch function
//! - is_legacy_gguf_quant: quantization type detection
//! - safetensors_arch_to_template_hint: architecture mapping
//! - apr_arch_to_template_hint: architecture mapping
//! - PreparedTokens: public API
//! - clean_model_output: additional edge cases

use super::*;
use std::path::PathBuf;

// ============================================================================
// qtype_to_dtype_str
// ============================================================================

#[test]
fn test_qtype_to_dtype_str_all_known() {
    assert_eq!(qtype_to_dtype_str(0), "F32");
    assert_eq!(qtype_to_dtype_str(1), "F16");
    assert_eq!(qtype_to_dtype_str(2), "Q4_0");
    assert_eq!(qtype_to_dtype_str(3), "Q4_1");
    assert_eq!(qtype_to_dtype_str(6), "Q5_0");
    assert_eq!(qtype_to_dtype_str(7), "Q5_1");
    assert_eq!(qtype_to_dtype_str(8), "Q8_0");
    assert_eq!(qtype_to_dtype_str(9), "Q8_1");
    assert_eq!(qtype_to_dtype_str(10), "Q2_K");
    assert_eq!(qtype_to_dtype_str(11), "Q3_K");
    assert_eq!(qtype_to_dtype_str(12), "Q4_K");
    assert_eq!(qtype_to_dtype_str(13), "Q5_K");
    assert_eq!(qtype_to_dtype_str(14), "Q6_K");
    assert_eq!(qtype_to_dtype_str(16), "IQ2_XXS");
    assert_eq!(qtype_to_dtype_str(17), "IQ2_XS");
    assert_eq!(qtype_to_dtype_str(30), "BF16");
}

#[test]
fn test_qtype_to_dtype_str_unknown() {
    assert_eq!(qtype_to_dtype_str(4), "Unknown");
    assert_eq!(qtype_to_dtype_str(5), "Unknown");
    assert_eq!(qtype_to_dtype_str(15), "Unknown");
    assert_eq!(qtype_to_dtype_str(18), "Unknown");
    assert_eq!(qtype_to_dtype_str(100), "Unknown");
    assert_eq!(qtype_to_dtype_str(u32::MAX), "Unknown");
}

// ============================================================================
// InferenceConfig builder
// ============================================================================

#[test]
fn test_config_with_trace_output() {
    let config = InferenceConfig::new("/model.gguf").with_trace_output("/tmp/trace.json");
    assert_eq!(config.trace_output, Some(PathBuf::from("/tmp/trace.json")));
}

#[test]
fn test_config_full_chain() {
    let config = InferenceConfig::new("/model.gguf")
        .with_prompt("Hello")
        .with_max_tokens(64)
        .with_temperature(0.7)
        .with_top_k(40)
        .without_gpu()
        .with_verbose(true)
        .with_trace(true)
        .with_trace_output("/tmp/trace.json");

    assert_eq!(config.model_path, PathBuf::from("/model.gguf"));
    assert_eq!(config.prompt, Some("Hello".to_string()));
    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 40);
    assert!(config.no_gpu);
    assert!(config.verbose);
    assert!(config.trace);
    assert_eq!(config.trace_output, Some(PathBuf::from("/tmp/trace.json")));
}

#[test]
fn test_config_mock_backend() {
    let config = InferenceConfig::new("/model.gguf").with_mock_backend();
    assert!(config.use_mock_backend);
}

// ============================================================================
// tok_per_sec
// ============================================================================

#[test]
fn test_tok_per_sec_basic() {
    let tps = tok_per_sec(100, 1000.0); // 100 tokens in 1 second
    assert!((tps - 100.0).abs() < 0.01);
}

#[test]
fn test_tok_per_sec_zero_ms() {
    let tps = tok_per_sec(100, 0.0);
    assert_eq!(tps, 0.0);
}

#[test]
fn test_tok_per_sec_zero_tokens() {
    let tps = tok_per_sec(0, 1000.0);
    assert_eq!(tps, 0.0);
}

#[test]
fn test_tok_per_sec_fast() {
    let tps = tok_per_sec(1000, 100.0); // 1000 tokens in 0.1 second = 10000 tok/s
    assert!((tps - 10000.0).abs() < 1.0);
}

// ============================================================================
// prefault_mmap
// ============================================================================

#[test]
fn test_prefault_mmap_empty() {
    prefault_mmap(&[]);
}

#[test]
fn test_prefault_mmap_small() {
    let data = vec![0u8; 100];
    prefault_mmap(&data);
}

#[test]
fn test_prefault_mmap_page_boundary() {
    let data = vec![0u8; 4096]; // Exactly one page
    prefault_mmap(&data);
}

#[test]
fn test_prefault_mmap_multi_page() {
    let data = vec![0u8; 4096 * 3 + 100]; // 3+ pages
    prefault_mmap(&data);
}

// ============================================================================
// is_legacy_gguf_quant
// ============================================================================

#[test]
fn test_is_legacy_gguf_quant() {
    // PMAT-782: Q4_0/Q4_1/Q5_0 candle-layout kernels now match CPU (parity gate
    // PASSES, cosine≈0.9998) → GPU-eligible.
    // PMAT-783: the gate FAILS CLOSED for every quant WITHOUT a verified GPU kernel
    // (anything outside the from_ggml_type whitelist), because resolve_qtype's
    // `.unwrap_or(Q4K)` would otherwise silently decode it as Q4_K → garbage.

    // GPU-eligible (have a real WeightQuantType kernel) → NOT gated:
    assert!(!is_legacy_gguf_quant(0)); // F32
    assert!(!is_legacy_gguf_quant(2)); // Q4_0 — fixed (candle layout)
    assert!(!is_legacy_gguf_quant(3)); // Q4_1 — fixed (PMAT-782 candle layout)
    assert!(!is_legacy_gguf_quant(6)); // Q5_0 — fixed (candle layout)
    assert!(!is_legacy_gguf_quant(8)); // Q8_0
    assert!(!is_legacy_gguf_quant(12)); // Q4_K
    assert!(!is_legacy_gguf_quant(13)); // Q5_K
    assert!(!is_legacy_gguf_quant(14)); // Q6_K

    // No GPU kernel → MUST be gated to CPU (else silent Q4_K garbage):
    assert!(is_legacy_gguf_quant(1)); // F16 (no GGUF f16 GEMV here)
    assert!(is_legacy_gguf_quant(7)); // Q5_1 — no GPU kernel
    assert!(is_legacy_gguf_quant(9)); // Q8_1
    assert!(is_legacy_gguf_quant(10)); // Q2_K
    assert!(is_legacy_gguf_quant(11)); // Q3_K
    assert!(is_legacy_gguf_quant(15)); // Q8_K
    assert!(is_legacy_gguf_quant(30)); // BF16
    assert!(is_legacy_gguf_quant(100)); // Unknown / IQ* families
}

// PMAT-783: model_has_legacy_quant must inspect EVERY projection tensor —
// including QKV and the FFN gate, which the pre-PMAT-783 gate omitted. A Q2_K(10)
// tensor hidden in QKV or the gate must still force CPU (else resolve_qtype's
// `.unwrap_or(Q4K)` decodes Q2_K bytes as Q4_K → garbage on the GPU-resident path).
#[test]
fn test_model_has_legacy_quant_checks_qkv_and_gate() {
    use crate::gguf::test_helpers::create_test_model_with_config;
    use crate::gguf::{ArchConstraints, GGUFConfig, OwnedQKVWeights};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    // Baseline: an all-Q4K model is GPU-eligible.
    let base = create_test_model_with_config(&config);
    assert!(
        !model_has_legacy_quant(&base),
        "all-Q4K model must be GPU-eligible"
    );

    // Q2_K (type 10) hidden ONLY in the fused QKV tensor must still gate to CPU.
    let mut qkv_model = create_test_model_with_config(&config);
    if let OwnedQKVWeights::Fused(t) = &mut qkv_model.layers[0].qkv_weight {
        t.qtype = 10; // Q2_K — no GPU kernel
    }
    assert!(
        model_has_legacy_quant(&qkv_model),
        "Q2_K in QKV must force CPU"
    );

    // Q3_K (type 11) hidden ONLY in the FFN gate must still gate to CPU.
    let mut gate_model = create_test_model_with_config(&config);
    if let Some(g) = gate_model.layers[0].ffn_gate_weight.as_mut() {
        g.qtype = 11; // Q3_K — no GPU kernel
        assert!(
            model_has_legacy_quant(&gate_model),
            "Q3_K in FFN gate must force CPU"
        );
    }
}

// ============================================================================
// validate_model_path
// ============================================================================

#[test]
fn test_validate_model_path_traversal() {
    let path = std::path::Path::new("../../../etc/passwd");
    let result = validate_model_path(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("traversal") || err.contains("Security"),
        "Expected traversal error, got: {}",
        err
    );
}

#[test]
fn test_validate_model_path_bad_extension() {
    // Create a temp file with bad extension
    let tmp = std::env::temp_dir().join("test_validate.txt");
    std::fs::write(&tmp, "dummy").unwrap();

    let result = validate_model_path(&tmp);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("extension") || err.contains("Security"),
        "Expected extension error, got: {}",
        err
    );

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_validate_model_path_nonexistent() {
    let path = std::path::Path::new("/nonexistent/model.gguf");
    let result = validate_model_path(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not found") || err.contains("File"),
        "Expected not-found error, got: {}",
        err
    );
}

#[test]
fn test_validate_model_path_no_extension() {
    let path = std::path::Path::new("/tmp/modelfile");
    let result = validate_model_path(path);
    assert!(result.is_err());
}

#[test]
fn test_validate_model_path_valid_extensions() {
    // Can't fully test without creating files, but verify valid extension list
    for ext in &["gguf", "safetensors", "apr", "bin", "json"] {
        let tmp = std::env::temp_dir().join(format!("test_validate.{}", ext));
        std::fs::write(&tmp, "dummy").unwrap();

        let result = validate_model_path(&tmp);
        // Should pass extension check (may fail on "not a file" for directories)
        if let Err(e) = &result {
            let err = e.to_string();
            assert!(
                !err.contains("extension"),
                "Extension .{} should be valid but got: {}",
                ext,
                err
            );
        }

        std::fs::remove_file(&tmp).ok();
    }
}

#[test]
fn test_validate_model_path_directory() {
    let tmp_dir = std::env::temp_dir().join("test_validate_dir.gguf");
    std::fs::create_dir_all(&tmp_dir).ok();

    let result = validate_model_path(&tmp_dir);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not a regular file") || err.contains("Security"),
        "Expected not-a-file error, got: {}",
        err
    );

    std::fs::remove_dir_all(&tmp_dir).ok();
}

// ============================================================================
// PreparedTokens
// ============================================================================

#[test]
fn test_prepared_tokens_from_raw() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![1, 2, 3, 4, 5]);
    let prepared = prepare_tokens(&config, &ModelFormat::Gguf).unwrap();
    assert_eq!(prepared.tokens(), &[1, 2, 3, 4, 5]);
    assert_eq!(prepared.input_count(), 5);
}

#[test]
fn test_prepared_tokens_no_prompt() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.gguf");
    let prepared = prepare_tokens(&config, &ModelFormat::Gguf).unwrap();
    // No prompt and no input_tokens -> BOS token [1]
    assert_eq!(prepared.tokens(), &[1u32]);
    assert_eq!(prepared.input_count(), 1);
}

// ============================================================================
// safetensors_arch_to_template_hint
// ============================================================================

#[test]
fn test_safetensors_arch_qwen() {
    // GH-317/318: Exact match via normalize_architecture contract
    assert_eq!(
        safetensors_arch_to_template_hint("Qwen2ForCausalLM", "model"),
        "qwen2"
    );
    // "QwenModel" is not in the contract — defaults to "llama"
    assert_eq!(
        safetensors_arch_to_template_hint("QwenModel", "model"),
        "llama"
    );
}

#[test]
fn test_safetensors_arch_llama() {
    assert_eq!(
        safetensors_arch_to_template_hint("LlamaForCausalLM", "model"),
        "llama"
    );
}

#[test]
fn test_safetensors_arch_mistral() {
    assert_eq!(
        safetensors_arch_to_template_hint("MistralForCausalLM", "model"),
        "mistral"
    );
}

#[test]
fn test_safetensors_arch_phi() {
    // GH-317/318: PhiForCausalLM maps to "phi2" (Phi-1.5/Phi-2), Phi3ForCausalLM maps to "phi"
    assert_eq!(
        safetensors_arch_to_template_hint("PhiForCausalLM", "model"),
        "phi2"
    );
    assert_eq!(
        safetensors_arch_to_template_hint("Phi3ForCausalLM", "model"),
        "phi"
    );
}

#[test]
fn test_safetensors_arch_unknown() {
    // GH-317/318: Unknown HF class names default to "llama" via normalize_architecture
    let result = safetensors_arch_to_template_hint("CustomModel", "my-model-instruct");
    assert_eq!(result, "llama");
}

// ============================================================================
// apr_arch_to_template_hint
// ============================================================================

#[test]
fn test_apr_arch_qwen() {
    // GH-318: normalize_architecture does exact matching from contract
    assert_eq!(apr_arch_to_template_hint("qwen2", "model"), "qwen2");
    assert_eq!(apr_arch_to_template_hint("qwen3", "model"), "qwen3");
    // "Qwen" not in contract → defaults to "llama" (no fallback to model_name)
    assert_eq!(apr_arch_to_template_hint("Qwen", "model"), "llama");
}

#[test]
fn test_apr_arch_llama() {
    assert_eq!(apr_arch_to_template_hint("llama", "model"), "llama");
    // "LLaMA" not in contract → defaults to "llama"
    assert_eq!(apr_arch_to_template_hint("LLaMA", "model"), "llama");
}

#[test]
fn test_apr_arch_mistral() {
    assert_eq!(apr_arch_to_template_hint("mistral", "model"), "mistral");
}

#[test]
fn test_apr_arch_phi() {
    assert_eq!(apr_arch_to_template_hint("phi3", "model"), "phi");
    // "Phi" not in contract → defaults to "llama"
    assert_eq!(apr_arch_to_template_hint("Phi", "model"), "llama");
}

#[test]
fn test_apr_arch_unknown_defaults_to_llama() {
    // GH-318: No fallback to model_name — contract always returns canonical key
    assert_eq!(apr_arch_to_template_hint("unknown", "my-model"), "llama");
    assert_eq!(apr_arch_to_template_hint("custom", "instruct-v2"), "llama");
}

// ============================================================================
// clean_model_output edge cases
// ============================================================================

#[test]
fn test_clean_model_output_empty() {
    let cleaned = clean_model_output("");
    assert_eq!(cleaned, "");
}

#[test]
fn test_clean_model_output_only_markers() {
    let cleaned = clean_model_output("<|im_start|>assistant\n<|im_end|>");
    assert_eq!(cleaned, "");
}

#[test]
fn test_clean_model_output_endoftext() {
    // clean_model_output removes markers but keeps surrounding text
    let cleaned = clean_model_output("Hello world<|endoftext|>more text");
    assert_eq!(cleaned, "Hello worldmore text");
}

#[test]
fn test_clean_model_output_preserves_content() {
    let cleaned = clean_model_output("The answer is 42");
    assert_eq!(cleaned, "The answer is 42");
}

#[test]
fn test_clean_model_output_complex_markers() {
    // clean_model_output strips markers but keeps text around them
    let cleaned = clean_model_output(
        "<|im_start|>assistant\nLine 1\nLine 2\nLine 3<|im_end|><|endoftext|>garbage",
    );
    assert_eq!(cleaned, "Line 1\nLine 2\nLine 3garbage");
}

// ============================================================================
// InferenceResult construction
// ============================================================================

#[test]
fn test_inference_result_fields() {
    let result = InferenceResult {
        text: "Hello".to_string(),
        tokens: vec![1, 2, 3],
        input_token_count: 1,
        generated_token_count: 2,
        inference_ms: 100.0,
        tok_per_sec: 20.0,
        load_ms: 50.0,
        format: "GGUF".to_string(),
        used_gpu: false,
    };
    assert_eq!(result.text, "Hello");
    assert_eq!(result.tokens.len(), 3);
    assert_eq!(result.input_token_count, 1);
    assert_eq!(result.generated_token_count, 2);
    assert!((result.inference_ms - 100.0).abs() < 0.01);
    assert!((result.tok_per_sec - 20.0).abs() < 0.01);
    assert!((result.load_ms - 50.0).abs() < 0.01);
    assert_eq!(result.format, "GGUF");
    assert!(!result.used_gpu);
}
