use super::*;
use std::collections::BTreeMap;

// ========================================================================
// hf_to_gguf_name: Attention projection patterns
// ========================================================================

#[test]
fn test_hf_to_gguf_name_q_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.self_attn.q_proj.weight"),
        "blk.0.attn_q.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_q_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.5.self_attn.q_proj.bias"),
        "blk.5.attn_q.bias"
    );
}

#[test]
fn test_hf_to_gguf_name_k_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.3.self_attn.k_proj.weight"),
        "blk.3.attn_k.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_k_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.11.self_attn.k_proj.bias"),
        "blk.11.attn_k.bias"
    );
}

#[test]
fn test_hf_to_gguf_name_v_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.7.self_attn.v_proj.weight"),
        "blk.7.attn_v.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_v_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.self_attn.v_proj.bias"),
        "blk.0.attn_v.bias"
    );
}

#[test]
fn test_hf_to_gguf_name_o_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.2.self_attn.o_proj.weight"),
        "blk.2.attn_output.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_o_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.31.self_attn.o_proj.bias"),
        "blk.31.attn_output.bias"
    );
}

// ========================================================================
// hf_to_gguf_name: Fused QKV patterns
// ========================================================================

#[test]
fn test_hf_to_gguf_name_qkv_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.self_attn.qkv_proj.weight"),
        "blk.0.attn_qkv.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_qkv_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.4.self_attn.qkv_proj.bias"),
        "blk.4.attn_qkv.bias"
    );
}

// ========================================================================
// hf_to_gguf_name: MLP / FFN patterns
// ========================================================================

#[test]
fn test_hf_to_gguf_name_gate_proj() {
    assert_eq!(
        hf_to_gguf_name("model.layers.1.mlp.gate_proj.weight"),
        "blk.1.ffn_gate.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_up_proj() {
    assert_eq!(
        hf_to_gguf_name("model.layers.10.mlp.up_proj.weight"),
        "blk.10.ffn_up.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_down_proj() {
    assert_eq!(
        hf_to_gguf_name("model.layers.23.mlp.down_proj.weight"),
        "blk.23.ffn_down.weight"
    );
}

// ========================================================================
// hf_to_gguf_name: Layer norm patterns
// ========================================================================

#[test]
fn test_hf_to_gguf_name_input_layernorm() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.input_layernorm.weight"),
        "blk.0.attn_norm.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_post_attention_layernorm() {
    assert_eq!(
        hf_to_gguf_name("model.layers.15.post_attention_layernorm.weight"),
        "blk.15.ffn_norm.weight"
    );
}

// ========================================================================
// hf_to_gguf_name: Non-layer tensors (embedding, lm_head, output norm)
// ========================================================================

#[test]
fn test_hf_to_gguf_name_embed_tokens() {
    assert_eq!(
        hf_to_gguf_name("model.embed_tokens.weight"),
        "token_embd.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_lm_head() {
    assert_eq!(hf_to_gguf_name("lm_head.weight"), "output.weight");
}

#[test]
fn test_hf_to_gguf_name_output_norm() {
    assert_eq!(hf_to_gguf_name("model.norm.weight"), "output_norm.weight");
}

// ========================================================================
// hf_to_gguf_name: Unknown / passthrough
// ========================================================================

#[test]
fn test_hf_to_gguf_name_unknown_passthrough() {
    // Completely unknown names should pass through unchanged
    assert_eq!(hf_to_gguf_name("some.custom.tensor"), "some.custom.tensor");
}

#[test]
fn test_hf_to_gguf_name_unknown_layer_suffix_passthrough() {
    // A layer tensor with an unrecognized suffix passes through as-is
    assert_eq!(
        hf_to_gguf_name("model.layers.0.some_unknown_suffix.weight"),
        "blk.0.some_unknown_suffix.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_empty_string() {
    assert_eq!(hf_to_gguf_name(""), "");
}

// ========================================================================
// hf_to_gguf_name: Multi-digit layer indices (regression guard)
//
// Bug class: off-by-one in layer number parsing. Models with 100+ layers
// (e.g. Llama-70B has 80 layers) must produce correct multi-digit indices.
// ========================================================================

#[test]
fn test_hf_to_gguf_name_high_layer_index() {
    assert_eq!(
        hf_to_gguf_name("model.layers.79.self_attn.q_proj.weight"),
        "blk.79.attn_q.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_three_digit_layer_index() {
    // Some very deep models exceed 100 layers
    assert_eq!(
        hf_to_gguf_name("model.layers.127.mlp.gate_proj.weight"),
        "blk.127.ffn_gate.weight"
    );
}

// ========================================================================
// hf_to_gguf_name: Consistency — roundtrip with Architecture::qwen2_map_name
//
// Bug class: asymmetric mapping. If HF->GGUF and GGUF->HF aren't inverses,
// round-trip export/import corrupts tensor names silently.
// ========================================================================

#[test]
fn test_hf_to_gguf_name_all_layer_suffixes_mapped() {
    // Verify every known suffix produces a different GGUF name (no collisions)
    let suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ];

    let mut gguf_names: Vec<String> = suffixes
        .iter()
        .map(|s| hf_to_gguf_name(&format!("model.layers.0.{s}")))
        .collect();

    let count_before = gguf_names.len();
    gguf_names.sort();
    gguf_names.dedup();
    assert_eq!(
        gguf_names.len(),
        count_before,
        "Name collision detected in hf_to_gguf_name mapping"
    );
}

// ========================================================================
// infer_vocab_hidden: Embedding tensor present
// ========================================================================

#[test]
fn test_infer_vocab_hidden_from_embed_tokens() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Qwen2-0.5B: vocab=151936, hidden=896
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.0; 151936 * 896], vec![151936, 896]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 151936);
    assert_eq!(hidden, 896);
}

#[test]
fn test_infer_vocab_hidden_from_token_embd() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF-style naming
    tensors.insert(
        "token_embd.weight".to_string(),
        (vec![0.0; 32000 * 4096], vec![32000, 4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 32000);
    assert_eq!(hidden, 4096);
}

// ========================================================================
// infer_vocab_hidden: Fallback to lm_head
// ========================================================================

#[test]
fn test_infer_vocab_hidden_fallback_to_lm_head() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // No embedding tensor, but lm_head present
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.0; 32000 * 4096], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 4096 * 4096], vec![4096, 4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 32000);
    assert_eq!(hidden, 4096);
}

#[test]
fn test_infer_vocab_hidden_fallback_to_output_weight() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF-style output.weight as lm_head equivalent
    tensors.insert(
        "output.weight".to_string(),
        (vec![0.0; 128256 * 4096], vec![128256, 4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 128256);
    assert_eq!(hidden, 4096);
}

// ========================================================================
// infer_vocab_hidden: Fallback to q_proj for hidden_dim only
// ========================================================================

#[test]
fn test_infer_vocab_hidden_hidden_from_q_proj() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // No embedding or lm_head — only layer weights
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 4096 * 4096], vec![4096, 4096]),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        (vec![0.0; 11008 * 4096], vec![11008, 4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    // vocab_size cannot be inferred without embedding/lm_head
    assert_eq!(vocab, 0);
    assert_eq!(hidden, 4096);
}

// ========================================================================
// infer_vocab_hidden: Empty tensor map
// ========================================================================

#[test]
fn test_infer_vocab_hidden_empty_tensors() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 0);
    assert_eq!(hidden, 0);
}

// ========================================================================
// infer_vocab_hidden: 1D tensors (should NOT match — requires 2D)
//
// Bug class: 1D norm weights like input_layernorm.weight with shape [4096]
// could be misinterpreted as vocab_size=4096, hidden=0 if the dimension
// check is missing.
// ========================================================================

#[test]
fn test_infer_vocab_hidden_ignores_1d_embedding() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D tensor should NOT be treated as an embedding
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.0; 4096], vec![4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    // 1D tensor doesn't satisfy the shape.len() == 2 check
    assert_eq!(vocab, 0);
    assert_eq!(hidden, 0);
}

#[test]
fn test_infer_vocab_hidden_ignores_1d_lm_head() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D lm_head should NOT match
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.0; 32000], vec![32000]),
    );
    // But 2D q_proj should give hidden_dim
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 2048 * 2048], vec![2048, 2048]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 0); // 1D lm_head skipped
    assert_eq!(hidden, 2048); // from q_proj fallback
}

// ========================================================================
// infer_vocab_hidden: Embedding takes priority over lm_head
//
// Bug class: If lm_head is checked first and embed_tokens second, the
// wrong tensor could provide dimensions for tied-embedding models where
// lm_head and embed_tokens have different names but same data.
// ========================================================================

#[test]
fn test_infer_vocab_hidden_embedding_priority_over_lm_head() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Both present — embedding should win
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.0; 32000 * 4096], vec![32000, 4096]),
    );
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.0; 151936 * 896], vec![151936, 896]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    // Embedding shapes should be used, not lm_head
    assert_eq!(vocab, 151936);
    assert_eq!(hidden, 896);
}

// ========================================================================
// unfuse_qkv_tensors: Passthrough when no fused tensors present
// ========================================================================

#[test]
fn test_unfuse_qkv_tensors_no_fused_passthrough() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![1.0; 16], vec![4, 4]),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![2.0; 16], vec![4, 4]),
    );
    tensors.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        (vec![3.0; 16], vec![4, 4]),
    );

    // Non-APR path means read_apr_metadata returns None, but since no
    // fused tensors exist, the early return fires first.
    let result = unfuse_qkv_tensors(tensors.clone(), Path::new("/tmp/fake.safetensors"));

    assert_eq!(result.len(), 3);
    assert!(result.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(result.contains_key("model.layers.0.self_attn.k_proj.weight"));
    assert!(result.contains_key("model.layers.0.self_attn.v_proj.weight"));
}

// ========================================================================
// unfuse_qkv_tensors: Fused tensors present but no APR metadata
//
// Bug class: If metadata is unavailable (non-APR input, corrupt file),
// the function should return tensors unchanged rather than panicking.
// ========================================================================

#[test]
fn test_unfuse_qkv_tensors_fused_but_no_metadata_returns_unchanged() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.weight".to_string(),
        (vec![1.0; 48], vec![12, 4]),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        (vec![2.0; 16], vec![4, 4]),
    );

    // Non-APR path -> read_apr_metadata returns None -> early return with original tensors
    let result = unfuse_qkv_tensors(tensors.clone(), Path::new("/tmp/nonexistent.safetensors"));

    // Should be unchanged since metadata couldn't be read
    assert_eq!(result.len(), 2);
    assert!(result.contains_key("model.layers.0.self_attn.qkv_proj.weight"));
}

// ========================================================================
// ExportFormat: from_str edge cases beyond what coverage_types tests
// ========================================================================

#[test]
fn test_export_format_from_str_mixed_case() {
    use std::str::FromStr;
    // Mixed case should work because of to_lowercase()
    assert_eq!(
        ExportFormat::from_str("SafeTensors"),
        Ok(ExportFormat::SafeTensors)
    );
    assert_eq!(ExportFormat::from_str("GgUf"), Ok(ExportFormat::Gguf));
    assert_eq!(
        ExportFormat::from_str("TorchScript"),
        Ok(ExportFormat::TorchScript)
    );
    assert_eq!(ExportFormat::from_str("PT"), Ok(ExportFormat::TorchScript));
}

#[test]
fn test_export_format_from_str_error_message_contains_input() {
    use std::str::FromStr;
    // The error message should contain the bad input for debugging
    let err = ExportFormat::from_str("parquet").unwrap_err();
    assert!(
        err.contains("parquet"),
        "Error should contain the unrecognized input, got: {err}"
    );
}

// ========================================================================
// ExportFormat: extension + is_supported consistency
//
// Bug class: Adding a new format variant without updating extension() or
// is_supported() leads to runtime panics or silent mis-routing.
// ========================================================================

#[test]
fn test_export_format_all_variants_have_nonempty_extension() {
    let all_formats = [
        ExportFormat::SafeTensors,
        ExportFormat::Gguf,
        ExportFormat::Onnx,
        ExportFormat::TorchScript,
    ];
    for fmt in &all_formats {
        let ext = fmt.extension();
        assert!(!ext.is_empty(), "Format {:?} has empty extension", fmt);
        // Extension should not contain dots (it's the bare extension)
        assert!(
            !ext.contains('.'),
            "Format {:?} extension contains a dot: {}",
            fmt,
            ext
        );
    }
}

#[test]
fn test_export_format_supported_formats_have_valid_extensions() {
    // Every supported format should produce a known extension
    let supported: Vec<ExportFormat> = [
        ExportFormat::SafeTensors,
        ExportFormat::Gguf,
        ExportFormat::Onnx,
        ExportFormat::TorchScript,
    ]
    .into_iter()
    .filter(|f| f.is_supported())
    .collect();

    assert!(
        supported.len() >= 2,
        "At least SafeTensors and Gguf should be supported"
    );
    for fmt in &supported {
        assert!(
            ["safetensors", "gguf"].contains(&fmt.extension()),
            "Supported format {:?} has unexpected extension: {}",
            fmt,
            fmt.extension()
        );
    }
}

// ========================================================================
// ExportOptions: Default values
// ========================================================================

#[test]
fn test_export_options_default() {
    let opts = ExportOptions::default();
    assert_eq!(opts.format, ExportFormat::SafeTensors);
    assert!(opts.quantize.is_none());
    assert!(opts.include_tokenizer);
    assert!(opts.include_config);
}

// ========================================================================
// ExportReport: Clone and Debug
// ========================================================================

#[test]
fn test_export_report_clone_and_debug() {
    let report = ExportReport {
        original_size: 1024,
        exported_size: 512,
        tensor_count: 42,
        format: ExportFormat::Gguf,
        quantization: Some(QuantizationType::Q4K),
    };
    let cloned = report.clone();
    assert_eq!(cloned.original_size, 1024);
    assert_eq!(cloned.exported_size, 512);
    assert_eq!(cloned.tensor_count, 42);
    assert_eq!(cloned.format, ExportFormat::Gguf);

    // Debug should not panic
    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("ExportReport"));
}

// ========================================================================
// infer_vocab_hidden: Realistic multi-tensor model configurations
//
// Bug class: The function iterates a BTreeMap (sorted order). If the
// first matching tensor has unexpected shape, inference fails silently.
// These tests use realistic tensor maps mimicking real models.
// ========================================================================

#[test]
fn test_infer_vocab_hidden_llama_7b_layout() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Llama-7B: vocab=32000, hidden=4096, 32 layers
    // Use small data vectors (shapes are what matter)
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert("lm_head.weight".to_string(), (vec![], vec![32000, 4096]));
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    tensors.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        (vec![], vec![4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 32000);
    assert_eq!(hidden, 4096);
}

#[test]
fn test_infer_vocab_hidden_qwen2_05b_layout() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Qwen2-0.5B: vocab=151936, hidden=896, 24 layers
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![151936, 896]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![896, 896]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 151936);
    assert_eq!(hidden, 896);
}

#[test]
fn test_infer_vocab_hidden_only_unrelated_tensors() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Tensors that don't match any known patterns
    tensors.insert(
        "encoder.block.0.weight".to_string(),
        (vec![], vec![512, 512]),
    );
    tensors.insert(
        "decoder.block.0.weight".to_string(),
        (vec![], vec![512, 512]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 0);
    assert_eq!(hidden, 0);
}

// ========================================================================
// infer_model_config: Basic Qwen2-0.5B layout
// ========================================================================

#[test]
fn test_infer_model_config_qwen2_05b_layout() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Qwen2-0.5B: vocab=151936, hidden=896, 24 layers, 14 heads
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![151936, 896]),
    );
    // 24 layers (0..23)
    for i in 0..24 {
        tensors.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            (vec![], vec![896, 896]),
        );
        tensors.insert(
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            (vec![], vec![128, 896]),
        );
        tensors.insert(
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            (vec![], vec![4864, 896]),
        );
    }

    let config = infer_model_config(&tensors);
    // Parse as JSON to verify fields
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    assert_eq!(v["hidden_size"], 896);
    assert_eq!(v["num_hidden_layers"], 24);
    assert_eq!(v["vocab_size"], 151936);
    assert_eq!(v["num_attention_heads"], 14);
    assert_eq!(v["intermediate_size"], 4864);
}

// ========================================================================
// infer_model_config: Llama-7B layout
// ========================================================================

#[test]
fn test_infer_model_config_llama_7b_layout() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Llama-7B: vocab=32000, hidden=4096, 32 layers
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert("lm_head.weight".to_string(), (vec![], vec![32000, 4096]));
    for i in 0..32 {
        tensors.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            (vec![], vec![4096, 4096]),
        );
        tensors.insert(
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            (vec![], vec![4096, 4096]),
        );
        tensors.insert(
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            (vec![], vec![11008, 4096]),
        );
    }

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    assert_eq!(v["hidden_size"], 4096);
    assert_eq!(v["num_hidden_layers"], 32);
    assert_eq!(v["vocab_size"], 32000);
    assert_eq!(v["num_attention_heads"], 32);
    assert_eq!(v["intermediate_size"], 11008);
    // MHA: num_key_value_heads == num_attention_heads
    assert_eq!(v["num_key_value_heads"], 32);
}

// ========================================================================
// infer_model_config: Empty tensors - all defaults
// ========================================================================

#[test]
fn test_infer_model_config_empty_tensors_returns_defaults() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // Defaults when nothing is inferable
    assert_eq!(v["hidden_size"], 4096);
    assert_eq!(v["num_hidden_layers"], 12);
    assert_eq!(v["vocab_size"], 32000);
}

// ========================================================================
// infer_model_config: No embedding, fallback to lm_head
// ========================================================================

#[test]
fn test_infer_model_config_no_embedding_fallback_to_lm_head() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Only lm_head, no embed_tokens
    tensors.insert("lm_head.weight".to_string(), (vec![], vec![32000, 4096]));
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // hidden_size defaults to 4096 (no embed_tokens/token_embd found)
    assert_eq!(v["hidden_size"], 4096);
    // vocab_size inferred from lm_head (larger dim)
    assert_eq!(v["vocab_size"], 32000);
}

// ========================================================================
// infer_model_config: No layer tensors - num_layers defaults to 12
// ========================================================================

#[test]
fn test_infer_model_config_no_layers_defaults_to_12() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Only embedding, no layer tensors
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    assert_eq!(v["num_hidden_layers"], 12);
}

// ========================================================================
// infer_model_config: GGUF-style tensor names (blk.N / token_embd)
// ========================================================================

#[test]
fn test_infer_model_config_gguf_style_names() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF naming convention
    tensors.insert("token_embd.weight".to_string(), (vec![], vec![32000, 4096]));
    for i in 0..16 {
        tensors.insert(format!("blk.{i}.attn_q.weight"), (vec![], vec![4096, 4096]));
    }

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // hidden_size from token_embd (smaller dim)
    assert_eq!(v["hidden_size"], 4096);
    // num_layers from blk.0..blk.15 => 16 layers
    assert_eq!(v["num_hidden_layers"], 16);
    // vocab from token_embd (larger dim)
    assert_eq!(v["vocab_size"], 32000);
}

// ========================================================================
// infer_model_config: Vocab < hidden triggers warning but still works
// ========================================================================

#[test]
fn test_infer_model_config_vocab_less_than_hidden_unusual() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Unusual scenario: embedding where both dims are close
    // embed_tokens shape [100, 4096] -> hidden=100 (min), vocab from lm_head
    // lm_head shape [200, 100] -> vocab=200 (max)
    // This results in vocab=200 < hidden? No, hidden=100 and vocab=200 so vocab > hidden.
    // To trigger vocab < hidden, we need a specific arrangement.
    // embed_tokens [512, 4096] -> hidden=512 (min dim)
    // No lm_head -> fallback to embed_tokens for vocab, vocab = 4096 (max dim)
    // So vocab=4096 > hidden=512 -- still fine.
    // The warning triggers when vocab_inferred && hidden_inferred && vocab < hidden.
    // This happens if the embedding shape is [small, large] -> hidden=small, vocab=large
    // and lm_head shape [small2, large2] with max=large2 but large2 < hidden.
    // Use: embed_tokens [8192, 256] -> hidden=256, vocab=8192 from embed_tokens fallback.
    // Actually lm_head is checked first for vocab. If lm_head has shape [100, 256]:
    //   vocab = max(100, 256) = 256, but hidden = 256 -> vocab == hidden, not less.
    // Use: lm_head [100, 50] -> vocab=100 (max), but hidden from embed_tokens min dim.
    // embed_tokens [200, 50] -> hidden=50 (min). vocab from lm_head = max(100,50)=100.
    // Now vocab=100 > hidden=50. Still no warning.
    // We need vocab < hidden: embed_tokens [200,50] -> hidden=50.
    // lm_head must give vocab < 50. lm_head [30, 20] -> vocab=30.
    // vocab_inferred=true, hidden_inferred=true, vocab=30 < hidden=50 -> WARNING
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![200, 50]),
    );
    tensors.insert("lm_head.weight".to_string(), (vec![], vec![30, 20]));

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // hidden = min(200, 50) = 50
    assert_eq!(v["hidden_size"], 50);
    // vocab = max(30, 20) = 30 (from lm_head, checked before embed_tokens fallback)
    assert_eq!(v["vocab_size"], 30);
    // The function should still produce valid JSON despite the warning
}

// ========================================================================
// infer_model_config: GQA detection (k_proj < q_proj)
// ========================================================================

#[test]
fn test_infer_model_config_gqa_detection() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Model with GQA: hidden=4096, 32 attention heads, 8 KV heads
    // head_dim = 4096/32 = 128
    // k_proj shape: [num_kv_heads * head_dim, hidden] = [1024, 4096]
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![], vec![1024, 4096]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    assert_eq!(v["num_attention_heads"], 32);
    // GQA: k_proj first dim = 1024, head_dim = 128
    // num_kv_heads = 1024 / 128 = 8
    assert_eq!(v["num_key_value_heads"], 8);
}

// ========================================================================
// infer_model_config: 1D embedding tensor fallback
// ========================================================================

#[test]
fn test_infer_model_config_1d_embedding_fallback() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D embedding tensor - uses last dim as-is
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // 1D tensor: shape.last() = 4096
    assert_eq!(v["hidden_size"], 4096);
}

// ========================================================================
// infer_model_config: num_attention_heads fallback for various hidden sizes
// ========================================================================

#[test]
fn test_infer_model_config_heads_fallback_hidden_896() {
    // Qwen2.5-0.5B: hidden=896 -> 14 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 896]),
    );
    // No q_proj -> fallback path for num_attention_heads
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 14);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_1536() {
    // Qwen2.5-1.5B: hidden=1536 -> 12 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 1536]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 12);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_2048() {
    // Llama-style: hidden=2048 -> 16 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 2048]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 16);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_4096() {
    // Llama-7B: hidden=4096 -> 32 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 32);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_5120() {
    // Llama-13B: hidden=5120 -> 40 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 5120]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 40);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_8192() {
    // Llama-70B: hidden=8192 -> 64 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 8192]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 64);
}

#[test]
fn test_infer_model_config_heads_fallback_unknown_hidden_size() {
    // Non-standard hidden size: 3072 -> default formula: (3072 / 128).max(1) = 24
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 3072]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // 3072 / 128 = 24
    assert_eq!(v["num_attention_heads"], 24);
}

// ========================================================================
// infer_model_config: num_attention_heads with q_proj present (non-fallback)
// ========================================================================

#[test]
fn test_infer_model_config_heads_from_q_proj_small_hidden() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // hidden < 4096 -> head_dim = 64
    // hidden = 2048, head_dim = 64, num_heads = 2048/64 = 32
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 2048]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![2048, 2048]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // hidden=2048 < 4096 -> head_dim=64, num_heads = 2048/64 = 32
    assert_eq!(v["num_attention_heads"], 32);
}

#[test]
fn test_infer_model_config_heads_from_q_proj_large_hidden() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // hidden >= 4096 -> head_dim = 128
    // hidden = 4096, head_dim = 128, num_heads = 4096/128 = 32
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 32);
}

// ========================================================================
// infer_model_config: intermediate_size default when no MLP tensors
// ========================================================================

#[test]
fn test_infer_model_config_intermediate_size_default() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 2048]),
    );
    // No gate_proj/up_proj/feed_forward.w1 -> default to hidden_size * 4
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // hidden=2048, default intermediate = 2048*4 = 8192
    assert_eq!(v["intermediate_size"], 8192);
}

// ========================================================================
// infer_model_config: intermediate_size from up_proj
// ========================================================================

#[test]
fn test_infer_model_config_intermediate_size_from_up_proj() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    // up_proj with specific intermediate size
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        (vec![], vec![14336, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["intermediate_size"], 14336);
}

// ========================================================================
// infer_model_config: intermediate_size from feed_forward.w1
// ========================================================================

#[test]
fn test_infer_model_config_intermediate_size_from_feed_forward_w1() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.feed_forward.w1.weight".to_string(),
        (vec![], vec![11008, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["intermediate_size"], 11008);
}

// ========================================================================
// infer_model_config: num_key_value_heads defaults to num_attention_heads (MHA)
// ========================================================================

#[test]
fn test_infer_model_config_kv_heads_default_mha() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    // q_proj present but NO k_proj -> num_kv_heads defaults to num_attention_heads
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_key_value_heads"], v["num_attention_heads"]);
}

// ========================================================================
// infer_model_config: Output is always valid JSON
// ========================================================================

#[test]
fn test_infer_model_config_output_is_valid_json() {
    // Test with various tensor configurations to ensure JSON is always valid
    let configs: Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>> = vec![
        BTreeMap::new(), // empty
        {
            let mut m = BTreeMap::new();
            m.insert(
                "model.embed_tokens.weight".to_string(),
                (vec![], vec![100, 64]),
            );
            m
        },
        {
            let mut m = BTreeMap::new();
            m.insert("token_embd.weight".to_string(), (vec![], vec![50000, 768]));
            m.insert("blk.0.attn_q.weight".to_string(), (vec![], vec![768, 768]));
            m.insert("blk.5.attn_q.weight".to_string(), (vec![], vec![768, 768]));
            m
        },
    ];

    for tensors in &configs {
        let config = infer_model_config(tensors);
        let result: std::result::Result<serde_json::Value, _> = serde_json::from_str(&config);
        assert!(
            result.is_ok(),
            "infer_model_config produced invalid JSON for {} tensors: {config}",
            tensors.len()
        );
    }
}

// ========================================================================
// infer_model_config: Config contains all required HuggingFace fields
// ========================================================================

#[test]
fn test_infer_model_config_contains_all_required_fields() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // All fields required by HuggingFace SafeTensors inference (GH-193)
    let required_fields = [
        "architectures",
        "bos_token_id",
        "eos_token_id",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "sliding_window",
        "tie_word_embeddings",
        "torch_dtype",
        "use_cache",
        "use_sliding_window",
        "vocab_size",
    ];

    for field in &required_fields {
        assert!(
            !v[field].is_null(),
            "Required field '{field}' is missing from config JSON"
        );
    }
}

// ========================================================================
// infer_model_config: head_dim calculation
// ========================================================================

#[test]
fn test_infer_model_config_head_dim_affects_kv_heads() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // hidden=8192, 64 heads -> head_dim = 128
    // k_proj shape [2048, 8192] -> kv_heads = 2048/128 = 16
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 8192]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![8192, 8192]),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![], vec![2048, 8192]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // hidden=8192 >= 4096 -> head_dim=128
    // num_heads = 8192/128 = 64
    assert_eq!(v["num_attention_heads"], 64);
    // kv_heads = 2048/128 = 16
    assert_eq!(v["num_key_value_heads"], 16);
}

// ========================================================================
// infer_model_config: Embedding shape with GGUF transposed layout
// ========================================================================

#[test]
fn test_infer_model_config_transposed_embedding_gguf() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF sometimes stores embeddings as [hidden_size, vocab_size]
    // The function picks the smaller dim as hidden_size
    tensors.insert(
        "token_embd.weight".to_string(),
        (vec![], vec![4096, 151936]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // min(4096, 151936) = 4096 -> hidden
    assert_eq!(v["hidden_size"], 4096);
    // max(4096, 151936) = 151936 -> vocab
    assert_eq!(v["vocab_size"], 151936);
}

// ========================================================================
// infer_model_config: Alternative q_proj name patterns
// ========================================================================

#[test]
fn test_infer_model_config_attn_q_proj_pattern() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 2048]),
    );
    // "attn.q_proj" pattern (without "self_attn" prefix)
    tensors.insert(
        "model.layers.0.attn.q_proj.weight".to_string(),
        (vec![], vec![2048, 2048]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // Should detect q_proj via "attn.q_proj" pattern
    // hidden=2048 < 4096 -> head_dim=64, num_heads=32
    assert_eq!(v["num_attention_heads"], 32);
}

#[test]
fn test_infer_model_config_attention_wq_pattern() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    // "attention.wq" pattern (Llama-style)
    tensors.insert(
        "model.layers.0.attention.wq.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 32);
}

// ========================================================================
// infer_model_config: Alternative k_proj name patterns for GQA
// ========================================================================

#[test]
fn test_infer_model_config_gqa_via_attn_k_proj() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    // "attn.k_proj" pattern
    tensors.insert(
        "model.layers.0.attn.k_proj.weight".to_string(),
        (vec![], vec![512, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // kv_dim=512, head_dim=128, kv_heads = 512/128 = 4
    assert_eq!(v["num_key_value_heads"], 4);
}

#[test]
fn test_infer_model_config_gqa_via_attention_wk() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    // "attention.wk" pattern (Llama-style)
    tensors.insert(
        "model.layers.0.attention.wk.weight".to_string(),
        (vec![], vec![1024, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // kv_dim=1024, head_dim=128, kv_heads = 1024/128 = 8
    assert_eq!(v["num_key_value_heads"], 8);
}

// ========================================================================
// infer_model_config: output.weight used for vocab instead of lm_head
// ========================================================================

#[test]
fn test_infer_model_config_output_weight_for_vocab() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF-style output.weight instead of lm_head
    tensors.insert("output.weight".to_string(), (vec![], vec![128256, 4096]));
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["vocab_size"], 128256);
}

// ========================================================================
// infer_model_config: 1D vocab tensor fallback
// ========================================================================

#[test]
fn test_infer_model_config_1d_lm_head_vocab() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D lm_head - uses first dim
    tensors.insert("lm_head.weight".to_string(), (vec![], vec![50000]));
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["vocab_size"], 50000);
}

// ========================================================================
// infer_model_config: 1D embedding + 1D lm_head (degenerate case)
// ========================================================================

#[test]
fn test_infer_model_config_empty_1d_embedding() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D embedding with no elements - shape.last() returns Some(0)
    tensors.insert("token_embd.weight".to_string(), (vec![], vec![0]));
    let config = infer_model_config(&tensors);
    // Should still produce valid JSON
    let result: std::result::Result<serde_json::Value, _> = serde_json::from_str(&config);
    assert!(
        result.is_ok(),
        "Should produce valid JSON even with 0-dim tensor"
    );
}

// ========================================================================
// extract_user_metadata: Empty/short data
// ========================================================================

#[test]
fn test_extract_user_metadata_nonexistent_file() {
    let result = extract_user_metadata(Path::new("/tmp/nonexistent_apr_file_12345.apr"));
    assert!(result.is_empty());
}

#[test]
fn test_extract_user_metadata_short_data() {
    let dir = std::env::temp_dir().join("apr_test_short_data");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("short.apr");
    // Write less than 16 bytes
    fs::write(&path, &[0u8; 10]).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Invalid metadata JSON
// ========================================================================

#[test]
fn test_extract_user_metadata_invalid_json() {
    let dir = std::env::temp_dir().join("apr_test_invalid_json");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("invalid.apr");

    // Build APR-like bytes: magic(4) + version(4) + metadata_len(8) + garbage
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00"); // magic
    data.extend_from_slice(&2u32.to_le_bytes()); // version
    let garbage = b"not valid json{{{";
    let len = garbage.len() as u64;
    data.extend_from_slice(&len.to_le_bytes()); // metadata_len
    data.extend_from_slice(garbage);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Valid with source_metadata
// ========================================================================

#[test]
fn test_extract_user_metadata_with_source_metadata() {
    let dir = std::env::temp_dir().join("apr_test_source_metadata");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("with_meta.apr");

    let json = r#"{"custom":{"source_metadata":{"key1":"val1","key2":"val2"}}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00"); // magic
    data.extend_from_slice(&2u32.to_le_bytes()); // version
    let json_bytes = json.as_bytes();
    let len = json_bytes.len() as u64;
    data.extend_from_slice(&len.to_le_bytes()); // metadata_len
    data.extend_from_slice(json_bytes);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert_eq!(result.len(), 2);
    assert_eq!(result.get("key1").map(String::as_str), Some("val1"));
    assert_eq!(result.get("key2").map(String::as_str), Some("val2"));

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Valid without source_metadata
// ========================================================================

#[test]
fn test_extract_user_metadata_without_source_metadata() {
    let dir = std::env::temp_dir().join("apr_test_no_source_metadata");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("no_source_meta.apr");

    let json = r#"{"custom":{"other_key":"other_val"}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00");
    data.extend_from_slice(&2u32.to_le_bytes());
    let json_bytes = json.as_bytes();
    let len = json_bytes.len() as u64;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(json_bytes);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Non-string values in source_metadata skipped
// ========================================================================

#[test]
fn test_extract_user_metadata_non_string_values_skipped() {
    let dir = std::env::temp_dir().join("apr_test_non_string_values");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("mixed_types.apr");

    let json = r#"{"custom":{"source_metadata":{"str_key":"str_val","num_key":42,"bool_key":true,"null_key":null}}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00");
    data.extend_from_slice(&2u32.to_le_bytes());
    let json_bytes = json.as_bytes();
    let len = json_bytes.len() as u64;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(json_bytes);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    // Only the string value should be extracted
    assert_eq!(result.len(), 1);
    assert_eq!(result.get("str_key").map(String::as_str), Some("str_val"));

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Metadata length exceeds data length
// ========================================================================

#[test]
fn test_extract_user_metadata_metadata_len_exceeds_data() {
    let dir = std::env::temp_dir().join("apr_test_len_overflow");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("overflow.apr");

    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00");
    data.extend_from_slice(&2u32.to_le_bytes());
    // Claim metadata is 9999 bytes but only provide 5
    let len: u64 = 9999;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(b"hello");

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Invalid UTF-8 in metadata
// ========================================================================

#[test]
fn test_extract_user_metadata_invalid_utf8() {
    let dir = std::env::temp_dir().join("apr_test_invalid_utf8");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("bad_utf8.apr");

    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00");
    data.extend_from_slice(&2u32.to_le_bytes());
    // Invalid UTF-8 sequence
    let bad_bytes: &[u8] = &[0xFF, 0xFE, 0x80, 0x81, 0x82];
    let len = bad_bytes.len() as u64;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(bad_bytes);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// infer_tokenizer_json: Non-APR extension returns empty
// ========================================================================

#[test]
fn test_infer_tokenizer_json_non_apr_extension() {
    let result = infer_tokenizer_json(Path::new("/tmp/model.safetensors"));
    assert!(result.is_empty());
}

#[test]
fn test_infer_tokenizer_json_gguf_extension() {
    let result = infer_tokenizer_json(Path::new("/tmp/model.gguf"));
    assert!(result.is_empty());
}

#[test]
fn test_infer_tokenizer_json_no_extension() {
    let result = infer_tokenizer_json(Path::new("/tmp/model"));
    assert!(result.is_empty());
}

// ========================================================================
// infer_tokenizer_json: APR file that doesn't exist
// ========================================================================

#[test]
fn test_infer_tokenizer_json_nonexistent_apr() {
    let result = infer_tokenizer_json(Path::new("/tmp/nonexistent_12345.apr"));
    assert!(result.is_empty());
}

// ========================================================================
// infer_tokenizer_json: APR file too short
// ========================================================================

#[test]
fn test_infer_tokenizer_json_short_apr_file() {
    let dir = std::env::temp_dir().join("apr_test_short_apr_tokenizer");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("short.apr");
    // Write less than 44 bytes (header size)
    fs::write(&path, &[0u8; 20]).expect("write failed");

    let result = infer_tokenizer_json(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// infer_tokenizer_json: APR file with no tokenizer in metadata
// ========================================================================

#[test]
fn test_infer_tokenizer_json_apr_without_tokenizer() {
    let dir = std::env::temp_dir().join("apr_test_no_tokenizer");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("no_tok.apr");

    // Build fake APR: 44 bytes header + metadata JSON without tokenizer + terminator
    let mut data = vec![0u8; 44]; // header
    let metadata = r#"{"model_type": "qwen2", "hidden_size": 896}"#;
    data.extend_from_slice(metadata.as_bytes());
    data.extend_from_slice(b"}\n\n\n"); // terminator pattern

    fs::write(&path, &data).expect("write failed");

    let result = infer_tokenizer_json(&path);
    // Metadata doesn't contain "tokenizer" or "vocabulary"
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// GH-253-4: ValidatedGgufMetadata tests
// ========================================================================

#[test]
fn test_validated_metadata_requires_architecture() {
    use crate::format::gguf::GgufValue;
    let metadata = vec![(
        "general.name".to_string(),
        GgufValue::String("test".to_string()),
    )];
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("general.architecture"));
}

#[test]
fn test_validated_metadata_tokens_require_model() {
    use crate::format::gguf::GgufValue;
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("qwen2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(vec!["a".to_string()]),
        ),
    ];
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("tokenizer.ggml.model"));
}

#[test]
fn test_validated_metadata_model_requires_tokens() {
    use crate::format::gguf::GgufValue;
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("qwen2".to_string()),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gpt2".to_string()),
        ),
    ];
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("tokenizer.ggml.tokens"));
}

#[test]
fn test_validated_metadata_complete_passes() {
    use crate::format::gguf::GgufValue;
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("qwen2".to_string()),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(vec!["hello".to_string(), "world".to_string()]),
        ),
    ];
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().as_slice().len(), 3);
}

#[test]
fn test_validated_metadata_no_tokenizer_passes() {
    use crate::format::gguf::GgufValue;
    // Architecture-only metadata (no tokenizer) is valid
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        ),
        (
            "general.name".to_string(),
            GgufValue::String("test".to_string()),
        ),
    ];
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_ok());
}

// ========================================================================
// Bug 211: GGUF export tokenizer fallback from APR metadata
// ========================================================================

#[test]
fn test_bug_211_extract_apr_tokenizer_for_gguf_with_vocab() {
    use crate::format::v2::AprV2Metadata;
    let mut meta = AprV2Metadata::default();
    meta.architecture = Some("qwen2".to_string());
    meta.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["hello", "world", "<|im_start|>"]),
    );
    meta.custom.insert(
        "tokenizer.merges".to_string(),
        serde_json::json!(["h e", "l l"]),
    );
    meta.custom
        .insert("tokenizer.bos_token_id".to_string(), serde_json::json!(1));
    meta.custom
        .insert("tokenizer.eos_token_id".to_string(), serde_json::json!(2));

    let entries = extract_apr_tokenizer_for_gguf(&meta);
    // Should have at least: model, pre, tokens, merges, bos, eos
    assert!(
        entries.len() >= 6,
        "Expected >= 6 tokenizer entries, got {}",
        entries.len()
    );

    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"tokenizer.ggml.tokens"));
    assert!(keys.contains(&"tokenizer.ggml.merges"));
    assert!(keys.contains(&"tokenizer.ggml.model"));
    assert!(keys.contains(&"tokenizer.ggml.bos_token_id"));
    assert!(keys.contains(&"tokenizer.ggml.eos_token_id"));
}

#[test]
fn test_bug_211_extract_apr_tokenizer_for_gguf_empty() {
    use crate::format::v2::AprV2Metadata;
    let meta = AprV2Metadata::default();
    let entries = extract_apr_tokenizer_for_gguf(&meta);
    // Should still have model and pre even without vocab
    assert!(entries.len() >= 2);
}

// ========================================================================
// Bug 213: APR metadata → GGUF config round-trip
// ========================================================================

#[test]
fn test_bug_213_resolve_gguf_config_from_apr_metadata() {
    use crate::format::v2::AprV2Metadata;

    let mut meta = AprV2Metadata::default();
    meta.architecture = Some("qwen2".to_string());
    meta.hidden_size = Some(1536);
    meta.num_layers = Some(28);
    meta.num_heads = Some(12);
    meta.num_kv_heads = Some(2);
    meta.vocab_size = Some(151936);
    meta.intermediate_size = Some(8960);
    meta.max_position_embeddings = Some(32768);
    meta.rope_theta = Some(1_000_000.0);
    meta.rms_norm_eps = Some(1e-6);

    let cfg = resolve_gguf_config(Some(&meta), None);

    assert_eq!(cfg.arch, "qwen2");
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.num_layers, 28);
    assert_eq!(cfg.num_heads, 12);
    assert_eq!(cfg.num_kv_heads, 2);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.intermediate_size, 8960);
    assert_eq!(cfg.max_pos, 32768);
    assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9);
}

#[test]
fn test_bug_213_resolve_gguf_config_defaults_without_metadata() {
    let cfg = resolve_gguf_config(None, None);

    // Should use hardcoded defaults
    assert_eq!(cfg.arch, "qwen2");
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_layers, 32);
    assert_eq!(cfg.num_heads, 32);
}
