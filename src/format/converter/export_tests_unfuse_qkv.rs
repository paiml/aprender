use super::*;

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

    // C-16 (Meyer DbC): 0 = unknown when nothing is inferable
    assert_eq!(v["hidden_size"], 0);
    assert_eq!(v["num_hidden_layers"], 12);
    assert_eq!(v["vocab_size"], 0);
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

    // C-16 (Meyer DbC): hidden_size 0 = unknown (no embed_tokens/token_embd found)
    assert_eq!(v["hidden_size"], 0);
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
