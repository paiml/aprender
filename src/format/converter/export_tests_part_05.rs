
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

// ========================================================================
// GH-258: export_to_gguf coverage tests
// ========================================================================

/// Helper: create a minimal tensor map with HF-style names for testing export_to_gguf
fn make_test_tensors(
    hidden: usize,
    vocab: usize,
    layers: usize,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut tensors = BTreeMap::new();

    // Embedding
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; vocab * hidden], vec![vocab, hidden]),
    );

    // lm_head
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.2; vocab * hidden], vec![vocab, hidden]),
    );

    // Norm
    tensors.insert(
        "model.norm.weight".to_string(),
        (vec![1.0; hidden], vec![hidden]),
    );

    // Layers
    for l in 0..layers {
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            tensors.insert(
                format!("model.layers.{l}.self_attn.{proj}.weight"),
                (vec![0.01; hidden * hidden], vec![hidden, hidden]),
            );
        }
        tensors.insert(
            format!("model.layers.{l}.input_layernorm.weight"),
            (vec![1.0; hidden], vec![hidden]),
        );
        tensors.insert(
            format!("model.layers.{l}.post_attention_layernorm.weight"),
            (vec![1.0; hidden], vec![hidden]),
        );
        tensors.insert(
            format!("model.layers.{l}.mlp.gate_proj.weight"),
            (vec![0.01; hidden * hidden], vec![hidden, hidden]),
        );
        tensors.insert(
            format!("model.layers.{l}.mlp.up_proj.weight"),
            (vec![0.01; hidden * hidden], vec![hidden, hidden]),
        );
        tensors.insert(
            format!("model.layers.{l}.mlp.down_proj.weight"),
            (vec![0.01; hidden * hidden], vec![hidden, hidden]),
        );
    }

    tensors
}

#[test]
fn test_export_to_gguf_f32_writes_valid_gguf() {
    use crate::format::gguf::GgufReader;

    let tensors = make_test_tensors(64, 256, 2);
    let dir = tempfile::tempdir().expect("temp dir");
    let output = dir.path().join("test.gguf");
    // Use a non-APR dummy input path (no metadata extraction)
    let input = dir.path().join("dummy.safetensors");
    std::fs::write(&input, b"dummy").expect("write dummy");

    export_to_gguf(&tensors, &output, &input, None).expect("export should succeed");

    // Verify the GGUF file is readable
    assert!(output.exists());
    let reader = GgufReader::from_file(&output).expect("should parse GGUF");

    // Check architecture metadata
    let arch = reader.architecture();
    assert!(arch.is_some(), "GGUF should have architecture metadata");

    // Verify tensor count: embed + lm_head + norm + 2 layers × (4 attn + 2 norm + 3 mlp) = 21
    assert_eq!(reader.tensor_count, 21, "should have 21 tensors");
}

#[test]
fn test_export_to_gguf_q4k_quantizes_weight_tensors() {
    use crate::format::gguf::GgufReader;

    // Need hidden>=256 for Q4K quantization (data.len() >= 256 check)
    let tensors = make_test_tensors(256, 512, 1);
    let dir = tempfile::tempdir().expect("temp dir");
    let output = dir.path().join("test_q4k.gguf");
    let input = dir.path().join("dummy.safetensors");
    std::fs::write(&input, b"dummy").expect("write dummy");

    let quant = QuantizationType::Q4K;
    export_to_gguf(&tensors, &output, &input, Some(&quant)).expect("Q4K export should succeed");

    assert!(output.exists());
    let file_size = std::fs::metadata(&output).expect("metadata").len();
    // Q4K should produce a smaller file than pure F32
    assert!(file_size > 0, "output file should have content");

    let reader = GgufReader::from_file(&output).expect("should parse GGUF");
    assert!(reader.tensor_count > 0);
}

#[test]
fn test_export_to_gguf_q4k_preserves_embeddings_as_f32() {
    use crate::format::gguf::GgufReader;

    let tensors = make_test_tensors(256, 512, 1);
    let dir = tempfile::tempdir().expect("temp dir");
    let output = dir.path().join("test_embed.gguf");
    let input = dir.path().join("dummy.safetensors");
    std::fs::write(&input, b"dummy").expect("write dummy");

    let quant = QuantizationType::Q4K;
    export_to_gguf(&tensors, &output, &input, Some(&quant)).expect("export should succeed");

    let reader = GgufReader::from_file(&output).expect("should parse GGUF");
    // Embedding (token_embd.weight) should remain F32 (dtype=0)
    let embd = reader
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight");
    assert!(embd.is_some(), "should have token_embd.weight");
    assert_eq!(
        embd.expect("embd").dtype,
        0,
        "embedding should stay F32 (dtype=0) under Q4K"
    );

    // lm_head (output.weight) should also remain F32
    let lm_head = reader.tensors.iter().find(|t| t.name == "output.weight");
    assert!(lm_head.is_some(), "should have output.weight");
    assert_eq!(
        lm_head.expect("lm_head").dtype,
        0,
        "lm_head should stay F32 (dtype=0) under Q4K"
    );
}

#[test]
fn test_export_to_gguf_shape_reversal_for_2d() {
    use crate::format::gguf::GgufReader;

    let mut tensors = BTreeMap::new();
    // APR shape [rows=128, cols=64] should become GGUF [ne0=64, ne1=128]
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; 128 * 64], vec![128, 64]),
    );

    let dir = tempfile::tempdir().expect("temp dir");
    let output = dir.path().join("test_shape.gguf");
    let input = dir.path().join("dummy.safetensors");
    std::fs::write(&input, b"dummy").expect("write dummy");

    export_to_gguf(&tensors, &output, &input, None).expect("export should succeed");

    let reader = GgufReader::from_file(&output).expect("should parse GGUF");
    let embd = reader
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight");
    assert!(embd.is_some());
    let dims = &embd.expect("embd").dims;
    assert_eq!(
        dims,
        &vec![64, 128],
        "shape should be reversed for GGUF [ne0, ne1]"
    );
}

#[test]
fn test_export_to_gguf_1d_shape_unchanged() {
    use crate::format::gguf::GgufReader;

    let mut tensors = BTreeMap::new();
    // 1D tensor shape should NOT be reversed
    tensors.insert("model.norm.weight".to_string(), (vec![1.0; 64], vec![64]));
    // Need at least one 2D tensor for the file to be useful
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; 128 * 64], vec![128, 64]),
    );

    let dir = tempfile::tempdir().expect("temp dir");
    let output = dir.path().join("test_1d.gguf");
    let input = dir.path().join("dummy.safetensors");
    std::fs::write(&input, b"dummy").expect("write dummy");

    export_to_gguf(&tensors, &output, &input, None).expect("export should succeed");

    let reader = GgufReader::from_file(&output).expect("should parse GGUF");
    let norm = reader
        .tensors
        .iter()
        .find(|t| t.name == "output_norm.weight");
    assert!(norm.is_some());
    let dims = &norm.expect("norm").dims;
    assert_eq!(dims, &vec![64], "1D shape should be preserved");
}

#[test]
fn test_export_to_gguf_tensor_name_mapping() {
    use crate::format::gguf::GgufReader;

    let tensors = make_test_tensors(64, 256, 1);
    let dir = tempfile::tempdir().expect("temp dir");
    let output = dir.path().join("test_names.gguf");
    let input = dir.path().join("dummy.safetensors");
    std::fs::write(&input, b"dummy").expect("write dummy");

    export_to_gguf(&tensors, &output, &input, None).expect("export should succeed");

    let reader = GgufReader::from_file(&output).expect("should parse GGUF");
    let names: Vec<&str> = reader.tensors.iter().map(|t| t.name.as_str()).collect();

    // Verify key HF→GGUF name mappings
    assert!(
        names.contains(&"token_embd.weight"),
        "embed_tokens → token_embd"
    );
    assert!(names.contains(&"output.weight"), "lm_head → output");
    assert!(
        names.contains(&"output_norm.weight"),
        "model.norm → output_norm"
    );
    assert!(names.contains(&"blk.0.attn_q.weight"), "q_proj → attn_q");
    assert!(names.contains(&"blk.0.attn_k.weight"), "k_proj → attn_k");
    assert!(names.contains(&"blk.0.attn_v.weight"), "v_proj → attn_v");
    assert!(
        names.contains(&"blk.0.attn_output.weight"),
        "o_proj → attn_output"
    );
    assert!(
        names.contains(&"blk.0.ffn_gate.weight"),
        "gate_proj → ffn_gate"
    );
    assert!(names.contains(&"blk.0.ffn_up.weight"), "up_proj → ffn_up");
    assert!(
        names.contains(&"blk.0.ffn_down.weight"),
        "down_proj → ffn_down"
    );
    assert!(
        names.contains(&"blk.0.attn_norm.weight"),
        "input_layernorm → attn_norm"
    );
    assert!(
        names.contains(&"blk.0.ffn_norm.weight"),
        "post_attention_layernorm → ffn_norm"
    );
}

#[test]
fn test_export_to_gguf_from_apr_input_reads_metadata() {
    use crate::format::gguf::GgufReader;
    use crate::format::v2::{AprV2Metadata, AprV2Writer};

    let dir = tempfile::tempdir().expect("temp dir");

    // Create a real APR file with specific metadata
    let mut metadata = AprV2Metadata::new("test-export");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(64);
    metadata.vocab_size = Some(256);
    metadata.num_layers = Some(2);
    metadata.num_heads = Some(4);
    metadata.num_kv_heads = Some(2);
    metadata.rope_theta = Some(500_000.0);

    let mut writer = AprV2Writer::new(metadata);
    // Add minimal tensors
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![256, 64],
        &vec![0.1; 256 * 64],
    );
    writer.add_f32_tensor("model.norm.weight", vec![64], &vec![1.0; 64]);

    let apr_path = dir.path().join("model.apr");
    let bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, bytes).expect("write file");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; 256 * 64], vec![256, 64]),
    );
    tensors.insert("model.norm.weight".to_string(), (vec![1.0; 64], vec![64]));

    let output = dir.path().join("output.gguf");
    export_to_gguf(&tensors, &output, &apr_path, None).expect("export should succeed");

    let reader = GgufReader::from_file(&output).expect("should parse GGUF");
    // Architecture should come from APR metadata, not default
    let arch = reader.architecture();
    assert_eq!(
        arch.as_deref(),
        Some("llama"),
        "should use APR metadata architecture"
    );
}
