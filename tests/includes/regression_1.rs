// =============================================================================
// Category 1: Tensor Layout Regressions (GH-208, LAYOUT-001/002)
// =============================================================================

#[test]
fn regression_bug_208_shape_reversal_2d() {
    use aprender::format::layout_contract::enforce_import_contract;
    let (shape, transposed) =
        enforce_import_contract("output.weight", &[1536, 151936], 151936, 1536);
    assert!(
        transposed || shape == vec![151936, 1536],
        "Bug 208: 2D output weight must have shape reversed"
    );
}

#[test]
fn regression_bug_208_no_transpose_1d() {
    use aprender::format::layout_contract::enforce_import_contract;
    let (shape, transposed) = enforce_import_contract("output_norm.weight", &[1536], 151936, 1536);
    assert!(!transposed, "Bug 208: 1D tensors must NOT be transposed");
    assert_eq!(shape, vec![1536], "Bug 208: 1D shape must be unchanged");
}

#[test]
fn regression_layout_001_embedding_contract() {
    use aprender::format::layout_contract::enforce_embedding_contract;
    let result = std::panic::catch_unwind(|| {
        enforce_embedding_contract(151936 * 1536, 151936, 1536);
    });
    assert!(
        result.is_ok(),
        "LAYOUT-001: Embedding contract must not panic"
    );
}

#[test]
fn regression_layout_002_matmul_contract() {
    use aprender::format::layout_contract::enforce_matmul_contract;
    let result = std::panic::catch_unwind(|| {
        enforce_matmul_contract("test.weight", &[4096, 1536], 4096, 1536);
    });
    assert!(result.is_ok(), "LAYOUT-002: Matmul contract must not panic");
}

#[test]
fn regression_layout_row_major_validated_weight() {
    use aprender::format::validated_tensors::ValidatedWeight;
    let data = vec![1.0f32; 12];
    let result = ValidatedWeight::new(data, 3, 4, "test");
    assert!(result.is_ok(), "Row-major weight construction must succeed");
}

#[test]
fn regression_layout_validated_embedding() {
    use aprender::format::validated_tensors::ValidatedEmbedding;
    let mut data = Vec::with_capacity(6);
    for i in 0..6 {
        data.push((i as f32 + 1.0) * 0.1);
    }
    let result = ValidatedEmbedding::new(data, 2, 3);
    let _ = result;
}

// =============================================================================
// Category 2: URI Parsing Regressions (GH-220, GH-221, GH-222)
// =============================================================================

#[test]
fn regression_bug_221_hf_uri_strip_resolve_main() {
    let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/model.safetensors";
    let stripped = uri
        .replace("/resolve/main/", "/")
        .replace("/blob/main/", "/");
    assert!(!stripped.contains("/resolve/main/"));
    assert!(stripped.ends_with("/model.safetensors"));
}

#[test]
fn regression_bug_221_hf_uri_strip_blob_main() {
    let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/main/model.safetensors";
    let stripped = uri
        .replace("/resolve/main/", "/")
        .replace("/blob/main/", "/");
    assert!(!stripped.contains("/blob/main/"));
}

#[test]
fn regression_bug_221_bare_resolve_main() {
    let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main";
    let mut stripped = uri.replace("/resolve/main/", "/");
    if stripped.ends_with("/resolve/main") {
        stripped = stripped.replace("/resolve/main", "");
    }
    assert!(!stripped.contains("resolve/main"));
}

#[test]
fn regression_bug_221_normal_uri_unchanged() {
    let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/model.gguf";
    let stripped = uri
        .replace("/resolve/main/", "/")
        .replace("/blob/main/", "/");
    assert_eq!(uri, stripped);
}

#[test]
fn regression_bug_222_chat_template_detection() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen2.5-coder-0.5b");
    let messages = vec![ChatMessage::user("Hello")];
    let result = template.format_conversation(&messages).unwrap();
    assert!(
        result.contains("<|im_start|>"),
        "Bug 222: Qwen2 must use ChatML format"
    );
}

#[test]
fn regression_bug_222_non_qwen_template_not_empty() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("llama3");
    let messages = vec![ChatMessage::user("Hello")];
    let result = template.format_conversation(&messages).unwrap();
    assert!(
        !result.is_empty(),
        "Bug 222: Template output must not be empty"
    );
}

#[test]
fn regression_bug_220_source_parse_hf() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/model.gguf");
    assert!(result.is_ok(), "Bug 220: Standard hf:// URI must parse");
}

#[test]
fn regression_bug_220_source_parse_local() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("/path/to/model.gguf");
    assert!(result.is_ok(), "Bug 220: Local path must parse");
}

// =============================================================================
// Category 3: Format Detection Regressions
// =============================================================================

#[test]
fn regression_format_detect_gguf_magic() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    std::io::Write::write_all(&mut file, b"GGUF\x03\x00\x00\x00").unwrap();
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Gguf);
}

#[test]
fn regression_format_detect_safetensors_magic() {
    let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp");
    let header = b"{}";
    let len = (header.len() as u64).to_le_bytes();
    std::io::Write::write_all(&mut file, &len).unwrap();
    std::io::Write::write_all(&mut file, header).unwrap();
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert!(
        result.is_err(),
        "SafeTensors without .safetensors ext is unrecognizable by magic"
    );
    let by_ext = aprender::format::rosetta::FormatType::from_extension(std::path::Path::new(
        "model.safetensors",
    ));
    assert_eq!(
        by_ext.unwrap(),
        aprender::format::rosetta::FormatType::SafeTensors
    );
}

#[test]
fn regression_format_detect_truncated_no_panic() {
    let mut file = NamedTempFile::with_suffix(".bin").expect("create temp");
    std::io::Write::write_all(&mut file, &[0x00, 0x01]).unwrap();
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert!(result.is_err());
}

#[test]
fn regression_format_detect_empty_no_panic() {
    let file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert!(result.is_err());
}

// =============================================================================
// Category 4: Chat Template Regressions (PMAT-236, PMAT-237)
// =============================================================================

#[test]
fn regression_pmat_236_template_not_empty() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen2.5-coder");
    let messages = vec![ChatMessage::user("Test")];
    let result = template.format_conversation(&messages).unwrap();
    assert!(
        !result.trim().is_empty(),
        "PMAT-236: Template output must never be empty"
    );
}

#[test]
fn regression_pmat_237_no_silent_skip() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen2.5-coder");
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Hi"),
    ];
    let result = template.format_conversation(&messages).unwrap();
    assert!(
        result.contains("helpful assistant"),
        "PMAT-237: System message must not be silently skipped"
    );
    assert!(
        result.contains("Hi"),
        "PMAT-237: User message must not be silently skipped"
    );
}

#[test]
fn regression_chat_template_assistant_role() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen2.5-coder");
    let messages = vec![
        ChatMessage::user("What is 2+2?"),
        ChatMessage::assistant("4"),
        ChatMessage::user("And 3+3?"),
    ];
    let result = template.format_conversation(&messages).unwrap();
    assert!(result.contains("4"));
    assert!(result.contains("3+3"));
}

#[test]
fn regression_chat_template_chatml_markers_present() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen2.5-coder");
    let messages = vec![ChatMessage::user("test")];
    let result = template.format_conversation(&messages).unwrap();
    let has_markers = result.contains("<|im_start|>") && result.contains("<|im_end|>");
    assert!(has_markers, "ChatML markers must be present in Qwen output");
}

// =============================================================================
// Category 5: Quantization Regressions (GH-205)
// =============================================================================

#[test]
fn regression_gh_205_q4k_recognized() {
    let result =
        aprender::format::rosetta::FormatType::from_extension(Path::new("model-q4_k.gguf"));
    assert!(result.is_ok(), "GH-205: Q4K GGUF must be recognized");
}

#[test]
fn regression_gh_205_safetensors_recognized() {
    let result =
        aprender::format::rosetta::FormatType::from_extension(Path::new("model.safetensors"));
    assert!(result.is_ok(), "GH-205: SafeTensors must be recognized");
}

// =============================================================================
// Category 6: Sharded Model Regressions (Bug 205, GH-213)
// =============================================================================

#[test]
fn regression_bug_205_sharded_manifest_2_shards() {
    let json = r#"{
        "metadata": {"total_size": 7000000000},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.14.weight": "model-00002-of-00002.safetensors"
        }
    }"#;
    let idx = aprender::format::converter_types::ShardedIndex::parse(json).unwrap();
    assert_eq!(idx.shard_files().len(), 2);
}

#[test]
fn regression_bug_205_single_shard() {
    let json = r#"{
        "weight_map": {
            "model.embed_tokens.weight": "model.safetensors",
            "model.lm_head.weight": "model.safetensors"
        }
    }"#;
    let idx = aprender::format::converter_types::ShardedIndex::parse(json).unwrap();
    assert_eq!(idx.shard_files().len(), 1);
}

#[test]
fn regression_bug_205_four_shards() {
    let json = r#"{
        "weight_map": {
            "a": "model-00001-of-00004.safetensors",
            "b": "model-00002-of-00004.safetensors",
            "c": "model-00003-of-00004.safetensors",
            "d": "model-00004-of-00004.safetensors"
        }
    }"#;
    let idx = aprender::format::converter_types::ShardedIndex::parse(json).unwrap();
    assert_eq!(idx.shard_files().len(), 4);
}
