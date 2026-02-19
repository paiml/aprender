
#[test]
fn edge_safetensors_header_size_exceeds_file() {
    let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp");
    let header_len: u64 = 1_000_000_000;
    std::io::Write::write_all(&mut file, &header_len.to_le_bytes()).unwrap();
    std::io::Write::write_all(&mut file, b"{\"test\"").unwrap();
    // Must not panic
    let _ = aprender::format::rosetta::FormatType::from_magic(file.path());
}

#[test]
fn edge_safetensors_header_zero_length() {
    let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp");
    let header_len: u64 = 0;
    std::io::Write::write_all(&mut file, &header_len.to_le_bytes()).unwrap();
    let _ = aprender::format::rosetta::FormatType::from_magic(file.path());
}

#[test]
fn edge_safetensors_valid_magic_pattern() {
    let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp");
    let header_len: u64 = 2;
    std::io::Write::write_all(&mut file, &header_len.to_le_bytes()).unwrap();
    std::io::Write::write_all(&mut file, b"{}").unwrap();
    // SafeTensors has no magic bytes — from_magic can't detect it
    // It must use from_extension instead
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    // Even with .safetensors extension, magic detection may not identify it
    // (SafeTensors starts with u64 header length, not a fixed magic)
    let by_ext = aprender::format::rosetta::FormatType::from_extension(file.path());
    assert_eq!(
        by_ext.unwrap(),
        aprender::format::rosetta::FormatType::SafeTensors
    );
    // from_magic should at least not panic
    let _ = result;
}

// =============================================================================
// APR Format Edge Cases (~4 tests)
// =============================================================================

#[test]
fn edge_apr_empty_file() {
    let file = NamedTempFile::with_suffix(".apr").expect("create temp");
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert!(result.is_err(), "Empty file must fail APR detection");
}

#[test]
fn edge_apr_extension_detection() {
    let result = aprender::format::rosetta::FormatType::from_extension(Path::new("model.apr"));
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Apr);
}

#[test]
fn edge_apr_nonexistent_path() {
    let result = aprender::format::rosetta::FormatType::from_magic(Path::new(
        "/nonexistent/path/to/model.apr",
    ));
    assert!(result.is_err(), "Non-existent path must return error");
}

// =============================================================================
// ShardedIndex Parsing Edge Cases (~10 tests)
// =============================================================================

#[test]
fn edge_sharded_empty_string() {
    let result = aprender::format::converter_types::ShardedIndex::parse("");
    assert!(result.is_err(), "Empty string must fail ShardedIndex parse");
}

#[test]
fn edge_sharded_empty_json_object() {
    let result = aprender::format::converter_types::ShardedIndex::parse("{}");
    assert!(
        result.is_err(),
        "Empty JSON object (no weight_map) must fail"
    );
}

#[test]
fn edge_sharded_valid_minimal() {
    let json = r#"{"weight_map": {"layer.weight": "shard-00001.safetensors"}}"#;
    let result = aprender::format::converter_types::ShardedIndex::parse(json);
    assert!(result.is_ok(), "Valid minimal ShardedIndex must parse");
}

#[test]
fn edge_sharded_empty_weight_map() {
    let json = r#"{"weight_map": {}}"#;
    let result = aprender::format::converter_types::ShardedIndex::parse(json);
    assert!(
        result.is_ok(),
        "Empty weight_map should parse (just no tensors)"
    );
}

#[test]
fn edge_sharded_multiple_shards() {
    let json = r#"{
        "metadata": {"total_size": 14000000000},
        "weight_map": {
            "model.encoder.weight": "model-00001-of-00002.safetensors",
            "model.decoder.weight": "model-00002-of-00002.safetensors"
        }
    }"#;
    let idx = aprender::format::converter_types::ShardedIndex::parse(json).unwrap();
    let files = idx.shard_files();
    assert_eq!(files.len(), 2, "Should have 2 unique shard files");
}

#[test]
fn edge_sharded_not_json() {
    let result = aprender::format::converter_types::ShardedIndex::parse("not json at all");
    assert!(result.is_err(), "Non-JSON must fail ShardedIndex parse");
}

#[test]
fn edge_sharded_json_array() {
    let result = aprender::format::converter_types::ShardedIndex::parse("[1, 2, 3]");
    assert!(result.is_err(), "JSON array must fail ShardedIndex parse");
}

#[test]
fn edge_sharded_nested_json() {
    let json = r#"{"weight_map": {"a": "s1.safetensors"}, "nested": {"weight_map": {"b": "s2.safetensors"}}}"#;
    let result = aprender::format::converter_types::ShardedIndex::parse(json);
    assert!(result.is_ok());
}

#[test]
fn edge_sharded_unicode_tensor_names() {
    let json = r#"{"weight_map": {"模型.重量": "shard.safetensors"}}"#;
    // Unicode tensor names should not cause panics
    let _ = aprender::format::converter_types::ShardedIndex::parse(json);
}

// =============================================================================
// Chat Template Edge Cases (~8 tests)
// =============================================================================

#[test]
fn edge_chat_template_auto_detect_empty_string() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("");
    let messages = vec![ChatMessage::user("Hello")];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "Template should handle empty model name");
}

#[test]
fn edge_chat_template_auto_detect_unknown_model() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("nonexistent-model-xyz");
    let messages = vec![ChatMessage::user("Test")];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "Unknown model should use default template");
}

#[test]
fn edge_chat_template_empty_messages() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen");
    let messages: Vec<ChatMessage> = vec![];
    let result = template.format_conversation(&messages);
    // Empty messages should produce something valid (or clean error), not panic
    let _ = result;
}

#[test]
fn edge_chat_template_empty_content() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen");
    let messages = vec![ChatMessage::user("")];
    let result = template.format_conversation(&messages);
    assert!(
        result.is_ok(),
        "Empty content message should not cause panic"
    );
}

#[test]
fn edge_chat_template_only_system_message() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen");
    let messages = vec![ChatMessage::system("You are a helpful assistant.")];
    let result = template.format_conversation(&messages);
    assert!(
        result.is_ok(),
        "System-only messages should produce valid output"
    );
}

#[test]
fn edge_chat_template_consecutive_user_messages() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen");
    let messages = vec![
        ChatMessage::user("First message"),
        ChatMessage::user("Second message"),
    ];
    let result = template.format_conversation(&messages);
    // Multiple consecutive user messages should not panic
    let _ = result;
}

#[test]
fn edge_chat_template_large_content() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen");
    let large_content = "x".repeat(100_000);
    let messages = vec![ChatMessage::user(large_content)];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
}

#[test]
fn edge_chat_template_qwen_format_markers() {
    use aprender::text::chat_template::{auto_detect_template, ChatMessage};
    let template = auto_detect_template("qwen2.5-coder");
    let messages = vec![ChatMessage::user("Hello")];
    let result = template.format_conversation(&messages).unwrap();
    assert!(
        result.contains("<|im_start|>"),
        "Qwen template must contain <|im_start|>"
    );
    assert!(
        result.contains("<|im_end|>"),
        "Qwen template must contain <|im_end|>"
    );
}

// =============================================================================
// Layout Contract Edge Cases (~5 tests)
// =============================================================================

#[test]
fn edge_layout_contract_1d_tensor_no_transpose() {
    use aprender::format::layout_contract::enforce_import_contract;
    let result = std::panic::catch_unwind(|| {
        let _ = enforce_import_contract("output_norm.weight", &[1536], 151936, 1536);
    });
    assert!(result.is_ok(), "1D tensor should not panic in contract");
}

#[test]
fn edge_layout_contract_2d_tensor_transpose() {
    use aprender::format::layout_contract::enforce_import_contract;
    let result = std::panic::catch_unwind(|| {
        let _ = enforce_import_contract("output.weight", &[1536, 151936], 151936, 1536);
    });
    assert!(result.is_ok(), "2D tensor contract should not panic");
}

#[test]
fn edge_layout_contract_embedding() {
    use aprender::format::layout_contract::enforce_embedding_contract;
    let result = std::panic::catch_unwind(|| {
        enforce_embedding_contract(151936 * 1536, 151936, 1536);
    });
    assert!(result.is_ok(), "Embedding contract should not panic");
}

#[test]
fn edge_layout_contract_matmul() {
    use aprender::format::layout_contract::enforce_matmul_contract;
    let result = std::panic::catch_unwind(|| {
        enforce_matmul_contract("test.weight", &[4096, 1536], 4096, 1536);
    });
    assert!(result.is_ok(), "Matmul contract should not panic");
}

// =============================================================================
// Model Family Detection Edge Cases (~3 tests)
// =============================================================================

#[test]
fn edge_model_family_known_families_not_empty() {
    assert!(
        !aprender::format::model_family::KNOWN_FAMILIES.is_empty(),
        "KNOWN_FAMILIES must not be empty"
    );
}

#[test]
fn edge_model_family_registry_has_qwen() {
    let registry = aprender::format::model_family::build_default_registry();
    let names = registry.family_names();
    let found = names.iter().any(|n| n.contains("qwen2"));
    assert!(found, "Registry must contain qwen2 family");
}

#[test]
fn edge_model_family_registry_unique_names() {
    let registry = aprender::format::model_family::build_default_registry();
    let names = registry.family_names();
    let count = names.len();
    let mut unique = names.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        count,
        unique.len(),
        "Registry must have unique family names"
    );
}

// =============================================================================
// Source Parsing Edge Cases (~5 tests)
// =============================================================================

#[test]
fn edge_source_parse_empty() {
    let result = aprender::format::converter_types::Source::parse("");
    // Empty string becomes a local path, not panic
    assert!(result.is_ok(), "Empty string should parse as local path");
}

#[test]
fn edge_source_parse_hf_prefix() {
    let result = aprender::format::converter_types::Source::parse("hf://Qwen/model");
    assert!(result.is_ok(), "Valid hf:// prefix should parse");
    match result.unwrap() {
        aprender::format::converter_types::Source::HuggingFace { org, repo, .. } => {
            assert_eq!(org, "Qwen");
            assert_eq!(repo, "model");
        }
        _ => panic!("Expected HuggingFace variant"),
    }
}

#[test]
fn edge_source_parse_local_path() {
    let result = aprender::format::converter_types::Source::parse("/path/to/model.safetensors");
    assert!(result.is_ok(), "Local path should parse");
    assert!(matches!(
        result.unwrap(),
        aprender::format::converter_types::Source::Local(_)
    ));
}

#[test]
fn edge_source_parse_https_url() {
    let result = aprender::format::converter_types::Source::parse("https://example.com/model.gguf");
    assert!(result.is_ok(), "HTTPS URL should parse");
    assert!(matches!(
        result.unwrap(),
        aprender::format::converter_types::Source::Url(_)
    ));
}

#[test]
fn edge_source_parse_hf_invalid() {
    let result = aprender::format::converter_types::Source::parse("hf://only-one-part");
    assert!(
        result.is_err(),
        "hf:// with only one path segment must fail"
    );
}

#[test]
fn edge_source_parse_hf_with_file() {
    let result = aprender::format::converter_types::Source::parse(
        "hf://Qwen/Qwen2.5-Coder-0.5B/model.safetensors",
    );
    assert!(result.is_ok());
    match result.unwrap() {
        aprender::format::converter_types::Source::HuggingFace { org, repo, file } => {
            assert_eq!(org, "Qwen");
            assert_eq!(repo, "Qwen2.5-Coder-0.5B");
            assert!(file.is_some());
        }
        _ => panic!("Expected HuggingFace variant"),
    }
}
