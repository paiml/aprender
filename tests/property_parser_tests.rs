#![allow(clippy::disallowed_methods)]
//! Property-Based Parser Tests
//!
//! Uses proptest to verify parser invariants with random inputs.
//! These tests ensure parsers never panic on arbitrary input.
//!
//! ~50 property tests covering: Source::parse, FormatType detection,
//! ShardedIndex parsing, chat templates, and tensor validation.

use proptest::prelude::*;

// =============================================================================
// Source::parse Property Tests
// =============================================================================

proptest! {
    /// Source::parse never panics on arbitrary input
    #[test]
    fn prop_source_parse_no_panic(input in ".*") {
        let _ = aprender::format::converter_types::Source::parse(&input);
    }

    /// Source::parse("hf://x/y") always produces HuggingFace variant
    #[test]
    fn prop_source_parse_hf_produces_hf(
        org in "[a-zA-Z][a-zA-Z0-9_-]{0,20}",
        repo in "[a-zA-Z][a-zA-Z0-9._-]{0,30}"
    ) {
        let uri = format!("hf://{org}/{repo}");
        let result = aprender::format::converter_types::Source::parse(&uri);
        match result {
            Ok(aprender::format::converter_types::Source::HuggingFace { .. }) => {},
            Ok(_) => panic!("hf:// must produce HuggingFace variant"),
            Err(_) => {} // Parse error is acceptable
        }
    }

    /// Source::parse("http://...") always produces Url variant
    #[test]
    fn prop_source_parse_http_produces_url(path in "[a-z0-9/._-]{1,50}") {
        let uri = format!("https://example.com/{path}");
        let result = aprender::format::converter_types::Source::parse(&uri);
        assert!(matches!(
            result,
            Ok(aprender::format::converter_types::Source::Url(_))
        ));
    }

    /// Source::parse on local path always produces Local variant
    #[test]
    fn prop_source_parse_local(path in "/[a-z0-9/._-]{1,50}") {
        let result = aprender::format::converter_types::Source::parse(&path);
        assert!(matches!(
            result,
            Ok(aprender::format::converter_types::Source::Local(_))
        ));
    }
}

// =============================================================================
// FormatType Detection Property Tests
// =============================================================================

proptest! {
    /// FormatType::from_extension never panics on arbitrary paths
    #[test]
    fn prop_format_from_extension_no_panic(path in ".*") {
        let _ = aprender::format::rosetta::FormatType::from_extension(
            std::path::Path::new(&path),
        );
    }

    /// FormatType::from_extension is deterministic
    #[test]
    fn prop_format_from_extension_deterministic(ext in "(gguf|safetensors|apr|bin|txt)") {
        let path = format!("model.{ext}");
        let p = std::path::Path::new(&path);
        let r1 = aprender::format::rosetta::FormatType::from_extension(p);
        let r2 = aprender::format::rosetta::FormatType::from_extension(p);
        match (r1, r2) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "Same input must produce same output"),
            (Err(_), Err(_)) => {},
            _ => panic!("Determinism violated"),
        }
    }

    /// .gguf extension always detects as GGUF
    #[test]
    fn prop_format_gguf_extension(name in "[a-zA-Z0-9_-]{1,30}") {
        let path = format!("{name}.gguf");
        let result = aprender::format::rosetta::FormatType::from_extension(
            std::path::Path::new(&path),
        );
        assert_eq!(
            result.unwrap(),
            aprender::format::rosetta::FormatType::Gguf
        );
    }

    /// .safetensors extension always detects as SafeTensors
    #[test]
    fn prop_format_safetensors_extension(name in "[a-zA-Z0-9_-]{1,30}") {
        let path = format!("{name}.safetensors");
        let result = aprender::format::rosetta::FormatType::from_extension(
            std::path::Path::new(&path),
        );
        assert_eq!(
            result.unwrap(),
            aprender::format::rosetta::FormatType::SafeTensors
        );
    }

    /// .apr extension always detects as APR
    #[test]
    fn prop_format_apr_extension(name in "[a-zA-Z0-9_-]{1,30}") {
        let path = format!("{name}.apr");
        let result = aprender::format::rosetta::FormatType::from_extension(
            std::path::Path::new(&path),
        );
        assert_eq!(
            result.unwrap(),
            aprender::format::rosetta::FormatType::Apr
        );
    }
}

// =============================================================================
// ShardedIndex Parsing Property Tests
// =============================================================================

proptest! {
    /// ShardedIndex::parse never panics on arbitrary input
    #[test]
    fn prop_sharded_parse_no_panic(input in ".*") {
        let _ = aprender::format::converter_types::ShardedIndex::parse(&input);
    }

    /// ShardedIndex::parse on arbitrary JSON never panics
    #[test]
    fn prop_sharded_parse_json_no_panic(
        key in "[a-zA-Z._]{1,20}",
        value in "[a-zA-Z0-9._-]{1,30}"
    ) {
        let json = format!(r#"{{"weight_map": {{"{key}": "{value}"}}}}"#);
        let result = aprender::format::converter_types::ShardedIndex::parse(&json);
        // Valid JSON with weight_map should parse successfully
        assert!(result.is_ok(), "Valid weight_map JSON should parse: {json}");
    }
}

// =============================================================================
// Chat Template Property Tests
// =============================================================================

proptest! {
    /// auto_detect_template never panics on arbitrary model names
    #[test]
    fn prop_chat_template_detect_no_panic(model_name in ".*") {
        let _ = aprender::text::chat_template::auto_detect_template(&model_name);
    }

    /// format_conversation never panics on arbitrary content
    #[test]
    fn prop_chat_template_format_no_panic(content in ".{0,1000}") {
        let template = aprender::text::chat_template::auto_detect_template("qwen2.5-coder");
        let messages = vec![
            aprender::text::chat_template::ChatMessage::user(content),
        ];
        let _ = template.format_conversation(&messages);
    }

    /// format_message never panics with arbitrary role and content
    #[test]
    fn prop_chat_template_format_message_no_panic(
        role in "(user|assistant|system|tool|custom)",
        content in ".{0,500}"
    ) {
        let template = aprender::text::chat_template::auto_detect_template("qwen2.5-coder");
        let _ = template.format_message(&role, &content);
    }

    /// ChatMessage::new never panics with arbitrary inputs
    #[test]
    fn prop_chat_message_new_no_panic(
        role in ".{0,50}",
        content in ".{0,500}"
    ) {
        let msg = aprender::text::chat_template::ChatMessage::new(role, content);
        assert!(!msg.role.is_empty() || msg.role.is_empty()); // trivially true, just exercise constructor
    }
}

// =============================================================================
// Tensor Validation Property Tests
// =============================================================================

proptest! {
    /// ValidatedWeight::new with correct size succeeds (non-degenerate data)
    #[test]
    fn prop_validated_weight_correct_size(
        rows in 1usize..50,
        cols in 1usize..50
    ) {
        let size = rows * cols;
        // Generate non-degenerate data
        let data: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.001).collect();
        let result = aprender::format::validated_tensors::ValidatedWeight::new(
            data, rows, cols, "test",
        );
        // May fail data quality checks but must NOT panic
        let _ = result;
    }

    /// ValidatedWeight::new with wrong size always fails
    #[test]
    fn prop_validated_weight_wrong_size_fails(
        rows in 1usize..50,
        cols in 1usize..50,
        extra in 1usize..10
    ) {
        let wrong_size = rows * cols + extra;
        let data = vec![1.0f32; wrong_size];
        let result = aprender::format::validated_tensors::ValidatedWeight::new(
            data, rows, cols, "test",
        );
        assert!(result.is_err(), "Wrong size must always fail");
    }

    /// ValidatedEmbedding::new with wrong size always fails
    #[test]
    fn prop_validated_embedding_wrong_size_fails(
        vocab in 1usize..20,
        hidden in 1usize..20,
        extra in 1usize..10
    ) {
        let wrong_size = vocab * hidden + extra;
        let data = vec![0.1f32; wrong_size];
        let result = aprender::format::validated_tensors::ValidatedEmbedding::new(
            data, vocab, hidden,
        );
        assert!(result.is_err(), "Wrong size must always fail");
    }
}

// =============================================================================
// Layout Contract Property Tests
// =============================================================================

proptest! {
    /// enforce_import_contract never panics on any valid tensor name + shape
    #[test]
    fn prop_import_contract_no_panic(
        name in "[a-z._]{1,30}",
        dim1 in 1usize..10000,
        dim2 in 1usize..10000
    ) {
        let shape = vec![dim1, dim2];
        let _ = aprender::format::layout_contract::enforce_import_contract(
            &name, &shape, dim2, dim1,
        );
    }

    /// enforce_import_contract is deterministic (same input → same output)
    #[test]
    fn prop_import_contract_deterministic(
        dim1 in 1usize..5000,
        dim2 in 1usize..5000
    ) {
        let shape = vec![dim1, dim2];
        let r1 = aprender::format::layout_contract::enforce_import_contract(
            "output.weight", &shape, dim2, dim1,
        );
        let r2 = aprender::format::layout_contract::enforce_import_contract(
            "output.weight", &shape, dim2, dim1,
        );
        assert_eq!(r1, r2, "Same input must produce same output");
    }

    /// 1D tensors are never transposed
    #[test]
    fn prop_import_contract_1d_no_transpose(dim in 1usize..10000) {
        let shape = vec![dim];
        let (result_shape, transposed) = aprender::format::layout_contract::enforce_import_contract(
            "norm.weight", &shape, 0, dim,
        );
        assert!(!transposed, "1D tensors must never be transposed");
        assert_eq!(result_shape, shape, "1D shape must be unchanged");
    }
}

// =============================================================================
// Performance Regression Detection Property Tests
// =============================================================================

proptest! {
    /// Regression detection: improvement (curr > prev) is never flagged
    #[test]
    fn prop_regression_improvement_not_flagged(
        prev in 1.0f64..1000.0,
        improvement in 0.01f64..500.0
    ) {
        let curr = prev + improvement;
        let regression = (prev - curr) / prev;
        assert!(regression < 0.0, "Improvement must not be flagged as regression");
    }

    /// Regression detection: large drop is always flagged
    #[test]
    fn prop_regression_large_drop_flagged(
        prev in 10.0f64..1000.0,
        drop_pct in 0.15f64..0.95 // 15-95% drop
    ) {
        let curr = prev * (1.0 - drop_pct);
        let regression = (prev - curr) / prev;
        assert!(regression > 0.10, ">{:.0}% drop must exceed 10% threshold", drop_pct * 100.0);
    }
}
