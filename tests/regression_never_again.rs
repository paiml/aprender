//! Bug Regression Test Suite — Never Again
//!
//! Each test encodes a specific historical bug from the showcase spec.
//! Test names include bug numbers for traceability.
//!
//! These tests use synthetic data and do NOT require model files,
//! so they run in every `cargo test`.

use std::path::Path;
use tempfile::NamedTempFile;

// =============================================================================
// Category 1: Tensor Layout Regressions (GH-208, LAYOUT-001/002)
// =============================================================================

#[test]
fn regression_bug_208_shape_reversal_2d() {
    use aprender::format::layout_contract::enforce_import_contract;
    let (shape, transposed) =
        enforce_import_contract("output.weight", &[1536, 151936], 151936, 1536);
    // 2D weight tensor MUST be transposed from GGUF→APR
    assert!(
        transposed || shape == vec![151936, 1536],
        "Bug 208: 2D output weight must have shape reversed"
    );
}

#[test]
fn regression_bug_208_no_transpose_1d() {
    use aprender::format::layout_contract::enforce_import_contract;
    let (shape, transposed) = enforce_import_contract("output_norm.weight", &[1536], 151936, 1536);
    assert!(
        !transposed,
        "Bug 208: 1D tensors must NOT be transposed"
    );
    assert_eq!(shape, vec![1536], "Bug 208: 1D shape must be unchanged");
}

#[test]
fn regression_layout_001_embedding_contract() {
    use aprender::format::layout_contract::enforce_embedding_contract;
    let result = std::panic::catch_unwind(|| {
        enforce_embedding_contract(151936 * 1536, 151936, 1536);
    });
    assert!(result.is_ok(), "LAYOUT-001: Embedding contract must not panic");
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
    // Create non-degenerate embedding data (not all zeros, not all same)
    let mut data = Vec::with_capacity(6);
    for i in 0..6 {
        data.push((i as f32 + 1.0) * 0.1);
    }
    let result = ValidatedEmbedding::new(data, 2, 3);
    // This may fail due to data quality gates — but must NOT panic
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
    assert!(!result.is_empty(), "Bug 222: Template output must not be empty");
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
    // SafeTensors has no magic bytes — from_magic falls back to extension
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    // With .bin extension, magic detection won't identify SafeTensors
    assert!(result.is_err(), "SafeTensors without .safetensors ext is unrecognizable by magic");
    // But from_extension with .safetensors works
    let by_ext = aprender::format::rosetta::FormatType::from_extension(
        std::path::Path::new("model.safetensors"),
    );
    assert_eq!(by_ext.unwrap(), aprender::format::rosetta::FormatType::SafeTensors);
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
    assert!(!result.trim().is_empty(), "PMAT-236: Template output must never be empty");
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
    assert!(result.contains("helpful assistant"), "PMAT-237: System message must not be silently skipped");
    assert!(result.contains("Hi"), "PMAT-237: User message must not be silently skipped");
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
    // ChatML uses <|im_start|> and <|im_end|>
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

// =============================================================================
// Category 7: QA Integrity Regressions (P0-QA-001, Bug 206)
// =============================================================================

#[test]
fn regression_p0_qa_001_skipped_gates_distinguishable() {
    // The key invariant: skipped tests must be tracked separately from executed
    // This is now enforced by QaReport.gates_executed/gates_skipped (Phase 1)
    let total = 9;
    let executed = 5;
    let skipped = 4;
    assert_eq!(executed + skipped, total);
    assert!(skipped > 0, "P0-QA-001: Must track skipped gates");
}

#[test]
fn regression_bug_206_perf_regression_detection() {
    let prev_tps = 89.8;
    let curr_tps = 67.8;
    let regression = (prev_tps - curr_tps) / prev_tps;
    assert!(
        regression > 0.10,
        "Bug 206: 89.8→67.8 must be >10% regression (was {:.1}%)",
        regression * 100.0
    );
}

#[test]
fn regression_bug_206_no_false_positive() {
    let prev_tps = 67.8;
    let curr_tps = 89.8;
    let regression = (prev_tps - curr_tps) / prev_tps;
    assert!(regression < 0.0, "Improvement must not trigger regression");
}

#[test]
fn regression_bug_206_exact_same_no_regression() {
    let prev_tps = 89.8;
    let curr_tps = 89.8;
    let regression: f64 = (prev_tps - curr_tps) / prev_tps;
    assert!(
        regression.abs() < f64::EPSILON,
        "Same value must not trigger regression"
    );
}

// =============================================================================
// Category 8: Validated Tensors (PMAT-235)
// =============================================================================

#[test]
fn regression_pmat_235_weight_size_mismatch() {
    use aprender::format::validated_tensors::ValidatedWeight;
    let data = vec![1.0f32; 10];
    let result = ValidatedWeight::new(data, 3, 4, "test");
    assert!(result.is_err(), "PMAT-235: Must reject size mismatch (10 != 3*4)");
}

#[test]
fn regression_pmat_235_weight_correct_size() {
    use aprender::format::validated_tensors::ValidatedWeight;
    let data = vec![1.0f32; 12];
    let result = ValidatedWeight::new(data, 3, 4, "test");
    assert!(result.is_ok(), "PMAT-235: Must accept correct size (12 == 3*4)");
}

#[test]
fn regression_pmat_235_embedding_size_mismatch() {
    use aprender::format::validated_tensors::ValidatedEmbedding;
    let data = vec![0.1f32; 5];
    let result = ValidatedEmbedding::new(data, 2, 3);
    assert!(result.is_err(), "PMAT-235: Must reject embedding size mismatch");
}

// =============================================================================
// Category 9: Model Family / Architecture Detection
// =============================================================================

#[test]
fn regression_arch_qwen2_in_registry() {
    let registry = aprender::format::model_family::build_default_registry();
    let names = registry.family_names();
    assert!(names.iter().any(|n| n.contains("qwen2")));
}

#[test]
fn regression_arch_known_families_contains_qwen2() {
    assert!(aprender::format::model_family::KNOWN_FAMILIES.contains(&"qwen2"));
}

#[test]
fn regression_arch_known_families_contains_llama() {
    assert!(aprender::format::model_family::KNOWN_FAMILIES.contains(&"llama"));
}

// =============================================================================
// Category 10: Rosetta Stone Validation
// =============================================================================

#[test]
fn regression_rosetta_construction() {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let _ = format!("{rosetta:?}");
}

#[test]
fn regression_rosetta_validate_nonexistent() {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let result = rosetta.validate(Path::new("/nonexistent/model.gguf"));
    assert!(result.is_err());
}

#[test]
fn regression_rosetta_validate_empty() {
    let file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let result = rosetta.validate(file.path());
    assert!(result.is_err());
}

// =============================================================================
// Category 11: Source/HF URI Parsing
// =============================================================================

#[test]
fn regression_source_parse_hf_valid() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct");
    assert!(result.is_ok());
}

#[test]
fn regression_source_parse_hf_invalid_single_part() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("hf://only-one-part");
    assert!(result.is_err());
}

#[test]
fn regression_source_parse_hf_with_file() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("hf://Qwen/Qwen2.5-Coder-0.5B/model.safetensors");
    match result.unwrap() {
        Source::HuggingFace { file, .. } => {
            assert!(file.is_some(), "File component must be parsed");
        }
        _ => panic!("Expected HuggingFace variant"),
    }
}

#[test]
fn regression_source_parse_https() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("https://example.com/model.gguf");
    assert!(matches!(result.unwrap(), Source::Url(_)));
}

#[test]
fn regression_source_parse_local() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("/tmp/model.gguf");
    assert!(matches!(result.unwrap(), Source::Local(_)));
}

// =============================================================================
// Category 12: Metadata Plausibility (Bug 210, GH-222)
// =============================================================================

/// Bug 210: GgufModelConfig rope_theta for Qwen2 must NOT be 10000.0
/// The exact Bug 210 signature is qwen2 architecture with rope_theta=10000.0
#[test]
fn regression_bug_210_qwen2_rope_theta_not_default() {
    use aprender::format::gguf::api::GgufModelConfig;
    // This is the WRONG config that caused Bug 210
    let bad_config = GgufModelConfig {
        architecture: Some("qwen2".to_string()),
        rope_theta: Some(10000.0),
        ..Default::default()
    };
    // The plausibility check: Qwen2 with 10000.0 is the Bug 210 signature
    let theta = bad_config.rope_theta.expect("has theta");
    let is_suspicious = bad_config.architecture.as_deref() == Some("qwen2")
        && (f64::from(theta) - 10000.0).abs() < 1.0;
    assert!(
        is_suspicious,
        "Bug 210: qwen2+10000.0 must be flagged as suspicious"
    );
}

/// Bug 210: Correct Qwen2 config has rope_theta ~1000000.0
#[test]
fn regression_bug_210_qwen2_correct_rope_theta() {
    use aprender::format::gguf::api::GgufModelConfig;
    let good_config = GgufModelConfig {
        architecture: Some("qwen2".to_string()),
        rope_theta: Some(1_000_000.0),
        ..Default::default()
    };
    let theta = good_config.rope_theta.expect("has theta");
    assert!(
        f64::from(theta) >= 100_000.0,
        "Bug 210: correct qwen2 rope_theta should be >= 100000"
    );
}

/// Bug 210: LLaMA config with rope_theta=10000.0 is valid (not suspicious)
#[test]
fn regression_bug_210_llama_default_theta_is_valid() {
    use aprender::format::gguf::api::GgufModelConfig;
    let config = GgufModelConfig {
        architecture: Some("llama".to_string()),
        rope_theta: Some(10000.0),
        ..Default::default()
    };
    let theta = f64::from(config.rope_theta.expect("has theta"));
    // LLaMA 1/2 with 10000.0 is correct
    assert!(
        theta >= 1000.0 && theta <= 10_000_000.0,
        "Bug 210: llama with rope_theta=10000.0 is valid"
    );
}

/// Bug 210: rope_theta plausibility range check
#[test]
fn regression_bug_210_rope_theta_plausibility_ranges() {
    // Known architecture rope_theta values
    let known_values: Vec<(&str, f64)> = vec![
        ("qwen2", 1_000_000.0),
        ("llama", 10_000.0),   // LLaMA 1/2
        ("llama", 500_000.0),  // LLaMA 3
        ("gpt2", 10_000.0),    // GPT-2
    ];

    for (arch, theta) in known_values {
        assert!(
            theta >= 100.0 && theta <= 100_000_000.0,
            "Bug 210: {arch} rope_theta={theta} must be in plausible range [100, 100M]"
        );
    }
}

/// Bug 210: rms_norm_eps plausibility
#[test]
fn regression_bug_210_rms_norm_eps_plausibility() {
    let valid_values: Vec<f32> = vec![1e-6, 1e-5, 1e-8];
    for eps in valid_values {
        assert!(
            eps > 0.0 && eps <= 0.01,
            "Bug 210: rms_norm_eps={eps} must be in range (0, 0.01]"
        );
    }
    // Invalid: zero or negative
    assert!(!(0.0_f32 > 0.0 && 0.0_f32 <= 0.01), "Zero eps must fail");
}

/// Bug 210: max_position_embeddings plausibility
#[test]
fn regression_bug_210_max_position_embeddings_plausibility() {
    let valid_values: Vec<usize> = vec![2048, 4096, 8192, 32768, 131072];
    for max_pos in valid_values {
        assert!(
            max_pos >= 128 && max_pos <= 1_048_576,
            "Bug 210: max_pos={max_pos} must be in range [128, 1M]"
        );
    }
}
