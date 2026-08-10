use super::*;

// =========================================================================
// execute() error case tests
// =========================================================================

#[test]
fn test_execute_invalid_repo_id_no_slash() {
    let temp_dir = std::env::temp_dir().join("apr_pub_invalid_repo_1");
    let _ = fs::create_dir_all(&temp_dir);

    let result = execute(
        &temp_dir,
        "invalid-repo-name", // No slash
        None,
        "mit",
        "text-generation",
        None,
        &[],
        None,
        true,
        false,
        None,
        &[],
    );

    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Invalid repo ID"));
            assert!(msg.contains("Expected format: org/repo-name"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_execute_invalid_repo_id_too_many_slashes() {
    let temp_dir = std::env::temp_dir().join("apr_pub_invalid_repo_2");
    let _ = fs::create_dir_all(&temp_dir);

    let result = execute(
        &temp_dir,
        "org/repo/extra", // Too many slashes
        None,
        "mit",
        "text-generation",
        None,
        &[],
        None,
        true,
        false,
        None,
        &[],
    );

    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Invalid repo ID"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_execute_directory_not_found() {
    let result = execute(
        Path::new("/nonexistent/directory"),
        "paiml/test-model",
        None,
        "mit",
        "text-generation",
        None,
        &[],
        None,
        true,
        false,
        None,
        &[],
    );

    assert!(result.is_err());
    match result {
        Err(CliError::FileNotFound(_)) => {}
        other => panic!("Expected FileNotFound, got {:?}", other),
    }
}

#[test]
fn test_execute_no_model_files() {
    let temp_dir = std::env::temp_dir().join("apr_pub_no_models");
    let _ = fs::create_dir_all(&temp_dir);
    // Create non-model files
    let txt_file = temp_dir.join("readme.txt");
    let _ = fs::write(&txt_file, "test");

    let result = execute(
        &temp_dir,
        "paiml/test-model",
        None,
        "mit",
        "text-generation",
        None,
        &[],
        None,
        true,
        false,
        None,
        &[],
    );

    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("No model files found"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_execute_dry_run_success() {
    let temp_dir = std::env::temp_dir().join("apr_pub_dry_run");
    let _ = fs::create_dir_all(&temp_dir);

    // Create a model file
    let model_file = temp_dir.join("model.apr");
    let _ = fs::write(&model_file, "APR2test");

    let result = execute(
        &temp_dir,
        "paiml/test-model",
        Some("My Test Model"),
        "apache-2.0",
        "text-generation",
        Some("aprender"),
        &["rust".to_string(), "transformer".to_string()],
        Some("Test commit"),
        true, // dry_run
        true, // verbose
        None,
        &[],
    );

    assert!(result.is_ok());

    let _ = fs::remove_dir_all(&temp_dir);
}

// =========================================================================
// F-PUBLISH-EXTRA-001 falsification tests
// (contracts/apr-cli-publish-extra-v1.yaml)
// =========================================================================

/// FALSIFY-PUB-EXTRA-002: sha256 mismatch must abort BEFORE any network I/O.
/// Guard runs ahead of the dry_run branch, so dry_run=true still trips it.
#[test]
fn test_falsify_pub_extra_002_sha_mismatch_aborts() {
    let temp_dir = std::env::temp_dir().join("apr_pub_falsify_sha_mismatch");
    let _ = fs::remove_dir_all(&temp_dir);
    let _ = fs::create_dir_all(&temp_dir);

    let model_file = temp_dir.join("model.apr");
    fs::write(&model_file, b"APR2mismatch").expect("write model");

    // Deliberately wrong sha256 (64 hex zeros).
    let manifest_path = temp_dir.join("manifest.yaml");
    let manifest_yaml = format!(
        "schema_version: \"1.0.0\"\n\
         name: \"paiml/test-model\"\n\
         artifact_url: \"https://example.com/model.apr\"\n\
         sha256: \"{}\"\n",
        "0".repeat(64)
    );
    fs::write(&manifest_path, manifest_yaml).expect("write manifest");

    let result = execute(
        &temp_dir,
        "paiml/test-model",
        None,
        "apache-2.0",
        "text-generation",
        None,
        &[],
        None,
        true,  // dry_run — proves guard runs before network path
        false, // verbose
        Some(&manifest_path),
        &[],
    );

    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("sha256 mismatch"),
                "Expected sha256 mismatch error, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed(sha256 mismatch), got {other:?}"),
    }

    let _ = fs::remove_dir_all(&temp_dir);
}

/// FALSIFY-PUB-EXTRA-003: extra-files and a valid manifest propagate through
/// dry-run without error. Proves the `--extra-file` + `--manifest` paths are
/// reachable and the pre-flight guard accepts a correctly-hashed artifact.
#[test]
fn test_falsify_pub_extra_003_extra_file_passthrough() {
    let temp_dir = std::env::temp_dir().join("apr_pub_falsify_extra_passthrough");
    let _ = fs::remove_dir_all(&temp_dir);
    let _ = fs::create_dir_all(&temp_dir);

    let model_file = temp_dir.join("model.apr");
    let artifact_bytes = b"APR2three-format-ship";
    fs::write(&model_file, artifact_bytes).expect("write model");

    // Compute the REAL sha256 so the manifest matches.
    let artifact_sha = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(artifact_bytes);
        format!("{:x}", h.finalize())
    };

    let manifest_path = temp_dir.join("manifest.yaml");
    let manifest_yaml = format!(
        "schema_version: \"1.0.0\"\n\
         name: \"paiml/test-model\"\n\
         artifact_url: \"https://example.com/model.apr\"\n\
         sha256: \"{artifact_sha}\"\n\
         size_bytes: {}\n",
        artifact_bytes.len()
    );
    fs::write(&manifest_path, manifest_yaml).expect("write manifest");

    let extra_file = temp_dir.join("tokenizer.json");
    fs::write(&extra_file, br#"{"version":"1.0"}"#).expect("write extra");

    let result = execute(
        &temp_dir,
        "paiml/test-model",
        None,
        "apache-2.0",
        "text-generation",
        None,
        &[],
        None,
        true, // dry_run — avoids network
        true, // verbose
        Some(&manifest_path),
        std::slice::from_ref(&extra_file),
    );

    assert!(
        result.is_ok(),
        "Expected Ok from manifest + extra-file dry-run, got {result:?}"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

// =========================================================================
// find_model_files() tests
// =========================================================================

#[test]
fn test_find_model_files_empty() {
    let temp_dir = std::env::temp_dir().join("apr_publish_test_empty");
    let _ = fs::create_dir_all(&temp_dir);

    let files = find_model_files(&temp_dir).expect("value");
    assert!(files.is_empty());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_find_model_files_apr() {
    let temp_dir = std::env::temp_dir().join("apr_pub_find_apr");
    let _ = fs::create_dir_all(&temp_dir);

    let apr_file = temp_dir.join("model.apr");
    let _ = fs::write(&apr_file, "APR2");

    let files = find_model_files(&temp_dir).expect("value");
    assert_eq!(files.len(), 1);
    assert!(files[0].ends_with("model.apr"));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_find_model_files_safetensors() {
    let temp_dir = std::env::temp_dir().join("apr_pub_find_st");
    let _ = fs::create_dir_all(&temp_dir);

    let st_file = temp_dir.join("model.safetensors");
    let _ = fs::write(&st_file, "safetensors");

    let files = find_model_files(&temp_dir).expect("value");
    assert_eq!(files.len(), 1);
    assert!(files[0].ends_with("model.safetensors"));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_find_model_files_gguf() {
    let temp_dir = std::env::temp_dir().join("apr_pub_find_gguf");
    let _ = fs::create_dir_all(&temp_dir);

    let gguf_file = temp_dir.join("model.gguf");
    let _ = fs::write(&gguf_file, "GGUF");

    let files = find_model_files(&temp_dir).expect("value");
    assert_eq!(files.len(), 1);
    assert!(files[0].ends_with("model.gguf"));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_find_model_files_multiple_formats() {
    let temp_dir = std::env::temp_dir().join("apr_pub_find_multi");
    let _ = fs::create_dir_all(&temp_dir);

    let _ = fs::write(temp_dir.join("model.apr"), "APR2");
    let _ = fs::write(temp_dir.join("model.safetensors"), "st");
    let _ = fs::write(temp_dir.join("model.gguf"), "GGUF");
    let _ = fs::write(temp_dir.join("readme.txt"), "ignored");

    let files = find_model_files(&temp_dir).expect("value");
    assert_eq!(files.len(), 3);
    // Files are sorted alphabetically
    assert!(files[0].ends_with("model.apr"));
    assert!(files[1].ends_with("model.gguf"));
    assert!(files[2].ends_with("model.safetensors"));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_find_model_files_ignores_non_model_files() {
    let temp_dir = std::env::temp_dir().join("apr_pub_find_ignore");
    let _ = fs::create_dir_all(&temp_dir);

    let _ = fs::write(temp_dir.join("model.txt"), "text");
    let _ = fs::write(temp_dir.join("config.json"), "{}");
    let _ = fs::write(temp_dir.join("tokenizer.json"), "{}");
    let _ = fs::write(temp_dir.join("README.md"), "# Readme");

    let files = find_model_files(&temp_dir).expect("value");
    assert!(files.is_empty());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_find_model_files_case_insensitive() {
    let temp_dir = std::env::temp_dir().join("apr_pub_find_case");
    let _ = fs::create_dir_all(&temp_dir);

    // Extensions are case-insensitive (APR, GGUF, SAFETENSORS work too)
    let _ = fs::write(temp_dir.join("model.APR"), "APR2");
    let _ = fs::write(temp_dir.join("model.GGUF"), "GGUF");

    let files = find_model_files(&temp_dir).expect("value");
    assert_eq!(files.len(), 2);

    let _ = fs::remove_dir_all(&temp_dir);
}

// =========================================================================
// generate_model_card() tests
// =========================================================================

#[test]
fn test_generate_model_card() {
    let (card, file_names) = generate_model_card(
        "paiml/test-model",
        Some("Test Model"),
        "mit",
        "text-generation",
        None,
        &[],
        &[],
    );

    assert_eq!(card.model_id, "paiml/test-model");
    assert_eq!(card.name, "Test Model");
    assert_eq!(card.license, Some("mit".to_string()));
    assert!(file_names.is_empty());
}

#[test]
fn test_generate_model_card_default_name() {
    let (card, _) = generate_model_card(
        "paiml/my-awesome-model",
        None, // No explicit name
        "apache-2.0",
        "text-generation",
        None,
        &[],
        &[],
    );

    // Should use last part of repo_id as name
    assert_eq!(card.name, "my-awesome-model");
}

#[test]
fn test_generate_model_card_description_generated() {
    let (card, _) = generate_model_card(
        "paiml/whisper-tiny",
        Some("Whisper Tiny"),
        "mit",
        "automatic-speech-recognition",
        Some("whisper"),
        &["speech".to_string()],
        &[],
    );

    assert!(card.description.is_some());
    assert!(card
        .description
        .expect("description")
        .contains("Whisper Tiny"));
}

#[test]
fn test_generate_model_card_captures_file_names() {
    let files = vec![
        std::path::PathBuf::from("/tmp/model.gguf"),
        std::path::PathBuf::from("/tmp/model.safetensors"),
    ];
    let (_, file_names) = generate_model_card(
        "paiml/test",
        None,
        "mit",
        "text-generation",
        None,
        &[],
        &files,
    );

    assert_eq!(file_names, vec!["model.gguf", "model.safetensors"]);
}

// =========================================================================
// ModelCardExt::to_huggingface_extended() tests
// =========================================================================

#[test]
fn test_model_card_extended_asr() {
    let card = ModelCard::new("paiml/whisper-test", "1.0.0")
        .with_name("Whisper Test")
        .with_license("MIT");

    let output = card.to_huggingface_extended(
        "automatic-speech-recognition",
        Some("whisper-apr"),
        &["whisper".to_string()],
        &[],
    );

    assert!(output.contains("pipeline_tag: automatic-speech-recognition"));
    assert!(output.contains("library_name: whisper-apr"));
    assert!(output.contains("- speech-recognition"));
    assert!(output.contains("- whisper"));
}

#[test]
fn test_model_card_extended_text_generation() {
    let card = ModelCard::new("paiml/gpt-test", "1.0.0")
        .with_name("GPT Test")
        .with_license("apache-2.0");

    let output = card.to_huggingface_extended(
        "text-generation",
        Some("aprender"),
        &["transformer".to_string(), "causal-lm".to_string()],
        &[],
    );

    assert!(output.contains("pipeline_tag: text-generation"));
    assert!(output.contains("library_name: aprender"));
    assert!(output.contains("- transformer"));
    assert!(output.contains("- causal-lm"));
    assert!(output.contains("- aprender"));
    assert!(output.contains("- rust"));
    // Should NOT have ASR-specific tags
    assert!(!output.contains("- speech-recognition"));
}

#[test]
fn test_model_card_extended_yaml_front_matter() {
    let card = ModelCard::new("paiml/test", "1.0.0")
        .with_name("Test")
        .with_license("mit");

    let output = card.to_huggingface_extended("text-generation", None, &[], &[]);

    // Should start with YAML front matter
    assert!(output.starts_with("---\n"));
    assert!(output.contains("\n---\n\n"));
}

#[test]
fn test_model_card_extended_contains_sections() {
    let card = ModelCard::new("paiml/test", "1.0.0")
        .with_name("Test Model")
        .with_license("mit");

    let output = card.to_huggingface_extended("text-generation", None, &[], &[]);

    // Should contain all expected sections
    assert!(output.contains("# Test Model"));
    assert!(output.contains("## Available Formats"));
    assert!(output.contains("## Usage"));
    assert!(output.contains("## Framework"));
    assert!(output.contains("## Citation"));
}

#[test]
fn test_model_card_extended_code_example() {
    let card = ModelCard::new("paiml/test", "1.0.0").with_name("Test");

    let output = card.to_huggingface_extended("text-generation", None, &[], &[]);

    // Should contain Rust code example
    assert!(output.contains("```rust"));
    assert!(output.contains("use aprender::Model;"));
    assert!(output.contains("Model::load"));
}

#[test]
fn test_model_card_extended_bibtex_citation() {
    let card = ModelCard::new("paiml/test", "1.0.0").with_name("Test");

    let output = card.to_huggingface_extended("text-generation", None, &[], &[]);

    assert!(output.contains("```bibtex"));
    assert!(output.contains("@software{aprender,"));
    assert!(output.contains("title = {aprender: Rust ML Library}"));
}

#[test]
fn test_model_card_extended_model_index() {
    let card = ModelCard::new("paiml/test-model", "1.0.0").with_name("Test Model");

    let output = card.to_huggingface_extended("text-generation", None, &[], &[]);

    assert!(output.contains("model-index:"));
    assert!(output.contains("- name: paiml/test-model"));
    assert!(output.contains("type: text-generation"));
}

#[test]
fn test_model_card_extended_no_library_name() {
    let card = ModelCard::new("paiml/test", "1.0.0").with_name("Test");

    let output = card.to_huggingface_extended(
        "text-generation",
        None, // No library name
        &[],
        &[],
    );

    // Should NOT contain library_name field
    assert!(!output.contains("library_name:"));
}

#[test]
fn test_model_card_extended_deduplicated_tags() {
    let card = ModelCard::new("paiml/test", "1.0.0").with_name("Test");

    let output = card.to_huggingface_extended(
        "text-generation",
        None,
        &[
            "rust".to_string(),     // Already added by default
            "aprender".to_string(), // Already added by default
            "custom".to_string(),   // New tag
        ],
        &[],
    );

    // Count occurrences of "- rust" (should be exactly 1)
    let rust_count = output.matches("  - rust\n").count();
    assert_eq!(rust_count, 1, "rust tag should appear exactly once");

    let aprender_count = output.matches("  - aprender\n").count();
    assert_eq!(aprender_count, 1, "aprender tag should appear exactly once");

    assert!(output.contains("  - custom\n"));
}

#[test]
fn test_model_card_extended_multilingual_asr() {
    let card = ModelCard::new("paiml/whisper", "1.0.0").with_name("Whisper");

    let output = card.to_huggingface_extended("automatic-speech-recognition", None, &[], &[]);

    // ASR models should have language specification
    assert!(output.contains("language:"));
    assert!(output.contains("  - en"));
    assert!(output.contains("  - multilingual"));
}

#[test]
fn test_model_card_extended_with_architecture() {
    let card = ModelCard::new("paiml/test", "1.0.0")
        .with_name("Test")
        .with_architecture("transformer");

    let output = card.to_huggingface_extended("text-generation", None, &[], &[]);

    // Architecture should appear in tags
    assert!(output.contains("  - transformer\n"));
}

// GH-511: Test that Available Formats section uses actual file names
#[test]
fn test_model_card_extended_dynamic_formats() {
    let card = ModelCard::new("paiml/test", "1.0.0").with_name("Test");
    let file_names = vec!["model.gguf".to_string(), "weights.safetensors".to_string()];

    let output = card.to_huggingface_extended("text-generation", None, &[], &file_names);

    // Should contain actual file names, not hardcoded defaults
    assert!(output.contains("| `model.gguf` | GGUF format (llama.cpp compatible) |"));
    assert!(output.contains("| `weights.safetensors` | HuggingFace SafeTensors format |"));
    // Available Formats table should NOT contain hardcoded model.apr when not in file list
    // (Note: model.apr still appears in the Usage code example section, so check table specifically)
    let formats_section = output.split("## Available Formats").nth(1).expect("nth(1");
    let formats_table = formats_section.split("## Usage").next().expect("next");
    assert!(!formats_table.contains("model.apr"));
}

#[test]
fn test_model_card_extended_empty_files_fallback() {
    let card = ModelCard::new("paiml/test", "1.0.0").with_name("Test");

    let output = card.to_huggingface_extended("text-generation", None, &[], &[]);

    // Empty file list should show fallback
    assert!(output.contains("model.apr"));
}

// =========================================================================
// FALSIFY-PUB-LFS-001: format_upload_route dry-run surfacing
// =========================================================================
//
// The dry-run reporter classifies each file by the HF upload path its size
// will take. These tests pin the exact strings the CLI emits so both the
// partitioning AND the user-facing message are tracked for regressions.

#[test]
fn dry_run_route_partitions_at_5_gib_exactly() {
    const GIB: u64 = 1024 * 1024 * 1024;
    // <= 5 GiB MUST route to HTTP LFS
    assert_eq!(format_upload_route(0), "[→ HTTP LFS (≤5 GiB)]");
    assert_eq!(format_upload_route(5 * GIB - 1), "[→ HTTP LFS (≤5 GiB)]");
    assert_eq!(format_upload_route(5 * GIB), "[→ HTTP LFS (≤5 GiB)]");
    // > 5 GiB MUST NOT route to HTTP LFS (same boundary as should_use_xet)
    assert_ne!(format_upload_route(5 * GIB + 1), "[→ HTTP LFS (≤5 GiB)]");
}

#[cfg(feature = "xet")]
#[test]
fn dry_run_route_above_5_gib_reports_xet_when_enabled() {
    const GIB: u64 = 1024 * 1024 * 1024;
    assert_eq!(format_upload_route(5 * GIB + 1), "[→ Xet CAS (>5 GiB)]");
    // Real SHIP-TWO-001 teacher sizes
    assert_eq!(format_upload_route(8_035_635_524), "[→ Xet CAS (>5 GiB)]");
    assert_eq!(format_upload_route(15_231_938_404), "[→ Xet CAS (>5 GiB)]");
}

#[cfg(all(feature = "hf-hub", not(feature = "xet")))]
#[test]
fn dry_run_route_above_5_gib_flags_missing_xet_feature() {
    const GIB: u64 = 1024 * 1024 * 1024;
    assert_eq!(
        format_upload_route(5 * GIB + 1),
        "[✗ would FAIL: rebuild with --features xet]"
    );
    assert_eq!(
        format_upload_route(8_035_635_524),
        "[✗ would FAIL: rebuild with --features xet]"
    );
}

// =========================================================================
// PMAT-690 P3-C-prep defect 6 — companion file discovery + safetensors alias
// =========================================================================

#[test]
fn test_find_companion_files_picks_all_hf_integration_files() {
    let dir = std::env::temp_dir().join(format!("apr_pub_companions_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    for name in [
        "README.md",
        "LICENSE",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
    ] {
        fs::write(dir.join(name), b"x").unwrap();
    }
    // Decoy: NOT in the allowlist — must NOT be picked up.
    fs::write(dir.join("random_other.json"), b"x").unwrap();
    fs::write(dir.join("notes.txt"), b"x").unwrap();
    // Binary artifacts also present — find_companion_files MUST NOT include them.
    fs::write(dir.join("model.apr"), b"x").unwrap();
    fs::write(dir.join("model.safetensors"), b"x").unwrap();

    let found = super::find_companion_files(&dir).unwrap();
    let names: Vec<String> = found
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    assert!(names.contains(&"README.md".to_string()));
    assert!(names.contains(&"LICENSE".to_string()));
    assert!(names.contains(&"config.json".to_string()));
    assert!(names.contains(&"generation_config.json".to_string()));
    assert!(names.contains(&"tokenizer.json".to_string()));
    assert!(names.contains(&"tokenizer_config.json".to_string()));
    assert!(names.contains(&"vocab.json".to_string()));
    assert!(names.contains(&"merges.txt".to_string()));
    assert!(names.contains(&"special_tokens_map.json".to_string()));
    assert!(names.contains(&"chat_template.jinja".to_string()));
    assert!(!names.contains(&"random_other.json".to_string()));
    assert!(!names.contains(&"notes.txt".to_string()));
    assert!(!names.contains(&"model.apr".to_string()));
    assert!(!names.contains(&"model.safetensors".to_string()));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_find_companion_files_empty_dir_returns_empty() {
    let dir = std::env::temp_dir().join(format!("apr_pub_companions_empty_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let found = super::find_companion_files(&dir).unwrap();
    assert!(found.is_empty());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_safetensors_needing_alias_descriptive_name_triggers_alias() {
    let paths = vec![
        std::path::PathBuf::from("/tmp/albor-370m-v1.safetensors"),
        std::path::PathBuf::from("/tmp/albor-370m-v1.apr"),
    ];
    let result = super::safetensors_needing_alias(&paths);
    assert!(result.is_some());
    assert_eq!(
        result.unwrap().file_name().unwrap().to_str(),
        Some("albor-370m-v1.safetensors")
    );
}

#[test]
fn test_safetensors_needing_alias_canonical_name_skips_alias() {
    // If the file is ALREADY named model.safetensors, no alias is needed
    // (HF Transformers auto-discovers it directly).
    let paths = vec![std::path::PathBuf::from("/tmp/model.safetensors")];
    assert!(super::safetensors_needing_alias(&paths).is_none());
}

#[test]
fn test_safetensors_needing_alias_no_safetensors_skips_alias() {
    // Pure .apr + .gguf publish (no safetensors at all) — no alias needed.
    let paths = vec![
        std::path::PathBuf::from("/tmp/model.apr"),
        std::path::PathBuf::from("/tmp/model.gguf"),
    ];
    assert!(super::safetensors_needing_alias(&paths).is_none());
}

// =========================================================================
// FALSIFY-PUBLISH-SPDX-001 — `--license` must agree with validate-manifest
// =========================================================================

/// 0.63.0 lower-cased `--license 'NOT-A-LICENSE'` straight into the model
/// card's YAML front matter, where the Hub ignores an unrecognised value,
/// while `apr validate-manifest` FALSIFY-PM-004 fails closed on the same
/// string (issue #2391).
#[test]
fn execute_rejects_non_spdx_license_before_any_io() {
    let temp_dir = std::env::temp_dir().join("apr_pub_spdx_reject");
    let _ = fs::create_dir_all(&temp_dir);

    for bad in ["NOT-A-LICENSE", "NOT-A-LICENSE-!!", "Apache2", "whatever"] {
        let result = execute(
            &temp_dir,
            "paiml/x",
            None,
            bad,
            "text-generation",
            None,
            &[],
            None,
            true, // dry_run — must still refuse
            false,
            None,
            &[],
        );
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("SPDX"), "must say why: {msg}");
                assert!(msg.contains(bad), "must echo the value: {msg}");
            }
            other => panic!("--license {bad:?} must be rejected, got {other:?}"),
        }
    }
}

/// Control: the shipped default and the identifiers validate-manifest accepts
/// must get past this guard (they fail later, on the empty directory).
#[test]
fn execute_accepts_valid_spdx_identifiers() {
    let temp_dir = std::env::temp_dir().join("apr_pub_spdx_accept");
    let _ = fs::create_dir_all(&temp_dir);

    for ok in ["mit", "MIT", "Apache-2.0", "llama3.1", "custom"] {
        let result = execute(
            &temp_dir,
            "paiml/x",
            None,
            ok,
            "text-generation",
            None,
            &[],
            None,
            true,
            false,
            None,
            &[],
        );
        if let Err(CliError::ValidationFailed(msg)) = &result {
            assert!(
                !msg.contains("SPDX"),
                "{ok} must not be rejected as non-SPDX: {msg}"
            );
        }
    }
}
