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
    );

    assert!(result.is_ok());

    let _ = fs::remove_dir_all(&temp_dir);
}

// =========================================================================
// find_model_files() tests
// =========================================================================

#[test]
fn test_find_model_files_empty() {
    let temp_dir = std::env::temp_dir().join("apr_publish_test_empty");
    let _ = fs::create_dir_all(&temp_dir);

    let files = find_model_files(&temp_dir).unwrap();
    assert!(files.is_empty());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_find_model_files_apr() {
    let temp_dir = std::env::temp_dir().join("apr_pub_find_apr");
    let _ = fs::create_dir_all(&temp_dir);

    let apr_file = temp_dir.join("model.apr");
    let _ = fs::write(&apr_file, "APR2");

    let files = find_model_files(&temp_dir).unwrap();
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

    let files = find_model_files(&temp_dir).unwrap();
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

    let files = find_model_files(&temp_dir).unwrap();
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

    let files = find_model_files(&temp_dir).unwrap();
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

    let files = find_model_files(&temp_dir).unwrap();
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

    let files = find_model_files(&temp_dir).unwrap();
    assert_eq!(files.len(), 2);

    let _ = fs::remove_dir_all(&temp_dir);
}

// =========================================================================
// generate_model_card() tests
// =========================================================================

#[test]
fn test_generate_model_card() {
    let card = generate_model_card(
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
}

#[test]
fn test_generate_model_card_default_name() {
    let card = generate_model_card(
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
    let card = generate_model_card(
        "paiml/whisper-tiny",
        Some("Whisper Tiny"),
        "mit",
        "automatic-speech-recognition",
        Some("whisper"),
        &["speech".to_string()],
        &[],
    );

    assert!(card.description.is_some());
    assert!(card.description.unwrap().contains("Whisper Tiny"));
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

    let output = card.to_huggingface_extended("text-generation", None, &[]);

    // Should start with YAML front matter
    assert!(output.starts_with("---\n"));
    assert!(output.contains("\n---\n\n"));
}

#[test]
fn test_model_card_extended_contains_sections() {
    let card = ModelCard::new("paiml/test", "1.0.0")
        .with_name("Test Model")
        .with_license("mit");

    let output = card.to_huggingface_extended("text-generation", None, &[]);

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

    let output = card.to_huggingface_extended("text-generation", None, &[]);

    // Should contain Rust code example
    assert!(output.contains("```rust"));
    assert!(output.contains("use aprender::Model;"));
    assert!(output.contains("Model::load"));
}

#[test]
fn test_model_card_extended_bibtex_citation() {
    let card = ModelCard::new("paiml/test", "1.0.0").with_name("Test");

    let output = card.to_huggingface_extended("text-generation", None, &[]);

    assert!(output.contains("```bibtex"));
    assert!(output.contains("@software{aprender,"));
    assert!(output.contains("title = {aprender: Rust ML Library}"));
}

#[test]
fn test_model_card_extended_model_index() {
    let card = ModelCard::new("paiml/test-model", "1.0.0").with_name("Test Model");

    let output = card.to_huggingface_extended("text-generation", None, &[]);

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

    let output = card.to_huggingface_extended("automatic-speech-recognition", None, &[]);

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

    let output = card.to_huggingface_extended("text-generation", None, &[]);

    // Architecture should appear in tags
    assert!(output.contains("  - transformer\n"));
}
