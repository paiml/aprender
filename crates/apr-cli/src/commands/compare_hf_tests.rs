use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

// ========================================================================
// map_hf_to_apr_name Tests
// ========================================================================

#[test]
fn test_map_hf_to_apr_name_with_model_prefix() {
    let hf_name = "model.decoder.layers.0.encoder_attn.q_proj.weight";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "decoder.layers.0.encoder_attn.q_proj.weight");
}

#[test]
fn test_map_hf_to_apr_name_without_prefix() {
    let hf_name = "decoder.layers.0.self_attn.k_proj.weight";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "decoder.layers.0.self_attn.k_proj.weight");
}

#[test]
fn test_map_hf_to_apr_name_encoder() {
    let hf_name = "model.encoder.embed_positions.weight";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "encoder.embed_positions.weight");
}

#[test]
fn test_map_hf_to_apr_name_proj_out() {
    let hf_name = "model.decoder.layers.3.fc2.weight";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "decoder.layers.3.fc2.weight");
}

#[test]
fn test_map_hf_to_apr_name_empty() {
    let hf_name = "";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "");
}

#[test]
fn test_map_hf_to_apr_name_only_model() {
    let hf_name = "model.";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "");
}

#[test]
fn test_map_hf_to_apr_name_no_model_prefix() {
    let hf_name = "lm_head.weight";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "lm_head.weight");
}

// ========================================================================
// run Command Tests
// ========================================================================

#[test]
fn test_run_file_not_found() {
    let result = run(
        Path::new("/nonexistent/model.apr"),
        "openai/whisper-tiny",
        None,
        1e-5,
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_invalid_apr() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not a valid apr file").expect("write");

    let result = run(file.path(), "openai/whisper-tiny", None, 1e-5, false);
    // Either FeatureDisabled or InvalidFormat error
    assert!(result.is_err());
}

#[test]
fn test_run_with_tensor_filter() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(
        file.path(),
        "openai/whisper-tiny",
        Some("decoder.layers.0"),
        1e-5,
        false,
    );
    // Should fail (invalid file or feature disabled)
    assert!(result.is_err());
}

#[test]
fn test_run_with_json_output() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(
        file.path(),
        "openai/whisper-tiny",
        None,
        1e-5,
        true, // json_output
    );
    // Should fail (invalid file or feature disabled)
    assert!(result.is_err());
}

#[test]
fn test_run_with_custom_threshold() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(
        file.path(),
        "openai/whisper-tiny",
        None,
        1e-3, // looser threshold
        false,
    );
    // Should fail (invalid file or feature disabled)
    assert!(result.is_err());
}

#[test]
fn test_run_with_strict_threshold() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(
        file.path(),
        "openai/whisper-tiny",
        None,
        1e-8, // strict threshold
        false,
    );
    // Should fail (invalid file or feature disabled)
    assert!(result.is_err());
}

#[test]
#[cfg(not(feature = "safetensors-compare"))]
fn test_run_feature_disabled() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(file.path(), "openai/whisper-tiny", None, 1e-5, false);
    match result {
        Err(CliError::FeatureDisabled(feature)) => {
            assert_eq!(feature, "safetensors-compare");
        }
        _ => panic!("Expected FeatureDisabled error"),
    }
}

// ========================================================================
// Edge Cases
// ========================================================================

#[test]
fn test_map_hf_to_apr_name_special_chars() {
    let hf_name = "model.layer_norm.weight";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "layer_norm.weight");
}

#[test]
fn test_map_hf_to_apr_name_deep_nesting() {
    let hf_name = "model.decoder.layers.23.self_attn.k_proj.weight";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "decoder.layers.23.self_attn.k_proj.weight");
}

#[test]
fn test_map_hf_to_apr_name_bias() {
    let hf_name = "model.encoder.layers.0.fc1.bias";
    let apr_name = map_hf_to_apr_name(hf_name);
    assert_eq!(apr_name, "encoder.layers.0.fc1.bias");
}

// ========================================================================
// map_hf_to_apr_name: "model" prefix edge cases
// ========================================================================

#[test]
fn map_hf_name_starting_with_model_underscore_is_not_stripped() {
    // "model_" is NOT "model." so should NOT be stripped
    let apr_name = map_hf_to_apr_name("model_weights.layer.weight");
    assert_eq!(apr_name, "model_weights.layer.weight");
}

#[test]
fn map_hf_name_model_alone_without_dot_is_unchanged() {
    let apr_name = map_hf_to_apr_name("model");
    assert_eq!(apr_name, "model");
}

#[test]
fn map_hf_name_double_model_prefix_strips_first_only() {
    // "model.model.x" should strip ONE "model." prefix
    let apr_name = map_hf_to_apr_name("model.model.layer.weight");
    assert_eq!(apr_name, "model.layer.weight");
}

#[test]
fn map_hf_name_single_segment_no_dots() {
    let apr_name = map_hf_to_apr_name("embed_tokens");
    assert_eq!(apr_name, "embed_tokens");
}

#[test]
fn map_hf_name_model_dot_single_segment() {
    let apr_name = map_hf_to_apr_name("model.weight");
    assert_eq!(apr_name, "weight");
}

// ========================================================================
// map_hf_to_apr_name: preserves layer numbering
// ========================================================================

#[test]
fn map_hf_name_preserves_large_layer_index() {
    let apr_name = map_hf_to_apr_name("model.decoder.layers.127.self_attn.v_proj.weight");
    assert_eq!(apr_name, "decoder.layers.127.self_attn.v_proj.weight");
}

#[test]
fn map_hf_name_preserves_zero_layer_index() {
    let apr_name = map_hf_to_apr_name("model.encoder.layers.0.layer_norm.weight");
    assert_eq!(apr_name, "encoder.layers.0.layer_norm.weight");
}

// ========================================================================
// map_hf_to_apr_name: common model architectures
// ========================================================================

#[test]
fn map_hf_name_gpt_style_no_prefix() {
    // GPT-style models often lack "model." prefix
    let apr_name = map_hf_to_apr_name("transformer.h.0.attn.c_attn.weight");
    assert_eq!(apr_name, "transformer.h.0.attn.c_attn.weight");
}

#[test]
fn map_hf_name_bert_style_with_prefix() {
    let apr_name = map_hf_to_apr_name("model.embeddings.word_embeddings.weight");
    assert_eq!(apr_name, "embeddings.word_embeddings.weight");
}

// ========================================================================
// run: error type validation
// ========================================================================

#[test]
fn run_nonexistent_file_returns_correct_variant() {
    let result = run(
        Path::new("/this/path/does/not/exist.apr"),
        "test/repo",
        None,
        1e-5,
        false,
    );
    // Without safetensors-compare feature, returns FeatureDisabled
    // With the feature, returns FileNotFound
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            CliError::FeatureDisabled(_) | CliError::FileNotFound(_)
        ),
        "Expected FeatureDisabled or FileNotFound, got: {err:?}"
    );
}

#[test]
fn run_with_zero_threshold() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(file.path(), "test/repo", None, 0.0, false);
    assert!(result.is_err());
}

#[test]
fn run_with_negative_threshold() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(file.path(), "test/repo", None, -1.0, false);
    assert!(result.is_err());
}

#[test]
fn run_with_empty_repo_name() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(file.path(), "", None, 1e-5, false);
    assert!(result.is_err());
}

#[test]
fn run_with_all_options_combined() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");

    let result = run(
        file.path(),
        "openai/whisper-tiny",
        Some("encoder"),
        1e-3,
        true,
    );
    assert!(result.is_err());
}

#[test]
#[cfg(not(feature = "safetensors-compare"))]
fn run_feature_disabled_ignores_all_args() {
    // When feature is disabled, all args are silently unused
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"valid or not, doesn't matter")
        .expect("write");

    let result = run(file.path(), "any/repo", Some("any filter"), 42.0, true);
    assert!(
        matches!(result, Err(CliError::FeatureDisabled(ref f)) if f == "safetensors-compare"),
        "Feature disabled should be returned regardless of args"
    );
}
