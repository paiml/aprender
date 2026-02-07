//! Compare APR model against HuggingFace source (GH-121)
//!
//! Toyota Way: Genchi Genbutsu - Go and see the actual weight values.
//! Compare .apr model weights bit-for-bit against HuggingFace safetensors.
//!
//! Usage:
//!   apr compare-hf model.apr --hf openai/whisper-tiny
//!   apr compare-hf model.apr --hf openai/whisper-tiny --tensor "decoder.layers.0.encoder_attn.q_proj.weight"
//!   apr compare-hf model.apr --hf openai/whisper-tiny --threshold 1e-5

use crate::error::CliError;
use colored::Colorize;
use std::path::Path;

#[cfg(feature = "safetensors-compare")]
use aprender::inspect::safetensors::{BatchComparison, HfSafetensors, TensorComparison};

/// Run the compare-hf command
pub(crate) fn run(
    apr_path: &Path,
    hf_repo: &str,
    tensor_filter: Option<&str>,
    threshold: f64,
    json_output: bool,
) -> Result<(), CliError> {
    #[cfg(not(feature = "safetensors-compare"))]
    {
        let _ = (apr_path, hf_repo, tensor_filter, threshold, json_output);
        return Err(CliError::FeatureDisabled("safetensors-compare".to_string()));
    }

    #[cfg(feature = "safetensors-compare")]
    {
        run_compare(apr_path, hf_repo, tensor_filter, threshold, json_output)
    }
}

#[cfg(feature = "safetensors-compare")]
fn run_compare(
    apr_path: &Path,
    hf_repo: &str,
    tensor_filter: Option<&str>,
    threshold: f64,
    json_output: bool,
) -> Result<(), CliError> {
    use aprender::serialization::AprReader;

    if !apr_path.exists() {
        return Err(CliError::FileNotFound(apr_path.to_path_buf()));
    }

    // Load APR model
    if !json_output {
        println!(
            "Loading APR model: {}",
            apr_path.display().to_string().cyan()
        );
    }

    let apr_reader = AprReader::open(apr_path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read APR: {e}")))?;

    // Download and load HF model
    if !json_output {
        println!("Downloading HF model: {}", hf_repo.cyan());
    }

    let hf_model = HfSafetensors::from_hub(hf_repo)
        .map_err(|e| CliError::NetworkError(format!("Failed to download HF model: {e}")))?;

    if !json_output {
        println!(
            "Found {} tensors in HF model\n",
            hf_model.tensor_names().len()
        );
    }

    // Compare tensors
    let comparisons: Vec<TensorComparison> = hf_model
        .tensor_names()
        .iter()
        .filter(|name| tensor_filter.map_or(true, |f| name.contains(f)))
        .filter_map(|name| {
            let hf_tensor = hf_model.tensor(name).ok()?;

            // Try to load corresponding APR tensor
            // HF uses different naming, try common mappings
            let apr_name = map_hf_to_apr_name(name);
            let apr_data = apr_reader.read_tensor_f32(&apr_name).ok()?;

            Some(TensorComparison::compare(
                name, &hf_tensor, &apr_data, threshold,
            ))
        })
        .collect();

    let batch = BatchComparison::from_comparisons(comparisons);

    if json_output {
        output_json(&batch);
    } else {
        output_text(&batch, threshold);
    }

    // Exit with error if any comparisons failed
    if !batch.all_passed() {
        return Err(CliError::ValidationFailed(format!(
            "{} tensors failed threshold test",
            batch.total_compared - batch.total_passed
        )));
    }

    Ok(())
}

/// Map HuggingFace tensor names to APR naming convention
fn map_hf_to_apr_name(hf_name: &str) -> String {
    // HF Whisper uses: model.decoder.layers.0.encoder_attn.q_proj.weight
    // APR uses: decoder.layers.0.encoder_attn.q_proj.weight
    let name = hf_name
        .strip_prefix("model.")
        .unwrap_or(hf_name)
        .to_string();

    // Additional mappings can be added here for other model types
    name
}

#[cfg(feature = "safetensors-compare")]
fn output_json(batch: &BatchComparison) {
    use serde::Serialize;

    #[derive(Serialize)]
    struct JsonOutput {
        total_compared: usize,
        total_passed: usize,
        shape_mismatches: usize,
        worst_tensor: Option<String>,
        worst_diff: f64,
        all_passed: bool,
        comparisons: Vec<ComparisonEntry>,
    }

    #[derive(Serialize)]
    struct ComparisonEntry {
        name: String,
        shape_match: bool,
        passes_threshold: bool,
        max_diff: Option<f64>,
        l2_distance: Option<f64>,
        cosine_similarity: Option<f64>,
    }

    let comparisons: Vec<ComparisonEntry> = batch
        .comparisons
        .iter()
        .map(|c| ComparisonEntry {
            name: c.name.clone(),
            shape_match: c.shape_match,
            passes_threshold: c.passes_threshold,
            max_diff: c.weight_diff.as_ref().map(|d| d.max_diff),
            l2_distance: c.weight_diff.as_ref().map(|d| d.l2_distance),
            cosine_similarity: c.weight_diff.as_ref().map(|d| d.cosine_similarity),
        })
        .collect();

    let output = JsonOutput {
        total_compared: batch.total_compared,
        total_passed: batch.total_passed,
        shape_mismatches: batch.shape_mismatches,
        worst_tensor: batch.worst_tensor.clone(),
        worst_diff: batch.worst_diff,
        all_passed: batch.all_passed(),
        comparisons,
    };

    if let Ok(json) = serde_json::to_string_pretty(&output) {
        println!("{json}");
    }
}

#[cfg(feature = "safetensors-compare")]
fn output_text(batch: &BatchComparison, threshold: f64) {
    println!("{}", "=".repeat(70));
    println!("{}", "HuggingFace vs APR Weight Comparison".bold());
    println!("{}", "=".repeat(70));
    println!();

    // Summary
    println!(
        "Total tensors compared: {}",
        batch.total_compared.to_string().cyan()
    );
    println!(
        "Passed threshold (< {:.0e}): {}",
        threshold,
        if batch.total_passed == batch.total_compared {
            batch.total_passed.to_string().green()
        } else {
            batch.total_passed.to_string().yellow()
        }
    );

    if batch.shape_mismatches > 0 {
        println!(
            "Shape mismatches: {}",
            batch.shape_mismatches.to_string().red()
        );
    }

    println!();

    // Show failed comparisons
    let failed: Vec<_> = batch
        .comparisons
        .iter()
        .filter(|c| !c.passes_threshold)
        .collect();

    if !failed.is_empty() {
        println!("{}", "FAILED COMPARISONS:".red().bold());
        for c in &failed {
            let diff_str = c.weight_diff.as_ref().map_or_else(
                || "shape mismatch".to_string(),
                |d| format!("max_diff={:.6}", d.max_diff),
            );
            println!("  {} {}: {}", "✗".red(), c.name, diff_str.red());
        }
        println!();
    }

    // Show worst tensor
    if let Some(worst) = &batch.worst_tensor {
        println!(
            "Worst tensor: {} (diff={:.6})",
            worst.yellow(),
            batch.worst_diff
        );
    }

    println!();

    // Final verdict
    if batch.all_passed() {
        println!("{}", "✓ All tensors match within threshold!".green().bold());
    } else {
        println!(
            "{}",
            "✗ Weight mismatch detected - check conversion!"
                .red()
                .bold()
        );
        println!();
        println!("Possible causes:");
        println!("  1. Weight transpose issue (HF is [out, in], check APR layout)");
        println!("  2. Tensor name mapping incorrect");
        println!("  3. Quantization/precision loss during conversion");
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
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
}
