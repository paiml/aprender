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
    use aprender::format::rosetta::{FormatType, RosettaStone};
    use aprender::serialization::AprReader;

    if !apr_path.exists() {
        return Err(CliError::FileNotFound(apr_path.to_path_buf()));
    }

    // PMAT-267: Detect format and load via appropriate reader
    let format = FormatType::from_magic(apr_path)
        .or_else(|_| FormatType::from_extension(apr_path))
        .map_err(|e| CliError::InvalidFormat(format!("Unsupported format: {e}")))?;

    if !json_output {
        println!(
            "Loading local model: {} ({:?})",
            apr_path.display().to_string().cyan(),
            format
        );
    }

    // For non-APR formats, use RosettaStone to read tensor data
    let rosetta = RosettaStone::new();
    let local_report = rosetta
        .inspect(apr_path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to inspect local model: {e}")))?;

    // APR reader (only works for APR files)
    let apr_reader = AprReader::open(apr_path).ok();

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

    // PMAT-267: Compare tensors — use APR reader when available, RosettaStone otherwise
    let local_tensor_names: Vec<String> = local_report
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .collect();

    let comparisons: Vec<TensorComparison> = hf_model
        .tensor_names()
        .iter()
        .filter(|name| tensor_filter.map_or(true, |f| name.contains(f)))
        .filter_map(|name| {
            let hf_tensor = hf_model.tensor(name).ok()?;

            // Try to load corresponding local tensor
            // HF uses different naming, try common mappings
            let local_name = map_hf_to_apr_name(name);

            // Try APR reader first (fastest), then RosettaStone
            let local_data = if let Some(ref reader) = apr_reader {
                reader.read_tensor_f32(&local_name).ok()
            } else {
                // For GGUF/SafeTensors, check if tensor exists in report
                if local_tensor_names.contains(&local_name) {
                    rosetta.load_tensor_f32(apr_path, &local_name).ok()
                } else {
                    None
                }
            }?;

            Some(TensorComparison::compare(
                name,
                &hf_tensor,
                &local_data,
                threshold,
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
#[path = "compare_hf_tests.rs"]
mod tests;
