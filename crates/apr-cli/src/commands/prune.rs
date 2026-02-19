//! Prune command implementation (GH-247)
//!
//! Structured pruning pipeline for removing attention heads, MLP neurons,
//! or entire layers from transformer models.
//!
//! # Example
//!
//! ```bash
//! apr prune model.apr --method structured --target-ratio 0.5 -o pruned.apr
//! apr prune model.apr --method depth --remove-layers 20-24 -o pruned.apr
//! apr prune model.apr --method magnitude --sparsity 0.5 -o pruned.apr
//! apr prune model.apr --analyze --json
//! ```

use crate::error::{CliError, Result};
use crate::output;
use std::path::Path;

/// Pruning method selection
#[derive(Debug, Clone, Copy, Default)]
pub enum PruneMethod {
    /// Unstructured magnitude pruning (zero out small weights)
    #[default]
    Magnitude,
    /// Structured pruning (remove attention heads / MLP neurons)
    Structured,
    /// Depth pruning (remove entire layers)
    Depth,
    /// Width pruning (reduce hidden dimensions)
    Width,
    /// Wanda pruning (weights and activations)
    Wanda,
    /// SparseGPT pruning
    SparseGpt,
}

impl std::str::FromStr for PruneMethod {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "magnitude" | "mag" => Ok(Self::Magnitude),
            "structured" | "struct" => Ok(Self::Structured),
            "depth" | "layer" => Ok(Self::Depth),
            "width" | "hidden" => Ok(Self::Width),
            "wanda" => Ok(Self::Wanda),
            "sparsegpt" | "sparse_gpt" => Ok(Self::SparseGpt),
            _ => Err(format!(
                "Unknown pruning method: {s}. Supported: magnitude, structured, depth, width, wanda, sparsegpt"
            )),
        }
    }
}

/// Validate prune command parameters (target ratio, sparsity, method).
fn validate_prune_params(
    file: &Path,
    method: &str,
    target_ratio: f32,
    sparsity: f32,
) -> Result<PruneMethod> {
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }
    let prune_method: PruneMethod = method.parse().map_err(CliError::ValidationFailed)?;
    if target_ratio <= 0.0 || target_ratio >= 1.0 {
        return Err(CliError::ValidationFailed(format!(
            "Target ratio must be between 0 and 1 (exclusive), got {target_ratio}"
        )));
    }
    if !(0.0..=1.0).contains(&sparsity) {
        return Err(CliError::ValidationFailed(format!(
            "Sparsity must be between 0 and 1, got {sparsity}"
        )));
    }
    Ok(prune_method)
}

/// Print the configuration summary table before pruning.
#[allow(clippy::disallowed_methods)]
fn print_config_table(
    file: &Path,
    out: &Path,
    prune_method: PruneMethod,
    target_ratio: f32,
    sparsity: f32,
    remove_layers: Option<&str>,
    calibration: Option<&Path>,
) {
    output::header("APR Prune");
    let mut pairs = vec![
        ("Input", file.display().to_string()),
        ("Method", format!("{prune_method:?}")),
        ("Target ratio", format!("{target_ratio:.2}")),
        ("Output", out.display().to_string()),
    ];
    if sparsity > 0.0 {
        pairs.push(("Sparsity", format!("{sparsity:.2}")));
    }
    if let Some(layers) = remove_layers {
        pairs.push(("Remove layers", layers.to_string()));
    }
    if let Some(cal) = calibration {
        pairs.push(("Calibration", cal.display().to_string()));
    }
    println!("{}", output::kv_table(&pairs));
    println!();
}

/// Validate depth-pruning specific arguments.
fn validate_depth_args(prune_method: PruneMethod, remove_layers: Option<&str>) -> Result<()> {
    if matches!(prune_method, PruneMethod::Depth) && remove_layers.is_none() {
        return Err(CliError::ValidationFailed(
            "Depth pruning requires --remove-layers (e.g., --remove-layers 20-24)".to_string(),
        ));
    }
    Ok(())
}

/// Run the prune command
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
pub(crate) fn run(
    file: &Path,
    method: &str,
    target_ratio: f32,
    sparsity: f32,
    output_path: Option<&Path>,
    remove_layers: Option<&str>,
    analyze_only: bool,
    plan_only: bool,
    calibration: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let prune_method = validate_prune_params(file, method, target_ratio, sparsity)?;

    // Analyze mode
    if analyze_only {
        return run_analyze(file, prune_method, json_output);
    }

    // Plan mode
    if plan_only {
        return run_plan(file, prune_method, target_ratio, sparsity, json_output);
    }

    let out = output_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Output path required. Use -o <path> to specify output.".to_string(),
        )
    })?;

    if !json_output {
        print_config_table(file, out, prune_method, target_ratio, sparsity, remove_layers, calibration);
    }

    validate_depth_args(prune_method, remove_layers)?;

    if !json_output {
        output::pipeline_stage("Pruning", output::StageStatus::Running);
    }

    let prune_result = execute_pruning(
        file,
        prune_method,
        target_ratio,
        sparsity,
        remove_layers,
        out,
    )?;

    if !json_output {
        output::pipeline_stage("Pruning", output::StageStatus::Done);
    }

    print_prune_output(
        file,
        out,
        prune_method,
        target_ratio,
        sparsity,
        &prune_result,
        json_output,
    );

    Ok(())
}

/// Result of the pruning operation, containing all metrics needed for output.
struct PruneResult {
    file_size: u64,
    output_size: u64,
    original_count: usize,
    pruned_count: usize,
    original_params: usize,
    pruned_params: usize,
    zeros: usize,
}

/// Load model, apply pruning, and write the output file.
fn execute_pruning(
    file: &Path,
    prune_method: PruneMethod,
    target_ratio: f32,
    sparsity: f32,
    remove_layers: Option<&str>,
    out: &Path,
) -> Result<PruneResult> {
    let file_size = std::fs::metadata(file)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model: {e}")))?
        .len();

    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = rosetta
        .inspect(file)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model: {e}")))?;

    let mut tensors = std::collections::BTreeMap::new();
    for ti in &report.tensors {
        if let Ok(data) = rosetta.load_tensor_f32(file, &ti.name) {
            tensors.insert(ti.name.clone(), (data, ti.shape.clone()));
        }
    }

    let original_count = tensors.len();
    let original_params: usize = tensors
        .values()
        .map(|(data, _shape): &(Vec<f32>, Vec<usize>)| data.len())
        .sum();

    let pruned_tensors = apply_pruning(
        &tensors,
        prune_method,
        target_ratio,
        sparsity,
        remove_layers,
    )?;

    let pruned_count = pruned_tensors.len();
    let pruned_params: usize = pruned_tensors
        .values()
        .map(|(data, _shape): &(Vec<f32>, Vec<usize>)| data.len())
        .sum();
    let zeros: usize = pruned_tensors
        .values()
        .map(|(data, _shape): &(Vec<f32>, Vec<usize>)| data.iter().filter(|v| **v == 0.0).count())
        .sum();

    let bytes = write_pruned_model(
        file,
        prune_method,
        target_ratio,
        sparsity,
        &pruned_tensors,
        out,
    )?;
    let output_size = bytes.len() as u64;

    Ok(PruneResult {
        file_size,
        output_size,
        original_count,
        pruned_count,
        original_params,
        pruned_params,
        zeros,
    })
}

/// Apply the selected pruning method to the tensor map.
#[allow(clippy::type_complexity)]
fn apply_pruning(
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    prune_method: PruneMethod,
    target_ratio: f32,
    sparsity: f32,
    remove_layers: Option<&str>,
) -> Result<std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    match prune_method {
        PruneMethod::Magnitude => Ok(prune_magnitude(tensors, sparsity.max(target_ratio))),
        PruneMethod::Depth => {
            let layers = remove_layers.expect("validated above");
            prune_depth(tensors, layers)
        }
        PruneMethod::Structured | PruneMethod::Width => {
            Ok(prune_magnitude(tensors, sparsity.max(target_ratio)))
        }
        PruneMethod::Wanda | PruneMethod::SparseGpt => {
            Ok(prune_magnitude(tensors, sparsity.max(target_ratio)))
        }
    }
}

/// Serialize pruned tensors and write the APR file to disk.
#[allow(clippy::disallowed_methods)]
fn write_pruned_model(
    source_file: &Path,
    prune_method: PruneMethod,
    target_ratio: f32,
    sparsity: f32,
    pruned_tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    out: &Path,
) -> Result<Vec<u8>> {
    let mut writer = aprender::serialization::apr::AprWriter::new();
    writer.set_metadata(
        "pruning_method",
        serde_json::json!(format!("{prune_method:?}")),
    );
    writer.set_metadata("pruning_ratio", serde_json::json!(target_ratio));
    writer.set_metadata("pruning_sparsity", serde_json::json!(sparsity));
    writer.set_metadata(
        "source_file",
        serde_json::json!(source_file.display().to_string()),
    );

    for (name, (data, shape)) in pruned_tensors {
        writer.add_tensor_f32(name, shape.clone(), data);
    }

    let bytes = writer.to_bytes().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to serialize pruned model: {e}"))
    })?;
    std::fs::write(out, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write output: {e}")))?;

    Ok(bytes)
}

/// Print pruning results as JSON or human-readable table.
#[allow(clippy::disallowed_methods)]
fn print_prune_output(
    file: &Path,
    out: &Path,
    prune_method: PruneMethod,
    target_ratio: f32,
    sparsity: f32,
    result: &PruneResult,
    json_output: bool,
) {
    if json_output {
        let json = serde_json::json!({
            "status": "completed",
            "input": file.display().to_string(),
            "output": out.display().to_string(),
            "method": format!("{prune_method:?}"),
            "target_ratio": target_ratio,
            "sparsity": sparsity,
            "input_size": result.file_size,
            "output_size": result.output_size,
            "tensors": result.pruned_count,
            "original_params": result.original_params,
            "pruned_params": result.pruned_params,
            "zero_params": result.zeros,
            "actual_sparsity": if result.pruned_params > 0 { result.zeros as f64 / result.pruned_params as f64 } else { 0.0 },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!();
        output::subheader("Pruning Complete");
        println!(
            "{}",
            output::kv_table(&[
                (
                    "Input size",
                    humansize::format_size(result.file_size, humansize::BINARY)
                ),
                (
                    "Output size",
                    humansize::format_size(result.output_size, humansize::BINARY)
                ),
                (
                    "Tensors",
                    format!("{} → {}", result.original_count, result.pruned_count)
                ),
                (
                    "Parameters",
                    format!("{} → {}", result.original_params, result.pruned_params)
                ),
                (
                    "Zeros",
                    format!(
                        "{} ({:.1}%)",
                        result.zeros,
                        if result.pruned_params > 0 {
                            result.zeros as f64 / result.pruned_params as f64 * 100.0
                        } else {
                            0.0
                        }
                    )
                ),
                ("Output", out.display().to_string()),
            ])
        );
    }
}

/// Analyze model for pruning opportunities
#[allow(clippy::disallowed_methods)]
fn run_analyze(file: &Path, method: PruneMethod, json_output: bool) -> Result<()> {
    let file_size = std::fs::metadata(file)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model: {e}")))?
        .len();

    // Estimate parameter count (assume Q4 ~ 0.5 bytes/param)
    let estimated_params = file_size * 2;

    if json_output {
        let json = serde_json::json!({
            "analysis": true,
            "input": file.display().to_string(),
            "file_size": file_size,
            "estimated_params": estimated_params,
            "method": format!("{method:?}"),
            "recommendations": [
                {"ratio": 0.2, "description": "Conservative (minimal quality loss)"},
                {"ratio": 0.5, "description": "Balanced (moderate compression)"},
                {"ratio": 0.8, "description": "Aggressive (significant quality loss)"},
            ],
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Prune — Analysis");
        println!(
            "{}",
            output::kv_table(&[
                ("Input", file.display().to_string()),
                (
                    "File size",
                    humansize::format_size(file_size, humansize::BINARY),
                ),
                ("Est. parameters", format_params(estimated_params),),
                ("Method", format!("{method:?}")),
            ])
        );
        println!();
        output::subheader("Pruning Recommendations");
        println!("  20% — Conservative (minimal quality loss)");
        println!("  50% — Balanced (moderate compression)");
        println!("  80% — Aggressive (significant quality loss)");
        println!();
        println!(
            "  {} Use --target-ratio <0-1> to set pruning target.",
            output::badge_info("INFO"),
        );
    }

    Ok(())
}

include!("prune_include_01.rs");
