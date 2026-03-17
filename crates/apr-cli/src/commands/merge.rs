//! Merge command implementation
//!
//! Implements APR-SPEC §4.9: Merge Command
//!
//! Merges multiple models into a single model using various strategies:
//! - average: Simple average of weights (ensemble)
//! - weighted: Weighted average by specified weights
//! - slerp: Spherical linear interpolation (2 models)
//! - ties: TIES merging (trim, elect sign, merge)
//! - dare: DARE merging (drop and rescale)

use crate::error::{CliError, Result};
use crate::output;
use aprender::format::{apr_merge, MergeOptions, MergeReport, MergeStrategy};
use humansize::{format_size, BINARY};
use std::path::{Path, PathBuf};

/// Validate and resolve merge weights based on strategy (BUG-MERGE-001..004).
fn validate_merge_weights(
    merge_strategy: MergeStrategy,
    weights: Option<Vec<f32>>,
    file_count: usize,
    strategy_name: &str,
    json_output: bool,
) -> Result<Option<Vec<f32>>> {
    match merge_strategy {
        MergeStrategy::Weighted => {
            let w = weights.ok_or_else(|| {
                CliError::ValidationFailed(
                    "Weighted merge strategy requires --weights argument. Example: --weights 0.7 0.3"
                        .to_string(),
                )
            })?;
            validate_weight_values(&w, file_count)?;
            // GH-514: Only print weights in text mode to avoid contaminating JSON output
            if !json_output {
                println!("Weights: {:?}", w);
            }
            Ok(Some(w))
        }
        MergeStrategy::Slerp | MergeStrategy::Dare => {
            // SLERP uses first weight as interpolation t (default 0.5)
            // DARE uses weights for per-model scaling (optional)
            if let Some(ref w) = weights {
                // GH-514: Only print weights in text mode
                if !json_output {
                    println!("Weights: {:?}", w);
                }
            }
            Ok(weights)
        }
        _ => {
            if weights.is_some() {
                eprintln!(
                    "  {} --weights argument ignored for '{}' strategy.",
                    output::badge_warn("WARN"),
                    strategy_name
                );
            }
            Ok(None)
        }
    }
}

/// Validate individual weight values: count, non-negative, finite, sum warning.
fn validate_weight_values(w: &[f32], file_count: usize) -> Result<()> {
    if w.len() != file_count {
        return Err(CliError::ValidationFailed(format!(
            "Weight count ({}) must match file count ({file_count}). Provide one weight per input model.",
            w.len()
        )));
    }
    for (i, &weight) in w.iter().enumerate() {
        if weight < 0.0 {
            return Err(CliError::ValidationFailed(format!(
                "Weight {} is negative ({weight:.3}). All weights must be >= 0.",
                i + 1
            )));
        }
        if !weight.is_finite() {
            return Err(CliError::ValidationFailed(format!(
                "Weight {} is not a valid number ({weight:.3}). Use finite values.",
                i + 1
            )));
        }
    }
    let sum: f32 = w.iter().sum();
    if (sum - 1.0).abs() > 0.01 {
        eprintln!(
            "  {} Weights sum to {sum:.3}, not 1.0. Results will be scaled accordingly.",
            output::badge_warn("WARN"),
        );
    }
    Ok(())
}

/// Validate merge inputs: at least 2 files and all must exist.
fn validate_merge_inputs(files: &[PathBuf]) -> Result<()> {
    if files.len() < 2 {
        return Err(CliError::ValidationFailed(
            "Merge requires at least 2 input models".to_string(),
        ));
    }
    for file in files {
        if !file.exists() {
            return Err(CliError::FileNotFound(file.clone()));
        }
    }
    Ok(())
}

/// Display the merge input/output summary header (non-JSON mode).
fn display_merge_header(files: &[PathBuf], output_path: &Path) {
    output::header("APR Merge");
    let mut input_pairs: Vec<(&str, String)> = Vec::new();
    for (i, file) in files.iter().enumerate() {
        input_pairs.push(("Input", format!("{}. {}", i + 1, file.display())));
    }
    input_pairs.push(("Output", output_path.display().to_string()));
    println!("{}", output::kv_table(&input_pairs));
}

/// Run the merge plan (dry-run validation).
#[allow(clippy::too_many_arguments, clippy::disallowed_methods)]
pub(crate) fn run_plan(
    files: &[PathBuf],
    strategy: &str,
    output: Option<&Path>,
    weights: Option<Vec<f32>>,
    base_model: Option<PathBuf>,
    drop_rate: f32,
    density: f32,
    seed: u64,
    json_output: bool,
) -> Result<()> {
    validate_merge_inputs(files)?;

    let merge_strategy: MergeStrategy = strategy.parse().map_err(|_| {
        CliError::ValidationFailed(format!(
            "Unknown merge strategy: {strategy}. Supported: average, weighted, slerp, ties, dare"
        ))
    })?;

    if !merge_strategy.is_supported() {
        return Err(CliError::ValidationFailed(format!(
            "Merge strategy '{strategy}' is not yet supported."
        )));
    }

    let validated_weights =
        validate_merge_weights(merge_strategy, weights, files.len(), strategy, json_output)?;

    // Compute total input size
    let total_input_size: u64 = files
        .iter()
        .filter_map(|f| std::fs::metadata(f).ok())
        .map(|m| m.len())
        .sum();

    if json_output {
        let json = serde_json::json!({
            "plan": true,
            "status": "valid",
            "inputs": files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>(),
            "output": output.map(|p| p.display().to_string()).unwrap_or_else(|| "(not specified)".into()),
            "strategy": strategy,
            "model_count": files.len(),
            "total_input_size": total_input_size,
            "weights": validated_weights,
            "base_model": base_model.as_ref().map(|p| p.display().to_string()),
            "drop_rate": drop_rate,
            "density": density,
            "seed": seed,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Merge — Plan");
        let mut pairs: Vec<(&str, String)> = Vec::new();
        for (i, file) in files.iter().enumerate() {
            let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
            pairs.push((
                "Input",
                format!(
                    "{}. {} ({})",
                    i + 1,
                    file.display(),
                    humansize::format_size(size, BINARY)
                ),
            ));
        }
        if let Some(out) = output {
            pairs.push(("Output", out.display().to_string()));
        }
        pairs.push(("Strategy", format!("{merge_strategy:?}")));
        pairs.push(("Models", files.len().to_string()));
        pairs.push((
            "Total input",
            humansize::format_size(total_input_size, BINARY),
        ));
        if let Some(ref w) = validated_weights {
            let w_str = w
                .iter()
                .map(|w| format!("{w:.3}"))
                .collect::<Vec<_>>()
                .join(", ");
            pairs.push(("Weights", w_str));
        }
        if let Some(ref base) = base_model {
            pairs.push(("Base model", base.display().to_string()));
        }
        println!("{}", output::kv_table(&pairs));
        println!();
        println!("  {}", output::badge_pass("Plan valid — ready to merge"));
    }
    Ok(())
}

/// Run the merge command
#[allow(clippy::too_many_arguments, clippy::disallowed_methods)]
pub(crate) fn run(
    files: &[PathBuf],
    strategy: &str,
    output: Option<&Path>,
    weights: Option<Vec<f32>>,
    base_model: Option<PathBuf>,
    drop_rate: f32,
    density: f32,
    seed: u64,
    json_output: bool,
    plan: bool,
) -> Result<()> {
    if plan {
        return run_plan(
            files,
            strategy,
            output,
            weights,
            base_model,
            drop_rate,
            density,
            seed,
            json_output,
        );
    }

    let output = output.ok_or_else(|| {
        CliError::ValidationFailed("--output is required for merge execution".into())
    })?;

    validate_merge_inputs(files)?;

    if !json_output {
        display_merge_header(files, output);
    }

    // Parse merge strategy
    let merge_strategy: MergeStrategy = strategy.parse().map_err(|_| {
        CliError::ValidationFailed(format!(
            "Unknown merge strategy: {strategy}. Supported: average, weighted, slerp, ties, dare"
        ))
    })?;

    // Check if strategy is supported
    if !merge_strategy.is_supported() {
        return Err(CliError::ValidationFailed(format!(
            "Merge strategy '{strategy}' is not yet supported."
        )));
    }

    if !json_output {
        println!("Strategy: {merge_strategy:?}");
    }

    let validated_weights =
        validate_merge_weights(merge_strategy, weights, files.len(), strategy, json_output)?;
    if !json_output {
        println!();
    }

    // Build options
    let options = MergeOptions {
        strategy: merge_strategy,
        weights: validated_weights,
        base_model,
        drop_rate,
        density,
        seed,
        scales: None,
        outlier_k: 3.0,
        layer_ranges: None,
    };

    // Run merge
    if !json_output {
        output::pipeline_stage("Merging", output::StageStatus::Running);
    }

    match apr_merge(files, output.to_path_buf(), options) {
        Ok(report) => {
            if json_output {
                display_report_json(&report, files, strategy, output);
            } else {
                display_report(&report);
            }
            Ok(())
        }
        Err(e) => {
            if !json_output {
                println!();
                println!("  {}", output::badge_fail("Merge failed"));
            }
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

/// Display merge report as JSON
#[allow(clippy::disallowed_methods)]
fn display_report_json(report: &MergeReport, files: &[PathBuf], strategy: &str, output: &Path) {
    let json = serde_json::json!({
        "status": "success",
        "inputs": files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>(),
        "output": output.display().to_string(),
        "strategy": strategy,
        "model_count": report.model_count,
        "tensor_count": report.tensor_count,
        "output_size": report.output_size,
        "weights_used": report.weights_used,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

/// Display merge report
fn display_report(report: &MergeReport) {
    println!();
    output::subheader("Merge Report");

    let mut pairs: Vec<(&str, String)> = vec![
        ("Models merged", report.model_count.to_string()),
        ("Tensors", output::count_fmt(report.tensor_count)),
        ("Output size", format_size(report.output_size, BINARY)),
        ("Strategy", format!("{:?}", report.strategy)),
    ];
    if let Some(ref weights) = report.weights_used {
        let w_str = weights
            .iter()
            .map(|w| format!("{w:.3}"))
            .collect::<Vec<_>>()
            .join(", ");
        pairs.push(("Weights used", w_str));
    }

    println!("{}", output::kv_table(&pairs));
    println!();
    println!("  {}", output::badge_pass("Merge successful"));
}

#[cfg(test)]
#[path = "merge_tests.rs"]
mod tests;
