//! Merge command implementation
//!
//! Implements APR-SPEC §4.9: Merge Command
//!
//! Merges multiple models into a single model using various strategies:
//! - average: Simple average of weights (ensemble)
//! - weighted: Weighted average by specified weights
//! - ties: TIES merging (planned)
//! - dare: DARE merging (planned)
//! - slerp: Spherical linear interpolation (planned)

use crate::error::{CliError, Result};
use aprender::format::{apr_merge, MergeOptions, MergeReport, MergeStrategy};
use colored::Colorize;
use humansize::{format_size, BINARY};
use std::path::{Path, PathBuf};

/// Run the merge command
pub(crate) fn run(
    files: &[PathBuf],
    strategy: &str,
    output: &Path,
    weights: Option<Vec<f32>>,
) -> Result<()> {
    // Validate we have at least 2 files
    if files.len() < 2 {
        return Err(CliError::ValidationFailed(
            "Merge requires at least 2 input models".to_string(),
        ));
    }

    // Validate all input files exist
    for file in files {
        if !file.exists() {
            return Err(CliError::FileNotFound(file.clone()));
        }
    }

    println!("{}", "=== APR Merge ===".cyan().bold());
    println!();
    println!("Merging {} models:", files.len());
    for (i, file) in files.iter().enumerate() {
        println!("  {}. {}", i + 1, file.display());
    }
    println!("Output: {}", output.display());

    // Parse merge strategy
    let merge_strategy = MergeStrategy::from_str(strategy).ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "Unknown merge strategy: {strategy}. Supported: average, weighted"
        ))
    })?;

    // Check if strategy is supported
    if !merge_strategy.is_supported() {
        return Err(CliError::ValidationFailed(format!(
            "Merge strategy '{strategy}' is not yet supported. Use 'average' or 'weighted'."
        )));
    }

    println!("Strategy: {:?}", merge_strategy);

    // Show weights if weighted merge
    if let Some(ref w) = weights {
        println!("Weights: {:?}", w);
    }
    println!();

    // Build options
    let options = MergeOptions {
        strategy: merge_strategy,
        weights,
    };

    // Run merge
    println!("{}", "Merging...".yellow());

    match apr_merge(files, output.to_path_buf(), options) {
        Ok(report) => {
            display_report(&report);
            Ok(())
        }
        Err(e) => {
            println!();
            println!("{}", "Merge failed".red().bold());
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

/// Display merge report
fn display_report(report: &MergeReport) {
    println!();
    println!("{}", "=== Merge Report ===".cyan().bold());
    println!();
    println!("Models merged:  {}", report.model_count);
    println!("Tensors:        {}", report.tensor_count);
    println!(
        "Output size:    {}",
        format_size(report.output_size, BINARY)
    );
    println!("Strategy:       {:?}", report.strategy);

    if let Some(ref weights) = report.weights_used {
        println!(
            "Weights used:   {:?}",
            weights
                .iter()
                .map(|w| format!("{:.3}", w))
                .collect::<Vec<_>>()
        );
    }

    println!();
    println!("{}", "Merge successful".green().bold());
}
