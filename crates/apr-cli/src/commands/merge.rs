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
            println!("Weights: {:?}", w);
            Ok(Some(w))
        }
        MergeStrategy::Slerp | MergeStrategy::Dare => {
            // SLERP uses first weight as interpolation t (default 0.5)
            // DARE uses weights for per-model scaling (optional)
            if let Some(ref w) = weights {
                println!("Weights: {:?}", w);
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

/// Run the merge command
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    files: &[PathBuf],
    strategy: &str,
    output: &Path,
    weights: Option<Vec<f32>>,
    base_model: Option<PathBuf>,
    drop_rate: f32,
    density: f32,
    seed: u64,
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

    output::header("APR Merge");
    let mut input_pairs: Vec<(&str, String)> = Vec::new();
    for (i, file) in files.iter().enumerate() {
        input_pairs.push(("Input", format!("{}. {}", i + 1, file.display())));
    }
    input_pairs.push(("Output", output.display().to_string()));
    println!("{}", output::kv_table(&input_pairs));

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

    println!("Strategy: {merge_strategy:?}");

    let validated_weights = validate_merge_weights(merge_strategy, weights, files.len(), strategy)?;
    println!();

    // Build options
    let options = MergeOptions {
        strategy: merge_strategy,
        weights: validated_weights,
        base_model,
        drop_rate,
        density,
        seed,
    };

    // Run merge
    output::pipeline_stage("Merging", output::StageStatus::Running);

    match apr_merge(files, output.to_path_buf(), options) {
        Ok(report) => {
            display_report(&report);
            Ok(())
        }
        Err(e) => {
            println!();
            println!("  {}", output::badge_fail("Merge failed"));
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
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
mod tests {
    use super::*;
    use aprender::format::MergeReport;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // Validation Error Tests
    // ========================================================================

    #[test]
    fn test_run_insufficient_files() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            &[file.path().to_path_buf()],
            "average",
            Path::new("/tmp/merged.apr"),
            None,
            None,
            0.9,
            0.2,
            42,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("at least 2"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_empty_files() {
        let result = run(&[], "average", Path::new("/tmp/merged.apr"), None, None, 0.9, 0.2, 42);
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("at least 2"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            &[
                PathBuf::from("/nonexistent/model1.apr"),
                PathBuf::from("/nonexistent/model2.apr"),
            ],
            "average",
            Path::new("/tmp/merged.apr"),
            None,
            None,
            0.9,
            0.2,
            42,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_second_file_not_found() {
        let file1 = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            &[
                file1.path().to_path_buf(),
                PathBuf::from("/nonexistent/model2.apr"),
            ],
            "average",
            Path::new("/tmp/merged.apr"),
            None,
            None,
            0.9,
            0.2,
            42,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_unknown_strategy() {
        let file1 = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let file2 = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            &[file1.path().to_path_buf(), file2.path().to_path_buf()],
            "unknown_strategy",
            Path::new("/tmp/merged.apr"),
            None,
            None,
            0.9,
            0.2,
            42,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown merge strategy"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_ties_without_base_model() {
        let file1 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        let file2 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");

        let result = run(
            &[file1.path().to_path_buf(), file2.path().to_path_buf()],
            "ties",
            Path::new("/tmp/merged.safetensors"),
            None,
            None, // no base model
            0.9,
            0.2,
            42,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("base-model") || msg.contains("base_model") || msg.contains("TIES"));
            }
            _ => panic!("Expected ValidationFailed error for missing base model"),
        }
    }

    #[test]
    fn test_run_dare_without_base_model() {
        let file1 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        let file2 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");

        let result = run(
            &[file1.path().to_path_buf(), file2.path().to_path_buf()],
            "dare",
            Path::new("/tmp/merged.safetensors"),
            None,
            None, // no base model
            0.9,
            0.2,
            42,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("base-model") || msg.contains("base_model") || msg.contains("DARE"));
            }
            _ => panic!("Expected ValidationFailed error for missing base model"),
        }
    }

    #[test]
    fn test_run_slerp_with_three_models() {
        let file1 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        let file2 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        let file3 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");

        let result = run(
            &[
                file1.path().to_path_buf(),
                file2.path().to_path_buf(),
                file3.path().to_path_buf(),
            ],
            "slerp",
            Path::new("/tmp/merged.safetensors"),
            None,
            None,
            0.9,
            0.2,
            42,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // Display Report Tests
    // ========================================================================

    #[test]
    fn test_display_report_basic() {
        let report = MergeReport {
            model_count: 2,
            tensor_count: 100,
            output_size: 1024 * 1024 * 100, // 100MB
            strategy: MergeStrategy::Average,
            weights_used: None,
        };
        display_report(&report);
    }

    #[test]
    fn test_display_report_with_weights() {
        let report = MergeReport {
            model_count: 3,
            tensor_count: 200,
            output_size: 1024 * 1024 * 500, // 500MB
            strategy: MergeStrategy::Weighted,
            weights_used: Some(vec![0.5, 0.3, 0.2]),
        };
        display_report(&report);
    }

    #[test]
    fn test_display_report_large_merge() {
        let report = MergeReport {
            model_count: 5,
            tensor_count: 1000,
            output_size: 7 * 1024 * 1024 * 1024, // 7GB
            strategy: MergeStrategy::Average,
            weights_used: None,
        };
        display_report(&report);
    }

    // ========================================================================
    // Invalid File Content Tests
    // ========================================================================

    #[test]
    fn test_run_invalid_apr_files() {
        let mut file1 = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let mut file2 = NamedTempFile::with_suffix(".apr").expect("create temp file");

        file1.write_all(b"not valid APR").expect("write to file");
        file2
            .write_all(b"also not valid APR")
            .expect("write to file");

        let result = run(
            &[file1.path().to_path_buf(), file2.path().to_path_buf()],
            "average",
            Path::new("/tmp/merged.apr"),
            None,
            None,
            0.9,
            0.2,
            42,
        );
        // Should fail because files are not valid APR
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_weights() {
        let mut file1 = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let mut file2 = NamedTempFile::with_suffix(".apr").expect("create temp file");

        file1.write_all(b"data1").expect("write");
        file2.write_all(b"data2").expect("write");

        let result = run(
            &[file1.path().to_path_buf(), file2.path().to_path_buf()],
            "weighted",
            Path::new("/tmp/merged.apr"),
            Some(vec![0.7, 0.3]),
            None,
            0.9,
            0.2,
            42,
        );
        // Will fail at actual merge, but tests weight parsing path
        assert!(result.is_err());
    }
}
