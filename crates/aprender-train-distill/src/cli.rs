//! Command implementations for the end-to-end distillation pipeline.
//!
//! These functions were the body of the `aprender-train-distill` binary's
//! `main.rs`. That binary is gone (APR-MONO Rule 1: `apr` is the only
//! user-facing binary); the capability is reachable as
//! `apr train distill <verb>`, which calls exactly these entry points.
//!
//! Note that the SafeTensors / GGUF / APR export writers were stranded in the
//! binary's `main.rs` — no library caller could reach them. They now live in
//! [`crate::export`], and this module drives them.

// The `serde_json::json!` macro expands to code containing `.unwrap()`, which
// trips clippy::disallowed_methods at the macro invocation site even though no
// author-written unwrap exists. Scope the allow to this presentation module.
#![allow(clippy::disallowed_methods)]

use crate::{config::DistillConfig, estimate_memory, run, validation::ConfigValidator};
use std::path::{Path, PathBuf};

/// Run the distillation pipeline described by a config file.
///
/// With `dry_run`, validates the config and prints a memory estimate without
/// training. `output` overrides `output.dir` in the config when given.
///
/// # Errors
///
/// Propagates config load failures, validation failures, and pipeline failures.
pub fn run_pipeline(
    config_path: &Path,
    output: Option<PathBuf>,
    dry_run: bool,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::header("apr train distill")
        );
    }

    // Load configuration
    let mut config = DistillConfig::from_file(config_path)?;

    // Override output if specified
    if let Some(out) = output {
        config.output.dir = out;
    }

    // Validate
    ConfigValidator::validate(&config)?;

    if dry_run {
        if !cli.is_quiet() {
            println!(
                "{}",
                entrenar_common::cli::styles::success("Configuration valid")
            );

            let estimate = estimate_memory(&config)?;
            println!("\n{}", estimate.to_human_readable());
        }
        return Ok(());
    }

    // Run pipeline
    let result = run(&config)?;

    if !cli.is_quiet() {
        println!(
            "\n{}",
            entrenar_common::cli::styles::success("Distillation complete")
        );
        println!("  Output: {}", result.output_path.display());
        println!("  Duration: {:.1}s", result.duration_seconds);
        println!(
            "  Improvement: {:.1}%",
            result.metrics.improvement_ratio() * 100.0
        );
    }

    Ok(())
}

/// Estimate distillation memory for a teacher/student pair.
///
/// `student` defaults to `teacher` when omitted, matching the pre-migration
/// binary.
///
/// # Errors
///
/// Propagates validation and estimation failures.
pub fn run_estimate(
    teacher: &str,
    student: Option<String>,
    batch_size: u32,
    seq_len: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let student_id = student.unwrap_or_else(|| teacher.to_string());

    let mut config = DistillConfig::minimal(teacher, &student_id);
    config.training.batch_size = batch_size;
    config.dataset.max_length = seq_len;

    let estimate = estimate_memory(&config)?;

    if cli.format == entrenar_common::OutputFormat::Json {
        println!(
            "{}",
            serde_json::json!({
                "model_bytes": estimate.model_bytes,
                "activation_bytes": estimate.activation_bytes,
                "optimizer_bytes": estimate.optimizer_bytes,
                "total_bytes": estimate.total_bytes,
                "fits_in_vram": estimate.fits_in_vram,
                "recommended_batch_size": estimate.recommended_batch_size,
            })
        );
    } else {
        println!("{}", estimate.to_human_readable());
    }

    Ok(())
}

/// Validate a distillation config file without running anything.
///
/// # Errors
///
/// Propagates config load failures and validation failures.
pub fn run_validate(config_path: &Path, cli: &entrenar_common::Cli) -> entrenar_common::Result<()> {
    let config = DistillConfig::from_file(config_path)?;
    ConfigValidator::validate(&config)?;

    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::success("Configuration valid")
        );
    }

    Ok(())
}

/// Export a trained student to SafeTensors, GGUF or APR.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::ModelNotFound`] when `input` does
/// not exist, [`entrenar_common::EntrenarError::UnsupportedFormat`] for an
/// unknown `format`, and propagates writer failures.
pub fn run_export(
    input: &Path,
    format: &str,
    output: &Path,
    quantize: &str,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !input.exists() {
        return Err(entrenar_common::EntrenarError::ModelNotFound {
            path: input.to_path_buf(),
        });
    }

    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::info(&format!(
                "Exporting {} to {} format (quantize: {})",
                input.display(),
                format,
                quantize
            ))
        );
    }

    let (weights, shapes) = crate::load_safetensors_weights(input)?;

    crate::export::ensure_parent_dir(output)?;
    crate::export::dispatch_export(format, &weights, &shapes, output, quantize)?;

    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::success(&format!("Exported to {}", output.display()))
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_export_refuses_missing_input() {
        let cli = entrenar_common::Cli::new().with_verbosity(0);
        let missing = Path::new("/nonexistent/student-that-does-not-exist.safetensors");
        let err = match run_export(
            missing,
            "safetensors",
            Path::new("/tmp/out.safetensors"),
            "none",
            &cli,
        ) {
            Ok(()) => panic!("run_export must refuse an input path that does not exist"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("student-that-does-not-exist"),
            "refusal must quote the missing input path, got: {err}"
        );
    }

    #[test]
    fn run_validate_refuses_missing_config() {
        let cli = entrenar_common::Cli::new().with_verbosity(0);
        let missing = Path::new("/nonexistent/distill-config-that-does-not-exist.yaml");
        assert!(
            run_validate(missing, &cli).is_err(),
            "run_validate must refuse a config path that does not exist"
        );
    }
}
