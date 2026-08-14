//! Command implementations for the LoRA/QLoRA planner.
//!
//! These functions were the body of the `aprender-train-lora` binary's
//! `main.rs`. That binary is gone (APR-MONO Rule 1: `apr` is the only
//! user-facing binary); the capability is reachable as
//! `apr train lora <verb>`, which calls exactly these entry points.

// The `serde_json::json!` macro expands to code containing `.unwrap()`, which
// trips clippy::disallowed_methods at the macro invocation site even though no
// author-written unwrap exists. Scope the allow to this presentation module.
#![allow(clippy::disallowed_methods)]

use crate::{plan, Method};
use entrenar_common::cli::styles;
use entrenar_common::output::{format_bytes, format_number, TableBuilder};
use std::path::Path;

/// Parse a model size string into a parameter count.
///
/// Accepts a `B`/`b` suffix (billions), an `M`/`m` suffix (millions), or a bare
/// integer. Unparseable values fall back to the defaults the pre-migration
/// binary used: 7.0 for `B`, 350.0 for `M`, and 7 000 000 000 for a bare value.
#[must_use]
pub fn parse_model_size(model: &str) -> u64 {
    let lower = model.to_lowercase();
    if lower.ends_with('b') {
        let num: f64 = lower.trim_end_matches('b').parse().unwrap_or(7.0);
        (num * 1e9) as u64
    } else if lower.ends_with('m') {
        let num: f64 = lower.trim_end_matches('m').parse().unwrap_or(350.0);
        (num * 1e6) as u64
    } else {
        lower.parse().unwrap_or(7_000_000_000)
    }
}

fn format_vram(gb: f64) -> String {
    format!("{gb:.0} GB")
}

/// Plan an optimal LoRA configuration for a model size and VRAM budget.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::ConfigValue`] when `method` is not
/// one of `full`, `lora`, `qlora`, `auto`, and propagates optimizer failures.
pub fn run_plan(
    model: &str,
    vram: f64,
    method: &str,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let model_params = parse_model_size(model);
    let method: Method =
        method
            .parse()
            .map_err(|e| entrenar_common::EntrenarError::ConfigValue {
                field: "method".into(),
                message: e,
                suggestion: "Use: full, lora, qlora, auto".into(),
            })?;

    let config = plan(model_params, vram, method)?;

    if cli.format == entrenar_common::OutputFormat::Json {
        println!(
            "{}",
            serde_json::json!({
                "method": format!("{:?}", config.method),
                "rank": config.rank,
                "alpha": config.alpha,
                "target_modules": config.target_modules,
                "trainable_params": config.trainable_params,
                "trainable_percent": config.trainable_percent,
                "memory_gb": config.memory_gb,
                "utilization_percent": config.utilization_percent,
                "speedup": config.speedup,
            })
        );
    } else {
        if !cli.is_quiet() {
            println!(
                "{}",
                styles::header(&format!(
                    "Optimal Configuration for {} VRAM",
                    format_vram(vram)
                ))
            );
        }

        let table = TableBuilder::new()
            .headers(vec!["Property", "Value"])
            .row(vec!["Method", &format!("{:?}", config.method)])
            .row(vec!["Rank", &config.rank.to_string()])
            .row(vec!["Alpha", &format!("{:.1}", config.alpha)])
            .row(vec!["Target Modules", &config.target_modules.join(", ")])
            .row(vec![
                "Trainable Parameters",
                &format!(
                    "{} ({:.2}%)",
                    format_number(config.trainable_params),
                    config.trainable_percent
                ),
            ])
            .row(vec![
                "Memory Required",
                &format!(
                    "{:.1} GB ({:.0}% utilization)",
                    config.memory_gb, config.utilization_percent
                ),
            ])
            .row(vec![
                "Training Speedup",
                &format!("{:.1}x vs full fine-tuning", config.speedup),
            ])
            .build();

        println!("{}", table.render());
    }

    Ok(())
}

/// Compare full / LoRA / QLoRA fine-tuning for a model size and VRAM budget.
///
/// # Errors
///
/// Infallible today; returns `Result` so the signature is stable if the
/// comparison gains fallible steps.
pub fn run_compare(
    model: &str,
    vram: f64,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let model_params = parse_model_size(model);
    let comparisons = crate::optimizer::compare_methods(model_params, vram);

    if cli.format == entrenar_common::OutputFormat::Json {
        let json: Vec<_> = comparisons
            .iter()
            .map(|c| {
                serde_json::json!({
                    "method": format!("{:?}", c.method),
                    "fits": c.fits,
                    "memory_gb": c.memory_gb,
                    "trainable_params": c.trainable_params,
                    "speedup": c.speedup,
                    "rank": c.rank,
                })
            })
            .collect();
        if let Ok(json_str) = serde_json::to_string_pretty(&json) {
            println!("{json_str}");
        }
    } else {
        if !cli.is_quiet() {
            println!("{}", styles::header("Method Comparison"));
        }

        let mut builder = TableBuilder::new().headers(vec![
            "Method", "Fits", "Memory", "Params", "Speedup", "Rank",
        ]);

        for c in &comparisons {
            let fits = if c.fits { "✓" } else { "✗" };
            builder = builder.row(vec![
                &format!("{:?}", c.method),
                fits,
                &format!("{:.1} GB", c.memory_gb),
                &format_number(c.trainable_params),
                &format!("{:.1}x", c.speedup),
                &c.rank.to_string(),
            ]);
        }

        println!("{}", builder.build().render());

        // Recommendation
        if let Some(best) = comparisons.iter().filter(|c| c.fits).max_by(|a, b| {
            a.speedup
                .partial_cmp(&b.speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            println!(
                "\n{}",
                styles::success(&format!(
                    "Recommendation: {:?} (rank {}) for optimal speed/memory balance",
                    best.method, best.rank
                ))
            );
        }
    }

    Ok(())
}

/// Merge a LoRA adapter into a base model, scaling the delta by `scale`.
///
/// # Errors
///
/// Propagates load/merge/write failures from [`crate::MergeEngine`].
pub fn run_merge(
    base: &Path,
    adapter: &Path,
    output: &Path,
    scale: f32,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let engine = crate::MergeEngine::new().with_scale(scale);
    let result = engine.merge_from_file(base, adapter, output)?;

    if !cli.is_quiet() {
        println!(
            "{}",
            styles::success(&format!(
                "Merged adapter into base model\n  Output: {}\n  Size: {}",
                result.output_path.display(),
                format_bytes(result.output_size_bytes)
            ))
        );
    }

    Ok(())
}

/// Inspect a LoRA adapter file.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::ModelNotFound`] when `path` does
/// not exist.
pub fn run_inspect(path: &Path, cli: &entrenar_common::Cli) -> entrenar_common::Result<()> {
    if !path.exists() {
        return Err(entrenar_common::EntrenarError::ModelNotFound {
            path: path.to_path_buf(),
        });
    }

    // In real implementation, would load and analyze the adapter
    if !cli.is_quiet() {
        println!(
            "{}",
            styles::header(&format!("Adapter Analysis: {}", path.display()))
        );
        println!("  (Detailed analysis requires loading adapter file)");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_model_size_reads_billions_suffix() {
        assert_eq!(parse_model_size("7B"), 7_000_000_000);
        assert_eq!(parse_model_size("1.5b"), 1_500_000_000);
    }

    #[test]
    fn parse_model_size_reads_millions_suffix() {
        assert_eq!(parse_model_size("350M"), 350_000_000);
    }

    #[test]
    fn parse_model_size_reads_bare_integer() {
        assert_eq!(parse_model_size("123456789"), 123_456_789);
    }

    #[test]
    fn run_plan_refuses_unknown_method() {
        let cli = entrenar_common::Cli::new().with_verbosity(0);
        let err = match run_plan("7B", 24.0, "dora", &cli) {
            Ok(()) => panic!("run_plan must refuse an unknown fine-tuning method"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("dora"),
            "refusal must quote the rejected method, got: {err}"
        );
    }

    #[test]
    fn run_inspect_refuses_missing_adapter() {
        let cli = entrenar_common::Cli::new().with_verbosity(0);
        let missing = std::path::Path::new("/nonexistent/adapter-that-does-not-exist.safetensors");
        // Asserting is_ok() here would lock in a defect: a missing adapter must
        // be refused, not silently "analysed".
        let err = match run_inspect(missing, &cli) {
            Ok(()) => panic!("run_inspect must refuse a path that does not exist"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("adapter-that-does-not-exist"),
            "refusal must quote the missing path, got: {err}"
        );
    }
}
