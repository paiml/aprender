//! Command implementations for the training-side model inspector.
//!
//! These functions were the body of the `aprender-train-inspect` binary's
//! `main.rs`. That binary is gone (APR-MONO Rule 1: `apr` is the only
//! user-facing binary); the capability is reachable as
//! `apr train inspect <verb>`, which calls exactly these entry points.

// The `serde_json::json!` macro expands to code containing `.unwrap()`, which
// trips clippy::disallowed_methods at the macro invocation site even though no
// author-written unwrap exists. Scope the allow to this presentation module.
#![allow(clippy::disallowed_methods)]

use crate::{inspect, OutputFormat};
use entrenar_common::cli::styles;
use entrenar_common::output::{format_bytes, format_number, TableBuilder};
use std::path::Path;

/// Memory footprint of a training step, in bytes.
///
/// Extracted from the binary's `memory_command`, where the arithmetic was
/// stranded in `main.rs`. Activation bytes assume 32 stored tensors per
/// sample position and 2 bytes per element (FP16); optimizer state is Adam's
/// 4× the model bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingMemory {
    /// Model weights, in bytes (the on-disk size).
    pub model_bytes: u64,
    /// Activation memory for one batch, in bytes.
    pub activation_bytes: u64,
    /// Adam optimizer state, in bytes.
    pub optimizer_bytes: u64,
    /// Sum of the three components, in bytes.
    pub total_bytes: u64,
}

impl TrainingMemory {
    /// Estimate training memory for a model of `model_bytes` on disk.
    #[must_use]
    pub fn estimate(model_bytes: u64, hidden_dim: usize, batch_size: u32, seq_len: usize) -> Self {
        let activation_bytes =
            u64::from(batch_size) * (seq_len as u64) * (hidden_dim as u64) * 32 * 2;
        let optimizer_bytes = model_bytes * 4; // Adam states
        Self {
            model_bytes,
            activation_bytes,
            optimizer_bytes,
            total_bytes: model_bytes + activation_bytes + optimizer_bytes,
        }
    }
}

/// Show format, architecture, parameter count and tensor count for a model.
///
/// # Errors
///
/// Propagates read/parse failures from [`crate::inspect`].
pub fn run_info(path: &Path, cli: &entrenar_common::Cli) -> entrenar_common::Result<()> {
    let info = inspect(path)?;

    if cli.format == entrenar_common::OutputFormat::Json {
        println!(
            "{}",
            serde_json::json!({
                "path": info.path.display().to_string(),
                "size_bytes": info.size_bytes,
                "format": format!("{:?}", info.format),
                "architecture": info.architecture.architecture.name(),
                "hidden_dim": info.architecture.hidden_dim,
                "num_layers": info.architecture.num_layers,
                "vocab_size": info.architecture.vocab_size,
                "total_params": info.total_params,
            })
        );
    } else {
        if !cli.is_quiet() {
            println!("{}", styles::header(&format!("Model: {}", path.display())));
        }

        let table = TableBuilder::new()
            .headers(vec!["Property", "Value"])
            .row(vec!["Format", &format!("{:?}", info.format)])
            .row(vec!["Size", &info.size_human()])
            .row(vec!["Architecture", info.architecture.architecture.name()])
            .row(vec![
                "Hidden Dimension",
                &info.architecture.hidden_dim.to_string(),
            ])
            .row(vec!["Layers", &info.architecture.num_layers.to_string()])
            .row(vec![
                "Vocab Size",
                &format_number(info.architecture.vocab_size as u64),
            ])
            .row(vec!["Parameters", &format!("{:.2}B", info.params_b())])
            .row(vec!["Tensors", &info.tensors.len().to_string()])
            .build();

        println!("{}", table.render());
    }

    Ok(())
}

/// Show a per-layer breakdown of tensors, parameters and bytes.
///
/// With `verbose`, also lists every tensor name and shape.
///
/// # Errors
///
/// Propagates read/parse failures from [`crate::inspect`].
pub fn run_layers(
    path: &Path,
    verbose: bool,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let info = inspect(path)?;
    let breakdown = inspect::layer_breakdown(&info);

    if !cli.is_quiet() {
        println!("{}", styles::header("Layer Breakdown"));
    }

    let mut builder = TableBuilder::new().headers(vec!["Layer", "Tensors", "Parameters", "Size"]);

    for layer in &breakdown {
        builder = builder.row(vec![
            &layer.layer_num.to_string(),
            &layer.tensor_count.to_string(),
            &format_number(layer.param_count),
            &format_bytes(layer.size_bytes),
        ]);
    }

    println!("{}", builder.build().render());

    if verbose {
        println!("\n{}", styles::header("All Tensors"));
        for tensor in &info.tensors {
            println!(
                "  {} [{:?}] - {} params",
                tensor.name,
                tensor.shape,
                format_number(tensor.num_elements)
            );
        }
    }

    Ok(())
}

/// Estimate training memory for a model at a given batch size and sequence length.
///
/// # Errors
///
/// Propagates read/parse failures from [`crate::inspect`].
pub fn run_memory(
    path: &Path,
    batch_size: u32,
    seq_len: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let info = inspect(path)?;

    let mem = TrainingMemory::estimate(
        info.size_bytes,
        info.architecture.hidden_dim,
        batch_size,
        seq_len,
    );

    if cli.format == entrenar_common::OutputFormat::Json {
        println!(
            "{}",
            serde_json::json!({
                "model_bytes": mem.model_bytes,
                "activation_bytes": mem.activation_bytes,
                "optimizer_bytes": mem.optimizer_bytes,
                "total_bytes": mem.total_bytes,
            })
        );
    } else {
        if !cli.is_quiet() {
            println!(
                "{}",
                styles::header(&format!(
                    "Memory Estimate (batch={batch_size}, seq={seq_len})"
                ))
            );
        }

        let table = TableBuilder::new()
            .headers(vec!["Component", "Memory"])
            .row(vec!["Model Weights", &format_bytes(mem.model_bytes)])
            .row(vec!["Activations", &format_bytes(mem.activation_bytes)])
            .row(vec!["Optimizer State", &format_bytes(mem.optimizer_bytes)])
            .row(vec!["Total", &format_bytes(mem.total_bytes)])
            .build();

        println!("{}", table.render());
    }

    Ok(())
}

/// Run the integrity checker over a model file.
///
/// Returns `Ok(true)` when the model is valid and `Ok(false)` when the checker
/// reported issues. The binary translated `false` into `exit(1)`; the `apr`
/// dispatcher maps it to a `CliError` so the same exit status is produced
/// without a `process::exit` buried in a library.
///
/// # Errors
///
/// Propagates read/parse failures from the integrity checker.
pub fn run_validate(
    path: &Path,
    strict: bool,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<bool> {
    let checker = if strict {
        crate::validate::IntegrityChecker::new().strict()
    } else {
        crate::validate::IntegrityChecker::new()
    };

    let result = checker.validate(path)?;

    if cli.format == entrenar_common::OutputFormat::Json {
        println!(
            "{}",
            serde_json::json!({
                "valid": result.valid,
                "issues": result.issues.len(),
                "warnings": result.warnings.len(),
                "checks": result.checks.len(),
            })
        );
    } else {
        println!("{}", result.to_report());
    }

    Ok(result.valid)
}

/// Convert a model to another format, optionally quantizing.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::ConfigValue`] when `to` is not a
/// known output format or `quantize` is not a known quantization, and
/// propagates conversion failures.
pub fn run_convert(
    input: &Path,
    to: &str,
    output: &Path,
    quantize: &str,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let format: OutputFormat =
        to.parse()
            .map_err(|e| entrenar_common::EntrenarError::ConfigValue {
                field: "to".into(),
                message: e,
                suggestion: "Use: safetensors, gguf, apr".into(),
            })?;

    let mut converter = crate::convert::FormatConverter::new();

    if quantize != "none" {
        let quant: crate::convert::Quantization =
            quantize
                .parse()
                .map_err(|e| entrenar_common::EntrenarError::ConfigValue {
                    field: "quantize".into(),
                    message: e,
                    suggestion: "Use: q4_0, q8_0, f16, none".into(),
                })?;
        converter = converter.with_quantization(quant);
    }

    let result = converter.convert(input, output, format)?;

    if !cli.is_quiet() {
        println!(
            "{}",
            styles::success(&format!(
                "Converted {} → {}\n  Size: {} → {} ({:+.1}%)\n  Duration: {:.2}s",
                result.input_path.display(),
                result.output_path.display(),
                format_bytes(result.input_size),
                format_bytes(result.output_size),
                result.size_change_percent(),
                result.duration_secs
            ))
        );
    }

    Ok(())
}

/// Compare two models side by side (format, size, params, layers, hidden dim).
///
/// # Errors
///
/// Propagates read/parse failures from [`crate::inspect`] for either model.
pub fn run_compare(
    model1: &Path,
    model2: &Path,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let info1 = inspect(model1)?;
    let info2 = inspect(model2)?;

    if !cli.is_quiet() {
        println!("{}", styles::header("Model Comparison"));
    }

    let table = TableBuilder::new()
        .headers(vec![
            "Property",
            &model1.display().to_string(),
            &model2.display().to_string(),
        ])
        .row(vec![
            "Format",
            &format!("{:?}", info1.format),
            &format!("{:?}", info2.format),
        ])
        .row(vec!["Size", &info1.size_human(), &info2.size_human()])
        .row(vec![
            "Parameters",
            &format!("{:.2}B", info1.params_b()),
            &format!("{:.2}B", info2.params_b()),
        ])
        .row(vec![
            "Layers",
            &info1.architecture.num_layers.to_string(),
            &info2.architecture.num_layers.to_string(),
        ])
        .row(vec![
            "Hidden Dim",
            &info1.architecture.hidden_dim.to_string(),
            &info2.architecture.hidden_dim.to_string(),
        ])
        .build();

    println!("{}", table.render());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_memory_sums_its_three_components() {
        let mem = TrainingMemory::estimate(1_000_000, 4096, 32, 512);
        assert_eq!(
            mem.total_bytes,
            mem.model_bytes + mem.activation_bytes + mem.optimizer_bytes
        );
    }

    #[test]
    fn training_memory_scales_activations_with_batch_size() {
        let one = TrainingMemory::estimate(1_000_000, 4096, 1, 512);
        let four = TrainingMemory::estimate(1_000_000, 4096, 4, 512);
        assert_eq!(four.activation_bytes, one.activation_bytes * 4);
    }

    #[test]
    fn training_memory_scales_activations_with_seq_len() {
        let short = TrainingMemory::estimate(1_000_000, 4096, 8, 128);
        let long = TrainingMemory::estimate(1_000_000, 4096, 8, 512);
        assert_eq!(long.activation_bytes, short.activation_bytes * 4);
    }

    #[test]
    fn training_memory_uses_adam_four_x_optimizer_state() {
        let mem = TrainingMemory::estimate(7_000_000, 1024, 1, 1);
        assert_eq!(mem.optimizer_bytes, 28_000_000);
    }

    #[test]
    fn run_convert_refuses_unknown_output_format() {
        let cli = entrenar_common::Cli::new().with_verbosity(0);
        let err = match run_convert(
            Path::new("in.safetensors"),
            "pickle",
            Path::new("out.bin"),
            "none",
            &cli,
        ) {
            Ok(()) => panic!("run_convert must refuse an unknown output format"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("pickle"),
            "refusal must quote the rejected format, got: {err}"
        );
    }

    #[test]
    fn run_convert_refuses_unknown_quantization() {
        let cli = entrenar_common::Cli::new().with_verbosity(0);
        let err = match run_convert(
            Path::new("in.safetensors"),
            "gguf",
            Path::new("out.gguf"),
            "q3_nonsense",
            &cli,
        ) {
            Ok(()) => panic!("run_convert must refuse an unknown quantization"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("q3_nonsense"),
            "refusal must quote the rejected quantization, got: {err}"
        );
    }
}
