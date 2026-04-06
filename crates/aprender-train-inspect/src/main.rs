//! entrenar-inspect CLI entry point.

use clap::{Parser, Subcommand};
use entrenar_common::cli::{styles, CommonArgs};
use entrenar_common::output::{format_bytes, format_number, TableBuilder};
use entrenar_inspect::{inspect, OutputFormat};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "entrenar-inspect")]
#[command(about = "SafeTensors model inspection and format conversion")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Show model information
    Info {
        /// Path to model file
        path: PathBuf,
    },

    /// Show layer-by-layer breakdown
    Layers {
        /// Path to model file
        path: PathBuf,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Estimate training memory requirements
    Memory {
        /// Path to model file
        path: PathBuf,

        /// Batch size
        #[arg(short, long, default_value = "32")]
        batch_size: u32,

        /// Sequence length
        #[arg(short, long, default_value = "512")]
        seq_len: usize,
    },

    /// Validate model integrity
    Validate {
        /// Path to model file
        path: PathBuf,

        /// Enable strict validation
        #[arg(long)]
        strict: bool,
    },

    /// Convert model format
    Convert {
        /// Input model path
        input: PathBuf,

        /// Output format: safetensors, gguf, apr
        #[arg(short, long)]
        to: String,

        /// Output path
        #[arg(short, long)]
        output: PathBuf,

        /// Quantization: q4_0, q8_0, f16, none
        #[arg(long, default_value = "none")]
        quantize: String,
    },

    /// Compare two models
    Compare {
        /// First model path
        model1: PathBuf,

        /// Second model path
        model2: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    let config = cli.common.to_cli();

    let result = match cli.command {
        Commands::Info { path } => info_command(&path, &config),
        Commands::Layers { path, verbose } => layers_command(&path, verbose, &config),
        Commands::Memory {
            path,
            batch_size,
            seq_len,
        } => memory_command(&path, batch_size, seq_len, &config),
        Commands::Validate { path, strict } => validate_command(&path, strict, &config),
        Commands::Convert {
            input,
            to,
            output,
            quantize,
        } => convert_command(&input, &to, &output, &quantize, &config),
        Commands::Compare { model1, model2 } => compare_command(&model1, &model2, &config),
    };

    if let Err(e) = result {
        if !config.is_quiet() {
            eprintln!("{}", styles::error(&e.to_string()));
        }
        std::process::exit(1);
    }
}

fn info_command(path: &PathBuf, cli: &entrenar_common::Cli) -> entrenar_common::Result<()> {
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

fn layers_command(
    path: &PathBuf,
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

fn memory_command(
    path: &PathBuf,
    batch_size: u32,
    seq_len: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let info = inspect(path)?;

    let model_bytes = info.size_bytes;
    let activation_bytes =
        u64::from(batch_size) * (seq_len as u64) * (info.architecture.hidden_dim as u64) * 32 * 2;
    let optimizer_bytes = model_bytes * 4; // Adam states
    let total_bytes = model_bytes + activation_bytes + optimizer_bytes;

    if cli.format == entrenar_common::OutputFormat::Json {
        println!(
            "{}",
            serde_json::json!({
                "model_bytes": model_bytes,
                "activation_bytes": activation_bytes,
                "optimizer_bytes": optimizer_bytes,
                "total_bytes": total_bytes,
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
            .row(vec!["Model Weights", &format_bytes(model_bytes)])
            .row(vec!["Activations", &format_bytes(activation_bytes)])
            .row(vec!["Optimizer State", &format_bytes(optimizer_bytes)])
            .row(vec!["Total", &format_bytes(total_bytes)])
            .build();

        println!("{}", table.render());
    }

    Ok(())
}

fn validate_command(
    path: &std::path::Path,
    strict: bool,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let checker = if strict {
        entrenar_inspect::validate::IntegrityChecker::new().strict()
    } else {
        entrenar_inspect::validate::IntegrityChecker::new()
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

    if !result.valid {
        std::process::exit(1);
    }

    Ok(())
}

fn convert_command(
    input: &std::path::Path,
    to: &str,
    output: &std::path::Path,
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

    let mut converter = entrenar_inspect::convert::FormatConverter::new();

    if quantize != "none" {
        let quant: entrenar_inspect::convert::Quantization =
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

fn compare_command(
    model1: &PathBuf,
    model2: &PathBuf,
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
