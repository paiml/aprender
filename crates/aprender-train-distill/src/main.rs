//! entrenar-distill CLI entry point.

use clap::{Parser, Subcommand};
use entrenar_common::cli::CommonArgs;
use entrenar_distill::{config::DistillConfig, estimate_memory, run, validation::ConfigValidator};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "entrenar-distill")]
#[command(about = "End-to-end knowledge distillation CLI")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the distillation pipeline
    Run {
        /// Path to configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Override output directory
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Dry run (validate only, don't train)
        #[arg(long)]
        dry_run: bool,
    },

    /// Estimate memory requirements
    Estimate {
        /// Teacher model ID
        #[arg(long)]
        teacher: String,

        /// Student model ID
        #[arg(long)]
        student: Option<String>,

        /// Batch size
        #[arg(long, default_value = "32")]
        batch_size: u32,

        /// Sequence length
        #[arg(long, default_value = "512")]
        seq_len: usize,
    },

    /// Validate configuration file
    Validate {
        /// Path to configuration file
        #[arg(short, long)]
        config: PathBuf,
    },

    /// Export a trained model to different formats
    Export {
        /// Input model path
        #[arg(short, long)]
        input: PathBuf,

        /// Output format: safetensors, gguf, apr
        #[arg(short, long, default_value = "safetensors")]
        format: String,

        /// Output path
        #[arg(short, long)]
        output: PathBuf,

        /// Quantization: none, q8_0, q4_0
        #[arg(long, default_value = "none")]
        quantize: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let config = cli.common.to_cli();

    let result = match cli.command {
        Commands::Run {
            config: config_path,
            output,
            dry_run,
        } => run_command(&config_path, output, dry_run, &config),

        Commands::Estimate {
            teacher,
            student,
            batch_size,
            seq_len,
        } => estimate_command(&teacher, student, batch_size, seq_len, &config),

        Commands::Validate {
            config: config_path,
        } => validate_command(&config_path, &config),

        Commands::Export {
            input,
            format,
            output,
            quantize,
        } => export_command(&input, &format, &output, &quantize, &config),
    };

    if let Err(e) = result {
        if !config.is_quiet() {
            eprintln!("{}", entrenar_common::cli::styles::error(&e.to_string()));
        }
        std::process::exit(1);
    }
}

fn run_command(
    config_path: &PathBuf,
    output: Option<PathBuf>,
    dry_run: bool,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::header("entrenar-distill")
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

fn estimate_command(
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

fn validate_command(
    config_path: &PathBuf,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
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

/// Ensure the parent directory of `path` exists, creating it if necessary.
fn ensure_parent_dir(path: &std::path::Path) -> entrenar_common::Result<()> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => return Ok(()),
    };
    std::fs::create_dir_all(parent).map_err(|e| entrenar_common::EntrenarError::Io {
        context: format!("creating output directory: {}", parent.display()),
        source: e,
    })
}

/// Dispatch export to the correct format handler.
fn dispatch_export(
    format: &str,
    weights: &std::collections::HashMap<String, Vec<f32>>,
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    output: &std::path::Path,
    quantize: &str,
) -> entrenar_common::Result<()> {
    match format {
        "safetensors" => export_safetensors(weights, shapes, output),
        "gguf" => export_gguf(weights, shapes, output, quantize),
        "apr" | "json" => export_apr(weights, output),
        other => Err(entrenar_common::EntrenarError::UnsupportedFormat {
            format: other.to_string(),
        }),
    }
}

fn export_command(
    input: &std::path::Path,
    format: &str,
    output: &std::path::Path,
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

    let (weights, shapes) = entrenar_distill::load_safetensors_weights(input)?;

    ensure_parent_dir(output)?;
    dispatch_export(format, &weights, &shapes, output, quantize)?;

    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::success(&format!("Exported to {}", output.display()))
        );
    }

    Ok(())
}

/// Export weights as SafeTensors format.
fn export_safetensors(
    weights: &std::collections::HashMap<String, Vec<f32>>,
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    output: &std::path::Path,
) -> entrenar_common::Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    let mut sorted_names: Vec<&String> = weights.keys().collect();
    sorted_names.sort();

    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = sorted_names
        .iter()
        .map(|name| {
            let data = &weights[*name];
            let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
            let shape = shapes
                .get(*name)
                .cloned()
                .unwrap_or_else(|| vec![data.len()]);
            ((*name).clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| -> entrenar_common::Result<_> {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes).map_err(|e| {
                entrenar_common::EntrenarError::Serialization {
                    message: format!("TensorView creation failed for {name}: {e}"),
                }
            })?;
            Ok((name.as_str(), view))
        })
        .collect::<entrenar_common::Result<Vec<_>>>()?;

    let st_bytes = safetensors::serialize(views, None).map_err(|e| {
        entrenar_common::EntrenarError::Serialization {
            message: format!("SafeTensors serialization failed: {e}"),
        }
    })?;

    std::fs::write(output, st_bytes).map_err(|e| entrenar_common::EntrenarError::Io {
        context: format!("writing SafeTensors output: {}", output.display()),
        source: e,
    })
}

/// Export weights as GGUF format (requires `hub` feature for real quantization).
fn export_gguf(
    weights: &std::collections::HashMap<String, Vec<f32>>,
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    output: &std::path::Path,
    quantize: &str,
) -> entrenar_common::Result<()> {
    #[cfg(feature = "hub")]
    {
        let quant = match quantize {
            "q4_0" | "Q4_0" => entrenar::hf_pipeline::GgufQuantization::Q4_0,
            "q8_0" | "Q8_0" => entrenar::hf_pipeline::GgufQuantization::Q8_0,
            "none" | "None" | "f32" => entrenar::hf_pipeline::GgufQuantization::None,
            other => {
                return Err(entrenar_common::EntrenarError::ConfigValue {
                    field: "quantize".into(),
                    message: format!("unknown quantization: {other}"),
                    suggestion: "Use one of: none, q4_0, q8_0".into(),
                });
            }
        };

        let mw =
            entrenar_distill::weights::weights_to_model_weights(weights.clone(), shapes.clone());

        let output_dir = output.parent().unwrap_or_else(|| std::path::Path::new("."));
        let filename = output
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));

        let exporter = entrenar::hf_pipeline::Exporter::new()
            .output_dir(output_dir)
            .gguf_quantization(quant);

        exporter
            .export(&mw, entrenar::hf_pipeline::ExportFormat::GGUF, filename)
            .map_err(|e| entrenar_common::EntrenarError::Internal {
                message: format!("GGUF export failed: {e}"),
            })?;

        Ok(())
    }

    #[cfg(not(feature = "hub"))]
    {
        let _ = (weights, shapes, output, quantize);
        Err(entrenar_common::EntrenarError::HuggingFace {
            message: "GGUF export requires the 'hub' feature. \
                      Rebuild with: cargo build -p entrenar-distill --features hub"
                .to_string(),
        })
    }
}

/// Export weights as APR (JSON) format via entrenar's io module.
fn export_apr(
    weights: &std::collections::HashMap<String, Vec<f32>>,
    output: &std::path::Path,
) -> entrenar_common::Result<()> {
    // Build a simple JSON representation of the model weights
    let model_data = serde_json::json!({
        "format": "apr",
        "version": "1.0",
        "tensors": weights.iter().map(|(name, data)| {
            serde_json::json!({
                "name": name,
                "shape": [data.len()],
                "data": data,
            })
        }).collect::<Vec<_>>(),
    });

    let json = serde_json::to_string_pretty(&model_data).map_err(|e| {
        entrenar_common::EntrenarError::Serialization {
            message: format!("APR JSON serialization failed: {e}"),
        }
    })?;

    std::fs::write(output, json).map_err(|e| entrenar_common::EntrenarError::Io {
        context: format!("writing APR output: {}", output.display()),
        source: e,
    })
}
