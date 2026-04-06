//! entrenar-lora CLI entry point.

use clap::{Parser, Subcommand};
use entrenar_common::cli::{styles, CommonArgs};
use entrenar_common::output::{format_bytes, format_number, TableBuilder};
use entrenar_lora::{plan, Method};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "entrenar-lora")]
#[command(about = "LoRA/QLoRA configuration optimizer and memory planner")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Plan optimal LoRA configuration for given constraints
    Plan {
        /// Model size in parameters (e.g., "7B", "13B") or exact number
        #[arg(short, long)]
        model: String,

        /// Available VRAM in GB
        #[arg(short, long)]
        vram: f64,

        /// Fine-tuning method: full, lora, qlora, auto
        #[arg(short = 'm', long, default_value = "auto")]
        method: String,
    },

    /// Compare different fine-tuning methods
    Compare {
        /// Model size
        #[arg(short, long)]
        model: String,

        /// Available VRAM in GB
        #[arg(short, long, default_value = "24")]
        vram: f64,
    },

    /// Merge LoRA adapter with base model
    Merge {
        /// Path to base model
        #[arg(short, long)]
        base: PathBuf,

        /// Path to LoRA adapter
        #[arg(short, long)]
        adapter: PathBuf,

        /// Output path
        #[arg(short, long)]
        output: PathBuf,

        /// Scale factor for adapter (default: 1.0)
        #[arg(short, long, default_value = "1.0")]
        scale: f32,
    },

    /// Inspect LoRA adapter structure
    Inspect {
        /// Path to adapter file
        path: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    let config = cli.common.to_cli();

    let result = match cli.command {
        Commands::Plan { model, vram, method } => plan_command(&model, vram, &method, &config),
        Commands::Compare { model, vram } => compare_command(&model, vram, &config),
        Commands::Merge { base, adapter, output, scale } => {
            merge_command(&base, &adapter, &output, scale, &config)
        }
        Commands::Inspect { path } => inspect_command(&path, &config),
    };

    if let Err(e) = result {
        if !config.is_quiet() {
            eprintln!("{}", styles::error(&e.to_string()));
        }
        std::process::exit(1);
    }
}

fn parse_model_size(model: &str) -> u64 {
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

fn plan_command(
    model: &str,
    vram: f64,
    method: &str,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let model_params = parse_model_size(model);
    let method: Method =
        method.parse().map_err(|e| entrenar_common::EntrenarError::ConfigValue {
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
                styles::header(&format!("Optimal Configuration for {} VRAM", format_vram(vram)))
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
            .row(vec!["Training Speedup", &format!("{:.1}x vs full fine-tuning", config.speedup)])
            .build();

        println!("{}", table.render());
    }

    Ok(())
}

fn compare_command(
    model: &str,
    vram: f64,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let model_params = parse_model_size(model);
    let comparisons = entrenar_lora::optimizer::compare_methods(model_params, vram);

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

        let mut builder = TableBuilder::new()
            .headers(vec!["Method", "Fits", "Memory", "Params", "Speedup", "Rank"]);

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
        if let Some(best) = comparisons
            .iter()
            .filter(|c| c.fits)
            .max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap_or(std::cmp::Ordering::Equal))
        {
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

fn merge_command(
    base: &std::path::Path,
    adapter: &std::path::Path,
    output: &std::path::Path,
    scale: f32,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let engine = entrenar_lora::MergeEngine::new().with_scale(scale);
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

fn inspect_command(
    path: &std::path::Path,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !path.exists() {
        return Err(entrenar_common::EntrenarError::ModelNotFound { path: path.to_path_buf() });
    }

    // In real implementation, would load and analyze the adapter
    if !cli.is_quiet() {
        println!("{}", styles::header(&format!("Adapter Analysis: {}", path.display())));
        println!("  (Detailed analysis requires loading adapter file)");
    }

    Ok(())
}

fn format_vram(gb: f64) -> String {
    format!("{gb:.0} GB")
}
