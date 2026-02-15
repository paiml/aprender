//! Fine-tuning command implementation (GH-244)
//!
//! Surfaces entrenar's LoRA/QLoRA fine-tuning pipeline through the apr CLI.
//! Supports planning mode (VRAM estimation) and training execution.
//!
//! # Example
//!
//! ```bash
//! apr finetune model.apr --method lora --data train.jsonl -o adapter/
//! apr finetune model.apr --method qlora --rank 16 --plan --json
//! apr finetune merge model.apr --adapter adapter/ -o merged.apr
//! ```

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use entrenar_lora::{plan, MemoryPlanner, Method};
use std::path::Path;

/// Fine-tuning method selection
#[derive(Debug, Clone, Copy, Default)]
pub enum FinetuneMethod {
    #[default]
    Auto,
    Full,
    LoRA,
    QLoRA,
}

impl std::str::FromStr for FinetuneMethod {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "full" => Ok(Self::Full),
            "lora" => Ok(Self::LoRA),
            "qlora" => Ok(Self::QLoRA),
            _ => Err(format!(
                "Unknown fine-tuning method: {s}. Use: auto, full, lora, qlora"
            )),
        }
    }
}

impl From<FinetuneMethod> for Method {
    fn from(m: FinetuneMethod) -> Self {
        match m {
            FinetuneMethod::Auto => Method::Auto,
            FinetuneMethod::Full => Method::Full,
            FinetuneMethod::LoRA => Method::LoRA,
            FinetuneMethod::QLoRA => Method::QLoRA,
        }
    }
}

/// Run the finetune command
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
pub(crate) fn run(
    model_path: Option<&Path>,
    method: &str,
    rank: Option<u32>,
    vram_gb: f64,
    plan_only: bool,
    data_path: Option<&Path>,
    output_path: Option<&Path>,
    adapter_path: Option<&Path>,
    merge_mode: bool,
    epochs: u32,
    learning_rate: f64,
    model_size: Option<&str>,
    json_output: bool,
) -> Result<()> {
    // Handle merge subcommand
    if merge_mode {
        return run_merge(model_path, adapter_path, output_path, json_output);
    }

    let ft_method: FinetuneMethod = method
        .parse()
        .map_err(CliError::ValidationFailed)?;

    if !json_output {
        output::section("apr finetune (GH-244: LoRA/QLoRA Fine-tuning)");
        println!();
    }

    // Determine model parameters
    let model_params = if let Some(size) = model_size {
        parse_model_size(size)?
    } else if let Some(path) = model_path {
        estimate_params_from_file(path)?
    } else {
        return Err(CliError::ValidationFailed(
            "Either model path or --model-size required".to_string(),
        ));
    };

    if !json_output {
        output::kv("Model parameters", format_params(model_params));
        output::kv("Available VRAM", format!("{vram_gb:.1} GB"));
        output::kv("Method", format!("{ft_method:?}"));
        if let Some(r) = rank {
            output::kv("Requested rank", r.to_string());
        }
        output::kv("Epochs", epochs.to_string());
        output::kv("Learning rate", format!("{learning_rate:.1e}"));
        println!();
    }

    // Plan configuration using entrenar-lora
    let config = plan(model_params, vram_gb, ft_method.into())
        .map_err(|e| CliError::ValidationFailed(format!("Failed to plan config: {e}")))?;

    // Memory breakdown
    let planner = MemoryPlanner::new(model_params);
    let req = planner.estimate(config.method, config.rank);

    if json_output {
        let json = serde_json::json!({
            "model_params": model_params,
            "vram_gb": vram_gb,
            "recommended_method": format!("{:?}", config.method),
            "recommended_rank": config.rank,
            "recommended_alpha": config.alpha,
            "trainable_params": config.trainable_params,
            "trainable_percent": config.trainable_percent,
            "memory_gb": config.memory_gb,
            "utilization_percent": config.utilization_percent,
            "speedup": config.speedup,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "plan_only": plan_only,
            "memory_breakdown": {
                "model_bytes": req.model_bytes,
                "adapter_bytes": req.adapter_bytes,
                "optimizer_bytes": req.optimizer_bytes,
                "activation_bytes": req.activation_bytes,
                "total_bytes": req.total_bytes,
                "savings_percent": req.savings_percent,
            },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
        if plan_only {
            return Ok(());
        }
    } else {
        // Display recommended config
        println!("{}", "RECOMMENDED CONFIGURATION".white().bold());
        println!("{}", "═".repeat(50));
        println!();
        println!(
            "  Method:           {}",
            format!("{:?}", config.method).cyan().bold()
        );
        println!("  Rank:             {}", config.rank.to_string().green());
        println!("  Alpha:            {:.1}", config.alpha);
        println!(
            "  Trainable params: {} ({:.2}%)",
            format_params(config.trainable_params).yellow(),
            config.trainable_percent
        );
        println!(
            "  Memory required:  {:.2} GB ({:.0}% utilization)",
            config.memory_gb, config.utilization_percent
        );
        println!(
            "  Speedup:          {:.1}x vs full fine-tuning",
            config.speedup
        );
        println!();

        // Memory breakdown
        println!("{}", "MEMORY BREAKDOWN".white().bold());
        println!("{}", "─".repeat(50));

        let model_gb = req.model_bytes as f64 / 1e9;
        let adapter_gb = req.adapter_bytes as f64 / 1e9;
        let optimizer_gb = req.optimizer_bytes as f64 / 1e9;
        let activation_gb = req.activation_bytes as f64 / 1e9;
        let total_gb = req.total_bytes as f64 / 1e9;

        println!("  Base model:       {model_gb:.2} GB");
        println!("  Adapter:          {adapter_gb:.2} GB");
        println!("  Optimizer states: {optimizer_gb:.2} GB");
        println!("  Activations:      {activation_gb:.2} GB");
        println!("{}", "─".repeat(50));
        println!("  {}:            {total_gb:.2} GB", "TOTAL".bold());
        println!(
            "  Savings:          {:.0}% vs full fine-tuning",
            req.savings_percent
        );
        println!();

        // Feasibility check
        if total_gb <= vram_gb {
            println!(
                "{} Configuration fits in {vram_gb:.1} GB VRAM",
                "✓".green().bold(),
            );
        } else {
            println!(
                "{} Configuration requires {total_gb:.2} GB but only {vram_gb:.1} GB available",
                "⚠".yellow().bold(),
            );
            println!();
            println!("  Suggestions:");
            println!("    - Use QLoRA (4-bit quantization)");
            println!("    - Reduce rank (--rank 4)");
            println!("    - Use gradient checkpointing");
        }
    }

    if plan_only {
        return Ok(());
    }

    // Training execution
    if data_path.is_none() {
        if !json_output {
            println!();
            println!("{}", "NEXT STEPS".white().bold());
            println!("{}", "─".repeat(50));
            println!("  Provide --data <train.jsonl> to start training.");
            println!("  Example: apr finetune model.apr --method lora --data train.jsonl -o adapter/");
        }
        return Ok(());
    }

    let data = data_path.expect("data checked above");
    if !data.exists() {
        return Err(CliError::FileNotFound(data.to_path_buf()));
    }

    if !json_output {
        println!();
        output::pipeline_stage("Training", output::StageStatus::Running);
        println!("  Data: {}", data.display());
        println!("  Epochs: {epochs}");
        println!("  Learning rate: {learning_rate:.1e}");
    }

    // Training execution placeholder
    // entrenar training loop integration goes here
    let out = output_path.unwrap_or(Path::new("adapter/"));
    if !json_output {
        println!();
        println!(
            "  {} Training pipeline configured. Adapter output: {}",
            output::badge_info("INFO"),
            out.display()
        );
        println!("  Full training execution requires entrenar training backend.");
        println!("  Use `apr tune` for planning-only mode.");
    }

    Ok(())
}

/// Run adapter merge (finetune merge)
fn run_merge(
    model_path: Option<&Path>,
    adapter_path: Option<&Path>,
    output_path: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let model = model_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Model path required for merge. Usage: apr finetune merge model.apr --adapter adapter/"
                .to_string(),
        )
    })?;

    let adapter = adapter_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Adapter path required for merge. Use --adapter <path>".to_string(),
        )
    })?;

    if !model.exists() {
        return Err(CliError::FileNotFound(model.to_path_buf()));
    }
    if !adapter.exists() {
        return Err(CliError::FileNotFound(adapter.to_path_buf()));
    }

    let out = output_path.unwrap_or(Path::new("merged.apr"));

    if !json_output {
        output::header("APR Finetune — Merge Adapter");
        println!(
            "{}",
            output::kv_table(&[
                ("Base model", model.display().to_string()),
                ("Adapter", adapter.display().to_string()),
                ("Output", out.display().to_string()),
            ])
        );
        println!();
        println!(
            "  {} Adapter merge requires entrenar merge backend.",
            output::badge_info("INFO")
        );
    }

    Ok(())
}

/// Parse model size string (e.g., "7B", "1.5B", "70B")
fn parse_model_size(size: &str) -> Result<u64> {
    let size = size.to_uppercase();
    let (num_str, multiplier) = if size.ends_with('B') {
        (&size[..size.len() - 1], 1_000_000_000u64)
    } else if size.ends_with('M') {
        (&size[..size.len() - 1], 1_000_000u64)
    } else {
        return Err(CliError::ValidationFailed(format!(
            "Invalid model size format: {size}. Use: 7B, 1.5B, 70B, etc."
        )));
    };

    let num: f64 = num_str.parse().map_err(|_| {
        CliError::ValidationFailed(format!("Invalid number in model size: {num_str}"))
    })?;

    Ok((num * multiplier as f64) as u64)
}

/// Estimate parameters from model file size
fn estimate_params_from_file(path: &Path) -> Result<u64> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model file: {e}")))?;
    // Conservative estimate: assume Q4 (~0.5 bytes/param)
    Ok(metadata.len() * 2)
}

/// Format parameter count for display
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else {
        format!("{params}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_finetune_method_parse() {
        assert!(matches!("auto".parse::<FinetuneMethod>(), Ok(FinetuneMethod::Auto)));
        assert!(matches!("full".parse::<FinetuneMethod>(), Ok(FinetuneMethod::Full)));
        assert!(matches!("lora".parse::<FinetuneMethod>(), Ok(FinetuneMethod::LoRA)));
        assert!(matches!("qlora".parse::<FinetuneMethod>(), Ok(FinetuneMethod::QLoRA)));
        assert!("unknown".parse::<FinetuneMethod>().is_err());
    }

    #[test]
    fn test_finetune_method_to_entrenar() {
        assert!(matches!(Method::from(FinetuneMethod::Auto), Method::Auto));
        assert!(matches!(Method::from(FinetuneMethod::LoRA), Method::LoRA));
        assert!(matches!(Method::from(FinetuneMethod::QLoRA), Method::QLoRA));
        assert!(matches!(Method::from(FinetuneMethod::Full), Method::Full));
    }

    #[test]
    fn test_parse_model_size() {
        assert_eq!(parse_model_size("7B").expect("7B"), 7_000_000_000);
        assert_eq!(parse_model_size("1.5B").expect("1.5B"), 1_500_000_000);
        assert_eq!(parse_model_size("135M").expect("135M"), 135_000_000);
        assert!(parse_model_size("invalid").is_err());
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(7_000_000_000), "7.0B");
        assert_eq!(format_params(135_000_000), "135.0M");
        assert_eq!(format_params(1000), "1000");
    }

    #[test]
    fn test_run_no_model() {
        let result = run(
            None, "auto", None, 16.0, false, None, None, None, false, 3, 2e-4, None, false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_plan_with_model_size() {
        let result = run(
            None, "lora", None, 16.0, true, None, None, None, false, 3, 2e-4,
            Some("7B"), false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_plan_json() {
        let result = run(
            None, "qlora", None, 24.0, true, None, None, None, false, 3, 2e-4,
            Some("14B"), true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_model_file() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 4096]).expect("write");
        let result = run(
            Some(input.path()), "auto", None, 16.0, true, None, None, None, false,
            3, 2e-4, None, false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_merge_no_model() {
        let result = run_merge(None, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_no_adapter() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run_merge(Some(input.path()), None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_model_not_found() {
        let result = run_merge(
            Some(Path::new("/nonexistent.apr")),
            Some(Path::new("/nonexistent_adapter/")),
            None,
            false,
        );
        assert!(result.is_err());
    }
}
