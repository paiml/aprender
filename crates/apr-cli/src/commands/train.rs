//! `apr train plan` — Forjar-style training pre-flight validation.
//!
//! Generates a `TrainingPlan` by validating data quality, model compatibility,
//! HPO search space, resource estimates, and pre-flight checks — all without
//! touching the GPU.
//!
//! Analogous to `forjar plan` which shows what will change before `forjar apply`.

use colored::Colorize;

use crate::{error::CliError, output};

type Result<T> = std::result::Result<T, CliError>;

/// Run `apr train plan` — generate and display a training plan.
pub(crate) fn run_plan(
    data: &std::path::Path,
    model_size: &str,
    model_path: Option<&std::path::Path>,
    num_classes: usize,
    task: &str,
    output_dir: &std::path::Path,
    strategy: &str,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
    val_data: Option<&std::path::Path>,
    test_data: Option<&std::path::Path>,
    format: &str,
    json_output: bool,
) -> Result<()> {
    if task != "classify" {
        return Err(CliError::ValidationFailed(format!(
            "Unknown task type: {task}. Currently supported: classify"
        )));
    }

    let config = entrenar::finetune::PlanConfig {
        task: task.to_string(),
        data_path: data.to_path_buf(),
        val_path: val_data.map(|p| p.to_path_buf()),
        test_path: test_data.map(|p| p.to_path_buf()),
        model_size: model_size.to_string(),
        model_path: model_path.map(|p| p.to_path_buf()),
        num_classes,
        output_dir: output_dir.to_path_buf(),
        strategy: strategy.to_string(),
        budget,
        scout,
        max_epochs,
        manual_lr,
        manual_lora_rank,
        manual_batch_size,
    };

    let plan = entrenar::finetune::training_plan(&config)
        .map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    // Choose output format
    let effective_format = if json_output { "json" } else { format };

    match effective_format {
        "json" => {
            println!("{}", plan.to_json());
        }
        "yaml" => {
            println!("{}", plan.to_yaml());
        }
        _ => {
            print_plan_text(&plan);
        }
    }

    Ok(())
}

include!("train_output.rs");

/// Run `apr train apply` — execute a training plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_apply(
    plan_file: Option<&std::path::Path>,
    data: Option<&std::path::Path>,
    model_size: &str,
    model_path: Option<&std::path::Path>,
    num_classes: usize,
    output_dir: &std::path::Path,
    strategy: &str,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
    json_output: bool,
) -> Result<()> {
    // ── Load or generate plan ──────────────────────────────────────────
    let plan = if let Some(plan_path) = plan_file {
        // Load plan from YAML/JSON file
        let content = std::fs::read_to_string(plan_path).map_err(|_| {
            CliError::FileNotFound(plan_path.to_path_buf())
        })?;

        entrenar::finetune::TrainingPlan::from_str(&content)
            .map_err(|e| {
                CliError::ValidationFailed(format!(
                    "Failed to parse plan file {}: {e}",
                    plan_path.display()
                ))
            })?
    } else {
        // Generate plan inline (same as `apr train plan`)
        let data_path = data.ok_or_else(|| {
            CliError::ValidationFailed(
                "Either --plan <file> or --data <file> is required".to_string(),
            )
        })?;

        let config = entrenar::finetune::PlanConfig {
            task: "classify".to_string(),
            data_path: data_path.to_path_buf(),
            val_path: None,
            test_path: None,
            model_size: model_size.to_string(),
            model_path: model_path.map(|p| p.to_path_buf()),
            num_classes,
            output_dir: output_dir.to_path_buf(),
            strategy: strategy.to_string(),
            budget,
            scout,
            max_epochs,
            manual_lr,
            manual_lora_rank,
            manual_batch_size,
        };

        entrenar::finetune::training_plan(&config)
            .map_err(|e| CliError::ValidationFailed(e.to_string()))?
    };

    // ── Check verdict ──────────────────────────────────────────────────
    if plan.verdict == entrenar::finetune::PlanVerdict::Blocked {
        if json_output {
            println!("{}", plan.to_json());
        } else {
            println!(
                "  {} Plan is blocked — resolve failures before applying",
                "BLOCKED".red().bold()
            );
            print_plan_text(&plan);
        }
        return Err(CliError::ValidationFailed(
            "Plan is blocked — resolve all failures first".to_string(),
        ));
    }

    // ── Resolve model path ─────────────────────────────────────────────
    let resolved_model_path = model_path
        .map(|p| p.to_path_buf())
        .or_else(|| {
            // Try to extract from plan
            plan.model.weights_available.then(|| {
                // Default: look in standard locations
                std::path::PathBuf::from("/home/noah/src/models/qwen2.5-coder-0.5b")
            })
        })
        .ok_or_else(|| {
            CliError::ValidationFailed(
                "--model-path is required for apply (model weights directory)".to_string(),
            )
        })?;

    let resolved_data_path = data
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from(&plan.data.train_path));

    if !json_output {
        output::header("apr train apply — Executing Training Plan");
        println!();
        output::kv("  Strategy", &plan.hyperparameters.strategy);
        if plan.hyperparameters.strategy == "manual" {
            if let Some(ref m) = plan.hyperparameters.manual {
                output::kv("  Learning rate", format!("{:.2e}", m.learning_rate));
                output::kv("  LoRA rank", m.lora_rank.to_string());
                output::kv("  Batch size", m.batch_size.to_string());
            }
        } else {
            output::kv("  Budget", format!("{} trials", plan.hyperparameters.budget));
            output::kv(
                "  Mode",
                if plan.hyperparameters.scout { "scout (1 epoch/trial)" } else { "full" },
            );
        }
        output::kv("  Model", &plan.model.size);
        output::kv("  Data", &plan.data.train_path);
        output::kv("  Output", output_dir.display().to_string());
        println!();
    }

    // ── Execute ────────────────────────────────────────────────────────
    let apply_config = entrenar::finetune::ApplyConfig {
        model_path: resolved_model_path,
        data_path: resolved_data_path,
        output_dir: output_dir.to_path_buf(),
        on_trial_complete: None,
    };

    let result = entrenar::finetune::execute_plan(&plan, &apply_config)
        .map_err(|e| CliError::ValidationFailed(format!("Training failed: {e}")))?;

    // ── Display results ────────────────────────────────────────────────
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        print_apply_result(&result);
    }

    Ok(())
}

/// Format a parameter count for display.
fn format_params(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// Format minutes into human-readable duration.
fn format_duration(minutes: f64) -> String {
    if minutes < 1.0 {
        format!("{:.0} sec", minutes * 60.0)
    } else if minutes < 60.0 {
        format!("{:.1} min", minutes)
    } else {
        let hours = minutes / 60.0;
        format!("{:.1} hours", hours)
    }
}
