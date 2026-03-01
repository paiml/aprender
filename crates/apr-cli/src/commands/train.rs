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
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_plan(
    data: Option<&std::path::Path>,
    model_size: &str,
    model_path: Option<&std::path::Path>,
    num_classes: usize,
    task: &str,
    config_path: Option<&std::path::Path>,
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
    match task {
        "pretrain" | "causal_lm" => run_plan_pretrain(config_path, json_output),
        "classify" => {
            let data = data.ok_or_else(|| {
                CliError::ValidationFailed(
                    "--data is required for --task classify".to_string(),
                )
            })?;
            run_plan_classify(
                data, model_size, model_path, num_classes, output_dir, strategy,
                budget, scout, max_epochs, manual_lr, manual_lora_rank,
                manual_batch_size, val_data, test_data, format, json_output,
            )
        }
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown task type: {task}. Supported: classify, pretrain"
        ))),
    }
}

/// Plan for classification fine-tuning (existing behavior).
#[allow(clippy::too_many_arguments)]
fn run_plan_classify(
    data: &std::path::Path,
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
    val_data: Option<&std::path::Path>,
    test_data: Option<&std::path::Path>,
    format: &str,
    json_output: bool,
) -> Result<()> {
    let config = entrenar::finetune::PlanConfig {
        task: "classify".to_string(),
        data_path: data.to_path_buf(),
        val_path: val_data.map(std::path::Path::to_path_buf),
        test_path: test_data.map(std::path::Path::to_path_buf),
        model_size: model_size.to_string(),
        model_path: model_path.map(std::path::Path::to_path_buf),
        num_classes,
        output_dir: output_dir.to_path_buf(),
        strategy: strategy.to_string(),
        budget,
        scout,
        max_epochs,
        manual_lr,
        manual_lora_rank,
        manual_batch_size,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };

    let plan = entrenar::finetune::training_plan(&config)
        .map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    let effective_format = if json_output { "json" } else { format };

    match effective_format {
        "json" => println!("{}", plan.to_json()),
        "yaml" => println!("{}", plan.to_yaml()),
        _ => print_plan_text(&plan),
    }

    Ok(())
}

/// Plan for causal LM pre-training from YAML config (ALB-009).
fn run_plan_pretrain(
    config_path: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    let config_path = config_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--config <yaml> is required for --task pretrain".to_string(),
        )
    })?;

    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    // Load and validate the training config
    let spec = entrenar::config::load_config(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Config error: {e}")))?;

    entrenar::config::validate_config(&spec)
        .map_err(|e| CliError::ValidationFailed(format!("Validation error: {e}")))?;

    // Display the pre-training plan
    if json_output {
        let plan = serde_json::json!({
            "task": "pretrain",
            "config": config_path.display().to_string(),
            "model": {
                "path": spec.model.path.display().to_string(),
                "mode": format!("{:?}", spec.model.mode),
            },
            "data": {
                "train": spec.data.train.display().to_string(),
                "batch_size": spec.data.batch_size,
                "seq_len": spec.data.seq_len,
            },
            "optimizer": {
                "name": spec.optimizer.name,
                "lr": spec.optimizer.lr,
            },
            "training": {
                "epochs": spec.training.epochs,
                "mode": format!("{:?}", spec.training.mode),
                "warmup_steps": spec.training.warmup_steps,
                "gradient_accumulation": spec.training.gradient_accumulation,
                "mixed_precision": spec.training.mixed_precision,
            },
            "verdict": "ready",
        });
        println!("{}", serde_json::to_string_pretty(&plan).unwrap_or_default());
    } else {
        output::header("apr train plan — Pre-training Plan (Causal LM)");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Model path", spec.model.path.display().to_string());
        output::kv("  Model mode", format!("{:?}", spec.model.mode));
        output::kv("  Training mode", format!("{:?}", spec.training.mode));
        println!();
        output::kv("  Train data", spec.data.train.display().to_string());
        output::kv("  Batch size", spec.data.batch_size.to_string());
        if let Some(seq_len) = spec.data.seq_len {
            output::kv("  Sequence length", seq_len.to_string());
        }
        println!();
        output::kv("  Optimizer", &spec.optimizer.name);
        output::kv("  Learning rate", format!("{:.2e}", spec.optimizer.lr));
        output::kv("  Epochs", spec.training.epochs.to_string());
        output::kv("  Warmup steps", spec.training.warmup_steps.to_string());
        if let Some(accum) = spec.training.gradient_accumulation {
            output::kv("  Gradient accumulation", accum.to_string());
        }
        if let Some(ref mp) = spec.training.mixed_precision {
            output::kv("  Mixed precision", mp);
        }
        if let Some(ref lora) = spec.lora {
            println!();
            output::kv("  LoRA rank", lora.rank.to_string());
            output::kv("  LoRA alpha", format!("{:.1}", lora.alpha));
        }
        println!();
        println!("  {} Config validated, ready for apply", "READY".green().bold());
    }

    Ok(())
}

include!("train_output.rs");

/// Load a training plan from file, or generate one inline.
#[allow(clippy::too_many_arguments)]
fn load_or_generate_plan(
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
) -> Result<entrenar::finetune::TrainingPlan> {
    if let Some(plan_path) = plan_file {
        let content = std::fs::read_to_string(plan_path).map_err(|_| {
            CliError::FileNotFound(plan_path.to_path_buf())
        })?;
        entrenar::finetune::TrainingPlan::from_str(&content)
            .map_err(|e| {
                CliError::ValidationFailed(format!(
                    "Failed to parse plan file {}: {e}",
                    plan_path.display()
                ))
            })
    } else {
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
            model_path: model_path.map(std::path::Path::to_path_buf),
            num_classes,
            output_dir: output_dir.to_path_buf(),
            strategy: strategy.to_string(),
            budget,
            scout,
            max_epochs,
            manual_lr,
            manual_lora_rank,
            manual_batch_size,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        entrenar::finetune::training_plan(&config)
            .map_err(|e| CliError::ValidationFailed(e.to_string()))
    }
}

/// Print apply summary before execution.
fn print_apply_summary(
    plan: &entrenar::finetune::TrainingPlan,
    output_dir: &std::path::Path,
) {
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

/// Run `apr train apply` — execute a training plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_apply(
    plan_file: Option<&std::path::Path>,
    config_path: Option<&std::path::Path>,
    task: &str,
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
    match task {
        "pretrain" | "causal_lm" => run_apply_pretrain(config_path, json_output),
        "classify" => run_apply_classify(
            plan_file, data, model_size, model_path, num_classes,
            output_dir, strategy, budget, scout, max_epochs,
            manual_lr, manual_lora_rank, manual_batch_size, json_output,
        ),
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown task type: {task}. Supported: classify, pretrain"
        ))),
    }
}

/// Execute classification fine-tuning (existing behavior).
#[allow(clippy::too_many_arguments)]
fn run_apply_classify(
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
    let plan = load_or_generate_plan(
        plan_file, data, model_size, model_path, num_classes,
        output_dir, strategy, budget, scout, max_epochs,
        manual_lr, manual_lora_rank, manual_batch_size,
    )?;

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

    let resolved_model_path = model_path
        .map(std::path::Path::to_path_buf)
        .or_else(|| {
            plan.model.weights_available.then(|| {
                std::path::PathBuf::from("/home/noah/src/models/qwen2.5-coder-0.5b")
            })
        })
        .ok_or_else(|| {
            CliError::ValidationFailed(
                "--model-path is required for apply (model weights directory)".to_string(),
            )
        })?;

    let resolved_data_path = match data {
        Some(p) => p.to_path_buf(),
        None => std::path::PathBuf::from(&plan.data.train_path),
    };

    if !json_output {
        print_apply_summary(&plan, output_dir);
    }

    let apply_config = entrenar::finetune::ApplyConfig {
        model_path: resolved_model_path,
        data_path: resolved_data_path,
        output_dir: output_dir.to_path_buf(),
        on_trial_complete: None,
    };

    let result = entrenar::finetune::execute_plan(&plan, &apply_config)
        .map_err(|e| CliError::ValidationFailed(format!("Training failed: {e}")))?;

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

/// Execute causal LM pre-training from YAML config (ALB-009).
fn run_apply_pretrain(
    config_path: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    let config_path = config_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--config <yaml> is required for --task pretrain".to_string(),
        )
    })?;

    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    if !json_output {
        output::header("apr train apply — Causal LM Pre-training");
        println!();
        output::kv("  Config", config_path.display().to_string());
        println!();
    }

    entrenar::config::train_from_yaml(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Training failed: {e}")))?;

    if json_output {
        println!("{{\"status\":\"completed\",\"task\":\"pretrain\"}}");
    } else {
        println!();
        println!("  {} Pre-training completed", "DONE".green().bold());
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
