//! Distill command implementation (GH-247)
//!
//! Knowledge distillation pipeline for transferring knowledge from a
//! teacher model to a smaller student model.
//!
//! # Example
//!
//! ```bash
//! apr distill teacher.apr --student pruned.apr --data train.jsonl -o distilled.apr
//! apr distill teacher.apr --progressive --target-ratio 0.5 --data train.jsonl -o distilled.apr
//! apr distill teacher.apr --plan --json
//! ```

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use serde::Deserialize;
use std::path::Path;

/// Validate distillation parameters (temperature, alpha).
fn validate_distill_params(temperature: f64, alpha: f64) -> Result<()> {
    if temperature <= 0.0 {
        return Err(CliError::ValidationFailed(format!(
            "Temperature must be positive, got {temperature}"
        )));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(CliError::ValidationFailed(format!(
            "Alpha must be between 0 and 1, got {alpha}"
        )));
    }
    Ok(())
}

/// Validate that optional file paths exist on disk.
fn validate_optional_paths(student_path: Option<&Path>, data_path: Option<&Path>) -> Result<()> {
    if let Some(student) = student_path {
        if !student.exists() {
            return Err(CliError::FileNotFound(student.to_path_buf()));
        }
    }
    if let Some(data) = data_path {
        if !data.exists() {
            return Err(CliError::FileNotFound(data.to_path_buf()));
        }
    }
    Ok(())
}

/// Print the distill run header (file-based mode).
#[allow(clippy::too_many_arguments)]
fn print_distill_header(
    teacher_path: &Path,
    student_path: Option<&Path>,
    data_path: Option<&Path>,
    distill_strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    out: &Path,
    json_output: bool,
) {
    if !json_output {
        output::header("APR Distill");
        let mut pairs = vec![
            ("Teacher", teacher_path.display().to_string()),
            ("Strategy", format!("{distill_strategy:?}")),
            ("Temperature", format!("{temperature:.1}")),
            ("Alpha", format!("{alpha:.2}")),
            ("Epochs", epochs.to_string()),
            ("Output", out.display().to_string()),
        ];
        if let Some(student) = student_path {
            pairs.insert(1, ("Student", student.display().to_string()));
        }
        if let Some(data) = data_path {
            pairs.push(("Training data", data.display().to_string()));
        }
        println!("{}", output::kv_table(&pairs));
        println!();
    }
}

/// Run the distill command — dispatches between file-based and config-driven modes.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "mutating_output_contract"
)]
pub(crate) fn run(
    teacher_path: Option<&Path>,
    student_path: Option<&Path>,
    data_path: Option<&Path>,
    output_path: Option<&Path>,
    strategy: &str,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    plan_only: bool,
    config_path: Option<&Path>,
    stage: Option<&str>,
    json_output: bool,
) -> Result<()> {
    // Config-driven mode (ALB-011): --config <yaml> [--stage precompute|train]
    if let Some(config) = config_path {
        return run_config_mode(config, stage, plan_only, json_output);
    }

    let teacher_path = teacher_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Teacher model path required. Use positional arg or --config <yaml>".to_string(),
        )
    })?;

    if !teacher_path.exists() {
        return Err(CliError::FileNotFound(teacher_path.to_path_buf()));
    }

    let distill_strategy: DistillStrategy = strategy.parse().map_err(CliError::ValidationFailed)?;
    validate_distill_params(temperature, alpha)?;

    if plan_only {
        return run_plan(
            teacher_path,
            student_path,
            distill_strategy,
            temperature,
            alpha,
            epochs,
            json_output,
        );
    }

    if student_path.is_none() && !matches!(distill_strategy, DistillStrategy::Progressive) {
        return Err(CliError::ValidationFailed(
            "Student model required for standard distillation. Use --student <path>".to_string(),
        ));
    }

    let out = output_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Output path required. Use -o <path> to specify output.".to_string(),
        )
    })?;

    print_distill_header(
        teacher_path,
        student_path,
        data_path,
        distill_strategy,
        temperature,
        alpha,
        epochs,
        out,
        json_output,
    );
    validate_optional_paths(student_path, data_path)?;

    if !json_output {
        output::pipeline_stage("Distilling", output::StageStatus::Running);
    }

    let distill_result = execute_distillation(
        teacher_path,
        student_path,
        distill_strategy,
        temperature,
        alpha,
        epochs,
        out,
    )?;

    if !json_output {
        output::pipeline_stage("Distilling", output::StageStatus::Done);
    }

    print_distill_output(
        teacher_path,
        student_path,
        out,
        distill_strategy,
        temperature,
        alpha,
        epochs,
        &distill_result,
        json_output,
    );

    Ok(())
}

/// Config-driven distillation mode (ALB-011).
///
/// Supports two-stage workflow:
///   --plan: validate config + show estimates
///   --stage precompute: extract teacher logits to sharded files
///   --stage train: train student with KD loss from precomputed logits
fn run_config_mode(
    config_path: &Path,
    stage: Option<&str>,
    plan_only: bool,
    json_output: bool,
) -> Result<()> {
    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    let content = std::fs::read_to_string(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read config: {e}")))?;

    // Detect config type: text-based (has synthetic_data) vs logit KD (has distillation/dataset)
    let raw: serde_json::Value = serde_yaml::from_str(&content)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse YAML: {e}")))?;

    if raw.get("synthetic_data").is_some() {
        let config: TextDistillConfig = serde_yaml::from_str(&content)
            .map_err(|e| CliError::ValidationFailed(format!("Config error: {e}")))?;
        return run_text_config_mode(&config, config_path, stage, plan_only, json_output);
    }

    // Original logit KD config
    let config = DistillYamlConfig::load(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Config error: {e}")))?;

    config
        .validate()
        .map_err(|e| CliError::ValidationFailed(format!("Validation error: {e}")))?;

    if plan_only {
        return run_config_plan(&config, config_path, json_output);
    }

    match stage {
        Some("precompute") => run_config_precompute(&config, config_path, json_output),
        Some("train") => run_config_train(&config, config_path, json_output),
        Some(other) => Err(CliError::ValidationFailed(format!(
            "Unknown stage: {other}. Supported: precompute, train"
        ))),
        None => Err(CliError::ValidationFailed(
            "--stage <precompute|train> required with --config. Use --plan to see the plan."
                .to_string(),
        )),
    }
}

/// Text-based distillation config mode dispatch (GH-455).
fn run_text_config_mode(
    config: &TextDistillConfig,
    config_path: &Path,
    stage: Option<&str>,
    plan_only: bool,
    json_output: bool,
) -> Result<()> {
    // GH-504: Handle --plan for text-based configs
    if plan_only {
        return run_text_config_plan(config, config_path, json_output);
    }

    match stage {
        Some("generate") => run_text_generate(config, config_path, json_output),
        Some(other) => Err(CliError::ValidationFailed(format!(
            "Unknown stage: {other}. Supported: generate"
        ))),
        None => Err(CliError::ValidationFailed(
            "--stage generate required with text-based distillation config.".to_string(),
        )),
    }
}

/// Plan mode for text-based distillation (GH-504).
#[allow(clippy::disallowed_methods)]
fn run_text_config_plan(
    config: &TextDistillConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    let prompts_path = config_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(&config.synthetic_data.prompts);
    let prompt_count = if prompts_path.exists() {
        std::fs::read_to_string(&prompts_path)
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };
    let estimated_samples =
        prompt_count as u64 * u64::from(config.synthetic_data.samples_per_prompt);
    let estimated_tokens = estimated_samples * u64::from(config.teacher.max_tokens);

    if json_output {
        let json = serde_json::json!({
            "plan": true,
            "mode": "text-distillation",
            "config": config_path.display().to_string(),
            "teacher_model": config.teacher.model,
            "prompts_file": config.synthetic_data.prompts,
            "prompt_count": prompt_count,
            "samples_per_prompt": config.synthetic_data.samples_per_prompt,
            "estimated_samples": estimated_samples,
            "target_tokens": config.synthetic_data.target_tokens,
            "estimated_tokens": estimated_tokens,
            "max_tokens_per_sample": config.teacher.max_tokens,
            "temperature": config.teacher.temperature,
            "output_dir": config.synthetic_data.output,
            "stages": ["generate"],
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Distill — Text Config Plan");
        println!(
            "{}",
            output::kv_table(&[
                ("Config", config_path.display().to_string()),
                ("Teacher", config.teacher.model.clone()),
                ("Prompts", config.synthetic_data.prompts.clone()),
                ("Prompt count", format!("{prompt_count}")),
                (
                    "Samples/prompt",
                    format!("{}", config.synthetic_data.samples_per_prompt),
                ),
                ("Est. samples", format!("{estimated_samples}")),
                ("Est. tokens", format!("{estimated_tokens}")),
                ("Output", config.synthetic_data.output.clone()),
            ])
        );
        println!();
        println!("  Stages:");
        println!("    1. generate — Generate synthetic data from teacher");
        println!();
        println!(
            "  {} Run with --stage generate to execute.",
            output::badge_info("INFO")
        );
    }

    Ok(())
}

/// Plan mode for config-driven distillation.
/// Validates config, estimates resource usage, shows two-stage plan.
#[allow(clippy::disallowed_methods)]
fn run_config_plan(
    config: &DistillYamlConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    let dataset_path = std::path::Path::new(&config.dataset.path);
    let dataset_exists = dataset_path.exists();
    let dataset_size = if dataset_exists {
        std::fs::metadata(dataset_path)
            .map(|m| m.len())
            .unwrap_or(0)
    } else {
        0
    };
    let teacher_path = std::path::Path::new(&config.teacher.model_id);
    let teacher_exists = teacher_path.exists();
    let teacher_size = if teacher_exists {
        dir_size(teacher_path)
    } else {
        0
    };

    if json_output {
        print_config_plan_json(
            config,
            config_path,
            teacher_exists,
            teacher_size,
            dataset_exists,
            dataset_size,
        );
    } else {
        print_config_plan_text(
            config,
            config_path,
            teacher_exists,
            teacher_size,
            dataset_exists,
            dataset_size,
        );
    }
    Ok(())
}

/// Plan distillation (estimate only)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
fn run_plan(
    teacher_path: &Path,
    student_path: Option<&Path>,
    strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    json_output: bool,
) -> Result<()> {
    let teacher_size = std::fs::metadata(teacher_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read teacher: {e}")))?
        .len();

    let student_size = student_path
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(teacher_size / 2, |m| m.len());

    let peak_memory = teacher_size + student_size;

    if json_output {
        let json = serde_json::json!({
            "plan": true,
            "teacher": teacher_path.display().to_string(),
            "teacher_size": teacher_size,
            "student_size": student_size,
            "strategy": format!("{strategy:?}"),
            "temperature": temperature,
            "alpha": alpha,
            "epochs": epochs,
            "peak_memory": peak_memory,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Distill — Plan");
        println!(
            "{}",
            output::kv_table(&[
                ("Teacher", teacher_path.display().to_string()),
                (
                    "Teacher size",
                    humansize::format_size(teacher_size, humansize::BINARY),
                ),
                (
                    "Student size",
                    humansize::format_size(student_size, humansize::BINARY),
                ),
                ("Strategy", format!("{strategy:?}")),
                ("Temperature", format!("{temperature:.1}")),
                ("Alpha", format!("{alpha:.2}")),
                ("Epochs", epochs.to_string()),
                (
                    "Peak memory",
                    humansize::format_size(peak_memory, humansize::BINARY),
                ),
            ])
        );
        println!();
        println!(
            "  {} Run without --plan to execute.",
            output::badge_info("INFO"),
        );
    }

    Ok(())
}

include!("distill_types.rs");
include!("distill_config_and_execute.rs");
include!("distill_train_and_write.rs");
include!("distill_text_generate.rs");
include!("distill_include_01.rs");
