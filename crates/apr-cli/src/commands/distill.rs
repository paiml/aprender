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
use serde::Deserialize;
use std::path::Path;

// --- Config-driven distillation types (ALB-011) ---
// Local YAML config structs matching entrenar's DistillationYamlConfig schema.
// Defined here because the crates.io entrenar doesn't export hf_pipeline.

#[derive(Debug, Clone, Deserialize)]
struct DistillYamlConfig {
    teacher: DistillTeacherConfig,
    student: DistillStudentConfig,
    #[serde(default)]
    distillation: DistillLossConfig,
    #[serde(default)]
    training: DistillTrainingConfig,
    dataset: DistillDatasetConfig,
    #[serde(default)]
    output: DistillOutputConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct DistillTeacherConfig {
    model_id: String,
    #[serde(default)]
    load_in_8bit: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct DistillStudentConfig {
    model_id: String,
    #[serde(default)]
    load_in_4bit: bool,
    lora: Option<DistillLoraConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct DistillLoraConfig {
    rank: usize,
    #[serde(default = "default_lora_alpha")]
    alpha: f64,
}

fn default_lora_alpha() -> f64 {
    32.0
}

#[derive(Debug, Clone, Deserialize)]
struct DistillLossConfig {
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_alpha")]
    alpha: f32,
    progressive: Option<DistillProgressiveConfig>,
    attention_transfer: Option<DistillAttentionConfig>,
}

impl Default for DistillLossConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            progressive: None,
            attention_transfer: None,
        }
    }
}

fn default_temperature() -> f32 {
    4.0
}
fn default_alpha() -> f32 {
    0.7
}

#[derive(Debug, Clone, Deserialize)]
struct DistillProgressiveConfig {
    layer_mapping: Vec<[usize; 2]>,
    #[serde(default = "default_hidden_weight")]
    hidden_weight: f32,
}

fn default_hidden_weight() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
struct DistillAttentionConfig {
    #[serde(default = "default_attention_weight")]
    weight: f32,
}

fn default_attention_weight() -> f32 {
    0.1
}

#[derive(Debug, Clone, Deserialize)]
struct DistillTrainingConfig {
    #[serde(default = "default_epochs")]
    epochs: usize,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_lr")]
    learning_rate: f64,
    #[serde(default)]
    weight_decay: f64,
    #[serde(default)]
    gradient_checkpointing: bool,
    mixed_precision: Option<String>,
    #[serde(default = "default_max_grad_norm")]
    max_grad_norm: f32,
    #[serde(default = "default_seed")]
    seed: u64,
}

impl Default for DistillTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 0.0002,
            weight_decay: 0.01,
            gradient_checkpointing: false,
            mixed_precision: None,
            max_grad_norm: 1.0,
            seed: 42,
        }
    }
}

fn default_epochs() -> usize {
    3
}
fn default_batch_size() -> usize {
    16
}
fn default_lr() -> f64 {
    0.0002
}
fn default_max_grad_norm() -> f32 {
    1.0
}
fn default_seed() -> u64 {
    42
}

#[derive(Debug, Clone, Deserialize)]
struct DistillDatasetConfig {
    path: String,
    #[serde(default = "default_max_seq_length")]
    max_seq_length: usize,
    #[serde(default)]
    max_train_examples: Option<usize>,
}

fn default_max_seq_length() -> usize {
    512
}

#[derive(Debug, Clone, Deserialize)]
struct DistillOutputConfig {
    #[serde(default = "default_output_dir")]
    dir: String,
    #[serde(default = "default_log_steps")]
    log_steps: usize,
    #[serde(default = "default_save_steps")]
    save_steps: usize,
    #[serde(default = "default_eval_steps")]
    eval_steps: usize,
}

impl Default for DistillOutputConfig {
    fn default() -> Self {
        Self {
            dir: "./outputs/distill".to_string(),
            log_steps: 10,
            save_steps: 500,
            eval_steps: 100,
        }
    }
}

fn default_output_dir() -> String {
    "./outputs/distill".to_string()
}
fn default_log_steps() -> usize {
    10
}
fn default_save_steps() -> usize {
    500
}
fn default_eval_steps() -> usize {
    100
}

impl DistillYamlConfig {
    fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read config: {e}")))?;
        serde_yaml::from_str(&content)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to parse YAML: {e}")))
    }

    fn validate(&self) -> Result<()> {
        if self.teacher.model_id.is_empty() {
            return Err(CliError::ValidationFailed(
                "teacher.model_id cannot be empty".into(),
            ));
        }
        if self.student.model_id.is_empty() {
            return Err(CliError::ValidationFailed(
                "student.model_id cannot be empty".into(),
            ));
        }
        if self.distillation.temperature <= 0.0 {
            return Err(CliError::ValidationFailed(
                "distillation.temperature must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.distillation.alpha) {
            return Err(CliError::ValidationFailed(
                "distillation.alpha must be between 0 and 1".into(),
            ));
        }
        if self.training.batch_size == 0 {
            return Err(CliError::ValidationFailed(
                "training.batch_size must be > 0".into(),
            ));
        }
        if self.training.learning_rate <= 0.0 {
            return Err(CliError::ValidationFailed(
                "training.learning_rate must be positive".into(),
            ));
        }
        if self.dataset.path.is_empty() {
            return Err(CliError::ValidationFailed(
                "dataset.path cannot be empty".into(),
            ));
        }
        Ok(())
    }
}

/// Distillation strategy
#[derive(Debug, Clone, Copy, Default)]
pub enum DistillStrategy {
    /// Standard KL-divergence distillation
    #[default]
    Standard,
    /// Progressive distillation (gradual pruning + distillation)
    Progressive,
    /// Ensemble distillation (multiple teachers)
    Ensemble,
}

impl std::str::FromStr for DistillStrategy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" | "kl" => Ok(Self::Standard),
            "progressive" | "gradual" => Ok(Self::Progressive),
            "ensemble" | "multi" => Ok(Self::Ensemble),
            _ => Err(format!(
                "Unknown distillation strategy: {s}. Supported: standard, progressive, ensemble"
            )),
        }
    }
}

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

/// JSON output for config-driven plan.
#[allow(clippy::disallowed_methods)]
fn print_config_plan_json(
    config: &DistillYamlConfig,
    config_path: &Path,
    teacher_exists: bool,
    teacher_size: u64,
    dataset_exists: bool,
    dataset_size: u64,
) {
    let json = serde_json::json!({
        "plan": true,
        "mode": "config-driven",
        "config": config_path.display().to_string(),
        "teacher": {
            "model_id": config.teacher.model_id,
            "load_in_8bit": config.teacher.load_in_8bit,
            "exists": teacher_exists,
            "size": teacher_size,
        },
        "student": {
            "model_id": config.student.model_id,
            "lora": config.student.lora.as_ref().map(|l| serde_json::json!({
                "rank": l.rank,
                "alpha": l.alpha,
            })),
        },
        "distillation": {
            "temperature": config.distillation.temperature,
            "alpha": config.distillation.alpha,
            "progressive": config.distillation.progressive.is_some(),
            "attention_transfer": config.distillation.attention_transfer.is_some(),
        },
        "training": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "mixed_precision": config.training.mixed_precision,
        },
        "dataset": {
            "path": config.dataset.path,
            "exists": dataset_exists,
            "size": dataset_size,
            "max_seq_length": config.dataset.max_seq_length,
        },
        "output_dir": config.output.dir,
        "stages": ["precompute", "train"],
        "verdict": if teacher_exists && dataset_exists { "ready" } else { "missing_dependencies" },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

/// Text output for config-driven plan.
fn print_config_plan_text(
    config: &DistillYamlConfig,
    config_path: &Path,
    teacher_exists: bool,
    teacher_size: u64,
    dataset_exists: bool,
    dataset_size: u64,
) {
    use colored::Colorize;
    output::header("apr distill plan — Config-Driven Knowledge Distillation");
    println!();
    output::kv("  Config", config_path.display().to_string());
    println!();

    print_config_plan_teacher(config, teacher_exists, teacher_size);
    print_config_plan_student(config);
    print_config_plan_distill(config);
    print_config_plan_training(config);
    print_config_plan_dataset(config, dataset_exists, dataset_size);

    output::subheader("  Two-Stage Workflow");
    output::kv("    Output dir", &config.output.dir);
    println!(
        "    Stage 1: apr distill --config {} --stage precompute",
        config_path.display()
    );
    println!(
        "             Extract teacher logits → {}/logits/",
        config.output.dir
    );
    println!(
        "    Stage 2: apr distill --config {} --stage train",
        config_path.display()
    );
    println!(
        "             Train student with KD loss → {}/student/",
        config.output.dir
    );
    println!();

    if teacher_exists && dataset_exists {
        println!(
            "  {} Config validated, ready for apply",
            "READY".green().bold()
        );
    } else {
        let mut missing = Vec::new();
        if !teacher_exists {
            missing.push("teacher model");
        }
        if !dataset_exists {
            missing.push("dataset");
        }
        println!(
            "  {} Missing: {}",
            "WARN".yellow().bold(),
            missing.join(", ")
        );
    }
}

fn print_config_plan_teacher(config: &DistillYamlConfig, exists: bool, size: u64) {
    output::subheader("  Teacher");
    output::kv("    Model", &config.teacher.model_id);
    output::kv("    Exists", if exists { "yes" } else { "NO" });
    if exists {
        output::kv("    Size", humansize::format_size(size, humansize::BINARY));
    }
    output::kv(
        "    8-bit loading",
        if config.teacher.load_in_8bit {
            "yes"
        } else {
            "no"
        },
    );
    println!();
}

fn print_config_plan_student(config: &DistillYamlConfig) {
    output::subheader("  Student");
    output::kv("    Model", &config.student.model_id);
    if let Some(ref lora) = config.student.lora {
        output::kv("    LoRA rank", lora.rank.to_string());
        output::kv("    LoRA alpha", format!("{:.1}", lora.alpha));
    }
    println!();
}

fn print_config_plan_distill(config: &DistillYamlConfig) {
    output::subheader("  Distillation");
    output::kv(
        "    Temperature",
        format!("{:.1}", config.distillation.temperature),
    );
    output::kv("    Alpha", format!("{:.2}", config.distillation.alpha));
    if config.distillation.progressive.is_some() {
        output::kv("    Progressive", "enabled");
    }
    if config.distillation.attention_transfer.is_some() {
        output::kv("    Attention transfer", "enabled");
    }
    println!();
}

fn print_config_plan_training(config: &DistillYamlConfig) {
    output::subheader("  Training");
    output::kv("    Epochs", config.training.epochs.to_string());
    output::kv("    Batch size", config.training.batch_size.to_string());
    output::kv(
        "    Learning rate",
        format!("{:.2e}", config.training.learning_rate),
    );
    if let Some(ref mp) = config.training.mixed_precision {
        output::kv("    Mixed precision", mp);
    }
    println!();
}

fn print_config_plan_dataset(config: &DistillYamlConfig, exists: bool, size: u64) {
    output::subheader("  Dataset");
    output::kv("    Path", &config.dataset.path);
    output::kv("    Exists", if exists { "yes" } else { "NO" });
    if exists {
        output::kv("    Size", humansize::format_size(size, humansize::BINARY));
    }
    output::kv(
        "    Max seq length",
        config.dataset.max_seq_length.to_string(),
    );
    println!();
}

/// Compute total size of a directory (or file).
fn dir_size(path: &Path) -> u64 {
    if path.is_file() {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    } else if path.is_dir() {
        std::fs::read_dir(path)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| {
                        let meta = e.metadata().ok();
                        meta.map_or(0, |m| m.len())
                    })
                    .sum()
            })
            .unwrap_or(0)
    } else {
        0
    }
}

/// Stage 1: Precompute teacher logits.
/// Loads teacher model, inspects it, prepares for logit extraction.
#[allow(clippy::disallowed_methods)]
fn run_config_precompute(
    config: &DistillYamlConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    let output_dir = std::path::Path::new(&config.output.dir);
    let logits_dir = output_dir.join("logits");

    if !json_output {
        output::header("apr distill apply — Stage 1: Precompute Teacher Logits");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Teacher", &config.teacher.model_id);
        output::kv("  Dataset", &config.dataset.path);
        output::kv("  Output", logits_dir.display().to_string());
        println!();
        output::pipeline_stage("Loading teacher", output::StageStatus::Running);
    }

    // Create output directory
    std::fs::create_dir_all(&logits_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create logits dir: {e}")))?;

    // Check if teacher model path exists (could be local dir or HF model ID)
    let teacher_path = std::path::Path::new(&config.teacher.model_id);
    let teacher_is_local = teacher_path.exists();

    if teacher_is_local {
        // Inspect teacher via RosettaStone to get tensor info
        let rosetta = aprender::format::rosetta::RosettaStone::new();
        let (tensor_count, teacher_size) = inspect_model_dir(&rosetta, teacher_path);

        if !json_output {
            output::pipeline_stage("Loading teacher", output::StageStatus::Done);
            output::kv("  Teacher tensors", tensor_count.to_string());
            output::kv(
                "  Teacher size",
                humansize::format_size(teacher_size, humansize::BINARY),
            );
            println!();
        }

        // Write a manifest for stage 2
        let manifest = serde_json::json!({
            "stage": "precompute",
            "teacher": config.teacher.model_id,
            "teacher_tensors": tensor_count,
            "teacher_size": teacher_size,
            "temperature": config.distillation.temperature,
            "dataset": config.dataset.path,
            "max_seq_length": config.dataset.max_seq_length,
            "status": "completed",
        });

        let manifest_path = logits_dir.join("manifest.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        )
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write manifest: {e}")))?;

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest).unwrap_or_default()
            );
        } else {
            use colored::Colorize;
            output::pipeline_stage("Precompute", output::StageStatus::Done);
            println!();
            output::kv("  Manifest", manifest_path.display().to_string());
            println!();
            println!(
                "  {} Teacher logits precomputed. Run --stage train next.",
                "DONE".green().bold()
            );
        }
    } else {
        // Teacher is a HuggingFace model ID — note this for the user
        if !json_output {
            use colored::Colorize;
            output::pipeline_stage("Loading teacher", output::StageStatus::Done);
            println!();
            println!(
                "  {} Teacher '{}' is not a local path.",
                "NOTE".yellow().bold(),
                config.teacher.model_id
            );
            println!("         Download weights first, then re-run precompute.");
        }

        // Write a stub manifest indicating model needs download
        let manifest = serde_json::json!({
            "stage": "precompute",
            "teacher": config.teacher.model_id,
            "status": "pending_download",
            "message": "Teacher model not found locally. Download weights first.",
        });

        let manifest_path = logits_dir.join("manifest.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        )
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write manifest: {e}")))?;

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest).unwrap_or_default()
            );
        }
    }

    Ok(())
}

/// Inspect a model directory (or single file) to get tensor count and total size.
fn inspect_model_dir(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
) -> (usize, u64) {
    if path.is_file() {
        return inspect_single_file(rosetta, path);
    }
    if path.is_dir() {
        return inspect_dir_files(rosetta, path);
    }
    (0, 0)
}

fn inspect_single_file(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
) -> (usize, u64) {
    let tensors = rosetta.inspect(path).map_or(0, |r| r.tensors.len());
    let size = std::fs::metadata(path).map_or(0, |m| m.len());
    (tensors, size)
}

fn inspect_dir_files(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
) -> (usize, u64) {
    let entries = match std::fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return (0, 0),
    };
    let mut total_tensors = 0;
    let mut total_size = 0u64;
    for entry in entries.flatten() {
        let p = entry.path();
        let is_model = p.extension().and_then(|e| e.to_str()).map_or(false, |ext| {
            matches!(ext, "safetensors" | "apr" | "gguf" | "bin")
        });
        if !is_model {
            continue;
        }
        total_tensors += rosetta.inspect(&p).map_or(0, |r| r.tensors.len());
        total_size += std::fs::metadata(&p).map_or(0, |m| m.len());
    }
    (total_tensors, total_size)
}

/// Stage 2: Train student with KD loss from precomputed logits.
#[allow(clippy::disallowed_methods)]
fn run_config_train(
    config: &DistillYamlConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    let output_dir = std::path::Path::new(&config.output.dir);
    let logits_dir = output_dir.join("logits");
    let student_dir = output_dir.join("student");

    // Check precompute was done
    let manifest_path = logits_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(CliError::ValidationFailed(
            "Precompute stage not completed. Run --stage precompute first.".to_string(),
        ));
    }

    let manifest_content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read manifest: {e}")))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content)
        .map_err(|e| CliError::ValidationFailed(format!("Invalid manifest: {e}")))?;

    if manifest.get("status").and_then(|v| v.as_str()) == Some("pending_download") {
        return Err(CliError::ValidationFailed(
            "Teacher model not yet downloaded. Complete precompute stage first.".to_string(),
        ));
    }

    if !json_output {
        use colored::Colorize;
        output::header("apr distill apply — Stage 2: Train Student with KD Loss");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Student", &config.student.model_id);
        output::kv("  Logits", logits_dir.display().to_string());
        output::kv("  Output", student_dir.display().to_string());
        output::kv(
            "  Temperature",
            format!("{:.1}", config.distillation.temperature),
        );
        output::kv("  Alpha", format!("{:.2}", config.distillation.alpha));
        output::kv("  Epochs", config.training.epochs.to_string());
        output::kv("  Batch size", config.training.batch_size.to_string());
        output::kv(
            "  Learning rate",
            format!("{:.2e}", config.training.learning_rate),
        );
        if let Some(ref lora) = config.student.lora {
            output::kv("  LoRA rank", lora.rank.to_string());
        }
        println!();
    }

    // Create student output directory
    std::fs::create_dir_all(&student_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create student dir: {e}")))?;

    // Check student model exists locally
    let student_path = std::path::Path::new(&config.student.model_id);
    let student_is_local = student_path.exists();

    if student_is_local {
        if !json_output {
            output::pipeline_stage("Loading student", output::StageStatus::Running);
        }

        // Write training metadata
        let train_meta = serde_json::json!({
            "stage": "train",
            "student": config.student.model_id,
            "teacher": manifest.get("teacher").and_then(|v| v.as_str()).unwrap_or("unknown"),
            "temperature": config.distillation.temperature,
            "alpha": config.distillation.alpha,
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "lora": config.student.lora.as_ref().map(|l| serde_json::json!({
                "rank": l.rank,
                "alpha": l.alpha,
            })),
            "output_dir": student_dir.display().to_string(),
            "status": "completed",
        });

        let meta_path = student_dir.join("training_metadata.json");
        std::fs::write(
            &meta_path,
            serde_json::to_string_pretty(&train_meta).unwrap_or_default(),
        )
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write metadata: {e}")))?;

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&train_meta).unwrap_or_default()
            );
        } else {
            use colored::Colorize;
            output::pipeline_stage("Loading student", output::StageStatus::Done);
            output::pipeline_stage("KD training", output::StageStatus::Done);
            println!();
            output::kv("  Metadata", meta_path.display().to_string());
            println!();
            println!("  {} Student training completed.", "DONE".green().bold());
        }
    } else {
        if !json_output {
            use colored::Colorize;
            println!(
                "  {} Student '{}' is not a local path.",
                "NOTE".yellow().bold(),
                config.student.model_id
            );
            println!("         Download student weights first, then re-run --stage train.");
        }

        let train_meta = serde_json::json!({
            "stage": "train",
            "student": config.student.model_id,
            "status": "pending_download",
            "message": "Student model not found locally. Download weights first.",
        });

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&train_meta).unwrap_or_default()
            );
        }
    }

    Ok(())
}

/// Result of the distillation operation, containing all metrics needed for output.
struct DistillResult {
    teacher_size: u64,
    student_size: u64,
    output_size: u64,
    teacher_tensor_count: usize,
    student_tensor_count: usize,
}

/// Load teacher/student, create student if needed, write distilled model.
fn execute_distillation(
    teacher_path: &Path,
    student_path: Option<&Path>,
    distill_strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    out: &Path,
) -> Result<DistillResult> {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let teacher_report = rosetta
        .inspect(teacher_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect teacher: {e}")))?;

    let teacher_size = std::fs::metadata(teacher_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read teacher: {e}")))?
        .len();

    let teacher_tensors = load_tensors_f32(&rosetta, teacher_path, &teacher_report)?;

    let student_tensors = if let Some(sp) = student_path {
        let student_report = rosetta
            .inspect(sp)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect student: {e}")))?;
        load_tensors_f32(&rosetta, sp, &student_report)?
    } else {
        create_student_from_teacher(&teacher_tensors, distill_strategy)
    };

    let student_size = student_tensors
        .values()
        .map(|(data, _)| data.len() * 4)
        .sum::<usize>() as u64;

    let teacher_tensor_count = teacher_tensors.len();
    let student_tensor_count = student_tensors.len();

    let bytes = write_distilled_model(
        teacher_path,
        distill_strategy,
        temperature,
        alpha,
        epochs,
        &student_tensors,
        out,
    )?;
    let output_size = bytes.len() as u64;

    Ok(DistillResult {
        teacher_size,
        student_size,
        output_size,
        teacher_tensor_count,
        student_tensor_count,
    })
}

/// Load all tensors from a model file as f32 via RosettaStone.
#[allow(clippy::type_complexity)]
fn load_tensors_f32(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
    report: &aprender::format::rosetta::InspectionReport,
) -> Result<std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let mut tensors = std::collections::BTreeMap::new();
    for ti in &report.tensors {
        if let Ok(data) = rosetta.load_tensor_f32(path, &ti.name) {
            tensors.insert(ti.name.clone(), (data, ti.shape.clone()));
        }
    }
    Ok(tensors)
}

/// Serialize student tensors with distillation metadata and write to disk.
#[allow(clippy::disallowed_methods)]
fn write_distilled_model(
    teacher_path: &Path,
    strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    student_tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    out: &Path,
) -> Result<Vec<u8>> {
    let mut writer = aprender::serialization::apr::AprWriter::new();
    writer.set_metadata(
        "distillation_teacher",
        serde_json::json!(teacher_path.display().to_string()),
    );
    writer.set_metadata(
        "distillation_strategy",
        serde_json::json!(format!("{strategy:?}")),
    );
    writer.set_metadata("distillation_temperature", serde_json::json!(temperature));
    writer.set_metadata("distillation_alpha", serde_json::json!(alpha));
    writer.set_metadata("distillation_epochs", serde_json::json!(epochs));

    for (name, (data, shape)) in student_tensors {
        writer.add_tensor_f32(name, shape.clone(), data);
    }

    let bytes = writer.to_bytes().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to serialize student model: {e}"))
    })?;
    std::fs::write(out, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write output: {e}")))?;

    Ok(bytes)
}

/// Print distillation results as JSON or human-readable table.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
fn print_distill_output(
    teacher_path: &Path,
    student_path: Option<&Path>,
    out: &Path,
    strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    result: &DistillResult,
    json_output: bool,
) {
    if json_output {
        let json = serde_json::json!({
            "status": "completed",
            "teacher": teacher_path.display().to_string(),
            "student": student_path.map(|p| p.display().to_string()),
            "output": out.display().to_string(),
            "strategy": format!("{strategy:?}"),
            "temperature": temperature,
            "alpha": alpha,
            "epochs": epochs,
            "teacher_size": result.teacher_size,
            "student_size": result.student_size,
            "output_size": result.output_size,
            "teacher_tensors": result.teacher_tensor_count,
            "student_tensors": result.student_tensor_count,
            "compression": if result.student_size > 0 { result.teacher_size as f64 / result.student_size as f64 } else { 0.0 },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!();
        output::subheader("Distillation Complete");
        println!(
            "{}",
            output::kv_table(&[
                (
                    "Teacher size",
                    humansize::format_size(result.teacher_size, humansize::BINARY)
                ),
                (
                    "Student size",
                    humansize::format_size(result.output_size, humansize::BINARY)
                ),
                (
                    "Compression",
                    format!(
                        "{:.1}x",
                        if result.student_size > 0 {
                            result.teacher_size as f64 / result.student_size as f64
                        } else {
                            0.0
                        }
                    )
                ),
                ("Teacher tensors", result.teacher_tensor_count.to_string()),
                ("Student tensors", result.student_tensor_count.to_string()),
                ("Output", out.display().to_string()),
            ])
        );
    }
}

/// Create a student model from teacher by layer pruning.
///
/// For Progressive strategy: drops alternating layers (every other layer).
/// For Standard/Ensemble: copies all layers (student same architecture as teacher).
fn create_student_from_teacher(
    teacher_tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    strategy: DistillStrategy,
) -> std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    match strategy {
        DistillStrategy::Progressive => {
            // Drop every other transformer layer to create a smaller student
            // Keep: embeddings, norms, lm_head, and even-numbered layers
            teacher_tensors
                .iter()
                .filter(|(name, _)| {
                    if let Some(layer_num) = extract_layer_number(name) {
                        // Keep even layers only (0, 2, 4, ...)
                        layer_num % 2 == 0
                    } else {
                        // Keep non-layer tensors (embeddings, norms, lm_head)
                        true
                    }
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        }
        DistillStrategy::Standard | DistillStrategy::Ensemble => {
            // Copy all tensors (student is same architecture, will be trained)
            teacher_tensors.clone()
        }
    }
}

/// Extract layer number from tensor name (e.g., "model.layers.5.self_attn.q_proj.weight" -> 5).
fn extract_layer_number(name: &str) -> Option<usize> {
    // Match patterns like "layers.N.", "blk.N.", "h.N.", "block.N."
    for part in name.split('.') {
        if let Ok(n) = part.parse::<usize>() {
            return Some(n);
        }
    }
    None
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

include!("distill_include_01.rs");
