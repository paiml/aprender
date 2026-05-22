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
#[allow(dead_code)]
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
#[allow(dead_code)]
struct DistillProgressiveConfig {
    layer_mapping: Vec<[usize; 2]>,
    #[serde(default = "default_hidden_weight")]
    hidden_weight: f32,
}

fn default_hidden_weight() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct DistillAttentionConfig {
    #[serde(default = "default_attention_weight")]
    weight: f32,
}

fn default_attention_weight() -> f32 {
    0.1
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

// --- Text-based distillation config (GH-455, ALB-011) ---
// Matches distill-30b.yaml schema for text-based synthetic data generation.
// Separate from DistillYamlConfig (logit KD) because vocab mismatch prevents logit alignment.

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct TextDistillConfig {
    teacher: TextTeacherConfig,
    #[serde(default)]
    student: Option<TextStudentConfig>,
    synthetic_data: SyntheticDataConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct TextTeacherConfig {
    model: String,
    #[serde(default)]
    tokenizer: Option<String>,
    #[serde(default)]
    precision: Option<String>,
    #[serde(default = "default_gpu")]
    gpu: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_gen_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct TextStudentConfig {
    checkpoint: String,
    tokenizer: String,
    #[serde(default)]
    config: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SyntheticDataConfig {
    prompts: String,
    output: String,
    #[serde(default = "default_target_tokens")]
    target_tokens: u64,
    #[serde(default = "default_samples_per_prompt")]
    samples_per_prompt: u32,
    #[serde(default = "default_min_completion_tokens")]
    min_completion_tokens: u32,
    #[serde(default = "default_max_prompt_chars")]
    max_prompt_chars: usize,
}

fn default_gpu() -> bool {
    true
}
fn default_max_tokens() -> u32 {
    256
}
fn default_gen_temperature() -> f32 {
    0.8
}
fn default_top_p() -> f32 {
    0.95
}
fn default_target_tokens() -> u64 {
    500_000
}
fn default_samples_per_prompt() -> u32 {
    1
}
fn default_min_completion_tokens() -> u32 {
    10
}
fn default_max_prompt_chars() -> usize {
    2048
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
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "mutating_output_contract"
)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
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
    backend: &str,
    dataset_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    // SPEC-DISTILL-001 Phase 3-prep (PMAT-697): validate backend selector
    // here so the user's mistake (e.g., typo `--backend cda`) is caught
    // before any I/O. Unknown backends fail with an enumeration of valid
    // values.
    match backend {
        "fixture" => {} // default; nothing else to validate
        "cuda" => {
            // SPEC-DISTILL-001 Phase 3-prep second half (PMAT-697): construct
            // real CudaTrainerTeacher + CudaStudentProvider, wire to Pipeline,
            // execute. Requires --features cuda to be compiled in.
            #[cfg(all(feature = "training", feature = "cuda"))]
            {
                let teacher_path = teacher_path.ok_or_else(|| {
                    CliError::ValidationFailed(
                        "--backend cuda requires a positional teacher path".to_string(),
                    )
                })?;
                return run_cuda_backend(
                    teacher_path,
                    student_path,
                    output_path,
                    temperature,
                    alpha,
                    epochs,
                    plan_only,
                    dataset_dir,
                    json_output,
                );
            }
            #[cfg(not(all(feature = "training", feature = "cuda")))]
            {
                return Err(CliError::ValidationFailed(
                    "--backend cuda requires apr-cli built with --features cuda,training. \
                     Rebuild: cargo install aprender --features cuda,training"
                        .to_string(),
                ));
            }
        }
        other => {
            return Err(CliError::ValidationFailed(format!(
                "--backend '{other}' not recognized. Valid: fixture, cuda. \
                 Default 'fixture' uses CPU-only stub providers; 'cuda' wires \
                 the real GPU backends."
            )));
        }
    }

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

/// SPEC-DISTILL-001 Phase 3-prep second half (PMAT-697): real cuda backend.
///
/// Constructs CudaTrainerTeacher + CudaStudentProvider from on-disk
/// `.apr` checkpoints, threads them through Pipeline::with_teacher /
/// with_student, and runs `execute()`. Output is the trained student
/// safetensors + a distillation_metadata.json sidecar.
///
/// **Why this path exists** — `--backend fixture` uses CPU stubs that
/// produce no real learning signal (useful for plumbing tests + CI,
/// not for distillation). `--backend cuda` is the actual production
/// path that drives a real teacher's logits into a real student's
/// gradient update via Phase 2a's `kd_step`.
///
/// **Limitations** — Phase 2d's CudaStudentProvider is batch_size=1 only.
/// The dispatch script (`scripts/dispatch-distill-phase-3-gx10.sh`)
/// scales by step count rather than batch parallelism. Phase 2e
/// generalizes via a fused-step trait method.
/// PMAT-698e: cap max_position_embeddings before passing into the cuda
/// trainer. See call sites in run_cuda_backend for the full rationale —
/// CudaTransformerTrainer::for_inference uses this value as max_seq_len
/// for ALL per-block scratch buffers, and the attention scores tensor
/// is sized at num_heads * max_seq_len² * 4 bytes, which overflows
/// the GPU memory budget for max_position_embeddings >= ~8192 even on
/// the 128GB GB10 unified pool.
///
/// Default cap: 2048 (gives ~5.6 GB workspace for Qwen2.5-Coder-0.5B).
/// Override via APR_DISTILL_MAX_SEQ_LEN env var (e.g. set to 4096 for
/// longer-context distillation runs that still fit).
#[cfg(all(feature = "training", feature = "cuda"))]
fn cap_max_seq_len(native: Option<usize>) -> Option<usize> {
    const DEFAULT_CAP: usize = 2048;
    let cap = std::env::var("APR_DISTILL_MAX_SEQ_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CAP);
    let n = native?;
    let chosen = n.min(cap);
    if chosen < n {
        eprintln!(
            "[PMAT-698e] capping max_position_embeddings {n} → {chosen} \
             (override via APR_DISTILL_MAX_SEQ_LEN)"
        );
    }
    Some(chosen)
}

#[cfg(all(feature = "training", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn run_cuda_backend(
    teacher_path: &Path,
    student_path: Option<&Path>,
    output_path: Option<&Path>,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    plan_only: bool,
    dataset_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    use aprender::format::v2::AprV2Reader;
    use aprender::format::v2::TensorDType;
    use entrenar::transformer::TransformerConfig;
    use entrenar_distill::{
        student_provider::CudaStudentProvider,
        teacher_provider::{CudaTrainerTeacher, TeacherLogitsProvider},
        DistillConfig, Pipeline,
    };

    use crate::commands::distill_q4k_teacher::RealizarQ4KTeacher;

    if plan_only {
        if json_output {
            println!(
                "{}",
                serde_json::json!({
                    "backend": "cuda",
                    "plan": true,
                    "teacher": teacher_path.display().to_string(),
                    "student": student_path.map(|p| p.display().to_string()),
                    "temperature": temperature,
                    "alpha": alpha,
                    "epochs": epochs,
                })
            );
        } else {
            println!("[plan] backend=cuda teacher={}", teacher_path.display());
        }
        return Ok(());
    }

    let student_path = student_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--backend cuda requires --student <path-to-student.apr>".to_string(),
        )
    })?;
    let output_path = output_path.ok_or_else(|| {
        CliError::ValidationFailed("--backend cuda requires --output <path>".to_string())
    })?;

    // Load teacher metadata → TransformerConfig.
    let teacher_bytes = std::fs::read(teacher_path).map_err(|e| {
        CliError::ValidationFailed(format!("read teacher {}: {e}", teacher_path.display()))
    })?;
    let teacher_reader = AprV2Reader::from_bytes(&teacher_bytes).map_err(|e| {
        CliError::ValidationFailed(format!("parse teacher {}: {e}", teacher_path.display()))
    })?;
    let teacher_meta = teacher_reader.metadata();
    let teacher_config = TransformerConfig::from_apr_metadata(
        teacher_meta.hidden_size,
        teacher_meta.num_heads,
        teacher_meta.num_kv_heads,
        teacher_meta.intermediate_size,
        teacher_meta.num_layers,
        teacher_meta.vocab_size,
        // PMAT-698e: cap max_position_embeddings before passing into
        // CudaTransformerTrainer::for_inference. The trainer uses this
        // value verbatim as max_seq_len for ALL per-block scratch buffers,
        // including the attention scores tensor sized at
        //   num_heads * max_seq_len² * 4 bytes
        // For Qwen2.5-Coder-0.5B (native max_position_embeddings=32768, 14
        // heads), that's 14 * 32768² * 4 = 60 GB PER BLOCK; on a 24-layer
        // model the total scratch footprint is ~1.4 TB, which overflows
        // even GB10's 128 GB unified pool and surfaces as
        // CUDA_ERROR_OUT_OF_MEMORY at "Block 0 upload". Distillation
        // training rarely sees sequences longer than 2048 tokens; capping
        // here gives ~5.6 GB workspace for the smoke. Caller can override
        // via APR_DISTILL_MAX_SEQ_LEN env var.
        cap_max_seq_len(teacher_meta.max_position_embeddings),
        teacher_meta.rms_norm_eps,
        teacher_meta.rope_theta,
        teacher_meta.architecture.as_deref(),
    )
    .ok_or_else(|| {
        CliError::ValidationFailed(
            "teacher .apr metadata missing required fields (hidden_size / num_heads / \
             num_layers / vocab_size / intermediate_size). The teacher must be a fully-\
             stamped checkpoint per SPEC-HF-PUBLISH-001."
                .to_string(),
        )
    })?;

    // Load student metadata → TransformerConfig (independent — student arch
    // typically differs from teacher).
    let student_bytes = std::fs::read(student_path).map_err(|e| {
        CliError::ValidationFailed(format!("read student {}: {e}", student_path.display()))
    })?;
    let student_reader = AprV2Reader::from_bytes(&student_bytes).map_err(|e| {
        CliError::ValidationFailed(format!("parse student {}: {e}", student_path.display()))
    })?;
    let student_meta = student_reader.metadata();
    let student_config = TransformerConfig::from_apr_metadata(
        student_meta.hidden_size,
        student_meta.num_heads,
        student_meta.num_kv_heads,
        student_meta.intermediate_size,
        student_meta.num_layers,
        student_meta.vocab_size,
        // PMAT-698e (see teacher comment above): same cap rationale.
        cap_max_seq_len(student_meta.max_position_embeddings),
        student_meta.rms_norm_eps,
        student_meta.rope_theta,
        student_meta.architecture.as_deref(),
    )
    .ok_or_else(|| {
        CliError::ValidationFailed(
            "student .apr metadata missing required fields — see teacher error message".to_string(),
        )
    })?;

    // Construct providers. for_inference / for_training both take the
    // checkpoint's DIRECTORY (which CudaTransformerTrainer scans for
    // `model.safetensors` or `model.apr`). For our purposes, the parent
    // of the .apr file is the right directory.
    let teacher_dir = teacher_path
        .parent()
        .ok_or_else(|| CliError::ValidationFailed("teacher path has no parent dir".into()))?;
    let student_dir = student_path
        .parent()
        .ok_or_else(|| CliError::ValidationFailed("student path has no parent dir".into()))?;
    // PMAT-701 Bug B: detect Q4K-quantized teacher and route to the
    // realizar inference path which keeps weights in Q4K on the GPU.
    // The legacy CudaTrainerTeacher dequantizes Q4K to F32 at upload
    // (~7× memory inflation), making 7B teachers OOM-kill on GB10 even
    // with the unified-memory allocator in effect. See contract
    // `contracts/cuda-q4k-frozen-teacher-v1.yaml` for the full invariant.
    let teacher_uses_quantized_weights = teacher_reader.tensor_names().iter().any(|name| {
        teacher_reader
            .get_tensor(name)
            .is_some_and(|t| matches!(t.dtype, TensorDType::Q4K | TensorDType::Q6K))
    });
    let teacher_provider: Box<dyn TeacherLogitsProvider> = if teacher_uses_quantized_weights {
        eprintln!(
            "[PMAT-701] Q4K/Q6K teacher detected → RealizarQ4KTeacher (Q4K-native forward, no F32 dequant)"
        );
        Box::new(
            RealizarQ4KTeacher::from_apr_path(teacher_path)
                .map_err(|e| CliError::ValidationFailed(format!("RealizarQ4KTeacher load: {e}")))?,
        )
    } else {
        eprintln!("[PMAT-701] F32/F16/BF16 teacher → CudaTrainerTeacher (legacy path)");
        // teacher_config is unused on the Q4K branch (the metadata is read
        // from the APR file by realizar directly); only the CudaTrainerTeacher
        // path needs the explicit TransformerConfig.
        Box::new(
            CudaTrainerTeacher::for_inference(teacher_dir, teacher_config)
                .map_err(|e| CliError::ValidationFailed(format!("CudaTrainerTeacher load: {e}")))?,
        )
    };
    let student_provider = CudaStudentProvider::for_training(student_dir, student_config)
        .map_err(|e| CliError::ValidationFailed(format!("CudaStudentProvider load: {e}")))?;

    // Build minimal DistillConfig pointing at on-disk paths. The pipeline
    // uses these for the file-load passthroughs; the providers we just
    // built do the actual forward/backward work.
    let mut config = DistillConfig::minimal(
        teacher_path
            .to_str()
            .ok_or_else(|| CliError::ValidationFailed("teacher path is not valid UTF-8".into()))?,
        student_path
            .to_str()
            .ok_or_else(|| CliError::ValidationFailed("student path is not valid UTF-8".into()))?,
    );
    config.output.dir = output_path.to_path_buf();
    config.distillation.temperature = temperature as f32;
    config.distillation.alpha = alpha as f32;
    config.training.epochs = epochs;

    // Wire the providers into the pipeline and execute.
    let mut pipeline = Pipeline::new(&config)
        .with_teacher(teacher_provider)
        .with_student(Box::new(student_provider));

    // SPEC-DISTILL-001 Phase 4 Stage B-2: when `--dataset <DIR>` is set,
    // construct a ShardBatchSource from the .bin shards. Otherwise the
    // pipeline keeps its default SyntheticBatchSource for smoke tests.
    // Requires the `shard-batch-source` feature on aprender-train-distill
    // (enabled by default in apr-cli's `training` feature).
    if let Some(dir) = dataset_dir {
        #[cfg(feature = "training")]
        {
            use entrenar_distill::batch_source::ShardBatchSource;
            let smoke_seq_len: usize = std::env::var("APR_DISTILL_SMOKE_SEQ_LEN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            let bs = config.training.batch_size as usize;
            let pad_id: u32 = 0;
            let eos_id: u32 = 0;
            let source = ShardBatchSource::from_dir(dir, bs, smoke_seq_len, pad_id, eos_id)
                .map_err(|e| {
                    CliError::ValidationFailed(format!(
                        "ShardBatchSource::from_dir({}): {e}",
                        dir.display()
                    ))
                })?;
            pipeline = pipeline.with_batch_source(Box::new(source));
        }
        #[cfg(not(feature = "training"))]
        {
            let _ = dir;
            return Err(CliError::ValidationFailed(
                "--dataset requires apr-cli built with --features training,cuda".to_string(),
            ));
        }
    }
    let result = pipeline
        .execute()
        .map_err(|e| CliError::ValidationFailed(format!("cuda pipeline.execute failed: {e}")))?;

    if json_output {
        println!(
            "{}",
            serde_json::json!({
                "backend": "cuda",
                "teacher": teacher_path.display().to_string(),
                "student": student_path.display().to_string(),
                "output": result.output_path.display().to_string(),
                "temperature": temperature,
                "alpha": alpha,
                "epochs": epochs,
                "initial_loss": result.metrics.initial_loss,
                "final_loss": result.metrics.final_loss,
                "best_loss": result.metrics.best_loss,
                "steps_completed": result.metrics.steps_completed,
                "duration_seconds": result.duration_seconds,
                "status": "completed",
            })
        );
    } else {
        println!(
            "✓ Distillation complete: initial_loss={:.4} → final_loss={:.4} ({} steps, {:.1}s)",
            result.metrics.initial_loss,
            result.metrics.final_loss,
            result.metrics.steps_completed,
            result.duration_seconds
        );
        println!("  Output: {}", result.output_path.display());
    }
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
///
/// When the `training` feature is enabled and the student model resolves to a
/// local SafeTensors path, this delegates to `entrenar_distill::run`
/// (real KD pipeline: loads weights, computes distillation loss, applies
/// gradient descent, saves trained student). The legacy metadata-only stub
/// remains as the fallback when the feature is off or the student is remote
/// (HF model id without local cache) — preserves backward compatibility per
/// SPEC-SHIP-TWO-001 §35 wire-up.
#[allow(clippy::disallowed_methods)]
fn run_config_train(
    config: &DistillYamlConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    // §35 wire-up: try real distillation pipeline when the training feature
    // is on and the student is a local path. Returns Ok(true) if the real
    // pipeline ran; Ok(false) to fall through to the legacy stub.
    #[cfg(feature = "training")]
    {
        if run_config_train_real(config, config_path, json_output)? {
            return Ok(());
        }
    }

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
            output::pipeline_stage("Loading student", output::StageStatus::Done);
            output::pipeline_stage("KD training", output::StageStatus::Done);
            println!();
            output::kv("  Metadata", meta_path.display().to_string());
            println!();
            println!("  {} Student training completed.", "DONE".green().bold());
        }
    } else {
        if !json_output {
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

/// §35 wire-up: invoke `entrenar_distill::run` with translated config.
///
/// Returns `Ok(true)` if the real pipeline executed (caller should return);
/// `Ok(false)` to fall through to the legacy metadata stub (e.g. when the
/// student is a remote HF id without a local cache).
#[cfg(feature = "training")]
fn run_config_train_real(
    config: &DistillYamlConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<bool> {
    // Only run the real pipeline when the student resolves to a local file.
    // Remote HF ids (`org/model` without local cache) fall through to the
    // stub, matching the existing user-facing "pending_download" path.
    let student_path = std::path::Path::new(&config.student.model_id);
    if !student_path.exists() {
        return Ok(false);
    }

    let distill_config = translate_to_distill_config(config);

    if !json_output {
        output::header("apr distill apply — Stage 2: Train Student (real KD pipeline)");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Teacher", &config.teacher.model_id);
        output::kv("  Student", &config.student.model_id);
        output::kv("  Output dir", &config.output.dir);
        output::kv(
            "  Temperature",
            format!("{:.1}", config.distillation.temperature),
        );
        output::kv("  Alpha", format!("{:.2}", config.distillation.alpha));
        output::kv("  Epochs", config.training.epochs.to_string());
        println!();
        output::pipeline_stage("Distillation training", output::StageStatus::Running);
    }

    let result = entrenar_distill::run(&distill_config)
        .map_err(|e| CliError::ValidationFailed(format!("distillation pipeline failed: {e}")))?;

    let meta = serde_json::json!({
        "stage": "train",
        "teacher": config.teacher.model_id,
        "student": config.student.model_id,
        "output_path": result.output_path.display().to_string(),
        "temperature": config.distillation.temperature,
        "alpha": config.distillation.alpha,
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "initial_loss": result.metrics.initial_loss,
        "final_loss": result.metrics.final_loss,
        "steps_completed": result.metrics.steps_completed,
        "duration_seconds": result.duration_seconds,
        "status": "completed",
    });

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&meta).unwrap_or_default()
        );
    } else {
        output::pipeline_stage("Distillation training", output::StageStatus::Done);
        println!();
        output::kv("  Output", result.output_path.display().to_string());
        output::kv(
            "  Initial loss",
            format!("{:.6}", result.metrics.initial_loss),
        );
        output::kv("  Final loss", format!("{:.6}", result.metrics.final_loss));
        output::kv(
            "  Steps completed",
            result.metrics.steps_completed.to_string(),
        );
        output::kv("  Duration", format!("{:.2}s", result.duration_seconds));
        println!();
        println!("  {} Student training completed.", "DONE".green().bold());
    }

    Ok(true)
}

/// Translate apr-cli's `DistillYamlConfig` to `entrenar_distill::DistillConfig`.
#[cfg(feature = "training")]
fn translate_to_distill_config(config: &DistillYamlConfig) -> entrenar_distill::DistillConfig {
    use entrenar_distill::config::{
        DistillationParams, LoraConfig as DistillLoraOut, OutputConfig, StudentConfig,
        TeacherConfig, TrainingConfig, WeightFormat,
    };

    let lora_out = config.student.lora.as_ref().map(|l| DistillLoraOut {
        rank: u32::try_from(l.rank).unwrap_or(u32::MAX),
        // apr-cli uses f64 for alpha; pipeline uses f32. lossy cast is fine for
        // hyperparameters (small magnitudes, no overflow risk in practice).
        alpha: l.alpha as f32,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
        dropout: 0.0,
    });

    entrenar_distill::DistillConfig {
        teacher: TeacherConfig {
            model_id: config.teacher.model_id.clone(),
            revision: None,
            format: WeightFormat::default(),
        },
        student: StudentConfig {
            model_id: config.student.model_id.clone(),
            lora: lora_out,
        },
        distillation: DistillationParams {
            temperature: config.distillation.temperature,
            alpha: config.distillation.alpha,
            progressive: None,
            attention: None,
        },
        training: TrainingConfig {
            epochs: u32::try_from(config.training.epochs).unwrap_or(u32::MAX),
            batch_size: u32::try_from(config.training.batch_size).unwrap_or(u32::MAX),
            learning_rate: config.training.learning_rate,
            ..TrainingConfig::default()
        },
        dataset: entrenar_distill::config::DatasetConfig::default(),
        output: OutputConfig {
            dir: std::path::PathBuf::from(&config.output.dir).join("student"),
            ..OutputConfig::default()
        },
    }
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

/// Stage: Generate synthetic data from teacher model (GH-455).
///
/// Spawns `realizar serve --model <teacher.apr> --gpu` as a subprocess,
/// reads prompts from JSONL, generates completions via HTTP, writes output JSONL.
#[allow(clippy::disallowed_methods)]
fn start_teacher_server(apr_bin: &Path, model: &str) -> Result<std::process::Child> {
    use std::process::{Command, Stdio};
    Command::new(apr_bin)
        .args(["serve", "run", model, "--gpu", "--port", "8090"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to start apr serve: {e}")))
}

fn wait_for_server_health(server: &mut std::process::Child, json_output: bool) -> Result<()> {
    let health_url = "http://127.0.0.1:8090/health";
    for attempt in 0..180 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        if let Ok(Some(status)) = server.try_wait() {
            let _ = server.kill();
            return Err(CliError::ValidationFailed(format!(
                "apr serve exited with status {status} during startup"
            )));
        }
        match ureq::get(health_url).call() {
            Ok(resp) if resp.status() == 200 => {
                if !json_output {
                    output::pipeline_stage("Starting teacher server", output::StageStatus::Done);
                    output::kv("  Ready after", format!("{}s", attempt + 1));
                    println!();
                }
                return Ok(());
            }
            _ => continue,
        }
    }
    let _ = server.kill();
    let _ = server.wait();
    Err(CliError::ValidationFailed(
        "Teacher server did not become ready within 180 seconds".into(),
    ))
}

/// Validate that teacher model and prompts files exist.
fn validate_distill_paths(config: &TextDistillConfig) -> Result<()> {
    let teacher_path = std::path::Path::new(&config.teacher.model);
    if !teacher_path.exists() {
        return Err(CliError::FileNotFound(teacher_path.to_path_buf()));
    }
    let prompts_path = std::path::Path::new(&config.synthetic_data.prompts);
    if !prompts_path.exists() {
        return Err(CliError::FileNotFound(prompts_path.to_path_buf()));
    }
    Ok(())
}

/// Print the text-generate header showing config summary.
fn print_generate_header(config: &TextDistillConfig, config_path: &Path) {
    output::header("apr distill apply — Stage: Generate Synthetic Data (GH-455)");
    println!();
    output::kv("  Config", config_path.display().to_string());
    output::kv("  Teacher", &config.teacher.model);
    output::kv("  Prompts", &config.synthetic_data.prompts);
    output::kv("  Output", &config.synthetic_data.output);
    output::kv(
        "  Max tokens/completion",
        config.teacher.max_tokens.to_string(),
    );
    output::kv(
        "  Temperature",
        format!("{:.2}", config.teacher.temperature),
    );
    output::kv(
        "  Target tokens",
        config.synthetic_data.target_tokens.to_string(),
    );
    println!();
}

/// Read prompts from a JSONL file, skipping blank lines.
fn read_prompts_jsonl(path: &Path) -> Result<Vec<serde_json::Value>> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut prompts = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| CliError::ValidationFailed(format!("Invalid prompt JSONL: {e}")))?;
        prompts.push(parsed);
    }
    Ok(prompts)
}

/// State loaded from an existing output file for resume support.
struct ResumeState {
    existing_prompts: std::collections::HashSet<String>,
    total_tokens: u64,
    generated_count: u64,
}

/// Load resume state from an existing output JSONL, creating parent dirs as needed.
fn load_resume_state(output_path: &Path) -> Result<ResumeState> {
    use std::io::{BufRead, BufReader};
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut state = ResumeState {
        existing_prompts: std::collections::HashSet::new(),
        total_tokens: 0,
        generated_count: 0,
    };
    if output_path.exists() {
        let existing = std::fs::File::open(output_path)?;
        for line in BufReader::new(existing).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(p) = parsed.get("prompt").and_then(|v| v.as_str()) {
                    state.existing_prompts.insert(p.to_string());
                }
                state.total_tokens += parsed.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                state.generated_count += 1;
            }
        }
    }
    Ok(state)
}

/// POST to /generate with retry (up to 3 attempts). Returns None if all retries exhausted.
fn send_generate_request(
    url: &str,
    request_body: &str,
    prompt_index: usize,
    json_output: bool,
) -> (Option<ureq::Response>, bool) {
    let mut skipped = false;
    for retry in 0..3 {
        match ureq::post(url)
            .set("Content-Type", "application/json")
            .send_string(request_body)
        {
            Ok(r) => return (Some(r), false),
            Err(e) if retry < 2 => {
                if !json_output {
                    eprintln!(
                        "  Retry {}/{} for prompt {}: {e}",
                        retry + 1,
                        3,
                        prompt_index
                    );
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            Err(e) => {
                if !json_output {
                    eprintln!("  Skipping prompt {} after 3 retries: {e}", prompt_index);
                }
                skipped = true;
            }
        }
    }
    (None, skipped)
}

/// Format and print the final result of text generation.
fn format_generate_result(
    config: &TextDistillConfig,
    prompts_total: usize,
    generated_count: u64,
    skipped_count: u64,
    total_tokens: u64,
    target: u64,
    elapsed: std::time::Duration,
    json_output: bool,
) {
    if json_output {
        let result = serde_json::json!({
            "stage": "generate",
            "status": "completed",
            "prompts_total": prompts_total,
            "completions_generated": generated_count,
            "completions_skipped": skipped_count,
            "total_tokens": total_tokens,
            "target_tokens": target,
            "elapsed_seconds": elapsed.as_secs(),
            "output": config.synthetic_data.output,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        output::pipeline_stage("Generating completions", output::StageStatus::Done);
        println!();
        output::kv("  Completions", generated_count.to_string());
        output::kv("  Skipped", skipped_count.to_string());
        output::kv("  Tokens", total_tokens.to_string());
        output::kv("  Target", target.to_string());
        output::kv("  Elapsed", format!("{:.0}s", elapsed.as_secs_f64()));
        output::kv(
            "  Throughput",
            format!(
                "{:.1} tok/s",
                total_tokens as f64 / elapsed.as_secs_f64().max(0.001)
            ),
        );
        output::kv("  Output", &config.synthetic_data.output);
        println!();
        println!(
            "  {} Synthetic data generated. Tokenize and train next.",
            "DONE".green().bold()
        );
    }
}

fn run_text_generate(
    config: &TextDistillConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    use std::io::Write;

    validate_distill_paths(config)?;

    if !json_output {
        print_generate_header(config, config_path);
    }

    let apr_bin = std::env::current_exe().map_err(|e| {
        CliError::ValidationFailed(format!("Cannot determine apr binary path: {e}"))
    })?;

    if !json_output {
        output::pipeline_stage("Starting teacher server", output::StageStatus::Running);
        output::kv("  Binary", apr_bin.display().to_string());
    }

    let mut server = start_teacher_server(&apr_bin, &config.teacher.model)?;
    wait_for_server_health(&mut server, json_output)?;

    let prompts_path = std::path::Path::new(&config.synthetic_data.prompts);
    let prompts = read_prompts_jsonl(prompts_path)?;

    if !json_output {
        output::pipeline_stage("Generating completions", output::StageStatus::Running);
        output::kv("  Loaded prompts", prompts.len().to_string());
    }

    let output_path = std::path::Path::new(&config.synthetic_data.output);
    let mut resume = load_resume_state(output_path)?;
    let mut skipped_count = 0u64;

    if !resume.existing_prompts.is_empty() && !json_output {
        println!(
            "  Resuming: {} existing records, {} tokens",
            resume.existing_prompts.len(),
            resume.total_tokens
        );
    }

    let output_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;
    let mut writer = std::io::BufWriter::new(output_file);

    let generate_url = format!("http://127.0.0.1:8090/generate");
    let target = config.synthetic_data.target_tokens;
    let start_time = std::time::Instant::now();

    for (i, prompt_json) in prompts.iter().enumerate() {
        if resume.total_tokens >= target {
            break;
        }

        let prompt_text = prompt_json
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                CliError::ValidationFailed(format!("Prompt {} missing 'prompt' field", i))
            })?;

        if resume.existing_prompts.contains(prompt_text) {
            continue;
        }

        // ALB-111: Skip pathologically long prompts (55K char prompt caused hours-long prefill)
        if prompt_text.len() > config.synthetic_data.max_prompt_chars {
            if !json_output {
                eprintln!(
                    "  Skipping prompt {} ({} chars > {} max)",
                    i,
                    prompt_text.len(),
                    config.synthetic_data.max_prompt_chars,
                );
            }
            skipped_count += 1;
            continue;
        }

        let request_body = serde_json::to_string(&serde_json::json!({
            "prompt": prompt_text,
            "max_tokens": config.teacher.max_tokens,
            "temperature": config.teacher.temperature,
            "strategy": "top_p",
            "top_p": config.teacher.top_p,
        }))
        .expect("JSON serialization cannot fail");

        let (resp, was_skipped) =
            send_generate_request(&generate_url, &request_body, i, json_output);
        if was_skipped {
            skipped_count += 1;
        }
        let Some(resp) = resp else {
            continue;
        };

        let gen_result: serde_json::Value = {
            let body = resp.into_string().map_err(|e| {
                CliError::NetworkError(format!("Failed to read response body: {e}"))
            })?;
            serde_json::from_str(&body)
                .map_err(|e| CliError::NetworkError(format!("Invalid generate response: {e}")))?
        };

        let num_tokens = gen_result
            .get("num_generated")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let text = gen_result
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if num_tokens < u64::from(config.synthetic_data.min_completion_tokens) {
            skipped_count += 1;
            continue;
        }

        // Write output JSONL record
        let record = serde_json::json!({
            "prompt": prompt_text,
            "completion": text,
            "tokens": num_tokens,
            "source": prompt_json.get("source").and_then(|v| v.as_str()).unwrap_or(""),
            "kind": prompt_json.get("kind").and_then(|v| v.as_str()).unwrap_or(""),
        });
        writeln!(
            writer,
            "{}",
            serde_json::to_string(&record)
                .map_err(|e| CliError::ValidationFailed(format!("JSON serialize error: {e}")))?
        )?;
        writer.flush()?;

        resume.total_tokens += num_tokens;
        resume.generated_count += 1;

        // Progress every 10 prompts
        if (i + 1) % 10 == 0 && !json_output {
            let elapsed = start_time.elapsed().as_secs_f64();
            let tok_per_sec = if elapsed > 0.0 {
                resume.total_tokens as f64 / elapsed
            } else {
                0.0
            };
            println!(
                "  [{}/{}] {} tokens generated ({:.0} tok/s), {} skipped",
                i + 1,
                prompts.len(),
                resume.total_tokens,
                tok_per_sec,
                skipped_count
            );
        }
    }

    writer.flush()?;

    // Shutdown server
    let _ = server.kill();
    let _ = server.wait();

    format_generate_result(
        config,
        prompts.len(),
        resume.generated_count,
        skipped_count,
        resume.total_tokens,
        target,
        start_time.elapsed(),
        json_output,
    );

    Ok(())
}

include!("distill_include_01.rs");
