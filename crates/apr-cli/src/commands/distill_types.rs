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
