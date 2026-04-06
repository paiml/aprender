//! Distillation configuration parsing and management.
//!
//! Supports YAML configuration files for zero-code distillation setup.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use entrenar_common::{EntrenarError, Result};

/// Complete distillation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillConfig {
    /// Teacher model configuration
    pub teacher: TeacherConfig,
    /// Student model configuration
    pub student: StudentConfig,
    /// Distillation hyperparameters
    pub distillation: DistillationParams,
    /// Training configuration
    pub training: TrainingConfig,
    /// Dataset configuration
    #[serde(default)]
    pub dataset: DatasetConfig,
    /// Output configuration
    #[serde(default)]
    pub output: OutputConfig,
}

impl DistillConfig {
    /// Load configuration from a YAML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| EntrenarError::Io {
            context: format!("reading config file: {}", path.display()),
            source: e,
        })?;

        Self::from_yaml(&content, path)
    }

    /// Parse configuration from YAML string.
    pub fn from_yaml(content: &str, path: &Path) -> Result<Self> {
        serde_yaml::from_str(content).map_err(|e| EntrenarError::ConfigParsing {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
    }

    /// Create a minimal configuration for testing.
    pub fn minimal(teacher_id: &str, student_id: &str) -> Self {
        Self {
            teacher: TeacherConfig {
                model_id: teacher_id.to_string(),
                revision: None,
                format: WeightFormat::SafeTensors,
            },
            student: StudentConfig { model_id: student_id.to_string(), lora: None },
            distillation: DistillationParams::default(),
            training: TrainingConfig::default(),
            dataset: DatasetConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

/// Teacher model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherConfig {
    /// HuggingFace model ID or local path
    pub model_id: String,
    /// Git revision/branch/tag
    #[serde(default)]
    pub revision: Option<String>,
    /// Weight format preference
    #[serde(default)]
    pub format: WeightFormat,
}

/// Student model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentConfig {
    /// HuggingFace model ID or local path
    pub model_id: String,
    /// Optional LoRA configuration
    #[serde(default)]
    pub lora: Option<LoraConfig>,
}

/// LoRA adapter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// LoRA rank
    pub rank: u32,
    /// LoRA alpha scaling
    pub alpha: f32,
    /// Target modules to apply LoRA
    pub target_modules: Vec<String>,
    /// Dropout probability
    #[serde(default)]
    pub dropout: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 64,
            alpha: 16.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            dropout: 0.1,
        }
    }
}

/// Weight format preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightFormat {
    /// SafeTensors format (secure, recommended)
    #[default]
    SafeTensors,
    /// GGUF format (llama.cpp compatible)
    Gguf,
    /// APR format (JSON metadata)
    Apr,
}

/// Distillation hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationParams {
    /// Temperature for soft targets
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Balance between soft and hard targets (0-1)
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    /// Progressive distillation config
    #[serde(default)]
    pub progressive: Option<ProgressiveConfig>,
    /// Attention transfer config
    #[serde(default)]
    pub attention: Option<AttentionConfig>,
}

impl Default for DistillationParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            alpha: default_alpha(),
            progressive: None,
            attention: None,
        }
    }
}

fn default_temperature() -> f32 {
    4.0
}

fn default_alpha() -> f32 {
    0.7
}

/// Progressive distillation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Whether progressive distillation is enabled
    #[serde(default = "bool_true")]
    pub enabled: bool,
    /// Layer mapping: (student_layer, teacher_layer)
    pub layer_mapping: Vec<(usize, usize)>,
    /// Weight for progressive loss
    #[serde(default = "default_progressive_weight")]
    pub weight: f32,
}

fn default_progressive_weight() -> f32 {
    0.3
}

fn bool_true() -> bool {
    true
}

/// Attention transfer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Whether attention transfer is enabled
    #[serde(default = "bool_true")]
    pub enabled: bool,
    /// Weight for attention loss
    #[serde(default = "default_attention_weight")]
    pub weight: f32,
}

fn default_attention_weight() -> f32 {
    0.1
}

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    /// Batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    /// Learning rate
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    /// Warmup steps
    #[serde(default)]
    pub warmup_steps: u32,
    /// Gradient accumulation steps
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation: u32,
    /// Mixed precision mode
    #[serde(default)]
    pub mixed_precision: MixedPrecision,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            learning_rate: default_lr(),
            warmup_steps: 0,
            gradient_accumulation: default_grad_accum(),
            mixed_precision: MixedPrecision::default(),
        }
    }
}

fn default_epochs() -> u32 {
    10
}

fn default_batch_size() -> u32 {
    32
}

fn default_lr() -> f64 {
    1e-4
}

fn default_grad_accum() -> u32 {
    1
}

/// Mixed precision mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MixedPrecision {
    /// No mixed precision
    #[default]
    None,
    /// FP16 mixed precision
    Fp16,
    /// BF16 mixed precision
    Bf16,
}

/// Dataset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset source: "huggingface" or "local"
    #[serde(default = "default_source")]
    pub source: String,
    /// Dataset name (for HuggingFace) or path (for local)
    #[serde(default)]
    pub name: String,
    /// Dataset path (for local)
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Split to use
    #[serde(default = "default_split")]
    pub split: String,
    /// Maximum sequence length
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    /// Whether to use streaming
    #[serde(default)]
    pub streaming: bool,
}

fn default_source() -> String {
    "huggingface".to_string()
}

fn default_split() -> String {
    "train".to_string()
}

fn default_max_length() -> usize {
    512
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            source: default_source(),
            name: String::new(),
            path: None,
            split: default_split(),
            max_length: default_max_length(),
            streaming: false,
        }
    }
}

/// Output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory
    #[serde(default = "default_output_dir")]
    pub dir: PathBuf,
    /// Checkpoint frequency (steps)
    #[serde(default = "default_checkpoint_every")]
    pub checkpoint_every: u32,
    /// Export format
    #[serde(default)]
    pub format: WeightFormat,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: default_output_dir(),
            checkpoint_every: default_checkpoint_every(),
            format: WeightFormat::default(),
        }
    }
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("./distilled-model")
}

fn default_checkpoint_every() -> u32 {
    1000
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_YAML: &str = r#"
teacher:
  model_id: "meta-llama/Llama-2-7b"
  format: safetensors

student:
  model_id: "TinyLlama/TinyLlama-1.1B"
  lora:
    rank: 64
    alpha: 16
    target_modules: [q_proj, k_proj, v_proj, o_proj]

distillation:
  temperature: 4.0
  alpha: 0.7

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4

output:
  dir: "./output"
"#;

    #[test]
    fn test_parse_yaml_config() {
        let config: DistillConfig =
            serde_yaml::from_str(SAMPLE_YAML).expect("Failed to parse YAML");

        assert_eq!(config.teacher.model_id, "meta-llama/Llama-2-7b");
        assert_eq!(config.student.model_id, "TinyLlama/TinyLlama-1.1B");
        assert_eq!(config.distillation.temperature, 4.0);
        assert_eq!(config.distillation.alpha, 0.7);
        assert_eq!(config.training.epochs, 10);
    }

    #[test]
    fn test_default_values() {
        let minimal = DistillConfig::minimal("teacher", "student");

        assert_eq!(minimal.distillation.temperature, 4.0);
        assert_eq!(minimal.distillation.alpha, 0.7);
        assert_eq!(minimal.training.epochs, 10);
        assert_eq!(minimal.training.batch_size, 32);
    }

    #[test]
    fn test_lora_config_defaults() {
        let lora = LoraConfig::default();
        assert_eq!(lora.rank, 64);
        assert_eq!(lora.alpha, 16.0);
        assert!(!lora.target_modules.is_empty());
    }

    #[test]
    fn test_weight_format_serialization() {
        let json = serde_json::to_string(&WeightFormat::SafeTensors)
            .expect("JSON serialization should succeed");
        assert_eq!(json, "\"safetensors\"");

        let format: WeightFormat =
            serde_json::from_str("\"gguf\"").expect("JSON deserialization should succeed");
        assert_eq!(format, WeightFormat::Gguf);
    }

    #[test]
    fn test_progressive_config() {
        let yaml = r"
enabled: true
layer_mapping: [[0, 3], [1, 7], [2, 11]]
weight: 0.3
";
        let config: ProgressiveConfig = serde_yaml::from_str(yaml).expect("config should be valid");
        assert!(config.enabled);
        assert_eq!(config.layer_mapping.len(), 3);
        assert_eq!(config.layer_mapping[0], (0, 3));
    }
}
