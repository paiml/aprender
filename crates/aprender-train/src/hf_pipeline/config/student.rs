//! Student model configuration with LoRA/QLoRA

use crate::lora::LoRAConfig;
use serde::{Deserialize, Serialize};

use super::teacher::default_revision;

/// Student model configuration with LoRA/QLoRA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentConfig {
    /// Model ID (can be same as teacher or smaller model)
    pub model_id: String,
    /// Revision/branch
    #[serde(default = "default_revision")]
    pub revision: String,
    /// LoRA configuration (if None, full fine-tuning)
    pub lora: Option<LoRAYamlConfig>,
    /// Use 4-bit quantization (QLoRA)
    #[serde(default)]
    pub load_in_4bit: bool,
}

/// LoRA configuration in YAML format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAYamlConfig {
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha (scaling factor)
    pub alpha: f32,
    /// Target modules
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
    /// Target layers (optional)
    pub layers: Option<Vec<usize>>,
}

fn default_target_modules() -> Vec<String> {
    vec!["q_proj".to_string(), "v_proj".to_string()]
}

impl From<&LoRAYamlConfig> for LoRAConfig {
    fn from(yaml: &LoRAYamlConfig) -> Self {
        let mut config = LoRAConfig::new(yaml.rank, yaml.alpha);
        let modules: Vec<&str> = yaml.target_modules.iter().map(String::as_str).collect();
        config = config.target_modules(&modules);
        if let Some(ref layers) = yaml.layers {
            config = config.target_layers(layers);
        }
        config
    }
}
