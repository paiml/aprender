//! PEFT-compatible adapter_config.json generation
//!
//! Generates adapter configuration files compatible with HuggingFace PEFT library,
//! enabling direct loading in `transformers` and `peft` Python packages.

use crate::lora::LoRAConfig;
use serde::{Deserialize, Serialize};

/// PEFT adapter configuration matching the HuggingFace PEFT schema
///
/// This struct serializes to `adapter_config.json` format that can be loaded by
/// `peft.PeftModel.from_pretrained()`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeftAdapterConfig {
    /// PEFT method type (always "LORA" for LoRA adapters)
    pub peft_type: String,
    /// LoRA rank
    pub r: usize,
    /// LoRA alpha scaling parameter
    pub lora_alpha: f32,
    /// Target module names for LoRA adaptation
    pub target_modules: Vec<String>,
    /// LoRA dropout rate (0.0 if not used)
    pub lora_dropout: f32,
    /// Bias handling: "none", "all", or "lora_only"
    pub bias: String,
    /// Base model name or path (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model_name_or_path: Option<String>,
    /// Task type (e.g., "CAUSAL_LM", "SEQ_CLS")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    /// Fan-in/fan-out setting
    #[serde(default)]
    pub fan_in_fan_out: bool,
    /// Inference mode
    #[serde(default)]
    pub inference_mode: bool,
}

impl PeftAdapterConfig {
    /// Convert from entrenar's LoRAConfig to PEFT-compatible config
    pub fn from_lora_config(config: &LoRAConfig, base_model: Option<&str>) -> Self {
        let mut target_modules: Vec<String> = config.target_modules.iter().cloned().collect();
        target_modules.sort();

        Self {
            peft_type: "LORA".to_string(),
            r: config.rank,
            lora_alpha: config.alpha,
            target_modules,
            lora_dropout: 0.0,
            bias: "none".to_string(),
            base_model_name_or_path: base_model.map(String::from),
            task_type: None,
            fan_in_fan_out: false,
            inference_mode: false,
        }
    }

    /// Set bias handling mode
    pub fn with_bias(mut self, bias: impl Into<String>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.lora_dropout = dropout;
        self
    }

    /// Set task type
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self
    }

    /// Set inference mode
    pub fn with_inference_mode(mut self, inference_mode: bool) -> Self {
        self.inference_mode = inference_mode;
        self
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_lora_config() -> LoRAConfig {
        LoRAConfig::new(16, 32.0).target_attention_projections()
    }

    #[test]
    fn test_from_lora_config() {
        let lora_config = make_test_lora_config();
        let peft = PeftAdapterConfig::from_lora_config(&lora_config, Some("meta-llama/Llama-2-7b"));

        assert_eq!(peft.peft_type, "LORA");
        assert_eq!(peft.r, 16);
        assert_eq!(peft.lora_alpha, 32.0);
        assert_eq!(peft.target_modules.len(), 4);
        assert!(peft.target_modules.contains(&"q_proj".to_string()));
        assert!(peft.target_modules.contains(&"k_proj".to_string()));
        assert!(peft.target_modules.contains(&"v_proj".to_string()));
        assert!(peft.target_modules.contains(&"o_proj".to_string()));
        assert_eq!(peft.bias, "none");
        assert_eq!(peft.base_model_name_or_path, Some("meta-llama/Llama-2-7b".to_string()));
    }

    #[test]
    fn test_from_lora_config_no_base_model() {
        let lora_config = LoRAConfig::new(8, 8.0).target_qv_projections();
        let peft = PeftAdapterConfig::from_lora_config(&lora_config, None);

        assert_eq!(peft.r, 8);
        assert!(peft.base_model_name_or_path.is_none());
        assert_eq!(peft.target_modules.len(), 2);
    }

    #[test]
    fn test_json_roundtrip() {
        let lora_config = make_test_lora_config();
        let peft = PeftAdapterConfig::from_lora_config(&lora_config, Some("test/model"));

        let json = peft.to_json().expect("operation should succeed");
        let deserialized =
            PeftAdapterConfig::from_json(&json).expect("deserialization should succeed");

        assert_eq!(peft, deserialized);
    }

    #[test]
    fn test_json_schema_keys() {
        let lora_config = make_test_lora_config();
        let peft = PeftAdapterConfig::from_lora_config(&lora_config, Some("test/model"));

        let json = peft.to_json().expect("operation should succeed");

        // Verify expected PEFT schema keys are present
        assert!(json.contains("\"peft_type\""));
        assert!(json.contains("\"r\""));
        assert!(json.contains("\"lora_alpha\""));
        assert!(json.contains("\"target_modules\""));
        assert!(json.contains("\"lora_dropout\""));
        assert!(json.contains("\"bias\""));
        assert!(json.contains("\"base_model_name_or_path\""));
    }

    #[test]
    fn test_json_no_base_model_omitted() {
        let peft = PeftAdapterConfig::from_lora_config(&LoRAConfig::new(4, 4.0), None);
        let json = peft.to_json().expect("operation should succeed");
        // base_model_name_or_path should be omitted when None
        assert!(!json.contains("base_model_name_or_path"));
    }

    #[test]
    fn test_builder_methods() {
        let config = LoRAConfig::new(8, 8.0).target_qv_projections();
        let peft = PeftAdapterConfig::from_lora_config(&config, None)
            .with_bias("lora_only")
            .with_dropout(0.1)
            .with_task_type("CAUSAL_LM")
            .with_inference_mode(true);

        assert_eq!(peft.bias, "lora_only");
        assert_eq!(peft.lora_dropout, 0.1);
        assert_eq!(peft.task_type, Some("CAUSAL_LM".to_string()));
        assert!(peft.inference_mode);
    }

    #[test]
    fn test_target_modules_sorted() {
        let config = LoRAConfig::new(8, 8.0).target_attention_projections();
        let peft = PeftAdapterConfig::from_lora_config(&config, None);

        // target_modules should be sorted for deterministic output
        let mut sorted = peft.target_modules.clone();
        sorted.sort();
        assert_eq!(peft.target_modules, sorted);
    }

    #[test]
    fn test_empty_target_modules() {
        let config = LoRAConfig::new(8, 8.0);
        let peft = PeftAdapterConfig::from_lora_config(&config, None);

        assert!(peft.target_modules.is_empty());
    }
}
