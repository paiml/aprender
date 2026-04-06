//! Distillation loss configuration

use serde::{Deserialize, Serialize};

/// Distillation loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DistillationConfig {
    /// Temperature for softening distributions
    pub temperature: f32,
    /// Alpha weight for soft vs hard loss
    pub alpha: f32,
    /// Progressive distillation config
    pub progressive: Option<ProgressiveConfig>,
    /// Attention transfer config
    pub attention_transfer: Option<AttentionTransferConfig>,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self { temperature: 4.0, alpha: 0.7, progressive: None, attention_transfer: None }
    }
}

/// Progressive distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Layer mapping [[student_layer, teacher_layer], ...]
    pub layer_mapping: Vec<[usize; 2]>,
    /// Weight for hidden state loss
    #[serde(default = "default_hidden_weight")]
    pub hidden_weight: f32,
}

fn default_hidden_weight() -> f32 {
    1.0
}

/// Attention transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionTransferConfig {
    /// Weight for attention transfer loss
    #[serde(default = "default_attention_weight")]
    pub weight: f32,
}

fn default_attention_weight() -> f32 {
    0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert!((config.temperature - 4.0).abs() < 1e-6);
        assert!((config.alpha - 0.7).abs() < 1e-6);
        assert!(config.progressive.is_none());
        assert!(config.attention_transfer.is_none());
    }

    #[test]
    fn test_distillation_config_custom() {
        let config = DistillationConfig {
            temperature: 2.0,
            alpha: 0.5,
            progressive: Some(ProgressiveConfig {
                layer_mapping: vec![[0, 0], [1, 2], [2, 4]],
                hidden_weight: 0.5,
            }),
            attention_transfer: Some(AttentionTransferConfig { weight: 0.2 }),
        };

        assert!((config.temperature - 2.0).abs() < 1e-6);
        assert!((config.alpha - 0.5).abs() < 1e-6);
        assert!(config.progressive.is_some());
        assert!(config.attention_transfer.is_some());
    }

    #[test]
    fn test_progressive_config_layer_mapping() {
        let config =
            ProgressiveConfig { layer_mapping: vec![[0, 0], [1, 2], [2, 4]], hidden_weight: 1.0 };

        assert_eq!(config.layer_mapping.len(), 3);
        assert_eq!(config.layer_mapping[0], [0, 0]);
        assert_eq!(config.layer_mapping[1], [1, 2]);
    }

    #[test]
    fn test_default_hidden_weight() {
        assert!((default_hidden_weight() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_attention_weight() {
        assert!((default_attention_weight() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_distillation_config_serde() {
        let config = DistillationConfig {
            temperature: 3.0,
            alpha: 0.6,
            progressive: None,
            attention_transfer: None,
        };

        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let deserialized: DistillationConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert!((config.temperature - deserialized.temperature).abs() < 1e-6);
        assert!((config.alpha - deserialized.alpha).abs() < 1e-6);
    }

    #[test]
    fn test_distillation_config_serde_with_optional() {
        let config = DistillationConfig {
            temperature: 3.0,
            alpha: 0.6,
            progressive: Some(ProgressiveConfig {
                layer_mapping: vec![[0, 1]],
                hidden_weight: 0.8,
            }),
            attention_transfer: Some(AttentionTransferConfig { weight: 0.15 }),
        };

        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let deserialized: DistillationConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert!(deserialized.progressive.is_some());
        let prog = deserialized.progressive.expect("deserialization should succeed");
        assert_eq!(prog.layer_mapping.len(), 1);
        assert!((prog.hidden_weight - 0.8).abs() < 1e-6);

        assert!(deserialized.attention_transfer.is_some());
        let attn = deserialized.attention_transfer.expect("deserialization should succeed");
        assert!((attn.weight - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_distillation_config_from_partial_json() {
        // Test that defaults are used when fields are missing
        let json = r#"{"temperature": 5.0}"#;
        let config: DistillationConfig =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert!((config.temperature - 5.0).abs() < 1e-6);
        assert!((config.alpha - 0.7).abs() < 1e-6); // default
    }

    #[test]
    fn test_progressive_config_serde_default_weight() {
        // Test that hidden_weight defaults to 1.0 when not specified
        let json = r#"{"layer_mapping": [[0, 0]]}"#;
        let config: ProgressiveConfig =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert!((config.hidden_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_attention_transfer_config_serde_default_weight() {
        // Test that weight defaults to 0.1 when not specified
        let json = r"{}";
        let config: AttentionTransferConfig =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert!((config.weight - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_distillation_config_debug() {
        let config = DistillationConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("DistillationConfig"));
        assert!(debug_str.contains("temperature"));
    }

    #[test]
    fn test_progressive_config_clone() {
        let config = ProgressiveConfig { layer_mapping: vec![[0, 0], [1, 2]], hidden_weight: 0.5 };
        let cloned = config.clone();
        assert_eq!(config.layer_mapping, cloned.layer_mapping);
        assert!((config.hidden_weight - cloned.hidden_weight).abs() < 1e-6);
    }

    #[test]
    fn test_attention_transfer_config_clone() {
        let config = AttentionTransferConfig { weight: 0.2 };
        let cloned = config.clone();
        assert!((config.weight - cloned.weight).abs() < 1e-6);
    }
}
