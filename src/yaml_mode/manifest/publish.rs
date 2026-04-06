//! Publish Configuration
//!
//! Defines the `publish:` section for auto-publishing trained models to HuggingFace Hub.

use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

fn default_format() -> String {
    "safetensors".to_string()
}

/// Publish configuration for auto-uploading to HuggingFace Hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishConfig {
    /// HuggingFace repo ID (e.g., myuser/my-model)
    pub repo: String,

    /// Make the repository private
    #[serde(default)]
    pub private: bool,

    /// Generate and upload a model card
    #[serde(default = "default_true")]
    pub model_card: bool,

    /// Merge LoRA adapters before publishing
    #[serde(default)]
    pub merge_adapters: bool,

    /// Export format (safetensors or gguf)
    #[serde(default = "default_format")]
    pub format: String,
}
