//! Output configuration

use serde::{Deserialize, Serialize};

/// Default number of training steps between checkpoint saves.
const DEFAULT_SAVE_STEPS: usize = 500;

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Output directory
    pub dir: String,
    /// Save checkpoints every N steps
    pub save_steps: usize,
    /// Evaluate every N steps
    pub eval_steps: usize,
    /// Log every N steps
    pub log_steps: usize,
    /// Push to HuggingFace Hub
    pub push_to_hub: bool,
    /// Hub repository ID
    pub hub_repo_id: Option<String>,
    /// Generate model card (README.md) on publish
    pub generate_model_card: bool,
    /// License for Hub publication (SPDX identifier, e.g., "apache-2.0")
    pub hub_license: Option<String>,
    /// Tags for Hub discoverability
    pub hub_tags: Vec<String>,
    /// Whether the Hub repository should be private
    pub hub_private: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: "./output".to_string(),
            save_steps: DEFAULT_SAVE_STEPS,
            eval_steps: 100,
            log_steps: 10,
            push_to_hub: false,
            hub_repo_id: None,
            generate_model_card: true,
            hub_license: None,
            hub_tags: Vec::new(),
            hub_private: false,
        }
    }
}
