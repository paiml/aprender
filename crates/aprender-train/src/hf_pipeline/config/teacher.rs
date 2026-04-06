//! Teacher model configuration

use serde::{Deserialize, Serialize};

/// Teacher model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherConfig {
    /// Model ID on HuggingFace
    pub model_id: String,
    /// Revision/branch (default: "main")
    #[serde(default = "default_revision")]
    pub revision: String,
    /// Use 8-bit quantization for teacher
    #[serde(default)]
    pub load_in_8bit: bool,
}

pub(crate) fn default_revision() -> String {
    "main".to_string()
}
