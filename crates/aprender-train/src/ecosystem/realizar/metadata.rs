//! GGUF metadata types.

use super::provenance::ExperimentProvenance;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GGUF metadata container.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GgufMetadata {
    /// General model information
    pub general: GeneralMetadata,
    /// Experiment provenance (optional)
    pub provenance: Option<ExperimentProvenance>,
    /// Custom key-value pairs
    pub custom: HashMap<String, String>,
}

/// General model metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneralMetadata {
    /// Model architecture (e.g., "llama", "mistral", "qwen")
    pub architecture: String,
    /// Model name
    pub name: String,
    /// Author or organization
    pub author: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// License identifier
    pub license: Option<String>,
    /// URL for more information
    pub url: Option<String>,
    /// File type (quantization level)
    pub file_type: Option<String>,
}

impl GeneralMetadata {
    /// Create new general metadata.
    pub fn new(architecture: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            architecture: architecture.into(),
            name: name.into(),
            author: None,
            description: None,
            license: None,
            url: None,
            file_type: None,
        }
    }

    /// Set author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set license.
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }
}
