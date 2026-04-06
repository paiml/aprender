//! Core types for the model registry.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Model source type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelSource {
    /// HuggingFace Hub model
    HuggingFace {
        /// Repository ID (e.g., "bert-base-uncased")
        repo_id: String,
    },
    /// Local file path
    LocalFile {
        /// Path to the model file
        path: PathBuf,
    },
    /// Custom URL source
    Custom {
        /// Download URL
        url: String,
    },
}

impl ModelSource {
    /// Create HuggingFace source
    pub fn huggingface(repo_id: impl Into<String>) -> Self {
        Self::HuggingFace { repo_id: repo_id.into() }
    }

    /// Create local file source
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::LocalFile { path: path.into() }
    }

    /// Create custom URL source
    pub fn custom(url: impl Into<String>) -> Self {
        Self::Custom { url: url.into() }
    }

    /// Get a display string for the source
    pub fn display_string(&self) -> String {
        match self {
            Self::HuggingFace { repo_id } => format!("hf://{repo_id}"),
            Self::LocalFile { path } => format!("file://{}", path.display()),
            Self::Custom { url } => url.clone(),
        }
    }
}

/// Model entry in the registry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model name (e.g., "bert-base-uncased")
    pub name: String,
    /// Model version
    pub version: String,
    /// SHA-256 checksum
    pub sha256: String,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Model source
    pub source: ModelSource,
    /// Local path if mirrored
    pub local_path: Option<PathBuf>,
    /// Model format (gguf, safetensors, etc.)
    pub format: Option<String>,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl ModelEntry {
    /// Create a new model entry
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
        source: ModelSource,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            sha256: sha256.into(),
            size_bytes,
            source,
            local_path: None,
            format: None,
            metadata: HashMap::new(),
        }
    }

    /// Set local path
    pub fn with_local_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.local_path = Some(path.into());
        self
    }

    /// Set format
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if model is available locally
    pub fn is_local(&self) -> bool {
        self.local_path.as_ref().is_some_and(|p| p.exists())
    }

    /// Get size in megabytes
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get size in gigabytes
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}
