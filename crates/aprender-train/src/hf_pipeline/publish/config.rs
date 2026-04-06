//! Publishing configuration types

use serde::{Deserialize, Serialize};

/// Configuration for publishing a model to HuggingFace Hub
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PublishConfig {
    /// HuggingFace repository ID (e.g., "username/my-model")
    pub repo_id: String,
    /// Repository type
    pub repo_type: RepoType,
    /// Whether the repository should be private
    pub private: bool,
    /// HuggingFace API token (if not set, resolved from env/file)
    #[serde(skip)]
    pub token: Option<String>,
    /// License identifier (e.g., "apache-2.0", "mit")
    pub license: Option<String>,
    /// Tags for discoverability
    pub tags: Vec<String>,
}

/// HuggingFace repository type
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RepoType {
    /// Model repository
    #[default]
    Model,
    /// Dataset repository
    Dataset,
    /// Space (app) repository
    Space,
}

impl RepoType {
    /// API path segment for this repo type
    #[must_use]
    pub fn api_path(&self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
            Self::Space => "spaces",
        }
    }
}

impl std::fmt::Display for RepoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model => write!(f, "model"),
            Self::Dataset => write!(f, "dataset"),
            Self::Space => write!(f, "space"),
        }
    }
}
