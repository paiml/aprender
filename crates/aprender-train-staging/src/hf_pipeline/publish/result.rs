//! Publishing result and error types

use std::fmt;

/// Successful publish result
#[derive(Clone, Debug)]
pub struct PublishResult {
    /// Repository URL on HuggingFace
    pub repo_url: String,
    /// Repository ID
    pub repo_id: String,
    /// Number of files uploaded
    pub files_uploaded: usize,
    /// Whether the model card was generated
    pub model_card_generated: bool,
}

impl fmt::Display for PublishResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Published to {} ({} files{})",
            self.repo_url,
            self.files_uploaded,
            if self.model_card_generated { " + model card" } else { "" }
        )
    }
}

/// Errors during publishing
#[derive(Debug, thiserror::Error)]
pub enum PublishError {
    /// Repository creation failed
    #[error("Failed to create repository '{repo_id}': {message}")]
    RepoCreationFailed { repo_id: String, message: String },

    /// File upload failed
    #[error("Failed to upload '{path}': {message}")]
    UploadFailed { path: String, message: String },

    /// Authentication required
    #[error("Authentication required: set HF_TOKEN or use with_token()")]
    AuthRequired,

    /// Invalid repository ID
    #[error("Invalid repository ID '{repo_id}': must be 'owner/name'")]
    InvalidRepoId { repo_id: String },

    /// HTTP error
    #[error("HTTP error: {message}")]
    Http { message: String },

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}
