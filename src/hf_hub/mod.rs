//! Hugging Face Hub Integration (GH-100, APR-PUB-001)
//!
//! Enables publishing and pulling .apr models to/from Hugging Face Hub.
//!
//! # Features
//!
//! - Push trained models to HF Hub with auto-generated model cards
//! - Pull models from HF Hub for inference
//! - Automatic model card generation from training metadata
//! - **Large file upload via LFS** (APR-PUB-001 fix)
//! - **Progress tracking for uploads**
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::hf_hub::{HfHubClient, PushOptions};
//!
//! // Initialize client (uses HF_TOKEN env var)
//! let client = HfHubClient::new()?;
//!
//! // Push model to hub
//! client.push_to_hub(
//!     "paiml/my-model",
//!     &model_bytes,
//!     PushOptions::default()
//!         .with_commit_message("Initial model upload")
//! )?;
//!
//! // Pull model from hub
//! let model_path = client.pull_from_hub("paiml/my-model")?;
//! ```
//!
//! # Authentication
//!
//! Set the `HF_TOKEN` environment variable with your Hugging Face token:
//! ```bash
//! export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
//! ```
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Auto-generate model cards to prevent incomplete documentation
//! - **Muda**: Eliminate manual model card creation
//! - **Kaizen**: Continuous model versioning through HF Hub
//! - **Andon Cord**: Fail loudly on upload errors (APR-PUB-001)

use crate::format::model_card::ModelCard;
use std::path::PathBuf;
use std::sync::Arc;

/// Base64 encode bytes for HF Hub API (APR-PUB-001)
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);

    for chunk in data.chunks(3) {
        let mut buf = [0u8; 3];
        buf[..chunk.len()].copy_from_slice(chunk);

        let n = (u32::from(buf[0]) << 16) | (u32::from(buf[1]) << 8) | u32::from(buf[2]);

        result.push(ALPHABET[(n >> 18) as usize & 0x3F] as char);
        result.push(ALPHABET[(n >> 12) as usize & 0x3F] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[(n >> 6) as usize & 0x3F] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[n as usize & 0x3F] as char);
        } else {
            result.push('=');
        }
    }

    result
}

/// Error type for HF Hub operations
#[derive(Debug)]
pub enum HfHubError {
    /// Missing authentication token
    MissingToken,
    /// Network or API error
    NetworkError(String),
    /// Repository not found
    RepoNotFound(String),
    /// File not found in repository
    FileNotFound(String),
    /// Invalid repository ID format
    InvalidRepoId(String),
    /// IO error
    IoError(std::io::Error),
    /// Model card generation error
    ModelCardError(String),
}

impl std::fmt::Display for HfHubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingToken => write!(f, "HF_TOKEN environment variable not set"),
            Self::NetworkError(e) => write!(f, "Network error: {e}"),
            Self::RepoNotFound(repo) => write!(f, "Repository not found: {repo}"),
            Self::FileNotFound(file) => write!(f, "File not found: {file}"),
            Self::InvalidRepoId(id) => write!(f, "Invalid repo ID (expected 'org/name'): {id}"),
            Self::IoError(e) => write!(f, "IO error: {e}"),
            Self::ModelCardError(e) => write!(f, "Model card error: {e}"),
        }
    }
}

impl std::error::Error for HfHubError {}

impl From<std::io::Error> for HfHubError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Result type for HF Hub operations
pub type Result<T> = std::result::Result<T, HfHubError>;

// ============================================================================
// Upload Progress Types (APR-PUB-001)
// ============================================================================

/// Upload progress information for tracking large file uploads
#[derive(Debug, Clone)]
pub struct UploadProgress {
    /// Bytes sent so far
    pub bytes_sent: u64,
    /// Total bytes to send
    pub total_bytes: u64,
    /// Current file being uploaded
    pub current_file: String,
    /// Number of files completed
    pub files_completed: usize,
    /// Total number of files
    pub total_files: usize,
}

impl UploadProgress {
    /// Calculate upload percentage (0.0 to 100.0)
    #[must_use]
    pub fn percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            100.0
        } else {
            (self.bytes_sent as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

/// Progress callback type for upload tracking
pub type ProgressCallback = Arc<dyn Fn(UploadProgress) + Send + Sync>;

/// Result of a successful upload
#[derive(Debug, Clone)]
pub struct UploadResult {
    /// URL to the repository
    pub repo_url: String,
    /// Commit SHA of the upload
    pub commit_sha: String,
    /// List of files uploaded
    pub files_uploaded: Vec<String>,
    /// Total bytes transferred
    pub bytes_transferred: u64,
}

/// Options for pushing models to HF Hub
#[derive(Clone)]
pub struct PushOptions {
    /// Commit message for the upload
    pub commit_message: Option<String>,
    /// Model card to include (auto-generated if None)
    pub model_card: Option<ModelCard>,
    /// Create repo if it doesn't exist
    pub create_repo: bool,
    /// Make repository private
    pub private: bool,
    /// Filename for the model (default: model.apr)
    pub filename: String,
    /// Progress callback for tracking uploads (APR-PUB-001)
    pub progress_callback: Option<ProgressCallback>,
    /// Maximum retry attempts for failed uploads
    pub max_retries: usize,
    /// Initial backoff in milliseconds for retries
    pub initial_backoff_ms: u64,
}

impl std::fmt::Debug for PushOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PushOptions")
            .field("commit_message", &self.commit_message)
            .field("model_card", &self.model_card)
            .field("create_repo", &self.create_repo)
            .field("private", &self.private)
            .field("filename", &self.filename)
            .field("progress_callback", &self.progress_callback.is_some())
            .field("max_retries", &self.max_retries)
            .field("initial_backoff_ms", &self.initial_backoff_ms)
            .finish()
    }
}

impl Default for PushOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl PushOptions {
    /// Create new push options with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            commit_message: None,
            model_card: None,
            create_repo: true,
            private: false,
            filename: "model.apr".to_string(),
            progress_callback: None,
            max_retries: 3,
            initial_backoff_ms: 1000,
        }
    }

    /// Set commit message
    #[must_use]
    pub fn with_commit_message(mut self, msg: impl Into<String>) -> Self {
        self.commit_message = Some(msg.into());
        self
    }

    /// Set model card
    #[must_use]
    pub fn with_model_card(mut self, card: ModelCard) -> Self {
        self.model_card = Some(card);
        self
    }

    /// Set create_repo flag
    #[must_use]
    pub fn with_create_repo(mut self, create: bool) -> Self {
        self.create_repo = create;
        self
    }

    /// Set private flag
    #[must_use]
    pub fn with_private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Set filename
    #[must_use]
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = filename.into();
        self
    }

    /// Set progress callback for upload tracking (APR-PUB-001)
    #[must_use]
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Set maximum retry attempts
    #[must_use]
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }
}

/// Hugging Face Hub client for model operations
#[derive(Debug)]
pub struct HfHubClient {
    /// HF API token
    token: Option<String>,
    /// Cache directory for downloaded models
    cache_dir: PathBuf,
    /// HF Hub API base URL
    api_base: String,
}

include!("mod_part_02.rs");
include!("mod_part_03.rs");
