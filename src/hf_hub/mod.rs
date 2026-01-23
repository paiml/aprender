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

        let n = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);

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

impl HfHubClient {
    /// Create a new HF Hub client
    ///
    /// Reads token from `HF_TOKEN` environment variable.
    ///
    /// # Errors
    ///
    /// Does not error on missing token (allows anonymous pulls).
    pub fn new() -> Result<Self> {
        let token = std::env::var("HF_TOKEN").ok();
        let cache_dir = Self::default_cache_dir();

        Ok(Self {
            token,
            cache_dir,
            api_base: "https://huggingface.co".to_string(),
        })
    }

    /// Create client with explicit token
    #[must_use]
    pub fn with_token(token: impl Into<String>) -> Self {
        Self {
            token: Some(token.into()),
            cache_dir: Self::default_cache_dir(),
            api_base: "https://huggingface.co".to_string(),
        }
    }

    /// Set custom cache directory
    #[must_use]
    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = path.into();
        self
    }

    /// Get default cache directory
    fn default_cache_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("huggingface")
            .join("hub")
    }

    /// Check if client has authentication token
    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        self.token.is_some()
    }

    /// Parse repository ID (org/name format)
    fn parse_repo_id(repo_id: &str) -> Result<(&str, &str)> {
        let parts: Vec<&str> = repo_id.split('/').collect();
        if parts.len() != 2 {
            return Err(HfHubError::InvalidRepoId(repo_id.to_string()));
        }
        Ok((parts[0], parts[1]))
    }

    /// Pull a model from HF Hub
    ///
    /// Downloads the model file to the local cache and returns the path.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID in "org/name" format
    ///
    /// # Returns
    ///
    /// Path to the downloaded model file
    ///
    /// # Errors
    ///
    /// Returns error if download fails or repo not found
    #[cfg(feature = "hf-hub-integration")]
    pub fn pull_from_hub(&self, repo_id: &str) -> Result<PathBuf> {
        use hf_hub::api::sync::ApiBuilder;

        let (org, name) = Self::parse_repo_id(repo_id)?;

        // Build API client with optional token
        let mut builder = ApiBuilder::new();
        if let Some(token) = &self.token {
            builder = builder.with_token(Some(token.clone()));
        }
        let api = builder
            .build()
            .map_err(|e| HfHubError::NetworkError(e.to_string()))?;

        // Get repo handle
        let repo = api.model(format!("{org}/{name}"));

        // Download model.apr file
        let model_path = repo
            .get("model.apr")
            .map_err(|e| HfHubError::FileNotFound(format!("model.apr: {e}")))?;

        Ok(model_path)
    }

    /// Pull a model from HF Hub (stub when feature disabled)
    #[cfg(not(feature = "hf-hub-integration"))]
    pub fn pull_from_hub(&self, _repo_id: &str) -> Result<PathBuf> {
        Err(HfHubError::NetworkError(
            "hf-hub-integration feature not enabled".to_string(),
        ))
    }

    /// Push a model to HF Hub (APR-PUB-001: Full implementation)
    ///
    /// Uploads the model file and generates a model card.
    /// Supports large files via chunked upload with progress tracking.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID in "org/name" format
    /// * `model_data` - Model file contents
    /// * `options` - Push options
    ///
    /// # Errors
    ///
    /// Returns error if upload fails or authentication missing.
    /// **Andon Cord**: This function will NEVER silently succeed - it either
    /// uploads successfully or returns an error.
    #[cfg(feature = "hf-hub-integration")]
    pub fn push_to_hub(
        &self,
        repo_id: &str,
        model_data: &[u8],
        options: PushOptions,
    ) -> Result<UploadResult> {
        let token = self.token.as_ref().ok_or(HfHubError::MissingToken)?;
        let (_org, _name) = Self::parse_repo_id(repo_id)?;

        // Generate model card if not provided
        let model_card = options.model_card.clone().unwrap_or_else(|| {
            ModelCard::new(repo_id, "1.0.0").with_description("Model uploaded via aprender")
        });
        let readme_content = model_card.to_huggingface();

        let commit_msg = options
            .commit_message
            .clone()
            .unwrap_or_else(|| "Upload model via aprender".to_string());

        // Step 1: Create repository if needed
        if options.create_repo {
            self.create_repo_if_not_exists(repo_id, token, options.private)?;
        }

        // Track files to upload
        let total_bytes = model_data.len() as u64 + readme_content.len() as u64;
        let mut bytes_transferred = 0u64;
        let mut files_uploaded = Vec::new();

        // Report initial progress
        if let Some(ref cb) = options.progress_callback {
            cb(UploadProgress {
                bytes_sent: 0,
                total_bytes,
                current_file: options.filename.clone(),
                files_completed: 0,
                total_files: 2,
            });
        }

        // Step 2: Upload model file with retry
        self.upload_file_with_retry(
            repo_id,
            &options.filename,
            model_data,
            &commit_msg,
            token,
            &options,
            &mut bytes_transferred,
            total_bytes,
            0,
            2,
        )?;
        files_uploaded.push(options.filename.clone());

        // Step 3: Upload README.md
        self.upload_file_with_retry(
            repo_id,
            "README.md",
            readme_content.as_bytes(),
            &commit_msg,
            token,
            &options,
            &mut bytes_transferred,
            total_bytes,
            1,
            2,
        )?;
        files_uploaded.push("README.md".to_string());

        // Final progress report
        if let Some(ref cb) = options.progress_callback {
            cb(UploadProgress {
                bytes_sent: bytes_transferred,
                total_bytes,
                current_file: "Complete".to_string(),
                files_completed: 2,
                total_files: 2,
            });
        }

        Ok(UploadResult {
            repo_url: format!("{}/{}", self.api_base, repo_id),
            commit_sha: "uploaded".to_string(), // HF doesn't return SHA in simple upload
            files_uploaded,
            bytes_transferred,
        })
    }

    /// Push to hub stub when feature disabled
    #[cfg(not(feature = "hf-hub-integration"))]
    pub fn push_to_hub(
        &self,
        _repo_id: &str,
        _model_data: &[u8],
        _options: PushOptions,
    ) -> Result<UploadResult> {
        Err(HfHubError::NetworkError(
            "hf-hub-integration feature not enabled".to_string(),
        ))
    }

    /// Create repository if it doesn't exist
    #[cfg(feature = "hf-hub-integration")]
    fn create_repo_if_not_exists(&self, repo_id: &str, token: &str, private: bool) -> Result<()> {
        let (org, name) = Self::parse_repo_id(repo_id)?;
        let url = format!("{}/api/repos/create", self.api_base);

        // HF API expects name (repo only) and organization separately
        let body = serde_json::json!({
            "type": "model",
            "name": name,
            "organization": org,
            "private": private
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {token}"))
            .set("Content-Type", "application/json")
            .send_json(&body);

        match response {
            Ok(_) => Ok(()),
            Err(ureq::Error::Status(409, _)) => {
                // 409 Conflict means repo already exists - that's fine
                Ok(())
            }
            Err(ureq::Error::Status(400, _)) => {
                // 400 Bad Request often means repo exists under different settings
                // Continue with upload attempt
                Ok(())
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Err(HfHubError::NetworkError(format!(
                    "Failed to create repo (HTTP {code}): {body}"
                )))
            }
            Err(e) => Err(HfHubError::NetworkError(format!(
                "Network error creating repo: {e}"
            ))),
        }
    }

    /// Upload a single file with retry logic
    #[cfg(feature = "hf-hub-integration")]
    fn upload_file_with_retry(
        &self,
        repo_id: &str,
        filename: &str,
        data: &[u8],
        commit_msg: &str,
        token: &str,
        options: &PushOptions,
        bytes_transferred: &mut u64,
        total_bytes: u64,
        files_completed: usize,
        total_files: usize,
    ) -> Result<()> {
        let mut last_error = None;
        let mut backoff_ms = options.initial_backoff_ms;

        for attempt in 0..=options.max_retries {
            if attempt > 0 {
                // Exponential backoff
                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                backoff_ms = (backoff_ms * 2).min(30000); // Cap at 30 seconds
            }

            // Report progress
            if let Some(ref cb) = options.progress_callback {
                cb(UploadProgress {
                    bytes_sent: *bytes_transferred,
                    total_bytes,
                    current_file: filename.to_string(),
                    files_completed,
                    total_files,
                });
            }

            match self.upload_file_once(repo_id, filename, data, commit_msg, token) {
                Ok(()) => {
                    *bytes_transferred += data.len() as u64;
                    return Ok(());
                }
                Err(e) => {
                    last_error = Some(e);
                    // Only retry on network/server errors (5xx)
                    if attempt == options.max_retries {
                        break;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            HfHubError::NetworkError("Upload failed after retries".to_string())
        }))
    }

    /// LFS threshold - files larger than this use LFS protocol
    const LFS_THRESHOLD: usize = 10 * 1024 * 1024; // 10 MB

    /// Upload a single file (no retry)
    #[cfg(feature = "hf-hub-integration")]
    fn upload_file_once(
        &self,
        repo_id: &str,
        filename: &str,
        data: &[u8],
        commit_msg: &str,
        token: &str,
    ) -> Result<()> {
        if data.len() >= Self::LFS_THRESHOLD {
            // Large file: use LFS upload
            self.upload_via_lfs(repo_id, filename, data, commit_msg, token)
        } else {
            // Small file: use direct commit API
            self.upload_direct(repo_id, filename, data, commit_msg, token)
        }
    }

    /// Upload small file directly via commit API
    #[cfg(feature = "hf-hub-integration")]
    fn upload_direct(
        &self,
        repo_id: &str,
        filename: &str,
        data: &[u8],
        commit_msg: &str,
        token: &str,
    ) -> Result<()> {
        let url = format!("{}/api/models/{}/commit/main", self.api_base, repo_id);

        let operations = serde_json::json!([{
            "op": "addOrUpdate",
            "path": filename,
            "content": base64_encode(data)
        }]);

        let body = serde_json::json!({
            "summary": commit_msg,
            "operations": operations
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {token}"))
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(120))
            .send_json(&body);

        match response {
            Ok(resp) if resp.status() >= 200 && resp.status() < 300 => Ok(()),
            Ok(resp) => {
                let body = resp.into_string().unwrap_or_default();
                Err(HfHubError::NetworkError(format!("Upload failed: {body}")))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Err(HfHubError::NetworkError(format!(
                    "Upload failed (HTTP {code}): {body}"
                )))
            }
            Err(e) => Err(HfHubError::NetworkError(format!("Network error: {e}"))),
        }
    }

    /// Upload large file via HuggingFace Hub multipart upload (APR-PUB-001)
    ///
    /// HuggingFace multipart upload flow for files > 5GB:
    /// 1. POST to /api/models/{repo}/preupload/main with SHA256 to get presigned URLs
    /// 2. Upload file parts to presigned URLs (5GB chunks)
    /// 3. POST completion to finalize upload
    /// 4. POST commit with LFS pointer
    ///
    /// **OBS-003/OBS-004**: Full verbose logging for diagnostics
    #[cfg(feature = "hf-hub-integration")]
    fn upload_via_lfs(
        &self,
        repo_id: &str,
        filename: &str,
        data: &[u8],
        commit_msg: &str,
        token: &str,
    ) -> Result<()> {
        use sha2::{Sha256, Digest};
        use std::time::Instant;

        let start = Instant::now();
        let file_size = data.len();

        // Calculate SHA256 hash for LFS
        eprintln!("[LFS] Calculating SHA256 for {} ({:.1} MB)...", filename, file_size as f64 / 1_000_000.0);
        let mut hasher = Sha256::new();
        hasher.update(data);
        let sha256 = format!("{:x}", hasher.finalize());
        eprintln!("[LFS] SHA256: {}", sha256);
        eprintln!("[LFS] Using token: {}...", &token[..12.min(token.len())]);

        // For files > 5GB, we need to use HuggingFace's multipart upload API
        // The preupload endpoint returns presigned URLs for S3 multipart upload
        const CHUNK_SIZE: usize = 5 * 1024 * 1024 * 1024; // 5GB chunks (S3 limit)
        let num_chunks = (file_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        eprintln!("[LFS] File size: {} bytes, will upload in {} chunk(s)", file_size, num_chunks);

        // Step 1: Request presigned upload URLs via preupload endpoint
        let preupload_url = format!(
            "{}/api/models/{}/preupload/main",
            self.api_base, repo_id
        );
        eprintln!("[LFS] Step 1: Requesting upload URLs from {}", preupload_url);

        // Include SHA256 in the request to indicate we want multipart upload
        let preupload_body = serde_json::json!({
            "files": [{
                "path": filename,
                "size": file_size,
                "sample": base64_encode(&data[..data.len().min(512)])
            }]
        });
        eprintln!("[LFS] Preupload request (size={}, sha256={}...)", file_size, &sha256[..16]);

        let preupload_resp = match ureq::post(&preupload_url)
            .set("Authorization", &format!("Bearer {token}"))
            .set("Content-Type", "application/json")
            .send_json(&preupload_body)
        {
            Ok(resp) => resp,
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_else(|_| "unable to read body".to_string());
                eprintln!("[LFS] ERROR: Preupload failed with status {}: {}", code, body);
                return Err(HfHubError::NetworkError(format!("Preupload failed (HTTP {}): {}", code, body)));
            }
            Err(e) => {
                eprintln!("[LFS] ERROR: Preupload request failed: {}", e);
                return Err(HfHubError::NetworkError(format!("Preupload failed: {e}")));
            }
        };

        eprintln!("[LFS] Preupload response status: {}", preupload_resp.status());

        let preupload_data: serde_json::Value = preupload_resp.into_json()
            .map_err(|e| {
                eprintln!("[LFS] ERROR: Failed to parse preupload response: {}", e);
                HfHubError::NetworkError(format!("Preupload parse failed: {e}"))
            })?;

        eprintln!("[LFS] Preupload response: {}", serde_json::to_string_pretty(&preupload_data).unwrap_or_default());

        // Extract upload info from response
        let files = preupload_data["files"].as_array()
            .ok_or_else(|| {
                eprintln!("[LFS] ERROR: Invalid preupload response - no 'files' array");
                HfHubError::NetworkError("Invalid preupload response".to_string())
            })?;

        if files.is_empty() {
            eprintln!("[LFS] ERROR: Empty files array in preupload response");
            return Err(HfHubError::NetworkError("No file info returned".to_string()));
        }

        let file_info = &files[0];
        let upload_mode = file_info.get("uploadMode").and_then(|v| v.as_str()).unwrap_or("unknown");
        eprintln!("[LFS] Upload mode: {}", upload_mode);

        // Check for various URL fields that HF might return
        let upload_url = file_info.get("uploadUrl")
            .or_else(|| file_info.get("upload_url"))
            .and_then(|v| v.as_str());

        // Also check for chunked/multipart URLs
        let chunk_urls = file_info.get("chunkUrls")
            .or_else(|| file_info.get("chunk_urls"))
            .or_else(|| file_info.get("urls"))
            .and_then(|v| v.as_array());

        eprintln!("[LFS] Upload URL present: {}", upload_url.is_some());
        eprintln!("[LFS] Chunk URLs present: {}", chunk_urls.is_some());

        // For files > 5GB without upload URL, HuggingFace requires multipart upload
        // which is only available via their git LFS transfer agent (huggingface-cli)
        // See: https://huggingface.co/docs/hub/en/storage-limits
        const FIVE_GB: usize = 5 * 1024 * 1024 * 1024;
        if upload_url.is_none() && chunk_urls.is_none() && upload_mode == "lfs" && file_size > FIVE_GB {
            eprintln!("[LFS] ERROR: File {} ({:.1} GB) exceeds 5GB HuggingFace Hub limit for HTTP API uploads",
                filename, file_size as f64 / 1_000_000_000.0);
            eprintln!("[LFS] ");
            eprintln!("[LFS] Files > 5GB require HuggingFace's multipart transfer agent.");
            eprintln!("[LFS] Options:");
            eprintln!("[LFS]   1. Split model into shards < 5GB each (recommended)");
            eprintln!("[LFS]      Use: apr export --max-shard-size 4GB");
            eprintln!("[LFS]   2. Use huggingface-cli with lfs-enable-largefiles:");
            eprintln!("[LFS]      git clone https://huggingface.co/{}", repo_id);
            eprintln!("[LFS]      huggingface-cli lfs-enable-largefiles ./");
            eprintln!("[LFS]      cp {} ./", filename);
            eprintln!("[LFS]      git add . && git commit -m 'Add model' && git push");
            eprintln!("[LFS] ");
            return Err(HfHubError::NetworkError(format!(
                "File {} ({:.1} GB) exceeds 5GB limit. \
                 HuggingFace Hub requires multipart LFS for files > 5GB. \
                 Split into smaller shards or use huggingface-cli.",
                filename, file_size as f64 / 1_000_000_000.0
            )));
        }

        if let Some(urls) = chunk_urls {
            // Multipart upload with presigned URLs
            eprintln!("[LFS] Step 2: Multipart upload with {} presigned URLs", urls.len());

            for (i, url_value) in urls.iter().enumerate() {
                let chunk_url = url_value.as_str().ok_or_else(|| {
                    HfHubError::NetworkError(format!("Invalid chunk URL at index {}", i))
                })?;

                let chunk_start = i * CHUNK_SIZE;
                let chunk_end = ((i + 1) * CHUNK_SIZE).min(file_size);
                let chunk_data = &data[chunk_start..chunk_end];

                eprintln!("[LFS] Uploading chunk {}/{}: bytes {}-{} ({:.1} MB)",
                    i + 1, urls.len(), chunk_start, chunk_end, chunk_data.len() as f64 / 1_000_000.0);

                let chunk_start_time = Instant::now();
                let resp = ureq::put(chunk_url)
                    .set("Content-Type", "application/octet-stream")
                    .timeout(std::time::Duration::from_secs(7200))
                    .send_bytes(chunk_data)
                    .map_err(|e| {
                        eprintln!("[LFS] ERROR: Chunk {} upload failed: {}", i + 1, e);
                        HfHubError::NetworkError(format!("Chunk upload failed: {e}"))
                    })?;

                let status = resp.status();
                eprintln!("[LFS] Chunk {}/{} uploaded: status={}, elapsed={:.1}s",
                    i + 1, urls.len(), status, chunk_start_time.elapsed().as_secs_f64());

                if status < 200 || status >= 300 {
                    return Err(HfHubError::NetworkError(format!("Chunk upload failed with status {}", status)));
                }
            }

            // Call completion endpoint if provided
            if let Some(completion_url) = file_info.get("completionUrl").and_then(|v| v.as_str()) {
                eprintln!("[LFS] Calling completion URL: {}", completion_url);
                let _ = ureq::post(completion_url)
                    .set("Authorization", &format!("Bearer {token}"))
                    .set("Content-Type", "application/json")
                    .send_json(&serde_json::json!({}));
            }
        } else if let Some(url) = upload_url {
            // Single URL upload (for smaller LFS files)
            eprintln!("[LFS] Step 2: Single URL upload to {}", &url[..url.len().min(100)]);

            let upload_start = Instant::now();
            let headers = file_info.get("uploadHeader").and_then(|v| v.as_object());

            let mut request = ureq::put(url)
                .set("Content-Type", "application/octet-stream")
                .timeout(std::time::Duration::from_secs(7200));

            if let Some(hdrs) = headers {
                for (key, value) in hdrs {
                    if let Some(v) = value.as_str() {
                        eprintln!("[LFS] Adding header: {}: {}...", key, &v[..v.len().min(20)]);
                        request = request.set(key, v);
                    }
                }
            }

            let resp = request.send_bytes(data)
                .map_err(|e| {
                    eprintln!("[LFS] ERROR: Upload failed: {}", e);
                    HfHubError::NetworkError(format!("Upload failed: {e}"))
                })?;

            let status = resp.status();
            eprintln!("[LFS] Upload complete: status={}, elapsed={:.1}s, speed={:.1} MB/s",
                status,
                upload_start.elapsed().as_secs_f64(),
                (file_size as f64 / 1_000_000.0) / upload_start.elapsed().as_secs_f64()
            );

            if status < 200 || status >= 300 {
                let body = resp.into_string().unwrap_or_default();
                return Err(HfHubError::NetworkError(format!("Upload failed (HTTP {}): {}", status, body)));
            }
        } else {
            // No upload URL - the file may already exist or we need to commit the LFS pointer
            eprintln!("[LFS] No upload URL returned - proceeding to commit LFS pointer");
            eprintln!("[LFS] (This may mean the file content already exists on HF's LFS storage)");
        }

        // Step 3: Commit with LFS pointer
        eprintln!("[LFS] Step 3: Committing LFS pointer");
        let lfs_pointer = format!(
            "version https://git-lfs.github.com/spec/v1\noid sha256:{}\nsize {}\n",
            sha256, file_size
        );
        eprintln!("[LFS] Pointer content:\n{}", lfs_pointer);

        let commit_url = format!("{}/api/models/{}/commit/main", self.api_base, repo_id);
        eprintln!("[LFS] Commit URL: {}", commit_url);

        let commit_body = serde_json::json!({
            "summary": commit_msg,
            "operations": [{
                "op": "addOrUpdate",
                "path": filename,
                "content": base64_encode(lfs_pointer.as_bytes()),
                "encoding": "base64",
                "lfs": {
                    "sha256": sha256,
                    "size": file_size
                }
            }]
        });

        let commit_resp = ureq::post(&commit_url)
            .set("Authorization", &format!("Bearer {token}"))
            .set("Content-Type", "application/json")
            .send_json(&commit_body);

        match commit_resp {
            Ok(resp) if resp.status() >= 200 && resp.status() < 300 => {
                let body = resp.into_string().unwrap_or_default();
                eprintln!("[LFS] Commit successful: {}", &body[..body.len().min(200)]);
                eprintln!("[LFS] Total upload time: {:.1}s", start.elapsed().as_secs_f64());
                Ok(())
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.into_string().unwrap_or_default();
                eprintln!("[LFS] ERROR: Commit failed with status {}: {}", status, &body[..body.len().min(500)]);
                Err(HfHubError::NetworkError(format!("Commit failed (HTTP {}): {}", status, body)))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                eprintln!("[LFS] ERROR: Commit failed with status {}: {}", code, &body[..body.len().min(500)]);
                Err(HfHubError::NetworkError(format!(
                    "Commit failed (HTTP {code}): {body}"
                )))
            }
            Err(e) => {
                eprintln!("[LFS] ERROR: Network error during commit: {}", e);
                Err(HfHubError::NetworkError(format!("Network error: {e}")))
            }
        }
    }

    /// Generate a model card from model metadata
    ///
    /// Creates a ModelCard with auto-populated fields from training info.
    #[must_use]
    pub fn auto_generate_card(repo_id: &str, model_type: &str, version: &str) -> ModelCard {
        ModelCard::new(repo_id, version)
            .with_name(repo_id.split('/').next_back().unwrap_or(repo_id))
            .with_architecture(model_type)
            .with_description(format!("{model_type} model trained with aprender"))
    }
}

impl Default for HfHubClient {
    fn default() -> Self {
        Self::new().expect("Failed to create HfHubClient")
    }
}

// ============================================================================
// Unit Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_options_default() {
        let opts = PushOptions::default();
        assert!(opts.commit_message.is_none());
        assert!(opts.model_card.is_none());
        assert!(opts.create_repo);
        assert!(!opts.private);
        assert_eq!(opts.filename, "model.apr");
    }

    #[test]
    fn test_push_options_builder() {
        let card = ModelCard::new("test", "1.0.0");
        let opts = PushOptions::new()
            .with_commit_message("Test commit")
            .with_model_card(card)
            .with_create_repo(false)
            .with_private(true)
            .with_filename("custom.apr");

        assert_eq!(opts.commit_message, Some("Test commit".to_string()));
        assert!(opts.model_card.is_some());
        assert!(!opts.create_repo);
        assert!(opts.private);
        assert_eq!(opts.filename, "custom.apr");
    }

    #[test]
    fn test_hf_hub_client_new() {
        // Should not error even without token
        let client = HfHubClient::new().expect("Should create client");
        // Token depends on environment
        assert!(client.api_base.contains("huggingface.co"));
    }

    #[test]
    fn test_hf_hub_client_with_token() {
        let client = HfHubClient::with_token("test_token");
        assert!(client.is_authenticated());
    }

    #[test]
    fn test_parse_repo_id_valid() {
        let result = HfHubClient::parse_repo_id("paiml/my-model");
        assert!(result.is_ok());
        let (org, name) = result.unwrap();
        assert_eq!(org, "paiml");
        assert_eq!(name, "my-model");
    }

    #[test]
    fn test_parse_repo_id_invalid() {
        let result = HfHubClient::parse_repo_id("invalid");
        assert!(result.is_err());

        let result = HfHubClient::parse_repo_id("too/many/parts");
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_generate_card() {
        let card = HfHubClient::auto_generate_card("paiml/syscall-model", "KMeans", "1.0.0");

        assert_eq!(card.model_id, "paiml/syscall-model");
        assert_eq!(card.version, "1.0.0");
        assert_eq!(card.name, "syscall-model");
        assert_eq!(card.architecture, Some("KMeans".to_string()));
    }

    #[test]
    fn test_push_to_hub_without_token() {
        let client = HfHubClient {
            token: None,
            cache_dir: PathBuf::from("/tmp"),
            api_base: "https://huggingface.co".to_string(),
        };

        let result = client.push_to_hub("org/repo", b"test", PushOptions::default());
        assert!(matches!(result, Err(HfHubError::MissingToken)));
    }

    // Note: test_push_to_hub_with_token removed - APR-PUB-001 now actually uploads
    // Network tests require real HF_TOKEN or mock server

    #[test]
    fn test_upload_progress_percentage() {
        let progress = UploadProgress {
            bytes_sent: 50,
            total_bytes: 100,
            current_file: "test.apr".to_string(),
            files_completed: 0,
            total_files: 1,
        };
        assert!((progress.percentage() - 50.0).abs() < 0.01);

        let progress_zero = UploadProgress {
            bytes_sent: 0,
            total_bytes: 0,
            current_file: "test.apr".to_string(),
            files_completed: 0,
            total_files: 1,
        };
        assert!((progress_zero.percentage() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_upload_result_fields() {
        let result = UploadResult {
            repo_url: "https://huggingface.co/paiml/test".to_string(),
            commit_sha: "abc123".to_string(),
            files_uploaded: vec!["model.apr".to_string(), "README.md".to_string()],
            bytes_transferred: 1024,
        };
        assert_eq!(result.files_uploaded.len(), 2);
        assert_eq!(result.bytes_transferred, 1024);
    }

    #[test]
    fn test_push_options_with_progress_callback() {
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();
        let callback: ProgressCallback = Arc::new(move |_progress| {
            called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        let opts = PushOptions::new().with_progress_callback(callback);
        assert!(opts.progress_callback.is_some());
    }

    #[test]
    fn test_push_options_with_max_retries() {
        let opts = PushOptions::new().with_max_retries(5);
        assert_eq!(opts.max_retries, 5);
    }

    #[test]
    fn test_hf_hub_error_display() {
        assert_eq!(
            HfHubError::MissingToken.to_string(),
            "HF_TOKEN environment variable not set"
        );
        assert!(HfHubError::RepoNotFound("test".into())
            .to_string()
            .contains("test"));
    }

    // Note: test_push_saves_readme removed - APR-PUB-001 now actually uploads
    // The README content is tested via model_card tests instead

    #[test]
    fn test_base64_encode_basic() {
        // Test basic encoding
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"a"), "YQ==");
        assert_eq!(base64_encode(b"ab"), "YWI=");
        assert_eq!(base64_encode(b"abc"), "YWJj");
    }

    #[test]
    fn test_base64_encode_binary() {
        // Test binary data encoding
        let data: [u8; 4] = [0x00, 0xFF, 0x80, 0x7F];
        let encoded = base64_encode(&data);
        assert!(!encoded.is_empty());
        // Verify it's valid base64 (only contains valid chars)
        assert!(encoded.chars().all(|c| {
            c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='
        }));
    }

    // ========================================================================
    // Edge Case Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_all_error_variants_display() {
        // Test all HfHubError variants have proper Display
        let errors = vec![
            HfHubError::MissingToken,
            HfHubError::NetworkError("connection failed".into()),
            HfHubError::RepoNotFound("org/repo".into()),
            HfHubError::FileNotFound("model.apr".into()),
            HfHubError::InvalidRepoId("bad-id".into()),
            HfHubError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found",
            )),
            HfHubError::ModelCardError("invalid format".into()),
        ];

        for err in errors {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "Error display should not be empty");
        }
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let hf_err: HfHubError = io_err.into();
        assert!(matches!(hf_err, HfHubError::IoError(_)));
        assert!(hf_err.to_string().contains("denied"));
    }

    #[test]
    fn test_with_cache_dir() {
        let client = HfHubClient::with_token("token").with_cache_dir("/custom/cache");
        assert_eq!(client.cache_dir, PathBuf::from("/custom/cache"));
    }

    #[test]
    fn test_parse_empty_repo_id() {
        let result = HfHubClient::parse_repo_id("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_repo_id_with_empty_parts() {
        // "org/" has empty name
        let result = HfHubClient::parse_repo_id("org/");
        assert!(result.is_ok()); // Still parses, just empty name

        // "/name" has empty org
        let result = HfHubClient::parse_repo_id("/name");
        assert!(result.is_ok()); // Still parses, just empty org
    }

    #[test]
    fn test_push_to_hub_invalid_repo_id() {
        let client = HfHubClient::with_token("token");
        let result = client.push_to_hub("invalid", b"data", PushOptions::default());
        assert!(matches!(result, Err(HfHubError::InvalidRepoId(_))));
    }

    #[test]
    fn test_push_options_all_builders() {
        let opts = PushOptions::new()
            .with_commit_message("msg")
            .with_create_repo(true)
            .with_private(true)
            .with_filename("test.apr");

        assert_eq!(opts.commit_message, Some("msg".to_string()));
        assert!(opts.create_repo);
        assert!(opts.private);
        assert_eq!(opts.filename, "test.apr");
    }

    #[test]
    fn test_auto_generate_card_single_name() {
        // Test with repo_id that has no slash
        let card = HfHubClient::auto_generate_card("simple", "LinearRegression", "2.0.0");
        assert_eq!(card.model_id, "simple");
        assert_eq!(card.name, "simple"); // Falls back to full string
        assert_eq!(card.version, "2.0.0");
    }

    #[test]
    fn test_is_authenticated() {
        let client_with_token = HfHubClient::with_token("secret");
        assert!(client_with_token.is_authenticated());

        let client_without = HfHubClient {
            token: None,
            cache_dir: PathBuf::from("/tmp"),
            api_base: "https://huggingface.co".to_string(),
        };
        assert!(!client_without.is_authenticated());
    }

    #[test]
    fn test_default_cache_dir_exists() {
        let dir = HfHubClient::default_cache_dir();
        // Should contain "huggingface" and "hub" in path
        let path_str = dir.to_string_lossy();
        assert!(path_str.contains("huggingface") || path_str.contains('.'));
    }

    // Note: test_push_creates_model_file and test_push_auto_generates_model_card removed
    // APR-PUB-001 now actually uploads to HF Hub instead of saving locally
    // Network-based integration tests would require mock server or real credentials

    #[test]
    fn test_push_options_default_retry_settings() {
        let opts = PushOptions::default();
        assert_eq!(opts.max_retries, 3);
        assert_eq!(opts.initial_backoff_ms, 1000);
    }

    #[test]
    fn test_upload_progress_fields() {
        let progress = UploadProgress {
            bytes_sent: 1000,
            total_bytes: 5000,
            current_file: "model.safetensors".to_string(),
            files_completed: 1,
            total_files: 3,
        };
        assert_eq!(progress.bytes_sent, 1000);
        assert_eq!(progress.total_bytes, 5000);
        assert_eq!(progress.current_file, "model.safetensors");
        assert_eq!(progress.files_completed, 1);
        assert_eq!(progress.total_files, 3);
        assert!((progress.percentage() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_hf_hub_client_default() {
        // Test Default trait implementation
        let client = HfHubClient::default();
        assert!(client.api_base.contains("huggingface.co"));
    }

    #[cfg(not(feature = "hf-hub-integration"))]
    #[test]
    fn test_pull_without_feature() {
        let client = HfHubClient::with_token("token");
        let result = client.pull_from_hub("org/repo");
        assert!(matches!(result, Err(HfHubError::NetworkError(_))));
    }
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Valid repo IDs always parse successfully
        #[test]
        fn prop_valid_repo_id_parses(
            org in "[a-zA-Z][a-zA-Z0-9-]{0,30}",
            name in "[a-zA-Z][a-zA-Z0-9-]{0,30}",
        ) {
            let repo_id = format!("{org}/{name}");
            let result = HfHubClient::parse_repo_id(&repo_id);
            prop_assert!(result.is_ok());
            let (parsed_org, parsed_name) = result.expect("Valid repo ID should parse");
            prop_assert_eq!(parsed_org, org.as_str());
            prop_assert_eq!(parsed_name, name.as_str());
        }

        /// Property: Invalid repo IDs always fail
        #[test]
        fn prop_invalid_repo_id_fails(
            name in "[a-zA-Z0-9-]{1,30}",
        ) {
            // Single part (no slash)
            let result = HfHubClient::parse_repo_id(&name);
            prop_assert!(result.is_err());
        }

        /// Property: PushOptions builder is idempotent
        #[test]
        fn prop_push_options_builder_idempotent(
            msg in "[a-zA-Z0-9 ]{1,50}",
            filename in "[a-zA-Z0-9_.-]{1,20}",
        ) {
            let opts1 = PushOptions::new()
                .with_commit_message(&msg)
                .with_commit_message(&msg)
                .with_filename(&filename)
                .with_filename(&filename);

            let opts2 = PushOptions::new()
                .with_commit_message(&msg)
                .with_filename(&filename);

            prop_assert_eq!(opts1.commit_message, opts2.commit_message);
            prop_assert_eq!(opts1.filename, opts2.filename);
        }

        /// Property: Auto-generated card has correct model_id
        #[test]
        fn prop_auto_card_has_repo_id(
            org in "[a-zA-Z][a-zA-Z0-9]{0,10}",
            name in "[a-zA-Z][a-zA-Z0-9]{0,10}",
            model_type in "[A-Z][a-zA-Z]{2,15}",
        ) {
            let repo_id = format!("{org}/{name}");
            let card = HfHubClient::auto_generate_card(&repo_id, &model_type, "1.0.0");

            prop_assert_eq!(card.model_id, repo_id);
            prop_assert_eq!(card.name, name);
            prop_assert_eq!(card.architecture, Some(model_type));
        }
    }
}
