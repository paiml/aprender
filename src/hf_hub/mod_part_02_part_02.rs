use super::super::{
    base64_encode, HfHubClient, HfHubError, ModelCard, PushOptions, UploadProgress, UploadResult,
    Result,
};
use std::path::PathBuf;

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
    pub(crate) fn default_cache_dir() -> PathBuf {
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
    pub(crate) fn parse_repo_id(repo_id: &str) -> Result<(&str, &str)> {
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
    #[allow(clippy::needless_pass_by_value)] // PushOptions is consumed/cloned for API simplicity
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
    #[allow(clippy::disallowed_methods)] // serde_json::json! macro internally uses unwrap()
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

        Err(last_error
            .unwrap_or_else(|| HfHubError::NetworkError("Upload failed after retries".to_string())))
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
    #[allow(clippy::disallowed_methods)] // serde_json::json! macro internally uses unwrap()
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
}
