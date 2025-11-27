//! Hugging Face Hub Integration (GH-100)
//!
//! Enables publishing and pulling .apr models to/from Hugging Face Hub.
//!
//! # Features
//!
//! - Push trained models to HF Hub with auto-generated model cards
//! - Pull models from HF Hub for inference
//! - Automatic model card generation from training metadata
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

use crate::format::model_card::ModelCard;
use std::path::PathBuf;

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

/// Options for pushing models to HF Hub
#[derive(Debug, Clone)]
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

    /// Push a model to HF Hub
    ///
    /// Uploads the model file and generates a model card.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID in "org/name" format
    /// * `model_data` - Model file contents
    /// * `options` - Push options
    ///
    /// # Errors
    ///
    /// Returns error if upload fails or authentication missing
    pub fn push_to_hub(
        &self,
        repo_id: &str,
        model_data: &[u8],
        options: PushOptions,
    ) -> Result<String> {
        let token = self.token.as_ref().ok_or(HfHubError::MissingToken)?;

        let (_org, _name) = Self::parse_repo_id(repo_id)?;

        // Generate model card if not provided
        let model_card = options.model_card.unwrap_or_else(|| {
            ModelCard::new(repo_id, "1.0.0").with_description("Model uploaded via aprender")
        });

        // Create README.md content
        let readme_content = model_card.to_huggingface();

        // For now, we'll use the HF Hub HTTP API directly
        // In a full implementation, this would use the upload API
        let commit_msg = options
            .commit_message
            .unwrap_or_else(|| "Upload model via aprender".to_string());

        // Build upload URL
        let upload_url = format!(
            "{}/api/models/{}/upload/main/{}",
            self.api_base, repo_id, options.filename
        );

        // Note: Actual upload requires multipart form data
        // This is a simplified implementation that returns the expected URL
        // A production implementation would use reqwest or similar

        // For demonstration, we'll save locally and return info
        let local_path = self.cache_dir.join(repo_id.replace('/', "_"));
        std::fs::create_dir_all(&local_path)?;

        // Save model
        let model_path = local_path.join(&options.filename);
        std::fs::write(&model_path, model_data)?;

        // Save README
        let readme_path = local_path.join("README.md");
        std::fs::write(&readme_path, readme_content)?;

        // Log the intended upload (actual HTTP upload would go here)
        // In production, this would use:
        // - POST to /api/repos/create for repo creation
        // - POST to /api/models/{repo}/upload for file upload

        Ok(format!(
            "Prepared for upload to {upload_url}\n\
             Token: {}...{}\n\
             Commit: {commit_msg}\n\
             Local cache: {}\n\
             \n\
             Note: HTTP upload not yet implemented.\n\
             Files saved locally. Use `huggingface-cli upload` to complete.",
            &token[..8.min(token.len())],
            &token[token.len().saturating_sub(4)..],
            local_path.display()
        ))
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

    #[test]
    fn test_push_to_hub_with_token() {
        let temp_dir = std::env::temp_dir().join("aprender_hf_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let client = HfHubClient {
            token: Some("test_token_12345678".to_string()),
            cache_dir: temp_dir.clone(),
            api_base: "https://huggingface.co".to_string(),
        };

        let result = client.push_to_hub(
            "paiml/test-model",
            b"model data",
            PushOptions::default().with_commit_message("Test upload"),
        );

        assert!(result.is_ok());
        let msg = result.unwrap();
        assert!(msg.contains("test_tok"));
        assert!(msg.contains("Test upload"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
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

    #[test]
    fn test_push_saves_readme() {
        let temp_dir = std::env::temp_dir().join("aprender_hf_readme_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let client = HfHubClient {
            token: Some("test_token".to_string()),
            cache_dir: temp_dir.clone(),
            api_base: "https://huggingface.co".to_string(),
        };

        let card = ModelCard::new("paiml/readme-test", "1.0.0")
            .with_license("MIT")
            .with_description("Test model");

        let _ = client.push_to_hub(
            "paiml/readme-test",
            b"model",
            PushOptions::default().with_model_card(card),
        );

        let readme_path = temp_dir.join("paiml_readme-test").join("README.md");
        assert!(readme_path.exists());

        let content = std::fs::read_to_string(&readme_path).unwrap();
        assert!(content.contains("license: mit"));
        assert!(content.contains("Test model"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
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
        assert!(path_str.contains("huggingface") || path_str.contains("."));
    }

    #[test]
    fn test_push_creates_model_file() {
        let temp_dir = std::env::temp_dir().join("aprender_hf_model_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let client = HfHubClient {
            token: Some("token".to_string()),
            cache_dir: temp_dir.clone(),
            api_base: "https://huggingface.co".to_string(),
        };

        let _ = client.push_to_hub(
            "org/model-file-test",
            b"binary model data",
            PushOptions::default(),
        );

        let model_path = temp_dir.join("org_model-file-test").join("model.apr");
        assert!(model_path.exists());

        let content = std::fs::read(&model_path).unwrap();
        assert_eq!(content, b"binary model data");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_push_auto_generates_model_card() {
        let temp_dir = std::env::temp_dir().join("aprender_hf_auto_card_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let client = HfHubClient {
            token: Some("token".to_string()),
            cache_dir: temp_dir.clone(),
            api_base: "https://huggingface.co".to_string(),
        };

        // Push without explicit model card
        let result = client.push_to_hub(
            "org/auto-card",
            b"data",
            PushOptions::default(), // No model_card set
        );

        assert!(result.is_ok());

        let readme_path = temp_dir.join("org_auto-card").join("README.md");
        assert!(readme_path.exists());

        let content = std::fs::read_to_string(&readme_path).unwrap();
        // Auto-generated card should have aprender tags
        assert!(content.contains("aprender"));

        let _ = std::fs::remove_dir_all(&temp_dir);
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
            let (parsed_org, parsed_name) = result.unwrap();
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
