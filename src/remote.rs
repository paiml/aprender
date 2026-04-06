//! Remote registry client for Pacha
//!
//! Provides HTTP client for interacting with remote Pacha registries.
//! This module requires the `remote` feature to be enabled.
//!
//! # Example
//!
//! ```no_run
//! use pacha::remote::{RemoteRegistry, RegistryAuth};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let registry = RemoteRegistry::new("https://registry.example.com")
//!     .with_auth(RegistryAuth::Token("my-token".to_string()));
//!
//! // Pull a model
//! let model = registry.pull_model("llama3", "1.0.0").await?;
//!
//! // List available models
//! let models = registry.list_models().await?;
//! # Ok(())
//! # }
//! ```

use crate::error::{PachaError, Result};
use crate::model::{Model, ModelVersion};
use serde::{Deserialize, Serialize};

// ============================================================================
// API Types
// ============================================================================

/// Remote registry API response for listing models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// List of model names
    pub models: Vec<String>,
    /// Total count
    pub total: usize,
    /// Pagination cursor for next page
    pub next_cursor: Option<String>,
}

/// Remote registry API response for listing versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListVersionsResponse {
    /// Model name
    pub model: String,
    /// List of versions
    pub versions: Vec<VersionInfo>,
}

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Version string (semver)
    pub version: String,
    /// Content hash (BLAKE3)
    pub hash: String,
    /// Size in bytes
    pub size: u64,
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
    /// Stage (development, staging, production, archived)
    pub stage: String,
}

/// Remote registry API response for model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadataResponse {
    /// Model name
    pub name: String,
    /// Version
    pub version: String,
    /// Content hash
    pub hash: String,
    /// Size in bytes
    pub size: u64,
    /// Model card (description, metrics, etc.)
    pub card: Option<serde_json::Value>,
    /// Lineage information
    pub lineage: Option<LineageInfo>,
}

/// Lineage information from remote registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageInfo {
    /// Parent model (if derived)
    pub parent: Option<String>,
    /// Dataset used for training
    pub dataset: Option<String>,
    /// Recipe used
    pub recipe: Option<String>,
}

/// Push request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushRequest {
    /// Model name
    pub name: String,
    /// Version
    pub version: String,
    /// Content hash (BLAKE3)
    pub hash: String,
    /// Model card
    pub card: Option<serde_json::Value>,
}

/// Push response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushResponse {
    /// Upload URL for the artifact
    pub upload_url: String,
    /// Upload ID for tracking
    pub upload_id: String,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub enum RegistryAuth {
    /// No authentication
    None,
    /// Bearer token
    Token(String),
    /// Basic authentication
    Basic {
        /// Username
        username: String,
        /// Password
        password: String,
    },
    /// API key (header-based)
    ApiKey {
        /// Header name
        header: String,
        /// Key value
        key: String,
    },
}

impl Default for RegistryAuth {
    fn default() -> Self {
        Self::None
    }
}

// ============================================================================
// Remote Registry Client
// ============================================================================

/// Remote registry client
#[derive(Debug)]
pub struct RemoteRegistry {
    /// Base URL of the registry
    base_url: String,
    /// Authentication configuration
    auth: RegistryAuth,
    /// HTTP client (only available with `remote` feature)
    #[cfg(feature = "remote")]
    client: reqwest::Client,
}

impl RemoteRegistry {
    /// Create a new remote registry client
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();

        Self {
            base_url,
            auth: RegistryAuth::None,
            #[cfg(feature = "remote")]
            client: reqwest::Client::builder()
                .user_agent(concat!("pacha/", env!("CARGO_PKG_VERSION")))
                .connect_timeout(std::time::Duration::from_secs(30))
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Set authentication
    #[must_use]
    pub fn with_auth(mut self, auth: RegistryAuth) -> Self {
        self.auth = auth;
        self
    }

    /// Get the base URL
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Check if authentication is configured
    #[must_use]
    pub fn has_auth(&self) -> bool {
        !matches!(self.auth, RegistryAuth::None)
    }

    // -------------------------------------------------------------------------
    // Read Operations
    // -------------------------------------------------------------------------

    /// List all models in the registry
    #[cfg(feature = "remote")]
    pub async fn list_models(&self) -> Result<ListModelsResponse> {
        let url = format!("{}/api/v1/models", self.base_url);
        let response = self
            .build_request(reqwest::Method::GET, &url)
            .send()
            .await
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        self.handle_response(response).await
    }

    /// List all models (stub for non-remote builds)
    #[cfg(not(feature = "remote"))]
    pub async fn list_models(&self) -> Result<ListModelsResponse> {
        Err(PachaError::UnsupportedOperation {
            operation: "list_models".to_string(),
            reason: "Remote feature not enabled. Rebuild with --features remote".to_string(),
        })
    }

    /// List versions of a model
    #[cfg(feature = "remote")]
    pub async fn list_versions(&self, model: &str) -> Result<ListVersionsResponse> {
        let url = format!("{}/api/v1/models/{}/versions", self.base_url, model);
        let response = self
            .build_request(reqwest::Method::GET, &url)
            .send()
            .await
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        self.handle_response(response).await
    }

    /// List versions (stub for non-remote builds)
    #[cfg(not(feature = "remote"))]
    pub async fn list_versions(&self, _model: &str) -> Result<ListVersionsResponse> {
        Err(PachaError::UnsupportedOperation {
            operation: "list_versions".to_string(),
            reason: "Remote feature not enabled. Rebuild with --features remote".to_string(),
        })
    }

    /// Get model metadata
    #[cfg(feature = "remote")]
    pub async fn get_metadata(&self, model: &str, version: &str) -> Result<ModelMetadataResponse> {
        let url = format!("{}/api/v1/models/{}/versions/{}", self.base_url, model, version);
        let response = self
            .build_request(reqwest::Method::GET, &url)
            .send()
            .await
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        self.handle_response(response).await
    }

    /// Get metadata (stub for non-remote builds)
    #[cfg(not(feature = "remote"))]
    pub async fn get_metadata(
        &self,
        _model: &str,
        _version: &str,
    ) -> Result<ModelMetadataResponse> {
        Err(PachaError::UnsupportedOperation {
            operation: "get_metadata".to_string(),
            reason: "Remote feature not enabled. Rebuild with --features remote".to_string(),
        })
    }

    /// Pull model artifact data
    #[cfg(feature = "remote")]
    pub async fn pull_model(&self, model: &str, version: &str) -> Result<Vec<u8>> {
        let url =
            format!("{}/api/v1/models/{}/versions/{}/artifact", self.base_url, model, version);
        let response = self
            .build_request(reqwest::Method::GET, &url)
            .send()
            .await
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))
    }

    /// Pull model (stub for non-remote builds)
    #[cfg(not(feature = "remote"))]
    pub async fn pull_model(&self, _model: &str, _version: &str) -> Result<Vec<u8>> {
        Err(PachaError::UnsupportedOperation {
            operation: "pull_model".to_string(),
            reason: "Remote feature not enabled. Rebuild with --features remote".to_string(),
        })
    }

    // -------------------------------------------------------------------------
    // Write Operations
    // -------------------------------------------------------------------------

    /// Initiate a push operation
    #[cfg(feature = "remote")]
    pub async fn init_push(&self, request: &PushRequest) -> Result<PushResponse> {
        let url = format!("{}/api/v1/models/{}/versions", self.base_url, request.name);
        let response = self
            .build_request(reqwest::Method::POST, &url)
            .json(request)
            .send()
            .await
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        self.handle_response(response).await
    }

    /// Init push (stub for non-remote builds)
    #[cfg(not(feature = "remote"))]
    pub async fn init_push(&self, _request: &PushRequest) -> Result<PushResponse> {
        Err(PachaError::UnsupportedOperation {
            operation: "init_push".to_string(),
            reason: "Remote feature not enabled. Rebuild with --features remote".to_string(),
        })
    }

    /// Upload artifact data to the provided URL
    #[cfg(feature = "remote")]
    pub async fn upload_artifact(&self, upload_url: &str, data: Vec<u8>) -> Result<()> {
        let response = self
            .build_request(reqwest::Method::PUT, upload_url)
            .body(data)
            .send()
            .await
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        Ok(())
    }

    /// Upload artifact (stub for non-remote builds)
    #[cfg(not(feature = "remote"))]
    pub async fn upload_artifact(&self, _upload_url: &str, _data: Vec<u8>) -> Result<()> {
        Err(PachaError::UnsupportedOperation {
            operation: "upload_artifact".to_string(),
            reason: "Remote feature not enabled. Rebuild with --features remote".to_string(),
        })
    }

    /// Complete push operation (full workflow)
    #[cfg(feature = "remote")]
    pub async fn push_model(
        &self,
        name: &str,
        version: &ModelVersion,
        data: &[u8],
        card: Option<serde_json::Value>,
    ) -> Result<()> {
        let hash = blake3::hash(data).to_hex().to_string();

        let request =
            PushRequest { name: name.to_string(), version: version.to_string(), hash, card };

        let response = self.init_push(&request).await?;
        self.upload_artifact(&response.upload_url, data.to_vec()).await
    }

    /// Push model (stub for non-remote builds)
    #[cfg(not(feature = "remote"))]
    pub async fn push_model(
        &self,
        _name: &str,
        _version: &ModelVersion,
        _data: &[u8],
        _card: Option<serde_json::Value>,
    ) -> Result<()> {
        Err(PachaError::UnsupportedOperation {
            operation: "push_model".to_string(),
            reason: "Remote feature not enabled. Rebuild with --features remote".to_string(),
        })
    }

    // -------------------------------------------------------------------------
    // Internal Helpers
    // -------------------------------------------------------------------------

    #[cfg(feature = "remote")]
    fn build_request(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let mut request = self.client.request(method, url);

        match &self.auth {
            RegistryAuth::None => {}
            RegistryAuth::Token(token) => {
                request = request.bearer_auth(token);
            }
            RegistryAuth::Basic { username, password } => {
                request = request.basic_auth(username, Some(password));
            }
            RegistryAuth::ApiKey { header, key } => {
                request = request.header(header.as_str(), key.as_str());
            }
        }

        request
    }

    #[cfg(feature = "remote")]
    async fn handle_response<T: serde::de::DeserializeOwned>(
        &self,
        response: reqwest::Response,
    ) -> Result<T> {
        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        response.json().await.map_err(|e| {
            PachaError::Json(serde_json::Error::io(std::io::Error::other(e.to_string())))
        })
    }

    #[cfg(feature = "remote")]
    async fn handle_error_response(&self, response: reqwest::Response) -> PachaError {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if status == reqwest::StatusCode::NOT_FOUND {
            PachaError::NotFound {
                kind: "remote".to_string(),
                name: body,
                version: "unknown".to_string(),
            }
        } else if status == reqwest::StatusCode::UNAUTHORIZED
            || status == reqwest::StatusCode::FORBIDDEN
        {
            PachaError::Validation(format!("Authentication failed: {body}"))
        } else {
            PachaError::Io(std::io::Error::other(format!("HTTP {}: {}", status, body)))
        }
    }
}

// ============================================================================
// Resolver Integration
// ============================================================================

/// Pull a model from remote registry and optionally cache locally
pub async fn pull_to_local(
    remote: &RemoteRegistry,
    local: &crate::registry::Registry,
    model: &str,
    version: &str,
) -> Result<Model> {
    // Get metadata first
    let metadata = remote.get_metadata(model, version).await?;

    // Parse version
    let model_version = parse_version(&metadata.version)?;

    // Check if already cached locally
    if let Ok(local_model) = local.get_model(model, &model_version) {
        // Verify hash matches
        let local_artifact = local.get_model_artifact(model, &model_version)?;
        let local_hash = blake3::hash(&local_artifact).to_hex().to_string();

        if local_hash == metadata.hash {
            return Ok(local_model);
        }
        // Hash mismatch - need to re-pull
    }

    // Pull artifact
    let data = remote.pull_model(model, version).await?;

    // Verify hash
    let hash = blake3::hash(&data).to_hex().to_string();
    if hash != metadata.hash {
        return Err(PachaError::HashMismatch { expected: metadata.hash, actual: hash });
    }

    // Register locally
    let card = metadata
        .card
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_else(|| crate::model::ModelCard::new("Pulled from remote registry"));

    local.register_model(model, &model_version, &data, card)?;

    local.get_model(model, &model_version)
}

/// Push a local model to remote registry
pub async fn push_to_remote(
    local: &crate::registry::Registry,
    remote: &RemoteRegistry,
    model: &str,
    version: &ModelVersion,
) -> Result<()> {
    // Get local model and artifact
    let local_model = local.get_model(model, version)?;
    let data = local.get_model_artifact(model, version)?;

    // Convert card to JSON
    let card = serde_json::to_value(&local_model.card).ok();

    // Push to remote
    remote.push_model(model, version, &data, card).await
}

/// Parse version string into ModelVersion
fn parse_version(s: &str) -> Result<ModelVersion> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() == 3 {
        let major: u32 = parts[0].parse().map_err(|_| PachaError::InvalidVersion(s.to_string()))?;
        let minor: u32 = parts[1].parse().map_err(|_| PachaError::InvalidVersion(s.to_string()))?;
        let patch: u32 = parts[2].parse().map_err(|_| PachaError::InvalidVersion(s.to_string()))?;
        return Ok(ModelVersion::new(major, minor, patch));
    }
    Err(PachaError::InvalidVersion(s.to_string()))
}

// ============================================================================
// TESTS - EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // API Types Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_list_models_response_serialize() {
        let response = ListModelsResponse {
            models: vec!["llama3".to_string(), "mistral".to_string()],
            total: 2,
            next_cursor: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("llama3"));
        assert!(json.contains("mistral"));
    }

    #[test]
    fn test_list_models_response_deserialize() {
        let json = r#"{"models":["llama3"],"total":1,"next_cursor":null}"#;
        let response: ListModelsResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0], "llama3");
        assert_eq!(response.total, 1);
        assert!(response.next_cursor.is_none());
    }

    #[test]
    fn test_version_info_serialize() {
        let info = VersionInfo {
            version: "1.0.0".to_string(),
            hash: "abc123".to_string(),
            size: 1024,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            stage: "production".to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("1.0.0"));
        assert!(json.contains("abc123"));
    }

    #[test]
    fn test_version_info_deserialize() {
        let json = r#"{"version":"2.0.0","hash":"def456","size":2048,"created_at":"2024-06-01T00:00:00Z","stage":"staging"}"#;
        let info: VersionInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.version, "2.0.0");
        assert_eq!(info.hash, "def456");
        assert_eq!(info.size, 2048);
        assert_eq!(info.stage, "staging");
    }

    #[test]
    fn test_model_metadata_response() {
        let response = ModelMetadataResponse {
            name: "test-model".to_string(),
            version: "1.2.3".to_string(),
            hash: "hash123".to_string(),
            size: 4096,
            card: Some(serde_json::json!({"description": "Test model"})),
            lineage: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: ModelMetadataResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "test-model");
        assert_eq!(parsed.version, "1.2.3");
    }

    #[test]
    fn test_lineage_info() {
        let lineage = LineageInfo {
            parent: Some("base-model:1.0.0".to_string()),
            dataset: Some("training-data:1.0.0".to_string()),
            recipe: Some("fine-tune-recipe:1.0.0".to_string()),
        };

        let json = serde_json::to_string(&lineage).unwrap();
        assert!(json.contains("base-model"));
        assert!(json.contains("training-data"));
    }

    #[test]
    fn test_push_request() {
        let request = PushRequest {
            name: "new-model".to_string(),
            version: "0.1.0".to_string(),
            hash: "newhash".to_string(),
            card: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("new-model"));
        assert!(json.contains("0.1.0"));
    }

    #[test]
    fn test_push_response() {
        let json =
            r#"{"upload_url":"https://storage.example.com/upload/123","upload_id":"upload-123"}"#;
        let response: PushResponse = serde_json::from_str(json).unwrap();

        assert!(response.upload_url.contains("storage.example.com"));
        assert_eq!(response.upload_id, "upload-123");
    }

    // -------------------------------------------------------------------------
    // Authentication Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_auth_default() {
        let auth = RegistryAuth::default();
        assert!(matches!(auth, RegistryAuth::None));
    }

    #[test]
    fn test_registry_auth_token() {
        let auth = RegistryAuth::Token("my-token".to_string());
        assert!(matches!(auth, RegistryAuth::Token(_)));
    }

    #[test]
    fn test_registry_auth_basic() {
        let auth =
            RegistryAuth::Basic { username: "user".to_string(), password: "pass".to_string() };
        assert!(matches!(auth, RegistryAuth::Basic { .. }));
    }

    #[test]
    fn test_registry_auth_api_key() {
        let auth =
            RegistryAuth::ApiKey { header: "X-Api-Key".to_string(), key: "secret-key".to_string() };
        assert!(matches!(auth, RegistryAuth::ApiKey { .. }));
    }

    // -------------------------------------------------------------------------
    // RemoteRegistry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_remote_registry_new() {
        let registry = RemoteRegistry::new("https://registry.example.com");
        assert_eq!(registry.base_url(), "https://registry.example.com");
        assert!(!registry.has_auth());
    }

    #[test]
    fn test_remote_registry_trailing_slash() {
        let registry = RemoteRegistry::new("https://registry.example.com/");
        assert_eq!(registry.base_url(), "https://registry.example.com");
    }

    #[test]
    fn test_remote_registry_with_auth() {
        let registry = RemoteRegistry::new("https://registry.example.com")
            .with_auth(RegistryAuth::Token("token".to_string()));
        assert!(registry.has_auth());
    }

    #[test]
    fn test_remote_registry_no_auth() {
        let registry =
            RemoteRegistry::new("https://registry.example.com").with_auth(RegistryAuth::None);
        assert!(!registry.has_auth());
    }

    // -------------------------------------------------------------------------
    // Version Parsing Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_version_valid() {
        let v = parse_version("1.2.3").unwrap();
        assert_eq!(v, ModelVersion::new(1, 2, 3));
    }

    #[test]
    fn test_parse_version_zeros() {
        let v = parse_version("0.0.0").unwrap();
        assert_eq!(v, ModelVersion::new(0, 0, 0));
    }

    #[test]
    fn test_parse_version_large() {
        let v = parse_version("100.200.300").unwrap();
        assert_eq!(v, ModelVersion::new(100, 200, 300));
    }

    #[test]
    fn test_parse_version_invalid_format() {
        assert!(parse_version("1.2").is_err());
        assert!(parse_version("1").is_err());
        assert!(parse_version("1.2.3.4").is_err());
    }

    #[test]
    fn test_parse_version_non_numeric() {
        assert!(parse_version("a.b.c").is_err());
        assert!(parse_version("1.x.0").is_err());
    }

    // -------------------------------------------------------------------------
    // Serialization Round-Trip Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_list_versions_response_roundtrip() {
        let response = ListVersionsResponse {
            model: "test".to_string(),
            versions: vec![
                VersionInfo {
                    version: "1.0.0".to_string(),
                    hash: "hash1".to_string(),
                    size: 100,
                    created_at: "2024-01-01T00:00:00Z".to_string(),
                    stage: "production".to_string(),
                },
                VersionInfo {
                    version: "2.0.0".to_string(),
                    hash: "hash2".to_string(),
                    size: 200,
                    created_at: "2024-06-01T00:00:00Z".to_string(),
                    stage: "staging".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: ListVersionsResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.model, "test");
        assert_eq!(parsed.versions.len(), 2);
    }

    #[test]
    fn test_metadata_with_lineage_roundtrip() {
        let response = ModelMetadataResponse {
            name: "derived-model".to_string(),
            version: "1.0.0".to_string(),
            hash: "hash".to_string(),
            size: 1000,
            card: Some(serde_json::json!({"description": "A derived model"})),
            lineage: Some(LineageInfo {
                parent: Some("base:1.0.0".to_string()),
                dataset: Some("data:1.0.0".to_string()),
                recipe: None,
            }),
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: ModelMetadataResponse = serde_json::from_str(&json).unwrap();

        assert!(parsed.lineage.is_some());
        let lineage = parsed.lineage.unwrap();
        assert_eq!(lineage.parent.unwrap(), "base:1.0.0");
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_models_list() {
        let response = ListModelsResponse { models: vec![], total: 0, next_cursor: None };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: ListModelsResponse = serde_json::from_str(&json).unwrap();

        assert!(parsed.models.is_empty());
        assert_eq!(parsed.total, 0);
    }

    #[test]
    fn test_pagination_cursor() {
        let response = ListModelsResponse {
            models: vec!["model1".to_string()],
            total: 100,
            next_cursor: Some("cursor-abc".to_string()),
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: ListModelsResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.next_cursor.unwrap(), "cursor-abc");
    }

    #[test]
    fn test_push_request_with_card() {
        let request = PushRequest {
            name: "model".to_string(),
            version: "1.0.0".to_string(),
            hash: "hash".to_string(),
            card: Some(serde_json::json!({
                "description": "Test model",
                "metrics": {"accuracy": 0.95}
            })),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("accuracy"));
        assert!(json.contains("0.95"));
    }
}
