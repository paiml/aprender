//! Pacha URI scheme support for model loading
//!
//! Enables direct model loading from the Pacha registry using URIs like:
//! - `pacha://model-name:version`
//! - `pacha://model-name` (latest version)
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::uri::{PachaUri, resolve_model_uri};
//!
//! let uri = PachaUri::parse("pacha://llama-7b:1.0.0")?;
//! let model_path = resolve_model_uri(&uri).await?;
//! ```
//!
//! ## Integration
//!
//! The Pacha URI scheme integrates with:
//! - Model registry for automatic metadata retrieval
//! - Lineage tracking for inference metrics
//! - Content addressing for integrity verification

#[cfg(feature = "registry")]
use std::path::PathBuf;

use crate::error::{RealizarError, Result};

/// Parsed Pacha URI
///
/// Represents a `pacha://model:version` URI for model loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PachaUri {
    /// Model name/identifier
    pub model: String,
    /// Version (None means latest)
    pub version: Option<String>,
}

impl PachaUri {
    /// Parse a Pacha URI string
    ///
    /// Supported formats:
    /// - `pacha://model-name:version`
    /// - `pacha://model-name` (latest version)
    ///
    /// # Errors
    ///
    /// Returns error if URI is malformed or uses unsupported scheme
    pub fn parse(uri: &str) -> Result<Self> {
        // Check for pacha:// scheme
        let path = uri
            .strip_prefix("pacha://")
            .ok_or_else(|| RealizarError::InvalidUri(format!("Expected pacha:// scheme: {uri}")))?;

        if path.is_empty() {
            return Err(RealizarError::InvalidUri(
                "Model name required after pacha://".to_string(),
            ));
        }

        // Split model:version
        let (model, version) = if let Some(colon_pos) = path.rfind(':') {
            let model = &path[..colon_pos];
            let version = &path[colon_pos + 1..];

            if model.is_empty() {
                return Err(RealizarError::InvalidUri(
                    "Model name cannot be empty".to_string(),
                ));
            }
            if version.is_empty() {
                return Err(RealizarError::InvalidUri(
                    "Version cannot be empty after colon".to_string(),
                ));
            }

            (model.to_string(), Some(version.to_string()))
        } else {
            (path.to_string(), None)
        };

        Ok(Self { model, version })
    }

    /// Check if this is a Pacha URI
    #[must_use]
    pub fn is_pacha_uri(uri: &str) -> bool {
        uri.starts_with("pacha://")
    }

    /// Convert back to URI string
    #[must_use]
    pub fn to_uri_string(&self) -> String {
        match &self.version {
            Some(v) => format!("pacha://{}:{v}", self.model),
            None => format!("pacha://{}", self.model),
        }
    }

    /// Get the version or "latest" if none specified
    #[must_use]
    pub fn version_or_latest(&self) -> &str {
        self.version.as_deref().unwrap_or("latest")
    }
}

impl std::fmt::Display for PachaUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uri_string())
    }
}

/// Model metadata retrieved from Pacha registry
#[derive(Debug, Clone)]
#[cfg(feature = "registry")]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Content hash (BLAKE3)
    pub content_hash: String,
    /// Signer public key (Ed25519) if signed
    pub signer_key: Option<String>,
    /// Local cached path
    pub local_path: PathBuf,
}

/// Resolver for Pacha URIs
///
/// Handles model resolution from the Pacha registry.
#[cfg(feature = "registry")]
pub struct PachaResolver {
    registry: pacha::Registry,
}

#[cfg(feature = "registry")]
impl PachaResolver {
    /// Create a new resolver with default registry
    ///
    /// # Errors
    ///
    /// Returns error if registry cannot be opened
    pub fn new() -> Result<Self> {
        let registry = pacha::Registry::open_default()
            .map_err(|e| RealizarError::RegistryError(format!("Failed to open registry: {e}")))?;
        Ok(Self { registry })
    }

    /// Create a resolver with a specific registry
    #[must_use]
    pub fn with_registry(registry: pacha::Registry) -> Self {
        Self { registry }
    }

    /// Resolve a Pacha URI to model metadata and artifact data
    ///
    /// This function:
    /// 1. Parses the Pacha URI
    /// 2. Retrieves model from registry
    /// 3. Returns the metadata and artifact data
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model not found in registry
    /// - Retrieval fails
    pub fn resolve(&self, uri: &PachaUri) -> Result<(ModelMetadata, Vec<u8>)> {
        use pacha::model::ModelVersion;

        // Parse version
        let version = match &uri.version {
            Some(v) => ModelVersion::parse(v)
                .map_err(|e| RealizarError::RegistryError(format!("Invalid version: {e}")))?,
            None => ModelVersion::new(0, 0, 0), // Will get latest
        };

        // Get model metadata from registry
        let model = self
            .registry
            .get_model(&uri.model, &version)
            .map_err(|e| RealizarError::ModelNotFound(format!("{}: {e}", uri.model)))?;

        // Get the artifact data
        let artifact = self
            .registry
            .get_model_artifact(&uri.model, &version)
            .map_err(|e| {
                RealizarError::RegistryError(format!("Failed to get model artifact: {e}"))
            })?;

        let metadata = ModelMetadata {
            name: uri.model.clone(),
            version: model.version.to_string(),
            content_hash: model.content_address.to_string(),
            signer_key: None, // Signing info retrieved separately via pacha::signing
            local_path: PathBuf::new(), // Artifact is returned directly, no file path
        };

        Ok((metadata, artifact))
    }

    /// Get model metadata without loading artifact
    ///
    /// # Errors
    ///
    /// Returns error if model not found
    pub fn get_metadata(&self, uri: &PachaUri) -> Result<ModelMetadata> {
        use pacha::model::ModelVersion;

        let version = match &uri.version {
            Some(v) => ModelVersion::parse(v)
                .map_err(|e| RealizarError::RegistryError(format!("Invalid version: {e}")))?,
            None => ModelVersion::new(0, 0, 0),
        };

        let model = self
            .registry
            .get_model(&uri.model, &version)
            .map_err(|e| RealizarError::ModelNotFound(format!("{}: {e}", uri.model)))?;

        Ok(ModelMetadata {
            name: uri.model.clone(),
            version: model.version.to_string(),
            content_hash: model.content_address.to_string(),
            signer_key: None, // Signing info retrieved separately via pacha::signing
            local_path: PathBuf::new(),
        })
    }
}

/// Lineage information for model tracing
///
/// Propagated to inference metrics for observability.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelLineage {
    /// Original Pacha URI
    pub uri: String,
    /// Model name
    pub model: String,
    /// Model version
    pub version: String,
    /// Content hash (BLAKE3) for integrity
    pub content_hash: String,
    /// Whether the model was verified (signature check)
    pub verified: bool,
    /// Timestamp when lineage was captured
    pub captured_at: u64,
}

impl ModelLineage {
    /// Create lineage from Pacha URI and metadata
    #[cfg(feature = "registry")]
    #[must_use]
    pub fn from_metadata(uri: &PachaUri, metadata: &ModelMetadata) -> Self {
        Self {
            uri: uri.to_uri_string(),
            model: metadata.name.clone(),
            version: metadata.version.clone(),
            content_hash: metadata.content_hash.clone(),
            verified: metadata.signer_key.is_some(),
            captured_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Create lineage without registry metadata
    #[must_use]
    pub fn from_uri(uri: &PachaUri) -> Self {
        Self {
            uri: uri.to_uri_string(),
            model: uri.model.clone(),
            version: uri.version.clone().unwrap_or_else(|| "unknown".to_string()),
            content_hash: String::new(),
            verified: false,
            captured_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_uri() {
        let uri = PachaUri::parse("pacha://llama-7b:1.0.0").expect("test");
        assert_eq!(uri.model, "llama-7b");
        assert_eq!(uri.version, Some("1.0.0".to_string()));
    }

    #[test]
    fn test_parse_uri_without_version() {
        let uri = PachaUri::parse("pacha://mistral-7b").expect("test");
        assert_eq!(uri.model, "mistral-7b");
        assert_eq!(uri.version, None);
    }

    #[test]
    fn test_parse_uri_with_semver() {
        let uri = PachaUri::parse("pacha://gpt2:2.0.0-beta.1").expect("test");
        assert_eq!(uri.model, "gpt2");
        assert_eq!(uri.version, Some("2.0.0-beta.1".to_string()));
    }

    #[test]
    fn test_parse_uri_with_org() {
        let uri = PachaUri::parse("pacha://paiml/trueno-llm:1.0").expect("test");
        assert_eq!(uri.model, "paiml/trueno-llm");
        assert_eq!(uri.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_invalid_scheme() {
        let result = PachaUri::parse("http://example.com/model");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_model() {
        let result = PachaUri::parse("pacha://");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_version() {
        let result = PachaUri::parse("pacha://model:");
        assert!(result.is_err());
    }

    #[test]
    fn test_is_pacha_uri() {
        assert!(PachaUri::is_pacha_uri("pacha://model:1.0"));
        assert!(PachaUri::is_pacha_uri("pacha://model"));
        assert!(!PachaUri::is_pacha_uri("http://example.com"));
        assert!(!PachaUri::is_pacha_uri("/path/to/model.gguf"));
    }

    #[test]
    fn test_to_uri_string() {
        let uri = PachaUri::parse("pacha://model:1.0").expect("test");
        assert_eq!(uri.to_uri_string(), "pacha://model:1.0");

        let uri_no_version = PachaUri::parse("pacha://model").expect("test");
        assert_eq!(uri_no_version.to_uri_string(), "pacha://model");
    }

    #[test]
    fn test_version_or_latest() {
        let uri = PachaUri::parse("pacha://model:2.0").expect("test");
        assert_eq!(uri.version_or_latest(), "2.0");

        let uri_no_version = PachaUri::parse("pacha://model").expect("test");
        assert_eq!(uri_no_version.version_or_latest(), "latest");
    }

    #[test]
    fn test_display() {
        let uri = PachaUri::parse("pacha://llama:1.0.0").expect("test");
        assert_eq!(format!("{uri}"), "pacha://llama:1.0.0");
    }

    #[test]
    fn test_lineage_from_uri() {
        let uri = PachaUri::parse("pacha://model:1.0").expect("test");
        let lineage = ModelLineage::from_uri(&uri);

        assert_eq!(lineage.uri, "pacha://model:1.0");
        assert_eq!(lineage.model, "model");
        assert_eq!(lineage.version, "1.0");
        assert!(!lineage.verified);
        assert!(lineage.captured_at > 0);
    }

    // =========================================================================
    // Coverage Tests
    // =========================================================================

    #[test]
    fn test_pacha_uri_clone_and_eq() {
        let uri1 = PachaUri::parse("pacha://model:1.0").expect("test");
        let uri2 = uri1.clone();
        assert_eq!(uri1, uri2);
    }

    #[test]
    fn test_lineage_from_uri_no_version() {
        let uri = PachaUri::parse("pacha://model-name").expect("test");
        let lineage = ModelLineage::from_uri(&uri);
        assert_eq!(lineage.version, "unknown");
    }

    #[test]
    fn test_parse_empty_model_error_msg() {
        let result = PachaUri::parse("pacha://");
        match result {
            Err(e) => assert!(e.to_string().contains("required")),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_parse_empty_model_name_before_colon() {
        let result = PachaUri::parse("pacha://:1.0");
        match result {
            Err(e) => assert!(e.to_string().contains("empty")),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_lineage_content_hash_empty() {
        let uri = PachaUri::parse("pacha://test").expect("test");
        let lineage = ModelLineage::from_uri(&uri);
        assert!(lineage.content_hash.is_empty());
    }

    #[test]
    fn test_model_lineage_serialize() {
        let uri = PachaUri::parse("pacha://model:1.0").expect("test");
        let lineage = ModelLineage::from_uri(&uri);
        let json = serde_json::to_string(&lineage).expect("test");
        assert!(json.contains("model"));
        assert!(json.contains("1.0"));
    }

    #[test]
    fn test_model_lineage_deserialize() {
        let json = r#"{"uri":"pacha://test:1.0","model":"test","version":"1.0","content_hash":"","verified":false,"captured_at":12345}"#;
        let lineage: ModelLineage = serde_json::from_str(json).expect("test");
        assert_eq!(lineage.model, "test");
        assert_eq!(lineage.version, "1.0");
    }

    #[test]
    fn test_pacha_uri_debug() {
        let uri = PachaUri::parse("pacha://model:1.0").expect("test");
        let debug = format!("{:?}", uri);
        assert!(debug.contains("PachaUri"));
        assert!(debug.contains("model"));
    }

    #[test]
    fn test_model_lineage_debug() {
        let uri = PachaUri::parse("pacha://model:1.0").expect("test");
        let lineage = ModelLineage::from_uri(&uri);
        let debug = format!("{:?}", lineage);
        assert!(debug.contains("ModelLineage"));
    }
}
