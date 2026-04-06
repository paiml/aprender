//! URI resolver for model references
//!
//! Resolves `ModelUri` to actual model data from:
//! - Local files (file://)
//! - Pacha registry (pacha://)
//! - HuggingFace Hub (hf://) - future work
//!
//! # Example
//!
//! ```no_run
//! use pacha::resolver::ModelResolver;
//! use pacha::uri::ModelUri;
//!
//! let resolver = ModelResolver::new_default().unwrap();
//!
//! // Resolve from local file
//! let uri = ModelUri::parse("./model.gguf").unwrap();
//! let data = resolver.resolve(&uri).unwrap();
//!
//! // Resolve from registry
//! let uri = ModelUri::parse("pacha://llama3:1.0.0").unwrap();
//! let data = resolver.resolve(&uri).unwrap();
//! ```

use crate::error::{PachaError, Result};
use crate::model::{Model, ModelVersion};
use crate::registry::{Registry, RegistryConfig};
use crate::remote::RegistryAuth;
#[cfg(feature = "remote")]
use crate::remote::RemoteRegistry;
use crate::uri::{ModelUri, UriScheme};
use std::fs;
use std::path::Path;

/// Resolution result containing model data and metadata
#[derive(Debug)]
pub struct ResolvedModel {
    /// The model data (binary)
    pub data: Vec<u8>,
    /// Source of the model
    pub source: ModelSource,
    /// Model metadata (if from registry)
    pub model: Option<Model>,
}

/// Source of a resolved model
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// Local file system
    LocalFile(String),
    /// Pacha registry (local)
    PachaLocal {
        /// Model name
        name: String,
        /// Model version
        version: String,
    },
    /// Pacha registry (remote)
    PachaRemote {
        /// Remote registry host
        host: String,
        /// Model name
        name: String,
        /// Model version
        version: String,
    },
    /// HuggingFace Hub
    HuggingFace {
        /// Repository ID (e.g., meta-llama/Llama-3-8B)
        repo_id: String,
        /// Git revision (branch, tag, or commit)
        revision: Option<String>,
    },
}

/// Model resolver that handles URI resolution to model data
pub struct ModelResolver {
    /// Local registry
    registry: Option<Registry>,
    /// Remote registry authentication (for pacha:// URIs with hosts)
    remote_auth: Option<RegistryAuth>,
}

impl ModelResolver {
    /// Create a resolver with the default registry location (~/.pacha)
    pub fn new_default() -> Result<Self> {
        let registry = Registry::open_default().ok();
        Ok(Self { registry, remote_auth: None })
    }

    /// Create a resolver with a custom registry path
    pub fn new(registry_path: impl AsRef<Path>) -> Result<Self> {
        let config = RegistryConfig::new(registry_path);
        let registry = Registry::open(config).ok();
        Ok(Self { registry, remote_auth: None })
    }

    /// Create a resolver without a registry (file-only mode)
    #[must_use]
    pub fn file_only() -> Self {
        Self { registry: None, remote_auth: None }
    }

    /// Set remote authentication for resolving pacha:// URIs with hosts
    #[must_use]
    pub fn with_remote_auth(mut self, auth: RegistryAuth) -> Self {
        self.remote_auth = Some(auth);
        self
    }

    /// Check if registry is available
    #[must_use]
    pub fn has_registry(&self) -> bool {
        self.registry.is_some()
    }

    /// Check if remote authentication is configured
    #[must_use]
    pub fn has_remote_auth(&self) -> bool {
        self.remote_auth.is_some()
    }

    /// Resolve a URI to model data
    pub fn resolve(&self, uri: &ModelUri) -> Result<ResolvedModel> {
        match uri.scheme {
            UriScheme::File => self.resolve_file(uri),
            UriScheme::Pacha => self.resolve_pacha(uri),
            UriScheme::HuggingFace => self.resolve_huggingface(uri),
        }
    }

    /// Resolve a URI string to model data
    pub fn resolve_str(&self, uri: &str) -> Result<ResolvedModel> {
        let parsed = ModelUri::parse(uri)?;
        self.resolve(&parsed)
    }

    /// Check if a URI can be resolved (exists)
    pub fn exists(&self, uri: &ModelUri) -> bool {
        match uri.scheme {
            UriScheme::File => uri.as_path().map_or(false, |p| p.exists()),
            UriScheme::Pacha => {
                if uri.is_remote() {
                    // Remote check not implemented
                    false
                } else if let Some(ref registry) = self.registry {
                    let version = uri.version.as_deref().unwrap_or("latest");
                    if let Ok(version) = parse_version(version) {
                        registry.get_model(&uri.name, &version).is_ok()
                    } else {
                        // Try as tag - for now just check any version exists
                        registry.list_model_versions(&uri.name).map_or(false, |v| !v.is_empty())
                    }
                } else {
                    false
                }
            }
            UriScheme::HuggingFace => {
                // HuggingFace check not implemented
                false
            }
        }
    }

    fn resolve_file(&self, uri: &ModelUri) -> Result<ResolvedModel> {
        let path = uri
            .as_path()
            .ok_or_else(|| PachaError::InvalidUri("File URI has no path".to_string()))?;

        if !path.exists() {
            return Err(PachaError::NotFound {
                kind: "file".to_string(),
                name: path.display().to_string(),
                version: "N/A".to_string(),
            });
        }

        let data = fs::read(&path).map_err(|e| {
            PachaError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read {}: {}", path.display(), e),
            ))
        })?;

        Ok(ResolvedModel {
            data,
            source: ModelSource::LocalFile(path.display().to_string()),
            model: None,
        })
    }

    fn resolve_pacha(&self, uri: &ModelUri) -> Result<ResolvedModel> {
        // Check for remote registry
        if uri.is_remote() {
            return self.resolve_pacha_remote(uri);
        }

        // Local registry resolution
        let registry = self
            .registry
            .as_ref()
            .ok_or_else(|| PachaError::NotInitialized(std::path::PathBuf::from("~/.pacha")))?;

        // Parse version
        let version_str = uri.version.as_deref().unwrap_or("latest");

        // Handle "latest" by getting the latest version
        let version = if version_str == "latest" {
            let versions = registry.list_model_versions(&uri.name)?;
            if versions.is_empty() {
                return Err(PachaError::NotFound {
                    kind: "model".to_string(),
                    name: uri.name.clone(),
                    version: "latest".to_string(),
                });
            }
            // Get the latest version (assume versions are sorted)
            versions.into_iter().max().ok_or_else(|| PachaError::NotFound {
                kind: "model".to_string(),
                name: uri.name.clone(),
                version: "latest".to_string(),
            })?
        } else {
            parse_version(version_str)?
        };

        // Get model and artifact
        let model = registry.get_model(&uri.name, &version)?;
        let data = registry.get_model_artifact(&uri.name, &version)?;

        Ok(ResolvedModel {
            data,
            source: ModelSource::PachaLocal {
                name: uri.name.clone(),
                version: version.to_string(),
            },
            model: Some(model),
        })
    }

    #[cfg(feature = "remote")]
    fn resolve_pacha_remote(&self, uri: &ModelUri) -> Result<ResolvedModel> {
        let host = uri
            .host
            .as_ref()
            .ok_or_else(|| PachaError::InvalidUri("Remote URI missing host".to_string()))?;

        let version = uri.version.as_deref().unwrap_or("latest");
        // Host may include port (e.g., "registry.example.com:8080")
        let base_url = format!("https://{host}");

        // Create remote client
        let mut remote = RemoteRegistry::new(&base_url);
        if let Some(ref auth) = self.remote_auth {
            remote = remote.with_auth(auth.clone());
        }

        // Use blocking runtime
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        // Pull model
        let data = rt.block_on(remote.pull_model(&uri.name, version))?;

        // Optionally cache to local registry
        if let Some(ref registry) = self.registry {
            let model_version =
                parse_version(version).unwrap_or_else(|_| ModelVersion::new(0, 0, 0));
            let _ = registry.register_model(
                &uri.name,
                &model_version,
                &data,
                crate::model::ModelCard::new(&format!("Pulled from {host}")),
            );
        }

        Ok(ResolvedModel {
            data,
            source: ModelSource::PachaRemote {
                host: host.clone(),
                name: uri.name.clone(),
                version: version.to_string(),
            },
            model: None,
        })
    }

    #[cfg(not(feature = "remote"))]
    fn resolve_pacha_remote(&self, uri: &ModelUri) -> Result<ResolvedModel> {
        let host = uri
            .host
            .as_ref()
            .ok_or_else(|| PachaError::InvalidUri("Remote URI missing host".to_string()))?;

        Err(PachaError::UnsupportedOperation {
            operation: "remote_registry".to_string(),
            reason: format!(
                "Remote feature not enabled. Rebuild with --features remote. Host: {}",
                host
            ),
        })
    }

    #[cfg(feature = "remote")]
    fn resolve_huggingface(&self, uri: &ModelUri) -> Result<ResolvedModel> {
        // Parse HuggingFace URI: hf://org/model or hf://org/model@revision
        let (repo_id, revision) = if uri.name.contains('@') {
            let parts: Vec<&str> = uri.name.splitn(2, '@').collect();
            (parts[0].to_string(), parts.get(1).map(|s| s.to_string()))
        } else {
            (uri.name.clone(), uri.version.clone())
        };

        let revision = revision.as_deref().unwrap_or("main");

        // Determine the file to download (default to model.safetensors or config.json)
        let filename = uri.path.as_deref().unwrap_or("model.safetensors");

        // Build HuggingFace URL
        let url = format!("https://huggingface.co/{}/resolve/{}/{}", repo_id, revision, filename);

        // Use blocking runtime for HTTP
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        let client = reqwest::Client::builder()
            .user_agent(concat!("pacha/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

        let data = rt.block_on(async {
            let response = client
                .get(&url)
                .send()
                .await
                .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))?;

            if !response.status().is_success() {
                return Err(PachaError::NotFound {
                    kind: "huggingface".to_string(),
                    name: repo_id.clone(),
                    version: revision.to_string(),
                });
            }

            response
                .bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| PachaError::Io(std::io::Error::other(e.to_string())))
        })?;

        // Optionally cache to local registry
        if let Some(ref registry) = self.registry {
            // Use repo name as model name, revision as version
            let model_name = repo_id.replace('/', "-");
            let model_version =
                parse_version(revision).unwrap_or_else(|_| ModelVersion::new(0, 0, 0));
            let _ = registry.register_model(
                &model_name,
                &model_version,
                &data,
                crate::model::ModelCard::new(&format!("Downloaded from HuggingFace: {repo_id}")),
            );
        }

        Ok(ResolvedModel {
            data,
            source: ModelSource::HuggingFace { repo_id, revision: Some(revision.to_string()) },
            model: None,
        })
    }

    #[cfg(not(feature = "remote"))]
    fn resolve_huggingface(&self, uri: &ModelUri) -> Result<ResolvedModel> {
        Err(PachaError::UnsupportedOperation {
            operation: "huggingface".to_string(),
            reason: format!("HuggingFace Hub requires --features remote. Model: {}", uri.name),
        })
    }

    /// List all models in the local registry
    pub fn list_models(&self) -> Result<Vec<String>> {
        let registry = self
            .registry
            .as_ref()
            .ok_or_else(|| PachaError::NotInitialized(std::path::PathBuf::from("~/.pacha")))?;
        registry.list_models()
    }

    /// List versions of a model
    pub fn list_versions(&self, model_name: &str) -> Result<Vec<ModelVersion>> {
        let registry = self
            .registry
            .as_ref()
            .ok_or_else(|| PachaError::NotInitialized(std::path::PathBuf::from("~/.pacha")))?;
        registry.list_model_versions(model_name)
    }
}

/// Parse a version string into a ModelVersion
fn parse_version(s: &str) -> Result<ModelVersion> {
    // Try semantic version first (x.y.z)
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() == 3 {
        let major: u32 = parts[0]
            .parse()
            .map_err(|_| PachaError::InvalidUri(format!("Invalid version: {s}")))?;
        let minor: u32 = parts[1]
            .parse()
            .map_err(|_| PachaError::InvalidUri(format!("Invalid version: {s}")))?;
        let patch: u32 = parts[2]
            .parse()
            .map_err(|_| PachaError::InvalidUri(format!("Invalid version: {s}")))?;
        return Ok(ModelVersion::new(major, minor, patch));
    }

    // Single number -> assume x.0.0
    if let Ok(major) = s.parse::<u32>() {
        return Ok(ModelVersion::new(major, 0, 0));
    }

    Err(PachaError::InvalidUri(format!("Cannot parse version: {s}. Expected format: x.y.z")))
}

// ============================================================================
// TESTS - EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelCard;
    use std::io::Write;
    use tempfile::TempDir;

    // -------------------------------------------------------------------------
    // Setup Helpers
    // -------------------------------------------------------------------------

    fn setup_registry() -> (TempDir, ModelResolver) {
        let dir = TempDir::new().unwrap();
        let config = RegistryConfig::new(dir.path());
        let registry = Registry::open(config).unwrap();

        // Register a test model
        registry
            .register_model(
                "test-model",
                &ModelVersion::new(1, 0, 0),
                b"model data v1.0.0",
                ModelCard::new("Test model v1"),
            )
            .unwrap();

        registry
            .register_model(
                "test-model",
                &ModelVersion::new(1, 1, 0),
                b"model data v1.1.0",
                ModelCard::new("Test model v1.1"),
            )
            .unwrap();

        let resolver = ModelResolver::new(dir.path()).unwrap();
        (dir, resolver)
    }

    fn create_temp_file(content: &[u8]) -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(content).unwrap();
        (dir, path)
    }

    // -------------------------------------------------------------------------
    // File Resolution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_file() {
        let (_dir, path) = create_temp_file(b"GGUF model data");
        let resolver = ModelResolver::file_only();

        let uri = ModelUri::parse(&format!("file://{}", path.display())).unwrap();
        let resolved = resolver.resolve(&uri).unwrap();

        assert_eq!(resolved.data, b"GGUF model data");
        assert!(matches!(resolved.source, ModelSource::LocalFile(_)));
        assert!(resolved.model.is_none());
    }

    #[test]
    fn test_resolve_bare_path() {
        let (_dir, path) = create_temp_file(b"model content");
        let resolver = ModelResolver::file_only();

        let uri = ModelUri::parse(path.to_str().unwrap()).unwrap();
        let resolved = resolver.resolve(&uri).unwrap();

        assert_eq!(resolved.data, b"model content");
    }

    #[test]
    fn test_resolve_nonexistent_file() {
        let resolver = ModelResolver::file_only();
        let uri = ModelUri::parse("file:///nonexistent/model.gguf").unwrap();

        let result = resolver.resolve(&uri);
        assert!(matches!(result, Err(PachaError::NotFound { .. })));
    }

    #[test]
    fn test_exists_file() {
        let (_dir, path) = create_temp_file(b"data");
        let resolver = ModelResolver::file_only();

        let uri = ModelUri::parse(path.to_str().unwrap()).unwrap();
        assert!(resolver.exists(&uri));

        let uri = ModelUri::parse("file:///nonexistent.gguf").unwrap();
        assert!(!resolver.exists(&uri));
    }

    // -------------------------------------------------------------------------
    // Pacha Registry Resolution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_pacha_with_version() {
        let (_dir, resolver) = setup_registry();

        let uri = ModelUri::parse("pacha://test-model:1.0.0").unwrap();
        let resolved = resolver.resolve(&uri).unwrap();

        assert_eq!(resolved.data, b"model data v1.0.0");
        assert!(matches!(
            resolved.source,
            ModelSource::PachaLocal { ref name, ref version }
            if name == "test-model" && version == "1.0.0"
        ));
        assert!(resolved.model.is_some());
    }

    #[test]
    fn test_resolve_pacha_latest() {
        let (_dir, resolver) = setup_registry();

        let uri = ModelUri::parse("pacha://test-model:latest").unwrap();
        let resolved = resolver.resolve(&uri).unwrap();

        // Should get v1.1.0 (the latest)
        assert_eq!(resolved.data, b"model data v1.1.0");
    }

    #[test]
    fn test_resolve_pacha_no_version() {
        let (_dir, resolver) = setup_registry();

        let uri = ModelUri::parse("pacha://test-model").unwrap();
        let resolved = resolver.resolve(&uri).unwrap();

        // Should get latest
        assert_eq!(resolved.data, b"model data v1.1.0");
    }

    #[test]
    fn test_resolve_pacha_not_found() {
        let (_dir, resolver) = setup_registry();

        let uri = ModelUri::parse("pacha://nonexistent:1.0.0").unwrap();
        let result = resolver.resolve(&uri);

        assert!(matches!(result, Err(PachaError::NotFound { .. })));
    }

    #[test]
    fn test_resolve_pacha_no_registry() {
        let resolver = ModelResolver::file_only();

        let uri = ModelUri::parse("pacha://test-model:1.0.0").unwrap();
        let result = resolver.resolve(&uri);

        assert!(matches!(result, Err(PachaError::NotInitialized(_))));
    }

    #[test]
    fn test_exists_pacha() {
        let (_dir, resolver) = setup_registry();

        let uri = ModelUri::parse("pacha://test-model:1.0.0").unwrap();
        assert!(resolver.exists(&uri));

        let uri = ModelUri::parse("pacha://nonexistent:1.0.0").unwrap();
        assert!(!resolver.exists(&uri));
    }

    // -------------------------------------------------------------------------
    // Remote Pacha Tests
    // -------------------------------------------------------------------------

    #[test]
    #[cfg(not(feature = "remote"))]
    fn test_resolve_pacha_remote_not_implemented() {
        let (_dir, resolver) = setup_registry();

        let uri = ModelUri::parse("pacha://registry.example.com/model:1.0.0").unwrap();
        let result = resolver.resolve(&uri);

        assert!(matches!(result, Err(PachaError::UnsupportedOperation { .. })));
    }

    #[test]
    #[cfg(feature = "remote")]
    fn test_resolve_pacha_remote_connection_error() {
        let (_dir, resolver) = setup_registry();

        // Remote resolution to a non-existent server should fail with IO error
        let uri = ModelUri::parse("pacha://nonexistent.invalid/model:1.0.0").unwrap();
        let result = resolver.resolve(&uri);

        // Should fail with network error (connection refused or DNS failure)
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // HuggingFace Tests
    // -------------------------------------------------------------------------

    #[test]
    #[cfg(not(feature = "remote"))]
    fn test_resolve_huggingface_not_implemented() {
        let resolver = ModelResolver::file_only();

        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B").unwrap();
        let result = resolver.resolve(&uri);

        assert!(matches!(result, Err(PachaError::UnsupportedOperation { .. })));
    }

    #[test]
    #[cfg(feature = "remote")]
    fn test_resolve_huggingface_nonexistent_repo() {
        let resolver = ModelResolver::file_only();

        // Try to resolve a definitely nonexistent repo
        let uri = ModelUri::parse("hf://nonexistent-user-12345/nonexistent-model-67890").unwrap();
        let result = resolver.resolve(&uri);

        // Should fail with NotFound (404 from HuggingFace)
        assert!(result.is_err());
    }

    #[test]
    fn test_huggingface_uri_parsing() {
        // Basic HuggingFace URI
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B").unwrap();
        assert_eq!(uri.name, "meta-llama/Llama-3-8B");
        assert_eq!(uri.scheme, UriScheme::HuggingFace);

        // With revision in version
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B:main").unwrap();
        assert_eq!(uri.name, "meta-llama/Llama-3-8B");
        assert_eq!(uri.version, Some("main".to_string()));
    }

    #[test]
    fn test_huggingface_uri_with_path() {
        // HuggingFace URI with specific file path
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B/config.json").unwrap();
        assert_eq!(uri.name, "meta-llama/Llama-3-8B");
        assert_eq!(uri.path, Some("config.json".to_string()));
    }

    #[test]
    fn test_model_source_huggingface_clone() {
        let source = ModelSource::HuggingFace {
            repo_id: "meta-llama/Llama-3-8B".to_string(),
            revision: Some("main".to_string()),
        };
        let cloned = source.clone();
        assert_eq!(source, cloned);
    }

    #[test]
    fn test_model_source_huggingface_without_revision() {
        let source =
            ModelSource::HuggingFace { repo_id: "google/gemma-7b".to_string(), revision: None };
        assert!(matches!(source, ModelSource::HuggingFace { revision: None, .. }));
    }

    #[test]
    fn test_exists_huggingface() {
        let resolver = ModelResolver::file_only();

        // HuggingFace existence check is not implemented (returns false)
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B").unwrap();
        assert!(!resolver.exists(&uri));
    }

    // -------------------------------------------------------------------------
    // resolve_str Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_str() {
        let (_dir, path) = create_temp_file(b"test data");
        let resolver = ModelResolver::file_only();

        let resolved = resolver.resolve_str(path.to_str().unwrap()).unwrap();
        assert_eq!(resolved.data, b"test data");
    }

    #[test]
    fn test_resolve_str_invalid() {
        let resolver = ModelResolver::file_only();
        let result = resolver.resolve_str("invalid://uri");
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // List Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_list_models() {
        let (_dir, resolver) = setup_registry();

        let models = resolver.list_models().unwrap();
        assert!(models.contains(&"test-model".to_string()));
    }

    #[test]
    fn test_list_versions() {
        let (_dir, resolver) = setup_registry();

        let versions = resolver.list_versions("test-model").unwrap();
        assert_eq!(versions.len(), 2);
    }

    #[test]
    fn test_list_models_no_registry() {
        let resolver = ModelResolver::file_only();
        let result = resolver.list_models();
        assert!(matches!(result, Err(PachaError::NotInitialized(_))));
    }

    // -------------------------------------------------------------------------
    // Version Parsing Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_version_semver() {
        let v = parse_version("1.2.3").unwrap();
        assert_eq!(v, ModelVersion::new(1, 2, 3));
    }

    #[test]
    fn test_parse_version_single() {
        let v = parse_version("2").unwrap();
        assert_eq!(v, ModelVersion::new(2, 0, 0));
    }

    #[test]
    fn test_parse_version_invalid() {
        assert!(parse_version("invalid").is_err());
        assert!(parse_version("1.2").is_err());
        assert!(parse_version("a.b.c").is_err());
    }

    // -------------------------------------------------------------------------
    // ModelSource Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_source_equality() {
        let s1 = ModelSource::LocalFile("/path/to/model".to_string());
        let s2 = ModelSource::LocalFile("/path/to/model".to_string());
        let s3 = ModelSource::LocalFile("/other/path".to_string());

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_model_source_pacha_local() {
        let source =
            ModelSource::PachaLocal { name: "llama3".to_string(), version: "8b".to_string() };
        assert!(matches!(source, ModelSource::PachaLocal { .. }));
    }

    // -------------------------------------------------------------------------
    // has_registry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_has_registry() {
        let (_dir, resolver) = setup_registry();
        assert!(resolver.has_registry());

        let resolver = ModelResolver::file_only();
        assert!(!resolver.has_registry());
    }

    // -------------------------------------------------------------------------
    // Remote Auth Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_with_remote_auth() {
        let resolver = ModelResolver::file_only()
            .with_remote_auth(RegistryAuth::Token("test-token".to_string()));

        assert!(resolver.has_remote_auth());
    }

    #[test]
    fn test_without_remote_auth() {
        let resolver = ModelResolver::file_only();
        assert!(!resolver.has_remote_auth());
    }

    #[test]
    fn test_remote_auth_basic() {
        let resolver = ModelResolver::file_only().with_remote_auth(RegistryAuth::Basic {
            username: "user".to_string(),
            password: "pass".to_string(),
        });

        assert!(resolver.has_remote_auth());
    }

    #[test]
    fn test_remote_auth_api_key() {
        let resolver = ModelResolver::file_only().with_remote_auth(RegistryAuth::ApiKey {
            header: "X-Api-Key".to_string(),
            key: "secret".to_string(),
        });

        assert!(resolver.has_remote_auth());
    }

    // -------------------------------------------------------------------------
    // ModelSource Remote Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_source_pacha_remote() {
        let source = ModelSource::PachaRemote {
            host: "registry.example.com".to_string(),
            name: "llama3".to_string(),
            version: "1.0.0".to_string(),
        };

        assert!(matches!(source, ModelSource::PachaRemote { .. }));
    }

    #[test]
    fn test_model_source_huggingface() {
        let source = ModelSource::HuggingFace {
            repo_id: "meta-llama/Llama-3-8B".to_string(),
            revision: Some("main".to_string()),
        };

        assert!(matches!(source, ModelSource::HuggingFace { .. }));
    }
}
