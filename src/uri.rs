//! URI scheme for model references
//!
//! Supports multiple URI schemes for model resolution:
//!
//! - `pacha://model-name:version` - Pacha registry (local or remote)
//! - `pacha://model-name:latest` - Latest version
//! - `pacha://model-name@sha256:abc123` - Content-addressed
//! - `pacha://model-name:production` - Stage alias
//! - `file://./model.gguf` - Local file
//! - `hf://meta-llama/Llama-3-8B` - HuggingFace Hub
//!
//! # Example
//!
//! ```
//! use pacha::uri::{ModelUri, UriScheme};
//!
//! let uri = ModelUri::parse("pacha://llama3:8b-q4").unwrap();
//! assert_eq!(uri.scheme, UriScheme::Pacha);
//! assert_eq!(uri.name, "llama3");
//! assert_eq!(uri.version.as_deref(), Some("8b-q4"));
//! ```

use crate::error::{PachaError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

/// URI scheme for model references
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UriScheme {
    /// Pacha registry (local or remote)
    Pacha,
    /// Local filesystem
    File,
    /// HuggingFace Hub
    HuggingFace,
}

impl fmt::Display for UriScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pacha => write!(f, "pacha"),
            Self::File => write!(f, "file"),
            Self::HuggingFace => write!(f, "hf"),
        }
    }
}

impl FromStr for UriScheme {
    type Err = PachaError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "pacha" => Ok(Self::Pacha),
            "file" => Ok(Self::File),
            "hf" | "huggingface" => Ok(Self::HuggingFace),
            _ => Err(PachaError::InvalidUri(format!("Unknown scheme: {s}"))),
        }
    }
}

/// Version reference type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VersionRef {
    /// Semantic version (e.g., "1.0.0")
    Version(String),
    /// Tag (e.g., "latest", "production", "8b-q4")
    Tag(String),
    /// Content hash (e.g., "sha256:abc123")
    Hash(String),
}

impl fmt::Display for VersionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version(v) | Self::Tag(v) => write!(f, "{v}"),
            Self::Hash(h) => write!(f, "@{h}"),
        }
    }
}

/// Parsed model URI
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelUri {
    /// URI scheme
    pub scheme: UriScheme,
    /// Model name or path
    pub name: String,
    /// Version reference (semantic version, tag, or hash)
    pub version: Option<String>,
    /// Content hash for content-addressed references
    pub hash: Option<String>,
    /// Remote registry host (if specified)
    pub host: Option<String>,
    /// File path within model (for HuggingFace URIs)
    pub path: Option<String>,
}

impl ModelUri {
    /// Parse a URI string into a `ModelUri`
    ///
    /// # Supported formats
    ///
    /// - `pacha://model:version`
    /// - `pacha://host/model:version`
    /// - `pacha://model@sha256:hash`
    /// - `file://path/to/model.gguf`
    /// - `hf://org/model`
    /// - `model:version` (assumes pacha://)
    /// - `./model.gguf` (assumes file://)
    pub fn parse(uri: &str) -> Result<Self> {
        let uri = uri.trim();

        // Parse scheme://rest first
        let (scheme, rest) = if let Some(idx) = uri.find("://") {
            let scheme_str = &uri[..idx];
            let rest = &uri[idx + 3..];
            (UriScheme::from_str(scheme_str)?, rest)
        } else if uri.starts_with("./") || uri.starts_with('/') {
            // Bare relative or absolute paths - assume file://
            return Ok(Self {
                scheme: UriScheme::File,
                name: uri.to_string(),
                version: None,
                hash: None,
                host: None,
                path: None,
            });
        } else if uri.ends_with(".gguf") || uri.ends_with(".safetensors") || uri.ends_with(".apr") {
            // Bare model files - assume file://
            return Ok(Self {
                scheme: UriScheme::File,
                name: uri.to_string(),
                version: None,
                hash: None,
                host: None,
                path: None,
            });
        } else if uri.contains(':') && !uri.contains('/') {
            // Bare model:version format - assume pacha
            (UriScheme::Pacha, uri)
        } else {
            return Err(PachaError::InvalidUri(format!("Cannot parse URI: {uri}")));
        };

        match scheme {
            UriScheme::File => Self::parse_file_uri(rest),
            UriScheme::HuggingFace => Self::parse_hf_uri(rest),
            UriScheme::Pacha => Self::parse_pacha_uri(rest),
        }
    }

    fn parse_file_uri(path: &str) -> Result<Self> {
        Ok(Self {
            scheme: UriScheme::File,
            name: path.to_string(),
            version: None,
            hash: None,
            host: None,
            path: None,
        })
    }

    fn parse_hf_uri(input: &str) -> Result<Self> {
        // Formats:
        // - hf://org/model
        // - hf://org/model:revision
        // - hf://org/model/path/to/file
        // - hf://org/model:revision/path/to/file

        // First, separate the model identifier from any path
        // The model id is always "org/model" (exactly two segments)
        let parts: Vec<&str> = input.splitn(3, '/').collect();

        if parts.len() < 2 {
            return Err(PachaError::InvalidUri(format!(
                "HuggingFace URI must have format org/model: {}",
                input
            )));
        }

        let org = parts[0];
        let model_and_rest = parts[1];

        // Check for version in model part (e.g., "model:revision")
        let (model, version) = if let Some(idx) = model_and_rest.find(':') {
            (&model_and_rest[..idx], Some(model_and_rest[idx + 1..].to_string()))
        } else {
            (model_and_rest, None)
        };

        // Build the model name (org/model)
        let name = format!("{org}/{model}");

        // Get file path if present (third segment and beyond)
        let file_path = if parts.len() > 2 { Some(parts[2].to_string()) } else { None };

        Ok(Self {
            scheme: UriScheme::HuggingFace,
            name,
            version,
            hash: None,
            host: None,
            path: file_path,
        })
    }

    fn parse_pacha_uri(rest: &str) -> Result<Self> {
        // Check for host: pacha://host/model:version
        let (host, model_part) = if let Some(idx) = rest.find('/') {
            (Some(rest[..idx].to_string()), &rest[idx + 1..])
        } else {
            (None, rest)
        };

        // Check for hash: model@sha256:abc123
        let (name_version, hash) = if let Some(idx) = model_part.find('@') {
            let hash_part = &model_part[idx + 1..];
            (&model_part[..idx], Some(hash_part.to_string()))
        } else {
            (model_part, None)
        };

        // Split name:version
        let (name, version) = if let Some(idx) = name_version.rfind(':') {
            (name_version[..idx].to_string(), Some(name_version[idx + 1..].to_string()))
        } else {
            (name_version.to_string(), None)
        };

        if name.is_empty() {
            return Err(PachaError::InvalidUri("Empty model name".to_string()));
        }

        Ok(Self { scheme: UriScheme::Pacha, name, version, hash, host, path: None })
    }

    /// Check if this is a local file reference
    pub fn is_local_file(&self) -> bool {
        self.scheme == UriScheme::File
    }

    /// Check if this is a remote reference
    pub fn is_remote(&self) -> bool {
        self.host.is_some() || self.scheme == UriScheme::HuggingFace
    }

    /// Get the local file path (if scheme is File)
    pub fn as_path(&self) -> Option<PathBuf> {
        if self.scheme == UriScheme::File {
            Some(PathBuf::from(&self.name))
        } else {
            None
        }
    }

    /// Get version or default to "latest"
    pub fn version_or_latest(&self) -> &str {
        self.version.as_deref().unwrap_or("latest")
    }
}

impl fmt::Display for ModelUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}://", self.scheme)?;

        if let Some(ref host) = self.host {
            write!(f, "{host}/")?;
        }

        write!(f, "{}", self.name)?;

        if let Some(ref hash) = self.hash {
            write!(f, "@{hash}")?;
        } else if let Some(ref version) = self.version {
            write!(f, ":{version}")?;
        }

        if let Some(ref path) = self.path {
            write!(f, "/{path}")?;
        }

        Ok(())
    }
}

impl FromStr for ModelUri {
    type Err = PachaError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

// ============================================================================
// TESTS - EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // UriScheme Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_uri_scheme_from_str() {
        assert_eq!(UriScheme::from_str("pacha").unwrap(), UriScheme::Pacha);
        assert_eq!(UriScheme::from_str("PACHA").unwrap(), UriScheme::Pacha);
        assert_eq!(UriScheme::from_str("file").unwrap(), UriScheme::File);
        assert_eq!(UriScheme::from_str("hf").unwrap(), UriScheme::HuggingFace);
        assert_eq!(UriScheme::from_str("huggingface").unwrap(), UriScheme::HuggingFace);
    }

    #[test]
    fn test_uri_scheme_from_str_invalid() {
        assert!(UriScheme::from_str("unknown").is_err());
        assert!(UriScheme::from_str("").is_err());
    }

    #[test]
    fn test_uri_scheme_display() {
        assert_eq!(UriScheme::Pacha.to_string(), "pacha");
        assert_eq!(UriScheme::File.to_string(), "file");
        assert_eq!(UriScheme::HuggingFace.to_string(), "hf");
    }

    // -------------------------------------------------------------------------
    // ModelUri Pacha Scheme Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_pacha_simple() {
        let uri = ModelUri::parse("pacha://llama3:8b").unwrap();
        assert_eq!(uri.scheme, UriScheme::Pacha);
        assert_eq!(uri.name, "llama3");
        assert_eq!(uri.version.as_deref(), Some("8b"));
        assert!(uri.host.is_none());
        assert!(uri.hash.is_none());
    }

    #[test]
    fn test_parse_pacha_with_host() {
        let uri = ModelUri::parse("pacha://registry.example.com/llama3:1.0.0").unwrap();
        assert_eq!(uri.scheme, UriScheme::Pacha);
        assert_eq!(uri.host.as_deref(), Some("registry.example.com"));
        assert_eq!(uri.name, "llama3");
        assert_eq!(uri.version.as_deref(), Some("1.0.0"));
    }

    #[test]
    fn test_parse_pacha_with_hash() {
        let uri = ModelUri::parse("pacha://llama3@sha256:abc123def").unwrap();
        assert_eq!(uri.scheme, UriScheme::Pacha);
        assert_eq!(uri.name, "llama3");
        assert_eq!(uri.hash.as_deref(), Some("sha256:abc123def"));
        assert!(uri.version.is_none());
    }

    #[test]
    fn test_parse_pacha_no_version() {
        let uri = ModelUri::parse("pacha://llama3").unwrap();
        assert_eq!(uri.name, "llama3");
        assert!(uri.version.is_none());
        assert_eq!(uri.version_or_latest(), "latest");
    }

    #[test]
    fn test_parse_pacha_stage_tag() {
        let uri = ModelUri::parse("pacha://fraud-detector:production").unwrap();
        assert_eq!(uri.name, "fraud-detector");
        assert_eq!(uri.version.as_deref(), Some("production"));
    }

    #[test]
    fn test_parse_bare_model_version() {
        let uri = ModelUri::parse("llama3:8b-q4").unwrap();
        assert_eq!(uri.scheme, UriScheme::Pacha);
        assert_eq!(uri.name, "llama3");
        assert_eq!(uri.version.as_deref(), Some("8b-q4"));
    }

    // -------------------------------------------------------------------------
    // ModelUri File Scheme Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_file_uri() {
        let uri = ModelUri::parse("file://./model.gguf").unwrap();
        assert_eq!(uri.scheme, UriScheme::File);
        assert_eq!(uri.name, "./model.gguf");
        assert!(uri.is_local_file());
    }

    #[test]
    fn test_parse_file_absolute() {
        let uri = ModelUri::parse("file:///home/user/model.gguf").unwrap();
        assert_eq!(uri.scheme, UriScheme::File);
        assert_eq!(uri.name, "/home/user/model.gguf");
    }

    #[test]
    fn test_parse_bare_relative_path() {
        let uri = ModelUri::parse("./models/llama.gguf").unwrap();
        assert_eq!(uri.scheme, UriScheme::File);
        assert_eq!(uri.name, "./models/llama.gguf");
    }

    #[test]
    fn test_parse_bare_absolute_path() {
        let uri = ModelUri::parse("/opt/models/llama.gguf").unwrap();
        assert_eq!(uri.scheme, UriScheme::File);
        assert_eq!(uri.name, "/opt/models/llama.gguf");
    }

    #[test]
    fn test_parse_bare_gguf_file() {
        let uri = ModelUri::parse("model.gguf").unwrap();
        assert_eq!(uri.scheme, UriScheme::File);
        assert_eq!(uri.name, "model.gguf");
    }

    #[test]
    fn test_as_path() {
        let uri = ModelUri::parse("file://./model.gguf").unwrap();
        assert_eq!(uri.as_path(), Some(PathBuf::from("./model.gguf")));

        let uri = ModelUri::parse("pacha://llama3:8b").unwrap();
        assert_eq!(uri.as_path(), None);
    }

    // -------------------------------------------------------------------------
    // ModelUri HuggingFace Scheme Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_hf_uri() {
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B").unwrap();
        assert_eq!(uri.scheme, UriScheme::HuggingFace);
        assert_eq!(uri.name, "meta-llama/Llama-3-8B");
        assert!(uri.version.is_none());
        assert!(uri.is_remote());
    }

    #[test]
    fn test_parse_hf_uri_with_revision() {
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B:main").unwrap();
        assert_eq!(uri.scheme, UriScheme::HuggingFace);
        assert_eq!(uri.name, "meta-llama/Llama-3-8B");
        assert_eq!(uri.version.as_deref(), Some("main"));
    }

    #[test]
    fn test_parse_hf_uri_with_path() {
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B/config.json").unwrap();
        assert_eq!(uri.scheme, UriScheme::HuggingFace);
        assert_eq!(uri.name, "meta-llama/Llama-3-8B");
        assert_eq!(uri.path.as_deref(), Some("config.json"));
        assert!(uri.version.is_none());
    }

    #[test]
    fn test_parse_hf_uri_with_revision_and_path() {
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B:v2.0/model.safetensors").unwrap();
        assert_eq!(uri.scheme, UriScheme::HuggingFace);
        assert_eq!(uri.name, "meta-llama/Llama-3-8B");
        assert_eq!(uri.version.as_deref(), Some("v2.0"));
        assert_eq!(uri.path.as_deref(), Some("model.safetensors"));
    }

    #[test]
    fn test_parse_hf_uri_invalid_format() {
        // Missing org
        assert!(ModelUri::parse("hf://model").is_err());
    }

    // -------------------------------------------------------------------------
    // ModelUri Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_pacha() {
        let uri = ModelUri::parse("pacha://llama3:8b").unwrap();
        assert_eq!(uri.to_string(), "pacha://llama3:8b");
    }

    #[test]
    fn test_display_pacha_with_host() {
        let uri = ModelUri::parse("pacha://registry.example.com/llama3:1.0.0").unwrap();
        assert_eq!(uri.to_string(), "pacha://registry.example.com/llama3:1.0.0");
    }

    #[test]
    fn test_display_pacha_with_hash() {
        let uri = ModelUri::parse("pacha://llama3@sha256:abc123").unwrap();
        assert_eq!(uri.to_string(), "pacha://llama3@sha256:abc123");
    }

    #[test]
    fn test_display_file() {
        let uri = ModelUri::parse("file://./model.gguf").unwrap();
        assert_eq!(uri.to_string(), "file://./model.gguf");
    }

    #[test]
    fn test_display_hf() {
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B").unwrap();
        assert_eq!(uri.to_string(), "hf://meta-llama/Llama-3-8B");
    }

    #[test]
    fn test_display_hf_with_path() {
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B/config.json").unwrap();
        assert_eq!(uri.to_string(), "hf://meta-llama/Llama-3-8B/config.json");
    }

    #[test]
    fn test_display_hf_with_revision_and_path() {
        let uri = ModelUri::parse("hf://meta-llama/Llama-3-8B:v2.0/model.safetensors").unwrap();
        assert_eq!(uri.to_string(), "hf://meta-llama/Llama-3-8B:v2.0/model.safetensors");
    }

    // -------------------------------------------------------------------------
    // ModelUri Error Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_empty_name() {
        assert!(ModelUri::parse("pacha://:8b").is_err());
    }

    #[test]
    fn test_parse_unknown_scheme() {
        assert!(ModelUri::parse("unknown://model").is_err());
    }

    // -------------------------------------------------------------------------
    // Roundtrip Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_roundtrip_pacha() {
        let original = "pacha://llama3:8b-q4";
        let uri = ModelUri::parse(original).unwrap();
        assert_eq!(uri.to_string(), original);
    }

    #[test]
    fn test_roundtrip_file() {
        let original = "file:///opt/models/llama.gguf";
        let uri = ModelUri::parse(original).unwrap();
        assert_eq!(uri.to_string(), original);
    }

    #[test]
    fn test_roundtrip_hf() {
        let original = "hf://meta-llama/Llama-3-8B:main";
        let uri = ModelUri::parse(original).unwrap();
        assert_eq!(uri.to_string(), original);
    }

    // -------------------------------------------------------------------------
    // FromStr Trait Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_str_trait() {
        let uri: ModelUri = "pacha://llama3:8b".parse().unwrap();
        assert_eq!(uri.name, "llama3");
    }

    // -------------------------------------------------------------------------
    // Property Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file() {
        assert!(ModelUri::parse("file://./model.gguf").unwrap().is_local_file());
        assert!(ModelUri::parse("./model.gguf").unwrap().is_local_file());
        assert!(!ModelUri::parse("pacha://llama3:8b").unwrap().is_local_file());
        assert!(!ModelUri::parse("hf://meta-llama/Llama-3").unwrap().is_local_file());
    }

    #[test]
    fn test_is_remote() {
        assert!(ModelUri::parse("hf://meta-llama/Llama-3").unwrap().is_remote());
        assert!(ModelUri::parse("pacha://registry.example.com/llama3:8b").unwrap().is_remote());
        assert!(!ModelUri::parse("pacha://llama3:8b").unwrap().is_remote());
        assert!(!ModelUri::parse("file://./model.gguf").unwrap().is_remote());
    }
}
