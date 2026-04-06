//! Artifact metadata types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Artifact metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    /// Original filename
    pub name: String,
    /// Content-addressable hash (SHA-256)
    pub hash: String,
    /// Size in bytes
    pub size: u64,
    /// Content type (MIME)
    pub content_type: Option<String>,
    /// Creation timestamp (Unix seconds)
    pub created_at: u64,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ArtifactMetadata {
    /// Create new artifact metadata
    pub fn new(name: &str, hash: &str, size: u64) -> Self {
        Self {
            name: name.to_string(),
            hash: hash.to_string(),
            size,
            content_type: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            metadata: HashMap::new(),
        }
    }

    /// Set content type
    pub fn with_content_type(mut self, content_type: &str) -> Self {
        self.content_type = Some(content_type.to_string());
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_metadata_new() {
        let meta = ArtifactMetadata::new("test.bin", "abc123", 1024);
        assert_eq!(meta.name, "test.bin");
        assert_eq!(meta.hash, "abc123");
        assert_eq!(meta.size, 1024);
    }

    #[test]
    fn test_artifact_metadata_with_content_type() {
        let meta = ArtifactMetadata::new("model.safetensors", "abc", 100)
            .with_content_type("application/octet-stream");
        assert_eq!(meta.content_type, Some("application/octet-stream".to_string()));
    }

    #[test]
    fn test_artifact_metadata_with_metadata() {
        let meta = ArtifactMetadata::new("model.bin", "abc123", 1024)
            .with_metadata("model_type", "bert")
            .with_metadata("version", "1.0");
        assert_eq!(meta.metadata.get("model_type"), Some(&"bert".to_string()));
        assert_eq!(meta.metadata.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_artifact_metadata_serde() {
        let meta = ArtifactMetadata::new("test.bin", "abc123", 1024)
            .with_content_type("application/octet-stream")
            .with_metadata("key", "value");

        let json = serde_json::to_string(&meta).expect("JSON serialization should succeed");
        let parsed: ArtifactMetadata =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(meta.name, parsed.name);
        assert_eq!(meta.hash, parsed.hash);
        assert_eq!(meta.size, parsed.size);
        assert_eq!(meta.content_type, parsed.content_type);
    }

    #[test]
    fn test_artifact_metadata_created_at_is_set() {
        let meta = ArtifactMetadata::new("test.bin", "hash", 100);
        assert!(meta.created_at > 0);
    }
}
