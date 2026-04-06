//! Artifact Record - content-addressable storage for run outputs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Artifact Record represents a stored artifact from a run.
///
/// Artifacts are stored using content-addressable storage (CAS),
/// where the `cas_hash` uniquely identifies the artifact content.
///
/// ## CAS Hash Format
///
/// The `cas_hash` follows the format: `algorithm:hex_digest`
///
/// Examples:
/// - `sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
/// - `blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactRecord {
    run_id: String,
    key: String,
    cas_hash: String,
    size_bytes: u64,
    created_at: DateTime<Utc>,
}

impl ArtifactRecord {
    /// Create a new artifact record.
    ///
    /// # Arguments
    ///
    /// * `run_id` - ID of the parent run
    /// * `key` - Artifact name/key (e.g., "model.pt", "checkpoint.ckpt")
    /// * `cas_hash` - Content-addressable hash (e.g., "sha256:abc123")
    /// * `size_bytes` - Size of the artifact in bytes
    ///
    /// # Returns
    ///
    /// A new `ArtifactRecord` with the current timestamp.
    #[must_use]
    pub fn new(
        run_id: impl Into<String>,
        key: impl Into<String>,
        cas_hash: impl Into<String>,
        size_bytes: u64,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            key: key.into(),
            cas_hash: cas_hash.into(),
            size_bytes,
            created_at: Utc::now(),
        }
    }

    /// Get the run ID.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Get the artifact key/name.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Get the content-addressable hash.
    #[must_use]
    pub fn cas_hash(&self) -> &str {
        &self.cas_hash
    }

    /// Get the artifact size in bytes.
    #[must_use]
    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Get the creation timestamp.
    #[must_use]
    pub const fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_record_new() {
        let artifact = ArtifactRecord::new("run-1", "model.pt", "sha256:abc123", 1000);
        assert_eq!(artifact.run_id(), "run-1");
        assert_eq!(artifact.key(), "model.pt");
        assert_eq!(artifact.cas_hash(), "sha256:abc123");
        assert_eq!(artifact.size_bytes(), 1000);
    }

    #[test]
    fn test_artifact_cas_hash_format() {
        let artifact = ArtifactRecord::new("run-1", "data.bin", "sha256:e3b0c44298fc1c14", 0);
        assert!(artifact.cas_hash().starts_with("sha256:"));
    }
}
