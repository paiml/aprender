//! Artifact backend trait and utilities

use crate::storage::cloud::error::Result;
use crate::storage::cloud::metadata::ArtifactMetadata;
use sha2::{Digest, Sha256};

/// Trait for artifact storage backends
pub trait ArtifactBackend: Send + Sync {
    /// Store an artifact and return its content-addressable hash
    fn put(&self, name: &str, data: &[u8]) -> Result<String>;

    /// Retrieve an artifact by hash
    fn get(&self, hash: &str) -> Result<Vec<u8>>;

    /// Check if an artifact exists
    fn exists(&self, hash: &str) -> Result<bool>;

    /// Delete an artifact by hash
    fn delete(&self, hash: &str) -> Result<()>;

    /// Get artifact metadata
    fn get_metadata(&self, hash: &str) -> Result<ArtifactMetadata>;

    /// List all artifacts
    fn list(&self) -> Result<Vec<ArtifactMetadata>>;

    /// Get backend type name
    fn backend_type(&self) -> &'static str;
}

/// Compute SHA-256 hash of data
pub fn compute_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash() {
        let data = b"hello world";
        let hash = compute_hash(data);
        assert_eq!(hash.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
        assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let data = b"test data";
        let hash1 = compute_hash(data);
        let hash2 = compute_hash(data);
        assert_eq!(hash1, hash2);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_hash_deterministic(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let hash1 = compute_hash(&data);
            let hash2 = compute_hash(&data);
            prop_assert_eq!(hash1, hash2);
        }

        #[test]
        fn prop_hash_length_constant(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let hash = compute_hash(&data);
            prop_assert_eq!(hash.len(), 64);
        }
    }
}
