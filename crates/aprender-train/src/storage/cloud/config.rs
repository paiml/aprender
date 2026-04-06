//! Unified backend configuration

use crate::storage::cloud::azure::AzureConfig;
use crate::storage::cloud::error::Result;
use crate::storage::cloud::gcs::GCSConfig;
use crate::storage::cloud::local::LocalBackend;
use crate::storage::cloud::memory::InMemoryBackend;
use crate::storage::cloud::s3::{MockS3Backend, S3Config};
use crate::storage::cloud::traits::ArtifactBackend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Unified artifact backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendConfig {
    /// Local filesystem
    Local { path: PathBuf },
    /// In-memory (for testing)
    Memory,
    /// Amazon S3
    S3(S3Config),
    /// Azure Blob Storage
    Azure(AzureConfig),
    /// Google Cloud Storage
    GCS(GCSConfig),
}

impl BackendConfig {
    /// Create a local backend configuration
    pub fn local(path: PathBuf) -> Self {
        Self::Local { path }
    }

    /// Create an in-memory backend configuration
    pub fn memory() -> Self {
        Self::Memory
    }

    /// Create an S3 backend configuration
    pub fn s3(bucket: &str, prefix: &str) -> Self {
        Self::S3(S3Config::new(bucket, prefix))
    }

    /// Create a backend from this configuration
    pub fn build(&self) -> Result<Box<dyn ArtifactBackend>> {
        match self {
            Self::Local { path } => Ok(Box::new(LocalBackend::new_and_init(path.clone())?)),
            Self::Memory => Ok(Box::new(InMemoryBackend::new())),
            Self::S3(config) => Ok(Box::new(MockS3Backend::new(config.clone()))),
            Self::Azure(_config) => {
                // For now, return a mock (real implementation would use azure SDK)
                Ok(Box::new(InMemoryBackend::new()))
            }
            Self::GCS(_config) => {
                // For now, return a mock (real implementation would use GCS SDK)
                Ok(Box::new(InMemoryBackend::new()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_backend_config_local() {
        let config = BackendConfig::local(PathBuf::from("/tmp/artifacts"));
        match config {
            BackendConfig::Local { path } => {
                assert_eq!(path, PathBuf::from("/tmp/artifacts"));
            }
            _ => panic!("Expected Local"),
        }
    }

    #[test]
    fn test_backend_config_memory() {
        let config = BackendConfig::memory();
        assert!(matches!(config, BackendConfig::Memory));
    }

    #[test]
    fn test_backend_config_s3() {
        let config = BackendConfig::s3("bucket", "prefix");
        match config {
            BackendConfig::S3(c) => {
                assert_eq!(c.bucket, "bucket");
                assert_eq!(c.prefix, "prefix");
            }
            _ => panic!("Expected S3"),
        }
    }

    #[test]
    fn test_backend_config_build_memory() {
        let config = BackendConfig::memory();
        let backend = config.build().expect("builder should produce valid result");
        assert_eq!(backend.backend_type(), "memory");
    }

    #[test]
    fn test_backend_config_build_s3() {
        let config = BackendConfig::s3("bucket", "prefix");
        let backend = config.build().expect("builder should produce valid result");
        assert_eq!(backend.backend_type(), "s3");
    }

    #[test]
    fn test_backend_config_build_local() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let config = BackendConfig::local(tmp.path().to_path_buf());
        let backend = config.build().expect("builder should produce valid result");
        assert_eq!(backend.backend_type(), "local");
    }

    #[test]
    fn test_backend_config_build_azure() {
        let config = BackendConfig::Azure(AzureConfig::new("account", "container"));
        let backend = config.build().expect("builder should produce valid result");
        // Currently returns InMemoryBackend as placeholder
        assert_eq!(backend.backend_type(), "memory");
    }

    #[test]
    fn test_backend_config_build_gcs() {
        let config = BackendConfig::GCS(GCSConfig::new("bucket"));
        let backend = config.build().expect("builder should produce valid result");
        // Currently returns InMemoryBackend as placeholder
        assert_eq!(backend.backend_type(), "memory");
    }

    #[test]
    fn test_backend_config_serde() {
        let configs = vec![
            BackendConfig::local(PathBuf::from("/tmp/test")),
            BackendConfig::memory(),
            BackendConfig::s3("bucket", "prefix"),
            BackendConfig::Azure(AzureConfig::new("account", "container")),
            BackendConfig::GCS(GCSConfig::new("bucket")),
        ];

        for config in configs {
            let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
            let parsed: BackendConfig =
                serde_json::from_str(&json).expect("JSON deserialization should succeed");
            // Just verify it parses without panic
            let _ = parsed;
        }
    }
}

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_metadata_size_matches(
            name in "[a-zA-Z0-9_]{1,20}",
            size in 0u64..1_000_000
        ) {
            use crate::storage::cloud::metadata::ArtifactMetadata;
            let meta = ArtifactMetadata::new(&name, "hash", size);
            prop_assert_eq!(meta.size, size);
        }
    }
}
