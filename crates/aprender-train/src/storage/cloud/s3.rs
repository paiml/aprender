//! S3 backend configuration and mock implementation

use crate::storage::cloud::error::Result;
use crate::storage::cloud::memory::InMemoryBackend;
use crate::storage::cloud::metadata::ArtifactMetadata;
use crate::storage::cloud::traits::ArtifactBackend;
use serde::{Deserialize, Serialize};

/// S3 backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Config {
    /// S3 bucket name
    pub bucket: String,
    /// Key prefix within bucket
    pub prefix: String,
    /// AWS region (e.g., "us-east-1")
    pub region: Option<String>,
    /// Custom endpoint (for MinIO, R2, etc.)
    pub endpoint: Option<String>,
    /// Access key ID (if not using IAM role)
    pub access_key_id: Option<String>,
    /// Secret access key (if not using IAM role)
    pub secret_access_key: Option<String>,
}

impl S3Config {
    /// Create a new S3 configuration
    pub fn new(bucket: &str, prefix: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            region: None,
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
        }
    }

    /// Set region
    pub fn with_region(mut self, region: &str) -> Self {
        self.region = Some(region.to_string());
        self
    }

    /// Set custom endpoint (for S3-compatible services)
    pub fn with_endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.to_string());
        self
    }

    /// Set credentials
    pub fn with_credentials(mut self, access_key_id: &str, secret_access_key: &str) -> Self {
        self.access_key_id = Some(access_key_id.to_string());
        self.secret_access_key = Some(secret_access_key.to_string());
        self
    }

    /// Get the full key path for a hash
    pub fn key_for_hash(&self, hash: &str) -> String {
        if self.prefix.is_empty() {
            hash.to_string()
        } else {
            format!("{}/{}", self.prefix.trim_end_matches('/'), hash)
        }
    }
}

/// Mock S3 backend for testing (simulates S3 behavior in memory)
#[derive(Debug)]
pub struct MockS3Backend {
    config: S3Config,
    inner: InMemoryBackend,
}

impl MockS3Backend {
    /// Create a new mock S3 backend
    pub fn new(config: S3Config) -> Self {
        Self { config, inner: InMemoryBackend::new() }
    }

    /// Get the configuration
    pub fn config(&self) -> &S3Config {
        &self.config
    }
}

impl ArtifactBackend for MockS3Backend {
    fn put(&self, name: &str, data: &[u8]) -> Result<String> {
        self.inner.put(name, data)
    }

    fn get(&self, hash: &str) -> Result<Vec<u8>> {
        self.inner.get(hash)
    }

    fn exists(&self, hash: &str) -> Result<bool> {
        self.inner.exists(hash)
    }

    fn delete(&self, hash: &str) -> Result<()> {
        self.inner.delete(hash)
    }

    fn get_metadata(&self, hash: &str) -> Result<ArtifactMetadata> {
        self.inner.get_metadata(hash)
    }

    fn list(&self) -> Result<Vec<ArtifactMetadata>> {
        self.inner.list()
    }

    fn backend_type(&self) -> &'static str {
        "s3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_config_new() {
        let config = S3Config::new("my-bucket", "artifacts/");
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.prefix, "artifacts/");
    }

    #[test]
    fn test_s3_config_with_region() {
        let config = S3Config::new("bucket", "prefix").with_region("us-west-2");
        assert_eq!(config.region, Some("us-west-2".to_string()));
    }

    #[test]
    fn test_s3_config_with_endpoint() {
        let config = S3Config::new("bucket", "prefix").with_endpoint("http://minio:9000");
        assert_eq!(config.endpoint, Some("http://minio:9000".to_string()));
    }

    #[test]
    fn test_s3_config_key_for_hash() {
        let config = S3Config::new("bucket", "artifacts");
        assert_eq!(config.key_for_hash("abc123"), "artifacts/abc123");

        let config = S3Config::new("bucket", "");
        assert_eq!(config.key_for_hash("abc123"), "abc123");
    }

    #[test]
    fn test_mock_s3_backend_put_get() {
        let config = S3Config::new("test-bucket", "prefix");
        let backend = MockS3Backend::new(config);

        let data = b"s3 test data";
        let hash = backend.put("file.bin", data).expect("operation should succeed");

        let retrieved = backend.get(&hash).expect("key should exist");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_mock_s3_backend_type() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);
        assert_eq!(backend.backend_type(), "s3");
    }

    #[test]
    fn test_s3_config_with_credentials() {
        let config = S3Config::new("bucket", "prefix").with_credentials("access_key", "secret_key");
        assert_eq!(config.access_key_id, Some("access_key".to_string()));
        assert_eq!(config.secret_access_key, Some("secret_key".to_string()));
    }

    #[test]
    fn test_s3_config_key_for_hash_with_trailing_slash() {
        let config = S3Config::new("bucket", "artifacts/");
        assert_eq!(config.key_for_hash("abc123"), "artifacts/abc123");
    }

    #[test]
    fn test_mock_s3_backend_exists() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);

        let hash = backend.put("file.bin", b"data").expect("operation should succeed");
        assert!(backend.exists(&hash).expect("operation should succeed"));
        assert!(!backend.exists("nonexistent").expect("operation should succeed"));
    }

    #[test]
    fn test_mock_s3_backend_delete() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);

        let hash = backend.put("file.bin", b"data").expect("operation should succeed");
        backend.delete(&hash).expect("operation should succeed");
        assert!(!backend.exists(&hash).expect("operation should succeed"));
    }

    #[test]
    fn test_mock_s3_backend_get_metadata() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);

        let hash = backend.put("model.bin", b"model data").expect("operation should succeed");
        let meta = backend.get_metadata(&hash).expect("operation should succeed");
        assert_eq!(meta.name, "model.bin");
    }

    #[test]
    fn test_mock_s3_backend_list() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);

        backend.put("file1.bin", b"data1").expect("operation should succeed");
        backend.put("file2.bin", b"data2").expect("operation should succeed");

        let list = backend.list().expect("operation should succeed");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_mock_s3_backend_config() {
        let config = S3Config::new("test-bucket", "models/").with_region("us-east-1");
        let backend = MockS3Backend::new(config);

        assert_eq!(backend.config().bucket, "test-bucket");
        assert_eq!(backend.config().prefix, "models/");
        assert_eq!(backend.config().region, Some("us-east-1".to_string()));
    }

    #[test]
    fn test_s3_config_serde() {
        let config = S3Config::new("bucket", "prefix")
            .with_region("us-west-2")
            .with_endpoint("http://localhost:9000")
            .with_credentials("key", "secret");

        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let parsed: S3Config =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(config.bucket, parsed.bucket);
        assert_eq!(config.region, parsed.region);
        assert_eq!(config.endpoint, parsed.endpoint);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_s3_key_contains_hash(
            prefix in "[a-zA-Z0-9/]{0,20}",
            hash in "[a-f0-9]{64}"
        ) {
            let config = S3Config::new("bucket", &prefix);
            let key = config.key_for_hash(&hash);
            prop_assert!(key.contains(&hash));
        }
    }
}
