//! Google Cloud Storage configuration

use serde::{Deserialize, Serialize};

/// Google Cloud Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCSConfig {
    /// GCS bucket name
    pub bucket: String,
    /// Object prefix
    pub prefix: String,
    /// Project ID
    pub project_id: Option<String>,
    /// Service account JSON key path
    pub service_account_key: Option<String>,
}

impl GCSConfig {
    /// Create a new GCS configuration
    pub fn new(bucket: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            prefix: String::new(),
            project_id: None,
            service_account_key: None,
        }
    }

    /// Set object prefix
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Set project ID
    pub fn with_project(mut self, project_id: &str) -> Self {
        self.project_id = Some(project_id.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcs_config_new() {
        let config = GCSConfig::new("my-bucket");
        assert_eq!(config.bucket, "my-bucket");
    }

    #[test]
    fn test_gcs_config_with_project() {
        let config = GCSConfig::new("bucket").with_project("my-project");
        assert_eq!(config.project_id, Some("my-project".to_string()));
    }

    #[test]
    fn test_gcs_config_with_prefix() {
        let config = GCSConfig::new("bucket").with_prefix("models/v1/");
        assert_eq!(config.prefix, "models/v1/");
    }

    #[test]
    fn test_gcs_config_serde() {
        let config = GCSConfig::new("bucket").with_prefix("artifacts/").with_project("my-project");

        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let parsed: GCSConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(config.bucket, parsed.bucket);
        assert_eq!(config.prefix, parsed.prefix);
        assert_eq!(config.project_id, parsed.project_id);
    }
}
