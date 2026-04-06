//! Registry index format and metadata structures.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Registry index containing all dataset information.
///
/// The index is stored as JSON and contains metadata for all datasets
/// in the registry.
///
/// # Example JSON
///
/// ```json
/// {
///   "version": "1.0",
///   "datasets": [
///     {
///       "name": "mnist",
///       "versions": ["1.0.0", "1.0.1"],
///       "latest": "1.0.1",
///       "size_bytes": 11490000,
///       "num_rows": 70000,
///       "schema": { ... },
///       "metadata": { ... }
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryIndex {
    /// Index format version.
    pub version: String,
    /// List of all datasets in the registry.
    pub datasets: Vec<DatasetInfo>,
}

impl RegistryIndex {
    /// Creates a new empty registry index.
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            datasets: Vec::new(),
        }
    }

    /// Returns the number of datasets in the index.
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    /// Returns true if the index contains no datasets.
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    /// Finds a dataset by name.
    pub fn find(&self, name: &str) -> Option<&DatasetInfo> {
        self.datasets.iter().find(|d| d.name == name)
    }

    /// Finds a dataset by name (mutable).
    pub fn find_mut(&mut self, name: &str) -> Option<&mut DatasetInfo> {
        self.datasets.iter_mut().find(|d| d.name == name)
    }
}

impl Default for RegistryIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a dataset in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Unique name of the dataset.
    pub name: String,
    /// List of available versions.
    pub versions: Vec<String>,
    /// Latest version tag.
    pub latest: String,
    /// Total size in bytes.
    pub size_bytes: u64,
    /// Number of rows in the dataset.
    pub num_rows: usize,
    /// Arrow schema as JSON.
    pub schema: Value,
    /// Dataset metadata.
    pub metadata: DatasetMetadata,
}

impl DatasetInfo {
    /// Checks if a specific version exists.
    pub fn has_version(&self, version: &str) -> bool {
        self.versions.contains(&version.to_string())
    }

    /// Returns the number of versions available.
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }
}

/// Metadata describing a dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Human-readable description.
    pub description: String,
    /// License identifier (e.g., "MIT", "Apache-2.0", "CC-BY-4.0").
    pub license: String,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Original source URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Citation information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation: Option<String>,
    /// SHA-256 hash of the dataset content (hex string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl DatasetMetadata {
    /// Creates a new metadata builder.
    pub fn builder() -> DatasetMetadataBuilder {
        DatasetMetadataBuilder::default()
    }

    /// Creates metadata with just a description.
    pub fn with_description(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            ..Default::default()
        }
    }
}

/// Builder for constructing DatasetMetadata.
#[derive(Debug, Default)]
pub struct DatasetMetadataBuilder {
    description: String,
    license: String,
    tags: Vec<String>,
    source: Option<String>,
    citation: Option<String>,
    sha256: Option<String>,
}

impl DatasetMetadataBuilder {
    /// Sets the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the license.
    #[must_use]
    pub fn license(mut self, license: impl Into<String>) -> Self {
        self.license = license.into();
        self
    }

    /// Adds a tag.
    #[must_use]
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Sets multiple tags.
    #[must_use]
    pub fn tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the source URL.
    #[must_use]
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Sets the citation.
    #[must_use]
    pub fn citation(mut self, citation: impl Into<String>) -> Self {
        self.citation = Some(citation.into());
        self
    }

    /// Sets the SHA-256 hash for data provenance.
    #[must_use]
    pub fn sha256(mut self, hash: impl Into<String>) -> Self {
        self.sha256 = Some(hash.into());
        self
    }

    /// Builds the metadata.
    pub fn build(self) -> DatasetMetadata {
        DatasetMetadata {
            description: self.description,
            license: self.license,
            tags: self.tags,
            source: self.source,
            citation: self.citation,
            sha256: self.sha256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_index_new() {
        let index = RegistryIndex::new();
        assert_eq!(index.version, "1.0");
        assert!(index.is_empty());
    }

    #[test]
    fn test_registry_index_find() {
        let mut index = RegistryIndex::new();
        index.datasets.push(DatasetInfo {
            name: "test".to_string(),
            versions: vec!["1.0.0".to_string()],
            latest: "1.0.0".to_string(),
            size_bytes: 1000,
            num_rows: 100,
            schema: serde_json::json!({}),
            metadata: DatasetMetadata::default(),
        });

        assert!(index.find("test").is_some());
        assert!(index.find("nonexistent").is_none());
    }

    #[test]
    fn test_dataset_info_has_version() {
        let info = DatasetInfo {
            name: "test".to_string(),
            versions: vec!["1.0.0".to_string(), "2.0.0".to_string()],
            latest: "2.0.0".to_string(),
            size_bytes: 1000,
            num_rows: 100,
            schema: serde_json::json!({}),
            metadata: DatasetMetadata::default(),
        };

        assert!(info.has_version("1.0.0"));
        assert!(info.has_version("2.0.0"));
        assert!(!info.has_version("3.0.0"));
        assert_eq!(info.version_count(), 2);
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = DatasetMetadata::builder()
            .description("A test dataset")
            .license("MIT")
            .tag("test")
            .tag("example")
            .source("https://example.com")
            .build();

        assert_eq!(metadata.description, "A test dataset");
        assert_eq!(metadata.license, "MIT");
        assert_eq!(metadata.tags, vec!["test", "example"]);
        assert_eq!(metadata.source, Some("https://example.com".to_string()));
        assert!(metadata.citation.is_none());
    }

    #[test]
    fn test_metadata_with_description() {
        let metadata = DatasetMetadata::with_description("Simple description");
        assert_eq!(metadata.description, "Simple description");
        assert!(metadata.license.is_empty());
    }

    #[test]
    fn test_registry_index_serialization() {
        let mut index = RegistryIndex::new();
        index.datasets.push(DatasetInfo {
            name: "test".to_string(),
            versions: vec!["1.0.0".to_string()],
            latest: "1.0.0".to_string(),
            size_bytes: 1000,
            num_rows: 100,
            schema: serde_json::json!({"fields": []}),
            metadata: DatasetMetadata::builder()
                .description("Test dataset")
                .license("MIT")
                .build(),
        });

        let json = serde_json::to_string(&index);
        assert!(json.is_ok());

        let parsed: Result<RegistryIndex, _> =
            serde_json::from_str(&json.ok().unwrap_or_else(|| panic!("Should serialize")));
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_registry_index_len() {
        let mut index = RegistryIndex::new();
        assert_eq!(index.len(), 0);

        index.datasets.push(DatasetInfo {
            name: "test".to_string(),
            versions: vec![],
            latest: String::new(),
            size_bytes: 0,
            num_rows: 0,
            schema: serde_json::json!({}),
            metadata: DatasetMetadata::default(),
        });
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_metadata_builder_tags() {
        let metadata = DatasetMetadata::builder().tags(["a", "b", "c"]).build();

        assert_eq!(metadata.tags, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_registry_index_find_mut() {
        let mut index = RegistryIndex::new();
        index.datasets.push(DatasetInfo {
            name: "test".to_string(),
            versions: vec!["1.0.0".to_string()],
            latest: "1.0.0".to_string(),
            size_bytes: 1000,
            num_rows: 100,
            schema: serde_json::json!({}),
            metadata: DatasetMetadata::default(),
        });

        let found = index.find_mut("test");
        assert!(found.is_some());
        found.unwrap().size_bytes = 2000;
        assert_eq!(index.find("test").unwrap().size_bytes, 2000);

        assert!(index.find_mut("nonexistent").is_none());
    }

    #[test]
    fn test_registry_index_default() {
        let index = RegistryIndex::default();
        assert_eq!(index.version, "1.0");
        assert!(index.is_empty());
    }

    #[test]
    fn test_registry_index_clone() {
        let mut index = RegistryIndex::new();
        index.datasets.push(DatasetInfo {
            name: "test".to_string(),
            versions: vec!["1.0.0".to_string()],
            latest: "1.0.0".to_string(),
            size_bytes: 1000,
            num_rows: 100,
            schema: serde_json::json!({}),
            metadata: DatasetMetadata::default(),
        });

        let cloned = index.clone();
        assert_eq!(cloned.len(), index.len());
        assert_eq!(cloned.version, index.version);
    }

    #[test]
    fn test_registry_index_debug() {
        let index = RegistryIndex::new();
        let debug = format!("{:?}", index);
        assert!(debug.contains("RegistryIndex"));
    }

    #[test]
    fn test_dataset_info_clone() {
        let info = DatasetInfo {
            name: "test".to_string(),
            versions: vec!["1.0.0".to_string()],
            latest: "1.0.0".to_string(),
            size_bytes: 1000,
            num_rows: 100,
            schema: serde_json::json!({}),
            metadata: DatasetMetadata::default(),
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
    }

    #[test]
    fn test_dataset_info_debug() {
        let info = DatasetInfo {
            name: "test".to_string(),
            versions: vec![],
            latest: String::new(),
            size_bytes: 0,
            num_rows: 0,
            schema: serde_json::json!({}),
            metadata: DatasetMetadata::default(),
        };
        let debug = format!("{:?}", info);
        assert!(debug.contains("DatasetInfo"));
    }

    #[test]
    fn test_dataset_metadata_clone() {
        let metadata = DatasetMetadata {
            description: "desc".to_string(),
            license: "MIT".to_string(),
            tags: vec!["a".to_string()],
            source: Some("http://example.com".to_string()),
            citation: Some("citation".to_string()),
            sha256: Some("abc123".to_string()),
        };
        let cloned = metadata.clone();
        assert_eq!(cloned.description, metadata.description);
        assert_eq!(cloned.sha256, metadata.sha256);
    }

    #[test]
    fn test_dataset_metadata_debug() {
        let metadata = DatasetMetadata::default();
        let debug = format!("{:?}", metadata);
        assert!(debug.contains("DatasetMetadata"));
    }

    #[test]
    fn test_metadata_builder_all_fields() {
        let metadata = DatasetMetadata::builder()
            .description("Test")
            .license("MIT")
            .tag("tag1")
            .source("http://source.com")
            .citation("Citation text")
            .sha256("abc123def456")
            .build();

        assert_eq!(metadata.description, "Test");
        assert_eq!(metadata.license, "MIT");
        assert_eq!(metadata.source, Some("http://source.com".to_string()));
        assert_eq!(metadata.citation, Some("Citation text".to_string()));
        assert_eq!(metadata.sha256, Some("abc123def456".to_string()));
    }

    #[test]
    fn test_dataset_metadata_builder_debug() {
        let builder = DatasetMetadataBuilder::default();
        let debug = format!("{:?}", builder);
        assert!(debug.contains("DatasetMetadataBuilder"));
    }

    #[test]
    fn test_dataset_metadata_default() {
        let metadata = DatasetMetadata::default();
        assert!(metadata.description.is_empty());
        assert!(metadata.license.is_empty());
        assert!(metadata.tags.is_empty());
        assert!(metadata.source.is_none());
        assert!(metadata.citation.is_none());
        assert!(metadata.sha256.is_none());
    }
}
