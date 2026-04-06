//! Dataset registry for sharing and discovery.
//!
//! The registry provides a way to publish, discover, and retrieve datasets
//! from various storage backends. It supports local registries, S3-based
//! registries, and HTTP-based read-only registries.

mod index;

pub use index::{DatasetInfo, DatasetMetadata, RegistryIndex};

use crate::{
    backend::StorageBackend,
    dataset::{ArrowDataset, Dataset},
    error::{Error, Result},
};

/// A dataset registry for publishing and retrieving datasets.
///
/// The registry stores datasets along with metadata in a structured format,
/// enabling discovery and versioning of datasets.
///
/// # Example
///
/// ```no_run
/// use alimentar::{backend::MemoryBackend, registry::Registry};
///
/// let backend = MemoryBackend::new();
/// let registry = Registry::new(Box::new(backend));
/// ```
pub struct Registry {
    backend: Box<dyn StorageBackend>,
    index_path: String,
}

impl Registry {
    /// Creates a new registry with the given storage backend.
    pub fn new(backend: Box<dyn StorageBackend>) -> Self {
        Self {
            backend,
            index_path: "registry-index.json".to_string(),
        }
    }

    /// Creates a new registry with a custom index path.
    pub fn with_index_path(
        backend: Box<dyn StorageBackend>,
        index_path: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            index_path: index_path.into(),
        }
    }

    /// Initializes the registry by creating an empty index if one doesn't
    /// exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be created.
    pub fn init(&self) -> Result<()> {
        if self.backend.exists(&self.index_path)? {
            return Ok(());
        }

        let index = RegistryIndex::new();
        self.save_index(&index)
    }

    /// Loads the registry index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be loaded or parsed.
    pub fn load_index(&self) -> Result<RegistryIndex> {
        let data = self.backend.get(&self.index_path)?;
        let index: RegistryIndex = serde_json::from_slice(&data)
            .map_err(|e| Error::storage(format!("Failed to parse registry index: {e}")))?;
        Ok(index)
    }

    /// Saves the registry index.
    fn save_index(&self, index: &RegistryIndex) -> Result<()> {
        let data = serde_json::to_vec_pretty(index)
            .map_err(|e| Error::storage(format!("Failed to serialize registry index: {e}")))?;
        self.backend.put(&self.index_path, data.into())
    }

    /// Lists all available datasets in the registry.
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be loaded.
    pub fn list(&self) -> Result<Vec<DatasetInfo>> {
        let index = self.load_index()?;
        Ok(index.datasets)
    }

    /// Gets information about a specific dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset is not found.
    pub fn get_info(&self, name: &str) -> Result<DatasetInfo> {
        let index = self.load_index()?;
        index
            .datasets
            .into_iter()
            .find(|d| d.name == name)
            .ok_or_else(|| Error::storage(format!("Dataset '{}' not found in registry", name)))
    }

    /// Publishes a dataset to the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for the dataset
    /// * `version` - Semantic version string (e.g., "1.0.0")
    /// * `dataset` - The dataset to publish
    /// * `metadata` - Metadata describing the dataset
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset cannot be saved or the index cannot be
    /// updated.
    pub fn publish(
        &self,
        name: &str,
        version: &str,
        dataset: &ArrowDataset,
        metadata: DatasetMetadata,
    ) -> Result<()> {
        // Validate inputs
        if name.is_empty() {
            return Err(Error::invalid_config("Dataset name cannot be empty"));
        }
        if version.is_empty() {
            return Err(Error::invalid_config("Version cannot be empty"));
        }

        // Create the data path
        let data_path = format!("datasets/{}/{}/data.parquet", name, version);

        // Save the dataset as parquet (use thread ID for uniqueness in parallel tests)
        let temp_dir = std::env::temp_dir();
        let unique_id = format!("{}_{:?}", std::process::id(), std::thread::current().id());
        let temp_path = temp_dir.join(format!("alimentar_publish_{}.parquet", unique_id));
        dataset.to_parquet(&temp_path)?;

        // Read the parquet file and upload
        let parquet_data = std::fs::read(&temp_path).map_err(|e| Error::io(e, &temp_path))?;
        self.backend.put(&data_path, parquet_data.into())?;

        // Delete scratch file after upload completes
        let _ = std::fs::remove_file(&temp_path);

        // Update the index
        let mut index = self.load_index().unwrap_or_else(|_| RegistryIndex::new());

        // Find or create dataset info
        let dataset_info = index.datasets.iter_mut().find(|d| d.name == name);

        let size_bytes = self.backend.size(&data_path)?;
        let num_rows = dataset.len();
        let schema = dataset.schema();

        if let Some(info) = dataset_info {
            // Update existing dataset
            if !info.versions.contains(&version.to_string()) {
                info.versions.push(version.to_string());
            }
            info.latest = version.to_string();
            info.size_bytes = size_bytes;
            info.num_rows = num_rows;
            info.metadata = metadata;
        } else {
            // Add new dataset
            // Convert schema to a simple JSON representation
            let schema_json = schema_to_json(&schema);

            index.datasets.push(DatasetInfo {
                name: name.to_string(),
                versions: vec![version.to_string()],
                latest: version.to_string(),
                size_bytes,
                num_rows,
                schema: schema_json,
                metadata,
            });
        }

        self.save_index(&index)
    }

    /// Pulls a dataset from the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the dataset
    /// * `version` - Optional version (uses latest if not specified)
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset cannot be found or loaded.
    pub fn pull(&self, name: &str, version: Option<&str>) -> Result<ArrowDataset> {
        let info = self.get_info(name)?;

        let version = version.unwrap_or(&info.latest);

        if !info.versions.contains(&version.to_string()) {
            return Err(Error::storage(format!(
                "Version '{}' not found for dataset '{}'. Available: {:?}",
                version, name, info.versions
            )));
        }

        let data_path = format!("datasets/{}/{}/data.parquet", name, version);

        // Fetch data and write to per-thread scratch file
        let data = self.backend.get(&data_path)?;

        let temp_dir = std::env::temp_dir();
        let unique_id = format!("{}_{:?}", std::process::id(), std::thread::current().id());
        let temp_path = temp_dir.join(format!("alimentar_pull_{}.parquet", unique_id));

        std::fs::write(&temp_path, &data).map_err(|e| Error::io(e, &temp_path))?;

        let dataset = ArrowDataset::from_parquet(&temp_path)?;

        // Delete scratch file after parquet load completes
        let _ = std::fs::remove_file(&temp_path);

        Ok(dataset)
    }

    /// Searches datasets by query string (matches name and description).
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be loaded.
    pub fn search(&self, query: &str) -> Result<Vec<DatasetInfo>> {
        let index = self.load_index()?;
        let query_lower = query.to_lowercase();

        let results: Vec<DatasetInfo> = index
            .datasets
            .into_iter()
            .filter(|d| {
                d.name.to_lowercase().contains(&query_lower)
                    || d.metadata.description.to_lowercase().contains(&query_lower)
            })
            .collect();

        Ok(results)
    }

    /// Searches datasets by tags.
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be loaded.
    pub fn search_tags(&self, tags: &[&str]) -> Result<Vec<DatasetInfo>> {
        let index = self.load_index()?;

        let results: Vec<DatasetInfo> = index
            .datasets
            .into_iter()
            .filter(|d| {
                tags.iter()
                    .any(|&tag| d.metadata.tags.iter().any(|t| t == tag))
            })
            .collect();

        Ok(results)
    }

    /// Deletes a dataset version from the registry.
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset cannot be deleted.
    pub fn delete(&self, name: &str, version: &str) -> Result<()> {
        let mut index = self.load_index()?;

        let dataset_idx = index
            .datasets
            .iter()
            .position(|d| d.name == name)
            .ok_or_else(|| Error::storage(format!("Dataset '{}' not found", name)))?;

        let info = &mut index.datasets[dataset_idx];

        if !info.versions.contains(&version.to_string()) {
            return Err(Error::storage(format!(
                "Version '{}' not found for dataset '{}'",
                version, name
            )));
        }

        // Delete the data file
        let data_path = format!("datasets/{}/{}/data.parquet", name, version);
        self.backend.delete(&data_path)?;

        // Update index
        info.versions.retain(|v| v != version);

        if info.versions.is_empty() {
            // Remove dataset entirely if no versions left
            index.datasets.remove(dataset_idx);
        } else if info.latest == version {
            // Update latest to most recent remaining version
            info.latest = info.versions.last().cloned().unwrap_or_default();
        }

        self.save_index(&index)
    }
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("index_path", &self.index_path)
            .finish_non_exhaustive()
    }
}

/// Converts an Arrow schema to a JSON representation.
fn schema_to_json(schema: &arrow::datatypes::SchemaRef) -> serde_json::Value {
    let fields: Vec<serde_json::Value> = schema
        .fields()
        .iter()
        .map(|field| {
            serde_json::json!({
                "name": field.name(),
                "data_type": format!("{:?}", field.data_type()),
                "nullable": field.is_nullable(),
            })
        })
        .collect();

    serde_json::json!({
        "fields": fields,
    })
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::{backend::MemoryBackend, Dataset};

    fn create_test_dataset(rows: usize) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"))
    }

    fn create_test_metadata() -> DatasetMetadata {
        DatasetMetadata {
            description: "Test dataset".to_string(),
            license: "MIT".to_string(),
            tags: vec!["test".to_string(), "example".to_string()],
            source: Some("unit test".to_string()),
            citation: None,
            sha256: None,
        }
    }

    #[test]
    fn test_registry_init() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));

        let result = registry.init();
        assert!(result.is_ok());

        // Second init should be idempotent
        let result = registry.init();
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry_publish_and_pull() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(10);
        let metadata = create_test_metadata();

        // Publish
        let result = registry.publish("test-dataset", "1.0.0", &dataset, metadata);
        assert!(result.is_ok());

        // Pull
        let pulled = registry.pull("test-dataset", Some("1.0.0"));
        assert!(pulled.is_ok());
        let pulled = pulled.ok().unwrap_or_else(|| panic!("Should pull"));
        assert_eq!(pulled.len(), 10);
    }

    #[test]
    fn test_registry_list() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("dataset-a", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));
        registry
            .publish("dataset-b", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        let list = registry.list();
        assert!(list.is_ok());
        let list = list.ok().unwrap_or_else(|| panic!("Should list"));
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_registry_search() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);

        let metadata1 = DatasetMetadata {
            description: "Machine learning dataset".to_string(),
            license: "MIT".to_string(),
            tags: vec!["ml".to_string()],
            source: None,
            citation: None,
            sha256: None,
        };

        let metadata2 = DatasetMetadata {
            description: "Natural language processing data".to_string(),
            license: "Apache-2.0".to_string(),
            tags: vec!["nlp".to_string()],
            source: None,
            citation: None,
            sha256: None,
        };

        registry
            .publish("ml-data", "1.0.0", &dataset, metadata1)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));
        registry
            .publish("nlp-data", "1.0.0", &dataset, metadata2)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        // Search by name
        let results = registry.search("ml");
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| panic!("Should search"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "ml-data");

        // Search by description
        let results = registry.search("natural language");
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| panic!("Should search"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "nlp-data");
    }

    #[test]
    fn test_registry_search_tags() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = DatasetMetadata {
            description: "Test".to_string(),
            license: "MIT".to_string(),
            tags: vec!["ml".to_string(), "vision".to_string()],
            source: None,
            citation: None,
            sha256: None,
        };

        registry
            .publish("vision-data", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        let results = registry.search_tags(&["vision"]);
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| panic!("Should search"));
        assert_eq!(results.len(), 1);

        let results = registry.search_tags(&["audio"]);
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| panic!("Should search"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_registry_versioning() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset_v1 = create_test_dataset(10);
        let dataset_v2 = create_test_dataset(20);
        let metadata = create_test_metadata();

        registry
            .publish("versioned", "1.0.0", &dataset_v1, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should publish v1"));
        registry
            .publish("versioned", "2.0.0", &dataset_v2, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish v2"));

        let info = registry.get_info("versioned");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("Should get info"));
        assert_eq!(info.versions.len(), 2);
        assert_eq!(info.latest, "2.0.0");

        // Pull specific version
        let v1 = registry.pull("versioned", Some("1.0.0"));
        assert!(v1.is_ok());
        let v1 = v1.ok().unwrap_or_else(|| panic!("Should pull v1"));
        assert_eq!(v1.len(), 10);

        // Pull latest
        let latest = registry.pull("versioned", None);
        assert!(latest.is_ok());
        let latest = latest.ok().unwrap_or_else(|| panic!("Should pull latest"));
        assert_eq!(latest.len(), 20);
    }

    #[test]
    fn test_registry_delete() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("to-delete", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));
        registry
            .publish("to-delete", "2.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        // Delete one version
        let result = registry.delete("to-delete", "1.0.0");
        assert!(result.is_ok());

        let info = registry.get_info("to-delete");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("Should get info"));
        assert_eq!(info.versions.len(), 1);
        assert!(!info.versions.contains(&"1.0.0".to_string()));

        // Delete last version removes dataset
        let result = registry.delete("to-delete", "2.0.0");
        assert!(result.is_ok());

        let info = registry.get_info("to-delete");
        assert!(info.is_err());
    }

    #[test]
    fn test_registry_empty_name_error() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        let result = registry.publish("", "1.0.0", &dataset, metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_not_found() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let result = registry.pull("nonexistent", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_with_index_path() {
        let backend = MemoryBackend::new();
        let registry = Registry::with_index_path(Box::new(backend), "custom-index.json");
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        let result = registry.publish("test", "1.0.0", &dataset, metadata);
        assert!(result.is_ok());

        let list = registry.list();
        assert!(list.is_ok());
        assert_eq!(list.ok().unwrap_or_else(|| panic!("Should list")).len(), 1);
    }

    #[test]
    fn test_registry_debug() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        let debug = format!("{:?}", registry);
        assert!(debug.contains("Registry"));
        assert!(debug.contains("index_path"));
    }

    #[test]
    fn test_registry_delete_nonexistent_dataset() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let result = registry.delete("nonexistent", "1.0.0");
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_delete_nonexistent_version() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("dataset", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        let result = registry.delete("dataset", "2.0.0");
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_delete_latest_updates_correctly() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("multi-version", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));
        registry
            .publish("multi-version", "2.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        // Latest is 2.0.0, delete it
        let result = registry.delete("multi-version", "2.0.0");
        assert!(result.is_ok());

        let info = registry.get_info("multi-version");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("Should get info"));
        assert_eq!(info.latest, "1.0.0");
    }

    #[test]
    fn test_registry_pull_nonexistent_version() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("versioned-test", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        let result = registry.pull("versioned-test", Some("9.9.9"));
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_empty_version_error() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        let result = registry.publish("test", "", &dataset, metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_publish_update_existing() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset1 = create_test_dataset(5);
        let dataset2 = create_test_dataset(10);
        let metadata = create_test_metadata();

        // Publish same version twice (should update)
        registry
            .publish("update-test", "1.0.0", &dataset1, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));
        registry
            .publish("update-test", "1.0.0", &dataset2, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish update"));

        let info = registry.get_info("update-test");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("Should get info"));
        // Versions should not be duplicated
        assert_eq!(info.versions.len(), 1);
        // Rows should be from second publish
        assert_eq!(info.num_rows, 10);
    }

    // === Additional coverage tests for registry ===

    #[test]
    fn test_registry_load_index_without_init() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));

        // Trying to load without init should fail
        let result = registry.load_index();
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_list_empty() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let list = registry.list();
        assert!(list.is_ok());
        assert!(list.ok().unwrap_or_else(|| panic!("list")).is_empty());
    }

    #[test]
    fn test_registry_search_no_results() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("my-dataset", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        let results = registry.search("nonexistent");
        assert!(results.is_ok());
        assert!(results.ok().unwrap_or_else(|| panic!("search")).is_empty());
    }

    #[test]
    fn test_registry_search_case_insensitive() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = DatasetMetadata {
            description: "UPPERCASE DESCRIPTION".to_string(),
            license: "MIT".to_string(),
            tags: vec![],
            source: None,
            citation: None,
            sha256: None,
        };

        registry
            .publish("TestDataset", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        // Search lowercase
        let results = registry.search("testdataset");
        assert_eq!(results.ok().unwrap_or_else(|| panic!("search")).len(), 1);

        // Search description lowercase
        let results = registry.search("uppercase");
        assert_eq!(results.ok().unwrap_or_else(|| panic!("search")).len(), 1);
    }

    #[test]
    fn test_registry_search_tags_no_match() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = DatasetMetadata {
            description: "Test".to_string(),
            license: "MIT".to_string(),
            tags: vec!["ml".to_string(), "vision".to_string()],
            source: None,
            citation: None,
            sha256: None,
        };

        registry
            .publish("tagged-data", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        let results = registry.search_tags(&["audio", "nlp"]);
        assert!(results.is_ok());
        assert!(results.ok().unwrap_or_else(|| panic!("search")).is_empty());
    }

    #[test]
    fn test_registry_search_tags_multiple_matches() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);

        let metadata1 = DatasetMetadata {
            description: "Dataset 1".to_string(),
            license: "MIT".to_string(),
            tags: vec!["ml".to_string()],
            source: None,
            citation: None,
            sha256: None,
        };

        let metadata2 = DatasetMetadata {
            description: "Dataset 2".to_string(),
            license: "MIT".to_string(),
            tags: vec!["ml".to_string(), "vision".to_string()],
            source: None,
            citation: None,
            sha256: None,
        };

        registry
            .publish("data-a", "1.0.0", &dataset, metadata1)
            .ok()
            .unwrap_or_else(|| panic!("publish a"));
        registry
            .publish("data-b", "1.0.0", &dataset, metadata2)
            .ok()
            .unwrap_or_else(|| panic!("publish b"));

        // Both have 'ml' tag
        let results = registry.search_tags(&["ml"]);
        assert_eq!(results.ok().unwrap_or_else(|| panic!("search")).len(), 2);
    }

    #[test]
    fn test_registry_get_info_schema_contains_fields() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(10);
        let metadata = create_test_metadata();

        registry
            .publish("schema-test", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("Should publish"));

        let info = registry.get_info("schema-test");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("info"));

        // Schema should be JSON with fields
        assert!(info.schema.is_object());
        let fields = info.schema.get("fields");
        assert!(fields.is_some());
    }

    #[test]
    fn test_registry_multiple_versions_ordering() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        // Publish in non-sequential order
        registry
            .publish("ordered", "2.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("publish 2.0"));
        registry
            .publish("ordered", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("publish 1.0"));
        registry
            .publish("ordered", "3.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("publish 3.0"));

        let info = registry.get_info("ordered");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("info"));

        // Latest should be the last published
        assert_eq!(info.latest, "3.0.0");
        assert_eq!(info.versions.len(), 3);
    }

    #[test]
    fn test_registry_delete_middle_version() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("three-versions", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("publish 1.0"));
        registry
            .publish("three-versions", "2.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("publish 2.0"));
        registry
            .publish("three-versions", "3.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("publish 3.0"));

        // Delete middle version
        registry
            .delete("three-versions", "2.0.0")
            .ok()
            .unwrap_or_else(|| panic!("delete"));

        let info = registry.get_info("three-versions");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("info"));

        assert_eq!(info.versions.len(), 2);
        assert!(!info.versions.contains(&"2.0.0".to_string()));
        // Latest should still be 3.0.0
        assert_eq!(info.latest, "3.0.0");
    }

    #[test]
    fn test_registry_pull_specific_version() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset_v1 = create_test_dataset(5);
        let dataset_v2 = create_test_dataset(15);
        let metadata = create_test_metadata();

        registry
            .publish("versioned", "1.0.0", &dataset_v1, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("publish v1"));
        registry
            .publish("versioned", "2.0.0", &dataset_v2, metadata)
            .ok()
            .unwrap_or_else(|| panic!("publish v2"));

        // Pull specific version 1.0.0 (not latest)
        let pulled = registry.pull("versioned", Some("1.0.0"));
        assert!(pulled.is_ok());
        let pulled = pulled.ok().unwrap_or_else(|| panic!("pulled"));
        assert_eq!(pulled.len(), 5);
    }

    #[test]
    fn test_registry_init_idempotent() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));

        // Multiple inits should be safe
        for _ in 0..5 {
            let result = registry.init();
            assert!(result.is_ok());
        }

        // Registry should still work
        let list = registry.list();
        assert!(list.is_ok());
    }

    #[test]
    fn test_registry_metadata_preserved() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(10);
        let metadata = DatasetMetadata {
            description: "Full metadata test".to_string(),
            license: "Apache-2.0".to_string(),
            tags: vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()],
            source: Some("http://example.com/source".to_string()),
            citation: Some("Citation text here".to_string()),
            sha256: Some("abc123".to_string()),
        };

        registry
            .publish("metadata-test", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap_or_else(|| panic!("publish"));

        let info = registry.get_info("metadata-test");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("info"));

        assert_eq!(info.metadata.description, metadata.description);
        assert_eq!(info.metadata.license, metadata.license);
        assert_eq!(info.metadata.tags, metadata.tags);
        assert_eq!(info.metadata.source, metadata.source);
        assert_eq!(info.metadata.citation, metadata.citation);
        assert_eq!(info.metadata.sha256, metadata.sha256);
    }

    #[test]
    fn test_registry_size_bytes_updated() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(100);
        let metadata = create_test_metadata();

        registry
            .publish("size-test", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap_or_else(|| panic!("publish"));

        let info = registry.get_info("size-test");
        assert!(info.is_ok());
        let info = info.ok().unwrap_or_else(|| panic!("info"));

        // Size should be > 0 for a dataset with data
        assert!(info.size_bytes > 0);
    }

    #[test]
    fn test_schema_to_json_function() {
        let dataset = create_test_dataset(5);
        let schema = dataset.schema();
        let json = schema_to_json(&schema);

        assert!(json.is_object());
        let fields = json.get("fields").and_then(|f| f.as_array());
        assert!(fields.is_some());

        let fields = fields.unwrap_or_else(|| panic!("fields"));
        assert_eq!(fields.len(), 2); // id and name

        // Check first field
        let first = &fields[0];
        assert_eq!(
            first
                .get("name")
                .and_then(|n: &serde_json::Value| n.as_str()),
            Some("id")
        );
        assert!(first.get("data_type").is_some());
        assert!(first.get("nullable").is_some());
    }

    #[test]
    fn test_registry_concurrent_publish_different_datasets() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry
            .init()
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        // Publish multiple different datasets
        for i in 0..5 {
            registry
                .publish(
                    &format!("dataset-{}", i),
                    "1.0.0",
                    &dataset,
                    metadata.clone(),
                )
                .ok()
                .unwrap_or_else(|| panic!("publish"));
        }

        let list = registry.list();
        assert!(list.is_ok());
        assert_eq!(list.ok().unwrap_or_else(|| panic!("list")).len(), 5);
    }

    // === Additional registry tests for coverage ===

    #[test]
    fn test_registry_search_partial_name_match() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("machine-learning-data", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap();
        registry
            .publish("deep-learning-data", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap();
        registry
            .publish("statistics-data", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap();

        // Search by partial name
        let results = registry.search("learning").ok().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_registry_search_by_description() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);

        let metadata1 = DatasetMetadata {
            description: "Image classification dataset".to_string(),
            license: "MIT".to_string(),
            tags: vec![],
            source: None,
            citation: None,
            sha256: None,
        };

        let metadata2 = DatasetMetadata {
            description: "Text sentiment analysis".to_string(),
            license: "MIT".to_string(),
            tags: vec![],
            source: None,
            citation: None,
            sha256: None,
        };

        registry
            .publish("images", "1.0.0", &dataset, metadata1)
            .ok()
            .unwrap();
        registry
            .publish("text", "1.0.0", &dataset, metadata2)
            .ok()
            .unwrap();

        // Search by description content
        let image_results = registry.search("classification").ok().unwrap();
        assert_eq!(image_results.len(), 1);
        assert_eq!(image_results[0].name, "images");

        let text_results = registry.search("sentiment").ok().unwrap();
        assert_eq!(text_results.len(), 1);
        assert_eq!(text_results[0].name, "text");
    }

    #[test]
    fn test_registry_search_tags_multiple_tags() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);

        let metadata = DatasetMetadata {
            description: "Multi-tag dataset".to_string(),
            license: "MIT".to_string(),
            tags: vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()],
            source: None,
            citation: None,
            sha256: None,
        };

        registry
            .publish("tagged", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap();

        // Any of the tags should match
        assert_eq!(registry.search_tags(&["tag1"]).ok().unwrap().len(), 1);
        assert_eq!(registry.search_tags(&["tag2"]).ok().unwrap().len(), 1);
        assert_eq!(registry.search_tags(&["tag3"]).ok().unwrap().len(), 1);
        assert_eq!(
            registry.search_tags(&["tag1", "tag2"]).ok().unwrap().len(),
            1
        );
    }

    #[test]
    fn test_registry_delete_first_of_multiple_versions() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("versioned", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap();
        registry
            .publish("versioned", "2.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap();
        registry
            .publish("versioned", "3.0.0", &dataset, metadata)
            .ok()
            .unwrap();

        // Delete first version
        registry.delete("versioned", "1.0.0").ok().unwrap();

        let info = registry.get_info("versioned").ok().unwrap();
        assert_eq!(info.versions.len(), 2);
        assert!(!info.versions.contains(&"1.0.0".to_string()));
        // Latest should still be 3.0.0
        assert_eq!(info.latest, "3.0.0");
    }

    #[test]
    fn test_registry_publish_without_init() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        // Don't call init()

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        // Should still work because publish calls init internally via load_index
        let result = registry.publish("no-init", "1.0.0", &dataset, metadata);
        // This might fail or succeed depending on implementation
        // The current implementation creates index on first publish
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry_get_info_contains_schema() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(10);
        let metadata = create_test_metadata();

        registry
            .publish("schema-check", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap();

        let info = registry.get_info("schema-check").ok().unwrap();

        // Schema should be a JSON object with fields
        assert!(info.schema.is_object());
        let fields = info.schema.get("fields").and_then(|f| f.as_array());
        assert!(fields.is_some());
        assert_eq!(fields.unwrap().len(), 2); // id and name
    }

    #[test]
    fn test_registry_delete_all_versions_removes_dataset() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("temp", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap();
        registry
            .publish("temp", "2.0.0", &dataset, metadata)
            .ok()
            .unwrap();

        // Delete all versions
        registry.delete("temp", "1.0.0").ok().unwrap();
        registry.delete("temp", "2.0.0").ok().unwrap();

        // Dataset should no longer exist
        let result = registry.get_info("temp");
        assert!(result.is_err());

        // List should be empty
        let list = registry.list().ok().unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_registry_pull_uses_latest_when_none() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset_v1 = create_test_dataset(10);
        let dataset_v2 = create_test_dataset(20);
        let metadata = create_test_metadata();

        registry
            .publish("latest-test", "1.0.0", &dataset_v1, metadata.clone())
            .ok()
            .unwrap();
        registry
            .publish("latest-test", "2.0.0", &dataset_v2, metadata)
            .ok()
            .unwrap();

        // Pull without specifying version
        let pulled = registry.pull("latest-test", None).ok().unwrap();
        assert_eq!(pulled.len(), 20); // Should be v2 (20 rows)
    }

    #[test]
    fn test_registry_with_custom_index_path() {
        let backend = MemoryBackend::new();
        let registry = Registry::with_index_path(Box::new(backend), "my-custom-index.json");

        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("custom-path", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap();

        // Should be able to list
        let list = registry.list().ok().unwrap();
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_registry_metadata_all_fields() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);

        let metadata = DatasetMetadata {
            description: "Full metadata test".to_string(),
            license: "Apache-2.0".to_string(),
            tags: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            source: Some("https://example.com/source".to_string()),
            citation: Some("Cite this dataset".to_string()),
            sha256: Some("abc123def456".to_string()),
        };

        registry
            .publish("full-meta", "1.0.0", &dataset, metadata.clone())
            .ok()
            .unwrap();

        let info = registry.get_info("full-meta").ok().unwrap();
        assert_eq!(info.metadata.description, "Full metadata test");
        assert_eq!(info.metadata.license, "Apache-2.0");
        assert_eq!(info.metadata.tags.len(), 3);
        assert_eq!(
            info.metadata.source,
            Some("https://example.com/source".to_string())
        );
        assert_eq!(
            info.metadata.citation,
            Some("Cite this dataset".to_string())
        );
    }

    #[test]
    fn test_registry_republish_same_version() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset1 = create_test_dataset(10);
        let dataset2 = create_test_dataset(20);
        let metadata = create_test_metadata();

        // First publish
        registry
            .publish("republish", "1.0.0", &dataset1, metadata.clone())
            .ok()
            .unwrap();

        let info1 = registry.get_info("republish").ok().unwrap();
        assert_eq!(info1.num_rows, 10);

        // Republish same version with different data
        registry
            .publish("republish", "1.0.0", &dataset2, metadata)
            .ok()
            .unwrap();

        let info2 = registry.get_info("republish").ok().unwrap();
        assert_eq!(info2.num_rows, 20);
        // Should only have 1 version, not 2
        assert_eq!(info2.versions.len(), 1);
    }

    #[test]
    fn test_registry_search_empty_string() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);
        let metadata = create_test_metadata();

        registry
            .publish("test-data", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap();

        // Empty search should match everything (contains "")
        let results = registry.search("").ok().unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_registry_search_tags_empty_array() {
        let backend = MemoryBackend::new();
        let registry = Registry::new(Box::new(backend));
        registry.init().ok().unwrap();

        let dataset = create_test_dataset(5);
        let metadata = DatasetMetadata {
            description: "Test".to_string(),
            license: "MIT".to_string(),
            tags: vec!["ml".to_string()],
            source: None,
            citation: None,
            sha256: None,
        };

        registry
            .publish("tagged-ds", "1.0.0", &dataset, metadata)
            .ok()
            .unwrap();

        // Empty tag array should return empty results
        let results = registry.search_tags(&[]).ok().unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_schema_to_json_with_nullable_fields() {
        // Create schema with nullable and non-nullable fields
        let schema = Arc::new(Schema::new(vec![
            Field::new("required_id", DataType::Int32, false),
            Field::new("optional_name", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let json = schema_to_json(&dataset.schema());
        let fields = json.get("fields").and_then(|f| f.as_array()).unwrap();

        assert_eq!(fields.len(), 2);

        // Check nullable values
        let field0_nullable = fields[0].get("nullable").and_then(|v| v.as_bool());
        let field1_nullable = fields[1].get("nullable").and_then(|v| v.as_bool());

        assert_eq!(field0_nullable, Some(false));
        assert_eq!(field1_nullable, Some(true));
    }
}
