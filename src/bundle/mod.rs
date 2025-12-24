//! Model Bundling and Memory Paging
//!
//! This module provides efficient packaging and memory management for ML models,
//! enabling deployment on resource-constrained devices and handling of large models.
//!
//! # Features
//!
//! - **Model Bundling**: Package multiple models into a single `.apbundle` file
//! - **Memory Paging**: LRU-based component loading for models larger than RAM
//! - **Memory Mapping**: OS-level memory management via mmap for efficient I/O
//! - **Pre-fetching**: Proactive loading of anticipated model components
//!
//! # Example
//!
//! ```ignore
//! use aprender::bundle::{ModelBundle, BundleConfig};
//!
//! // Create a bundle from models
//! let bundle = ModelBundle::new("my_models.apbundle")
//!     .add_model("classifier", classifier_weights)?
//!     .add_model("encoder", encoder_weights)?
//!     .build()?;
//!
//! // Load with memory paging
//! let loaded = ModelBundle::load_paged("my_models.apbundle", 10_000_000)?; // 10MB limit
//! let weights = loaded.get_model("classifier")?;
//! ```

mod format;
mod manifest;
mod mmap;
mod paging;

pub use format::{BundleFormat, BundleReader, BundleWriter};
pub use manifest::{BundleManifest, ModelEntry, ModelMetadata};
pub use mmap::{MappedFile, MappedRegion, MemoryMappedFile};
pub use paging::{PagedBundle, PagingConfig, PagingStats};

use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Bundle Constants
// ============================================================================

/// Magic bytes for .apbundle format.
pub const BUNDLE_MAGIC: &[u8; 8] = b"APBUNDLE";

/// Current bundle format version.
pub const BUNDLE_VERSION: u32 = 1;

/// Default page size for memory paging (4KB).
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// Default maximum memory for paged bundles (100MB).
pub const DEFAULT_MAX_MEMORY: usize = 100 * 1024 * 1024;

// ============================================================================
// Model Bundle
// ============================================================================

/// A bundle of machine learning models with efficient storage and loading.
///
/// The `ModelBundle` struct provides:
/// - Atomic deployment of multiple related models
/// - Efficient memory-mapped access to model weights
/// - LRU-based paging for large models
#[derive(Debug)]
pub struct ModelBundle {
    /// Path to the bundle file.
    path: Option<String>,
    /// Bundle manifest.
    manifest: BundleManifest,
    /// Loaded model data (in-memory).
    models: HashMap<String, Vec<u8>>,
    /// Bundle configuration.
    config: BundleConfig,
}

/// Configuration for model bundle creation and loading.
#[derive(Debug, Clone)]
pub struct BundleConfig {
    /// Enable compression for model weights.
    pub compress: bool,
    /// Maximum memory for paged loading (bytes).
    pub max_memory: usize,
    /// Page size for paging (bytes).
    pub page_size: usize,
    /// Enable pre-fetching of components.
    pub prefetch: bool,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            compress: false,
            max_memory: DEFAULT_MAX_MEMORY,
            page_size: DEFAULT_PAGE_SIZE,
            prefetch: true,
        }
    }
}

impl BundleConfig {
    /// Create a new bundle configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable compression.
    #[must_use]
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }

    /// Set maximum memory for paged loading.
    #[must_use]
    pub fn with_max_memory(mut self, max_memory: usize) -> Self {
        self.max_memory = max_memory.max(DEFAULT_PAGE_SIZE);
        self
    }

    /// Set page size for paging.
    #[must_use]
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size.max(512);
        self
    }

    /// Enable or disable pre-fetching.
    #[must_use]
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }
}

impl ModelBundle {
    /// Create a new empty model bundle.
    #[must_use]
    pub fn new() -> Self {
        Self {
            path: None,
            manifest: BundleManifest::new(),
            models: HashMap::new(),
            config: BundleConfig::default(),
        }
    }

    /// Create a bundle builder for a specific path.
    #[must_use]
    pub fn builder(path: impl Into<String>) -> BundleBuilder {
        BundleBuilder::new(path)
    }

    /// Load a bundle from a file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let mut reader = BundleReader::open(path)?;
        let manifest = reader.read_manifest()?;
        let models = reader.read_all_models(&manifest)?;

        Ok(Self {
            path: Some(path_str),
            manifest,
            models,
            config: BundleConfig::default(),
        })
    }

    /// Load a bundle with memory paging enabled.
    pub fn load_paged(path: impl AsRef<Path>, max_memory: usize) -> Result<PagedBundle> {
        PagedBundle::open(path, PagingConfig::new().with_max_memory(max_memory))
    }

    /// Save the bundle to a file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let writer = BundleWriter::create(path)?;
        writer.write_bundle(&self.manifest, &self.models)
    }

    /// Add a model to the bundle.
    pub fn add_model(&mut self, name: impl Into<String>, data: Vec<u8>) -> &mut Self {
        let name = name.into();
        let size = data.len();

        self.manifest.add_model(ModelEntry::new(&name, size));
        self.models.insert(name, data);
        self
    }

    /// Get a model's data by name.
    #[must_use]
    pub fn get_model(&self, name: &str) -> Option<&[u8]> {
        self.models.get(name).map(Vec::as_slice)
    }

    /// Get model metadata by name.
    #[must_use]
    pub fn get_metadata(&self, name: &str) -> Option<&ModelEntry> {
        self.manifest.get_model(name)
    }

    /// Get all model names in the bundle.
    #[must_use]
    pub fn model_names(&self) -> Vec<&str> {
        self.manifest.model_names()
    }

    /// Get the number of models in the bundle.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if the bundle is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Get the total size of all models in bytes.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.models.values().map(Vec::len).sum()
    }

    /// Get the bundle manifest.
    #[must_use]
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    /// Get the bundle configuration.
    #[must_use]
    pub fn config(&self) -> &BundleConfig {
        &self.config
    }
}

impl Default for ModelBundle {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Bundle Builder
// ============================================================================

/// Builder for creating model bundles.
#[derive(Debug)]
pub struct BundleBuilder {
    path: String,
    models: HashMap<String, Vec<u8>>,
    metadata: HashMap<String, ModelMetadata>,
    config: BundleConfig,
}

impl BundleBuilder {
    /// Create a new bundle builder.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            models: HashMap::new(),
            metadata: HashMap::new(),
            config: BundleConfig::default(),
        }
    }

    /// Add a model to the bundle.
    #[must_use]
    pub fn add_model(mut self, name: impl Into<String>, data: Vec<u8>) -> Self {
        let name = name.into();
        self.metadata
            .insert(name.clone(), ModelMetadata::new(data.len()));
        self.models.insert(name, data);
        self
    }

    /// Add a model with custom metadata.
    #[must_use]
    pub fn add_model_with_metadata(
        mut self,
        name: impl Into<String>,
        data: Vec<u8>,
        metadata: ModelMetadata,
    ) -> Self {
        let name = name.into();
        self.metadata.insert(name.clone(), metadata);
        self.models.insert(name, data);
        self
    }

    /// Set bundle configuration.
    #[must_use]
    pub fn with_config(mut self, config: BundleConfig) -> Self {
        self.config = config;
        self
    }

    /// Build and save the bundle.
    pub fn build(self) -> Result<ModelBundle> {
        let mut bundle = ModelBundle::new();
        bundle.config = self.config;

        for (name, data) in self.models {
            bundle.add_model(name, data);
        }

        bundle.save(&self.path)?;
        bundle.path = Some(self.path);

        Ok(bundle)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_bundle_config_default() {
        let config = BundleConfig::default();
        assert!(!config.compress);
        assert_eq!(config.max_memory, DEFAULT_MAX_MEMORY);
        assert_eq!(config.page_size, DEFAULT_PAGE_SIZE);
        assert!(config.prefetch);
    }

    #[test]
    fn test_bundle_config_builder() {
        let config = BundleConfig::new()
            .with_compression(true)
            .with_max_memory(50_000_000)
            .with_page_size(8192)
            .with_prefetch(false);

        assert!(config.compress);
        assert_eq!(config.max_memory, 50_000_000);
        assert_eq!(config.page_size, 8192);
        assert!(!config.prefetch);
    }

    #[test]
    fn test_model_bundle_new() {
        let bundle = ModelBundle::new();
        assert!(bundle.is_empty());
        assert_eq!(bundle.len(), 0);
        assert_eq!(bundle.total_size(), 0);
    }

    #[test]
    fn test_model_bundle_add_model() {
        let mut bundle = ModelBundle::new();
        let data = vec![1u8, 2, 3, 4, 5];

        bundle.add_model("test_model", data.clone());

        assert_eq!(bundle.len(), 1);
        assert!(!bundle.is_empty());
        assert_eq!(bundle.get_model("test_model"), Some(data.as_slice()));
        assert!(bundle.model_names().contains(&"test_model"));
    }

    #[test]
    fn test_model_bundle_multiple_models() {
        let mut bundle = ModelBundle::new();

        bundle.add_model("model1", vec![1, 2, 3]);
        bundle.add_model("model2", vec![4, 5, 6, 7]);
        bundle.add_model("model3", vec![8]);

        assert_eq!(bundle.len(), 3);
        assert_eq!(bundle.total_size(), 8);
        assert_eq!(bundle.model_names().len(), 3);
    }

    #[test]
    fn test_model_bundle_get_nonexistent() {
        let bundle = ModelBundle::new();
        assert!(bundle.get_model("nonexistent").is_none());
    }

    #[test]
    fn test_bundle_builder() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path().to_string_lossy().to_string();

        let bundle = BundleBuilder::new(&path)
            .add_model("model1", vec![1, 2, 3])
            .add_model("model2", vec![4, 5])
            .with_config(BundleConfig::new().with_compression(false))
            .build()
            .expect("Failed to build bundle");

        assert_eq!(bundle.len(), 2);
        assert_eq!(bundle.get_model("model1"), Some(&[1u8, 2, 3][..]));
    }

    #[test]
    fn test_bundle_save_load() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        // Create and save
        let mut original = ModelBundle::new();
        original.add_model("weights", vec![1, 2, 3, 4, 5]);
        original.add_model("biases", vec![6, 7, 8]);
        original.save(path).expect("Failed to save bundle");

        // Load and verify
        let loaded = ModelBundle::load(path).expect("Failed to load bundle");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get_model("weights"), Some(&[1u8, 2, 3, 4, 5][..]));
        assert_eq!(loaded.get_model("biases"), Some(&[6u8, 7, 8][..]));
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_bundle_config_clone() {
        let config = BundleConfig::new()
            .with_compression(true)
            .with_max_memory(10_000_000)
            .with_page_size(8192)
            .with_prefetch(false);

        let cloned = config.clone();
        assert_eq!(cloned.compress, config.compress);
        assert_eq!(cloned.max_memory, config.max_memory);
        assert_eq!(cloned.page_size, config.page_size);
        assert_eq!(cloned.prefetch, config.prefetch);
    }

    #[test]
    fn test_bundle_config_debug() {
        let config = BundleConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("BundleConfig"));
        assert!(debug_str.contains("compress"));
    }

    #[test]
    fn test_bundle_config_min_memory_enforced() {
        let config = BundleConfig::new().with_max_memory(100);
        // Should enforce minimum to page size
        assert_eq!(config.max_memory, DEFAULT_PAGE_SIZE);
    }

    #[test]
    fn test_bundle_config_min_page_size_enforced() {
        let config = BundleConfig::new().with_page_size(100);
        // Should enforce minimum to 512
        assert_eq!(config.page_size, 512);
    }

    #[test]
    fn test_model_bundle_builder_static() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path().to_string_lossy().to_string();

        // Test the static builder method
        let bundle = ModelBundle::builder(&path)
            .add_model("model1", vec![1, 2, 3])
            .build()
            .expect("Failed to build bundle");

        assert_eq!(bundle.len(), 1);
    }

    #[test]
    fn test_model_bundle_manifest() {
        let mut bundle = ModelBundle::new();
        bundle.add_model("test", vec![1, 2, 3]);

        let manifest = bundle.manifest();
        assert_eq!(manifest.model_names().len(), 1);
    }

    #[test]
    fn test_model_bundle_config_getter() {
        let bundle = ModelBundle::new();
        let config = bundle.config();
        assert!(!config.compress);
        assert!(config.prefetch);
    }

    #[test]
    fn test_model_bundle_get_metadata() {
        let mut bundle = ModelBundle::new();
        bundle.add_model("test_model", vec![1, 2, 3, 4, 5]);

        let metadata = bundle.get_metadata("test_model");
        assert!(metadata.is_some());
        let entry = metadata.expect("Metadata should exist");
        assert_eq!(entry.name, "test_model");
    }

    #[test]
    fn test_model_bundle_get_metadata_nonexistent() {
        let bundle = ModelBundle::new();
        assert!(bundle.get_metadata("nonexistent").is_none());
    }

    #[test]
    fn test_model_bundle_default() {
        let bundle = ModelBundle::default();
        assert!(bundle.is_empty());
        assert_eq!(bundle.len(), 0);
    }

    #[test]
    fn test_model_bundle_debug() {
        let bundle = ModelBundle::new();
        let debug_str = format!("{:?}", bundle);
        assert!(debug_str.contains("ModelBundle"));
    }

    #[test]
    fn test_bundle_builder_debug() {
        let builder = BundleBuilder::new("test.apbundle");
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("BundleBuilder"));
    }

    #[test]
    fn test_bundle_builder_with_metadata() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path().to_string_lossy().to_string();

        let metadata = ModelMetadata::new(100)
            .with_version("1.0.0");

        let bundle = BundleBuilder::new(&path)
            .add_model_with_metadata("model1", vec![1, 2, 3], metadata)
            .build()
            .expect("Failed to build bundle");

        assert_eq!(bundle.len(), 1);
    }

    #[test]
    fn test_model_bundle_load_paged() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        // Create and save a bundle first
        let mut bundle = ModelBundle::new();
        bundle.add_model("weights", vec![1, 2, 3, 4, 5]);
        bundle.save(path).expect("Failed to save bundle");

        // Load with paging
        let paged = ModelBundle::load_paged(path, 1_000_000).expect("Failed to load paged");
        let names = paged.model_names();
        assert!(names.iter().any(|n| *n == "weights"));
    }

    #[test]
    fn test_bundle_constants() {
        assert_eq!(BUNDLE_MAGIC.len(), 8);
        assert_eq!(BUNDLE_VERSION, 1);
        assert_eq!(DEFAULT_PAGE_SIZE, 4096);
        assert!(DEFAULT_MAX_MEMORY > 0);
    }
}
