//! Bundle Manifest
//!
//! Defines the structure and serialization of bundle manifests.

use std::collections::HashMap;

// ============================================================================
// Model Metadata
// ============================================================================

/// Metadata for a single model in the bundle.
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// Version of the model.
    pub version: String,
    /// Model architecture description.
    pub architecture: String,
    /// Training hyperparameters.
    pub hyperparameters: HashMap<String, String>,
    /// Custom metadata fields.
    pub custom: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new metadata with size only.
    #[must_use]
    pub fn new(size: usize) -> Self {
        let mut meta = Self::default();
        meta.custom.insert("size".to_string(), size.to_string());
        meta
    }

    /// Set version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set architecture description.
    #[must_use]
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = arch.into();
        self
    }

    /// Add a hyperparameter.
    #[must_use]
    pub fn with_hyperparameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.hyperparameters.insert(key.into(), value.into());
        self
    }

    /// Add custom metadata.
    #[must_use]
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }
}

// ============================================================================
// Model Entry
// ============================================================================

/// An entry in the bundle manifest representing a single model.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Model name (unique identifier within bundle).
    pub name: String,
    /// Size of model data in bytes.
    pub size: usize,
    /// Offset in the bundle file.
    pub offset: u64,
    /// Model metadata.
    pub metadata: ModelMetadata,
    /// Component names (for paging).
    pub components: Vec<String>,
}

impl ModelEntry {
    /// Create a new model entry.
    #[must_use]
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size,
            offset: 0,
            metadata: ModelMetadata::new(size),
            components: Vec::new(),
        }
    }

    /// Set the offset in the bundle file.
    #[must_use]
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Set metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a component name.
    #[must_use]
    pub fn with_component(mut self, name: impl Into<String>) -> Self {
        self.components.push(name.into());
        self
    }
}

// ============================================================================
// Bundle Manifest
// ============================================================================

/// Manifest describing the contents of a model bundle.
///
/// The manifest includes:
/// - Bundle metadata (version, creation time, etc.)
/// - List of models with their offsets and sizes
/// - Model-specific metadata (architecture, hyperparameters)
#[derive(Debug, Clone, Default)]
pub struct BundleManifest {
    /// Bundle format version.
    pub version: u32,
    /// Creation timestamp (Unix epoch seconds).
    pub created_at: u64,
    /// Bundle description.
    pub description: String,
    /// Model entries indexed by name.
    models: HashMap<String, ModelEntry>,
    /// Order of models (for deterministic iteration).
    order: Vec<String>,
}

impl BundleManifest {
    /// Create a new empty manifest.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: super::BUNDLE_VERSION,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            description: String::new(),
            models: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a model entry.
    pub fn add_model(&mut self, entry: ModelEntry) {
        let name = entry.name.clone();
        if !self.models.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.models.insert(name, entry);
    }

    /// Get a model entry by name.
    #[must_use]
    pub fn get_model(&self, name: &str) -> Option<&ModelEntry> {
        self.models.get(name)
    }

    /// Get a mutable model entry by name.
    pub fn get_model_mut(&mut self, name: &str) -> Option<&mut ModelEntry> {
        self.models.get_mut(name)
    }

    /// Get all model names in order.
    #[must_use]
    pub fn model_names(&self) -> Vec<&str> {
        self.order.iter().map(String::as_str).collect()
    }

    /// Get the number of models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if manifest is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Calculate total size of all models.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.models.values().map(|e| e.size).sum()
    }

    /// Iterate over model entries in order.
    pub fn iter(&self) -> impl Iterator<Item = &ModelEntry> {
        self.order.iter().filter_map(|name| self.models.get(name))
    }

    /// Serialize manifest to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version (4 bytes)
        bytes.extend_from_slice(&self.version.to_le_bytes());

        // Created at (8 bytes)
        bytes.extend_from_slice(&self.created_at.to_le_bytes());

        // Description length + data
        let desc_bytes = self.description.as_bytes();
        bytes.extend_from_slice(&(desc_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(desc_bytes);

        // Number of models (4 bytes)
        bytes.extend_from_slice(&(self.models.len() as u32).to_le_bytes());

        // Model entries
        for name in &self.order {
            if let Some(entry) = self.models.get(name) {
                // Name length + name
                let name_bytes = entry.name.as_bytes();
                bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
                bytes.extend_from_slice(name_bytes);

                // Size (8 bytes)
                bytes.extend_from_slice(&(entry.size as u64).to_le_bytes());

                // Offset (8 bytes)
                bytes.extend_from_slice(&entry.offset.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize manifest from bytes.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 16 {
            return None;
        }

        let mut pos = 0;

        // Version
        let version = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;

        // Created at
        let created_at = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;

        // Description
        let desc_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        if pos + desc_len > data.len() {
            return None;
        }
        let description = String::from_utf8(data[pos..pos + desc_len].to_vec()).ok()?;
        pos += desc_len;

        // Number of models
        if pos + 4 > data.len() {
            return None;
        }
        let num_models = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        let mut manifest = Self {
            version,
            created_at,
            description,
            models: HashMap::new(),
            order: Vec::new(),
        };

        // Read model entries
        for _ in 0..num_models {
            // Name
            if pos + 4 > data.len() {
                return None;
            }
            let name_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            if pos + name_len > data.len() {
                return None;
            }
            let name = String::from_utf8(data[pos..pos + name_len].to_vec()).ok()?;
            pos += name_len;

            // Size
            if pos + 8 > data.len() {
                return None;
            }
            let size = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?) as usize;
            pos += 8;

            // Offset
            if pos + 8 > data.len() {
                return None;
            }
            let offset = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
            pos += 8;

            manifest.add_model(ModelEntry::new(&name, size).with_offset(offset));
        }

        Some(manifest)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_default() {
        let meta = ModelMetadata::default();
        assert!(meta.version.is_empty());
        assert!(meta.architecture.is_empty());
        assert!(meta.hyperparameters.is_empty());
    }

    #[test]
    fn test_model_metadata_builder() {
        let meta = ModelMetadata::new(1000)
            .with_version("1.0.0")
            .with_architecture("transformer")
            .with_hyperparameter("layers", "12")
            .with_custom("author", "test");

        assert_eq!(meta.version, "1.0.0");
        assert_eq!(meta.architecture, "transformer");
        assert_eq!(meta.hyperparameters.get("layers"), Some(&"12".to_string()));
        assert_eq!(meta.custom.get("author"), Some(&"test".to_string()));
    }

    #[test]
    fn test_model_entry_new() {
        let entry = ModelEntry::new("test_model", 1024);
        assert_eq!(entry.name, "test_model");
        assert_eq!(entry.size, 1024);
        assert_eq!(entry.offset, 0);
        assert!(entry.components.is_empty());
    }

    #[test]
    fn test_model_entry_builder() {
        let entry = ModelEntry::new("model", 512)
            .with_offset(100)
            .with_component("layer1")
            .with_component("layer2");

        assert_eq!(entry.offset, 100);
        assert_eq!(entry.components.len(), 2);
    }

    #[test]
    fn test_bundle_manifest_new() {
        let manifest = BundleManifest::new();
        assert_eq!(manifest.version, super::super::BUNDLE_VERSION);
        assert!(manifest.is_empty());
        assert_eq!(manifest.len(), 0);
    }

    #[test]
    fn test_bundle_manifest_add_model() {
        let mut manifest = BundleManifest::new();
        manifest.add_model(ModelEntry::new("model1", 100));
        manifest.add_model(ModelEntry::new("model2", 200));

        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest.total_size(), 300);
        assert!(manifest.get_model("model1").is_some());
        assert!(manifest.get_model("model2").is_some());
        assert!(manifest.get_model("model3").is_none());
    }

    #[test]
    fn test_bundle_manifest_model_names() {
        let mut manifest = BundleManifest::new();
        manifest.add_model(ModelEntry::new("first", 10));
        manifest.add_model(ModelEntry::new("second", 20));

        let names = manifest.model_names();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "first");
        assert_eq!(names[1], "second");
    }

    #[test]
    fn test_bundle_manifest_serialization() {
        let mut manifest = BundleManifest::new().with_description("Test bundle");
        manifest.add_model(ModelEntry::new("model1", 100).with_offset(64));
        manifest.add_model(ModelEntry::new("model2", 200).with_offset(164));

        let bytes = manifest.to_bytes();
        let restored = BundleManifest::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(restored.version, manifest.version);
        assert_eq!(restored.description, "Test bundle");
        assert_eq!(restored.len(), 2);

        let model1 = restored.get_model("model1").expect("model1 not found");
        assert_eq!(model1.size, 100);
        assert_eq!(model1.offset, 64);

        let model2 = restored.get_model("model2").expect("model2 not found");
        assert_eq!(model2.size, 200);
        assert_eq!(model2.offset, 164);
    }

    #[test]
    fn test_bundle_manifest_iter() {
        let mut manifest = BundleManifest::new();
        manifest.add_model(ModelEntry::new("a", 1));
        manifest.add_model(ModelEntry::new("b", 2));
        manifest.add_model(ModelEntry::new("c", 3));

        let entries: Vec<_> = manifest.iter().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "a");
        assert_eq!(entries[1].name, "b");
        assert_eq!(entries[2].name, "c");
    }
}
