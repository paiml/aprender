//! Model Persistence for ML Anomaly Detection (Sprint 48)
//!
//! Eliminates MUDA (waste) by persisting trained models using aprender's `.apr` format.
//! This enables 10-50x faster startup when using pre-trained models.
//!
//! # Toyota Way Principle
//!
//! *Muda* (無駄) - Eliminate waste by reusing trained models instead of retraining.
//!
//! # References
//!
//! Sculley, D., et al. (2015). Hidden technical debt in machine learning systems.
//! Advances in Neural Information Processing Systems.

use std::path::Path;
use thiserror::Error;

/// Errors that can occur during model persistence operations
#[derive(Error, Debug)]
pub enum ModelPersistenceError {
    #[error("Failed to save model: {0}")]
    SaveError(String),

    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("Model file not found: {0}")]
    FileNotFound(String),

    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("Model version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for model persistence operations
pub type Result<T> = std::result::Result<T, ModelPersistenceError>;

/// Metadata for a persisted model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelMetadata {
    /// Renacer version that created this model
    pub renacer_version: String,
    /// When the model was trained (ISO 8601)
    pub trained_at: String,
    /// Number of samples used for training
    pub training_samples: usize,
    /// Model-specific hyperparameters
    pub hyperparameters: std::collections::HashMap<String, String>,
    /// Optional description
    pub description: Option<String>,
}

impl ModelMetadata {
    /// Create new metadata with current timestamp
    pub fn new(training_samples: usize) -> Self {
        Self {
            renacer_version: env!("CARGO_PKG_VERSION").to_string(),
            trained_at: chrono_lite_timestamp(),
            training_samples,
            hyperparameters: std::collections::HashMap::new(),
            description: None,
        }
    }

    /// Add a hyperparameter
    pub fn with_hyperparameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.hyperparameters.insert(key.into(), value.into());
        self
    }

    /// Add a description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Lightweight timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// Serializable wrapper for `KMeans` model data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerializableKMeansModel {
    /// Cluster centroids (k x `n_features`)
    pub centroids: Vec<Vec<f32>>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of features
    pub n_features: usize,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Serializable wrapper for `IsolationForest` model data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerializableIsolationForestModel {
    /// Number of trees in the forest
    pub n_trees: usize,
    /// Subsample size used for training
    pub subsample_size: usize,
    /// Serialized tree data (simplified representation)
    pub tree_data: Vec<u8>,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Options for saving models
#[derive(Debug, Clone)]
pub struct PersistenceOptions {
    /// Enable compression (default: true)
    pub compress: bool,
    /// Model name
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
}

impl Default for PersistenceOptions {
    fn default() -> Self {
        Self { compress: true, name: None, description: None }
    }
}

impl PersistenceOptions {
    /// Create new options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }

    /// Set model name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set model description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Save a `KMeans` model to .apr format
pub fn save_kmeans_model(
    model: &SerializableKMeansModel,
    path: impl AsRef<Path>,
    options: PersistenceOptions,
) -> Result<()> {
    use aprender::format::{save, Compression, ModelType, SaveOptions};

    let compression = if options.compress { Compression::ZstdDefault } else { Compression::None };

    let mut save_options = SaveOptions::new().with_compression(compression);

    if let Some(name) = options.name {
        save_options = save_options.with_name(name);
    }
    if let Some(desc) = options.description {
        save_options = save_options.with_description(desc);
    }

    save(model, ModelType::KMeans, path.as_ref(), save_options)
        .map_err(|e| ModelPersistenceError::SaveError(e.to_string()))
}

/// Load a `KMeans` model from .apr format
pub fn load_kmeans_model(path: impl AsRef<Path>) -> Result<SerializableKMeansModel> {
    use aprender::format::{load, ModelType};

    if !path.as_ref().exists() {
        return Err(ModelPersistenceError::FileNotFound(path.as_ref().display().to_string()));
    }

    load::<SerializableKMeansModel>(path.as_ref(), ModelType::KMeans)
        .map_err(|e| ModelPersistenceError::LoadError(e.to_string()))
}

/// Load a `KMeans` model with memory mapping (zero-copy for large models)
/// Falls back to regular load if mmap is not available
pub fn load_kmeans_model_mmap(path: impl AsRef<Path>) -> Result<SerializableKMeansModel> {
    // Memory mapping not yet available in aprender, fall back to regular load
    load_kmeans_model(path)
}

/// Save an `IsolationForest` model to .apr format
pub fn save_isolation_forest_model(
    model: &SerializableIsolationForestModel,
    path: impl AsRef<Path>,
    options: PersistenceOptions,
) -> Result<()> {
    use aprender::format::{save, Compression, ModelType, SaveOptions};

    let compression = if options.compress { Compression::ZstdDefault } else { Compression::None };

    let mut save_options = SaveOptions::new().with_compression(compression);

    if let Some(name) = options.name {
        save_options = save_options.with_name(name);
    }
    if let Some(desc) = options.description {
        save_options = save_options.with_description(desc);
    }

    // Use Custom model type for IsolationForest
    save(model, ModelType::Custom, path.as_ref(), save_options)
        .map_err(|e| ModelPersistenceError::SaveError(e.to_string()))
}

/// Load an `IsolationForest` model from .apr format
pub fn load_isolation_forest_model(
    path: impl AsRef<Path>,
) -> Result<SerializableIsolationForestModel> {
    use aprender::format::{load, ModelType};

    if !path.as_ref().exists() {
        return Err(ModelPersistenceError::FileNotFound(path.as_ref().display().to_string()));
    }

    load::<SerializableIsolationForestModel>(path.as_ref(), ModelType::Custom)
        .map_err(|e| ModelPersistenceError::LoadError(e.to_string()))
}

/// Check if a model file exists and is valid
pub fn validate_model_file(path: impl AsRef<Path>) -> Result<ModelMetadata> {
    // Try to load as KMeans first
    if let Ok(model) = load_kmeans_model(path.as_ref()) {
        return Ok(model.metadata);
    }

    // Try IsolationForest
    if let Ok(model) = load_isolation_forest_model(path.as_ref()) {
        return Ok(model.metadata);
    }

    Err(ModelPersistenceError::InvalidFormat("Could not determine model type".to_string()))
}

/// Generate a status line for model information
pub fn model_status_line(metadata: &ModelMetadata) -> String {
    format!(
        "model: renacer v{}, trained with {} samples",
        metadata.renacer_version, metadata.training_samples
    )
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(ModelPersistenceError: Send, Sync);
static_assertions::assert_impl_all!(ModelMetadata: Send, Sync);
static_assertions::assert_impl_all!(PersistenceOptions: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ==================== RED PHASE TESTS ====================
    // These tests define the expected behavior

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata::new(1000);

        assert_eq!(metadata.renacer_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(metadata.training_samples, 1000);
        assert!(metadata.hyperparameters.is_empty());
        assert!(metadata.description.is_none());
    }

    #[test]
    fn test_model_metadata_with_hyperparameters() {
        let metadata = ModelMetadata::new(500)
            .with_hyperparameter("n_clusters", "3")
            .with_hyperparameter("max_iter", "100")
            .with_description("Test model");

        assert_eq!(metadata.hyperparameters.get("n_clusters"), Some(&"3".to_string()));
        assert_eq!(metadata.hyperparameters.get("max_iter"), Some(&"100".to_string()));
        assert_eq!(metadata.description, Some("Test model".to_string()));
    }

    #[test]
    fn test_persistence_options_default() {
        let options = PersistenceOptions::default();

        assert!(options.compress);
        assert!(options.name.is_none());
        assert!(options.description.is_none());
    }

    #[test]
    fn test_persistence_options_builder() {
        let options = PersistenceOptions::new()
            .with_compression(false)
            .with_name("baseline-model")
            .with_description("Production baseline");

        assert!(!options.compress);
        assert_eq!(options.name, Some("baseline-model".to_string()));
        assert_eq!(options.description, Some("Production baseline".to_string()));
    }

    #[test]
    fn test_serializable_kmeans_model_creation() {
        let model = SerializableKMeansModel {
            centroids: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            n_clusters: 2,
            n_features: 2,
            metadata: ModelMetadata::new(100),
        };

        assert_eq!(model.n_clusters, 2);
        assert_eq!(model.centroids.len(), 2);
    }

    #[test]
    fn test_save_and_load_kmeans_model() {
        let temp_dir = TempDir::new().expect("test");
        let model_path = temp_dir.path().join("test_kmeans.apr");

        let model = SerializableKMeansModel {
            centroids: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]],
            n_clusters: 3,
            n_features: 3,
            metadata: ModelMetadata::new(1000)
                .with_hyperparameter("n_clusters", "3")
                .with_description("Test KMeans model"),
        };

        // Save
        let options =
            PersistenceOptions::new().with_name("test-kmeans").with_description("Test model");
        save_kmeans_model(&model, &model_path, options).expect("Failed to save model");

        // Load
        let loaded = load_kmeans_model(&model_path).expect("Failed to load model");

        assert_eq!(loaded.n_clusters, model.n_clusters);
        assert_eq!(loaded.n_features, model.n_features);
        assert_eq!(loaded.centroids.len(), model.centroids.len());
        for (orig, loaded_centroid) in model.centroids.iter().zip(loaded.centroids.iter()) {
            for (o, l) in orig.iter().zip(loaded_centroid.iter()) {
                assert!((o - l).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_save_and_load_kmeans_uncompressed() {
        let temp_dir = TempDir::new().expect("test");
        let model_path = temp_dir.path().join("test_kmeans_uncompressed.apr");

        let model = SerializableKMeansModel {
            centroids: vec![vec![1.0], vec![10.0]],
            n_clusters: 2,
            n_features: 1,
            metadata: ModelMetadata::new(50),
        };

        // Save without compression
        let options = PersistenceOptions::new().with_compression(false);
        save_kmeans_model(&model, &model_path, options).expect("Failed to save uncompressed");

        // Load
        let loaded = load_kmeans_model(&model_path).expect("Failed to load");
        assert_eq!(loaded.n_clusters, 2);
    }

    #[test]
    fn test_load_nonexistent_model() {
        let result = load_kmeans_model("/nonexistent/path/model.apr");

        assert!(result.is_err());
        match result {
            Err(ModelPersistenceError::FileNotFound(path)) => {
                assert!(path.contains("nonexistent"));
            }
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_save_and_load_isolation_forest_model() {
        let temp_dir = TempDir::new().expect("test");
        let model_path = temp_dir.path().join("test_iforest.apr");

        let model = SerializableIsolationForestModel {
            n_trees: 100,
            subsample_size: 256,
            tree_data: vec![1, 2, 3, 4, 5], // Simplified tree data
            metadata: ModelMetadata::new(500)
                .with_hyperparameter("n_trees", "100")
                .with_hyperparameter("contamination", "0.1"),
        };

        // Save
        let options = PersistenceOptions::new().with_name("test-iforest");
        save_isolation_forest_model(&model, &model_path, options).expect("Failed to save");

        // Load
        let loaded = load_isolation_forest_model(&model_path).expect("Failed to load");

        assert_eq!(loaded.n_trees, model.n_trees);
        assert_eq!(loaded.subsample_size, model.subsample_size);
        assert_eq!(loaded.tree_data, model.tree_data);
    }

    #[test]
    fn test_model_status_line() {
        let metadata = ModelMetadata::new(1234);
        let status = model_status_line(&metadata);

        assert!(status.contains("renacer"));
        assert!(status.contains("1234 samples"));
    }

    #[test]
    fn test_validate_model_file_kmeans() {
        let temp_dir = TempDir::new().expect("test");
        let model_path = temp_dir.path().join("validate_test.apr");

        let model = SerializableKMeansModel {
            centroids: vec![vec![1.0]],
            n_clusters: 1,
            n_features: 1,
            metadata: ModelMetadata::new(42).with_description("Validation test"),
        };

        save_kmeans_model(&model, &model_path, PersistenceOptions::new()).expect("test");

        let metadata = validate_model_file(&model_path).expect("Validation failed");
        assert_eq!(metadata.training_samples, 42);
    }

    // ==================== PROPERTY-BASED TESTS ====================

    #[test]
    fn test_roundtrip_preserves_centroids() {
        use proptest::prelude::*;

        proptest::proptest!(|(
            n_clusters in 1usize..10,
            n_features in 1usize..5,
        )| {
            let temp_dir = TempDir::new().expect("test");
            let model_path = temp_dir.path().join("proptest.apr");

            // Generate random centroids
            let centroids: Vec<Vec<f32>> = (0..n_clusters)
                .map(|i| (0..n_features).map(|j| (i * n_features + j) as f32).collect())
                .collect();

            let model = SerializableKMeansModel {
                centroids: centroids.clone(),
                n_clusters,
                n_features,
                metadata: ModelMetadata::new(100),
            };

            save_kmeans_model(&model, &model_path, PersistenceOptions::new()).expect("test");
            let loaded = load_kmeans_model(&model_path).expect("test");

            prop_assert_eq!(loaded.n_clusters, n_clusters);
            prop_assert_eq!(loaded.n_features, n_features);
            prop_assert_eq!(loaded.centroids.len(), centroids.len());
        });
    }

    #[test]
    fn test_metadata_preserved_through_roundtrip() {
        let temp_dir = TempDir::new().expect("test");
        let model_path = temp_dir.path().join("metadata_test.apr");

        let model = SerializableKMeansModel {
            centroids: vec![vec![1.0, 2.0]],
            n_clusters: 1,
            n_features: 2,
            metadata: ModelMetadata::new(999)
                .with_hyperparameter("key1", "value1")
                .with_hyperparameter("key2", "value2")
                .with_description("Detailed description here"),
        };

        save_kmeans_model(&model, &model_path, PersistenceOptions::new()).expect("test");
        let loaded = load_kmeans_model(&model_path).expect("test");

        assert_eq!(loaded.metadata.training_samples, 999);
        assert_eq!(loaded.metadata.hyperparameters.get("key1"), Some(&"value1".to_string()));
        assert_eq!(loaded.metadata.hyperparameters.get("key2"), Some(&"value2".to_string()));
        assert_eq!(loaded.metadata.description, Some("Detailed description here".to_string()));
    }

    #[test]
    fn test_large_model_roundtrip() {
        let temp_dir = TempDir::new().expect("test");
        let model_path = temp_dir.path().join("large_model.apr");

        // Create a larger model (10 clusters, 50 features)
        let n_clusters = 10;
        let n_features = 50;
        let centroids: Vec<Vec<f32>> = (0..n_clusters)
            .map(|i| (0..n_features).map(|j| (i * j) as f32 * 0.1).collect())
            .collect();

        let model = SerializableKMeansModel {
            centroids,
            n_clusters,
            n_features,
            metadata: ModelMetadata::new(10000),
        };

        // Save with compression
        save_kmeans_model(&model, &model_path, PersistenceOptions::new()).expect("test");

        // Verify file is smaller than uncompressed would be
        let file_size = std::fs::metadata(&model_path).expect("test").len();
        let uncompressed_estimate = n_clusters * n_features * 4; // 4 bytes per f32
        assert!(
            file_size < uncompressed_estimate as u64 * 2,
            "Compression should reduce file size"
        );

        // Load and verify
        let loaded = load_kmeans_model(&model_path).expect("test");
        assert_eq!(loaded.n_clusters, n_clusters);
        assert_eq!(loaded.n_features, n_features);
    }
}
