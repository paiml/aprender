//! Model registry for multi-model serving
//!
//! Provides a central registry to manage multiple models in production environments.
//! Supports dynamic model loading/unloading, caching, and thread-safe concurrent access.
//!
//! ## Features
//!
//! - Thread-safe model registry with concurrent access
//! - Integration with `ModelCache` for efficient memory management
//! - Support for multiple model formats (GGUF, Safetensors)
//! - Model metadata and configuration
//! - Graceful loading/unloading with error handling
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::registry::{ModelRegistry, ModelConfig};
//!
//! let registry = ModelRegistry::new(5); // Max 5 models cached
//! registry.register("llama-7b", model, tokenizer)?;
//! let model = registry.get("llama-7b")?;
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use arc_swap::ArcSwap;

use crate::{
    cache::ModelCache,
    error::{RealizarError, Result},
    layers::Model,
    tokenizer::BPETokenizer,
};

/// Information about a registered model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier
    pub id: String,
    /// Human-readable model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model format (GGUF, Safetensors, etc.)
    pub format: String,
    /// Whether the model is currently loaded
    pub loaded: bool,
}

/// Entry in the model registry
#[derive(Clone)]
struct ModelEntry {
    /// Model instance
    model: Arc<Model>,
    /// Tokenizer for this model
    tokenizer: Arc<BPETokenizer>,
    /// Model metadata
    info: ModelInfo,
}

/// Type alias for the models map (immutable snapshot)
type ModelsMap = HashMap<String, ModelEntry>;

/// Type alias for model and tokenizer tuple
type ModelTuple = (Arc<Model>, Arc<BPETokenizer>);

/// Central registry for managing multiple models
///
/// The `ModelRegistry` provides thread-safe access to multiple models,
/// with automatic caching and lifecycle management.
///
/// Uses `ArcSwap` for lock-free reads (per `McKenney` 2011).
pub struct ModelRegistry {
    /// Registry of loaded models (lock-free reads via `ArcSwap`)
    models: ArcSwap<ModelsMap>,
    /// Write lock to serialize modifications
    write_lock: Mutex<()>,
    /// Model cache for memory management (reserved for future use)
    #[allow(dead_code)]
    cache: Arc<ModelCache>,
}

impl ModelRegistry {
    /// Create a new model registry
    ///
    /// # Arguments
    ///
    /// * `cache_capacity` - Maximum number of models to keep in cache
    #[must_use]
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            models: ArcSwap::from_pointee(HashMap::new()),
            write_lock: Mutex::new(()),
            cache: Arc::new(ModelCache::new(cache_capacity)),
        }
    }

    /// Register a new model
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the model
    /// * `model` - Model instance
    /// * `tokenizer` - Tokenizer for the model
    ///
    /// # Errors
    ///
    /// Returns error if model ID already exists
    pub fn register(&self, id: &str, model: Model, tokenizer: BPETokenizer) -> Result<()> {
        let _guard = self.write_lock.lock().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        let current = self.models.load();
        if current.contains_key(id) {
            return Err(RealizarError::ModelAlreadyExists(id.to_string()));
        }

        let entry = ModelEntry {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            info: ModelInfo {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                format: "unknown".to_string(),
                loaded: true,
            },
        };

        let mut new_map: ModelsMap = (**current).clone();
        new_map.insert(id.to_string(), entry);
        self.models.store(Arc::new(new_map));
        Ok(())
    }

    /// Register a model with full metadata
    ///
    /// # Arguments
    ///
    /// * `info` - Model metadata
    /// * `model` - Model instance
    /// * `tokenizer` - Tokenizer for the model
    ///
    /// # Errors
    ///
    /// Returns error if model ID already exists
    pub fn register_with_info(
        &self,
        mut info: ModelInfo,
        model: Model,
        tokenizer: BPETokenizer,
    ) -> Result<()> {
        let _guard = self.write_lock.lock().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        let current = self.models.load();
        if current.contains_key(&info.id) {
            return Err(RealizarError::ModelAlreadyExists(info.id));
        }

        info.loaded = true;
        let entry = ModelEntry {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            info,
        };

        let id = entry.info.id.clone();
        let mut new_map: ModelsMap = (**current).clone();
        new_map.insert(id, entry);
        self.models.store(Arc::new(new_map));
        Ok(())
    }

    /// Get a model by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Model identifier
    ///
    /// # Errors
    ///
    /// Returns error if model not found
    pub fn get(&self, id: &str) -> Result<ModelTuple> {
        // Lock-free read via ArcSwap::load()
        let models = self.models.load();

        let entry = models
            .get(id)
            .ok_or_else(|| RealizarError::ModelNotFound(id.to_string()))?;

        Ok((Arc::clone(&entry.model), Arc::clone(&entry.tokenizer)))
    }

    /// Get model info by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Model identifier
    ///
    /// # Errors
    ///
    /// Returns error if model not found
    pub fn get_info(&self, id: &str) -> Result<ModelInfo> {
        let models = self.models.load();

        let entry = models
            .get(id)
            .ok_or_else(|| RealizarError::ModelNotFound(id.to_string()))?;

        Ok(entry.info.clone())
    }

    /// List all registered models (lock-free)
    #[must_use]
    pub fn list(&self) -> Vec<ModelInfo> {
        let models = self.models.load();
        models.values().map(|entry| entry.info.clone()).collect()
    }

    /// Unregister a model
    ///
    /// # Arguments
    ///
    /// * `id` - Model identifier
    ///
    /// # Errors
    ///
    /// Returns error if model not found
    pub fn unregister(&self, id: &str) -> Result<()> {
        let _guard = self.write_lock.lock().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        let current = self.models.load();
        if !current.contains_key(id) {
            return Err(RealizarError::ModelNotFound(id.to_string()));
        }

        let mut new_map: ModelsMap = (**current).clone();
        new_map.remove(id);
        self.models.store(Arc::new(new_map));
        Ok(())
    }

    /// Atomically replace a model (for hot-reload)
    ///
    /// # Arguments
    ///
    /// * `id` - Model identifier to replace
    /// * `model` - New model instance
    /// * `tokenizer` - New tokenizer instance
    ///
    /// # Errors
    ///
    /// Returns error if model not found or lock acquisition fails
    pub fn replace(&self, id: &str, model: Model, tokenizer: BPETokenizer) -> Result<()> {
        let _guard = self.write_lock.lock().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        let current = self.models.load();
        if !current.contains_key(id) {
            return Err(RealizarError::ModelNotFound(id.to_string()));
        }

        // Get existing info to preserve metadata
        let existing_info = current.get(id).map_or_else(
            || ModelInfo {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                format: "unknown".to_string(),
                loaded: true,
            },
            |e| e.info.clone(),
        );

        let entry = ModelEntry {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            info: existing_info,
        };

        let mut new_map: ModelsMap = (**current).clone();
        new_map.insert(id.to_string(), entry);
        self.models.store(Arc::new(new_map));
        Ok(())
    }

    /// Check if a model is registered (lock-free)
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        let models = self.models.load();
        models.contains_key(id)
    }

    /// Get the number of registered models (lock-free)
    #[must_use]
    pub fn len(&self) -> usize {
        let models = self.models.load();
        models.len()
    }

    /// Check if the registry is empty (lock-free)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

include!("registry_get_info.rs");
