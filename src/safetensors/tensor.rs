
// ============================================================================
// Sharded SafeTensors Model (GH-213)
// ============================================================================

/// JSON structure for model.safetensors.index.json
#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    /// Mapping from tensor name to shard filename
    weight_map: HashMap<String, String>,
}

/// Sharded SafeTensors model container (GH-213)
///
/// Loads models split across multiple `.safetensors` shard files,
/// as produced by HuggingFace for models >3B parameters.
///
/// # Format
///
/// Sharded models have a `model.safetensors.index.json` that maps
/// tensor names to shard filenames (e.g., `model-00001-of-00002.safetensors`).
///
/// # Example
///
/// ```rust,ignore
/// let model = ShardedSafeTensorsModel::load_from_index("model.safetensors.index.json")?;
/// let weights = model.get_tensor_auto("model.layers.0.self_attn.q_proj.weight")?;
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct ShardedSafeTensorsModel {
    /// Memory-mapped shard files
    shards: Vec<MappedSafeTensorsModel>,
    /// Mapping from tensor name to shard index in `shards`
    tensor_to_shard: HashMap<String, usize>,
    /// Base directory path (parent of index.json)
    base_path: std::path::PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl ShardedSafeTensorsModel {
    /// Load a sharded SafeTensors model from its index.json file
    ///
    /// Parses the index.json, discovers unique shard files, and mmaps each one.
    ///
    /// # Arguments
    ///
    /// * `index_path` - Path to `model.safetensors.index.json`
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - index.json cannot be read or parsed
    /// - Any shard file cannot be opened or mmapped
    pub fn load_from_index(index_path: &std::path::Path) -> Result<Self> {
        let base_path = index_path
            .parent()
            .ok_or_else(|| RealizarError::IoError {
                message: format!(
                    "Cannot determine parent directory of '{}'",
                    index_path.display()
                ),
            })?
            .to_path_buf();

        // Parse index.json
        let index_content =
            std::fs::read_to_string(index_path).map_err(|e| RealizarError::IoError {
                message: format!(
                    "Failed to read index file '{}': {}",
                    index_path.display(),
                    e
                ),
            })?;

        let index: SafetensorsIndex =
            serde_json::from_str(&index_content).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to parse index.json: {}", e),
            })?;

        // Discover unique shard filenames (preserving order)
        let mut shard_filenames: Vec<String> = Vec::new();
        let mut filename_to_idx: HashMap<String, usize> = HashMap::new();

        for shard_file in index.weight_map.values() {
            if !filename_to_idx.contains_key(shard_file) {
                let idx = shard_filenames.len();
                filename_to_idx.insert(shard_file.clone(), idx);
                shard_filenames.push(shard_file.clone());
            }
        }

        // Load each shard via mmap
        let mut shards = Vec::with_capacity(shard_filenames.len());
        for filename in &shard_filenames {
            let shard_path = base_path.join(filename);
            let shard = MappedSafeTensorsModel::load(&shard_path)?;
            shards.push(shard);
        }

        // Build tensor-to-shard lookup
        let mut tensor_to_shard = HashMap::with_capacity(index.weight_map.len());
        for (tensor_name, shard_file) in &index.weight_map {
            let shard_idx = filename_to_idx[shard_file];
            tensor_to_shard.insert(tensor_name.clone(), shard_idx);
        }

        Ok(Self {
            shards,
            tensor_to_shard,
            base_path,
        })
    }

    /// Get tensor as F32 with automatic dtype conversion (routes to correct shard)
    ///
    /// Supports F32, F16, and BF16 dtypes with automatic conversion to F32.
    pub fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        let shard_idx =
            self.tensor_to_shard
                .get(name)
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "get_tensor_auto".to_string(),
                    reason: format!("Tensor '{}' not found in sharded model", name),
                })?;

        self.shards[*shard_idx].get_tensor_auto(name)
    }

    /// Get list of all tensor names across all shards
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_to_shard.keys().map(String::as_str).collect()
    }

    /// Get tensor info by name (routes to correct shard)
    #[must_use]
    pub fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        let shard_idx = self.tensor_to_shard.get(name)?;
        self.shards[*shard_idx].get_tensor_info(name)
    }

    /// Check if model has a tensor with given name
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_to_shard.contains_key(name)
    }

    /// Get the base directory path
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.base_path
    }

    /// Get total number of tensors across all shards
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        contract_pre_tensor_count!();
        self.tensor_to_shard.len()
    }

    /// Get number of shard files
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

// PMAT-234/235: Validation contract - makes bad loads IMPOSSIBLE
// Implements Poka-Yoke (mistake-proofing) via newtype pattern
pub mod validation;
pub use validation::{
    // Runtime validation functions (legacy API)
    enforce_embedding_validation,
    enforce_weight_validation,
    validate_embedding,
    validate_weight,
    // Compile-time enforcement via newtypes (PMAT-235)
    ContractValidationError,
    TensorStats,
    ValidatedAprTransformer,
    ValidatedEmbedding,
    ValidatedVector,
    ValidatedWeight,
    ValidationResult,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "tests_find_sibling.rs"]
mod safetensors_tests_part_02;
