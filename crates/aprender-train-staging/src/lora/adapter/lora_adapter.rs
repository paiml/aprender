//! LoRA adapter serialization and deserialization
//!
//! Contains the main LoRAAdapter struct for saving and loading adapters.

use super::error::AdapterError;
use super::metadata::AdapterMetadata;
use crate::lora::LoRALayer;
use crate::Tensor;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Serializable LoRA adapter format
///
/// Contains all information needed to reconstruct a LoRA adapter
/// (excluding the base weight, which remains frozen and separate)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LoRAAdapter {
    /// Format version for future compatibility
    version: String,
    /// LoRA rank
    rank: usize,
    /// LoRA alpha parameter
    alpha: f32,
    /// Output dimension
    d_out: usize,
    /// Input dimension
    d_in: usize,
    /// Computed scale factor (alpha/rank)
    scale: f32,
    /// LoRA A matrix weights [rank * d_in]
    lora_a: Vec<f32>,
    /// LoRA B matrix weights [d_out * rank]
    lora_b: Vec<f32>,
}

impl LoRAAdapter {
    /// Current adapter format version
    const VERSION: &'static str = "1.0";

    /// Create adapter from LoRALayer
    ///
    /// # Arguments
    /// * `layer` - LoRALayer to extract adapter from
    /// * `rank` - LoRA rank
    /// * `alpha` - LoRA alpha parameter
    pub fn from_layer(layer: &LoRALayer, rank: usize, alpha: f32) -> Self {
        Self {
            version: Self::VERSION.to_string(),
            rank,
            alpha,
            d_out: layer.d_out(),
            d_in: layer.d_in(),
            scale: layer.scale(),
            lora_a: layer.lora_a().data().to_vec(),
            lora_b: layer.lora_b().data().to_vec(),
        }
    }

    /// Load adapter and apply to base weight
    ///
    /// # Arguments
    /// * `base_weight` - Frozen base weight tensor [d_out * d_in]
    ///
    /// # Returns
    /// LoRALayer with loaded adapter weights
    pub fn to_layer(&self, base_weight: Tensor) -> Result<LoRALayer, AdapterError> {
        // Validate dimensions
        if base_weight.len() != self.d_out * self.d_in {
            return Err(AdapterError::DimensionMismatch {
                expected: format!("{}x{} = {}", self.d_out, self.d_in, self.d_out * self.d_in),
                actual: base_weight.len().to_string(),
            });
        }

        if self.lora_a.len() != self.rank * self.d_in {
            return Err(AdapterError::Validation(format!(
                "LoRA A size mismatch: expected {} (rank {} * d_in {}), got {}",
                self.rank * self.d_in,
                self.rank,
                self.d_in,
                self.lora_a.len()
            )));
        }

        if self.lora_b.len() != self.d_out * self.rank {
            return Err(AdapterError::Validation(format!(
                "LoRA B size mismatch: expected {} (d_out {} * rank {}), got {}",
                self.d_out * self.rank,
                self.d_out,
                self.rank,
                self.lora_b.len()
            )));
        }

        // Create layer with loaded weights
        let mut layer = LoRALayer::new(base_weight, self.d_out, self.d_in, self.rank, self.alpha);

        // Replace LoRA weights with loaded values
        *layer.lora_a_mut().data_mut() = ndarray::arr1(&self.lora_a);
        *layer.lora_b_mut().data_mut() = ndarray::arr1(&self.lora_b);

        Ok(layer)
    }

    /// Save adapter to JSON file
    ///
    /// # Arguments
    /// * `path` - File path to save to
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), AdapterError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load adapter from JSON file
    ///
    /// # Arguments
    /// * `path` - File path to load from
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, AdapterError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let adapter: LoRAAdapter = serde_json::from_reader(reader)?;

        // Validate version
        if adapter.version != Self::VERSION {
            return Err(AdapterError::Validation(format!(
                "Unsupported adapter version: {} (expected {})",
                adapter.version,
                Self::VERSION
            )));
        }

        Ok(adapter)
    }

    /// Get adapter metadata
    pub fn metadata(&self) -> AdapterMetadata {
        AdapterMetadata {
            version: self.version.clone(),
            rank: self.rank,
            alpha: self.alpha,
            d_out: self.d_out,
            d_in: self.d_in,
            scale: self.scale,
            num_params: self.lora_a.len() + self.lora_b.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_test_adapter() -> LoRAAdapter {
        LoRAAdapter {
            version: "1.0".to_string(),
            rank: 4,
            alpha: 8.0,
            d_out: 8,
            d_in: 16,
            scale: 2.0,
            lora_a: vec![0.1; 4 * 16], // rank * d_in
            lora_b: vec![0.2; 8 * 4],  // d_out * rank
        }
    }

    #[test]
    fn test_adapter_from_layer() {
        let base_weight = Tensor::zeros(8 * 16, false);
        let layer = LoRALayer::new(base_weight, 8, 16, 4, 8.0);
        let adapter = LoRAAdapter::from_layer(&layer, 4, 8.0);
        assert_eq!(adapter.rank, 4);
        assert_eq!(adapter.alpha, 8.0);
        assert_eq!(adapter.d_out, 8);
        assert_eq!(adapter.d_in, 16);
    }

    #[test]
    fn test_adapter_to_layer_valid() {
        let adapter = make_test_adapter();
        let base_weight = Tensor::zeros(8 * 16, false);
        let layer = adapter.to_layer(base_weight).expect("operation should succeed");
        assert_eq!(layer.d_out(), 8);
        assert_eq!(layer.d_in(), 16);
    }

    #[test]
    fn test_adapter_to_layer_dimension_mismatch() {
        let adapter = make_test_adapter();
        let base_weight = Tensor::zeros(100, false); // Wrong size
        let result = adapter.to_layer(base_weight);
        assert!(result.is_err());
        match result {
            Err(AdapterError::DimensionMismatch { .. }) => {}
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_adapter_to_layer_lora_a_mismatch() {
        let mut adapter = make_test_adapter();
        adapter.lora_a = vec![0.1; 10]; // Wrong size
        let base_weight = Tensor::zeros(8 * 16, false);
        let result = adapter.to_layer(base_weight);
        assert!(result.is_err());
        match result {
            Err(AdapterError::Validation(msg)) => {
                assert!(msg.contains("LoRA A size mismatch"));
            }
            _ => panic!("Expected Validation error"),
        }
    }

    #[test]
    fn test_adapter_to_layer_lora_b_mismatch() {
        let mut adapter = make_test_adapter();
        adapter.lora_b = vec![0.2; 10]; // Wrong size
        let base_weight = Tensor::zeros(8 * 16, false);
        let result = adapter.to_layer(base_weight);
        assert!(result.is_err());
        match result {
            Err(AdapterError::Validation(msg)) => {
                assert!(msg.contains("LoRA B size mismatch"));
            }
            _ => panic!("Expected Validation error"),
        }
    }

    #[test]
    fn test_adapter_save_load_roundtrip() {
        let adapter = make_test_adapter();
        let file = NamedTempFile::new().expect("temp file creation should succeed");

        adapter.save(file.path()).expect("save should succeed");
        let loaded = LoRAAdapter::load(file.path()).expect("load should succeed");

        assert_eq!(adapter.rank, loaded.rank);
        assert_eq!(adapter.alpha, loaded.alpha);
        assert_eq!(adapter.d_out, loaded.d_out);
        assert_eq!(adapter.d_in, loaded.d_in);
        assert_eq!(adapter.lora_a.len(), loaded.lora_a.len());
        assert_eq!(adapter.lora_b.len(), loaded.lora_b.len());
    }

    #[test]
    fn test_adapter_load_invalid_version() {
        let mut adapter = make_test_adapter();
        adapter.version = "0.0".to_string();
        let file = NamedTempFile::new().expect("temp file creation should succeed");
        adapter.save(file.path()).expect("save should succeed");

        let result = LoRAAdapter::load(file.path());
        assert!(result.is_err());
        match result {
            Err(AdapterError::Validation(msg)) => {
                assert!(msg.contains("Unsupported adapter version"));
            }
            _ => panic!("Expected Validation error"),
        }
    }

    #[test]
    fn test_adapter_load_nonexistent_file() {
        let result = LoRAAdapter::load("/nonexistent/path/adapter.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_adapter_save_invalid_path() {
        let adapter = make_test_adapter();
        let result = adapter.save("/nonexistent/dir/adapter.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_adapter_metadata() {
        let adapter = make_test_adapter();
        let meta = adapter.metadata();
        assert_eq!(meta.rank, 4);
        assert_eq!(meta.alpha, 8.0);
        assert_eq!(meta.d_out, 8);
        assert_eq!(meta.d_in, 16);
        assert_eq!(meta.num_params, 4 * 16 + 8 * 4);
    }

    #[test]
    fn test_adapter_clone() {
        let adapter = make_test_adapter();
        let cloned = adapter.clone();
        assert_eq!(adapter.rank, cloned.rank);
        assert_eq!(adapter.lora_a.len(), cloned.lora_a.len());
    }

    #[test]
    fn test_adapter_debug() {
        let adapter = make_test_adapter();
        let debug = format!("{adapter:?}");
        assert!(debug.contains("LoRAAdapter"));
    }
}
