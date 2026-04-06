//! PEFT-compatible adapter export (adapter_model.safetensors + adapter_config.json)
//!
//! Produces output compatible with `peft.PeftModel.from_pretrained()`.

use super::error::AdapterError;
use super::peft_config::PeftAdapterConfig;
use crate::lora::LoRAConfig;
use crate::lora::LoRALayer;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::path::Path;

/// A bundle of LoRA adapters keyed by layer path
///
/// Collects multiple LoRA layer adapters and exports them in PEFT format.
pub struct PeftAdapterBundle {
    /// Adapters keyed by layer path (e.g., "model.layers.0.self_attn.q_proj")
    adapters: Vec<(String, AdapterWeights)>,
    /// LoRA configuration
    config: LoRAConfig,
    /// Base model name (for adapter_config.json)
    base_model: Option<String>,
}

/// Extracted adapter weights for a single layer
struct AdapterWeights {
    /// LoRA A matrix [rank, d_in]
    lora_a: Vec<f32>,
    /// LoRA B matrix [d_out, rank]
    lora_b: Vec<f32>,
    /// LoRA rank
    rank: usize,
    /// Input dimension
    d_in: usize,
    /// Output dimension
    d_out: usize,
}

impl PeftAdapterBundle {
    /// Create a new bundle with the given LoRA config
    pub fn new(config: LoRAConfig) -> Self {
        Self { adapters: Vec::new(), config, base_model: None }
    }

    /// Set the base model name
    pub fn with_base_model(mut self, name: impl Into<String>) -> Self {
        self.base_model = Some(name.into());
        self
    }

    /// Add a LoRA layer adapter with its full layer path
    ///
    /// The layer path should follow the model's naming convention, e.g.:
    /// `"model.layers.0.self_attn.q_proj"`
    pub fn add_adapter(&mut self, layer_path: impl Into<String>, layer: &LoRALayer) {
        let weights = AdapterWeights {
            lora_a: layer.lora_a().data().to_vec(),
            lora_b: layer.lora_b().data().to_vec(),
            rank: layer.rank(),
            d_in: layer.d_in(),
            d_out: layer.d_out(),
        };
        self.adapters.push((layer_path.into(), weights));
    }

    /// Add raw LoRA weights (for GPU pipeline where LoRALayer isn't available)
    pub fn add_raw_adapter(
        &mut self,
        layer_path: impl Into<String>,
        lora_a: Vec<f32>,
        lora_b: Vec<f32>,
        rank: usize,
        d_in: usize,
        d_out: usize,
    ) {
        self.adapters
            .push((layer_path.into(), AdapterWeights { lora_a, lora_b, rank, d_in, d_out }));
    }

    /// Save PEFT-compatible adapter to output directory
    ///
    /// Creates:
    /// - `adapter_config.json` — PEFT configuration
    /// - `adapter_model.safetensors` — adapter weights in PEFT naming convention
    pub fn save_peft(&self, output_dir: impl AsRef<Path>) -> Result<(), AdapterError> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        // Write adapter_config.json
        let peft_config =
            PeftAdapterConfig::from_lora_config(&self.config, self.base_model.as_deref());
        let config_json = peft_config.to_json().map_err(|e| AdapterError::Serialization(e))?;
        std::fs::write(output_dir.join("adapter_config.json"), config_json)?;

        // Build tensor data for safetensors
        // PEFT naming convention: "base_model.model.{layer_path}.lora_A.weight" / "lora_B.weight"
        let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

        for (layer_path, weights) in &self.adapters {
            // LoRA A: [rank, d_in]
            let a_name = format!("base_model.model.{layer_path}.lora_A.weight");
            let a_bytes: Vec<u8> = bytemuck::cast_slice(&weights.lora_a).to_vec();
            let a_shape = vec![weights.rank, weights.d_in];
            tensor_data.push((a_name, a_bytes, a_shape));

            // LoRA B: [d_out, rank]
            let b_name = format!("base_model.model.{layer_path}.lora_B.weight");
            let b_bytes: Vec<u8> = bytemuck::cast_slice(&weights.lora_b).to_vec();
            let b_shape = vec![weights.d_out, weights.rank];
            tensor_data.push((b_name, b_bytes, b_shape));
        }

        // Create TensorViews
        let views: Vec<(&str, TensorView<'_>)> = tensor_data
            .iter()
            .map(|(name, bytes, shape)| {
                let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .expect("TensorView construction must not fail for valid F32 data");
                (name.as_str(), view)
            })
            .collect();

        // Metadata
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "pt".to_string());

        let safetensor_bytes = safetensors::serialize(views, Some(metadata)).map_err(|e| {
            AdapterError::SafeTensors(format!("SafeTensors serialization failed: {e}"))
        })?;

        std::fs::write(output_dir.join("adapter_model.safetensors"), safetensor_bytes)?;

        Ok(())
    }

    /// Number of adapter layers in the bundle
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Check if bundle is empty
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::LoRALayer;
    use crate::Tensor;
    use tempfile::TempDir;

    fn make_test_layer(d_out: usize, d_in: usize, rank: usize) -> LoRALayer {
        let base_weight = Tensor::zeros(d_out * d_in, false);
        LoRALayer::new(base_weight, d_out, d_in, rank, 16.0)
    }

    #[test]
    fn test_bundle_creation() {
        let config = LoRAConfig::new(8, 16.0).target_qv_projections();
        let bundle = PeftAdapterBundle::new(config);
        assert!(bundle.is_empty());
        assert_eq!(bundle.len(), 0);
    }

    #[test]
    fn test_add_adapter() {
        let config = LoRAConfig::new(8, 16.0).target_qv_projections();
        let mut bundle = PeftAdapterBundle::new(config);

        let layer = make_test_layer(64, 64, 8);
        bundle.add_adapter("model.layers.0.self_attn.q_proj", &layer);

        assert_eq!(bundle.len(), 1);
        assert!(!bundle.is_empty());
    }

    #[test]
    fn test_save_peft_creates_files() {
        let config = LoRAConfig::new(4, 8.0).target_qv_projections();
        let mut bundle = PeftAdapterBundle::new(config).with_base_model("meta-llama/Llama-2-7b");

        let layer = make_test_layer(16, 16, 4);
        bundle.add_adapter("model.layers.0.self_attn.q_proj", &layer);

        let tmp = TempDir::new().expect("temp file creation should succeed");
        bundle.save_peft(tmp.path()).expect("save should succeed");

        // Verify files exist
        assert!(tmp.path().join("adapter_config.json").exists());
        assert!(tmp.path().join("adapter_model.safetensors").exists());
    }

    #[test]
    fn test_save_peft_config_content() {
        let config = LoRAConfig::new(16, 32.0).target_attention_projections();
        let bundle = PeftAdapterBundle::new(config).with_base_model("test/model");

        let tmp = TempDir::new().expect("temp file creation should succeed");
        bundle.save_peft(tmp.path()).expect("save should succeed");

        let json = std::fs::read_to_string(tmp.path().join("adapter_config.json"))
            .expect("file read should succeed");
        let parsed: PeftAdapterConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(parsed.peft_type, "LORA");
        assert_eq!(parsed.r, 16);
        assert_eq!(parsed.lora_alpha, 32.0);
        assert_eq!(parsed.base_model_name_or_path, Some("test/model".to_string()));
    }

    #[test]
    fn test_save_peft_safetensors_content() {
        let config = LoRAConfig::new(4, 8.0).target_qv_projections();
        let mut bundle = PeftAdapterBundle::new(config);

        let layer = make_test_layer(8, 16, 4);
        bundle.add_adapter("model.layers.0.self_attn.q_proj", &layer);

        let tmp = TempDir::new().expect("temp file creation should succeed");
        bundle.save_peft(tmp.path()).expect("save should succeed");

        // Load and verify safetensors
        let data = std::fs::read(tmp.path().join("adapter_model.safetensors"))
            .expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");

        let names = loaded.names();
        assert!(names.contains(&"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"));
        assert!(names.contains(&"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"));

        // Check shapes
        let lora_a = loaded
            .tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
            .expect("operation should succeed");
        assert_eq!(lora_a.shape(), &[4, 16]); // [rank, d_in]

        let lora_b = loaded
            .tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight")
            .expect("operation should succeed");
        assert_eq!(lora_b.shape(), &[8, 4]); // [d_out, rank]
    }

    #[test]
    fn test_save_peft_multiple_layers() {
        let config = LoRAConfig::new(4, 8.0).target_qv_projections();
        let mut bundle = PeftAdapterBundle::new(config);

        for i in 0..3 {
            let layer = make_test_layer(8, 8, 4);
            bundle.add_adapter(format!("model.layers.{i}.self_attn.q_proj"), &layer);
        }
        assert_eq!(bundle.len(), 3);

        let tmp = TempDir::new().expect("temp file creation should succeed");
        bundle.save_peft(tmp.path()).expect("save should succeed");

        let data = std::fs::read(tmp.path().join("adapter_model.safetensors"))
            .expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");
        // 3 layers * 2 matrices (A + B) = 6 tensors
        assert_eq!(loaded.len(), 6);
    }

    #[test]
    fn test_save_peft_empty_bundle() {
        let config = LoRAConfig::new(4, 8.0);
        let bundle = PeftAdapterBundle::new(config);

        let tmp = TempDir::new().expect("temp file creation should succeed");
        bundle.save_peft(tmp.path()).expect("save should succeed");

        // Should still create both files
        assert!(tmp.path().join("adapter_config.json").exists());
        assert!(tmp.path().join("adapter_model.safetensors").exists());
    }
}
