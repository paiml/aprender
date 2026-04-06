//! Merged model export — merge LoRA/QLoRA adapters into base weights and export
//!
//! Supports merging adapters back into the base model and collecting the
//! merged weights. When the `hub` feature is enabled, also supports direct
//! export to SafeTensors and GGUF formats.

use super::error::AdapterError;
use crate::lora::LoRALayer;
use crate::lora::QLoRALayer;
use std::collections::HashMap;
use std::path::Path;

/// Merged model from combining LoRA/QLoRA adapters with base weights
pub struct MergedModel {
    /// Tensor data by name (merged base + adapter)
    pub tensors: HashMap<String, Vec<f32>>,
    /// Tensor shapes by name
    pub shapes: HashMap<String, Vec<usize>>,
    /// Number of layers merged
    pub layers_merged: usize,
}

impl MergedModel {
    /// Total parameter count
    pub fn param_count(&self) -> u64 {
        self.tensors.values().map(|t| t.len() as u64).sum()
    }

    /// Save merged model as SafeTensors
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<(), AdapterError> {
        use safetensors::tensor::{Dtype, TensorView};

        let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = self
            .tensors
            .iter()
            .map(|(name, data)| {
                let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
                let shape = self.shapes.get(name).cloned().unwrap_or_else(|| vec![data.len()]);
                (name.clone(), bytes, shape)
            })
            .collect();

        let views: Vec<(&str, TensorView<'_>)> = tensor_data
            .iter()
            .map(|(name, bytes, shape)| {
                let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .expect("TensorView construction must not fail for valid F32 data");
                (name.as_str(), view)
            })
            .collect();

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("format".to_string(), "entrenar-merged".to_string());

        let safetensor_bytes = safetensors::serialize(views, Some(metadata))
            .map_err(|e| AdapterError::SafeTensors(format!("Serialization failed: {e}")))?;

        std::fs::write(path, safetensor_bytes)?;
        Ok(())
    }
}

/// Merge LoRA layers into base weights and collect as merged model
///
/// Each entry is (layer_name, LoRALayer). The LoRA layer is cloned and merged,
/// producing the merged base weight.
pub fn merge_and_collect(layers: &[(&str, &LoRALayer)]) -> MergedModel {
    let mut tensors = HashMap::new();
    let mut shapes = HashMap::new();

    for &(name, layer) in layers {
        let mut cloned = layer.clone();
        cloned.merge();
        let data = cloned.base_weight().data().to_vec();
        shapes.insert(name.to_string(), vec![layer.d_out(), layer.d_in()]);
        tensors.insert(name.to_string(), data);
    }

    MergedModel { layers_merged: layers.len(), tensors, shapes }
}

/// Merge QLoRA layers into f32 weights and collect as merged model
///
/// Dequantizes 4-bit base + adapter contribution for each layer.
pub fn merge_qlora_and_collect(layers: &[(&str, &QLoRALayer)]) -> MergedModel {
    let mut tensors = HashMap::new();
    let mut shapes = HashMap::new();

    for &(name, layer) in layers {
        let data = layer.merge_to_f32();
        shapes.insert(name.to_string(), vec![layer.d_out(), layer.d_in()]);
        tensors.insert(name.to_string(), data);
    }

    MergedModel { layers_merged: layers.len(), tensors, shapes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use tempfile::TempDir;

    fn make_lora_layer(d_out: usize, d_in: usize, rank: usize) -> LoRALayer {
        let base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
        LoRALayer::new(base, d_out, d_in, rank, 8.0)
    }

    #[test]
    fn test_merge_and_collect_lora() {
        let layer1 = make_lora_layer(8, 16, 4);
        let layer2 = make_lora_layer(8, 16, 4);

        let layers: Vec<(&str, &LoRALayer)> = vec![
            ("model.layers.0.q_proj.weight", &layer1),
            ("model.layers.0.v_proj.weight", &layer2),
        ];

        let merged = merge_and_collect(&layers);

        assert_eq!(merged.layers_merged, 2);
        assert_eq!(merged.tensors.len(), 2);
        assert!(merged.param_count() > 0);
    }

    #[test]
    fn test_merge_qlora_and_collect() {
        let base = Tensor::from_vec(vec![0.5; 8 * 16], false);
        let qlora = QLoRALayer::new(base, 8, 16, 4, 8.0);

        let layers: Vec<(&str, &QLoRALayer)> = vec![("model.layers.0.q_proj.weight", &qlora)];

        let merged = merge_qlora_and_collect(&layers);

        assert_eq!(merged.layers_merged, 1);
        assert_eq!(merged.tensors.len(), 1);

        let data = merged.tensors.get("model.layers.0.q_proj.weight").expect("key should exist");
        assert_eq!(data.len(), 8 * 16);
    }

    #[test]
    fn test_save_safetensors() {
        let layer = make_lora_layer(8, 8, 4);
        let layers: Vec<(&str, &LoRALayer)> = vec![("weight", &layer)];
        let merged = merge_and_collect(&layers);

        let tmp = TempDir::new().expect("temp file creation should succeed");
        let path = tmp.path().join("merged.safetensors");
        merged.save_safetensors(&path).expect("save should succeed");

        // Verify file exists and is valid safetensors
        let data = std::fs::read(&path).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");
        assert_eq!(loaded.len(), 1);
        let names = loaded.names();
        assert!(names.contains(&"weight"));
    }

    #[test]
    fn test_merge_empty() {
        let layers: Vec<(&str, &LoRALayer)> = vec![];
        let merged = merge_and_collect(&layers);
        assert_eq!(merged.layers_merged, 0);
        assert!(merged.tensors.is_empty());
    }

    #[test]
    fn test_merge_preserves_shapes() {
        let layer = make_lora_layer(8, 16, 4);
        let layers: Vec<(&str, &LoRALayer)> = vec![("w", &layer)];
        let merged = merge_and_collect(&layers);

        assert_eq!(merged.shapes.get("w").expect("key should exist"), &vec![8, 16]);
    }
}
