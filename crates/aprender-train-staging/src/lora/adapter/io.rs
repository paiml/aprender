//! LoRA adapter I/O convenience functions

use super::error::AdapterError;
use super::lora_adapter::LoRAAdapter;
use super::peft_export::PeftAdapterBundle;
use crate::lora::{LoRAConfig, LoRALayer};
use crate::Tensor;
use std::path::Path;

/// Adapter serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterFormat {
    /// Entrenar's native JSON format (single-layer)
    EntrenarJson,
    /// HuggingFace PEFT format (adapter_config.json + adapter_model.safetensors)
    Peft,
}

/// Save LoRA adapter to file (Entrenar JSON format)
///
/// # Arguments
/// * `layer` - LoRALayer to save
/// * `rank` - LoRA rank
/// * `alpha` - LoRA alpha parameter
/// * `path` - File path to save to
pub fn save_adapter<P: AsRef<Path>>(
    layer: &LoRALayer,
    rank: usize,
    alpha: f32,
    path: P,
) -> Result<(), AdapterError> {
    let adapter = LoRAAdapter::from_layer(layer, rank, alpha);
    adapter.save(path)
}

/// Load LoRA adapter from file (Entrenar JSON format)
///
/// # Arguments
/// * `base_weight` - Frozen base weight to apply adapter to
/// * `path` - File path to load from
pub fn load_adapter<P: AsRef<Path>>(
    base_weight: Tensor,
    path: P,
) -> Result<LoRALayer, AdapterError> {
    let adapter = LoRAAdapter::load(path)?;
    adapter.to_layer(base_weight)
}

/// Save LoRA adapters in PEFT-compatible format
///
/// # Arguments
/// * `adapters` - Layer path to LoRA layer mappings
/// * `config` - LoRA configuration
/// * `base_model` - Optional base model name for adapter_config.json
/// * `output_dir` - Output directory (will contain adapter_config.json + adapter_model.safetensors)
pub fn save_adapter_peft<P: AsRef<Path>>(
    adapters: &[(&str, &LoRALayer)],
    config: &LoRAConfig,
    base_model: Option<&str>,
    output_dir: P,
) -> Result<(), AdapterError> {
    let mut bundle = PeftAdapterBundle::new(config.clone());
    if let Some(name) = base_model {
        bundle = bundle.with_base_model(name);
    }
    for (path, layer) in adapters {
        bundle.add_adapter(*path, layer);
    }
    bundle.save_peft(output_dir)
}

/// Load LoRA adapter from PEFT-compatible format
///
/// Reads `adapter_config.json` and `adapter_model.safetensors` from the given directory
/// and returns the adapter configuration along with tensor name → weight data.
pub fn load_adapter_peft<P: AsRef<Path>>(
    dir: P,
) -> Result<(super::peft_config::PeftAdapterConfig, Vec<(String, Vec<f32>)>), AdapterError> {
    let dir = dir.as_ref();

    // Read adapter_config.json
    let config_path = dir.join("adapter_config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config = super::peft_config::PeftAdapterConfig::from_json(&config_str)
        .map_err(|e| AdapterError::PeftFormatError(format!("Invalid adapter_config.json: {e}")))?;

    // Read adapter_model.safetensors
    let model_path = dir.join("adapter_model.safetensors");
    let model_data = std::fs::read(&model_path)?;
    let tensors = safetensors::SafeTensors::deserialize(&model_data).map_err(|e| {
        AdapterError::SafeTensors(format!("Failed to load adapter_model.safetensors: {e}"))
    })?;

    let mut weights = Vec::new();
    for name in tensors.names() {
        let tensor = tensors.tensor(name).map_err(|e| {
            AdapterError::SafeTensors(format!("Failed to read tensor '{name}': {e}"))
        })?;
        let data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(tensor.data()).to_vec();
        weights.push((name.to_string(), data));
    }

    Ok((config, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_adapter_format_eq() {
        assert_eq!(AdapterFormat::EntrenarJson, AdapterFormat::EntrenarJson);
        assert_eq!(AdapterFormat::Peft, AdapterFormat::Peft);
        assert_ne!(AdapterFormat::EntrenarJson, AdapterFormat::Peft);
    }

    #[test]
    fn test_save_load_peft_roundtrip() {
        let config = LoRAConfig::new(4, 8.0).target_qv_projections();

        let base = Tensor::zeros(8 * 16, false);
        let layer = LoRALayer::new(base, 8, 16, 4, 8.0);

        let tmp = TempDir::new().expect("temp file creation should succeed");
        save_adapter_peft(
            &[("model.layers.0.self_attn.q_proj", &layer)],
            &config,
            Some("test/model"),
            tmp.path(),
        )
        .expect("operation should succeed");

        let (loaded_config, weights) = load_adapter_peft(tmp.path()).expect("load should succeed");
        assert_eq!(loaded_config.r, 4);
        assert_eq!(loaded_config.lora_alpha, 8.0);
        assert_eq!(weights.len(), 2); // lora_A + lora_B
    }
}
