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

/// Decode raw safetensors bytes into f32 by dtype (F32, BF16, F16); any other dtype is an error.
fn decode_peft_tensor(
    name: &str,
    dtype: safetensors::Dtype,
    bytes: &[u8],
) -> Result<Vec<f32>, AdapterError> {
    use safetensors::Dtype;
    let (size, label) = match dtype {
        Dtype::F32 => (4, "F32"),
        Dtype::BF16 | Dtype::F16 => (2, "16-bit float"),
        other => {
            return Err(AdapterError::PeftFormatError(format!(
                "Tensor '{name}' has unsupported dtype {other:?} (expected F32, BF16 or F16)"
            )))
        }
    };
    if bytes.len() % size != 0 {
        return Err(AdapterError::PeftFormatError(format!(
            "Tensor '{name}' byte length {} is not a multiple of {size} ({label})",
            bytes.len()
        )));
    }
    Ok(match dtype {
        Dtype::F32 => {
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => {
            bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
        }
    })
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
        let data = decode_peft_tensor(name, tensor.dtype(), tensor.data())?;
        weights.push((name.to_string(), data));
    }
    // `names()` is HashMap order; sort for a deterministic result.
    weights.sort_by(|a, b| a.0.cmp(&b.0));

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

    /// Save a real PEFT dir (for the config), then replace the safetensors with `tensors`.
    fn peft_dir_with(tensors: Vec<(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)>) -> TempDir {
        let config = LoRAConfig::new(4, 8.0).target_qv_projections();
        let layer = LoRALayer::new(Tensor::zeros(8 * 16, false), 8, 16, 4, 8.0);
        let tmp = TempDir::new().expect("tempdir");
        save_adapter_peft(&[("m.q_proj", &layer)], &config, None, tmp.path()).expect("save");
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(n, d, sh, b)| {
                (
                    (*n).to_string(),
                    safetensors::tensor::TensorView::new(*d, sh.clone(), b).expect("view"),
                )
            })
            .collect();
        let bytes = safetensors::serialize(views, None).expect("serialize");
        std::fs::write(tmp.path().join("adapter_model.safetensors"), bytes).expect("write");
        tmp
    }

    #[test]
    fn test_peft_bf16_and_f16_convert_to_f32() {
        let src = [1.0f32, -2.5, 0.5, 3.0];
        let bf: Vec<u8> = src.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect();
        let f1: Vec<u8> = src.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        let tmp = peft_dir_with(vec![
            ("a.bf16", safetensors::Dtype::BF16, vec![4], bf),
            ("b.f16", safetensors::Dtype::F16, vec![4], f1),
        ]);
        let (_, w) = load_adapter_peft(tmp.path()).expect("load");
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].1, src.to_vec());
        assert_eq!(w[1].1, src.to_vec());
    }

    #[test]
    fn test_peft_unsupported_dtype_is_error() {
        for (dt, n) in [(safetensors::Dtype::I64, 8usize), (safetensors::Dtype::F64, 8)] {
            let tmp = peft_dir_with(vec![("w.bad", dt, vec![1], vec![0u8; n])]);
            let err = load_adapter_peft(tmp.path()).expect_err("must reject");
            assert!(matches!(err, AdapterError::PeftFormatError(_)));
            assert!(err.to_string().contains("w.bad"));
        }
    }

    #[test]
    fn test_peft_weights_sorted_by_name() {
        let b = 1.0f32.to_le_bytes().to_vec();
        let tmp = peft_dir_with(vec![
            ("z", safetensors::Dtype::F32, vec![1], b.clone()),
            ("m", safetensors::Dtype::F32, vec![1], b.clone()),
            ("a", safetensors::Dtype::F32, vec![1], b),
        ]);
        let (_, w) = load_adapter_peft(tmp.path()).expect("load");
        let names: Vec<&str> = w.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "m", "z"]);
    }
}
