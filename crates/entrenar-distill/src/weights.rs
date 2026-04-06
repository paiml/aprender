//! SafeTensors weight loading and conversion utilities.
//!
//! Provides helpers to load SafeTensors files into weight/shape maps
//! and convert them to entrenar's model weight types.

use entrenar_common::{EntrenarError, Result};
use std::collections::HashMap;
use std::path::Path;

/// Load a SafeTensors file into weight and shape maps.
///
/// Returns `(weights, shapes)` where:
/// - `weights`: tensor name → flattened f32 data
/// - `shapes`: tensor name → dimension sizes
pub fn load_safetensors_weights(
    path: impl AsRef<Path>,
) -> Result<(HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>)> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| EntrenarError::Io {
        context: format!("reading SafeTensors file: {}", path.display()),
        source: e,
    })?;

    let tensors =
        safetensors::SafeTensors::deserialize(&data).map_err(|e| EntrenarError::Serialization {
            message: format!("invalid SafeTensors file {}: {e}", path.display()),
        })?;

    let mut weights = HashMap::new();
    let mut shapes = HashMap::new();

    for name in tensors.names() {
        let tensor = tensors.tensor(name).map_err(|e| EntrenarError::Serialization {
            message: format!("failed to read tensor '{name}': {e}"),
        })?;

        let shape: Vec<usize> = tensor.shape().to_vec();
        let float_data: Vec<f32> = match tensor.dtype() {
            safetensors::Dtype::F32 => bytemuck::cast_slice(tensor.data()).to_vec(),
            safetensors::Dtype::F16 => {
                // Convert f16 → f32
                let halfs: &[u16] = bytemuck::cast_slice(tensor.data());
                halfs.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect()
            }
            safetensors::Dtype::BF16 => {
                let bits: &[u16] = bytemuck::cast_slice(tensor.data());
                bits.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect()
            }
            other => {
                return Err(EntrenarError::UnsupportedFormat {
                    format: format!("tensor dtype {other:?}"),
                });
            }
        };

        weights.insert(name.to_string(), float_data);
        shapes.insert(name.to_string(), shape);
    }

    Ok((weights, shapes))
}

/// Convert weight/shape maps to entrenar's `ModelWeights` type (hub feature).
#[cfg(feature = "hub")]
#[allow(clippy::implicit_hasher)]
pub fn weights_to_model_weights(
    weights: HashMap<String, Vec<f32>>,
    shapes: HashMap<String, Vec<usize>>,
) -> entrenar::hf_pipeline::ModelWeights {
    let mut mw = entrenar::hf_pipeline::ModelWeights::new();
    for (name, data) in weights {
        let shape = shapes.get(&name).cloned().unwrap_or_else(|| vec![data.len()]);
        mw.add_tensor(name, data, shape);
    }
    mw
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a test SafeTensors file and return its path.
    fn create_test_safetensors(dir: &Path) -> std::path::PathBuf {
        use safetensors::tensor::{Dtype, TensorView};

        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight_bytes: Vec<u8> = bytemuck::cast_slice(&weight_data).to_vec();
        let bias_data: Vec<f32> = vec![0.1, 0.2, 0.3];
        let bias_bytes: Vec<u8> = bytemuck::cast_slice(&bias_data).to_vec();

        let views = vec![
            (
                "layer.weight",
                TensorView::new(Dtype::F32, vec![2, 3], &weight_bytes)
                    .expect("operation should succeed"),
            ),
            (
                "layer.bias",
                TensorView::new(Dtype::F32, vec![3], &bias_bytes)
                    .expect("operation should succeed"),
            ),
        ];

        let bytes = safetensors::serialize(views, None).expect("operation should succeed");
        let path = dir.join("test.safetensors");
        std::fs::write(&path, bytes).expect("file write should succeed");
        path
    }

    #[test]
    fn test_load_safetensors_weights_basic() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let path = create_test_safetensors(tmp.path());

        let (weights, shapes) = load_safetensors_weights(&path).expect("load should succeed");

        assert_eq!(weights.len(), 2);
        assert_eq!(shapes.len(), 2);

        assert_eq!(weights["layer.weight"], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(shapes["layer.weight"], vec![2, 3]);

        assert_eq!(weights["layer.bias"], vec![0.1, 0.2, 0.3]);
        assert_eq!(shapes["layer.bias"], vec![3]);
    }

    #[test]
    fn test_load_safetensors_weights_missing_file() {
        let result = load_safetensors_weights("/nonexistent/model.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_weights_invalid_data() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let path = tmp.path().join("bad.safetensors");
        std::fs::write(&path, b"not a safetensors file").expect("file write should succeed");

        let result = load_safetensors_weights(&path);
        assert!(result.is_err());
    }
}
