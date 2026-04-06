//! Model saving functionality

use super::format::{ModelFormat, SaveConfig};
use super::model::Model;
use crate::Tensor;
use crate::{Error, Result};
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Save a model to a file
///
/// # Arguments
///
/// * `model` - The model to save
/// * `path` - Output file path
/// * `config` - Save configuration (format, options)
///
/// # Example
///
/// ```no_run
/// use entrenar::io::{Model, ModelMetadata, save_model, SaveConfig, ModelFormat};
/// # use entrenar::Tensor;
///
/// let params = vec![
///     ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true)),
/// ];
/// let model = Model::new(ModelMetadata::new("my-model", "linear"), params);
/// let config = SaveConfig::new(ModelFormat::Json);
///
/// save_model(&model, "model.json", &config).expect("failed to save model");
/// ```
pub fn save_model(model: &Model, path: impl AsRef<Path>, config: &SaveConfig) -> Result<()> {
    let path = path.as_ref();

    match config.format {
        ModelFormat::SafeTensors => save_safetensors(model, path),
        ModelFormat::Apr => save_apr(model, path),
        ModelFormat::Json => save_json(model, path, config.pretty),
        ModelFormat::Yaml => save_yaml(model, path),
        #[cfg(feature = "gguf")]
        ModelFormat::Gguf => Err(Error::Serialization(
            "GGUF format not yet implemented. Enable 'gguf' feature and use realizar integration."
                .to_string(),
        )),
    }
}

/// Serialize and save a model as JSON
fn save_json(model: &Model, path: &Path, pretty: bool) -> Result<()> {
    let state = model.to_state();
    let data = if pretty {
        serde_json::to_string_pretty(&state)
            .map_err(|e| Error::Serialization(format!("JSON serialization failed: {e}")))?
    } else {
        serde_json::to_string(&state)
            .map_err(|e| Error::Serialization(format!("JSON serialization failed: {e}")))?
    };
    let mut file = File::create(path)?;
    file.write_all(data.as_bytes())?;
    Ok(())
}

/// Serialize and save a model as YAML
fn save_yaml(model: &Model, path: &Path) -> Result<()> {
    let state = model.to_state();
    let data = serde_yaml::to_string(&state)
        .map_err(|e| Error::Serialization(format!("YAML serialization failed: {e}")))?;
    let mut file = File::create(path)?;
    file.write_all(data.as_bytes())?;
    Ok(())
}

/// ALB-086: Infer tensor shapes using config-aware batch analysis.
/// Scans all parameters to find hidden_size from norm weights, then
/// computes proper 2D shapes for all weight matrices.
fn infer_all_tensor_shapes(parameters: &[(String, Tensor)]) -> HashMap<String, Vec<usize>> {
    let mut shapes = HashMap::new();

    // Find hidden_size from a norm weight (always 1D [H])
    let hidden_size = parameters
        .iter()
        .find(|(n, _)| n.ends_with("layernorm.weight") || n == "model.norm.weight")
        .map_or(0, |(_, t)| t.len());

    for (name, tensor) in parameters {
        let numel = tensor.len();
        let shape = if name.ends_with("layernorm.weight") || name == "model.norm.weight" {
            vec![numel]
        } else if hidden_size > 0 && numel % hidden_size == 0 {
            let other_dim = numel / hidden_size;
            // For down_proj: [hidden_size, intermediate_size] — hidden is smaller dim
            // For gate/up_proj: [intermediate_size, hidden_size] — hidden is smaller dim
            // For q/o_proj: [hidden_size, hidden_size] — square
            // For k/v_proj: [kv_dim, hidden_size] — kv_dim < hidden
            // For embed/lm_head: [vocab_size, hidden_size]
            if name.ends_with("down_proj.weight") {
                vec![hidden_size, other_dim]
            } else {
                vec![other_dim, hidden_size]
            }
        } else {
            vec![numel]
        };
        shapes.insert(name.clone(), shape);
    }
    shapes
}

/// Save model in SafeTensors format (HuggingFace compatible)
fn save_safetensors(model: &Model, path: &Path) -> Result<()> {
    // ALB-086: Compute proper 2D shapes for HuggingFace compatibility
    let shapes = infer_all_tensor_shapes(&model.parameters);

    // Collect tensor data with proper lifetime management
    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = model
        .parameters
        .iter()
        .map(|(name, tensor)| {
            let data = tensor.data();
            let bytes: Vec<u8> =
                bytemuck::cast_slice(data.as_slice().expect("tensor data must be contiguous"))
                    .to_vec();
            let shape = shapes.get(name).cloned().unwrap_or_else(|| vec![tensor.len()]);
            (name.clone(), bytes, shape)
        })
        .collect();

    // Create TensorViews from collected data
    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .expect("TensorView construction must not fail for valid F32 data");
            (name.as_str(), view)
        })
        .collect();

    // Create metadata with model info
    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), model.metadata.name.clone());
    metadata.insert("architecture".to_string(), model.metadata.architecture.clone());
    metadata.insert("version".to_string(), model.metadata.version.clone());

    // Serialize to SafeTensors format
    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| Error::Serialization(format!("SafeTensors serialization failed: {e}")))?;

    // Write to file
    std::fs::write(path, safetensor_bytes)?;

    Ok(())
}

/// ALB-096: Save model in APR format (sovereign stack universal format).
///
/// Uses `AprWriter` for atomic single-file checkpoints with proper 2D shapes
/// and model metadata. Supports both model weights and `__training__.*` tensors.
fn save_apr(model: &Model, path: &Path) -> Result<()> {
    use aprender::serialization::apr::AprWriter;
    use serde_json::Value as JsonValue;

    let mut writer = AprWriter::new();

    // Embed model metadata (keys must match AprWriter's well-known key mapping)
    writer.set_metadata("model_name", JsonValue::String(model.metadata.name.clone()));
    writer.set_metadata("architecture", JsonValue::String(model.metadata.architecture.clone()));
    writer.set_metadata("version", JsonValue::String(model.metadata.version.clone()));
    writer.set_metadata("format", JsonValue::String("entrenar-checkpoint".into()));

    // Compute proper 2D shapes (same logic as SafeTensors — ALB-086)
    let shapes = infer_all_tensor_shapes(&model.parameters);

    for (name, tensor) in &model.parameters {
        let data = tensor.data();
        let slice = data.as_slice().expect("tensor data must be contiguous");
        let shape = shapes.get(name).cloned().unwrap_or_else(|| vec![tensor.len()]);
        writer.add_tensor_f32(name, shape, slice);
    }

    writer
        .write(path)
        .map_err(|e| Error::Serialization(format!("APR serialization failed: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{Model, ModelMetadata};
    use crate::Tensor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_model_json() {
        let params = vec![
            ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], true)),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let model = Model::new(ModelMetadata::new("test-model", "linear"), params);
        let config = SaveConfig::new(ModelFormat::Json);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        // Verify file was created and has content
        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        assert!(!content.is_empty());
        assert!(content.contains("test-model"));
        assert!(content.contains("linear"));
    }

    #[test]
    fn test_save_model_yaml() {
        let params = vec![("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true))];

        let model = Model::new(ModelMetadata::new("test", "simple"), params);
        let config = SaveConfig::new(ModelFormat::Yaml);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        assert!(content.contains("test"));
        assert!(content.contains("simple"));
    }

    #[test]
    fn test_save_model_json_pretty() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("pretty-test", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        // Pretty JSON should have newlines
        assert!(content.contains('\n'));
    }

    #[test]
    fn test_save_model_json_compact() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("compact-test", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(false);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        // Compact JSON should be single line (minus trailing)
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_save_model_empty_params() {
        let model = Model::new(ModelMetadata::new("empty", "test"), vec![]);
        let config = SaveConfig::new(ModelFormat::Json);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        assert!(content.contains("empty"));
    }

    #[test]
    fn test_save_model_large_tensor() {
        let large_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let params = vec![("large".to_string(), Tensor::from_vec(large_data, false))];
        let model = Model::new(ModelMetadata::new("large", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        assert!(content.len() > 1000);
    }

    #[test]
    fn test_save_config_builder() {
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
        assert!(config.pretty);
        assert_eq!(config.format, ModelFormat::Json);
    }

    #[test]
    fn test_save_model_with_compress_option() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("compress-test", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json).with_compress(true);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        // Currently compress is not implemented, but we can still save
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        assert!(content.contains("compress-test"));
    }

    #[test]
    fn test_save_model_multiple_tensors() {
        let params = vec![
            ("layer1.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true)),
            ("layer1.bias".to_string(), Tensor::from_vec(vec![0.1], true)),
            ("layer2.weight".to_string(), Tensor::from_vec(vec![3.0, 4.0], false)),
        ];
        let model = Model::new(ModelMetadata::new("multi", "deep"), params);
        let config = SaveConfig::new(ModelFormat::Yaml);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        assert!(content.contains("layer1.weight"));
        assert!(content.contains("layer2.weight"));
    }

    #[test]
    fn test_save_model_with_metadata() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let meta = ModelMetadata::new("meta-test", "test")
            .with_custom("version", serde_json::json!("1.0.0"))
            .with_custom("author", serde_json::json!("test"));
        let model = Model::new(meta, params);
        let config = SaveConfig::new(ModelFormat::Json);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let content = std::fs::read_to_string(temp_file.path()).expect("file read should succeed");
        assert!(content.contains("version"));
    }

    #[test]
    fn test_save_config_default() {
        let config = SaveConfig::default();
        assert_eq!(config.format, ModelFormat::Json);
        assert!(config.pretty);
        assert!(!config.compress);
    }

    #[test]
    fn test_save_model_invalid_path() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("test", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json);

        // Try to save to an invalid directory
        let result = save_model(&model, "/nonexistent/directory/model.json", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_model_safetensors() {
        let params = vec![
            ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], true)),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let model = Model::new(ModelMetadata::new("safetensor-test", "linear"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        // Verify file was created and is binary (starts with safetensors magic)
        let content = std::fs::read(temp_file.path()).expect("file read should succeed");
        assert!(!content.is_empty());
        // SafeTensors files start with a header length (8 bytes)
        assert!(content.len() > 8);
    }

    #[test]
    fn test_save_model_safetensors_can_be_loaded() {
        let params = vec![
            ("layer1.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true)),
            ("layer1.bias".to_string(), Tensor::from_vec(vec![0.5], false)),
        ];

        let model = Model::new(ModelMetadata::new("roundtrip-test", "mlp"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        // Verify we can load it back with safetensors crate
        let data = std::fs::read(temp_file.path()).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");

        // Check tensor names exist - names() returns Vec<&str>
        let names = loaded.names();
        assert!(names.contains(&"layer1.weight"));
        assert!(names.contains(&"layer1.bias"));

        // Check tensor data
        let weight = loaded.tensor("layer1.weight").expect("load should succeed");
        assert_eq!(weight.shape(), &[4]);
        let weight_data: &[f32] = bytemuck::cast_slice(weight.data());
        assert_eq!(weight_data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_save_safetensors_metadata() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("meta-model", "transformer"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        // Load and check metadata using read_metadata
        let data = std::fs::read(temp_file.path()).expect("file read should succeed");
        let (_, st_metadata) =
            safetensors::SafeTensors::read_metadata(&data).expect("deserialization should succeed");

        let metadata = st_metadata.metadata();
        assert!(metadata.is_some());
        let meta = metadata.as_ref().expect("operation should succeed");
        assert_eq!(meta.get("name").expect("key should exist"), "meta-model");
        assert_eq!(meta.get("architecture").expect("key should exist"), "transformer");
    }

    #[test]
    fn test_save_safetensors_large_tensor() {
        let large_data: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();
        let params =
            vec![("large_weights".to_string(), Tensor::from_vec(large_data.clone(), false))];
        let model = Model::new(ModelMetadata::new("large", "test"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        // Verify data integrity
        let data = std::fs::read(temp_file.path()).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");
        let tensor = loaded.tensor("large_weights").expect("load should succeed");
        let tensor_data: &[f32] = bytemuck::cast_slice(tensor.data());
        assert_eq!(tensor_data.len(), 10000);
        assert!((tensor_data[0] - 0.0).abs() < 1e-6);
        assert!((tensor_data[9999] - 9.999).abs() < 1e-3);
    }

    #[test]
    fn test_save_safetensors_invalid_path() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("test", "test"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let result = save_model(&model, "/nonexistent/directory/model.safetensors", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_safetensors_empty_params() {
        let model = Model::new(ModelMetadata::new("empty", "test"), vec![]);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        // Should still create valid file with metadata
        let data = std::fs::read(temp_file.path()).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_save_safetensors_multiple_tensors() {
        let params = vec![
            ("encoder.layer1.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true)),
            ("encoder.layer1.bias".to_string(), Tensor::from_vec(vec![0.1], true)),
            ("encoder.layer2.weight".to_string(), Tensor::from_vec(vec![3.0, 4.0, 5.0], false)),
            ("decoder.layer1.weight".to_string(), Tensor::from_vec(vec![6.0, 7.0], false)),
        ];
        let model = Model::new(ModelMetadata::new("encoder-decoder", "transformer"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        save_model(&model, temp_file.path(), &config).expect("save should succeed");

        let data = std::fs::read(temp_file.path()).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");
        assert_eq!(loaded.len(), 4);

        // names() returns Vec<&str> directly
        let names = loaded.names();
        assert!(names.contains(&"encoder.layer1.weight"));
        assert!(names.contains(&"decoder.layer1.weight"));
    }

    /// ALB-086: Verify SafeTensors saves proper 2D shapes for LlamaForCausalLM weights.
    #[test]
    fn test_safetensors_saves_2d_shapes() {
        let hidden = 64;
        let intermediate = 128;
        let vocab = 256;

        let params = vec![
            ("model.embed_tokens.weight".to_string(), Tensor::zeros(vocab * hidden, false)),
            ("model.norm.weight".to_string(), Tensor::zeros(hidden, false)),
            ("model.layers.0.input_layernorm.weight".to_string(), Tensor::zeros(hidden, false)),
            (
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                Tensor::zeros(hidden, false),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                Tensor::zeros(hidden * hidden, false),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".to_string(),
                Tensor::zeros(16 * hidden, false),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".to_string(),
                Tensor::zeros(16 * hidden, false),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".to_string(),
                Tensor::zeros(hidden * hidden, false),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                Tensor::zeros(intermediate * hidden, false),
            ),
            (
                "model.layers.0.mlp.up_proj.weight".to_string(),
                Tensor::zeros(intermediate * hidden, false),
            ),
            (
                "model.layers.0.mlp.down_proj.weight".to_string(),
                Tensor::zeros(hidden * intermediate, false),
            ),
        ];

        let metadata = ModelMetadata::new("test", "LlamaForCausalLM");
        let model = Model::new(metadata, params);
        let config =
            crate::io::format::SaveConfig::new(crate::io::format::ModelFormat::SafeTensors);
        let temp = NamedTempFile::new().unwrap();
        save_model(&model, temp.path(), &config).unwrap();

        let data = std::fs::read(temp.path()).unwrap();
        let loaded = safetensors::SafeTensors::deserialize(&data).unwrap();

        // Norm weights should be 1D
        assert_eq!(loaded.tensor("model.norm.weight").unwrap().shape(), &[hidden]);
        assert_eq!(
            loaded.tensor("model.layers.0.input_layernorm.weight").unwrap().shape(),
            &[hidden]
        );

        // Projection weights should be 2D
        assert_eq!(loaded.tensor("model.embed_tokens.weight").unwrap().shape(), &[vocab, hidden]);
        assert_eq!(
            loaded.tensor("model.layers.0.self_attn.q_proj.weight").unwrap().shape(),
            &[hidden, hidden]
        );
        assert_eq!(
            loaded.tensor("model.layers.0.self_attn.k_proj.weight").unwrap().shape(),
            &[16, hidden]
        );
        assert_eq!(
            loaded.tensor("model.layers.0.mlp.gate_proj.weight").unwrap().shape(),
            &[intermediate, hidden]
        );
        assert_eq!(
            loaded.tensor("model.layers.0.mlp.down_proj.weight").unwrap().shape(),
            &[hidden, intermediate]
        );
    }
}
