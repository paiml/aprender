//! Model loading functionality
//!
//! Uses bytemuck for zero_copy tensor deserialization where possible,
//! minimizing allocation_free overhead for large weight tensors.

use super::format::ModelFormat;
use super::model::{Model, ModelMetadata, ModelState};
use crate::{Error, Result, Tensor};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Load a model from a file
///
/// # Arguments
///
/// * `path` - Input file path
///
/// The format is automatically detected from the file extension.
///
/// # Example
///
/// ```no_run
/// use entrenar::io::load_model;
///
/// let model = load_model("model.json").expect("failed to load model");
/// println!("Loaded model: {}", model.metadata.name);
/// ```
pub fn load_model(path: impl AsRef<Path>) -> Result<Model> {
    let path = path.as_ref();

    // Detect format from extension
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::Serialization("File has no extension".to_string()))?;

    let format = ModelFormat::from_extension(ext)
        .ok_or_else(|| Error::Serialization(format!("Unsupported file extension: {ext}")))?;

    // Handle binary formats separately
    if format == ModelFormat::SafeTensors {
        return load_safetensors(path);
    }
    if format == ModelFormat::Apr {
        return load_apr(path);
    }
    #[cfg(feature = "gguf")]
    if format == ModelFormat::Gguf {
        return load_gguf(path);
    }

    // Read file content (text formats)
    let mut file = File::open(path)?;

    let mut content = String::new();
    file.read_to_string(&mut content)?;

    // Deserialize based on format
    let state: ModelState = match format {
        ModelFormat::Json => serde_json::from_str(&content)
            .map_err(|e| Error::Serialization(format!("JSON deserialization failed: {e}")))?,
        ModelFormat::Yaml => serde_yaml::from_str(&content)
            .map_err(|e| Error::Serialization(format!("YAML deserialization failed: {e}")))?,
        ModelFormat::SafeTensors => unreachable!(), // Handled above
        ModelFormat::Apr => unreachable!(),         // Handled above
        #[cfg(feature = "gguf")]
        ModelFormat::Gguf => unreachable!(), // Handled above
    };

    // Convert state to model
    Ok(Model::from_state(state))
}

/// Load model from GGUF format via aprender's GgufReader.
///
/// UCBD §5: All model loading goes through the canonical stack.
/// GGUF → aprender::format::gguf::GgufReader → dequantized f32 tensors → Model.
#[cfg(feature = "gguf")]
fn load_gguf(path: &Path) -> Result<Model> {
    use aprender::format::gguf::GgufReader;

    let reader = GgufReader::from_file(path)
        .map_err(|e| Error::Serialization(format!("GGUF parsing failed: {e}")))?;

    let arch = reader.architecture().unwrap_or_else(|| "unknown".to_string());
    let name = reader.model_name().unwrap_or_else(|| {
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("gguf-model").to_string()
    });

    let metadata = ModelMetadata::new(name, arch);

    // Extract all tensors as dequantized f32 (handles Q4_K, Q6_K, Q8_0, F16, etc.)
    let all_tensors = reader
        .get_all_tensors_f32()
        .map_err(|e| Error::Serialization(format!("GGUF tensor extraction failed: {e}")))?;

    let parameters: Vec<(String, Tensor)> = all_tensors
        .into_iter()
        .map(|(name, (data, _shape))| (name, Tensor::from_vec(data, false)))
        .collect();

    Ok(Model::new(metadata, parameters))
}

/// Load model from SafeTensors format (HuggingFace compatible)
fn load_safetensors(path: &Path) -> Result<Model> {
    // Read binary file
    let data = std::fs::read(path)
        .map_err(|e| Error::Serialization(format!("Failed to read file: {e}")))?;

    // Parse SafeTensors and get metadata
    let (_, st_metadata) = safetensors::SafeTensors::read_metadata(&data)
        .map_err(|e| Error::Serialization(format!("SafeTensors parsing failed: {e}")))?;

    // Extract custom metadata
    let custom_meta = st_metadata.metadata();
    let name = custom_meta
        .as_ref()
        .and_then(|m| m.get("name").cloned())
        .unwrap_or_else(|| "unknown".to_string());
    let architecture = custom_meta
        .as_ref()
        .and_then(|m| m.get("architecture").cloned())
        .unwrap_or_else(|| "unknown".to_string());

    let metadata = ModelMetadata::new(name, architecture);

    // Deserialize to access tensors
    let safetensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| Error::Serialization(format!("SafeTensors parsing failed: {e}")))?;

    // Convert tensors - names() returns Vec<&str>, not an iterator
    let parameters: Vec<(String, Tensor)> = safetensors
        .names()
        .into_iter()
        .map(|name| {
            let tensor_view = safetensors
                .tensor(name)
                .expect("tensor name from names() must exist in SafeTensors");
            let data: &[f32] = bytemuck::cast_slice(tensor_view.data());
            let tensor = Tensor::from_vec(data.to_vec(), false); // Default to no grad
            (name.to_string(), tensor)
        })
        .collect();

    Ok(Model::new(metadata, parameters))
}

/// ALB-096: Load model from APR format (sovereign stack universal format).
///
/// Uses `AprReader` for atomic single-file checkpoints. Skips `__training__.*`
/// tensors (optimizer state) — those are loaded separately by `CudaTransformerTrainer`.
fn load_apr(path: &Path) -> Result<Model> {
    use aprender::serialization::apr::AprReader;

    let reader = AprReader::open(path)
        .map_err(|e| Error::Serialization(format!("APR parsing failed: {e}")))?;

    // Extract metadata (AprReader maps v2_meta.name → "model_name")
    let name =
        reader.get_metadata("model_name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    let architecture = reader
        .get_metadata("architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let metadata = ModelMetadata::new(name, architecture);

    // Load model weight tensors, skip __training__.* (optimizer state)
    let parameters: Vec<(String, Tensor)> = reader
        .tensors
        .iter()
        .filter(|td| !td.name.starts_with("__training__"))
        .map(|td| {
            let data = reader
                .read_tensor_as_f32(&td.name)
                .map_err(|e| Error::Serialization(format!("APR tensor read failed: {e}")))
                .expect("tensor listed in descriptors must be readable");
            (td.name.clone(), Tensor::from_vec(data, false))
        })
        .collect();

    Ok(Model::new(metadata, parameters))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{save_model, Model, ModelMetadata, SaveConfig};
    use crate::Tensor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_model_json() {
        // Create and save a model
        let params = vec![
            ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], true)),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let original = Model::new(ModelMetadata::new("test-model", "linear"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("json");

        let config = SaveConfig::new(ModelFormat::Json);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        // Load it back
        let loaded = load_model(&temp_path).expect("load should succeed");

        // Verify
        assert_eq!(original.metadata.name, loaded.metadata.name);
        assert_eq!(original.metadata.architecture, loaded.metadata.architecture);
        assert_eq!(original.parameters.len(), loaded.parameters.len());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_model_yaml() {
        let params = vec![("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true))];

        let original = Model::new(ModelMetadata::new("yaml-test", "simple"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("yaml");

        let config = SaveConfig::new(ModelFormat::Yaml);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        let loaded = load_model(&temp_path).expect("load should succeed");

        assert_eq!(original.metadata.name, loaded.metadata.name);
        assert_eq!(original.parameters.len(), loaded.parameters.len());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_unsupported_extension() {
        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("unknown");

        let result = load_model(&temp_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_load_round_trip() {
        // Create a model with multiple parameters
        let params = vec![
            ("layer1.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true)),
            ("layer1.bias".to_string(), Tensor::from_vec(vec![0.1, 0.2], true)),
            ("layer2.weight".to_string(), Tensor::from_vec(vec![5.0, 6.0], false)),
        ];

        let meta = ModelMetadata::new("round-trip-test", "multi-layer")
            .with_custom("layers", serde_json::json!(2))
            .with_custom("hidden_size", serde_json::json!(4));

        let original = Model::new(meta, params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("json");

        // Save and load
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
        save_model(&original, &temp_path, &config).expect("save should succeed");
        let loaded = load_model(&temp_path).expect("load should succeed");

        // Verify all parameters match
        assert_eq!(original.parameters.len(), loaded.parameters.len());

        for (orig_name, orig_tensor) in &original.parameters {
            let loaded_tensor = loaded.get_parameter(orig_name).expect("load should succeed");
            assert_eq!(orig_tensor.data(), loaded_tensor.data());
            assert_eq!(orig_tensor.requires_grad(), loaded_tensor.requires_grad());
        }

        // Verify metadata
        assert_eq!(original.metadata.custom.len(), loaded.metadata.custom.len());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_model_file_not_found() {
        let result = load_model("nonexistent_file.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_no_extension() {
        let result = load_model("model_without_extension");
        assert!(result.is_err());
        // Use match instead of unwrap_err since Model doesn't implement Debug
        if let Err(err) = result {
            assert!(err.to_string().contains("no extension"));
        }
    }

    #[test]
    fn test_load_model_invalid_json() {
        use std::io::Write;
        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("json");

        // Write invalid JSON
        let mut f = File::create(&temp_path).expect("file write should succeed");
        f.write_all(b"{ invalid json }").expect("file write should succeed");
        drop(f);

        let result = load_model(&temp_path);
        assert!(result.is_err());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_model_invalid_yaml() {
        use std::io::Write;
        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("yaml");

        // Write invalid YAML
        let mut f = File::create(&temp_path).expect("file write should succeed");
        f.write_all(b"this: is: not: valid: yaml: [}").expect("file write should succeed");
        drop(f);

        let result = load_model(&temp_path);
        assert!(result.is_err());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_yml_extension() {
        let params = vec![("weight".to_string(), Tensor::from_vec(vec![1.0], true))];
        let original = Model::new(ModelMetadata::new("yml-test", "simple"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("yml");

        let config = SaveConfig::new(ModelFormat::Yaml);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        let loaded = load_model(&temp_path).expect("load should succeed");
        assert_eq!(original.metadata.name, loaded.metadata.name);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_model_safetensors() {
        let params = vec![
            ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], true)),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let original = Model::new(ModelMetadata::new("safetensor-test", "linear"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        let loaded = load_model(&temp_path).expect("load should succeed");

        assert_eq!(original.metadata.name, loaded.metadata.name);
        assert_eq!(original.metadata.architecture, loaded.metadata.architecture);
        assert_eq!(original.parameters.len(), loaded.parameters.len());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_safetensors_round_trip_data_integrity() {
        let params = vec![
            ("layer1.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true)),
            ("layer1.bias".to_string(), Tensor::from_vec(vec![0.5, 0.6], false)),
        ];

        let original = Model::new(ModelMetadata::new("round-trip", "mlp"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        let loaded = load_model(&temp_path).expect("load should succeed");

        // Verify data matches
        for (name, orig_tensor) in &original.parameters {
            let loaded_tensor = loaded.get_parameter(name).expect("load should succeed");
            assert_eq!(orig_tensor.data(), loaded_tensor.data());
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_safetensors_file_not_found() {
        let result = load_model("nonexistent.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_invalid_data() {
        use std::io::Write;
        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("safetensors");

        // Write invalid safetensors data
        let mut f = File::create(&temp_path).expect("file write should succeed");
        f.write_all(b"not valid safetensors binary data").expect("file write should succeed");
        drop(f);

        let result = load_model(&temp_path);
        assert!(result.is_err());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_safetensors_large_model() {
        let large_data: Vec<f32> = (0..5000).map(|i| i as f32 * 0.001).collect();
        let params = vec![
            ("large_weight".to_string(), Tensor::from_vec(large_data.clone(), false)),
            ("small_bias".to_string(), Tensor::from_vec(vec![0.1, 0.2], false)),
        ];

        let original = Model::new(ModelMetadata::new("large-model", "test"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        let loaded = load_model(&temp_path).expect("load should succeed");

        let loaded_large = loaded.get_parameter("large_weight").expect("load should succeed");
        assert_eq!(loaded_large.len(), 5000);

        // Verify some values
        let data = loaded_large.data();
        assert!((data[[0]] - 0.0).abs() < 1e-6);
        assert!((data[[4999]] - 4.999).abs() < 1e-3);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_safetensors_metadata_preserved() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let original = Model::new(ModelMetadata::new("meta-model", "transformer"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        let loaded = load_model(&temp_path).expect("load should succeed");

        assert_eq!(loaded.metadata.name, "meta-model");
        assert_eq!(loaded.metadata.architecture, "transformer");

        std::fs::remove_file(temp_path).ok();
    }

    /// Model loading performance: load_bench / loading_time (PW-07)
    #[test]
    fn load_bench_loading_time() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0; 1000], false))];
        let original = Model::new(ModelMetadata::new("bench-model", "test"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        let start = std::time::Instant::now();
        let _loaded = load_model(&temp_path).expect("load should succeed");
        let loading_time = start.elapsed();

        // Model loading should complete in reasonable time
        assert!(loading_time.as_millis() < 5000, "load_bench: {loading_time:?}");

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_apr_round_trip() {
        let params = vec![
            ("layer1.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true)),
            ("layer1.bias".to_string(), Tensor::from_vec(vec![0.5, 0.6], false)),
        ];

        let original = Model::new(ModelMetadata::new("apr-test", "transformer"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("apr");

        let config = SaveConfig::new(ModelFormat::Apr);
        save_model(&original, &temp_path, &config).expect("APR save should succeed");

        let loaded = load_model(&temp_path).expect("APR load should succeed");

        assert_eq!(loaded.metadata.name, "apr-test");
        assert_eq!(loaded.metadata.architecture, "transformer");
        assert_eq!(loaded.parameters.len(), 2);

        for (name, orig_tensor) in &original.parameters {
            let loaded_tensor = loaded.get_parameter(name).expect("tensor should exist");
            assert_eq!(orig_tensor.data(), loaded_tensor.data());
        }

        std::fs::remove_file(temp_path).ok();
    }
}
