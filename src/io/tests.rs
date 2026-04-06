//! Integration tests for Model I/O

use super::*;
use crate::Tensor;
use tempfile::NamedTempFile;

#[test]
fn test_full_workflow_json() {
    // Create a model
    let params = vec![
        ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], true)),
        ("bias".to_string(), Tensor::from_vec(vec![0.5], false)),
    ];

    let metadata = ModelMetadata::new("integration-test", "linear")
        .with_custom("input_dim", serde_json::json!(3))
        .with_custom("output_dim", serde_json::json!(1));

    let model = Model::new(metadata, params);

    // Save to JSON
    let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    let path = temp_file.path().with_extension("json");

    let config = SaveConfig::new(ModelFormat::Json);
    save_model(&model, &path, &config).expect("save should succeed");

    // Load back
    let loaded = load_model(&path).expect("load should succeed");

    // Verify
    assert_eq!(model.metadata.name, loaded.metadata.name);
    assert_eq!(model.parameters.len(), loaded.parameters.len());

    let orig_weight = model.get_parameter("weight").expect("parameter should exist");
    let loaded_weight = loaded.get_parameter("weight").expect("load should succeed");
    assert_eq!(orig_weight.data(), loaded_weight.data());

    // Clean up
    std::fs::remove_file(path).ok();
}

#[test]
fn test_full_workflow_yaml() {
    let params = vec![
        ("w1".to_string(), Tensor::from_vec(vec![1.0, 2.0], true)),
        ("w2".to_string(), Tensor::from_vec(vec![3.0, 4.0], true)),
    ];

    let model = Model::new(ModelMetadata::new("yaml-test", "dual-layer"), params);

    let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    let path = temp_file.path().with_extension("yaml");

    let config = SaveConfig::new(ModelFormat::Yaml);
    save_model(&model, &path, &config).expect("save should succeed");

    let loaded = load_model(&path).expect("load should succeed");

    assert_eq!(model.metadata.name, loaded.metadata.name);
    assert_eq!(model.parameters.len(), loaded.parameters.len());

    std::fs::remove_file(path).ok();
}

#[test]
fn test_model_with_complex_metadata() {
    let params = vec![("param".to_string(), Tensor::from_vec(vec![1.0], false))];

    let metadata = ModelMetadata::new("complex", "transformer")
        .with_custom("num_layers", serde_json::json!(12))
        .with_custom("hidden_size", serde_json::json!(768))
        .with_custom("vocab_size", serde_json::json!(50000))
        .with_custom("attention_heads", serde_json::json!(12));

    let model = Model::new(metadata, params);

    let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    let path = temp_file.path().with_extension("json");

    let config = SaveConfig::new(ModelFormat::Json);
    save_model(&model, &path, &config).expect("save should succeed");

    let loaded = load_model(&path).expect("load should succeed");

    assert_eq!(model.metadata.custom.get("num_layers"), loaded.metadata.custom.get("num_layers"));
    assert_eq!(model.metadata.custom.get("hidden_size"), loaded.metadata.custom.get("hidden_size"));

    std::fs::remove_file(path).ok();
}

#[test]
fn test_large_model_parameters() {
    // Test with a larger parameter set
    let large_data: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();

    let params = vec![("large_weight".to_string(), Tensor::from_vec(large_data, true))];

    let model = Model::new(ModelMetadata::new("large-test", "big"), params);

    let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    let path = temp_file.path().with_extension("json");

    let config = SaveConfig::new(ModelFormat::Json);
    save_model(&model, &path, &config).expect("save should succeed");

    let loaded = load_model(&path).expect("load should succeed");

    let orig = model.get_parameter("large_weight").expect("parameter should exist");
    let load = loaded.get_parameter("large_weight").expect("load should succeed");

    assert_eq!(orig.len(), load.len());
    assert_eq!(orig.data(), load.data());

    std::fs::remove_file(path).ok();
}
