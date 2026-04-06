//! Example: Model I/O - Saving and Loading Models
//!
//! Demonstrates how to save and load models using different formats (JSON, YAML).

use entrenar::io::{load_model, save_model, Model, ModelFormat, ModelMetadata, SaveConfig};
use entrenar::Tensor;

fn main() {
    println!("=== Model I/O Example ===\n");

    // Create a simple model
    println!("Creating model...");
    let params = vec![
        ("layer1.weight".to_string(), Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], true)),
        ("layer1.bias".to_string(), Tensor::from_vec(vec![0.01, 0.02], true)),
        ("layer2.weight".to_string(), Tensor::from_vec(vec![0.5, 0.6], true)),
        ("layer2.bias".to_string(), Tensor::from_vec(vec![0.1], false)),
    ];

    let metadata = ModelMetadata::new("example-model", "simple-mlp")
        .with_custom("input_dim", serde_json::json!(4))
        .with_custom("hidden_dim", serde_json::json!(2))
        .with_custom("output_dim", serde_json::json!(1))
        .with_custom("activation", serde_json::json!("relu"));

    let model = Model::new(metadata, params);

    println!("  Model name: {}", model.metadata.name);
    println!("  Architecture: {}", model.metadata.architecture);
    println!("  Parameters: {}", model.parameters.len());
    println!();

    // Save to JSON
    println!("Saving model to JSON...");
    let json_config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
    save_model(&model, "example_model.json", &json_config).expect("Failed to save JSON");
    println!("  ✓ Saved to example_model.json");
    println!();

    // Save to YAML
    println!("Saving model to YAML...");
    let yaml_config = SaveConfig::new(ModelFormat::Yaml);
    save_model(&model, "example_model.yaml", &yaml_config).expect("Failed to save YAML");
    println!("  ✓ Saved to example_model.yaml");
    println!();

    // Load from JSON
    println!("Loading model from JSON...");
    let loaded_json = load_model("example_model.json").expect("Failed to load JSON");
    println!("  ✓ Loaded model: {}", loaded_json.metadata.name);
    println!("  ✓ Parameters: {}", loaded_json.parameters.len());

    // Verify parameters match
    let orig_weight = model.get_parameter("layer1.weight").expect("operation should succeed");
    let loaded_weight =
        loaded_json.get_parameter("layer1.weight").expect("operation should succeed");
    println!("  ✓ Data integrity check: {}", orig_weight.data() == loaded_weight.data());
    println!();

    // Load from YAML
    println!("Loading model from YAML...");
    let loaded_yaml = load_model("example_model.yaml").expect("Failed to load YAML");
    println!("  ✓ Loaded model: {}", loaded_yaml.metadata.name);
    println!("  ✓ Parameters: {}", loaded_yaml.parameters.len());
    println!();

    // Show metadata
    println!("Model Metadata:");
    println!("  Name: {}", loaded_json.metadata.name);
    println!("  Architecture: {}", loaded_json.metadata.architecture);
    println!("  Version: {}", loaded_json.metadata.version);
    println!("\nCustom Metadata:");
    for (key, value) in &loaded_json.metadata.custom {
        println!("  {key}: {value}");
    }
    println!();

    // Show parameter details
    println!("Parameters:");
    for (name, tensor) in &loaded_json.parameters {
        println!("  {} (size={}, requires_grad={})", name, tensor.len(), tensor.requires_grad());
    }
    println!();

    // Cleanup
    println!("Cleaning up...");
    std::fs::remove_file("example_model.json").ok();
    std::fs::remove_file("example_model.yaml").ok();
    println!("  ✓ Removed temporary files");
    println!();

    println!("=== Example Complete ===");
}
