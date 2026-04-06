//! Build training components from configuration

use super::schema::{OptimSpec, TrainSpec};
use crate::error::{Error, Result};
use crate::io::{load_model, Model, ModelMetadata};
use crate::optim::{Adam, AdamW, Optimizer, SGD};
use crate::Tensor;

// Optimizer parameter field name constants (CB-525)
const PARAM_MOMENTUM: &str = "momentum";
const PARAM_BETA1: &str = "beta1";
const PARAM_BETA2: &str = "beta2";
const PARAM_EPS: &str = "eps";
const PARAM_WEIGHT_DECAY: &str = "weight_decay";

/// Build optimizer from configuration
pub fn build_optimizer(spec: &OptimSpec) -> Result<Box<dyn Optimizer>> {
    match spec.name.to_lowercase().as_str() {
        "sgd" => {
            let momentum =
                spec.params.get(PARAM_MOMENTUM).and_then(serde_json::Value::as_f64).unwrap_or(0.0)
                    as f32;

            Ok(Box::new(SGD::new(spec.lr, momentum)))
        }
        "adam" => {
            let beta1 =
                spec.params.get(PARAM_BETA1).and_then(serde_json::Value::as_f64).unwrap_or(0.9)
                    as f32;

            let beta2 =
                spec.params.get(PARAM_BETA2).and_then(serde_json::Value::as_f64).unwrap_or(0.999)
                    as f32;

            let eps = spec.params.get(PARAM_EPS).and_then(serde_json::Value::as_f64).unwrap_or(1e-8)
                as f32;

            Ok(Box::new(Adam::new(spec.lr, beta1, beta2, eps)))
        }
        "adamw" => {
            let beta1 =
                spec.params.get(PARAM_BETA1).and_then(serde_json::Value::as_f64).unwrap_or(0.9)
                    as f32;

            let beta2 =
                spec.params.get(PARAM_BETA2).and_then(serde_json::Value::as_f64).unwrap_or(0.999)
                    as f32;

            let eps = spec.params.get(PARAM_EPS).and_then(serde_json::Value::as_f64).unwrap_or(1e-8)
                as f32;

            let weight_decay = spec
                .params
                .get(PARAM_WEIGHT_DECAY)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.01) as f32;

            Ok(Box::new(AdamW::new(spec.lr, beta1, beta2, eps, weight_decay)))
        }
        name => Err(Error::ConfigError(format!(
            "Unknown optimizer: {name}. Supported: sgd, adam, adamw"
        ))),
    }
}

/// Build a model from configuration by loading from file
///
/// Loads the model from the path specified in the TrainSpec. Supports:
/// - SafeTensors (.safetensors) - HuggingFace compatible binary format
/// - JSON (.json) - Entrenar serialization format
/// - YAML (.yaml, .yml) - Entrenar serialization format
///
/// Falls back to demo mode (simple MLP) if the model file doesn't exist,
/// to support testing and demonstration workflows.
pub fn build_model(spec: &TrainSpec) -> Result<Model> {
    let model_path = &spec.model.path;

    // Try to load the actual model if it exists
    if model_path.exists() {
        println!("Loading model from: {}", model_path.display());
        let mut model = load_model(model_path)?;

        // Add training metadata
        model.metadata = model
            .metadata
            .with_custom("config_path", serde_json::json!(model_path))
            .with_custom("optimizer", serde_json::json!(spec.optimizer.name))
            .with_custom("learning_rate", serde_json::json!(spec.optimizer.lr))
            .with_custom("batch_size", serde_json::json!(spec.data.batch_size));

        // Enable gradients on all parameters for training
        for (_, tensor) in &mut model.parameters {
            tensor.set_requires_grad(true);
        }

        println!(
            "Loaded model '{}' with {} parameters",
            model.metadata.name,
            model.parameters.len()
        );

        return Ok(model);
    }

    // Demo mode fallback: create a simple model for testing
    eprintln!(
        "Warning: Model file not found at '{}', using demo mode (simple MLP)",
        model_path.display()
    );

    let params = vec![
        ("layer1.weight".to_string(), Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], true)),
        ("layer1.bias".to_string(), Tensor::from_vec(vec![0.01, 0.02], true)),
        ("layer2.weight".to_string(), Tensor::from_vec(vec![0.5, 0.6], true)),
        ("layer2.bias".to_string(), Tensor::from_vec(vec![0.1], true)),
    ];

    let metadata =
        ModelMetadata::new(format!("demo-model-from-{}", model_path.display()), "simple-mlp")
            .with_custom("demo_mode", serde_json::json!(true))
            .with_custom("config_path", serde_json::json!(model_path))
            .with_custom("optimizer", serde_json::json!(spec.optimizer.name))
            .with_custom("learning_rate", serde_json::json!(spec.optimizer.lr))
            .with_custom("batch_size", serde_json::json!(spec.data.batch_size));

    Ok(Model::new(metadata, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_build_optimizer_adam() {
        let mut params = std::collections::HashMap::new();
        params.insert("beta1".to_string(), serde_json::json!(0.9));
        params.insert("beta2".to_string(), serde_json::json!(0.999));

        let spec = OptimSpec { name: "adam".to_string(), lr: 0.001, params };

        let optimizer = build_optimizer(&spec).expect("operation should succeed");
        assert_eq!(optimizer.lr(), 0.001);
    }

    #[test]
    fn test_build_optimizer_sgd() {
        let mut params = std::collections::HashMap::new();
        params.insert("momentum".to_string(), serde_json::json!(0.9));

        let spec = OptimSpec { name: "sgd".to_string(), lr: 0.01, params };

        let optimizer = build_optimizer(&spec).expect("operation should succeed");
        assert_eq!(optimizer.lr(), 0.01);
    }

    #[test]
    fn test_build_optimizer_adamw() {
        let mut params = std::collections::HashMap::new();
        params.insert("weight_decay".to_string(), serde_json::json!(0.01));

        let spec = OptimSpec { name: "adamw".to_string(), lr: 0.001, params };

        let optimizer = build_optimizer(&spec).expect("operation should succeed");
        assert_eq!(optimizer.lr(), 0.001);
    }

    #[test]
    fn test_build_optimizer_unknown() {
        let spec = OptimSpec {
            name: "unknown".to_string(),
            lr: 0.001,
            params: std::collections::HashMap::new(),
        };

        let result = build_optimizer(&spec);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_model_demo_mode() {
        use super::super::schema::{DataConfig, ModelRef, TrainSpec, TrainingParams};

        // When model file doesn't exist, should fall back to demo mode
        let spec = TrainSpec {
            model: ModelRef { path: PathBuf::from("nonexistent.gguf"), ..Default::default() },
            data: DataConfig {
                train: PathBuf::from("train.parquet"),
                batch_size: 8,
                ..Default::default()
            },
            optimizer: OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: std::collections::HashMap::new(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: TrainingParams::default(),
            publish: None,
        };

        let model = build_model(&spec).expect("operation should succeed");
        assert_eq!(model.parameters.len(), 4);
        assert!(model.get_parameter("layer1.weight").is_some());
        // Verify demo mode indicator
        assert_eq!(model.metadata.architecture, "simple-mlp");
        assert!(model.metadata.name.starts_with("demo-model"));
    }

    #[test]
    fn test_build_model_loads_real_file() {
        use super::super::schema::{DataConfig, ModelRef, TrainSpec, TrainingParams};
        use crate::io::{save_model, ModelFormat, SaveConfig};
        use tempfile::NamedTempFile;

        // Create a real model file
        let params = vec![
            ("embed.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false)),
            ("attn.q".to_string(), Tensor::from_vec(vec![0.1, 0.2], false)),
            ("attn.k".to_string(), Tensor::from_vec(vec![0.3, 0.4], false)),
        ];
        let original = Model::new(ModelMetadata::new("test-transformer", "transformer"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        // Build model from the real file
        let spec = TrainSpec {
            model: ModelRef { path: temp_path.clone(), ..Default::default() },
            data: DataConfig {
                train: PathBuf::from("train.parquet"),
                batch_size: 8,
                ..Default::default()
            },
            optimizer: OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: std::collections::HashMap::new(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: TrainingParams::default(),
            publish: None,
        };

        let loaded = build_model(&spec).expect("load should succeed");

        // Verify it loaded the real model, not demo mode
        assert_eq!(loaded.parameters.len(), 3);
        assert!(loaded.get_parameter("embed.weight").is_some());
        assert!(loaded.get_parameter("attn.q").is_some());
        assert!(loaded.get_parameter("attn.k").is_some());

        // Verify metadata was preserved
        assert_eq!(loaded.metadata.name, "test-transformer");
        assert_eq!(loaded.metadata.architecture, "transformer");

        // Verify gradients are enabled for training
        for (_, tensor) in &loaded.parameters {
            assert!(
                tensor.requires_grad(),
                "All parameters should have requires_grad=true for training"
            );
        }

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_build_model_adds_training_metadata() {
        use super::super::schema::{DataConfig, ModelRef, TrainSpec, TrainingParams};
        use crate::io::{save_model, ModelFormat, SaveConfig};
        use tempfile::NamedTempFile;

        // Create a real model file
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let original = Model::new(ModelMetadata::new("meta-test", "linear"), params);

        let temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_path = temp_file.path().with_extension("json");

        let config = SaveConfig::new(ModelFormat::Json);
        save_model(&original, &temp_path, &config).expect("save should succeed");

        let spec = TrainSpec {
            model: ModelRef { path: temp_path.clone(), ..Default::default() },
            data: DataConfig {
                train: PathBuf::from("train.parquet"),
                batch_size: 32,
                ..Default::default()
            },
            optimizer: OptimSpec {
                name: "adamw".to_string(),
                lr: 0.0001,
                params: std::collections::HashMap::new(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: TrainingParams::default(),
            publish: None,
        };

        let loaded = build_model(&spec).expect("load should succeed");

        // Verify training metadata was added
        assert!(loaded.metadata.custom.contains_key("optimizer"));
        assert!(loaded.metadata.custom.contains_key("learning_rate"));
        assert!(loaded.metadata.custom.contains_key("batch_size"));

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }
}
