//! JSON Schema generation for YAML configuration validation
//!
//! Generates a JSON schema from the TrainSpec struct for external validation
//! tools and IDE autocompletion.
//!
//! Batuta: AI-05 (Declarative Schema Validation)

use serde_json::{json, Value};

/// Generate a JSON schema for the training configuration
#[allow(dead_code)]
///
/// This schema can be used by:
/// - IDE YAML plugins for autocompletion
/// - CI validation of config files
/// - Documentation generation
pub fn training_config_json_schema() -> Value {
    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "entrenar Training Configuration",
        "description": "Schema for entrenar YAML training configuration files",
        "type": "object",
        "required": ["model", "data", "optimizer", "training"],
        "properties": {
            "model": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to model weights or HuggingFace repo ID"
                    },
                    "hidden_size": { "type": "integer", "minimum": 1 },
                    "num_layers": { "type": "integer", "minimum": 1 },
                    "num_heads": { "type": "integer", "minimum": 1 },
                    "num_kv_heads": { "type": "integer", "minimum": 1 },
                    "intermediate_size": { "type": "integer", "minimum": 1 },
                    "vocab_size": { "type": "integer", "minimum": 1 },
                    "max_position_embeddings": { "type": "integer", "minimum": 1 }
                }
            },
            "data": {
                "type": "object",
                "required": ["train", "batch_size"],
                "properties": {
                    "train": { "type": "string", "description": "Training data path" },
                    "val": { "type": "string", "description": "Validation data path" },
                    "batch_size": { "type": "integer", "minimum": 1 },
                    "seq_len": { "type": "integer", "minimum": 1 }
                }
            },
            "optimizer": {
                "type": "object",
                "required": ["name", "lr"],
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["adam", "adamw", "sgd", "rmsprop", "adagrad", "lamb"]
                    },
                    "lr": { "type": "number", "exclusiveMinimum": 0, "maximum": 1 },
                    "beta1": { "type": "number", "minimum": 0, "maximum": 1 },
                    "beta2": { "type": "number", "minimum": 0, "maximum": 1 },
                    "epsilon": { "type": "number", "exclusiveMinimum": 0 },
                    "weight_decay": { "type": "number", "minimum": 0 }
                }
            },
            "training": {
                "type": "object",
                "required": ["epochs"],
                "properties": {
                    "epochs": { "type": "integer", "minimum": 1 },
                    "max_steps": { "type": "integer", "minimum": 1 },
                    "grad_clip": { "type": "number", "exclusiveMinimum": 0 },
                    "gradient_accumulation": { "type": "integer", "minimum": 1 },
                    "save_interval": { "type": "integer", "minimum": 1 },
                    "log_interval": { "type": "integer", "minimum": 1 },
                    "lr_scheduler": {
                        "type": "string",
                        "enum": ["cosine", "linear", "constant", "step", "exponential", "one_cycle", "plateau"]
                    },
                    "warmup_steps": { "type": "integer", "minimum": 0 },
                    "mixed_precision": {
                        "type": "string",
                        "enum": ["bf16", "fp16", "fp32"]
                    },
                    "deterministic": { "type": "boolean" },
                    "seed": { "type": "integer" },
                    "eval_interval": { "type": "integer", "minimum": 0 },
                    "patience": { "type": "integer", "minimum": 0 }
                }
            },
            "lora": {
                "type": "object",
                "properties": {
                    "rank": { "type": "integer", "minimum": 1, "maximum": 1024 },
                    "alpha": { "type": "number", "exclusiveMinimum": 0 },
                    "dropout": { "type": "number", "minimum": 0, "exclusiveMaximum": 1 },
                    "target_modules": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1
                    }
                }
            },
            "distributed": {
                "type": "object",
                "properties": {
                    "world_size": { "type": "integer", "minimum": 1 },
                    "rank": { "type": "integer", "minimum": 0 },
                    "local_rank": { "type": "integer", "minimum": 0 },
                    "role": { "type": "string", "enum": ["coordinator", "worker"] },
                    "backend": { "type": "string", "enum": ["cuda", "wgpu", "auto"] },
                    "coordinator_addr": { "type": "string" }
                }
            }
        }
    })
}

/// Validate a YAML config string against the JSON schema using jsonschema crate
#[allow(dead_code)]
pub fn validate_yaml_against_schema(yaml_str: &str) -> Result<(), Vec<String>> {
    let value: serde_json::Value = match serde_yaml::from_str(yaml_str) {
        Ok(v) => v,
        Err(e) => return Err(vec![format!("YAML parse error: {e}")]),
    };

    let schema = training_config_json_schema();
    let validator = jsonschema::validator_for(&schema)
        .map_err(|e| vec![format!("Schema compilation error: {e}")])?;

    let mut errors: Vec<String> = validator
        .iter_errors(&value)
        .map(|error| {
            let path = error.instance_path.to_string();
            if path.is_empty() {
                error.to_string()
            } else {
                format!("{path}: {error}")
            }
        })
        .collect();

    // Additional semantic checks beyond JSON Schema
    semantic_checks(&value, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Semantic validation checks that go beyond what JSON Schema can express.
#[allow(dead_code)]
fn semantic_checks(value: &serde_json::Value, errors: &mut Vec<String>) {
    let Some(obj) = value.as_object() else {
        return;
    };

    if let Some(lr) =
        obj.get("optimizer").and_then(|o| o.get("lr")).and_then(serde_json::Value::as_f64)
    {
        if lr <= 0.0 || lr > 1.0 {
            errors.push(format!("optimizer.lr must be in (0, 1], got {lr}"));
        }
    }

    if let Some(epochs) =
        obj.get("training").and_then(|t| t.get("epochs")).and_then(serde_json::Value::as_u64)
    {
        if epochs == 0 {
            errors.push("training.epochs must be >= 1".to_string());
        }
    }

    if let Some(bs) =
        obj.get("data").and_then(|d| d.get("batch_size")).and_then(serde_json::Value::as_u64)
    {
        if bs == 0 {
            errors.push("data.batch_size must be >= 1".to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_schema_has_required_fields() {
        let schema = training_config_json_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("model")));
        assert!(required.contains(&json!("data")));
        assert!(required.contains(&json!("optimizer")));
        assert!(required.contains(&json!("training")));
    }

    #[test]
    fn test_json_schema_optimizer_enum() {
        let schema = training_config_json_schema();
        let opt_enum = &schema["properties"]["optimizer"]["properties"]["name"]["enum"];
        assert!(opt_enum.as_array().unwrap().contains(&json!("adamw")));
    }

    #[test]
    fn test_validate_valid_yaml() {
        let yaml = r"
model:
  path: /tmp/model
data:
  train: /tmp/train
  batch_size: 4
optimizer:
  name: adamw
  lr: 0.001
training:
  epochs: 10
";
        assert!(validate_yaml_against_schema(yaml).is_ok());
    }

    #[test]
    fn test_validate_missing_required() {
        let yaml = r"
model:
  path: /tmp/model
";
        let result = validate_yaml_against_schema(yaml);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("data")));
    }

    #[test]
    fn test_validate_invalid_lr() {
        let yaml = r"
model:
  path: /tmp/model
data:
  train: /tmp/train
  batch_size: 4
optimizer:
  name: adamw
  lr: -0.1
training:
  epochs: 10
";
        let result = validate_yaml_against_schema(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_zero_epochs() {
        let yaml = r"
model:
  path: /tmp/model
data:
  train: /tmp/train
  batch_size: 4
optimizer:
  name: adamw
  lr: 0.001
training:
  epochs: 0
";
        let result = validate_yaml_against_schema(yaml);
        assert!(result.is_err());
    }
}
