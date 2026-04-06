//! Model structure for serialization

use crate::Tensor;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

/// Deserialize a bool from either a YAML boolean (`true`) or a quoted string (`"true"`).
/// This supports CB-950 compliance where all truthy values must be quoted in YAML.
fn deserialize_bool_lenient<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolOrString {
        Bool(bool),
        Str(String),
    }

    match BoolOrString::deserialize(deserializer)? {
        BoolOrString::Bool(b) => Ok(b),
        BoolOrString::Str(s) => match s.to_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            other => {
                Err(serde::de::Error::custom(format!("expected 'true' or 'false', got '{other}'")))
            }
        },
    }
}

/// Model metadata containing architecture and training information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name/identifier
    pub name: String,

    /// Model architecture type (e.g., "transformer", "linear", "custom")
    pub architecture: String,

    /// Model version
    pub version: String,

    /// Training configuration used
    pub training_config: Option<HashMap<String, serde_json::Value>>,

    /// Custom metadata fields
    pub custom: HashMap<String, serde_json::Value>,
}

impl ModelMetadata {
    /// Create new metadata with minimal fields
    pub fn new(name: impl Into<String>, architecture: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            architecture: architecture.into(),
            version: "0.1.0".to_string(),
            training_config: None,
            custom: HashMap::new(),
        }
    }

    /// Add custom metadata field
    pub fn with_custom(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.custom.insert(key.into(), value);
        self
    }
}

/// Information about a model parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name (e.g., "layer1.weight", "bias")
    pub name: String,

    /// Parameter shape
    pub shape: Vec<usize>,

    /// Data type (e.g., "f32", "i8")
    pub dtype: String,

    /// Whether this parameter requires gradients
    #[serde(deserialize_with = "deserialize_bool_lenient")]
    pub requires_grad: bool,
}

/// Serializable model state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// Model metadata
    pub metadata: ModelMetadata,

    /// Parameter information
    pub parameters: Vec<ParameterInfo>,

    /// Flattened parameter data
    pub data: Vec<f32>,
}

/// High-level model abstraction for I/O
pub struct Model {
    /// Model metadata
    pub metadata: ModelMetadata,

    /// Model parameters
    pub parameters: Vec<(String, Tensor)>,
}

impl Model {
    /// Create a new model
    pub fn new(metadata: ModelMetadata, parameters: Vec<(String, Tensor)>) -> Self {
        Self { metadata, parameters }
    }

    /// Get parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<&Tensor> {
        self.parameters.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// Get mutable parameter by name
    pub fn get_parameter_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.parameters.iter_mut().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// Convert model to serializable state
    pub fn to_state(&self) -> ModelState {
        let mut data = Vec::new();
        let parameters: Vec<ParameterInfo> = self
            .parameters
            .iter()
            .map(|(name, tensor)| {
                let shape = vec![tensor.len()];
                let param_data = tensor.data();
                data.extend_from_slice(
                    param_data.as_slice().expect("tensor data must be contiguous"),
                );

                ParameterInfo {
                    name: name.clone(),
                    shape,
                    dtype: "f32".to_string(),
                    requires_grad: tensor.requires_grad(),
                }
            })
            .collect();

        ModelState { metadata: self.metadata.clone(), parameters, data }
    }

    /// Create model from serializable state
    pub fn from_state(state: ModelState) -> Self {
        let mut data_offset = 0;
        let parameters: Vec<(String, Tensor)> = state
            .parameters
            .into_iter()
            .map(|param_info| {
                let size: usize = param_info.shape.iter().product();
                let param_data = state.data[data_offset..data_offset + size].to_vec();
                data_offset += size;

                let tensor = Tensor::from_vec(param_data, param_info.requires_grad);
                (param_info.name, tensor)
            })
            .collect();

        Self { metadata: state.metadata, parameters }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_creation() {
        let meta = ModelMetadata::new("test-model", "linear");
        assert_eq!(meta.name, "test-model");
        assert_eq!(meta.architecture, "linear");
        assert_eq!(meta.version, "0.1.0");
    }

    #[test]
    fn test_model_with_custom_metadata() {
        let meta = ModelMetadata::new("test", "custom")
            .with_custom("layers", serde_json::json!(12))
            .with_custom("hidden_size", serde_json::json!(768));

        assert_eq!(meta.custom.len(), 2);
        assert_eq!(meta.custom.get("layers").expect("key should exist"), &serde_json::json!(12));
    }

    #[test]
    fn test_model_parameter_access() {
        let params = vec![
            ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], true)),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let model = Model::new(ModelMetadata::new("test", "linear"), params);

        assert!(model.get_parameter("weight").is_some());
        assert!(model.get_parameter("bias").is_some());
        assert!(model.get_parameter("nonexistent").is_none());
    }

    #[test]
    fn test_model_state_round_trip() {
        let params = vec![
            ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], true)),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let original = Model::new(ModelMetadata::new("test", "linear"), params);
        let state = original.to_state();
        let restored = Model::from_state(state);

        assert_eq!(original.metadata.name, restored.metadata.name);
        assert_eq!(original.parameters.len(), restored.parameters.len());

        // Check parameter data
        let orig_weight = original.get_parameter("weight").expect("parameter should exist");
        let rest_weight = restored.get_parameter("weight").expect("parameter should exist");
        assert_eq!(orig_weight.data(), rest_weight.data());
    }

    #[test]
    fn test_model_get_parameter_mut() {
        let params = vec![("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true))];
        let mut model = Model::new(ModelMetadata::new("test", "linear"), params);

        let tensor = model.get_parameter_mut("weight").expect("parameter should exist");
        assert!(tensor.requires_grad());

        assert!(model.get_parameter_mut("nonexistent").is_none());
    }

    #[test]
    fn test_parameter_info_clone() {
        let info = ParameterInfo {
            name: "layer1.weight".to_string(),
            shape: vec![10, 20],
            dtype: "f32".to_string(),
            requires_grad: true,
        };
        let cloned = info.clone();
        assert_eq!(info.name, cloned.name);
        assert_eq!(info.shape, cloned.shape);
    }

    #[test]
    fn test_model_state_fields() {
        let state = ModelState {
            metadata: ModelMetadata::new("test", "arch"),
            parameters: vec![ParameterInfo {
                name: "w".to_string(),
                shape: vec![5],
                dtype: "f32".to_string(),
                requires_grad: true,
            }],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let cloned = state.clone();
        assert_eq!(state.parameters.len(), cloned.parameters.len());
        assert_eq!(state.data.len(), cloned.data.len());
    }

    #[test]
    fn test_model_metadata_clone() {
        let meta = ModelMetadata::new("model", "transformer");
        let cloned = meta.clone();
        assert_eq!(meta.name, cloned.name);
    }
}
