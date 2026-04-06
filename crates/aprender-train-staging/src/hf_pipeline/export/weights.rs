//! Model weights and metadata containers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model weights container for export
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Tensor data by name
    pub tensors: HashMap<String, Vec<f32>>,
    /// Tensor shapes by name
    pub shapes: HashMap<String, Vec<usize>>,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Model metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model architecture
    pub architecture: Option<String>,
    /// Model name
    pub model_name: Option<String>,
    /// Number of parameters
    pub num_params: u64,
    /// Hidden size
    pub hidden_size: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Training info
    pub training: Option<TrainingMetadata>,
}

/// Training metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Training epochs completed
    pub epochs: usize,
    /// Final training loss
    pub final_loss: Option<f32>,
    /// Final validation loss
    pub final_val_loss: Option<f32>,
    /// Learning rate used
    pub learning_rate: Option<f64>,
    /// Batch size used
    pub batch_size: Option<usize>,
    /// Distillation temperature (if applicable)
    pub temperature: Option<f32>,
    /// Teacher model (if distilled)
    pub teacher_model: Option<String>,
}

impl ModelWeights {
    /// Create new empty weights container
    #[must_use]
    pub fn new() -> Self {
        Self { tensors: HashMap::new(), shapes: HashMap::new(), metadata: ModelMetadata::default() }
    }

    /// Add a tensor
    pub fn add_tensor(&mut self, name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        let name = name.into();
        self.tensors.insert(name.clone(), data);
        self.shapes.insert(name, shape);
    }

    /// Get tensor by name
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<(&Vec<f32>, &Vec<usize>)> {
        let data = self.tensors.get(name)?;
        let shape = self.shapes.get(name)?;
        Some((data, shape))
    }

    /// Get all tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Count total parameters
    #[must_use]
    pub fn param_count(&self) -> u64 {
        self.tensors.values().map(|t| t.len() as u64).sum()
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Create mock weights for testing
    #[must_use]
    pub fn mock(num_layers: usize, hidden_size: usize) -> Self {
        let mut weights = Self::new();

        for layer in 0..num_layers {
            // Q, K, V, O projections
            for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                let name = format!("layer.{layer}.attention.{proj}.weight");
                let size = hidden_size * hidden_size;
                let data = vec![0.01; size];
                weights.add_tensor(name, data, vec![hidden_size, hidden_size]);
            }

            // MLP layers
            let mlp_size = hidden_size * 4;
            weights.add_tensor(
                format!("layer.{layer}.mlp.up.weight"),
                vec![0.01; hidden_size * mlp_size],
                vec![mlp_size, hidden_size],
            );
            weights.add_tensor(
                format!("layer.{layer}.mlp.down.weight"),
                vec![0.01; mlp_size * hidden_size],
                vec![hidden_size, mlp_size],
            );
        }

        weights.metadata = ModelMetadata {
            num_params: weights.param_count(),
            hidden_size: Some(hidden_size),
            num_layers: Some(num_layers),
            ..Default::default()
        };

        weights
    }
}

impl Default for ModelWeights {
    fn default() -> Self {
        Self::new()
    }
}
