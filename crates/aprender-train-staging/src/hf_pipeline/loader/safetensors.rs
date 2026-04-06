//! SafeTensors-based teacher model implementation

use crate::hf_pipeline::error::{FetchError, Result};
use ndarray::Array2;
use std::path::Path;

use super::{MemoryEstimate, TeacherModel};

/// SafeTensors-based teacher model
pub struct SafeTensorsTeacher {
    /// Model weights by tensor name
    weights: std::collections::HashMap<String, Array2<f32>>,
    /// Tensor names (in order)
    tensor_names: Vec<String>,
    /// Number of layers
    num_layers: usize,
    /// Hidden dimension
    hidden_size: usize,
    /// Total parameter count
    param_count: u64,
}

impl SafeTensorsTeacher {
    /// Load model from SafeTensors file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to model directory containing model.safetensors
    ///
    /// # Errors
    ///
    /// Returns error if file not found or parsing fails.
    pub fn load(path: &Path) -> Result<Self> {
        use safetensors::SafeTensors;

        let model_path = path.join("model.safetensors");
        if !model_path.exists() {
            return Err(FetchError::FileNotFound {
                repo: path.display().to_string(),
                file: "model.safetensors".into(),
            });
        }

        // Read the file into memory (safe approach for models up to ~10GB)
        let data = std::fs::read(&model_path)?;

        // Parse SafeTensors
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| FetchError::SafeTensorsParseError { message: e.to_string() })?;

        // Extract tensor names and compute statistics
        let tensor_names: Vec<String> = tensors.names().iter().map(|s| (*s).to_string()).collect();

        // Calculate total parameter count
        let mut param_count: u64 = 0;
        for name in &tensor_names {
            if let Ok(info) = tensors.tensor(name) {
                let numel: u64 = info.shape().iter().map(|&x| x as u64).product();
                param_count += numel;
            }
        }

        // Detect number of layers from tensor naming convention
        // Common patterns: "encoder.layer.N.", "layers.N.", "h.N."
        let num_layers = detect_layer_count(&tensor_names);

        // Detect hidden size from weight tensor shapes
        let hidden_size = detect_hidden_size(&tensors, &tensor_names);

        Ok(Self {
            weights: std::collections::HashMap::new(), // Lazy load on demand
            tensor_names,
            num_layers,
            hidden_size,
            param_count,
        })
    }

    /// Get list of tensor names in the model
    #[must_use]
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Get model weights by tensor name
    ///
    /// Note: Currently returns an empty map as weights are loaded on-demand
    /// for memory efficiency. Future versions will support lazy loading.
    #[must_use]
    pub fn weights(&self) -> &std::collections::HashMap<String, Array2<f32>> {
        &self.weights
    }

    /// Create mock teacher for testing
    #[cfg(test)]
    pub fn mock(num_layers: usize, hidden_size: usize) -> Self {
        let param_count = (num_layers as u64) * (hidden_size as u64).pow(2) * 4;
        Self {
            weights: std::collections::HashMap::new(),
            tensor_names: Vec::new(),
            num_layers,
            hidden_size,
            param_count,
        }
    }
}

/// Detect number of layers from tensor naming patterns
fn detect_layer_count(names: &[String]) -> usize {
    use std::collections::HashSet;

    let mut layer_indices: HashSet<usize> = HashSet::new();

    for name in names {
        // Match patterns like "encoder.layer.0.", "layers.0.", "h.0."
        if let Some(idx) = extract_layer_index(name) {
            layer_indices.insert(idx);
        }
    }

    if layer_indices.is_empty() {
        // Default to 12 if can't detect (BERT-base assumption)
        12
    } else {
        layer_indices.len()
    }
}

/// Parse a leading integer from `s`, stopping at the first `.` or end of string.
fn parse_leading_index(s: &str) -> Option<usize> {
    let numeric_part = match s.find('.') {
        Some(end) => &s[..end],
        None => s,
    };
    numeric_part.parse::<usize>().ok()
}

/// Extract layer index from tensor name
fn extract_layer_index(name: &str) -> Option<usize> {
    // Common patterns for layer indices
    const PATTERNS: &[&str] = &[".layer.", ".layers.", ".h."];

    PATTERNS.iter().find_map(|pattern| {
        let pos = name.find(pattern)?;
        let after_pattern = &name[pos + pattern.len()..];
        parse_leading_index(after_pattern)
    })
}

/// Extract the dimension of a square 2D tensor, optionally requiring a minimum size.
fn square_dim(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    min_size: usize,
) -> Option<usize> {
    let shape = tensors.tensor(name).ok()?.shape().to_vec();
    if shape.len() == 2 && shape[0] == shape[1] && shape[0] >= min_size {
        Some(shape[0])
    } else {
        None
    }
}

/// Detect hidden size from attention query weight tensors.
///
/// Looks for tensors matching known query-projection naming patterns
/// (e.g. `.query.weight`, `.q_proj.weight`) that are square matrices.
fn detect_from_query_weights(
    tensors: &safetensors::SafeTensors<'_>,
    names: &[String],
) -> Option<usize> {
    const QUERY_PATTERNS: &[&str] =
        &[".query.weight", ".q_proj.weight", ".self_attn.q_proj.weight"];

    names.iter().find_map(|name| {
        let matches_pattern = QUERY_PATTERNS.iter().any(|p| name.ends_with(p));
        matches_pattern.then(|| square_dim(tensors, name, 1)).flatten()
    })
}

/// Detect hidden size from any large square weight matrix (fallback heuristic).
fn detect_from_square_weights(
    tensors: &safetensors::SafeTensors<'_>,
    names: &[String],
) -> Option<usize> {
    names
        .iter()
        .filter(|name| name.contains("weight"))
        .find_map(|name| square_dim(tensors, name, 256))
}

/// Detect hidden size from tensor shapes
fn detect_hidden_size(tensors: &safetensors::SafeTensors<'_>, names: &[String]) -> usize {
    detect_from_query_weights(tensors, names)
        .or_else(|| detect_from_square_weights(tensors, names))
        // C-15 (Meyer DbC): 0 = unknown, no architecture-specific magic number.
        .unwrap_or(0)
}

impl TeacherModel for SafeTensorsTeacher {
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Mock implementation - just pass through
        Ok(input.clone())
    }

    fn hidden_states(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        // Return one hidden state per layer
        Ok(vec![input.clone(); self.num_layers])
    }

    fn attention_weights(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        // Return attention weights per layer
        let (batch, _seq) = input.dim();
        let attn = Array2::<f32>::ones((batch, batch));
        Ok(vec![attn; self.num_layers])
    }

    fn estimate_memory(&self, batch_size: usize, seq_len: usize) -> MemoryEstimate {
        MemoryEstimate::fp16(self.param_count, batch_size, seq_len, self.hidden_size)
    }

    fn param_count(&self) -> u64 {
        self.param_count
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
