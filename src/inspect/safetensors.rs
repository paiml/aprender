//! SafeTensors comparison for golden value testing (GH-121)
//!
//! Enables downloading HuggingFace models in safetensors format and comparing
//! their weights against .apr models bit-for-bit.
//!
//! # Usage
//!
//! ```rust,ignore
//! use aprender::inspect::safetensors::{HfSafetensors, TensorComparison};
//!
//! // Download and load HF model
//! let hf = HfSafetensors::from_hub("openai/whisper-tiny")?;
//!
//! // Compare specific tensor
//! let hf_tensor = hf.tensor("decoder.layers.0.encoder_attn.q_proj.weight")?;
//! let apr_tensor = apr_reader.load_tensor("decoder.layers.0.encoder_attn.q_proj.weight")?;
//!
//! let comparison = TensorComparison::compare(&hf_tensor, &apr_tensor);
//! println!("L2 diff: {}", comparison.l2_distance);
//! ```

#[cfg(feature = "safetensors-compare")]
use safetensors::SafeTensors;

use super::WeightDiff;

/// Error type for safetensors operations
#[derive(Debug)]
pub enum SafetensorsError {
    /// File not found
    FileNotFound(String),
    /// Parse error
    ParseError(String),
    /// Tensor not found
    TensorNotFound(String),
    /// Download error
    DownloadError(String),
    /// IO error
    IoError(std::io::Error),
}

impl std::fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "File not found: {p}"),
            Self::ParseError(e) => write!(f, "Parse error: {e}"),
            Self::TensorNotFound(n) => write!(f, "Tensor not found: {n}"),
            Self::DownloadError(e) => write!(f, "Download error: {e}"),
            Self::IoError(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for SafetensorsError {}

impl From<std::io::Error> for SafetensorsError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Result type for safetensors operations
pub type Result<T> = std::result::Result<T, SafetensorsError>;

/// Tensor data extracted from safetensors
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor name
    pub name: String,
    /// Shape
    pub shape: Vec<usize>,
    /// Data as f32 (converted from source dtype)
    pub data: Vec<f32>,
    /// Original dtype string
    pub dtype: String,
}

impl TensorData {
    /// Get number of elements
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if tensor is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compute L2 norm
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Compute mean
    #[must_use]
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            0.0
        } else {
            self.data.iter().sum::<f32>() / self.data.len() as f32
        }
    }
}

/// Comparison result between two tensors
#[derive(Debug, Clone)]
pub struct TensorComparison {
    /// Tensor name
    pub name: String,
    /// Shape match
    pub shape_match: bool,
    /// Shape A
    pub shape_a: Vec<usize>,
    /// Shape B
    pub shape_b: Vec<usize>,
    /// Weight diff (if shapes match)
    pub weight_diff: Option<WeightDiff>,
    /// Pass threshold test
    pub passes_threshold: bool,
}

impl TensorComparison {
    /// Compare two tensors
    #[must_use]
    pub fn compare(name: &str, a: &TensorData, b: &[f32], threshold: f64) -> Self {
        let shape_match = a.numel() == b.len();

        let weight_diff = if shape_match {
            Some(WeightDiff::from_slices(&a.data, b))
        } else {
            None
        };

        let passes_threshold = weight_diff.as_ref().is_some_and(|d| d.max_diff < threshold);

        Self {
            name: name.to_string(),
            shape_match,
            shape_a: a.shape.clone(),
            shape_b: vec![b.len()],
            weight_diff,
            passes_threshold,
        }
    }

    /// Quick check if tensors are close
    #[must_use]
    pub fn is_close(&self, threshold: f64) -> bool {
        self.shape_match
            && self
                .weight_diff
                .as_ref()
                .is_some_and(|d| d.max_diff < threshold)
    }
}

/// HuggingFace SafeTensors model loader
#[cfg(feature = "safetensors-compare")]
#[derive(Debug)]
pub struct HfSafetensors {
    /// Raw file data (kept for SafeTensors lifetime)
    data: Vec<u8>,
    /// Tensor name to index mapping
    tensor_names: Vec<String>,
}

#[cfg(feature = "safetensors-compare")]
impl HfSafetensors {
    /// Load from a local safetensors file
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_bytes(data)
    }

    /// Load from raw bytes
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        // Parse to get tensor names
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| SafetensorsError::ParseError(e.to_string()))?;

        let tensor_names: Vec<String> = tensors.names().into_iter().map(String::from).collect();

        Ok(Self { data, tensor_names })
    }

    /// Download from HuggingFace Hub
    #[cfg(feature = "hf-hub-integration")]
    pub fn from_hub(repo_id: &str) -> Result<Self> {
        use hf_hub::api::sync::ApiBuilder;

        let api = ApiBuilder::new()
            .build()
            .map_err(|e| SafetensorsError::DownloadError(e.to_string()))?;

        let repo = api.model(repo_id.to_string());

        // Try model.safetensors first, then pytorch_model.safetensors
        let path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.safetensors"))
            .map_err(|e| SafetensorsError::DownloadError(format!("No safetensors file: {e}")))?;

        Self::from_file(&path)
    }

    /// List all tensor names
    #[must_use]
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Get a specific tensor by name
    pub fn tensor(&self, name: &str) -> Result<TensorData> {
        let tensors = SafeTensors::deserialize(&self.data)
            .map_err(|e| SafetensorsError::ParseError(e.to_string()))?;

        let view = tensors
            .tensor(name)
            .map_err(|_| SafetensorsError::TensorNotFound(name.to_string()))?;

        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = format!("{:?}", view.dtype());

        // Convert to f32
        let data = Self::convert_to_f32(view.data(), view.dtype())?;

        Ok(TensorData {
            name: name.to_string(),
            shape,
            data,
            dtype,
        })
    }

    /// Convert raw bytes to f32 based on dtype
    fn convert_to_f32(bytes: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>> {
        use safetensors::Dtype;

        match dtype {
            Dtype::F32 => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(floats)
            }
            Dtype::F16 => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            Dtype::BF16 => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::bf16::from_bits(bits).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            _ => Err(SafetensorsError::ParseError(format!(
                "Unsupported dtype: {dtype:?}"
            ))),
        }
    }

    /// Compare all tensors against an APR model
    pub fn compare_all<F>(&self, get_apr_tensor: F, threshold: f64) -> Vec<TensorComparison>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let mut results = Vec::new();

        for name in &self.tensor_names {
            if let Ok(hf_tensor) = self.tensor(name) {
                if let Some(apr_data) = get_apr_tensor(name) {
                    results.push(TensorComparison::compare(
                        name, &hf_tensor, &apr_data, threshold,
                    ));
                }
            }
        }

        results
    }
}

/// Batch comparison results
#[derive(Debug)]
pub struct BatchComparison {
    /// Individual tensor comparisons
    pub comparisons: Vec<TensorComparison>,
    /// Number of tensors compared
    pub total_compared: usize,
    /// Number that passed threshold
    pub total_passed: usize,
    /// Number with shape mismatch
    pub shape_mismatches: usize,
    /// Worst tensor (highest max_diff)
    pub worst_tensor: Option<String>,
    /// Worst difference
    pub worst_diff: f64,
}

impl BatchComparison {
    /// Create from comparison results
    #[must_use]
    pub fn from_comparisons(comparisons: Vec<TensorComparison>) -> Self {
        let total_compared = comparisons.len();
        let total_passed = comparisons.iter().filter(|c| c.passes_threshold).count();
        let shape_mismatches = comparisons.iter().filter(|c| !c.shape_match).count();

        let (worst_tensor, worst_diff) = comparisons
            .iter()
            .filter_map(|c| c.weight_diff.as_ref().map(|d| (&c.name, d.max_diff)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((None, 0.0), |(name, diff)| (Some(name.clone()), diff));

        Self {
            comparisons,
            total_compared,
            total_passed,
            shape_mismatches,
            worst_tensor,
            worst_diff,
        }
    }

    /// Check if all comparisons passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.total_passed == self.total_compared && self.shape_mismatches == 0
    }

    /// Get summary string
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Compared {} tensors: {} passed, {} shape mismatches, worst diff: {:.6} ({})",
            self.total_compared,
            self.total_passed,
            self.shape_mismatches,
            self.worst_diff,
            self.worst_tensor.as_deref().unwrap_or("none")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data() {
        let data = TensorData {
            name: "test".to_string(),
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dtype: "F32".to_string(),
        };

        assert_eq!(data.numel(), 6);
        assert!(!data.is_empty());
        assert!((data.mean() - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_tensor_comparison() {
        let a = TensorData {
            name: "test".to_string(),
            shape: vec![3],
            data: vec![1.0, 2.0, 3.0],
            dtype: "F32".to_string(),
        };
        let b = vec![1.0, 2.0, 3.0];

        let comp = TensorComparison::compare("test", &a, &b, 1e-5);
        assert!(comp.shape_match);
        assert!(comp.passes_threshold);
        assert!(comp.is_close(1e-5));
    }

    #[test]
    fn test_tensor_comparison_mismatch() {
        let a = TensorData {
            name: "test".to_string(),
            shape: vec![3],
            data: vec![1.0, 2.0, 3.0],
            dtype: "F32".to_string(),
        };
        let b = vec![1.0, 2.0, 4.0]; // Different!

        let comp = TensorComparison::compare("test", &a, &b, 1e-5);
        assert!(comp.shape_match);
        assert!(!comp.passes_threshold); // Fails threshold
    }

    #[test]
    fn test_batch_comparison() {
        let comparisons = vec![
            TensorComparison {
                name: "a".to_string(),
                shape_match: true,
                shape_a: vec![3],
                shape_b: vec![3],
                weight_diff: Some(WeightDiff::empty()),
                passes_threshold: true,
            },
            TensorComparison {
                name: "b".to_string(),
                shape_match: true,
                shape_a: vec![3],
                shape_b: vec![3],
                weight_diff: Some(WeightDiff::empty()),
                passes_threshold: true,
            },
        ];

        let batch = BatchComparison::from_comparisons(comparisons);
        assert_eq!(batch.total_compared, 2);
        assert_eq!(batch.total_passed, 2);
        assert!(batch.all_passed());
    }

    #[test]
    fn test_safetensors_error_display() {
        let err = SafetensorsError::TensorNotFound("foo".to_string());
        assert!(err.to_string().contains("foo"));
    }
}
