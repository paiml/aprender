/// Transformation test configuration (quantize, import, prune, distill)
///
/// These tests validate model transformation operations — distinct from inference.
/// Each transformation takes a model as input and produces a different model as output.
/// Opt-in via the `transformations:` block in playbook YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationConfig {
    /// Quantization tests: compress model weights to lower precision
    #[serde(default)]
    pub quantize: Option<QuantizeConfig>,
    /// Import tests: convert between model formats
    #[serde(default)]
    pub import: Option<ImportConfig>,
    /// Pruning tests: remove redundant weights
    #[serde(default)]
    pub prune: Option<PruneConfig>,
    /// Distillation tests: train a smaller student model
    #[serde(default)]
    pub distill: Option<DistillConfig>,
}

/// Quantization test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeConfig {
    /// Quantization schemes to test (e.g., `["q4_k_m", "q8_0"]`)
    pub schemes: Vec<String>,
}

/// Format import test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    /// Source formats to import from (e.g., `["gguf", "safetensors"]`)
    pub source_formats: Vec<String>,
}

/// Weight pruning test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruneConfig {
    /// Pruning method: "magnitude", "structured", "wanda"
    pub method: String,
    /// Target sparsity ratio (0.0-1.0)
    pub target_ratio: f64,
}

/// Knowledge distillation test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillConfig {
    /// Student model HF repo or local path
    pub student_model: String,
    /// Calibration dataset path
    pub data_path: String,
}
