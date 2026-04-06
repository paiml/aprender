//! Configuration for type inference

/// Configuration for type inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum cardinality ratio to consider categorical (default: 0.05)
    pub categorical_threshold: f32,
    /// Minimum average string length to consider text (default: 20)
    pub text_min_avg_len: f32,
    /// Column names that should be treated as targets
    pub target_columns: Vec<String>,
    /// Column names to exclude from inference
    pub exclude_columns: Vec<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            categorical_threshold: 0.05,
            text_min_avg_len: 20.0,
            target_columns: vec!["label".to_string(), "target".to_string(), "y".to_string()],
            exclude_columns: vec![],
        }
    }
}
