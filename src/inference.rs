//! Inference — model inference and serving utilities
//!
//! Provides inference endpoints isolated from training loops
//! to prevent hidden feedback loops (MTD-09: Feedback Loop Detection).
//!
//! Training and inference are strictly separated:
//! - Training reads from data, writes model weights
//! - Inference reads model weights, serves predictions
//! - No inference output feeds back into training data

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Model checkpoint path
    pub model_path: String,
    /// Maximum sequence length for inference
    pub max_seq_len: usize,
    /// Whether to use greedy decoding
    pub greedy: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self { model_path: String::new(), max_seq_len: 2048, greedy: true }
    }
}
