//! Teacher model trait for distillation

use crate::hf_pipeline::error::Result;
use ndarray::Array2;

use super::MemoryEstimate;

/// Teacher model trait for distillation
///
/// Provides interface for frozen teacher models used in knowledge distillation.
pub trait TeacherModel: Send + Sync {
    /// Run forward pass, returning output logits
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch_size, seq_len, hidden_size]
    ///
    /// # Returns
    ///
    /// Output logits [batch_size, seq_len, vocab_size]
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>>;

    /// Get intermediate hidden states for progressive distillation
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    ///
    /// # Returns
    ///
    /// Hidden states for each layer
    fn hidden_states(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>>;

    /// Get attention weights for attention transfer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    ///
    /// # Returns
    ///
    /// Attention weights [batch, heads, seq, seq] for each layer
    fn attention_weights(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>>;

    /// Estimate memory requirements
    fn estimate_memory(&self, batch_size: usize, seq_len: usize) -> MemoryEstimate;

    /// Get number of parameters
    fn param_count(&self) -> u64;

    /// Get number of layers
    fn num_layers(&self) -> usize;

    /// Get hidden size
    fn hidden_size(&self) -> usize;
}
