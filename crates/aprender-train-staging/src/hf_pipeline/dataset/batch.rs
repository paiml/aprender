//! Batch struct for training

use ndarray::Array2;

/// Batch of examples for training
#[derive(Debug, Clone)]
pub struct Batch {
    /// Input IDs [batch_size, max_seq_len]
    pub input_ids: Array2<u32>,
    /// Attention mask [batch_size, max_seq_len]
    pub attention_mask: Array2<u8>,
    /// Labels [batch_size, max_seq_len] (optional)
    pub labels: Option<Array2<u32>>,
    /// Original sequence lengths
    pub lengths: Vec<usize>,
}

impl Batch {
    /// Get batch size
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.input_ids.nrows()
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        contract_pre_seq_len_from_data!();
        self.input_ids.ncols()
    }
}
