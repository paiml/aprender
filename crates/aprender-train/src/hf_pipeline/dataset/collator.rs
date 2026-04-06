//! Distillation collator for batching examples

use ndarray::Array2;

use super::batch::Batch;
use super::dataset_impl::Dataset;
use super::example::Example;

/// Collator for batching examples with dynamic padding
pub struct DistillationCollator {
    /// Padding token ID
    pub pad_token_id: u32,
    /// Maximum sequence length (truncate if longer)
    pub max_length: usize,
    /// Padding side (true = left, false = right)
    pub pad_left: bool,
}

impl Default for DistillationCollator {
    fn default() -> Self {
        Self { pad_token_id: 0, max_length: 512, pad_left: false }
    }
}

impl DistillationCollator {
    /// Create new collator
    #[must_use]
    pub fn new(pad_token_id: u32) -> Self {
        Self { pad_token_id, ..Default::default() }
    }

    /// Set maximum length
    #[must_use]
    pub fn max_length(mut self, len: usize) -> Self {
        self.max_length = len;
        self
    }

    /// Set padding side
    #[must_use]
    pub fn pad_left(mut self, left: bool) -> Self {
        self.pad_left = left;
        self
    }

    /// Collate examples into a batch
    #[must_use]
    pub fn collate(&self, examples: &[Example]) -> Batch {
        if examples.is_empty() {
            return Batch {
                input_ids: Array2::zeros((0, 0)),
                attention_mask: Array2::zeros((0, 0)),
                labels: None,
                lengths: vec![],
            };
        }

        // Find max length in batch (capped at max_length)
        let max_len = examples.iter().map(|e| e.len().min(self.max_length)).max().unwrap_or(0);

        let batch_size = examples.len();
        let mut input_ids = Array2::from_elem((batch_size, max_len), self.pad_token_id);
        let mut attention_mask = Array2::zeros((batch_size, max_len));
        let mut lengths = Vec::with_capacity(batch_size);

        let has_labels = examples.iter().any(|e| e.labels.is_some());
        let mut labels = if has_labels {
            Some(Array2::from_elem((batch_size, max_len), self.pad_token_id))
        } else {
            None
        };

        for (i, example) in examples.iter().enumerate() {
            let seq_len = example.len().min(self.max_length);
            lengths.push(seq_len);

            let (start, end) =
                if self.pad_left { (max_len - seq_len, max_len) } else { (0, seq_len) };

            // Copy input IDs
            for (j, &token) in example.input_ids.iter().take(seq_len).enumerate() {
                input_ids[[i, start + j]] = token;
            }

            // Set attention mask
            for j in start..end {
                attention_mask[[i, j]] = 1;
            }

            // Copy labels if present
            if let (Some(ref mut label_arr), Some(ref ex_labels)) = (&mut labels, &example.labels) {
                for (j, &token) in ex_labels.iter().take(seq_len).enumerate() {
                    label_arr[[i, start + j]] = token;
                }
            }
        }

        Batch { input_ids, attention_mask, labels, lengths }
    }

    /// Create batches from dataset
    pub fn batch_dataset(&self, dataset: &Dataset, batch_size: usize) -> Vec<Batch> {
        dataset.examples().chunks(batch_size).map(|chunk| self.collate(chunk)).collect()
    }
}
