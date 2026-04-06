//! Dataset example

/// A single example from a dataset
#[derive(Debug, Clone)]
pub struct Example {
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask (1 = attend, 0 = ignore)
    pub attention_mask: Vec<u8>,
    /// Target labels (for supervised learning)
    pub labels: Option<Vec<u32>>,
    /// Text content (if available)
    pub text: Option<String>,
}

impl Example {
    /// Create new example from token IDs
    #[must_use]
    pub fn from_tokens(input_ids: Vec<u32>) -> Self {
        let len = input_ids.len();
        Self { input_ids, attention_mask: vec![1; len], labels: None, text: None }
    }

    /// Set labels
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<u32>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Set text
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Get sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}
