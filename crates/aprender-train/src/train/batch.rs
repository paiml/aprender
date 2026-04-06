//! Batch data structure

use crate::Tensor;

/// A training batch containing inputs and targets
#[derive(Clone, Debug)]
pub struct Batch {
    /// Input features
    pub inputs: Tensor,
    /// Target labels/values
    pub targets: Tensor,
}

impl Batch {
    /// Create a new batch
    pub fn new(inputs: Tensor, targets: Tensor) -> Self {
        Self { inputs, targets }
    }

    /// Get batch size (length of inputs)
    pub fn size(&self) -> usize {
        self.inputs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_creation() {
        let inputs = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let targets = Tensor::from_vec(vec![4.0, 5.0, 6.0], false);

        let batch = Batch::new(inputs, targets);

        assert_eq!(batch.size(), 3);
    }
}
