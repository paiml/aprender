#[allow(clippy::wildcard_imports)]
use super::*;

impl NLLLoss {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn forward(&self, log_probs: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(log_probs.ndim(), 2);
        assert_eq!(targets.ndim(), 1);

        let batch_size = log_probs.shape()[0];
        let num_classes = log_probs.shape()[1];

        let mut losses = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let target = targets.data()[b] as usize;
            losses.push(-log_probs.data()[b * num_classes + target]);
        }

        let loss = Tensor::new(&losses, &[batch_size]);

        match self.reduction {
            Reduction::None => loss,
            Reduction::Mean => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Element-wise absolute value.
pub(super) fn abs(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| v.abs()).collect();
    Tensor::new(&data, x.shape())
}

/// Softmax computation for gradient tracking.
///
/// ONE PATH: Delegates to `nn::functional::softmax` (UCBD §4).
pub(super) fn softmax_2d(x: &Tensor) -> Tensor {
    crate::nn::functional::softmax(x, -1)
}

/// Log-softmax for numerical stability.
///
/// ONE PATH: Delegates to `nn::functional::log_softmax` (UCBD §4).
pub(super) fn log_softmax(x: &Tensor) -> Tensor {
    crate::nn::functional::log_softmax(x, -1)
}
