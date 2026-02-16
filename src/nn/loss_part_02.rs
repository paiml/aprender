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
pub(super) fn softmax_2d(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 2);

    let (batch, features) = (x.shape()[0], x.shape()[1]);
    let mut output = vec![0.0; batch * features];

    for b in 0..batch {
        let row_start = b * features;

        // Find max for numerical stability
        let max_val = x.data()[row_start..row_start + features]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max)
        let mut sum = 0.0;
        for j in 0..features {
            let exp_val = (x.data()[row_start + j] - max_val).exp();
            output[row_start + j] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for j in 0..features {
            output[row_start + j] /= sum;
        }
    }

    Tensor::new(&output, x.shape())
}

/// Log-softmax for numerical stability.
pub(super) fn log_softmax(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 2);

    let (batch, features) = (x.shape()[0], x.shape()[1]);
    let mut output = vec![0.0; batch * features];

    for b in 0..batch {
        let row_start = b * features;

        // Find max for numerical stability
        let max_val = x.data()[row_start..row_start + features]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute log(sum(exp(x - max)))
        let log_sum_exp: f32 = x.data()[row_start..row_start + features]
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f32>()
            .ln();

        // log_softmax = x - max - log_sum_exp
        for j in 0..features {
            output[row_start + j] = x.data()[row_start + j] - max_val - log_sum_exp;
        }
    }

    Tensor::new(&output, x.shape())
}
