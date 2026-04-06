//! Causal Language Modeling Loss

use crate::Tensor;
use ndarray::Array1;

use super::LossFn;

/// Causal Language Modeling Loss
///
/// Computes cross-entropy loss for next-token prediction tasks.
/// This is the standard loss function for autoregressive language models.
///
/// The loss is computed as:
/// L = -sum(log(softmax(logits)[target_token])) / num_tokens
///
/// # Example
///
/// ```
/// use entrenar::train::{CausalLMLoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = CausalLMLoss::new(10); // vocab_size = 10
/// let logits = Tensor::from_vec(vec![0.1; 3 * 10], true); // seq_len=3, vocab=10
/// let targets = Tensor::from_vec(vec![0.0, 1.0, 2.0], false); // target token IDs
///
/// let loss = loss_fn.forward(&logits, &targets);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct CausalLMLoss {
    /// Vocabulary size
    vocab_size: usize,
}

impl CausalLMLoss {
    /// Create new causal LM loss with given vocabulary size
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Compute softmax for a single position
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|&x| x / sum).collect()
    }
}

impl LossFn for CausalLMLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let seq_len = targets.len();
        let vocab_size = self.vocab_size;

        assert_eq!(
            predictions.len(),
            seq_len * vocab_size,
            "Predictions must be seq_len * vocab_size"
        );

        let pred_data = predictions.data();
        let target_data = targets.data();

        // Compute cross-entropy loss for each position
        let mut total_loss = 0.0;
        let mut grads = vec![0.0; predictions.len()];

        for pos in 0..seq_len {
            let start = pos * vocab_size;
            let end = start + vocab_size;
            let logits =
                &pred_data.as_slice().expect("prediction data must be contiguous")[start..end];

            // Softmax
            let probs = Self::softmax(logits);

            // Get target token ID
            let target_idx = target_data[pos] as usize;
            if target_idx < vocab_size {
                // Cross-entropy: -log(prob of correct token)
                let prob = probs[target_idx].max(1e-10);
                total_loss -= prob.ln();

                // Gradient: probs - one_hot(target)
                for (i, &p) in probs.iter().enumerate() {
                    grads[start + i] = if i == target_idx { p - 1.0 } else { p };
                }
            }
        }

        // Average loss over sequence
        let avg_loss = total_loss / seq_len as f32;
        let mut loss = Tensor::from_vec(vec![avg_loss], true);

        // Scale gradients by 1/seq_len
        let scale = 1.0 / seq_len as f32;
        for g in &mut grads {
            *g *= scale;
        }

        // Setup backward
        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct CausalLMBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            pred_backward_op: Option<Rc<dyn BackwardOp>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for CausalLMBackward {
            fn backward(&self) {
                // Set gradient on predictions
                let mut pred_grad = self.pred_grad_cell.borrow_mut();
                if let Some(existing) = pred_grad.as_mut() {
                    *existing = &*existing + &self.grad;
                } else {
                    *pred_grad = Some(self.grad.clone());
                }
                drop(pred_grad); // Release borrow before recursive call

                // Continue backward propagation through the computational graph
                if let Some(ref op) = self.pred_backward_op {
                    op.backward();
                }
            }
        }

        if predictions.requires_grad() {
            loss.set_backward_op(Rc::new(CausalLMBackward {
                pred_grad_cell: predictions.grad_cell(),
                pred_backward_op: predictions.backward_op(),
                grad: Array1::from(grads),
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "CausalLM"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_lm_loss_basic() {
        let loss_fn = CausalLMLoss::new(10); // vocab_size = 10
                                             // 3 positions, each with 10 logits
        let logits = Tensor::from_vec(vec![0.1; 30], true);
        // Targets: token 0, 1, 2
        let targets = Tensor::from_vec(vec![0.0, 1.0, 2.0], false);

        let loss = loss_fn.forward(&logits, &targets);

        // Loss should be positive and finite
        assert!(loss.data()[0] > 0.0);
        assert!(loss.data()[0].is_finite());
    }

    #[test]
    fn test_causal_lm_loss_perfect_prediction() {
        let loss_fn = CausalLMLoss::new(3); // vocab_size = 3
                                            // Perfect logits: high value at correct index
        let logits = Tensor::from_vec(
            vec![
                10.0, 0.0, 0.0, // position 0: target 0
                0.0, 10.0, 0.0, // position 1: target 1
            ],
            true,
        );
        let targets = Tensor::from_vec(vec![0.0, 1.0], false);

        let loss = loss_fn.forward(&logits, &targets);

        // Loss should be very small (near zero) for correct predictions
        assert!(loss.data()[0] < 0.1);
    }

    #[test]
    fn test_causal_lm_loss_gradient() {
        let loss_fn = CausalLMLoss::new(4); // vocab_size = 4
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let targets = Tensor::from_vec(vec![2.0], false); // target = token 2

        let loss = loss_fn.forward(&logits, &targets);

        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        let grad = logits.grad().expect("gradient should be available");
        // Gradient should be finite
        for g in &grad {
            assert!(g.is_finite());
        }
        // Gradient at correct position should be negative (prob - 1)
        // since target is index 2
        assert!(grad[2] < 0.0);
    }

    #[test]
    fn test_causal_lm_loss_name() {
        let loss_fn = CausalLMLoss::new(10);
        assert_eq!(loss_fn.name(), "CausalLM");
    }

    #[test]
    fn test_causal_lm_loss_longer_sequence() {
        let loss_fn = CausalLMLoss::new(100); // vocab_size = 100
        let seq_len = 10;
        let logits = Tensor::from_vec(vec![0.1; seq_len * 100], true);
        let targets: Vec<f32> = (0..seq_len).map(|i| (i % 100) as f32).collect();
        let targets = Tensor::from_vec(targets, false);

        let loss = loss_fn.forward(&logits, &targets);
        assert!(loss.data()[0] > 0.0);
        assert!(loss.data()[0].is_finite());
    }

    #[test]
    #[should_panic(expected = "seq_len * vocab_size")]
    fn test_causal_lm_loss_mismatched_sizes() {
        let loss_fn = CausalLMLoss::new(10);
        let logits = Tensor::from_vec(vec![0.1; 20], true); // Only 2 positions
        let targets = Tensor::from_vec(vec![0.0, 1.0, 2.0], false); // 3 targets
        loss_fn.forward(&logits, &targets);
    }

    #[test]
    fn test_causal_lm_loss_no_grad() {
        let loss_fn = CausalLMLoss::new(5);
        let logits = Tensor::from_vec(vec![0.1; 10], false); // no grad
        let targets = Tensor::from_vec(vec![0.0, 1.0], false);
        let loss = loss_fn.forward(&logits, &targets);
        assert!(loss.data()[0] > 0.0);
    }
}
