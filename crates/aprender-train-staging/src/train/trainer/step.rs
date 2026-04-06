//! Training step operations

use super::core::Trainer;
use crate::optim::clip_grad_norm;
use crate::train::Batch;
use crate::Tensor;
use provable_contracts_macros::ensures;

impl Trainer {
    /// Perform a single training step
    ///
    /// # Arguments
    ///
    /// * `batch` - Training batch with inputs and targets
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// Scalar loss value for this batch
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use entrenar::train::{Trainer, Batch};
    /// # use entrenar::Tensor;
    /// # let mut trainer: Trainer = todo!();
    /// # let batch: Batch = todo!();
    /// let loss = trainer.train_step(&batch, |inputs| {
    ///     // Forward pass: compute predictions
    ///     inputs.clone() // Simplified example
    /// });
    /// ```
    #[ensures(ret.is_finite())]
    pub fn train_step<F>(&mut self, batch: &Batch, forward_fn: F) -> f32
    where
        F: FnOnce(&Tensor) -> Tensor,
    {
        assert!(self.loss_fn.is_some(), "Loss function must be set before training");

        // Zero gradients
        self.optimizer.zero_grad(&mut self.params);

        // Forward pass
        let predictions = forward_fn(&batch.inputs);

        // Compute loss
        let loss = self
            .loss_fn
            .as_ref()
            .expect("loss function must be set before training")
            .forward(&predictions, &batch.targets);

        let loss_val = loss.data()[0];

        // Backward pass
        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        // Gradient clipping
        if let Some(max_norm) = self.config.max_grad_norm {
            clip_grad_norm(&mut self.params, max_norm);
        }

        // Optimizer step
        self.optimizer.step(&mut self.params);

        // Update metrics
        self.metrics.increment_step();

        loss_val
    }

    /// Perform forward and backward pass without optimizer step (for gradient accumulation)
    ///
    /// This is used internally for gradient accumulation. Gradients accumulate
    /// across calls until zero_grad is called.
    pub(crate) fn accumulate_gradients<F>(&mut self, batch: &Batch, forward_fn: F) -> f32
    where
        F: FnOnce(&Tensor) -> Tensor,
    {
        assert!(self.loss_fn.is_some(), "Loss function must be set before training");

        // Forward pass
        let predictions = forward_fn(&batch.inputs);

        // Compute loss
        let loss = self
            .loss_fn
            .as_ref()
            .expect("loss function must be set before training")
            .forward(&predictions, &batch.targets);

        let loss_val = loss.data()[0];

        // Backward pass (gradients accumulate)
        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        loss_val
    }
}

#[cfg(test)]
mod tests {
    use crate::optim::Adam;
    use crate::train::{Batch, MSELoss, TrainConfig, Trainer};
    use crate::Tensor;

    #[test]
    fn test_train_step() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0, 3.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Create a simple batch
        let inputs = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let targets = Tensor::from_vec(vec![2.0, 3.0, 4.0], false);
        let batch = Batch::new(inputs, targets);

        // Train step (identity function)
        let loss = trainer.train_step(&batch, std::clone::Clone::clone);

        // Loss should be positive (predictions != targets)
        assert!(loss > 0.0);
        assert!(loss.is_finite());
        assert_eq!(trainer.metrics.steps, 1);
    }

    #[test]
    #[should_panic(expected = "Loss function must be set")]
    fn test_train_step_without_loss() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);

        let batch = Batch::new(Tensor::zeros(10, false), Tensor::zeros(10, false));

        trainer.train_step(&batch, std::clone::Clone::clone);
    }
}
