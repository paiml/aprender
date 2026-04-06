//! Epoch-level training and validation operations

use super::core::Trainer;
use crate::train::Batch;
use crate::Tensor;

impl Trainer {
    /// Train for one epoch
    ///
    /// # Arguments
    ///
    /// * `batches` - Iterator over training batches
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// Average loss over the epoch
    pub fn train_epoch<F, I>(&mut self, batches: I, forward_fn: F) -> f32
    where
        F: Fn(&Tensor) -> Tensor,
        I: IntoIterator<Item = Batch>,
    {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (i, batch) in batches.into_iter().enumerate() {
            let loss = self.train_step(&batch, &forward_fn);
            total_loss += loss;
            num_batches += 1;

            // Log progress
            if (i + 1) % self.config.log_interval == 0 {
                let avg_loss = total_loss / num_batches as f32;
                println!(
                    "Epoch {}, Step {}: loss={:.4}, lr={:.6}",
                    self.metrics.epoch,
                    i + 1,
                    avg_loss,
                    self.lr()
                );
            }
        }

        let avg_loss = if num_batches > 0 { total_loss / num_batches as f32 } else { 0.0 };

        // Record epoch metrics
        self.metrics.record_epoch(avg_loss, self.lr());

        avg_loss
    }

    /// Validate on a dataset without updating parameters
    ///
    /// # Arguments
    ///
    /// * `batches` - Iterator over validation batches
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// Average validation loss
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use entrenar::train::{Trainer, Batch};
    /// # use entrenar::Tensor;
    /// # let mut trainer: Trainer = todo!();
    /// # let val_batches: Vec<Batch> = vec![];
    /// let val_loss = trainer.validate(val_batches, |x| x.clone());
    /// println!("Validation loss: {:.4}", val_loss);
    /// ```
    pub fn validate<F, I>(&mut self, batches: I, forward_fn: F) -> f32
    where
        F: Fn(&Tensor) -> Tensor,
        I: IntoIterator<Item = Batch>,
    {
        assert!(self.loss_fn.is_some(), "Loss function must be set before validation");

        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch in batches {
            // Forward pass only (no gradients, no optimizer step)
            let predictions = forward_fn(&batch.inputs);
            let loss = self
                .loss_fn
                .as_ref()
                .expect("loss function must be set before validation")
                .forward(&predictions, &batch.targets);
            total_loss += loss.data()[0];
            num_batches += 1;
        }

        let avg_loss = if num_batches > 0 { total_loss / num_batches as f32 } else { 0.0 };

        // Record validation loss
        self.metrics.record_val_loss(avg_loss);

        avg_loss
    }
}

#[cfg(test)]
mod tests {
    use crate::optim::Adam;
    use crate::train::{Batch, MSELoss, TrainConfig, Trainer};
    use crate::Tensor;

    #[test]
    fn test_train_epoch() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100); // Disable logging

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Create multiple batches
        let batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![2.0, 3.0], false),
                Tensor::from_vec(vec![3.0, 4.0], false),
            ),
        ];

        let avg_loss = trainer.train_epoch(batches, std::clone::Clone::clone);

        assert!(avg_loss > 0.0);
        assert_eq!(trainer.metrics.epoch, 1);
        assert_eq!(trainer.metrics.steps, 2);
    }

    #[test]
    fn test_train_epoch_with_empty_batches() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let batches: Vec<Batch> = vec![];
        let avg_loss = trainer.train_epoch(batches, std::clone::Clone::clone);

        // With empty batches, loss is 0.0
        assert_eq!(avg_loss, 0.0);
    }

    #[test]
    fn test_validate() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Validation batches
        let val_batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![2.0, 3.0], false),
                Tensor::from_vec(vec![3.0, 4.0], false),
            ),
        ];

        let val_loss = trainer.validate(val_batches, std::clone::Clone::clone);

        assert!(val_loss > 0.0);
        assert!(val_loss.is_finite());
        assert_eq!(trainer.metrics.val_losses.len(), 1);
        // Steps should not increase during validation
        assert_eq!(trainer.metrics.steps, 0);
    }

    #[test]
    fn test_validate_does_not_update_params() {
        let initial_params = vec![1.0, 2.0];
        let params = vec![Tensor::from_vec(initial_params.clone(), true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let val_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![5.0, 6.0], false), // Different targets to create loss
        )];

        trainer.validate(val_batches, std::clone::Clone::clone);

        // Parameters should remain unchanged after validation
        let params_after: Vec<f32> = trainer.params()[0].data().to_vec();
        assert_eq!(params_after, initial_params);
    }

    #[test]
    fn test_validate_with_empty_batches() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let batches: Vec<Batch> = vec![];
        let val_loss = trainer.validate(batches, std::clone::Clone::clone);

        // With empty batches, loss is 0.0
        assert_eq!(val_loss, 0.0);
    }
}
