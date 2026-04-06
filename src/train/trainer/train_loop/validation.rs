//! Training loop with validation support

use super::basic::safe_avg;
use crate::train::callback::CallbackAction;
use crate::train::trainer::core::Trainer;
use crate::train::trainer::result::TrainResult;
use crate::train::Batch;
use crate::Tensor;
use std::time::Instant;

impl Trainer {
    /// Train for multiple epochs with validation after each epoch
    ///
    /// This method runs training and validation each epoch, passing validation
    /// loss to callbacks for proper early stopping and checkpointing.
    ///
    /// # Arguments
    ///
    /// * `max_epochs` - Maximum number of epochs to train
    /// * `train_fn` - Function that returns training batches for each epoch
    /// * `val_fn` - Function that returns validation batches for each epoch
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// TrainResult with final metrics including best validation loss
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use entrenar::train::{Trainer, Batch, EarlyStopping};
    /// # use entrenar::Tensor;
    /// # let mut trainer: Trainer = todo!();
    /// # let train_batches: Vec<Batch> = vec![];
    /// # let val_batches: Vec<Batch> = vec![];
    /// trainer.add_callback(EarlyStopping::new(5, 0.001).monitor_validation());
    ///
    /// let result = trainer.train_with_val(
    ///     100,
    ///     || train_batches.clone(),
    ///     || val_batches.clone(),
    ///     |x| x.clone()
    /// );
    /// println!("Best val loss: {:.4}", result.best_loss);
    /// ```
    pub fn train_with_val<F, BT, BV, IT, IV>(
        &mut self,
        max_epochs: usize,
        train_fn: BT,
        val_fn: BV,
        forward_fn: F,
    ) -> TrainResult
    where
        F: Fn(&Tensor) -> Tensor,
        BT: Fn() -> IT,
        BV: Fn() -> IV,
        IT: IntoIterator<Item = Batch>,
        IV: IntoIterator<Item = Batch>,
    {
        self.start_time = Some(Instant::now());
        self.best_loss = None;
        let mut stopped_early = false;
        let mut final_loss = 0.0;
        let mut best_val_loss: Option<f32> = None;

        let ctx = self.build_context(0, max_epochs, 0, 0, 0.0, None);
        if self.callbacks.on_train_begin(&ctx) == CallbackAction::Stop {
            return self.make_early_stop_result();
        }

        for epoch in 0..max_epochs {
            let action = self.fire_epoch_begin(epoch, max_epochs, final_loss);
            if action == CallbackAction::Stop {
                stopped_early = true;
                break;
            }
            if action == CallbackAction::SkipEpoch {
                continue;
            }

            // Training phase
            let train_batches: Vec<Batch> = train_fn().into_iter().collect();
            let steps_per_epoch = train_batches.len();

            let step_result = self.run_epoch_steps(
                train_batches,
                steps_per_epoch,
                epoch,
                max_epochs,
                final_loss,
                &forward_fn,
            );
            stopped_early = step_result.stopped_early;
            if stopped_early {
                break;
            }

            let avg_train_loss = safe_avg(step_result.total_loss, step_result.num_batches);
            final_loss = avg_train_loss;

            // Validation phase
            let val_loss = self.compute_validation_loss(&val_fn, &forward_fn);

            // Update best losses
            let monitored_loss = val_loss.unwrap_or(avg_train_loss);
            update_tracked_best(&mut best_val_loss, monitored_loss);
            self.update_best_loss(avg_train_loss);

            self.metrics.record_epoch(avg_train_loss, self.lr());

            if self.fire_epoch_end(epoch, max_epochs, steps_per_epoch, avg_train_loss, val_loss) {
                stopped_early = true;
                break;
            }
        }

        let best = best_val_loss.unwrap_or(self.best_loss.unwrap_or(final_loss));
        self.finalize_training(max_epochs, final_loss, best, stopped_early)
    }

    /// Run validation batches and return average validation loss
    ///
    /// Returns `None` if there are no validation batches or no loss function.
    fn compute_validation_loss<F, BV, IV>(&mut self, val_fn: &BV, forward_fn: &F) -> Option<f32>
    where
        F: Fn(&Tensor) -> Tensor,
        BV: Fn() -> IV,
        IV: IntoIterator<Item = Batch>,
    {
        let val_batches: Vec<Batch> = val_fn().into_iter().collect();
        if val_batches.is_empty() {
            return None;
        }

        let mut val_total = 0.0;
        let mut val_count = 0;
        for batch in val_batches {
            if let Some(loss_fn) = self.loss_fn.as_ref() {
                let predictions = forward_fn(&batch.inputs);
                let loss = loss_fn.forward(&predictions, &batch.targets);
                val_total += loss.data()[0];
                val_count += 1;
            }
        }

        let val_avg = safe_avg(val_total, val_count);
        self.metrics.record_val_loss(val_avg);
        Some(val_avg)
    }
}

/// Update a tracked best value if the new value is lower
fn update_tracked_best(tracked: &mut Option<f32>, value: f32) {
    if tracked.is_none_or(|best| value < best) {
        *tracked = Some(value);
    }
}
