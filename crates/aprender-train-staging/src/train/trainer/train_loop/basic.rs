//! Basic training loop without validation

use crate::optim::clip_grad_norm;
use crate::train::callback::CallbackAction;
use crate::train::trainer::core::Trainer;
use crate::train::trainer::result::TrainResult;
use crate::train::Batch;
use crate::Tensor;
use std::time::Instant;

/// Result of running the inner step loop for one epoch
pub(super) struct EpochStepResult {
    pub total_loss: f32,
    pub num_batches: usize,
    pub stopped_early: bool,
}

impl Trainer {
    /// Train for multiple epochs with full callback support
    ///
    /// # Arguments
    ///
    /// * `max_epochs` - Maximum number of epochs to train
    /// * `batch_fn` - Function that returns batches for each epoch
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// TrainResult with final metrics
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use entrenar::train::{Trainer, Batch, EarlyStopping};
    /// # use entrenar::Tensor;
    /// # let mut trainer: Trainer = todo!();
    /// # let batches: Vec<Batch> = vec![];
    /// trainer.add_callback(EarlyStopping::new(5, 0.001));
    ///
    /// let result = trainer.train(100, || batches.clone(), |x| x.clone());
    /// println!("Trained {} epochs, final loss: {:.4}", result.final_epoch, result.final_loss);
    /// ```
    pub fn train<F, B, I>(&mut self, max_epochs: usize, batch_fn: B, forward_fn: F) -> TrainResult
    where
        F: Fn(&Tensor) -> Tensor,
        B: Fn() -> I,
        I: IntoIterator<Item = Batch>,
    {
        self.start_time = Some(Instant::now());
        self.best_loss = None;
        let mut stopped_early = false;
        let mut final_loss = 0.0;

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

            let batches: Vec<Batch> = batch_fn().into_iter().collect();
            let steps_per_epoch = batches.len();

            let step_result = self.run_epoch_steps(
                batches,
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

            let avg_loss = safe_avg(step_result.total_loss, step_result.num_batches);
            final_loss = avg_loss;
            self.update_best_loss(avg_loss);
            self.metrics.record_epoch(avg_loss, self.lr());

            if self.fire_epoch_end(epoch, max_epochs, steps_per_epoch, avg_loss, None) {
                stopped_early = true;
                break;
            }
        }

        self.finalize_training(
            max_epochs,
            final_loss,
            self.best_loss.unwrap_or(final_loss),
            stopped_early,
        )
    }

    // -- Shared helpers used by both basic.rs and validation.rs --

    /// Fire the epoch_begin callback and return the requested action
    pub(super) fn fire_epoch_begin(
        &mut self,
        epoch: usize,
        max_epochs: usize,
        current_loss: f32,
    ) -> CallbackAction {
        let ctx = self.build_context(epoch, max_epochs, 0, 0, current_loss, None);
        self.callbacks.on_epoch_begin(&ctx)
    }

    /// Fire the epoch_end callback; returns true if training should stop
    pub(super) fn fire_epoch_end(
        &mut self,
        epoch: usize,
        max_epochs: usize,
        steps_per_epoch: usize,
        loss: f32,
        val_loss: Option<f32>,
    ) -> bool {
        let ctx =
            self.build_context(epoch, max_epochs, steps_per_epoch, steps_per_epoch, loss, val_loss);
        self.callbacks.on_epoch_end(&ctx) == CallbackAction::Stop
    }

    /// Run the inner step loop for one epoch
    pub(super) fn run_epoch_steps<F>(
        &mut self,
        batches: Vec<Batch>,
        steps_per_epoch: usize,
        epoch: usize,
        max_epochs: usize,
        current_loss: f32,
        forward_fn: &F,
    ) -> EpochStepResult
    where
        F: Fn(&Tensor) -> Tensor,
    {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        let accum_steps = self.config.gradient_accumulation_steps.max(1);

        for (step, batch) in batches.into_iter().enumerate() {
            let ctx =
                self.build_context(epoch, max_epochs, step, steps_per_epoch, current_loss, None);
            if self.callbacks.on_step_begin(&ctx) == CallbackAction::Stop {
                return EpochStepResult { total_loss, num_batches, stopped_early: true };
            }

            if step % accum_steps == 0 {
                self.optimizer.zero_grad(&mut self.params);
            }

            let loss = self.accumulate_gradients(&batch, forward_fn);
            total_loss += loss;
            num_batches += 1;

            self.maybe_clip_and_step(step, steps_per_epoch, accum_steps);
            self.metrics.increment_step();

            let ctx = self.build_context(epoch, max_epochs, step, steps_per_epoch, loss, None);
            if self.callbacks.on_step_end(&ctx) == CallbackAction::Stop {
                return EpochStepResult { total_loss, num_batches, stopped_early: true };
            }
        }

        EpochStepResult { total_loss, num_batches, stopped_early: false }
    }

    /// Clip gradients and run optimizer step at accumulation boundaries
    fn maybe_clip_and_step(&mut self, step: usize, steps_per_epoch: usize, accum_steps: usize) {
        let is_accum_boundary = (step + 1).is_multiple_of(accum_steps);
        let is_last_batch = step + 1 == steps_per_epoch;
        if is_accum_boundary || is_last_batch {
            if let Some(max_norm) = self.config.max_grad_norm {
                clip_grad_norm(&mut self.params, max_norm);
            }
            self.optimizer.step(&mut self.params);
        }
    }

    /// Update best_loss if the new loss is lower
    pub(super) fn update_best_loss(&mut self, loss: f32) {
        if self.best_loss.is_none_or(|bl| loss < bl) {
            self.best_loss = Some(loss);
        }
    }

    /// Create a TrainResult for immediate early stop at train_begin
    pub(super) fn make_early_stop_result(&self) -> TrainResult {
        TrainResult {
            final_epoch: 0,
            final_loss: 0.0,
            best_loss: 0.0,
            stopped_early: true,
            elapsed_secs: self.elapsed_secs(),
        }
    }

    /// Fire train_end and build the final TrainResult
    pub(super) fn finalize_training(
        &mut self,
        max_epochs: usize,
        final_loss: f32,
        best_loss: f32,
        stopped_early: bool,
    ) -> TrainResult {
        let ctx = self.build_context(self.metrics.epoch, max_epochs, 0, 0, final_loss, None);
        self.callbacks.on_train_end(&ctx);

        TrainResult {
            final_epoch: self.metrics.epoch,
            final_loss,
            best_loss,
            stopped_early,
            elapsed_secs: self.elapsed_secs(),
        }
    }

    /// Compute elapsed seconds from start_time
    pub(super) fn elapsed_secs(&self) -> f64 {
        self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }
}

/// Safely compute average, returning 0.0 for empty sets
pub(super) fn safe_avg(total: f32, count: usize) -> f32 {
    if count > 0 {
        total / count as f32
    } else {
        0.0
    }
}
