//! Learning rate scheduler callback

use super::traits::{CallbackAction, CallbackContext, TrainerCallback};
use crate::optim::LRScheduler;

/// Callback that applies a learning rate scheduler during training
///
/// Can schedule per-step or per-epoch updates.
///
/// # Example
///
/// ```rust,ignore
/// use entrenar::train::LRSchedulerCallback;
/// use entrenar::optim::CosineAnnealingLR;
///
/// let scheduler = CosineAnnealingLR::new(0.001, 100, 0.0);
/// let callback = LRSchedulerCallback::per_epoch(scheduler);
/// trainer.add_callback(callback);
/// ```
pub struct LRSchedulerCallback<S: LRScheduler + Send> {
    scheduler: S,
    per_step: bool,
    initial_lr: Option<f32>,
}

impl<S: LRScheduler + Send> LRSchedulerCallback<S> {
    /// Create callback that steps scheduler per epoch
    pub fn per_epoch(scheduler: S) -> Self {
        Self { scheduler, per_step: false, initial_lr: None }
    }

    /// Create callback that steps scheduler per step
    pub fn per_step(scheduler: S) -> Self {
        Self { scheduler, per_step: true, initial_lr: None }
    }

    /// Get current learning rate from scheduler
    pub fn current_lr(&self) -> f32 {
        self.scheduler.get_lr()
    }
}

impl<S: LRScheduler + Send> TrainerCallback for LRSchedulerCallback<S> {
    fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.initial_lr = Some(ctx.lr);
        CallbackAction::Continue
    }

    fn on_epoch_end(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        if !self.per_step {
            self.scheduler.step();
        }
        CallbackAction::Continue
    }

    fn on_step_end(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        if self.per_step {
            self.scheduler.step();
        }
        CallbackAction::Continue
    }

    fn name(&self) -> &'static str {
        "LRSchedulerCallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::StepDecayLR;

    #[test]
    fn test_lr_scheduler_callback_per_epoch() {
        let scheduler = StepDecayLR::new(0.1, 10, 0.5);
        let mut cb = LRSchedulerCallback::per_epoch(scheduler);
        let ctx = CallbackContext { lr: 0.1, ..Default::default() };
        cb.on_train_begin(&ctx);
        assert_eq!(cb.initial_lr, Some(0.1));
        cb.on_epoch_end(&ctx);
    }

    #[test]
    fn test_lr_scheduler_callback_per_step() {
        let scheduler = StepDecayLR::new(0.1, 10, 0.5);
        let mut cb = LRSchedulerCallback::per_step(scheduler);
        cb.on_step_end(&CallbackContext::default());
    }

    #[test]
    fn test_lr_scheduler_callback_current_lr() {
        let scheduler = StepDecayLR::new(0.1, 10, 0.5);
        let cb = LRSchedulerCallback::per_epoch(scheduler);
        assert!((cb.current_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_lr_scheduler_callback_name() {
        let scheduler = StepDecayLR::new(0.1, 10, 0.5);
        let cb = LRSchedulerCallback::per_epoch(scheduler);
        assert_eq!(cb.name(), "LRSchedulerCallback");
    }
}
