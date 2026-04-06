//! Progress callback for logging training progress

use super::traits::{CallbackAction, CallbackContext, TrainerCallback};

/// Progress callback for logging training progress
#[derive(Clone, Debug)]
pub struct ProgressCallback {
    /// Log every N steps
    log_interval: usize,
}

impl ProgressCallback {
    /// Create progress callback
    pub fn new(log_interval: usize) -> Self {
        Self { log_interval }
    }
}

impl Default for ProgressCallback {
    fn default() -> Self {
        Self { log_interval: 10 }
    }
}

impl TrainerCallback for ProgressCallback {
    fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        println!("Epoch {}/{} starting (lr: {:.2e})", ctx.epoch + 1, ctx.max_epochs, ctx.lr);
        CallbackAction::Continue
    }

    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        let val_str = ctx.val_loss.map(|v| format!(", val_loss: {v:.4}")).unwrap_or_default();

        println!(
            "Epoch {}/{}: loss: {:.4}{} ({:.1}s)",
            ctx.epoch + 1,
            ctx.max_epochs,
            ctx.loss,
            val_str,
            ctx.elapsed_secs
        );
        CallbackAction::Continue
    }

    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        if ctx.step > 0 && ctx.step.is_multiple_of(self.log_interval) {
            println!("  Step {}/{}: loss: {:.4}", ctx.step, ctx.steps_per_epoch, ctx.loss);
        }
        CallbackAction::Continue
    }

    fn name(&self) -> &'static str {
        "ProgressCallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_callback() {
        let mut progress = ProgressCallback::new(5);
        let ctx = CallbackContext {
            epoch: 0,
            max_epochs: 10,
            step: 5,
            steps_per_epoch: 100,
            loss: 0.5,
            lr: 0.001,
            ..Default::default()
        };

        // Should not panic
        assert_eq!(progress.on_epoch_begin(&ctx), CallbackAction::Continue);
        assert_eq!(progress.on_step_end(&ctx), CallbackAction::Continue);
        assert_eq!(progress.on_epoch_end(&ctx), CallbackAction::Continue);
    }

    #[test]
    fn test_progress_callback_default() {
        let pc = ProgressCallback::default();
        assert_eq!(pc.log_interval, 10);
    }

    #[test]
    fn test_progress_callback_name() {
        let pc = ProgressCallback::new(5);
        assert_eq!(pc.name(), "ProgressCallback");
    }

    #[test]
    fn test_progress_callback_with_val_loss() {
        let mut pc = ProgressCallback::new(5);
        let ctx = CallbackContext {
            epoch: 0,
            max_epochs: 10,
            loss: 0.5,
            val_loss: Some(0.6),
            lr: 0.001,
            elapsed_secs: 1.0,
            ..Default::default()
        };
        assert_eq!(pc.on_epoch_end(&ctx), CallbackAction::Continue);
    }

    #[test]
    fn test_progress_callback_clone() {
        let pc = ProgressCallback::new(5);
        let cloned = pc.clone();
        assert_eq!(pc.log_interval, cloned.log_interval);
    }

    #[test]
    fn test_progress_callback_on_step_end_at_interval() {
        let mut cb = ProgressCallback::new(5);
        let mut ctx = CallbackContext::default();
        ctx.step = 5;
        ctx.steps_per_epoch = 10;

        let action = cb.on_step_end(&ctx);
        assert_eq!(action, CallbackAction::Continue);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Progress callback should always continue
        #[test]
        fn progress_callback_never_stops(
            epoch in 0usize..100,
            step in 0usize..1000,
            loss in -100.0f32..100.0,
        ) {
            let mut progress = ProgressCallback::new(10);
            let ctx = CallbackContext {
                epoch,
                max_epochs: 100,
                step,
                steps_per_epoch: 100,
                loss,
                lr: 0.001,
                ..Default::default()
            };

            prop_assert_eq!(progress.on_train_begin(&ctx), CallbackAction::Continue);
            prop_assert_eq!(progress.on_epoch_begin(&ctx), CallbackAction::Continue);
            prop_assert_eq!(progress.on_step_begin(&ctx), CallbackAction::Continue);
            prop_assert_eq!(progress.on_step_end(&ctx), CallbackAction::Continue);
            prop_assert_eq!(progress.on_epoch_end(&ctx), CallbackAction::Continue);
        }
    }
}
