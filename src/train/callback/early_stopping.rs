//! Early stopping callback to halt training when loss plateaus

use super::traits::{CallbackAction, CallbackContext, TrainerCallback};

/// Early stopping callback to halt training when loss plateaus
///
/// Monitors a metric and stops training if no improvement is seen
/// for `patience` epochs.
///
/// # Example
///
/// ```rust
/// use entrenar::train::callback::EarlyStopping;
///
/// // Stop if no improvement for 5 epochs, min improvement 0.001
/// let early_stop = EarlyStopping::new(5, 0.001);
/// ```
#[derive(Clone, Debug)]
pub struct EarlyStopping {
    /// Number of epochs to wait for improvement
    patience: usize,
    /// Minimum improvement to reset patience
    min_delta: f32,
    /// Best loss seen so far
    best_loss: f32,
    /// Epochs without improvement
    pub(crate) epochs_without_improvement: usize,
    /// Whether to restore best weights (placeholder)
    pub(crate) restore_best: bool,
    /// Monitor validation loss instead of training loss
    monitor_val: bool,
}

impl EarlyStopping {
    /// Create new early stopping callback
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            restore_best: false,
            monitor_val: false,
        }
    }

    /// Configure to restore best weights on stop
    pub fn with_restore_best(mut self) -> Self {
        self.restore_best = true;
        self
    }

    /// Configure to monitor validation loss (requires validation data)
    ///
    /// When enabled, early stopping will only consider validation loss.
    /// If validation loss is not available, training loss is used as fallback.
    pub fn monitor_validation(mut self) -> Self {
        self.monitor_val = true;
        self
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.best_loss = f32::INFINITY;
        self.epochs_without_improvement = 0;
    }

    /// Check if loss improved
    fn check_improvement(&mut self, loss: f32) -> bool {
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.epochs_without_improvement = 0;
            true
        } else {
            self.epochs_without_improvement += 1;
            false
        }
    }
}

impl TrainerCallback for EarlyStopping {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Use val_loss if monitoring validation (with fallback), otherwise use training loss
        let loss = if self.monitor_val { ctx.val_loss.unwrap_or(ctx.loss) } else { ctx.loss };
        self.check_improvement(loss);

        if self.epochs_without_improvement >= self.patience {
            eprintln!(
                "Early stopping: no improvement for {} epochs (best loss: {:.4})",
                self.patience, self.best_loss
            );
            CallbackAction::Stop
        } else {
            CallbackAction::Continue
        }
    }

    fn name(&self) -> &'static str {
        "EarlyStopping"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_patience() {
        let mut es = EarlyStopping::new(3, 0.001);
        let mut ctx = CallbackContext::default();

        // First epoch - establishes baseline
        ctx.loss = 1.0;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);

        // Improvement
        ctx.loss = 0.9;
        ctx.epoch = 1;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);

        // No improvement (within delta)
        ctx.loss = 0.899;
        ctx.epoch = 2;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);

        // Still no improvement
        ctx.loss = 0.899;
        ctx.epoch = 3;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);

        // Still no improvement - should stop (patience=3)
        ctx.loss = 0.899;
        ctx.epoch = 4;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Stop);
    }

    #[test]
    fn test_early_stopping_improvement_resets() {
        let mut es = EarlyStopping::new(2, 0.01);
        let mut ctx = CallbackContext::default();

        ctx.loss = 1.0;
        es.on_epoch_end(&ctx);

        ctx.loss = 1.0;
        ctx.epoch = 1;
        es.on_epoch_end(&ctx);

        // Improvement resets counter
        ctx.loss = 0.5;
        ctx.epoch = 2;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);
        assert_eq!(es.epochs_without_improvement, 0);
    }

    #[test]
    fn test_early_stopping_with_restore_best() {
        let es = EarlyStopping::new(3, 0.001).with_restore_best();
        assert!(es.restore_best);
    }

    #[test]
    fn test_early_stopping_monitor_validation() {
        let mut es = EarlyStopping::new(3, 0.001).monitor_validation();
        assert!(es.monitor_val);

        let mut ctx = CallbackContext::default();
        ctx.loss = 1.0;
        ctx.val_loss = Some(0.5);
        es.on_epoch_end(&ctx);
        assert_eq!(es.best_loss, 0.5);
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut es = EarlyStopping::new(3, 0.001);
        let mut ctx = CallbackContext::default();
        ctx.loss = 0.5;
        es.on_epoch_end(&ctx);
        assert_eq!(es.best_loss, 0.5);

        es.reset();
        assert_eq!(es.best_loss, f32::INFINITY);
        assert_eq!(es.epochs_without_improvement, 0);
    }

    #[test]
    fn test_early_stopping_name() {
        let es = EarlyStopping::new(3, 0.001);
        assert_eq!(es.name(), "EarlyStopping");
    }

    #[test]
    fn test_early_stopping_clone() {
        let es = EarlyStopping::new(5, 0.01);
        let cloned = es.clone();
        assert_eq!(es.patience, cloned.patience);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Early stopping should always stop after patience epochs without improvement
        #[test]
        fn early_stopping_respects_patience(
            patience in 1usize..10,
            min_delta in 0.0001f32..0.1,
            initial_loss in 0.1f32..10.0,
        ) {
            let mut es = EarlyStopping::new(patience, min_delta);
            let mut ctx = CallbackContext::default();

            // First epoch establishes baseline
            ctx.loss = initial_loss;
            es.on_epoch_end(&ctx);

            // Run for patience + 1 epochs without improvement
            for epoch in 1..=patience {
                ctx.epoch = epoch;
                ctx.loss = initial_loss; // No improvement
                let action = es.on_epoch_end(&ctx);

                if epoch < patience {
                    prop_assert_eq!(action, CallbackAction::Continue);
                } else {
                    prop_assert_eq!(action, CallbackAction::Stop);
                }
            }
        }

        /// Early stopping counter should reset on improvement
        #[test]
        fn early_stopping_resets_on_improvement(
            patience in 2usize..10,
            min_delta in 0.001f32..0.1,
            initial_loss in 1.0f32..10.0,
            improvement in 0.2f32..0.5,
        ) {
            let mut es = EarlyStopping::new(patience, min_delta);
            let mut ctx = CallbackContext::default();

            // Establish baseline
            ctx.loss = initial_loss;
            es.on_epoch_end(&ctx);

            // One epoch without improvement
            ctx.epoch = 1;
            es.on_epoch_end(&ctx);
            prop_assert!(es.epochs_without_improvement >= 1);

            // Improvement resets counter
            ctx.epoch = 2;
            ctx.loss = initial_loss - improvement;
            es.on_epoch_end(&ctx);
            prop_assert_eq!(es.epochs_without_improvement, 0);
        }
    }
}
