//! Core traits and types for the callback system
//!
//! This module provides the foundational types for training callbacks:
//! - `CallbackContext` - State passed to callbacks
//! - `CallbackAction` - Actions a callback can request
//! - `TrainerCallback` - The trait all callbacks implement

/// Context passed to callbacks with current training state
#[derive(Clone, Debug)]
pub struct CallbackContext {
    /// Current epoch (0-indexed)
    pub epoch: usize,
    /// Total epochs planned
    pub max_epochs: usize,
    /// Current step within epoch
    pub step: usize,
    /// Total steps in epoch
    pub steps_per_epoch: usize,
    /// Global step count
    pub global_step: usize,
    /// Current loss value
    pub loss: f32,
    /// Current learning rate
    pub lr: f32,
    /// Best loss seen so far
    pub best_loss: Option<f32>,
    /// Validation loss (if available)
    pub val_loss: Option<f32>,
    /// Training duration in seconds
    pub elapsed_secs: f64,
}

impl Default for CallbackContext {
    fn default() -> Self {
        Self {
            epoch: 0,
            max_epochs: 0,
            step: 0,
            steps_per_epoch: 0,
            global_step: 0,
            loss: 0.0,
            lr: 0.0,
            best_loss: None,
            val_loss: None,
            elapsed_secs: 0.0,
        }
    }
}

/// Action to take after a callback
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackAction {
    /// Continue training normally
    Continue,
    /// Stop training (early stopping)
    Stop,
    /// Skip rest of current epoch
    SkipEpoch,
}

/// Trait for training callbacks
///
/// Implement this trait to hook into training events. All methods have
/// default no-op implementations, so you only need to implement the
/// events you care about.
pub trait TrainerCallback: Send {
    /// Called before training starts
    fn on_train_begin(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after training ends
    fn on_train_end(&mut self, _ctx: &CallbackContext) {}

    /// Called before each epoch
    fn on_epoch_begin(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after each epoch
    fn on_epoch_end(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called before each training step
    fn on_step_begin(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after each training step
    fn on_step_end(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called when validation is performed
    fn on_validation(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Get callback name for logging
    fn name(&self) -> &'static str {
        "TrainerCallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_context_default() {
        let ctx = CallbackContext::default();
        assert_eq!(ctx.epoch, 0);
        assert_eq!(ctx.loss, 0.0);
        assert!(ctx.best_loss.is_none());
    }

    #[test]
    fn test_callback_action_clone_copy() {
        let action = CallbackAction::Continue;
        let cloned = action;
        assert_eq!(action, cloned);
        assert_ne!(CallbackAction::Stop, CallbackAction::SkipEpoch);
    }

    #[test]
    fn test_callback_context_clone() {
        let ctx = CallbackContext {
            epoch: 5,
            max_epochs: 10,
            step: 50,
            steps_per_epoch: 100,
            global_step: 550,
            loss: 0.5,
            lr: 0.001,
            best_loss: Some(0.4),
            val_loss: Some(0.6),
            elapsed_secs: 100.0,
        };
        let cloned = ctx.clone();
        assert_eq!(ctx.epoch, cloned.epoch);
    }

    #[test]
    fn test_default_trainer_callback_impl() {
        struct MinimalCallback;
        impl TrainerCallback for MinimalCallback {
            fn name(&self) -> &'static str {
                "MinimalCallback"
            }
        }

        let mut cb = MinimalCallback;
        let ctx = CallbackContext::default();
        assert_eq!(cb.on_train_begin(&ctx), CallbackAction::Continue);
        assert_eq!(cb.on_epoch_begin(&ctx), CallbackAction::Continue);
        assert_eq!(cb.on_epoch_end(&ctx), CallbackAction::Continue);
        assert_eq!(cb.on_step_begin(&ctx), CallbackAction::Continue);
        assert_eq!(cb.on_step_end(&ctx), CallbackAction::Continue);
        assert_eq!(cb.on_validation(&ctx), CallbackAction::Continue);
        cb.on_train_end(&ctx);
    }
}
