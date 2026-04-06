//! Callback system for training events
//!
//! Provides extensible hooks for training loop events:
//! - `on_train_begin` / `on_train_end`
//! - `on_epoch_begin` / `on_epoch_end`
//! - `on_step_begin` / `on_step_end`
//!
//! # Example
//!
//! ```rust
//! use entrenar::train::callback::{TrainerCallback, CallbackContext, CallbackAction};
//!
//! struct PrintCallback;
//!
//! impl TrainerCallback for PrintCallback {
//!     fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
//!         println!("Epoch {} finished with loss {:.4}", ctx.epoch, ctx.loss);
//!         CallbackAction::Continue
//!     }
//! }
//! ```

#![allow(clippy::field_reassign_with_default)]

mod checkpoint;
mod early_stopping;
mod explainability;
mod manager;
mod monitor;
mod progress;
mod scheduler;
mod traits;

// Re-export all public types
pub use checkpoint::CheckpointCallback;
pub use early_stopping::EarlyStopping;
pub use explainability::{ExplainMethod, ExplainabilityCallback, FeatureImportanceResult};
pub use manager::CallbackManager;
pub use monitor::MonitorCallback;
pub use progress::ProgressCallback;
pub use scheduler::LRSchedulerCallback;
pub use traits::{CallbackAction, CallbackContext, TrainerCallback};
