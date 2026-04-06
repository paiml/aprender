//! Trainer abstraction for training loops
//!
//! This module provides a high-level `Trainer` that orchestrates the training loop,
//! including:
//! - Single training steps
//! - Epoch-level training
//! - Multi-epoch training with callbacks
//! - Validation
//! - Gradient accumulation
//!
//! # Example
//!
//! ```no_run
//! use entrenar::train::{Trainer, TrainConfig, Batch, MSELoss, EarlyStopping};
//! use entrenar::optim::Adam;
//! use entrenar::Tensor;
//!
//! // Setup
//! let params = vec![Tensor::zeros(10, true)];
//! let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
//! let config = TrainConfig::default();
//!
//! let mut trainer = Trainer::new(params, Box::new(optimizer), config);
//! trainer.set_loss(Box::new(MSELoss));
//! trainer.add_callback(EarlyStopping::new(5, 0.001));
//!
//! // Training with callbacks
//! // let result = trainer.train(10, || batches.clone(), |x| x.clone());
//! ```

#![allow(clippy::field_reassign_with_default)]

mod core;
mod epoch;
mod result;
mod step;
mod train_loop;

pub use core::Trainer;
pub use result::TrainResult;
