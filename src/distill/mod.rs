//! Knowledge Distillation
//!
//! This module implements various knowledge distillation techniques for training
//! smaller student models from larger teacher models.
//!
//! ## Features
//!
//! - **Temperature-scaled KL divergence**: Standard distillation loss with soft targets
//! - **Multi-teacher ensemble**: Distill from multiple teachers simultaneously
//! - **Progressive distillation**: Layer-wise distillation for intermediate representations
//! - **Checkpoint saving**: Student model checkpoints with distillation metadata
//!
//! ## Example
//!
//! ```
//! use entrenar::distill::DistillationLoss;
//! use ndarray::array;
//!
//! let loss_fn = DistillationLoss::new(3.0, 0.5);
//! let student_logits = array![[1.0, 2.0, 1.5]];
//! let teacher_logits = array![[1.2, 1.8, 1.6]];
//! let labels = vec![1];
//! let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
//! assert!(loss > 0.0);
//! ```

pub mod checkpoint;
mod ensemble;
mod loss;
mod progressive;

#[cfg(test)]
mod tests;

pub use checkpoint::{save_student_checkpoint, DistillationCheckpoint};
pub use ensemble::EnsembleDistiller;
pub use loss::DistillationLoss;
pub use progressive::ProgressiveDistiller;
