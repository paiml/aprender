//! Knowledge Distillation Loss Functions
//!
//! Implements temperature-scaled KL divergence and progressive distillation
//! based on Hinton et al. (2015) and Sun et al. (2019).
//!
//! # References
//!
//! [1] Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge
//!     in a Neural Network." arXiv:1503.02531
//!
//! [2] Sun, S., Cheng, Y., Gan, Z., & Liu, J. (2019). "Patient Knowledge
//!     Distillation for BERT Model Compression." EMNLP 2019.
//!
//! [3] Zagoruyko, S., & Komodakis, N. (2017). "Paying More Attention to
//!     Attention: Improving the Performance of CNNs via Attention Transfer."
//!     ICLR 2017.

mod attention;
mod loss;
mod progressive;
mod utils;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use attention::AttentionTransfer;
pub use loss::DistillationLoss;
pub use progressive::ProgressiveDistillation;
