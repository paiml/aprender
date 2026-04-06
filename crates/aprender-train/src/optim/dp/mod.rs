//! Differential Privacy Module (MLOPS-015)
//!
//! DP-SGD implementation following Abadi et al. (2016) for privacy-preserving training.
//!
//! # Toyota Way: Jidoka (Autonomation)
//!
//! Built-in privacy protection stops data leakage. The privacy budget acts as
//! an Andon system, halting training when privacy guarantees would be violated.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::optim::dp::{DpSgdConfig, DpSgd, PrivacyBudget};
//! use entrenar::optim::SGD;
//!
//! let config = DpSgdConfig::new()
//!     .with_max_grad_norm(1.0)
//!     .with_noise_multiplier(1.1)
//!     .with_budget(PrivacyBudget::new(8.0, 1e-5));
//!
//! let dp_sgd = DpSgd::new(SGD::new(0.01), config);
//! ```
//!
//! # References
//!
//! \[3\] Abadi et al. (2016) - Deep Learning with Differential Privacy
//! \[4\] Mironov (2017) - Renyi Differential Privacy

pub mod accountant;
pub mod budget;
pub mod config;
pub mod dp_sgd;
pub mod error;
pub mod gradient;
pub mod utils;

#[cfg(test)]
mod tests;

// Re-exports for API compatibility
pub use accountant::{compute_rdp_gaussian, rdp_to_dp, RdpAccountant};
pub use budget::PrivacyBudget;
pub use config::DpSgdConfig;
pub use dp_sgd::DpSgd;
pub use error::{DpError, Result};
pub use gradient::{add_gaussian_noise, clip_gradient, grad_norm};
pub use utils::{estimate_noise_multiplier, privacy_cost_per_step};
