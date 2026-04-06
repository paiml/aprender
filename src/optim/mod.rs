//! Optimizers for training neural networks

mod adam;
mod adamw;
mod clip;
mod convergence_tests; // Tests split into convergence_tests/ directory
pub mod dp;
pub mod hpo;
mod optimizer;
mod scheduler;
mod sgd;
mod simd;

pub use adam::Adam;
pub use adamw::AdamW;
pub use clip::{clip_grad_norm, clip_grad_norm_refs};
pub use dp::{
    add_gaussian_noise, clip_gradient, estimate_noise_multiplier, grad_norm, privacy_cost_per_step,
    DpError, DpSgd, DpSgdConfig, PrivacyBudget, RdpAccountant,
};
pub use hpo::{
    AcquisitionFunction, GridSearch, HPOError, HyperbandScheduler, HyperparameterSpace,
    ParameterDomain, ParameterValue, SearchStrategy, SurrogateModel, TPEOptimizer, Trial,
    TrialStatus,
};
pub use optimizer::Optimizer;
pub use scheduler::{
    CosineAnnealingLR, LRScheduler, LinearWarmupLR, StepDecayLR, WarmupCosineDecayLR,
};
pub use sgd::SGD;
pub use simd::{simd_adam_update, simd_adamw_update, simd_axpy};
