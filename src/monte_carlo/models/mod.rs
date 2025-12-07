//! Financial simulation models
//!
//! Provides stock price models (GBM, jump-diffusion, empirical bootstrap)
//! and business revenue models.
//!
//! References:
//! - Black & Scholes (1973), "The Pricing of Options and Corporate Liabilities"
//! - Merton (1976), "Option pricing when underlying stock returns are discontinuous"

mod bootstrap;
mod gbm;
mod jump_diffusion;

pub use bootstrap::EmpiricalBootstrap;
pub use gbm::GeometricBrownianMotion;
pub use jump_diffusion::MertonJumpDiffusion;
