//! Configuration for auto-retraining.

use super::policy::RetrainPolicy;

/// Configuration for auto-retraining
#[derive(Clone, Debug)]
pub struct RetrainConfig {
    /// Policy for when to trigger retraining
    pub policy: RetrainPolicy,
    /// Cooldown period between retrains (in batches processed)
    pub cooldown_batches: usize,
    /// Maximum retrains per session (0 = unlimited)
    pub max_retrains: usize,
    /// Whether to log warnings for non-critical drift
    pub log_warnings: bool,
}

impl Default for RetrainConfig {
    fn default() -> Self {
        Self {
            policy: RetrainPolicy::default(),
            cooldown_batches: 100,
            max_retrains: 0,
            log_warnings: true,
        }
    }
}
