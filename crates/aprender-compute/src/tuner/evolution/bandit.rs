//! Bandit-based kernel selection (MLT-13)
//!
//! UCB1 and Thompson Sampling algorithms for kernel exploration vs exploitation.

use super::super::types::KernelType;

// ============================================================================
// KernelArm
// ============================================================================

/// Bandit arm for kernel selection (MLT-13)
#[derive(Debug, Clone, Default)]
pub struct KernelArm {
    /// Number of times this kernel was selected
    pub pulls: u32,
    /// Sum of rewards (normalized throughput)
    pub total_reward: f32,
    /// Sum of squared rewards (for variance estimation)
    pub total_reward_sq: f32,
}

impl KernelArm {
    /// Get mean reward
    #[allow(clippy::cast_precision_loss)] // Pull counts << 2^24
    pub fn mean(&self) -> f32 {
        if self.pulls == 0 {
            0.0
        } else {
            self.total_reward / self.pulls as f32
        }
    }

    /// Get UCB score (Upper Confidence Bound)
    #[allow(clippy::cast_precision_loss)] // Pull counts << 2^24
    pub fn ucb(&self, total_pulls: u32, c: f32) -> f32 {
        if self.pulls == 0 {
            f32::INFINITY // Unexplored arms have infinite UCB
        } else {
            self.mean() + c * (2.0 * (total_pulls.max(1) as f32).ln() / self.pulls as f32).sqrt()
        }
    }
}

// ============================================================================
// KernelBandit
// ============================================================================

/// Bandit-based kernel selector (MLT-13)
///
/// Uses UCB1 algorithm for exploration vs exploitation.
/// Reference: Li et al. (2010) "A Contextual-Bandit Approach"
#[derive(Debug, Clone, Default)]
pub struct KernelBandit {
    /// Arms for each kernel type
    pub(crate) arms: Vec<KernelArm>,
    /// Total number of pulls across all arms
    pub(crate) total_pulls: u32,
    /// Exploration parameter (higher = more exploration)
    pub(crate) exploration_c: f32,
    /// Whether to use Thompson Sampling (alternative to UCB)
    pub(crate) use_thompson: bool,
}

impl KernelBandit {
    /// Number of kernel types
    pub const NUM_KERNELS: usize = 12;

    /// Create a new bandit with default exploration
    pub fn new() -> Self {
        Self {
            arms: vec![KernelArm::default(); Self::NUM_KERNELS],
            total_pulls: 0,
            exploration_c: 2.0, // sqrt(2) is theoretically optimal
            use_thompson: false,
        }
    }

    /// Create a bandit with Thompson Sampling
    pub fn with_thompson_sampling() -> Self {
        Self {
            arms: vec![KernelArm::default(); Self::NUM_KERNELS],
            total_pulls: 0,
            exploration_c: 2.0,
            use_thompson: true,
        }
    }

    /// Select kernel using UCB1 or Thompson Sampling
    pub fn select(&self) -> KernelType {
        let idx = if self.use_thompson { self.select_thompson() } else { self.select_ucb() };
        KernelType::from_index(idx)
    }

    #[allow(clippy::cast_precision_loss)] // Index/count values << 2^24; modulo bounds ensure safe range
    fn select_ucb(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.ucb(self.total_pulls, self.exploration_c)
                    .partial_cmp(&b.ucb(self.total_pulls, self.exploration_c))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[allow(clippy::cast_precision_loss)]
    // SAFETY: usize index <= 11 (NUM_KERNELS) and u64 modulo 1000 -- both lossless in f32.
    fn select_thompson(&self) -> usize {
        // Thompson Sampling with Beta distribution approximation
        // For each arm, sample from Beta(successes+1, failures+1)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple pseudo-random based on current state
        let mut hasher = DefaultHasher::new();
        self.total_pulls.hash(&mut hasher);
        let seed = hasher.finish();

        self.arms
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                let sample_a =
                    a.mean() + 0.1 * ((seed.wrapping_add(*i as u64) % 1000) as f32 / 1000.0 - 0.5);
                let sample_b =
                    b.mean() + 0.1 * ((seed.wrapping_add(*j as u64) % 1000) as f32 / 1000.0 - 0.5);
                sample_a.partial_cmp(&sample_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update arm with observed reward
    pub fn update(&mut self, kernel: KernelType, reward: f32) {
        contract_pre_update!();
        let idx = kernel.to_index();
        if idx < self.arms.len() {
            self.arms[idx].pulls += 1;
            self.arms[idx].total_reward += reward;
            self.arms[idx].total_reward_sq += reward * reward;
            self.total_pulls += 1;
        }
    }

    /// Get the best kernel based on empirical mean
    pub fn best_kernel(&self) -> KernelType {
        let idx = self
            .arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.mean().partial_cmp(&b.mean()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        KernelType::from_index(idx)
    }

    /// Get exploration rate (fraction of pulls that were exploratory)
    #[allow(clippy::cast_precision_loss)] // Pull counts << 2^24
    pub fn exploration_rate(&self) -> f32 {
        if self.total_pulls == 0 {
            return 1.0;
        }
        let best_pulls = self.arms.iter().map(|a| a.pulls).max().unwrap_or(0);
        1.0 - (best_pulls as f32 / self.total_pulls as f32)
    }

    /// Get regret estimate (cumulative regret vs oracle)
    #[allow(clippy::cast_precision_loss)] // Pull counts << 2^24
    pub fn estimated_regret(&self) -> f32 {
        let best_mean = self.arms.iter().map(|a| a.mean()).fold(0.0f32, f32::max);
        self.arms.iter().map(|a| (best_mean - a.mean()) * a.pulls as f32).sum()
    }
}
