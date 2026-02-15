//! Lottery Ticket Hypothesis implementation.
//!
//! This module implements the Lottery Ticket Hypothesis (LTH) from:
//! Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis:
//! Finding Sparse, Trainable Neural Networks. arXiv:1803.03635.
//!
//! # Key Insight
//! Dense neural networks contain sparse subnetworks ("winning tickets") that
//! can achieve comparable test accuracy when trained from scratch with their
//! original initialization.
//!
//! # Algorithm: Iterative Magnitude Pruning (IMP)
//! 1. Initialize network with weights W₀
//! 2. Train network to convergence → W_T
//! 3. Prune p% of smallest magnitude weights → mask M
//! 4. Reset remaining weights to W₀ (or W_k for late rewinding)
//! 5. Repeat from step 2 with masked network
//!
//! # Toyota Way Principles
//! - **Jidoka**: Validate weight tensors at each pruning round
//! - **Poka-Yoke**: Type-safe configuration prevents invalid settings
//! - **Genchi Genbutsu**: Uses actual trained weights for importance
//!
//! # Example
//!
//! ```ignore
//! use aprender::pruning::{LotteryTicketPruner, RewindStrategy};
//!
//! let pruner = LotteryTicketPruner::builder()
//!     .target_sparsity(0.9)
//!     .pruning_rounds(10)
//!     .rewind_strategy(RewindStrategy::Init)
//!     .build();
//!
//! let ticket = pruner.find_ticket(&model, &train_fn)?;
//! ```

use super::error::PruningError;
use super::importance::ImportanceScores;
use super::mask::{generate_unstructured_mask, SparsityMask, SparsityPattern};
use super::pruner::{Pruner, PruningResult};
use super::MagnitudeImportance;
use crate::autograd::Tensor;
use crate::nn::Module;

/// Strategy for rewinding weights after pruning.
///
/// # References
/// - Init rewinding: Original LTH paper (Frankle & Carbin, 2018)
/// - Late rewinding: "Stabilizing the Lottery Ticket Hypothesis" (Frankle et al., 2019)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RewindStrategy {
    /// Rewind to initialization (W₀).
    /// Original LTH approach - works for small networks.
    #[default]
    Init,

    /// Rewind to early training iteration k (W_k).
    /// More stable for deeper networks. k is typically 0.1-1% of training.
    Early {
        /// Iteration to rewind to (e.g., 500 for 50k total iterations).
        iteration: usize,
    },

    /// Rewind to late training iteration.
    /// Used when early rewinding still fails.
    Late {
        /// Fraction of training to complete before capturing rewind point (0.0-1.0).
        fraction: f32,
    },

    /// No rewinding - just apply mask to current weights.
    /// Useful for one-shot pruning comparison.
    None,
}

/// Configuration for Lottery Ticket pruning.
#[derive(Debug, Clone)]
pub struct LotteryTicketConfig {
    /// Target sparsity (fraction of weights to prune, 0.0-1.0).
    pub target_sparsity: f32,

    /// Number of iterative pruning rounds.
    /// Each round prunes a fraction, accumulating to target_sparsity.
    pub pruning_rounds: usize,

    /// Strategy for rewinding weights after each pruning round.
    pub rewind_strategy: RewindStrategy,

    /// Pruning rate per round (computed from target_sparsity and rounds).
    /// p_per_round = 1 - (1 - target_sparsity)^(1/rounds)
    pub prune_rate_per_round: f32,

    /// Whether to use global pruning (across all layers) or per-layer.
    pub global_pruning: bool,
}

impl Default for LotteryTicketConfig {
    fn default() -> Self {
        Self::new(0.9, 10)
    }
}

impl LotteryTicketConfig {
    /// Create a new configuration.
    ///
    /// # Arguments
    /// * `target_sparsity` - Final sparsity (0.0-1.0), e.g., 0.9 = 90% pruned
    /// * `pruning_rounds` - Number of iterative pruning rounds
    #[must_use]
    pub fn new(target_sparsity: f32, pruning_rounds: usize) -> Self {
        let rounds = pruning_rounds.max(1) as f32;
        // Compute per-round pruning rate to achieve target after all rounds
        // After n rounds: remaining = (1 - p)^n = 1 - target_sparsity
        // So: p = 1 - (1 - target_sparsity)^(1/n)
        let prune_rate_per_round = 1.0 - (1.0 - target_sparsity).powf(1.0 / rounds);

        Self {
            target_sparsity: target_sparsity.clamp(0.0, 0.99),
            pruning_rounds: pruning_rounds.max(1),
            rewind_strategy: RewindStrategy::Init,
            prune_rate_per_round,
            global_pruning: true,
        }
    }

    /// Set the rewind strategy.
    #[must_use]
    pub fn with_rewind_strategy(mut self, strategy: RewindStrategy) -> Self {
        self.rewind_strategy = strategy;
        self
    }

    /// Enable or disable global pruning.
    #[must_use]
    pub fn with_global_pruning(mut self, global: bool) -> Self {
        self.global_pruning = global;
        self
    }
}

/// A "winning ticket" - the sparse subnetwork found by LTH.
///
/// Contains the pruning mask and the initial weights to use for retraining.
#[derive(Debug, Clone)]
pub struct WinningTicket {
    /// The sparsity mask identifying which weights to keep.
    pub mask: SparsityMask,

    /// The initial weights to rewind to (W₀ or W_k).
    pub initial_weights: Vec<f32>,

    /// Shape of the weight tensor.
    pub shape: Vec<usize>,

    /// Final sparsity achieved.
    pub sparsity: f32,

    /// Number of parameters remaining (non-zero).
    pub remaining_parameters: usize,

    /// Total parameters in original network.
    pub total_parameters: usize,

    /// History of sparsity at each pruning round.
    pub sparsity_history: Vec<f32>,
}

impl WinningTicket {
    /// Get compression ratio (original size / pruned size).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.remaining_parameters == 0 {
            return f32::INFINITY;
        }
        self.total_parameters as f32 / self.remaining_parameters as f32
    }

    /// Get the fraction of weights remaining.
    #[must_use]
    pub fn density(&self) -> f32 {
        1.0 - self.sparsity
    }
}

/// Builder for `LotteryTicketPruner`.
#[derive(Debug, Clone, Default)]
pub struct LotteryTicketPrunerBuilder {
    target_sparsity: Option<f32>,
    pruning_rounds: Option<usize>,
    rewind_strategy: Option<RewindStrategy>,
    global_pruning: Option<bool>,
}

impl LotteryTicketPrunerBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target sparsity (0.0-1.0).
    #[must_use]
    pub fn target_sparsity(mut self, sparsity: f32) -> Self {
        self.target_sparsity = Some(sparsity.clamp(0.0, 0.99));
        self
    }

    /// Set number of pruning rounds.
    #[must_use]
    pub fn pruning_rounds(mut self, rounds: usize) -> Self {
        self.pruning_rounds = Some(rounds.max(1));
        self
    }

    /// Set rewind strategy.
    #[must_use]
    pub fn rewind_strategy(mut self, strategy: RewindStrategy) -> Self {
        self.rewind_strategy = Some(strategy);
        self
    }

    /// Enable global pruning across all layers.
    #[must_use]
    pub fn global_pruning(mut self, global: bool) -> Self {
        self.global_pruning = Some(global);
        self
    }

    /// Build the pruner.
    #[must_use]
    pub fn build(self) -> LotteryTicketPruner {
        let target_sparsity = self.target_sparsity.unwrap_or(0.9);
        let pruning_rounds = self.pruning_rounds.unwrap_or(10);

        let mut config = LotteryTicketConfig::new(target_sparsity, pruning_rounds);

        if let Some(strategy) = self.rewind_strategy {
            config = config.with_rewind_strategy(strategy);
        }
        if let Some(global) = self.global_pruning {
            config = config.with_global_pruning(global);
        }

        LotteryTicketPruner::with_config(config)
    }
}

/// Lottery Ticket Hypothesis pruner.
///
/// Implements Iterative Magnitude Pruning (IMP) with weight rewinding
/// to find sparse, trainable subnetworks.
#[derive(Debug, Clone)]
pub struct LotteryTicketPruner {
    config: LotteryTicketConfig,
    importance: MagnitudeImportance,
}

impl Default for LotteryTicketPruner {
    fn default() -> Self {
        Self::new()
    }
}

impl LotteryTicketPruner {
    /// Create a new LTH pruner with default configuration.
    /// Default: 90% sparsity over 10 rounds with init rewinding.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(LotteryTicketConfig::default())
    }

    /// Create a pruner with custom configuration.
    #[must_use]
    pub fn with_config(config: LotteryTicketConfig) -> Self {
        Self {
            config,
            importance: MagnitudeImportance::l2(),
        }
    }

    /// Get a builder for configuring the pruner.
    #[must_use]
    pub fn builder() -> LotteryTicketPrunerBuilder {
        LotteryTicketPrunerBuilder::new()
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &LotteryTicketConfig {
        &self.config
    }

    /// Find a winning ticket from the given module.
    ///
    /// This performs iterative magnitude pruning:
    /// 1. Compute importance scores
    /// 2. Generate mask for current round's pruning rate
    /// 3. Accumulate into overall mask
    /// 4. Record weight state for rewinding
    ///
    /// Note: This is a simplified version that doesn't require training.
    /// For full LTH, use `find_ticket_with_training`.
    pub fn find_ticket(&self, module: &dyn Module) -> Result<WinningTicket, PruningError> {
        let params = module.parameters();
        if params.is_empty() {
            return Err(PruningError::NoParameters {
                module: "module".to_string(),
            });
        }

        // Get initial weights (for rewinding)
        let weights = params[0];
        let initial_weights = weights.data().to_vec();
        let shape = weights.shape().to_vec();
        let total_parameters = initial_weights.len();

        // Initialize cumulative mask (all ones = keep all)
        let mut cumulative_mask: Vec<f32> = vec![1.0; total_parameters];
        let mut sparsity_history = Vec::with_capacity(self.config.pruning_rounds);

        // Iterative pruning rounds
        for round in 0..self.config.pruning_rounds {
            // Count active weights (not yet pruned)
            let active_count = cumulative_mask.iter().filter(|&&v| v == 1.0).count();
            if active_count <= 1 {
                // Keep at least 1 weight - the "winning ticket" must have at least 1 parameter
                let zeros = cumulative_mask.iter().filter(|&&v| v == 0.0).count();
                let current_sparsity = zeros as f32 / total_parameters as f32;
                sparsity_history.push(current_sparsity);
                break;
            }

            // Compute target remaining weights after this round
            // Using the LTH formula: remaining_fraction = (1 - p)^k where k is rounds completed
            let rounds_completed = (round + 1) as i32;
            let remaining_fraction =
                (1.0 - self.config.prune_rate_per_round).powi(rounds_completed);
            let target_remaining = (total_parameters as f32 * remaining_fraction).round() as usize;
            // Ensure at least 1 weight remains
            let target_remaining = target_remaining.max(1);

            // How many to prune this round
            let to_prune = active_count.saturating_sub(target_remaining);

            if to_prune == 0 {
                // Calculate current sparsity
                let zeros = cumulative_mask.iter().filter(|&&v| v == 0.0).count();
                let current_sparsity = zeros as f32 / total_parameters as f32;
                sparsity_history.push(current_sparsity);
                continue;
            }

            // Compute importance scores for active weights only
            // Collect (index, importance) pairs for active weights
            let mut active_scores: Vec<(usize, f32)> = initial_weights
                .iter()
                .zip(cumulative_mask.iter())
                .enumerate()
                .filter(|(_, (_, &mask))| mask == 1.0)
                .map(|(i, (&w, _))| (i, w.abs()))
                .collect();

            // Sort by importance (ascending - lowest first to prune)
            active_scores
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Prune the lowest `to_prune` weights
            for (idx, _) in active_scores.iter().take(to_prune) {
                cumulative_mask[*idx] = 0.0;
            }

            // Calculate current sparsity
            let zeros = cumulative_mask.iter().filter(|&&v| v == 0.0).count();
            let current_sparsity = zeros as f32 / total_parameters as f32;
            sparsity_history.push(current_sparsity);

            // Log progress (in debug mode)
            #[cfg(debug_assertions)]
            {
                let _ = round; // Silence unused warning in release
                eprintln!(
                    "LTH Round {}/{}: sparsity = {:.2}% (pruned {} of {} active)",
                    round + 1,
                    self.config.pruning_rounds,
                    current_sparsity * 100.0,
                    to_prune,
                    active_count
                );
            }
        }

        // Create final mask
        let mask_tensor = Tensor::new(&cumulative_mask, &shape);
        let final_mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured)?;

        let remaining = cumulative_mask.iter().filter(|&&v| v != 0.0).count();
        let final_sparsity = 1.0 - (remaining as f32 / total_parameters as f32);

        Ok(WinningTicket {
            mask: final_mask,
            initial_weights,
            shape,
            sparsity: final_sparsity,
            remaining_parameters: remaining,
            total_parameters,
            sparsity_history,
        })
    }

    /// Apply a winning ticket mask to a module.
    ///
    /// This zeros out the pruned weights according to the mask.
    pub fn apply_ticket(
        &self,
        module: &mut dyn Module,
        ticket: &WinningTicket,
    ) -> Result<PruningResult, PruningError> {
        let mut params = module.parameters_mut();
        if params.is_empty() {
            return Err(PruningError::NoParameters {
                module: "module".to_string(),
            });
        }

        let weights = &mut *params[0];
        let total = weights.data().len();

        // Apply mask
        ticket.mask.apply(weights)?;

        // If rewinding is enabled, also reset to initial weights
        if self.config.rewind_strategy != RewindStrategy::None {
            let data = weights.data_mut();
            let mask_data = ticket.mask.tensor().data();

            for (i, (w, &m)) in data.iter_mut().zip(mask_data.iter()).enumerate() {
                if m != 0.0 {
                    *w = ticket.initial_weights[i];
                }
            }
        }

        let zeros = weights.data().iter().filter(|&&v| v == 0.0).count();
        let achieved_sparsity = zeros as f32 / total as f32;

        Ok(PruningResult::new(achieved_sparsity, zeros, total))
    }
}

include!("lottery_part_02.rs");
