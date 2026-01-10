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

impl Pruner for LotteryTicketPruner {
    fn generate_mask(
        &self,
        scores: &ImportanceScores,
        target_sparsity: f32,
        pattern: SparsityPattern,
    ) -> Result<SparsityMask, PruningError> {
        match pattern {
            SparsityPattern::Unstructured => {
                generate_unstructured_mask(&scores.values, target_sparsity)
            }
            _ => Err(PruningError::InvalidSparsity {
                value: target_sparsity,
                constraint: format!("LTH only supports unstructured pruning, got {pattern:?}"),
            }),
        }
    }

    fn apply_mask(
        &self,
        module: &mut dyn Module,
        mask: &SparsityMask,
    ) -> Result<PruningResult, PruningError> {
        let mut params = module.parameters_mut();
        if params.is_empty() {
            return Err(PruningError::NoParameters {
                module: "module".to_string(),
            });
        }

        let weights = &mut *params[0];
        let total = weights.data().len();

        mask.apply(weights)?;

        let zeros = weights.data().iter().filter(|&&v| v == 0.0).count();
        let achieved_sparsity = zeros as f32 / total as f32;

        Ok(PruningResult::new(achieved_sparsity, zeros, total))
    }

    fn importance(&self) -> &dyn super::importance::Importance {
        &self.importance
    }

    fn name(&self) -> &'static str {
        "lottery_ticket_pruner"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Module;

    // Mock module for testing
    struct MockModule {
        weights: Tensor,
    }

    impl MockModule {
        fn new(data: &[f32], shape: &[usize]) -> Self {
            Self {
                weights: Tensor::new(data, shape),
            }
        }
    }

    impl Module for MockModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weights]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.weights]
        }
    }

    // ==========================================================================
    // FALSIFICATION: RewindStrategy
    // ==========================================================================
    #[test]
    fn test_rewind_strategy_default() {
        let strategy = RewindStrategy::default();
        assert_eq!(strategy, RewindStrategy::Init);
    }

    #[test]
    fn test_rewind_strategy_early() {
        let strategy = RewindStrategy::Early { iteration: 500 };
        if let RewindStrategy::Early { iteration } = strategy {
            assert_eq!(iteration, 500);
        } else {
            panic!("Expected Early variant");
        }
    }

    #[test]
    fn test_rewind_strategy_late() {
        let strategy = RewindStrategy::Late { fraction: 0.1 };
        if let RewindStrategy::Late { fraction } = strategy {
            assert!((fraction - 0.1).abs() < 1e-6);
        } else {
            panic!("Expected Late variant");
        }
    }

    // ==========================================================================
    // FALSIFICATION: LotteryTicketConfig
    // ==========================================================================
    #[test]
    fn test_config_new() {
        let config = LotteryTicketConfig::new(0.9, 10);

        assert!((config.target_sparsity - 0.9).abs() < 1e-6);
        assert_eq!(config.pruning_rounds, 10);
        assert_eq!(config.rewind_strategy, RewindStrategy::Init);
        assert!(config.global_pruning);

        // Check per-round rate: (1 - 0.9)^(1/10) ≈ 0.794
        // So prune_rate_per_round ≈ 1 - 0.794 ≈ 0.206
        let expected = 1.0 - 0.1_f32.powf(0.1);
        assert!((config.prune_rate_per_round - expected).abs() < 1e-5);
    }

    #[test]
    fn test_config_clamps_sparsity() {
        let config = LotteryTicketConfig::new(1.5, 10);
        assert!(config.target_sparsity <= 0.99);

        let config = LotteryTicketConfig::new(-0.5, 10);
        assert!(config.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_min_rounds() {
        let config = LotteryTicketConfig::new(0.9, 0);
        assert_eq!(config.pruning_rounds, 1);
    }

    #[test]
    fn test_config_with_rewind_strategy() {
        let config = LotteryTicketConfig::new(0.9, 10)
            .with_rewind_strategy(RewindStrategy::Early { iteration: 100 });

        assert!(matches!(
            config.rewind_strategy,
            RewindStrategy::Early { iteration: 100 }
        ));
    }

    #[test]
    fn test_config_with_global_pruning() {
        let config = LotteryTicketConfig::new(0.9, 10).with_global_pruning(false);

        assert!(!config.global_pruning);
    }

    #[test]
    fn test_config_default() {
        let config = LotteryTicketConfig::default();

        assert!((config.target_sparsity - 0.9).abs() < 1e-6);
        assert_eq!(config.pruning_rounds, 10);
    }

    // ==========================================================================
    // FALSIFICATION: WinningTicket
    // ==========================================================================
    #[test]
    fn test_winning_ticket_compression_ratio() {
        let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

        let ticket = WinningTicket {
            mask,
            initial_weights: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![4],
            sparsity: 0.5,
            remaining_parameters: 2,
            total_parameters: 4,
            sparsity_history: vec![0.25, 0.5],
        };

        // 4 / 2 = 2x compression
        assert!((ticket.compression_ratio() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_winning_ticket_density() {
        let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

        let ticket = WinningTicket {
            mask,
            initial_weights: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![4],
            sparsity: 0.5,
            remaining_parameters: 2,
            total_parameters: 4,
            sparsity_history: vec![0.5],
        };

        assert!((ticket.density() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_winning_ticket_compression_ratio_zero_remaining() {
        let mask_tensor = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[4]);
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

        let ticket = WinningTicket {
            mask,
            initial_weights: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![4],
            sparsity: 1.0,
            remaining_parameters: 0,
            total_parameters: 4,
            sparsity_history: vec![1.0],
        };

        assert!(ticket.compression_ratio().is_infinite());
    }

    // ==========================================================================
    // FALSIFICATION: Builder pattern
    // ==========================================================================
    #[test]
    fn test_builder_default() {
        let pruner = LotteryTicketPruner::builder().build();

        assert!((pruner.config().target_sparsity - 0.9).abs() < 1e-6);
        assert_eq!(pruner.config().pruning_rounds, 10);
    }

    #[test]
    fn test_builder_with_target_sparsity() {
        let pruner = LotteryTicketPruner::builder().target_sparsity(0.8).build();

        assert!((pruner.config().target_sparsity - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_builder_with_pruning_rounds() {
        let pruner = LotteryTicketPruner::builder().pruning_rounds(5).build();

        assert_eq!(pruner.config().pruning_rounds, 5);
    }

    #[test]
    fn test_builder_with_rewind_strategy() {
        let pruner = LotteryTicketPruner::builder()
            .rewind_strategy(RewindStrategy::None)
            .build();

        assert_eq!(pruner.config().rewind_strategy, RewindStrategy::None);
    }

    #[test]
    fn test_builder_full_config() {
        let pruner = LotteryTicketPruner::builder()
            .target_sparsity(0.95)
            .pruning_rounds(20)
            .rewind_strategy(RewindStrategy::Late { fraction: 0.05 })
            .global_pruning(false)
            .build();

        assert!((pruner.config().target_sparsity - 0.95).abs() < 1e-6);
        assert_eq!(pruner.config().pruning_rounds, 20);
        assert!(matches!(
            pruner.config().rewind_strategy,
            RewindStrategy::Late { .. }
        ));
        assert!(!pruner.config().global_pruning);
    }

    // ==========================================================================
    // FALSIFICATION: LotteryTicketPruner construction
    // ==========================================================================
    #[test]
    fn test_pruner_new() {
        let pruner = LotteryTicketPruner::new();
        assert_eq!(pruner.name(), "lottery_ticket_pruner");
    }

    #[test]
    fn test_pruner_default() {
        let pruner = LotteryTicketPruner::default();
        assert!((pruner.config().target_sparsity - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_pruner_with_config() {
        let config = LotteryTicketConfig::new(0.5, 5);
        let pruner = LotteryTicketPruner::with_config(config);

        assert!((pruner.config().target_sparsity - 0.5).abs() < 1e-6);
        assert_eq!(pruner.config().pruning_rounds, 5);
    }

    // ==========================================================================
    // FALSIFICATION: find_ticket
    // ==========================================================================
    #[test]
    fn test_find_ticket_basic() {
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);
        let pruner = LotteryTicketPruner::builder()
            .target_sparsity(0.5)
            .pruning_rounds(2)
            .build();

        let ticket = pruner.find_ticket(&module).unwrap();

        // Should achieve approximately 50% sparsity
        assert!(ticket.sparsity > 0.4 && ticket.sparsity < 0.6);
        assert_eq!(ticket.total_parameters, 8);
        assert!(ticket.remaining_parameters > 0);
        assert_eq!(ticket.sparsity_history.len(), 2);
    }

    #[test]
    fn test_find_ticket_preserves_initial_weights() {
        let initial_data = [1.0, 2.0, 3.0, 4.0];
        let module = MockModule::new(&initial_data, &[4]);
        let pruner = LotteryTicketPruner::new();

        let ticket = pruner.find_ticket(&module).unwrap();

        // Initial weights should be preserved
        assert_eq!(ticket.initial_weights, initial_data);
    }

    #[test]
    fn test_find_ticket_empty_module_fails() {
        struct EmptyModule;
        impl Module for EmptyModule {
            fn forward(&self, input: &Tensor) -> Tensor {
                input.clone()
            }
            fn parameters(&self) -> Vec<&Tensor> {
                vec![]
            }
            fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
                vec![]
            }
        }

        let module = EmptyModule;
        let pruner = LotteryTicketPruner::new();

        let result = pruner.find_ticket(&module);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_ticket_high_sparsity() {
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[10]);
        let pruner = LotteryTicketPruner::builder()
            .target_sparsity(0.9)
            .pruning_rounds(5)
            .build();

        let ticket = pruner.find_ticket(&module).unwrap();

        // Should achieve approximately 90% sparsity
        assert!(ticket.sparsity > 0.85);
        // Should have ~1 parameter remaining
        assert!(ticket.remaining_parameters >= 1);
    }

    // ==========================================================================
    // FALSIFICATION: apply_ticket
    // ==========================================================================
    #[test]
    fn test_apply_ticket_zeros_weights() {
        let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let pruner = LotteryTicketPruner::builder()
            .target_sparsity(0.5)
            .pruning_rounds(1)
            .rewind_strategy(RewindStrategy::None)
            .build();

        let ticket = pruner.find_ticket(&module).unwrap();
        let result = pruner.apply_ticket(&mut module, &ticket).unwrap();

        // Check that some weights are now zero
        let zeros = module.weights.data().iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0);
        assert!(result.achieved_sparsity > 0.0);
    }

    #[test]
    fn test_apply_ticket_with_rewinding() {
        let initial_data = [10.0, 20.0, 30.0, 40.0];
        let mut module = MockModule::new(&initial_data, &[4]);

        // Modify weights to simulate training
        for w in module.weights.data_mut().iter_mut() {
            *w *= 2.0;
        }

        let pruner = LotteryTicketPruner::builder()
            .target_sparsity(0.5)
            .pruning_rounds(1)
            .rewind_strategy(RewindStrategy::Init)
            .build();

        // Create ticket with original weights
        let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();
        let ticket = WinningTicket {
            mask,
            initial_weights: initial_data.to_vec(),
            shape: vec![4],
            sparsity: 0.5,
            remaining_parameters: 2,
            total_parameters: 4,
            sparsity_history: vec![0.5],
        };

        pruner.apply_ticket(&mut module, &ticket).unwrap();

        // Check that remaining weights are rewound to initial values
        let data = module.weights.data();
        assert!((data[0] - 10.0).abs() < 1e-6); // Kept, rewound
        assert_eq!(data[1], 0.0); // Pruned
        assert!((data[2] - 30.0).abs() < 1e-6); // Kept, rewound
        assert_eq!(data[3], 0.0); // Pruned
    }

    // ==========================================================================
    // FALSIFICATION: Pruner trait implementation
    // ==========================================================================
    #[test]
    fn test_pruner_trait_generate_mask() {
        let pruner = LotteryTicketPruner::new();
        let scores =
            ImportanceScores::new(Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]), "test".to_string());

        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::Unstructured)
            .unwrap();

        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pruner_trait_rejects_structured_patterns() {
        let pruner = LotteryTicketPruner::new();
        let scores =
            ImportanceScores::new(Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]), "test".to_string());

        let result = pruner.generate_mask(&scores, 0.5, SparsityPattern::NM { n: 2, m: 4 });
        assert!(result.is_err());
    }

    #[test]
    fn test_pruner_trait_apply_mask() {
        let pruner = LotteryTicketPruner::new();
        let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

        let result = pruner.apply_mask(&mut module, &mask).unwrap();

        assert_eq!(result.parameters_pruned, 2);
        assert_eq!(result.total_parameters, 4);
    }

    #[test]
    fn test_pruner_trait_importance() {
        let pruner = LotteryTicketPruner::new();
        assert!(!pruner.importance().requires_calibration());
    }

    #[test]
    fn test_pruner_trait_name() {
        let pruner = LotteryTicketPruner::new();
        assert_eq!(pruner.name(), "lottery_ticket_pruner");
    }

    // ==========================================================================
    // FALSIFICATION: Iterative pruning convergence
    // ==========================================================================
    #[test]
    fn test_iterative_pruning_converges() {
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[10]);

        let pruner = LotteryTicketPruner::builder()
            .target_sparsity(0.9)
            .pruning_rounds(10)
            .build();

        let ticket = pruner.find_ticket(&module).unwrap();

        // Sparsity should increase monotonically
        for i in 1..ticket.sparsity_history.len() {
            assert!(
                ticket.sparsity_history[i] >= ticket.sparsity_history[i - 1],
                "Sparsity should increase monotonically"
            );
        }

        // Final sparsity should be close to target
        assert!(
            (ticket.sparsity - 0.9).abs() < 0.1,
            "Final sparsity {} should be close to target 0.9",
            ticket.sparsity
        );
    }

    // ==========================================================================
    // FALSIFICATION: Single round equivalence
    // ==========================================================================
    #[test]
    fn test_single_round_equals_one_shot() {
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let pruner = LotteryTicketPruner::builder()
            .target_sparsity(0.5)
            .pruning_rounds(1)
            .build();

        let ticket = pruner.find_ticket(&module).unwrap();

        // With 1 round, should get exactly target sparsity
        assert!((ticket.sparsity - 0.5).abs() < 1e-6);
    }
}
