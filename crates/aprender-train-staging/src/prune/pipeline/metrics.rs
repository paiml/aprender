//! Pruning metrics collection
//!
//! Tracks metrics collected during the pruning pipeline.

use super::stage::PruningStage;
use serde::{Deserialize, Serialize};

/// Metrics collected during pruning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PruningMetrics {
    /// Achieved sparsity (0.0 to 1.0).
    pub achieved_sparsity: f32,
    /// Target sparsity.
    pub target_sparsity: f32,
    /// Total parameters in model.
    pub total_parameters: usize,
    /// Parameters pruned (set to zero).
    pub parameters_pruned: usize,
    /// Parameters remaining (non-zero).
    pub parameters_remaining: usize,
    /// Per-layer sparsity.
    pub layer_sparsity: Vec<(String, f32)>,
    /// Pre-pruning perplexity (if evaluated).
    pub pre_prune_ppl: Option<f32>,
    /// Post-pruning perplexity (if evaluated).
    pub post_prune_ppl: Option<f32>,
    /// Perplexity increase percentage.
    pub ppl_increase_pct: Option<f32>,
    /// Fine-tuning loss curve.
    pub finetune_losses: Vec<f32>,
    /// Duration of each stage in seconds.
    pub stage_durations: Vec<(PruningStage, f64)>,
}

impl PruningMetrics {
    /// Create new metrics with target sparsity.
    pub fn new(target_sparsity: f32) -> Self {
        Self { target_sparsity, ..Default::default() }
    }

    /// Update achieved sparsity and parameter counts.
    pub fn update_sparsity(&mut self, pruned: usize, total: usize) {
        self.total_parameters = total;
        self.parameters_pruned = pruned;
        self.parameters_remaining = total.saturating_sub(pruned);
        self.achieved_sparsity = if total > 0 { pruned as f32 / total as f32 } else { 0.0 };
    }

    /// Add layer sparsity.
    pub fn add_layer_sparsity(&mut self, name: impl Into<String>, sparsity: f32) {
        self.layer_sparsity.push((name.into(), sparsity));
    }

    /// Set pre-pruning perplexity.
    pub fn set_pre_prune_ppl(&mut self, ppl: f32) {
        self.pre_prune_ppl = Some(ppl);
    }

    /// Set post-pruning perplexity and compute increase.
    pub fn set_post_prune_ppl(&mut self, ppl: f32) {
        self.post_prune_ppl = Some(ppl);
        if let Some(pre) = self.pre_prune_ppl {
            if pre > 0.0 {
                self.ppl_increase_pct = Some((ppl - pre) / pre * 100.0);
            }
        }
    }

    /// Record a fine-tuning loss.
    pub fn record_finetune_loss(&mut self, loss: f32) {
        self.finetune_losses.push(loss);
    }

    /// Record stage duration.
    pub fn record_stage_duration(&mut self, stage: PruningStage, duration_secs: f64) {
        self.stage_durations.push((stage, duration_secs));
    }

    /// Get sparsity gap (target - achieved).
    pub fn sparsity_gap(&self) -> f32 {
        self.target_sparsity - self.achieved_sparsity
    }

    /// Check if target sparsity was achieved.
    pub fn target_achieved(&self) -> bool {
        self.achieved_sparsity >= self.target_sparsity - 1e-4
    }

    /// Get mean layer sparsity.
    pub fn mean_layer_sparsity(&self) -> f32 {
        if self.layer_sparsity.is_empty() {
            return self.achieved_sparsity;
        }
        let sum: f32 = self.layer_sparsity.iter().map(|(_, s)| s).sum();
        sum / self.layer_sparsity.len() as f32
    }

    /// Get sparsity variance across layers.
    pub fn layer_sparsity_variance(&self) -> f32 {
        if self.layer_sparsity.is_empty() {
            return 0.0;
        }
        let mean = self.mean_layer_sparsity();
        let variance: f32 =
            self.layer_sparsity.iter().map(|(_, s)| (s - mean).powi(2)).sum::<f32>()
                / self.layer_sparsity.len().max(1) as f32;
        variance
    }

    /// Get total pipeline duration.
    pub fn total_duration_secs(&self) -> f64 {
        self.stage_durations.iter().map(|(_, d)| d).sum()
    }
}
