//! Statistics tracking for GAN training.

use std::collections::VecDeque;

/// Statistics from GAN training
#[derive(Debug, Clone)]
pub struct CodeGanStats {
    /// Total training steps
    pub steps: usize,
    /// Generator losses (recent history)
    pub gen_losses: VecDeque<f32>,
    /// Discriminator losses (recent history)
    pub disc_losses: VecDeque<f32>,
    /// Mode collapse score (0 = no collapse, 1 = full collapse)
    pub mode_collapse_score: f32,
    /// Number of unique tokens generated in last batch
    pub unique_tokens: usize,
}

impl Default for CodeGanStats {
    fn default() -> Self {
        Self {
            steps: 0,
            gen_losses: VecDeque::with_capacity(100),
            disc_losses: VecDeque::with_capacity(100),
            mode_collapse_score: 0.0,
            unique_tokens: 0,
        }
    }
}
