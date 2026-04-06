//! Speculative decoding and helper types for continuous batching.

use std::fmt;

use super::request::Token;

/// Exponential moving average for tracking metrics.
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    /// Current value
    value: f64,
    /// Smoothing factor (0-1)
    alpha: f64,
    /// Number of samples
    count: u64,
}

impl ExponentialMovingAverage {
    /// Create new EMA with smoothing factor.
    pub fn new(alpha: f64) -> Self {
        Self { value: 0.0, alpha: alpha.clamp(0.0, 1.0), count: 0 }
    }

    /// Update with new sample.
    pub fn update(&mut self, sample: f64) {
        if self.count == 0 {
            self.value = sample;
        } else {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        }
        self.count += 1;
    }

    /// Get current value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.count = 0;
    }
}

impl Default for ExponentialMovingAverage {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Output from speculative decoding step.
#[derive(Debug, Clone)]
pub struct SpeculativeOutput {
    /// Accepted tokens from draft
    pub accepted: Vec<Token>,
    /// Rejection index (first rejected draft token)
    pub rejection_idx: Option<usize>,
    /// Token from target model (after rejection or all accepted)
    pub target_token: Token,
    /// Total draft tokens proposed
    pub draft_count: usize,
}

impl SpeculativeOutput {
    /// Calculate acceptance rate for this step.
    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_count == 0 {
            return 0.0;
        }
        self.accepted.len() as f64 / self.draft_count as f64
    }

    /// Number of tokens produced (accepted + 1 target).
    pub fn total_tokens(&self) -> usize {
        self.accepted.len() + 1
    }
}

/// Speculative decoding coordinator.
///
/// Coordinates draft and target models for speculative decoding.
/// The draft model proposes tokens, target model verifies.
#[derive(Debug)]
pub struct SpeculativeDecoder {
    /// Speculation depth (draft tokens per step)
    k: usize,
    /// Acceptance rate tracker
    acceptance_rate: ExponentialMovingAverage,
    /// Total steps
    total_steps: u64,
    /// Total accepted tokens
    total_accepted: u64,
    /// Total draft tokens
    total_draft: u64,
}

impl SpeculativeDecoder {
    /// Create new speculative decoder.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            acceptance_rate: ExponentialMovingAverage::new(0.1),
            total_steps: 0,
            total_accepted: 0,
            total_draft: 0,
        }
    }

    /// Get speculation depth.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Set speculation depth.
    pub fn set_k(&mut self, k: usize) {
        self.k = k;
    }

    /// Simulate speculative decoding step.
    ///
    /// In a real implementation, this would:
    /// 1. Run draft model k times to get draft tokens
    /// 2. Run target model once on all draft positions
    /// 3. Compare and accept/reject
    pub fn simulate_step(
        &mut self,
        draft_tokens: &[Token],
        target_probs: &[(Token, f64)],
    ) -> SpeculativeOutput {
        let draft_count = draft_tokens.len().min(self.k);
        let mut accepted = Vec::new();
        let mut rejection_idx = None;

        // Simulate acceptance (simplified: accept if target agrees)
        for (i, &draft_token) in draft_tokens.iter().take(draft_count).enumerate() {
            if let Some((target_token, _)) = target_probs.get(i) {
                if *target_token == draft_token {
                    accepted.push(draft_token);
                } else {
                    rejection_idx = Some(i);
                    break;
                }
            } else {
                rejection_idx = Some(i);
                break;
            }
        }

        // Get target token (either after rejection or as the k+1 token)
        let target_token = if let Some(idx) = rejection_idx {
            target_probs.get(idx).map(|(t, _)| *t).unwrap_or(0)
        } else {
            target_probs.get(draft_count).map(|(t, _)| *t).unwrap_or(0)
        };

        let output = SpeculativeOutput {
            accepted: accepted.clone(),
            rejection_idx,
            target_token,
            draft_count,
        };

        // Update statistics
        self.total_steps += 1;
        self.total_accepted += accepted.len() as u64;
        self.total_draft += draft_count as u64;
        self.acceptance_rate.update(output.acceptance_rate());

        output
    }

    /// Current acceptance rate (EMA).
    pub fn acceptance_rate(&self) -> f64 {
        self.acceptance_rate.value()
    }

    /// Overall acceptance rate.
    pub fn overall_acceptance_rate(&self) -> f64 {
        if self.total_draft == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_draft as f64
    }

    /// Effective speedup vs naive decoding.
    ///
    /// Speedup = (accepted + 1) / (1 + verification_cost_ratio)
    /// Simplified: assume verification cost = 1/k of naive
    pub fn speedup(&self) -> f64 {
        let rate = self.acceptance_rate();
        // Expected tokens per step = 1 + k * acceptance_rate
        // Cost = 1 (target call) + k * draft_cost (assume draft_cost << 1)
        // Simplified model: speedup ≈ 1 + k * acceptance_rate
        1.0 + (self.k as f64) * rate
    }

    /// Get statistics.
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.total_steps, self.total_accepted, self.total_draft)
    }
}

impl fmt::Display for SpeculativeDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpeculativeDecoder(k={}, acceptance={:.1}%, speedup={:.2}x)",
            self.k,
            self.acceptance_rate() * 100.0,
            self.speedup()
        )
    }
}
