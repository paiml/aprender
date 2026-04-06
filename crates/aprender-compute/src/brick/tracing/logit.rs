// ============================================================================
// E.11.4: LogitEvolutionTrace (MLT-03)
// ============================================================================

/// Logit evolution for a single token through layers.
///
/// Tracks how a token's logit value and rank change as hidden states
/// pass through transformer layers.
#[derive(Debug, Clone, Default)]
pub struct TokenLogitEvolution {
    /// Token ID being tracked
    pub token_id: u32,
    /// Token string representation (for display)
    pub token_str: String,
    /// Logit value after each layer's contribution
    pub per_layer_logit: Vec<f32>,
    /// Rank among vocabulary at each layer (0 = highest probability)
    pub per_layer_rank: Vec<usize>,
    /// Final probability after softmax
    pub final_probability: f32,
    /// Final rank (0 = selected token)
    pub final_rank: usize,
}

impl TokenLogitEvolution {
    /// Create a new token evolution tracker.
    pub fn new(token_id: u32, token_str: String) -> Self {
        Self { token_id, token_str, ..Default::default() }
    }

    /// Record logit value at a layer.
    pub fn record_layer(&mut self, logit: f32, rank: usize) {
        self.per_layer_logit.push(logit);
        self.per_layer_rank.push(rank);
    }

    /// Get the layer where this token's rank changed most dramatically.
    pub fn decisive_layer(&self) -> Option<usize> {
        if self.per_layer_rank.len() < 2 {
            return None;
        }

        let mut max_change = 0i64;
        let mut decisive = 0;

        for i in 1..self.per_layer_rank.len() {
            let change = (self.per_layer_rank[i] as i64 - self.per_layer_rank[i - 1] as i64).abs();
            if change > max_change {
                max_change = change;
                decisive = i;
            }
        }

        Some(decisive)
    }
}

/// Full logit trace for one generation step.
#[derive(Debug, Clone, Default)]
pub struct LogitEvolutionTrace {
    /// Position being generated
    pub position: usize,
    /// Tokens being tracked (typically top-k candidates + ground truth)
    pub tracked_tokens: Vec<TokenLogitEvolution>,
    /// Which layer had the largest impact on the selected token
    pub decisive_layer: usize,
    /// Temperature used for sampling
    pub temperature: f32,
    /// Top-p (nucleus) value used
    pub top_p: f32,
}

impl LogitEvolutionTrace {
    /// Create a new logit evolution trace.
    pub fn new(position: usize, temperature: f32, top_p: f32) -> Self {
        Self { position, temperature, top_p, ..Default::default() }
    }

    /// Add a token to track.
    pub fn track_token(&mut self, token_id: u32, token_str: String) -> &mut TokenLogitEvolution {
        self.tracked_tokens.push(TokenLogitEvolution::new(token_id, token_str));
        self.tracked_tokens.last_mut().expect("invariant: just pushed")
    }

    /// Compute rank of a token in a logit distribution.
    pub fn compute_rank(logits: &[f32], token_id: u32) -> usize {
        let target_logit = logits.get(token_id as usize).copied().unwrap_or(f32::MIN);

        logits.iter().filter(|&&l| l > target_logit).count()
    }

    /// Finalize the trace after generation completes.
    pub fn finalize(&mut self, selected_token_id: u32) {
        // Find the decisive layer for the selected token
        for token in &self.tracked_tokens {
            if token.token_id == selected_token_id {
                if let Some(layer) = token.decisive_layer() {
                    self.decisive_layer = layer;
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_logit_evolution() {
        let mut evo = TokenLogitEvolution::new(42, "test".to_string());
        evo.record_layer(1.0, 100);
        evo.record_layer(2.0, 50);
        evo.record_layer(3.0, 10);

        assert_eq!(evo.per_layer_logit.len(), 3);
        assert_eq!(evo.per_layer_rank.len(), 3);
        assert_eq!(evo.decisive_layer(), Some(1)); // 100->50 is biggest jump
    }

    #[test]
    fn test_logit_evolution_trace_compute_rank() {
        let logits = vec![1.0, 5.0, 3.0, 2.0]; // sorted: 5, 3, 2, 1
                                               // Token 0 has logit 1.0, rank 3 (3 values above it)
        assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 0), 3);
        // Token 1 has logit 5.0, rank 0 (nothing above it)
        assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 1), 0);
    }
}
