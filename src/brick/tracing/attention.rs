// ============================================================================
// E.11.3: AttentionWeightTrace (MLT-02)
// ============================================================================

/// Sparse attention weight storage for a single head.
///
/// Records top-k attended positions to avoid storing the full attention matrix.
/// Useful for debugging repetition, context loss, and attention sinks.
#[derive(Debug, Clone, Default)]
pub struct AttentionWeightTrace {
    /// Layer index
    pub layer_idx: usize,
    /// Head index within the layer
    pub head_idx: usize,
    /// Query position (current token being generated)
    pub query_pos: usize,
    /// Top-k attended positions (sorted by weight descending)
    pub top_k_positions: Vec<usize>,
    /// Corresponding attention weights
    pub top_k_weights: Vec<f32>,
    /// Sum of weights outside top-k (attention mass lost to tail)
    pub tail_mass: f32,
    /// Entropy of attention distribution (higher = more uniform)
    pub entropy: f32,
}

impl AttentionWeightTrace {
    /// Create from full attention weights, extracting top-k.
    pub fn from_weights(
        layer_idx: usize,
        head_idx: usize,
        query_pos: usize,
        weights: &[f32],
        k: usize,
    ) -> Self {
        let k = k.min(weights.len());

        // Create position-weight pairs and sort by weight descending
        let mut pairs: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_positions: Vec<usize> = pairs.iter().take(k).map(|(pos, _)| *pos).collect();
        let top_k_weights: Vec<f32> = pairs.iter().take(k).map(|(_, w)| *w).collect();

        let top_k_mass: f32 = top_k_weights.iter().sum();
        let total_mass: f32 = weights.iter().sum();
        let tail_mass = (total_mass - top_k_mass).max(0.0);

        // Compute entropy: H = -sum(p * log(p)) for non-zero probabilities
        let entropy =
            weights.iter().filter(|&&w| w > 1e-10).map(|&w| -w * w.max(f32::EPSILON).ln()).sum();

        Self { layer_idx, head_idx, query_pos, top_k_positions, top_k_weights, tail_mass, entropy }
    }

    /// Check if attention is concentrated on first position (attention sink).
    pub fn is_attention_sink(&self, threshold: f32) -> bool {
        self.top_k_positions.first() == Some(&0)
            && self.top_k_weights.first().copied().unwrap_or(0.0) > threshold
    }

    /// Check if attention is too uniform (confused model).
    pub fn is_uniform(&self, entropy_threshold: f32) -> bool {
        self.entropy > entropy_threshold
    }

    /// Check for repetition pattern (high weight on recent positions).
    pub fn has_recency_bias(&self, recency_window: usize, threshold: f32) -> bool {
        if self.query_pos == 0 {
            return false;
        }
        let recency_start = self.query_pos.saturating_sub(recency_window);
        let recent_mass: f32 = self
            .top_k_positions
            .iter()
            .zip(self.top_k_weights.iter())
            .filter(|(pos, _)| **pos >= recency_start)
            .map(|(_, w)| w)
            .sum();
        recent_mass > threshold
    }
}

/// Configuration for attention weight tracing.
#[derive(Debug, Clone)]
pub struct AttentionTraceConfig {
    /// Number of top positions to record per head
    pub top_k: usize,
    /// Layers to trace (None = all)
    pub layers: Option<Vec<usize>>,
    /// Heads to trace (None = all)
    pub heads: Option<Vec<usize>>,
    /// Minimum weight to consider (positions with weight below this are ignored)
    pub weight_threshold: f32,
}

impl Default for AttentionTraceConfig {
    fn default() -> Self {
        Self { top_k: 10, layers: None, heads: None, weight_threshold: 0.01 }
    }
}

impl AttentionTraceConfig {
    /// Check if a layer should be traced.
    pub fn should_trace_layer(&self, layer_idx: usize) -> bool {
        self.layers.as_ref().is_none_or(|layers| layers.contains(&layer_idx))
    }

    /// Check if a head should be traced.
    pub fn should_trace_head(&self, head_idx: usize) -> bool {
        self.heads.as_ref().is_none_or(|heads| heads.contains(&head_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_weight_trace_from_weights() {
        let weights = vec![0.1, 0.3, 0.4, 0.2];
        let trace = AttentionWeightTrace::from_weights(0, 0, 3, &weights, 2);

        assert_eq!(trace.layer_idx, 0);
        assert_eq!(trace.head_idx, 0);
        assert_eq!(trace.query_pos, 3);
        assert_eq!(trace.top_k_positions.len(), 2);
        // Position 2 has highest weight (0.4), then position 1 (0.3)
        assert_eq!(trace.top_k_positions[0], 2);
        assert_eq!(trace.top_k_positions[1], 1);
    }

    #[test]
    fn test_attention_sink_detection() {
        let weights = vec![0.8, 0.1, 0.05, 0.05];
        let trace = AttentionWeightTrace::from_weights(0, 0, 3, &weights, 4);
        assert!(trace.is_attention_sink(0.5));
    }

    #[test]
    fn test_recency_bias_detection() {
        // Position 3 attending mostly to positions 1 and 2
        let weights = vec![0.05, 0.4, 0.5, 0.05];
        let trace = AttentionWeightTrace::from_weights(0, 0, 3, &weights, 4);
        assert!(trace.has_recency_bias(2, 0.5));
    }
}
