// ============================================================================
// E.11.2: LayerActivationTrace (MLT-01)
// ============================================================================

/// Statistics for a tensor without storing the tensor itself.
///
/// Computes min, max, mean, std, L2 norm, NaN/Inf counts in a single pass.
/// Used for anomaly detection (explosion, vanishing gradients, NaN propagation).
///
/// # Example
/// ```rust,ignore
/// let stats = TensorStats::from_slice(&tensor_data);
/// if stats.has_anomaly() {
///     log::warn!("Anomaly detected: {}", stats.anomaly_description());
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TensorStats {
    /// Number of elements analyzed
    pub count: usize,
    /// Minimum value (ignoring NaN/Inf)
    pub min: f32,
    /// Maximum value (ignoring NaN/Inf)
    pub max: f32,
    /// Mean value (ignoring NaN/Inf)
    pub mean: f32,
    /// Standard deviation (ignoring NaN/Inf)
    pub std: f32,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
    /// L2 norm (sqrt of sum of squares)
    pub l2_norm: f32,
}

impl TensorStats {
    /// Compute statistics from a slice in a single pass.
    ///
    /// Uses Welford's algorithm for numerically stable mean/variance.
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let mut count = 0usize;
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum_sq = 0.0f64;

        // Welford's algorithm for online mean/variance
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;

        for &val in data {
            if val.is_nan() {
                nan_count += 1;
                continue;
            }
            if val.is_infinite() {
                inf_count += 1;
                continue;
            }

            count += 1;
            min = min.min(val);
            max = max.max(val);
            sum_sq += (val as f64) * (val as f64);

            // Welford's update
            let delta = val as f64 - mean;
            mean += delta / count as f64;
            let delta2 = val as f64 - mean;
            m2 += delta * delta2;
        }

        let std = if count > 1 { (m2 / (count - 1) as f64).sqrt() as f32 } else { 0.0 };

        let l2_norm = sum_sq.sqrt() as f32;

        Self {
            count: data.len(),
            min: if count > 0 { min } else { 0.0 },
            max: if count > 0 { max } else { 0.0 },
            mean: mean as f32,
            std,
            nan_count,
            inf_count,
            l2_norm,
        }
    }

    /// Check if this tensor has any anomalies.
    ///
    /// Anomaly detection rules (from E.11.2):
    /// - NaN detected: `nan_count > 0`
    /// - Explosion: `max.abs() > 1e6` or `std > 1e4`
    /// - Vanishing: `std < 1e-6` (should check after first few layers)
    pub fn has_anomaly(&self) -> bool {
        self.nan_count > 0
            || self.inf_count > 0
            || self.max.abs() > 1e6
            || self.min.abs() > 1e6
            || self.std > 1e4
    }

    /// Check if values are vanishing (for layers past warmup).
    pub fn is_vanishing(&self) -> bool {
        self.std < 1e-6 && self.count > 0
    }

    /// Get a description of any anomaly detected.
    pub fn anomaly_description(&self) -> Option<String> {
        if self.nan_count > 0 {
            return Some(format!("NaN detected: {} values", self.nan_count));
        }
        if self.inf_count > 0 {
            return Some(format!("Inf detected: {} values", self.inf_count));
        }
        if self.max.abs() > 1e6 || self.min.abs() > 1e6 {
            return Some(format!("Explosion: min={:.2e}, max={:.2e}", self.min, self.max));
        }
        if self.std > 1e4 {
            return Some(format!("High variance: std={:.2e}", self.std));
        }
        None
    }
}

/// Activation trace for a single transformer layer.
///
/// Records tensor statistics at each stage of a transformer layer:
/// input -> norm -> attention -> residual -> ffn -> output
#[derive(Debug, Clone, Default)]
pub struct LayerActivationTrace {
    /// Layer index (0-indexed)
    pub layer_idx: usize,
    /// Input hidden state statistics
    pub input_stats: TensorStats,
    /// After RMSNorm/LayerNorm statistics
    pub post_norm_stats: TensorStats,
    /// After attention statistics
    pub post_attn_stats: TensorStats,
    /// After FFN statistics
    pub post_ffn_stats: TensorStats,
    /// Output hidden state statistics
    pub output_stats: TensorStats,
    /// Residual connection magnitude ratio (output_norm / (output_norm + attn_norm))
    pub residual_ratio: f32,
}

impl LayerActivationTrace {
    /// Create a new layer activation trace.
    pub fn new(layer_idx: usize) -> Self {
        Self { layer_idx, ..Default::default() }
    }

    /// Check if this layer has any anomalies.
    pub fn has_anomaly(&self) -> bool {
        self.input_stats.has_anomaly()
            || self.post_norm_stats.has_anomaly()
            || self.post_attn_stats.has_anomaly()
            || self.post_ffn_stats.has_anomaly()
            || self.output_stats.has_anomaly()
            || self.residual_ratio > 0.99 // Skip connection bypass
    }

    /// Get anomaly description for this layer.
    pub fn anomaly_description(&self) -> Option<String> {
        if let Some(desc) = self.input_stats.anomaly_description() {
            return Some(format!("Layer {} input: {}", self.layer_idx, desc));
        }
        if let Some(desc) = self.post_norm_stats.anomaly_description() {
            return Some(format!("Layer {} post_norm: {}", self.layer_idx, desc));
        }
        if let Some(desc) = self.post_attn_stats.anomaly_description() {
            return Some(format!("Layer {} post_attn: {}", self.layer_idx, desc));
        }
        if let Some(desc) = self.post_ffn_stats.anomaly_description() {
            return Some(format!("Layer {} post_ffn: {}", self.layer_idx, desc));
        }
        if let Some(desc) = self.output_stats.anomaly_description() {
            return Some(format!("Layer {} output: {}", self.layer_idx, desc));
        }
        if self.residual_ratio > 0.99 {
            return Some(format!(
                "Layer {} residual dominance: ratio={:.4}",
                self.layer_idx, self.residual_ratio
            ));
        }
        None
    }
}

/// Full model activation trace for one forward pass.
#[derive(Debug, Clone, Default)]
pub struct ModelActivationTrace {
    /// Per-layer activation traces
    pub layers: Vec<LayerActivationTrace>,
    /// Embedding output statistics
    pub embedding_stats: TensorStats,
    /// Final logits statistics
    pub logits_stats: TensorStats,
    /// Whether any anomaly was detected
    pub has_anomaly: bool,
    /// Description of first anomaly found
    pub anomaly_desc: Option<String>,
}

impl ModelActivationTrace {
    /// Create a new model activation trace with expected layer count.
    pub fn with_capacity(num_layers: usize) -> Self {
        Self { layers: Vec::with_capacity(num_layers), ..Default::default() }
    }

    /// Add a layer trace.
    pub fn add_layer(&mut self, trace: LayerActivationTrace) {
        if !self.has_anomaly {
            if let Some(desc) = trace.anomaly_description() {
                self.has_anomaly = true;
                self.anomaly_desc = Some(desc);
            }
        }
        self.layers.push(trace);
    }

    /// Finalize the trace and check embedding/logits.
    pub fn finalize(&mut self) {
        if !self.has_anomaly {
            if let Some(desc) = self.embedding_stats.anomaly_description() {
                self.has_anomaly = true;
                self.anomaly_desc = Some(format!("Embedding: {}", desc));
            }
        }
        if !self.has_anomaly {
            if let Some(desc) = self.logits_stats.anomaly_description() {
                self.has_anomaly = true;
                self.anomaly_desc = Some(format!("Logits: {}", desc));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // TensorStats Tests
    // ========================================================================

    #[test]
    fn test_tensor_stats_empty() {
        let stats = TensorStats::from_slice(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.nan_count, 0);
        assert!(!stats.has_anomaly());
    }

    #[test]
    fn test_tensor_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::from_slice(&data);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!(!stats.has_anomaly());
    }

    #[test]
    fn test_tensor_stats_nan_detection() {
        let data = vec![1.0, f32::NAN, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert_eq!(stats.nan_count, 1);
        assert!(stats.has_anomaly());
        assert!(stats.anomaly_description().unwrap().contains("NaN"));
    }

    #[test]
    fn test_tensor_stats_inf_detection() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert_eq!(stats.inf_count, 1);
        assert!(stats.has_anomaly());
    }

    #[test]
    fn test_tensor_stats_explosion() {
        let data = vec![1e7, 2e7];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.has_anomaly());
        assert!(stats.anomaly_description().unwrap().contains("Explosion"));
    }

    #[test]
    fn test_tensor_stats_vanishing() {
        let data = vec![1e-8, 1e-8, 1e-8];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.is_vanishing());
    }

    // ========================================================================
    // LayerActivationTrace Tests
    // ========================================================================

    #[test]
    fn test_layer_activation_trace_new() {
        let trace = LayerActivationTrace::new(5);
        assert_eq!(trace.layer_idx, 5);
        assert!(!trace.has_anomaly());
    }

    #[test]
    fn test_layer_activation_trace_anomaly() {
        let mut trace = LayerActivationTrace::new(0);
        trace.input_stats = TensorStats::from_slice(&[f32::NAN]);
        assert!(trace.has_anomaly());
        assert!(trace.anomaly_description().is_some());
    }

    #[test]
    fn test_layer_activation_trace_residual_dominance() {
        let mut trace = LayerActivationTrace::new(0);
        trace.residual_ratio = 0.999;
        assert!(trace.has_anomaly());
        assert!(trace.anomaly_description().unwrap().contains("residual"));
    }

    // ========================================================================
    // ModelActivationTrace Tests
    // ========================================================================

    #[test]
    fn test_model_activation_trace_add_layer() {
        let mut trace = ModelActivationTrace::with_capacity(32);
        trace.add_layer(LayerActivationTrace::new(0));
        trace.add_layer(LayerActivationTrace::new(1));
        assert_eq!(trace.layers.len(), 2);
        assert!(!trace.has_anomaly);
    }

    #[test]
    fn test_model_activation_trace_anomaly_propagation() {
        let mut trace = ModelActivationTrace::default();
        let mut bad_layer = LayerActivationTrace::new(0);
        bad_layer.input_stats = TensorStats::from_slice(&[f32::NAN]);
        trace.add_layer(bad_layer);
        assert!(trace.has_anomaly);
    }

    /// FALSIFICATION TEST: TensorStats Welford algorithm numerical stability
    ///
    /// Welford's algorithm must produce correct mean/std even for large values.
    #[test]
    fn test_falsify_tensor_stats_welford_stability() {
        // Test with large offset - naive algorithm would lose precision
        let large_offset = 1e9;
        let data: Vec<f32> = (0..1000).map(|i| large_offset + i as f32).collect();
        let stats = TensorStats::from_slice(&data);

        // Mean should be large_offset + 499.5
        let expected_mean = large_offset + 499.5;
        assert!(
            (stats.mean - expected_mean as f32).abs() < 1.0,
            "FALSIFICATION FAILED: Welford mean {} != expected {} (relative error too high)",
            stats.mean,
            expected_mean
        );

        // Std should be ~288.7 (uniform distribution 0-999)
        assert!(
            stats.std > 280.0 && stats.std < 300.0,
            "FALSIFICATION FAILED: Welford std {} outside expected range [280, 300]",
            stats.std
        );
    }
}
