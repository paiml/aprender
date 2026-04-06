//! Performance Metrics Breakdown
//!
//! LCP-04 pattern from llama.cpp for tracking inference timing.

// ----------------------------------------------------------------------------
// LCP-04: Performance Metrics Breakdown (llama.cpp pattern)
// ----------------------------------------------------------------------------

/// Performance metrics breakdown for inference phases.
///
/// Tracks timing for each phase of LLM inference:
/// - Model loading (t_load_ms)
/// - Prompt evaluation / prefill (t_p_eval_ms)
/// - Token generation / decode (t_eval_ms)
///
/// # Example
/// ```rust,ignore
/// use trueno::brick::PerfMetrics;
///
/// let mut metrics = PerfMetrics::default();
/// metrics.record_load(1500);  // 1.5s model load
/// metrics.record_prefill(200, 512);  // 200ms for 512 prompt tokens
/// metrics.record_decode(50);  // 50ms per generated token
///
/// println!("{}", metrics.summary());
/// ```
#[derive(Debug, Clone, Default)]
pub struct PerfMetrics {
    /// Model loading time (milliseconds)
    pub t_load_ms: u64,
    /// Prompt evaluation time - prefill phase (milliseconds)
    pub t_p_eval_ms: u64,
    /// Token generation time - decode phase (milliseconds)
    pub t_eval_ms: u64,
    /// Number of tokens in prompt (prefill)
    pub n_p_eval: u32,
    /// Number of tokens generated (decode)
    pub n_eval: u32,
    /// Sample count for t_eval (for averaging)
    pub n_samples: u32,
}

impl PerfMetrics {
    /// Create new metrics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record model loading time.
    pub fn record_load(&mut self, ms: u64) {
        self.t_load_ms = ms;
    }

    /// Record prefill (prompt evaluation) time.
    pub fn record_prefill(&mut self, ms: u64, tokens: u32) {
        self.t_p_eval_ms = ms;
        self.n_p_eval = tokens;
    }

    /// Record a single decode step.
    pub fn record_decode(&mut self, ms: u64) {
        self.t_eval_ms += ms;
        self.n_eval += 1;
        self.n_samples += 1;
    }

    /// Record batch decode step.
    pub fn record_decode_batch(&mut self, ms: u64, tokens: u32) {
        self.t_eval_ms += ms;
        self.n_eval += tokens;
        self.n_samples += 1;
    }

    /// Tokens per second during generation (decode throughput).
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.t_eval_ms == 0 {
            0.0
        } else {
            1000.0 * self.n_eval as f64 / self.t_eval_ms as f64
        }
    }

    /// Tokens per second during prompt evaluation (prefill throughput).
    #[must_use]
    pub fn prefill_tokens_per_second(&self) -> f64 {
        if self.t_p_eval_ms == 0 {
            0.0
        } else {
            1000.0 * self.n_p_eval as f64 / self.t_p_eval_ms as f64
        }
    }

    /// Total time for complete inference.
    #[must_use]
    pub fn total_ms(&self) -> u64 {
        self.t_load_ms + self.t_p_eval_ms + self.t_eval_ms
    }

    /// Time-to-first-token (TTFT).
    #[must_use]
    pub fn time_to_first_token_ms(&self) -> u64 {
        self.t_load_ms + self.t_p_eval_ms
    }

    /// Average time per token during decode.
    #[must_use]
    pub fn avg_token_latency_ms(&self) -> f64 {
        if self.n_eval == 0 {
            0.0
        } else {
            self.t_eval_ms as f64 / self.n_eval as f64
        }
    }

    /// Formatted summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "load: {}ms, prefill: {}ms ({:.1} tok/s, {} tokens), decode: {}ms ({:.1} tok/s, {} tokens), total: {}ms",
            self.t_load_ms,
            self.t_p_eval_ms,
            self.prefill_tokens_per_second(),
            self.n_p_eval,
            self.t_eval_ms,
            self.tokens_per_second(),
            self.n_eval,
            self.total_ms()
        )
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ----------------------------------------------------------------------------
// LCP-01: Inference Phase (for Arena Allocation)
// ----------------------------------------------------------------------------

/// Inference phase for dual-arena allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferencePhase {
    /// Processing prompt, large batches
    #[default]
    Prefill,
    /// Generating tokens, small batches
    Decode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_metrics_default() {
        let metrics = PerfMetrics::default();
        assert_eq!(metrics.t_load_ms, 0);
        assert_eq!(metrics.t_p_eval_ms, 0);
        assert_eq!(metrics.t_eval_ms, 0);
        assert_eq!(metrics.n_p_eval, 0);
        assert_eq!(metrics.n_eval, 0);
    }

    #[test]
    fn test_perf_metrics_record_load() {
        let mut metrics = PerfMetrics::new();
        metrics.record_load(1500);
        assert_eq!(metrics.t_load_ms, 1500);
    }

    #[test]
    fn test_perf_metrics_record_prefill() {
        let mut metrics = PerfMetrics::new();
        metrics.record_prefill(200, 512);
        assert_eq!(metrics.t_p_eval_ms, 200);
        assert_eq!(metrics.n_p_eval, 512);
    }

    #[test]
    fn test_perf_metrics_record_decode() {
        let mut metrics = PerfMetrics::new();
        metrics.record_decode(50);
        metrics.record_decode(50);
        assert_eq!(metrics.t_eval_ms, 100);
        assert_eq!(metrics.n_eval, 2);
        assert_eq!(metrics.n_samples, 2);
    }

    #[test]
    fn test_perf_metrics_tokens_per_second() {
        let mut metrics = PerfMetrics::new();
        metrics.record_decode_batch(1000, 100); // 100 tokens in 1 second
        assert!((metrics.tokens_per_second() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_perf_metrics_prefill_throughput() {
        let mut metrics = PerfMetrics::new();
        metrics.record_prefill(500, 1000); // 1000 tokens in 500ms = 2000 tok/s
        assert!((metrics.prefill_tokens_per_second() - 2000.0).abs() < 0.001);
    }

    #[test]
    fn test_perf_metrics_total_ms() {
        let mut metrics = PerfMetrics::new();
        metrics.record_load(1000);
        metrics.record_prefill(200, 512);
        metrics.record_decode_batch(300, 100);
        assert_eq!(metrics.total_ms(), 1500);
    }

    #[test]
    fn test_perf_metrics_time_to_first_token() {
        let mut metrics = PerfMetrics::new();
        metrics.record_load(1000);
        metrics.record_prefill(200, 512);
        assert_eq!(metrics.time_to_first_token_ms(), 1200);
    }

    #[test]
    fn test_perf_metrics_reset() {
        let mut metrics = PerfMetrics::new();
        metrics.record_load(1500);
        metrics.record_prefill(200, 512);
        metrics.reset();
        assert_eq!(metrics.t_load_ms, 0);
        assert_eq!(metrics.n_p_eval, 0);
    }

    #[test]
    fn test_inference_phase_default() {
        let phase = InferencePhase::default();
        assert_eq!(phase, InferencePhase::Prefill);
    }

    #[test]
    fn test_inference_phase_eq() {
        assert_eq!(InferencePhase::Prefill, InferencePhase::Prefill);
        assert_eq!(InferencePhase::Decode, InferencePhase::Decode);
        assert_ne!(InferencePhase::Prefill, InferencePhase::Decode);
    }
}
