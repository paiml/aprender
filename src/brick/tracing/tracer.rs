// ============================================================================
// E.11.7: Unified ModelTracer
// ============================================================================

use std::fmt;

use super::activation::{LayerActivationTrace, ModelActivationTrace};
use super::attention::{AttentionTraceConfig, AttentionWeightTrace};
use super::kv_cache::{KvCacheSessionTrace, KvCacheStateTrace};
use super::logit::LogitEvolutionTrace;
use super::quant_error::{ModelQuantizationError, QuantizationErrorTrace};

/// Configuration for model-level tracing.
#[derive(Debug, Clone, Default)]
pub struct ModelTracerConfig {
    /// Enable layer activation tracing (MLT-01)
    pub trace_activations: bool,
    /// Enable attention weight tracing (MLT-02)
    pub trace_attention: bool,
    /// Attention trace configuration
    pub attention_config: AttentionTraceConfig,
    /// Enable logit evolution tracing (MLT-03)
    pub trace_logits: bool,
    /// Specific tokens to track (None = auto-select top-k)
    pub tracked_tokens: Option<Vec<u32>>,
    /// Enable quantization error tracing (MLT-04) - expensive!
    pub trace_quant_error: bool,
    /// Enable KV cache state tracing (MLT-05)
    pub trace_kv_cache: bool,
}

impl ModelTracerConfig {
    /// Create a config that traces everything (for debugging).
    pub fn full() -> Self {
        Self {
            trace_activations: true,
            trace_attention: true,
            attention_config: AttentionTraceConfig::default(),
            trace_logits: true,
            tracked_tokens: None,
            trace_quant_error: true,
            trace_kv_cache: true,
        }
    }

    /// Create a lightweight config (activations + KV cache only).
    pub fn lightweight() -> Self {
        Self { trace_activations: true, trace_kv_cache: true, ..Default::default() }
    }

    /// Check if any tracing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.trace_activations
            || self.trace_attention
            || self.trace_logits
            || self.trace_quant_error
            || self.trace_kv_cache
    }
}

/// Unified model tracer that coordinates all trace types.
///
/// # Example
/// ```rust,ignore
/// let config = ModelTracerConfig::lightweight();
/// let mut tracer = ModelTracer::new(config);
///
/// tracer.begin_forward(position);
/// // ... forward pass with trace hooks ...
/// if let Some(anomaly) = tracer.end_forward() {
///     log::warn!("Anomaly: {}", anomaly);
/// }
/// ```
pub struct ModelTracer {
    config: ModelTracerConfig,
    /// Current forward pass position
    current_position: usize,
    /// Accumulated activation traces
    activation_traces: Vec<ModelActivationTrace>,
    /// Current activation trace (in progress)
    current_activation_trace: Option<ModelActivationTrace>,
    /// Accumulated attention traces
    attention_traces: Vec<AttentionWeightTrace>,
    /// Accumulated logit evolution traces
    logit_traces: Vec<LogitEvolutionTrace>,
    /// Current logit trace (in progress)
    current_logit_trace: Option<LogitEvolutionTrace>,
    /// Accumulated quantization error traces
    quant_traces: Vec<ModelQuantizationError>,
    /// KV cache session trace
    kv_trace: KvCacheSessionTrace,
}

impl ModelTracer {
    /// Create a new tracer with the given configuration.
    pub fn new(config: ModelTracerConfig) -> Self {
        Self {
            config,
            current_position: 0,
            activation_traces: Vec::new(),
            current_activation_trace: None,
            attention_traces: Vec::new(),
            logit_traces: Vec::new(),
            current_logit_trace: None,
            quant_traces: Vec::new(),
            kv_trace: KvCacheSessionTrace::default(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ModelTracerConfig {
        &self.config
    }

    /// Get a reference to the current logit trace (if any).
    pub fn current_logit_trace(&self) -> Option<&LogitEvolutionTrace> {
        self.current_logit_trace.as_ref()
    }

    /// Set the current logit trace (for testing purposes).
    pub fn set_current_logit_trace(&mut self, trace: Option<LogitEvolutionTrace>) {
        self.current_logit_trace = trace;
    }

    /// Begin a forward pass at the given position.
    pub fn begin_forward(&mut self, position: usize) {
        self.current_position = position;

        if self.config.trace_activations {
            self.current_activation_trace = Some(ModelActivationTrace::default());
        }

        if self.config.trace_logits {
            self.current_logit_trace = Some(LogitEvolutionTrace::new(position, 1.0, 1.0));
        }
    }

    /// Record layer activation (called by executor after each layer).
    pub fn record_layer_activation(&mut self, trace: LayerActivationTrace) {
        if let Some(ref mut activation) = self.current_activation_trace {
            activation.add_layer(trace);
        }
    }

    /// Record attention weights (called by attention brick).
    pub fn record_attention(&mut self, trace: AttentionWeightTrace) {
        if self.config.trace_attention {
            self.attention_traces.push(trace);
        }
    }

    /// Record logit state at a layer (called by lm_head or probe).
    pub fn record_logits(&mut self, layer_idx: usize, logits: &[f32]) {
        if let Some(ref mut logit_trace) = self.current_logit_trace {
            for token_evo in &mut logit_trace.tracked_tokens {
                let logit = logits.get(token_evo.token_id as usize).copied().unwrap_or(0.0);
                let rank = LogitEvolutionTrace::compute_rank(logits, token_evo.token_id);
                token_evo.record_layer(logit, rank);
            }
            // Store decisive layer based on rank changes
            logit_trace.decisive_layer = layer_idx;
        }
    }

    /// Record KV cache state (called after each generation step).
    pub fn record_kv_state(&mut self, trace: KvCacheStateTrace) {
        if self.config.trace_kv_cache {
            self.kv_trace.add_step(trace);
        }
    }

    /// Record quantization error for a brick.
    pub fn record_quant_error(&mut self, trace: QuantizationErrorTrace) {
        if self.config.trace_quant_error {
            if self.quant_traces.is_empty() {
                self.quant_traces.push(ModelQuantizationError::default());
            }
            if let Some(model_error) = self.quant_traces.last_mut() {
                model_error.add_error(trace);
            }
        }
    }

    /// Complete forward pass and check for anomalies.
    ///
    /// Returns a description of the first anomaly detected, if any.
    pub fn end_forward(&mut self) -> Option<String> {
        let mut anomaly = None;

        // Finalize activation trace
        if let Some(mut trace) = self.current_activation_trace.take() {
            trace.finalize();
            if trace.has_anomaly {
                anomaly = trace.anomaly_desc.clone();
            }
            self.activation_traces.push(trace);
        }

        // Finalize logit trace
        if let Some(trace) = self.current_logit_trace.take() {
            self.logit_traces.push(trace);
        }

        anomaly
    }

    /// Get summary statistics.
    pub fn summary(&self) -> ModelTracerSummary {
        ModelTracerSummary {
            total_forwards: self.activation_traces.len(),
            anomalies_detected: self.activation_traces.iter().filter(|t| t.has_anomaly).count(),
            attention_traces: self.attention_traces.len(),
            logit_traces: self.logit_traces.len(),
            kv_steps: self.kv_trace.steps.len(),
            total_evictions: self.kv_trace.total_evictions,
            avg_hit_rate: self.kv_trace.avg_hit_rate,
            quant_warnings: self.quant_traces.iter().map(|t| t.warning_count()).sum(),
            quant_criticals: self.quant_traces.iter().map(|t| t.critical_count()).sum(),
        }
    }

    /// Export summary as JSON for artifact validation.
    pub fn summary_to_json(&self) -> String {
        let summary = self.summary();
        format!(
            r#"{{"total_forwards":{},"anomalies_detected":{},"attention_traces":{},"logit_traces":{},"kv_steps":{},"total_evictions":{},"avg_hit_rate":{:.4},"quant_warnings":{},"quant_criticals":{}}}"#,
            summary.total_forwards,
            summary.anomalies_detected,
            summary.attention_traces,
            summary.logit_traces,
            summary.kv_steps,
            summary.total_evictions,
            summary.avg_hit_rate,
            summary.quant_warnings,
            summary.quant_criticals
        )
    }

    /// Clear all accumulated traces (free memory).
    pub fn clear(&mut self) {
        self.activation_traces.clear();
        self.attention_traces.clear();
        self.logit_traces.clear();
        self.quant_traces.clear();
        self.kv_trace = KvCacheSessionTrace::default();
    }
}

/// Summary of model tracer state.
#[derive(Debug, Clone, Default)]
pub struct ModelTracerSummary {
    /// Total forward passes traced
    pub total_forwards: usize,
    /// Number of forward passes with anomalies
    pub anomalies_detected: usize,
    /// Total attention traces collected
    pub attention_traces: usize,
    /// Total logit evolution traces
    pub logit_traces: usize,
    /// Total KV cache steps traced
    pub kv_steps: usize,
    /// Total KV cache evictions
    pub total_evictions: usize,
    /// Average KV cache hit rate
    pub avg_hit_rate: f32,
    /// Quantization warning count
    pub quant_warnings: usize,
    /// Quantization critical count
    pub quant_criticals: usize,
}

impl fmt::Display for ModelTracerSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ModelTracer Summary:")?;
        writeln!(f, "  Forward passes: {}", self.total_forwards)?;
        writeln!(f, "  Anomalies: {}", self.anomalies_detected)?;
        writeln!(f, "  Attention traces: {}", self.attention_traces)?;
        writeln!(f, "  Logit traces: {}", self.logit_traces)?;
        writeln!(f, "  KV cache steps: {}", self.kv_steps)?;
        writeln!(f, "  KV evictions: {}", self.total_evictions)?;
        writeln!(f, "  Avg hit rate: {:.2}%", self.avg_hit_rate * 100.0)?;
        writeln!(f, "  Quant warnings: {}", self.quant_warnings)?;
        write!(f, "  Quant criticals: {}", self.quant_criticals)
    }
}

#[cfg(test)]
mod tests {
    use super::super::activation::TensorStats;
    use super::*;

    #[test]
    fn test_model_tracer_lightweight() {
        let config = ModelTracerConfig::lightweight();
        assert!(config.trace_activations);
        assert!(config.trace_kv_cache);
        assert!(!config.trace_attention);
        assert!(!config.trace_quant_error);
    }

    #[test]
    fn test_model_tracer_full() {
        let config = ModelTracerConfig::full();
        assert!(config.trace_activations);
        assert!(config.trace_attention);
        assert!(config.trace_logits);
        assert!(config.trace_quant_error);
        assert!(config.trace_kv_cache);
    }

    #[test]
    fn test_model_tracer_forward_pass() {
        let config = ModelTracerConfig::lightweight();
        let mut tracer = ModelTracer::new(config);

        tracer.begin_forward(0);
        tracer.record_layer_activation(LayerActivationTrace::new(0));
        tracer.record_layer_activation(LayerActivationTrace::new(1));
        let anomaly = tracer.end_forward();

        assert!(anomaly.is_none());
        let summary = tracer.summary();
        assert_eq!(summary.total_forwards, 1);
        assert_eq!(summary.anomalies_detected, 0);
    }

    #[test]
    fn test_model_tracer_detects_anomaly() {
        let config = ModelTracerConfig::lightweight();
        let mut tracer = ModelTracer::new(config);

        tracer.begin_forward(0);
        let mut bad_layer = LayerActivationTrace::new(0);
        bad_layer.input_stats = TensorStats::from_slice(&[f32::NAN]);
        tracer.record_layer_activation(bad_layer);
        let anomaly = tracer.end_forward();

        assert!(anomaly.is_some());
        assert!(anomaly.unwrap().contains("NaN"));
        assert_eq!(tracer.summary().anomalies_detected, 1);
    }

    #[test]
    fn test_model_tracer_json_output() {
        let config = ModelTracerConfig::lightweight();
        let mut tracer = ModelTracer::new(config);

        tracer.begin_forward(0);
        tracer.end_forward();

        let json = tracer.summary_to_json();
        assert!(json.contains("\"total_forwards\":1"));
        assert!(json.contains("\"anomalies_detected\":0"));
    }

    /// FALSIFICATION TEST: ModelTracer layer count must match recorded layers
    #[test]
    fn test_falsify_tracer_layer_count() {
        let config = ModelTracerConfig::lightweight();
        let mut tracer = ModelTracer::new(config);

        tracer.begin_forward(0);
        let num_layers = 32;
        for i in 0..num_layers {
            tracer.record_layer_activation(LayerActivationTrace::new(i));
        }
        tracer.end_forward();

        // The activation trace should have exactly num_layers entries
        assert_eq!(
            tracer.activation_traces[0].layers.len(),
            num_layers,
            "FALSIFICATION FAILED: recorded {} layers but expected {}",
            tracer.activation_traces[0].layers.len(),
            num_layers
        );
    }
}
