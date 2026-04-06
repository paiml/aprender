//! Real-Time Audit Log & Explainability for APR Format Models
//!
//! # Toyota Way: 現地現物 (Genchi Genbutsu)
//! Every decision is traceable to ground truth. All predictions can be explained.
//!
//! # Architecture
//!
//! - **DecisionPath**: Model-specific explanation of how a decision was made
//! - **DecisionTrace**: Complete record of a single prediction
//! - **Explainable**: Trait for models that can explain their predictions
//! - **TraceCollector**: Strategy for collecting decision traces
//!
//! # Collectors
//!
//! - **RingCollector**: Stack-allocated, <100ns, for games/drones
//! - **StreamCollector**: Write-through, <1µs, for persistent logging
//! - **HashChainCollector**: SHA-256 chain, <10µs, for safety-critical
//!
//! # Example
//!
//! ```ignore
//! use entrenar::monitor::inference::{
//!     InferenceMonitor, RingCollector, LinearPath,
//! };
//!
//! let model = LinearRegressor::load("model.apr")?;
//! let collector = RingCollector::<LinearPath, 64>::new();
//! let mut monitor = InferenceMonitor::new(model, collector);
//!
//! let prediction = monitor.predict(&input);
//! let traces = monitor.traces();
//! ```

pub mod collector;
pub mod counterfactual;
pub mod path;
pub mod provenance;
pub mod safety_andon;
pub mod serialization;
pub mod trace;

#[cfg(test)]
mod tests;

// Re-exports
pub use collector::{
    ChainEntry, ChainVerification, HashChainCollector, RingCollector, StreamCollector,
    StreamFormat, TraceCollector,
};
pub use counterfactual::{Counterfactual, FeatureChange};
pub use path::{
    DecisionPath, ForestPath, KNNPath, LeafInfo, LinearPath, NeuralPath, PathError, TreePath,
    TreeSplit,
};
pub use provenance::{
    Anomaly, AttackPath, CausalRelation, IncidentReconstructor, NodeId, ProvenanceEdge,
    ProvenanceGraph, ProvenanceNode,
};
pub use safety_andon::{EmergencyCondition, SafetyAndon, SafetyIntegrityLevel};
pub use serialization::{
    PathType, SerializationError, TraceFormat, TraceSerializer, APRT_MAGIC, APRT_VERSION,
};
pub use trace::DecisionTrace;

use std::time::Instant;

/// Monotonic nanosecond timestamp
#[inline]
pub fn monotonic_ns() -> u64 {
    // Use Instant for monotonic clock
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// FNV-1a hash for input features
#[inline]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash a slice of f32 values
#[inline]
pub fn hash_features(features: &[f32]) -> u64 {
    let bytes: &[u8] = bytemuck::cast_slice(features);
    fnv1a_hash(bytes)
}

// GH-305: Explainable trait now lives in aprender (source of truth).
// Re-exported here for backwards compatibility.
pub use aprender::explainable::path::Explainable;

/// High-level inference monitor
///
/// Wraps a model and collector to automatically trace all predictions.
pub struct InferenceMonitor<M, C>
where
    M: Explainable,
    C: TraceCollector<M::Path>,
{
    model: M,
    collector: C,
    andon: Option<SafetyAndon>,
    latency_budget_ns: u64,
    sequence: u64,
}

impl<M, C> InferenceMonitor<M, C>
where
    M: Explainable,
    C: TraceCollector<M::Path>,
{
    /// Create a new inference monitor
    pub fn new(model: M, collector: C) -> Self {
        Self {
            model,
            collector,
            andon: None,
            latency_budget_ns: 10_000_000, // 10ms default
            sequence: 0,
        }
    }

    /// Set the Andon system for alerting
    pub fn with_andon(mut self, andon: SafetyAndon) -> Self {
        self.andon = Some(andon);
        self
    }

    /// Set the latency budget in nanoseconds
    pub fn with_latency_budget_ns(mut self, budget: u64) -> Self {
        self.latency_budget_ns = budget;
        self
    }

    /// Predict with automatic tracing
    pub fn predict(&mut self, x: &[f32], n_samples: usize) -> Vec<f32> {
        let start = Instant::now();
        let timestamp_ns = monotonic_ns();

        let (outputs, paths) = self.model.predict_explained(x, n_samples);

        let elapsed_ns = start.elapsed().as_nanos() as u64;

        let features_per_sample = x.len() / n_samples;

        for (i, (output, path)) in outputs.iter().zip(paths.into_iter()).enumerate() {
            let sample_start = i * features_per_sample;
            let sample_end = sample_start + features_per_sample;
            let sample_features = &x[sample_start..sample_end];

            let trace = DecisionTrace {
                timestamp_ns,
                sequence: self.sequence,
                input_hash: hash_features(sample_features),
                path,
                output: *output,
                latency_ns: elapsed_ns,
            };

            self.sequence += 1;

            // Andon checks
            if let Some(andon) = &mut self.andon {
                andon.check_trace(&trace, self.latency_budget_ns);
            }

            self.collector.record(trace);
        }

        outputs
    }

    /// Get reference to the collector
    pub fn collector(&self) -> &C {
        &self.collector
    }

    /// Get mutable reference to the collector
    pub fn collector_mut(&mut self) -> &mut C {
        &mut self.collector
    }

    /// Get reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get the current sequence number
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}
