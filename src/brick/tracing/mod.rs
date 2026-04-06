//! Model-Level Inference Tracing (Phase 13, E.11)
//!
//! Comprehensive tracing system for transformer model inference:
//! - MLT-01: LayerActivationTrace - anomaly detection per layer
//! - MLT-02: AttentionWeightTrace - attention pattern analysis
//! - MLT-03: LogitEvolutionTrace - token probability evolution
//! - MLT-04: QuantizationErrorTrace - quantization quality metrics
//! - MLT-05: KvCacheStateTrace - KV cache efficiency tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::brick::{ModelTracer, ModelTracerConfig};
//!
//! let config = ModelTracerConfig::lightweight();
//! let mut tracer = ModelTracer::new(config);
//!
//! tracer.begin_forward(position);
//! // ... forward pass with trace hooks ...
//! if let Some(anomaly) = tracer.end_forward() {
//!     log::warn!("Anomaly: {}", anomaly);
//! }
//! ```

mod activation;
mod attention;
mod kv_cache;
mod logit;
mod quant_error;
mod quant_type;
mod tracer;

pub use activation::{LayerActivationTrace, ModelActivationTrace, TensorStats};
pub use attention::{AttentionTraceConfig, AttentionWeightTrace};
pub use kv_cache::{KvCacheSessionTrace, KvCacheStateTrace};
pub use logit::{LogitEvolutionTrace, TokenLogitEvolution};
pub use quant_error::{ModelQuantizationError, QuantizationErrorTrace};
pub use quant_type::QuantType;
pub use tracer::{ModelTracer, ModelTracerConfig, ModelTracerSummary};
