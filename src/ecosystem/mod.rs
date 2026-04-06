//! Ecosystem Integration (Phase 9)
//!
//! This module provides integrations with other PAIML stack components:
//! - **Batuta**: GPU pricing and queue management
//! - **Realizar**: GGUF model export with quantization
//! - **Ruchy**: Session bridge for preserving training history
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Automatic fallback when services are unavailable
//! - **Just-in-Time**: Queue-aware ETA adjustments
//! - **Kaizen**: Provenance tracking for continuous improvement

pub mod batuta;
pub mod realizar;
#[cfg(feature = "ruchy-sessions")]
pub mod ruchy;

pub use batuta::{adjust_eta, BatutaClient, BatutaError, FallbackPricing, GpuPricing, QueueState};
pub use realizar::{
    ExperimentProvenance, GeneralMetadata, GgufExportError, GgufExportResult, GgufExporter,
    GgufMetadata, QuantizationType,
};
#[cfg(feature = "ruchy-sessions")]
pub use ruchy::{session_to_artifact, EntrenarSession, RuchyBridgeError, SessionMetrics};
