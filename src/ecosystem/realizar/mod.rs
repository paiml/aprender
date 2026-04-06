//! Realizar GGUF Export Integration (ENT-032)
//!
//! Provides GGUF model export with quantization support via Realizar.
//! Includes experiment provenance tracking in model metadata.

mod error;
mod exporter;
mod metadata;
mod provenance;
mod quantization;

#[cfg(test)]
mod tests;

// Re-export all public types for API compatibility
pub use error::GgufExportError;
pub use exporter::{GgufExportResult, GgufExporter};
pub use metadata::{GeneralMetadata, GgufMetadata};
pub use provenance::ExperimentProvenance;
pub use quantization::QuantizationType;
