//! Model Export Formats
//!
//! Supports exporting trained/distilled models to various formats:
//! - SafeTensors: Safe, fast tensor serialization
//! - APR: Aprender format for entrenar ecosystem
//! - GGUF: Quantized format for llama.cpp compatibility
//!
//! # Example
//!
//! ```ignore
//! use entrenar::hf_pipeline::export::{Exporter, ExportFormat};
//!
//! let exporter = Exporter::new();
//! exporter.export(&model_weights, ExportFormat::SafeTensors, "output/model.safetensors")?;
//! ```

mod exporter;
mod format;
pub(crate) mod gguf_verify;
pub(crate) mod gguf_writer;
pub(crate) mod pipeline;
#[cfg(feature = "hub-publish")]
pub(crate) mod publish_pipeline;
mod result;
mod types;
mod weights;

#[cfg(test)]
mod tests;

// Public re-exports
pub use exporter::Exporter;
pub use format::ExportFormat;
pub use gguf_verify::{verify_gguf, GgufSummary, GgufTensorInfo};
pub use gguf_writer::GgufQuantization;
pub use pipeline::{quantize_and_export, QuantExportResult};
#[cfg(feature = "hub-publish")]
pub use publish_pipeline::{quantize_export_publish, QuantPublishError, QuantPublishResult};
pub use result::ExportResult;
#[allow(unused_imports)]
pub use weights::TrainingMetadata;
pub use weights::{ModelMetadata, ModelWeights};
