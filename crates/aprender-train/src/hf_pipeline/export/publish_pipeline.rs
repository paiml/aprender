//! Quantize-Export-Publish pipeline (feature-gated: hub-publish)
//!
//! Combines quantization, GGUF export, and HuggingFace Hub publishing
//! into a single pipeline operation.

use crate::hf_pipeline::error::FetchError;
use crate::hf_pipeline::export::gguf_writer::GgufQuantization;
use crate::hf_pipeline::export::pipeline::{quantize_and_export, QuantExportResult};
use crate::hf_pipeline::export::weights::ModelWeights;
use crate::hf_pipeline::publish::config::PublishConfig;
use crate::hf_pipeline::publish::publisher::HfPublisher;
use crate::hf_pipeline::publish::result::{PublishError, PublishResult};
use std::path::Path;

/// Result of the full quantize-export-publish pipeline
#[derive(Debug, Clone)]
pub struct QuantPublishResult {
    /// Quantize-export result
    pub export: QuantExportResult,
    /// Publish result from HuggingFace Hub
    pub publish: PublishResult,
}

/// Quantize, export to GGUF, and publish to HuggingFace Hub
///
/// Full pipeline:
/// 1. Quantize weights (Q4_0, Q8_0, or unquantized)
/// 2. Export as GGUF with metadata
/// 3. Generate model card/README
/// 4. Publish to HuggingFace Hub
pub fn quantize_export_publish(
    weights: &ModelWeights,
    quantization: GgufQuantization,
    publish_config: PublishConfig,
    output_dir: impl AsRef<Path>,
) -> std::result::Result<QuantPublishResult, QuantPublishError> {
    let output_dir = output_dir.as_ref();
    let filename = "model.gguf";

    // Step 1+2: Quantize and export
    let export_result = quantize_and_export(weights, quantization, output_dir, filename)
        .map_err(QuantPublishError::Export)?;

    // Step 3: Create publisher and publish
    let publisher = HfPublisher::new(publish_config).map_err(QuantPublishError::Publish)?;

    let gguf_path = output_dir.join(filename);
    let files: Vec<(&Path, &str)> = vec![(&gguf_path, filename)];

    // Publish files (no ModelCard — we upload the README separately)
    let mut publish_result = publisher.publish(&files, None).map_err(QuantPublishError::Publish)?;

    // Upload generated README if available
    if let Some(readme) = &export_result.readme {
        publisher
            .upload_bytes(readme.as_bytes(), "README.md")
            .map_err(QuantPublishError::Publish)?;
        publish_result.model_card_generated = true;
    }

    Ok(QuantPublishResult { export: export_result, publish: publish_result })
}

/// Errors from the quantize-export-publish pipeline
#[derive(Debug, thiserror::Error)]
pub enum QuantPublishError {
    /// Export phase failed
    #[error("Export failed: {0}")]
    Export(FetchError),

    /// Publish phase failed
    #[error("Publish failed: {0}")]
    Publish(PublishError),
}
