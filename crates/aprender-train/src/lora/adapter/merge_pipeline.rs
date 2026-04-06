//! Merge-Export-Publish pipeline (feature-gated: hub-publish)
//!
//! Merges LoRA/QLoRA adapters into base weights, exports, and publishes
//! to HuggingFace Hub.

use super::error::AdapterError;
use super::merge_export::{merge_and_collect, merge_qlora_and_collect};
use crate::hf_pipeline::publish::config::PublishConfig;
use crate::hf_pipeline::publish::publisher::HfPublisher;
use crate::hf_pipeline::publish::result::{PublishError, PublishResult};
use crate::lora::{LoRALayer, QLoRALayer};
use std::path::Path;

/// Result of merge-export-publish pipeline
#[derive(Debug, Clone)]
pub struct MergePublishResult {
    /// Number of layers merged
    pub layers_merged: usize,
    /// Publish result
    pub publish: PublishResult,
}

/// Merge LoRA adapters, export as SafeTensors, and publish to HuggingFace Hub
pub fn merge_export_publish(
    layers: &[(&str, &LoRALayer)],
    publish_config: PublishConfig,
    output_dir: impl AsRef<Path>,
) -> Result<MergePublishResult, MergePublishError> {
    let output_dir = output_dir.as_ref();
    let filename = "model.safetensors";

    // Step 1: Merge
    let merged = merge_and_collect(layers);
    let layers_merged = merged.layers_merged;

    // Step 2: Export as SafeTensors
    let export_path = output_dir.join(filename);
    std::fs::create_dir_all(output_dir)
        .map_err(|e| MergePublishError::Merge(AdapterError::Io(e)))?;
    merged.save_safetensors(&export_path).map_err(MergePublishError::Merge)?;

    // Step 3: Publish
    let publisher = HfPublisher::new(publish_config).map_err(MergePublishError::Publish)?;
    let files: Vec<(&Path, &str)> = vec![(&export_path, filename)];

    let publish = publisher.publish(&files, None).map_err(MergePublishError::Publish)?;

    Ok(MergePublishResult { layers_merged, publish })
}

/// Merge QLoRA adapters, export as SafeTensors, and publish to HuggingFace Hub
pub fn merge_qlora_export_publish(
    layers: &[(&str, &QLoRALayer)],
    publish_config: PublishConfig,
    output_dir: impl AsRef<Path>,
) -> Result<MergePublishResult, MergePublishError> {
    let output_dir = output_dir.as_ref();
    let filename = "model.safetensors";

    let merged = merge_qlora_and_collect(layers);
    let layers_merged = merged.layers_merged;

    let export_path = output_dir.join(filename);
    std::fs::create_dir_all(output_dir)
        .map_err(|e| MergePublishError::Merge(AdapterError::Io(e)))?;
    merged.save_safetensors(&export_path).map_err(MergePublishError::Merge)?;

    let publisher = HfPublisher::new(publish_config).map_err(MergePublishError::Publish)?;
    let files: Vec<(&Path, &str)> = vec![(&export_path, filename)];

    let publish = publisher.publish(&files, None).map_err(MergePublishError::Publish)?;

    Ok(MergePublishResult { layers_merged, publish })
}

/// Errors from the merge-export-publish pipeline
#[derive(Debug, thiserror::Error)]
pub enum MergePublishError {
    /// Merge/export phase failed
    #[error("Merge/export failed: {0}")]
    Merge(AdapterError),

    /// Publish phase failed
    #[error("Publish failed: {0}")]
    Publish(PublishError),
}
