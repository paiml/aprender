//! YAML Mode Training - Declarative, No-Code Training Interface
//!
//! This module implements the YAML Mode Training specification (v1.0) which enables
//! ML practitioners to configure, execute, and monitor model training using only YAML.
//!
//! ## Core Principles (Toyota Way)
//!
//! - **Muda Elimination**: No redundant code; configuration-only workflows
//! - **Poka-yoke**: Schema validation catches errors at parse time, not runtime
//! - **Jidoka**: Built-in quality with automatic checkpointing and early stopping
//! - **Heijunka**: Reproducible training through deterministic seeding
//! - **Kaizen**: Experiment tracking enables iterative refinement
//!
//! ## Usage
//!
//! ```yaml
//! entrenar: "1.0"
//! name: "my-experiment"
//! version: "1.0.0"
//!
//! data:
//!   source: "./data/train.parquet"
//!
//! model:
//!   source: "./models/base.safetensors"
//!
//! training:
//!   epochs: 10
//! ```

pub mod bridge;
mod manifest;
mod templates;
mod validation;

#[cfg(test)]
mod tests;

pub use bridge::{manifest_to_spec, BridgeError, BridgeResult};
pub use manifest::{
    AuditConfig, BackpressureConfig, BenchmarkConfig, CallbackConfig, CallbackType, CitlConfig,
    DataConfig, DataLoader, DataSplit, DebugConfig, DistillModelRef, DistillationConfig,
    DriftDetectionConfig, GradientConfig, GraphConfig, InspectConfig, LoraConfig,
    MixedPrecisionConfig, ModelConfig, MonitoringConfig, OptimizerConfig, OutputConfig,
    PreprocessingStep, PrivacyConfig, PublishConfig, QuantizeConfig, RagConfig, SchedulerConfig,
    SessionConfig, SigningConfig, StressConfig, TerminalMonitor, TrackingConfig, TrainingConfig,
    TrainingManifest, VerificationConfig, WarmupConfig,
};
pub use templates::{generate_manifest, generate_manifest_with_hints, generate_yaml, Template};
pub use validation::{validate_manifest, ManifestError, ValidationResult};

use std::path::Path;

/// Load and validate a training manifest from a YAML file
pub fn load_manifest(path: &Path) -> crate::Result<TrainingManifest> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| crate::Error::Io(format!("Failed to read manifest: {e}")))?;

    let manifest: TrainingManifest = serde_yaml::from_str(&content)
        .map_err(|e| crate::Error::Parse(format!("Failed to parse manifest: {e}")))?;

    validate_manifest(&manifest).map_err(|e| crate::Error::Validation(e.to_string()))?;

    Ok(manifest)
}

/// Save a training manifest to a YAML file
pub fn save_manifest(manifest: &TrainingManifest, path: &Path) -> crate::Result<()> {
    let content = serde_yaml::to_string(manifest)
        .map_err(|e| crate::Error::Parse(format!("Failed to serialize manifest: {e}")))?;

    std::fs::write(path, content)
        .map_err(|e| crate::Error::Io(format!("Failed to write manifest: {e}")))?;

    Ok(())
}
