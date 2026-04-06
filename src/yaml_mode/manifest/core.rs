//! Core Training Manifest
//!
//! Contains the root TrainingManifest struct that represents the complete YAML Mode
//! training configuration.

use serde::{Deserialize, Serialize};

use super::callback::CallbackConfig;
use super::data::DataConfig;
use super::extended::{
    AuditConfig, BenchmarkConfig, CitlConfig, DebugConfig, DistillationConfig, GraphConfig,
    InspectConfig, PrivacyConfig, RagConfig, SessionConfig, SigningConfig, StressConfig,
    VerificationConfig,
};
use super::lora::LoraConfig;
use super::model::ModelConfig;
use super::monitoring::MonitoringConfig;
use super::optimizer::OptimizerConfig;
use super::output::OutputConfig;
use super::publish::PublishConfig;
use super::quantize::QuantizeConfig;
use super::scheduler::SchedulerConfig;
use super::training::TrainingConfig;

/// Complete training manifest (root structure)
///
/// # Required Fields
/// - `entrenar`: Specification version (must be "1.0")
/// - `name`: Experiment identifier
/// - `version`: Experiment version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingManifest {
    /// Specification version (required)
    pub entrenar: String,

    /// Experiment name (required)
    pub name: String,

    /// Experiment version (required)
    pub version: String,

    /// Human-readable description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Global random seed for reproducibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Dataset configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataConfig>,

    /// Model configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelConfig>,

    /// Optimizer configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<OptimizerConfig>,

    /// Learning rate scheduler configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<SchedulerConfig>,

    /// Training loop configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingConfig>,

    /// LoRA fine-tuning configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraConfig>,

    /// Quantization configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize: Option<QuantizeConfig>,

    /// Monitoring configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monitoring: Option<MonitoringConfig>,

    /// Training callbacks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callbacks: Option<Vec<CallbackConfig>>,

    /// Output and artifact configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputConfig>,

    /// Publish configuration for auto-uploading to HuggingFace Hub
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publish: Option<PublishConfig>,

    // Extended configurations for YAML Mode QA Epic
    /// CITL (Compiler-in-the-Loop) configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citl: Option<CitlConfig>,

    /// RAG configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rag: Option<RagConfig>,

    /// Graph output configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphConfig>,

    /// Distillation configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distillation: Option<DistillationConfig>,

    /// Inspection configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inspect: Option<InspectConfig>,

    /// Privacy configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy: Option<PrivacyConfig>,

    /// Audit configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit: Option<AuditConfig>,

    /// Session configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionConfig>,

    /// Stress testing configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stress: Option<StressConfig>,

    /// Benchmark configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<BenchmarkConfig>,

    /// Debug configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<DebugConfig>,

    /// Signing configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing: Option<SigningConfig>,

    /// Verification configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<VerificationConfig>,

    /// Lockfile path for reproducibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lockfile: Option<String>,

    /// Strict mode for lockfile enforcement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,

    /// Strict validation mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_validation: Option<bool>,

    /// Require peer review for production
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_peer_review: Option<bool>,
}
