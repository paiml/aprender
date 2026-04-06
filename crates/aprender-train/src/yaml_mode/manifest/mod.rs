//! Training Manifest Schema
//!
//! Defines the complete YAML Mode Training manifest structure as specified in
//! docs/specifications/yaml-mode-train.md
//!
//! This module is organized into submodules for better maintainability:
//! - `core` - TrainingManifest root struct
//! - `data` - DataConfig, DataSplit, PreprocessingStep, DataLoader, etc.
//! - `model` - ModelConfig, ArchitectureConfig
//! - `optimizer` - OptimizerConfig, ParamGroup
//! - `scheduler` - SchedulerConfig, WarmupConfig
//! - `training` - TrainingConfig, GradientConfig, MixedPrecisionConfig, etc.
//! - `lora` - LoraConfig
//! - `quantize` - QuantizeConfig, QatConfig, CalibrationConfig
//! - `monitoring` - MonitoringConfig, DriftDetectionConfig, TerminalMonitor, etc.
//! - `callback` - CallbackConfig, CallbackType
//! - `output` - OutputConfig, ModelOutputConfig, MetricsOutputConfig, etc.
//! - `extended` - Extended configurations for YAML Mode QA Epic

pub mod callback;
pub mod core;
pub mod data;
pub mod extended;
pub mod lora;
pub mod model;
pub mod monitoring;
pub mod optimizer;
pub mod output;
pub mod publish;
pub mod quantize;
pub mod scheduler;
pub mod shorthand;
pub mod training;

// Re-export all public types for API compatibility
pub use callback::{CallbackConfig, CallbackType};
pub use core::TrainingManifest;
pub use data::{DataConfig, DataLoader, DataSplit, PreprocessingStep};
pub use extended::{
    AuditConfig, BackpressureConfig, BenchmarkConfig, CitlConfig, DebugConfig, DistillModelRef,
    DistillationConfig, GraphConfig, InspectConfig, PrivacyConfig, RagConfig, SessionConfig,
    SigningConfig, StressConfig, VerificationConfig,
};
pub use lora::LoraConfig;
pub use model::ModelConfig;
pub use monitoring::{
    AlertConfig, ChartConfig, DriftDetectionConfig, MonitoringConfig, SystemMonitorConfig,
    TerminalMonitor, TrackingConfig,
};
pub use optimizer::OptimizerConfig;
pub use output::{
    MetricsOutputConfig, ModelOutputConfig, OutputConfig, RegistryConfig, ReportConfig,
};
pub use publish::PublishConfig;
pub use quantize::QuantizeConfig;
pub use scheduler::{SchedulerConfig, WarmupConfig};
pub use training::{
    CheckpointConfig, EarlyStoppingConfig, GradientConfig, MixedPrecisionConfig, TrainingConfig,
};
