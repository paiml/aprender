//! Declarative YAML configuration
//!
//! This module provides Ludwig-style declarative training configuration via YAML.
//!
//! # Example
//!
//! ```yaml
//! model:
//!   path: base-model.gguf
//!   layers: [q_proj, v_proj]
//!
//! data:
//!   train: train.parquet
//!   batch_size: 8
//!
//! optimizer:
//!   name: adam
//!   lr: 1e-4
//!
//! lora:
//!   rank: 64
//!   alpha: 16
//! ```

mod builder;
mod cli;
mod infer;
mod schema;
mod train;
mod validate;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;

pub use builder::{build_model, build_optimizer};
pub use cli::{
    apply_overrides, parse_args, ArchiveProviderArg, ArtifactTypeArg, AuditArgs, AuditType,
    BenchArgs, BundleArgs, CitationFormat, CiteArgs, Cli, Command, CompletionArgs, DepositArgs,
    ExperimentsArgs, ExperimentsCommand, ExportArgs, ExportFormat, FinetuneArgs, FinetuneCommand,
    InfoArgs, InitArgs, InitTemplate, InspectArgs, InspectMode, LicenseArg, MergeArgs, MergeMethod,
    MonitorArgs, OutputFormat, PreregisterArgs, PublishArgs, QuantMethod, QuantizeArgs,
    ResearchArgs, ResearchCommand, ResearchInitArgs, ShellType, TrainArgs, TrainingMethod,
    ValidateArgs, VerifyArgs,
};
pub use infer::{
    collect_stats_from_samples, infer_schema, infer_schema_from_path, infer_type, ColumnStats,
    FeatureType, InferenceConfig, InferredSchema,
};
pub use schema::{
    is_hf_repo_id, ArchitectureOverrides, CurriculumStage, DataConfig, LoRASpec, MergeSpec,
    ModelMode, ModelRef, OptimSpec, PublishSpec, QuantSpec, TrainSpec, TrainingMode,
    TrainingParams,
};
pub use train::{load_config, train_from_yaml, try_load_apr_for_inference};
pub use validate::{validate_config, ValidationError};
