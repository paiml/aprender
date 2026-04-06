//! CLI argument parsing and validation
//!
//! This module provides the command-line interface for entrenar training.
//!
//! # Usage
//!
//! ```bash
//! entrenar train config.yaml
//! entrenar train config.yaml --output-dir ./checkpoints
//! entrenar train config.yaml --resume checkpoint.json
//! entrenar validate config.yaml
//! entrenar info config.yaml
//! ```

mod core;
mod extended;
mod init;
mod quant_merge;
mod research;
mod types;

#[cfg(test)]
mod property_tests;
#[cfg(test)]
mod tests;

// Re-export all public types
pub use core::{apply_overrides, parse_args, Cli, Command, InfoArgs, TrainArgs, ValidateArgs};
pub use extended::{
    AuditArgs, BenchArgs, CompletionArgs, ExperimentsArgs, ExperimentsCommand, FinetuneArgs,
    FinetuneCommand, InspectArgs, MonitorArgs, PublishArgs,
};
pub use init::{InitArgs, InitTemplate, TrainingMethod};
pub use quant_merge::{MergeArgs, MergeMethod, QuantMethod, QuantizeArgs};
pub use research::{
    BundleArgs, CiteArgs, DepositArgs, ExportArgs, PreregisterArgs, ResearchArgs, ResearchCommand,
    ResearchInitArgs, VerifyArgs,
};
pub use types::{
    ArchiveProviderArg, ArtifactTypeArg, AuditType, CitationFormat, ExportFormat, InspectMode,
    LicenseArg, OutputFormat, ShellType,
};
