//! Export command arguments

use clap::Parser;
use std::path::PathBuf;

use super::super::types::ExportFormat;

/// Arguments for export command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ExportArgs {
    /// Path to artifact or document
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Export format
    #[arg(short, long)]
    pub format: ExportFormat,

    /// Output file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Anonymize for double-blind review
    #[arg(long)]
    pub anonymize: bool,

    /// Salt for anonymization (required with --anonymize)
    #[arg(long)]
    pub anon_salt: Option<String>,

    /// Jupyter kernel (for notebook export)
    #[arg(long, default_value = "python3")]
    pub kernel: String,
}
