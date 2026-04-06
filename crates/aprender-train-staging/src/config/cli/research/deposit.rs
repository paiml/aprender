//! Deposit command arguments

use clap::Parser;
use std::path::PathBuf;

use super::super::types::ArchiveProviderArg;

/// Arguments for deposit command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct DepositArgs {
    /// Path to artifact YAML
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Archive provider
    #[arg(short, long)]
    pub provider: ArchiveProviderArg,

    /// API token (or use env var ZENODO_TOKEN, etc.)
    #[arg(long)]
    pub token: Option<String>,

    /// Use sandbox/test environment
    #[arg(long)]
    pub sandbox: bool,

    /// Community to submit to
    #[arg(long)]
    pub community: Option<String>,

    /// Files to include (can be repeated)
    #[arg(short, long)]
    pub file: Vec<PathBuf>,

    /// Dry run (validate without uploading)
    #[arg(long)]
    pub dry_run: bool,
}
