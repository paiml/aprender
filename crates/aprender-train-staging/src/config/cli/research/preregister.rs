//! Pre-registration command arguments

use clap::Parser;
use std::path::PathBuf;

/// Arguments for preregister command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct PreregisterArgs {
    /// Research question or title
    #[arg(long)]
    pub title: String,

    /// Hypothesis being tested
    #[arg(long)]
    pub hypothesis: String,

    /// Methodology description
    #[arg(long)]
    pub methodology: String,

    /// Statistical analysis plan
    #[arg(long)]
    pub analysis_plan: String,

    /// Additional notes
    #[arg(long)]
    pub notes: Option<String>,

    /// Output path for pre-registration
    #[arg(short, long, default_value = "preregistration.yaml")]
    pub output: PathBuf,

    /// Path to Ed25519 private key for signing
    #[arg(long)]
    pub sign_key: Option<PathBuf>,

    /// Add git commit hash as timestamp proof
    #[arg(long)]
    pub git_timestamp: bool,
}
