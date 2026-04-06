//! Bundle command arguments

use clap::Parser;
use std::path::PathBuf;

/// Arguments for bundle command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct BundleArgs {
    /// Path to artifact YAML
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Output directory for RO-Crate
    #[arg(short, long)]
    pub output: PathBuf,

    /// Files to include (can be repeated)
    #[arg(short, long)]
    pub file: Vec<PathBuf>,

    /// Create ZIP archive instead of directory
    #[arg(long)]
    pub zip: bool,

    /// Include citation graph
    #[arg(long)]
    pub include_citations: bool,
}
