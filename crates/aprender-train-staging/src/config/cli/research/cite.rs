//! Citation command arguments

use clap::Parser;
use std::path::PathBuf;

use super::super::types::CitationFormat;

/// Arguments for cite command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct CiteArgs {
    /// Path to artifact YAML file
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Publication year
    #[arg(long)]
    pub year: u16,

    /// Output format
    #[arg(short, long, default_value = "bibtex")]
    pub format: CitationFormat,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Journal name
    #[arg(long)]
    pub journal: Option<String>,

    /// Volume number
    #[arg(long)]
    pub volume: Option<String>,

    /// Page range
    #[arg(long)]
    pub pages: Option<String>,

    /// URL
    #[arg(long)]
    pub url: Option<String>,
}
