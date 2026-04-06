//! Research init command arguments

use clap::Parser;
use std::path::PathBuf;

use super::super::types::{ArtifactTypeArg, LicenseArg};

/// Arguments for research init command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ResearchInitArgs {
    /// Artifact ID (unique identifier)
    #[arg(long)]
    pub id: String,

    /// Artifact title
    #[arg(long)]
    pub title: String,

    /// Artifact type
    #[arg(long, default_value = "dataset")]
    pub artifact_type: ArtifactTypeArg,

    /// License (e.g., CC-BY-4.0, MIT, Apache-2.0)
    #[arg(long, default_value = "cc-by-4.0")]
    pub license: LicenseArg,

    /// Output path for artifact YAML
    #[arg(short, long, default_value = "artifact.yaml")]
    pub output: PathBuf,

    /// Author name
    #[arg(long)]
    pub author: Option<String>,

    /// Author ORCID (format: 0000-0002-1825-0097)
    #[arg(long)]
    pub orcid: Option<String>,

    /// Author affiliation
    #[arg(long)]
    pub affiliation: Option<String>,

    /// Description of the artifact
    #[arg(long)]
    pub description: Option<String>,

    /// Keywords (comma-separated)
    #[arg(long)]
    pub keywords: Option<String>,

    /// DOI (if already assigned)
    #[arg(long)]
    pub doi: Option<String>,
}
