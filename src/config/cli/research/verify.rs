//! Verify command arguments

use clap::Parser;
use std::path::PathBuf;

/// Arguments for verify command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct VerifyArgs {
    /// Path to pre-registration or signed artifact
    #[arg(value_name = "FILE")]
    pub file: PathBuf,

    /// Path to Ed25519 public key for signature verification
    #[arg(long)]
    pub public_key: Option<PathBuf>,

    /// Original content to verify against commitment
    #[arg(long)]
    pub original: Option<PathBuf>,

    /// Verify git timestamp proof
    #[arg(long)]
    pub verify_git: bool,
}
