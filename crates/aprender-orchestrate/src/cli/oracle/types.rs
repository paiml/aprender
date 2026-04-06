//! Oracle types and enums
//!
//! This module contains shared types used across oracle submodules.

/// Oracle output format
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum OracleOutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output
    Json,
    /// Markdown output
    Markdown,
    /// Raw code output only - no metadata, no colors
    Code,
    /// Code with accompanying SVG diagram
    #[value(name = "code+svg")]
    CodeSvg,
}
