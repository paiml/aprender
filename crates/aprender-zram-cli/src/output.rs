//! Output formatting for CLI.

use clap::ValueEnum;

/// Output format selection.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum OutputFormat {
    /// Human-readable table format.
    #[default]
    Table,
    /// JSON output.
    Json,
    /// Raw values (for scripting).
    Raw,
}
