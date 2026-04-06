//! Shared infrastructure for entrenar CLI tools.
//!
//! This crate provides common utilities used across all entrenar sub-crates:
//! - CLI styling and output formatting
//! - Error handling with actionable diagnostics
//! - Table rendering for terminal output
//! - Progress indicators
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Rich error messages with actionable diagnostics
//! - **Andon**: Visual problem indication through consistent styling
//! - **Muda Elimination**: Single source of truth for shared code

pub mod cli;
pub mod error;
pub mod output;
pub mod progress;

pub use cli::{Cli, OutputFormat};
pub use error::{EntrenarError, Result};
pub use output::{Table, TableBuilder};

/// Re-export trueno-viz for visualization
pub use trueno_viz;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_has_actionable_message() {
        let err = EntrenarError::ConfigNotFound {
            path: "/path/to/config.yaml".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("config.yaml"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_table_builder_creates_valid_table() {
        let table = TableBuilder::new()
            .headers(vec!["Name", "Value"])
            .row(vec!["test", "123"])
            .build();

        assert_eq!(table.headers().len(), 2);
        assert_eq!(table.rows().len(), 1);
    }

    #[test]
    fn test_output_format_parsing() {
        assert!(matches!(
            "json".parse::<OutputFormat>(),
            Ok(OutputFormat::Json)
        ));
        assert!(matches!(
            "table".parse::<OutputFormat>(),
            Ok(OutputFormat::Table)
        ));
        assert!("invalid".parse::<OutputFormat>().is_err());
    }
}
