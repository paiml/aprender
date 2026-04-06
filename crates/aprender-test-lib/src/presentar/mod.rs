//! Presentar YAML support for probar.
//!
//! This module provides native support for testing presentar YAML configurations,
//! enabling automated validation of TUI dashboards, terminal rendering, and
//! falsification protocols.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                 Presentar Integration                    │
//! ├─────────────────────────────────────────────────────────┤
//! │  schema.rs      → YAML schema types (PresentarConfig)   │
//! │  validator.rs   → Config validation rules               │
//! │  terminal.rs    → CellBuffer snapshot assertions        │
//! │  falsification.rs → F001-F100 generator                 │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - Tretmans (2008): Model-Based Testing of Reactive Systems
//! - Claessen & Hughes (2000): QuickCheck property-based testing
//! - Jia & Harman (2011): Mutation Testing theory

mod falsification;
mod schema;
mod terminal;
mod validator;

pub use falsification::{generate_falsification_playbook, FalsificationCheck, FalsificationResult};
pub use schema::{
    KeybindingConfig, LayoutConfig, PanelConfig, PanelConfigs, PanelType, PresentarConfig,
    ThemeConfig,
};
pub use terminal::{Cell, Color, TerminalAssertion, TerminalSnapshot};
pub use validator::{parse_and_validate, validate_config, PresentarError, ValidationResult};

/// Presentar schema version supported by this module.
pub const SCHEMA_VERSION: &str = "1.0";

/// Number of falsification checks (F001-F100).
pub const FALSIFICATION_COUNT: usize = 100;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version() {
        assert_eq!(SCHEMA_VERSION, "1.0");
    }

    #[test]
    fn test_falsification_count() {
        assert_eq!(FALSIFICATION_COUNT, 100);
    }
}
