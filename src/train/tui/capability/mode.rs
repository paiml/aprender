//! Terminal mode types.

/// Terminal rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalMode {
    /// ASCII only (widest compatibility)
    Ascii,
    /// Unicode characters (modern terminals)
    #[default]
    Unicode,
    /// ANSI color codes
    Ansi,
}
