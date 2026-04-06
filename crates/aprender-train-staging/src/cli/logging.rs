//! Logging utilities for CLI output

/// Log level for CLI output
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Suppress all output
    Quiet,
    /// Normal output level
    Normal,
    /// Verbose output with additional details
    Verbose,
}

/// Log a message if the current level permits it
pub fn log(level: LogLevel, required: LogLevel, msg: &str) {
    if level != LogLevel::Quiet && (level == required || required == LogLevel::Normal) {
        println!("{msg}");
    }
}
