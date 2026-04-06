//! CLI module for entrenar
//!
//! This module contains all CLI command handlers and utilities.

mod commands;
mod logging;

pub use commands::run_command;
pub use logging::LogLevel;

// Re-export Cli from config for convenience
pub use crate::config::Cli;
