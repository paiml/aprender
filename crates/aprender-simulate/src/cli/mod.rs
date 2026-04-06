//! CLI module for simular.
//!
//! This module contains all CLI logic extracted from main.rs to enable
//! full test coverage. The entry point `run_cli` can be called from main.rs
//! with parsed arguments.

mod args;
mod commands;
mod output;
mod schema;

pub use args::{Args, Command, RenderFormat};
pub use commands::run_cli;
pub use output::{
    print_emc_report, print_emc_validation_results, print_experiment_result, print_help,
    print_version,
};
pub use schema::validate_emc_schema;

#[cfg(test)]
mod tests;
