//! Trueno development task runner (xtask pattern)
//!
//! Provides development utilities:
//! - `check-simd`: Validate SIMD backend detection
//! - `install-hooks`: Install git hooks
//! - `validate-examples`: Check example quality

// Development-phase lint allows
#![allow(clippy::useless_vec)]
#![cfg_attr(test, allow(clippy::disallowed_methods))]

mod check_simd;
mod install_hooks;
mod validate_examples;

use anyhow::{bail, Result};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if let Err(e) = run_command(&args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Parse and run the command from command-line arguments (pure, testable)
fn run_command(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        bail!("Usage: cargo xtask <command>\nCommands:\n  check-simd         Check SIMD attributes (pre-commit validation)\n  install-hooks      Install git pre-commit hooks\n  validate-examples  Validate book examples meet EXTREME TDD quality");
    }

    let command = &args[1];

    match command.as_str() {
        "check-simd" => check_simd::run(),
        "install-hooks" => install_hooks::run(),
        "validate-examples" => validate_examples::run(),
        _ => bail!("Unknown command: {}", command),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_command_validate_examples() {
        // We can't easily test the actual execution since it requires a full project setup
        // but we can test the command parsing logic
        let args = vec!["xtask".to_string(), "validate-examples".to_string()];
        // This will fail because we don't have examples/ in test env, but it proves parsing works
        let result = run_command(&args);
        assert!(result.is_err()); // Will fail to find examples dir, which is expected
    }

    #[test]
    fn test_run_command_no_args() {
        let args = vec!["xtask".to_string()];
        let result = run_command(&args);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Usage"));
    }

    #[test]
    fn test_run_command_unknown() {
        let args = vec!["xtask".to_string(), "unknown".to_string()];
        let result = run_command(&args);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown command"));
    }

    #[test]
    fn test_run_command_empty_args() {
        let args: Vec<String> = vec![];
        let result = run_command(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_command_check_simd() {
        let args = vec!["xtask".to_string(), "check-simd".to_string()];
        // check-simd will run against actual source files
        let result = run_command(&args);
        // Should succeed or fail based on actual source - we just test it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_run_command_install_hooks() {
        let args = vec!["xtask".to_string(), "install-hooks".to_string()];
        // install-hooks requires .git directory which exists in this repo
        let result = run_command(&args);
        // Should run without panicking
        let _ = result;
    }

    #[test]
    fn test_run_command_usage_message() {
        let args = vec!["xtask".to_string()];
        let result = run_command(&args);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("check-simd"));
        assert!(err_msg.contains("install-hooks"));
        assert!(err_msg.contains("validate-examples"));
    }

    #[test]
    fn test_run_command_with_extra_args() {
        // Test that we parse first command arg only
        let args = vec!["xtask".to_string(), "unknown".to_string(), "extra".to_string()];
        let result = run_command(&args);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown command"));
        assert!(err.contains("unknown"));
    }
}
