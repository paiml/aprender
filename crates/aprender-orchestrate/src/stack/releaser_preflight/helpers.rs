//! Shared helper functions for preflight checks
//!
//! This module contains free functions used by multiple preflight check methods.

use crate::stack::releaser::ReleaseOrchestrator;
use crate::stack::types::PreflightCheck;
use std::path::Path;
use std::process::Command;

/// Execute an external command described by a single whitespace-separated
/// config string and dispatch the result through a caller-provided closure.
///
/// Handles the three outcomes every check shares:
///   1. Empty command string  -> pass with `skip_msg`
///   2. Command not found     -> pass with "<tool> not found (skipped)"
///   3. Other spawn error     -> fail with error details
///   4. Successful spawn      -> delegate to `process_output`
pub fn run_check_command<F>(
    config_command: &str,
    check_id: &str,
    skip_msg: &str,
    crate_path: &Path,
    process_output: F,
) -> PreflightCheck
where
    F: FnOnce(&std::process::Output, &str, &str) -> PreflightCheck,
{
    let parts: Vec<&str> = config_command.split_whitespace().collect();
    if parts.is_empty() {
        return PreflightCheck::pass(check_id, skip_msg);
    }
    match Command::new(parts[0])
        .args(parts.get(1..).unwrap_or(&[]))
        .current_dir(crate_path)
        .output()
    {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            process_output(&output, &stdout, &stderr)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            PreflightCheck::pass(check_id, format!("{} not found (skipped)", parts[0]))
        }
        Err(e) => PreflightCheck::fail(check_id, format!("Failed to run {}: {}", parts[0], e)),
    }
}

/// Try several JSON keys in order and return the first successfully parsed f64.
pub fn parse_value_from_json(json: &str, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| ReleaseOrchestrator::parse_score_from_json(json, k))
}

/// Try several JSON keys in order and return the first successfully parsed u32.
pub fn parse_count_from_json_multi(json: &str, keys: &[&str]) -> Option<u32> {
    parse_value_from_json(json, keys).map(|f| f as u32)
}

/// Evaluate a numeric score against a threshold, producing a uniform
/// pass / fail / warning [`PreflightCheck`].
///
/// Used by TDG, Popper, and similar score-based gates.
pub fn score_check_result(
    check_id: &str,
    label: &str,
    value: Option<f64>,
    threshold: f64,
    fail_on_threshold: bool,
    status_success: bool,
) -> PreflightCheck {
    match value {
        Some(v) if v >= threshold => PreflightCheck::pass(
            check_id,
            format!("{}: {:.1} (minimum: {:.1})", label, v, threshold),
        ),
        Some(v) if fail_on_threshold => PreflightCheck::fail(
            check_id,
            format!("{} {:.1} below minimum {:.1}", label, v, threshold),
        ),
        Some(v) => PreflightCheck::pass(
            check_id,
            format!("{}: {:.1} (warning: below {:.1})", label, v, threshold),
        ),
        None if status_success => PreflightCheck::pass(check_id, format!("{} check passed", label)),
        None => {
            PreflightCheck::pass(check_id, format!("{} check completed (score not parsed)", label))
        }
    }
}
