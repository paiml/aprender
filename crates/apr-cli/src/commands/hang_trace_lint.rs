//! `apr hang-trace-lint` — CRUX-F-14 deadlock/hang-trace dir gate.
//!
//! Reads a captured `$APR_TRACE_DIR` after an `apr train` watchdog timeout
//! (or a successful run) and dispatches the pure classifiers in
//! `hang_trace_classifier`. Exits non-zero on any failure.
//!
//! Spec: `contracts/crux-F-14-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use super::hang_trace_classifier::{
    classify_empty_on_success, classify_exit_code, classify_timeout_dump,
    HangEmptyOnSuccessOutcome, HangExitOutcome, HangTimeoutOutcome, TraceDirListing,
};
use crate::error::{CliError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HangMode {
    Timeout,
    Success,
}

pub(crate) fn run(
    trace_dir: &Path,
    mode: HangMode,
    world_size: usize,
    exit_code: Option<i32>,
    expected_exit_code: Option<i32>,
    json: bool,
) -> Result<()> {
    if !trace_dir.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(trace_dir)));
    }
    // `--trace-dir` naming a regular file used to fall through to read_dir and
    // surface as a raw `IO error: Not a directory (os error 20)` with exit 7 —
    // an OS-failure code for what is really "the input you handed me is not a
    // usable observation" (exit 4 across the family; see commands::lint_error).
    // `--trace-dir` naming a regular file used to fall through to read_dir and
    // surface as a raw `IO error: Not a directory (os error 20)` with exit 7 —
    // an OS-failure code for what is really "the input you handed me is not a
    // usable observation" (exit 4 across the family; see commands::lint_error).
    if !trace_dir.is_dir() {
        return Err(CliError::InvalidInput(format!(
            "apr hang-trace-lint: --trace-dir is not a directory: {}",
            trace_dir.display()
        )));
    }
    let entries = std::fs::read_dir(trace_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            (name, size)
        })
        .collect::<Vec<_>>();
    let names: Vec<&str> = entries.iter().map(|e| e.0.as_str()).collect();
    let sizes: Vec<u64> = entries.iter().map(|e| e.1).collect();
    let listing = TraceDirListing { names, sizes };

    let dir_outcome_t = if mode == HangMode::Timeout {
        Some(classify_timeout_dump(&listing, world_size))
    } else {
        None
    };
    let dir_outcome_s = if mode == HangMode::Success {
        Some(classify_empty_on_success(&listing))
    } else {
        None
    };
    let exit_outcome = match (exit_code, expected_exit_code) {
        (Some(got), Some(expected)) => Some(classify_exit_code(got, expected)),
        _ => None,
    };

    print_report(
        trace_dir,
        dir_outcome_t.as_ref(),
        dir_outcome_s.as_ref(),
        exit_outcome.as_ref(),
        json,
    );

    if let Some(o) = &dir_outcome_t {
        if !matches!(o, HangTimeoutOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "hang-trace-lint timeout-dump gate rejected dir: {o:?}"
            )));
        }
    }
    if let Some(o) = &dir_outcome_s {
        if !matches!(o, HangEmptyOnSuccessOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "hang-trace-lint empty-on-success gate rejected dir: {o:?}"
            )));
        }
    }
    if let Some(o) = &exit_outcome {
        if matches!(o, HangExitOutcome::ExitCodeMismatch { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "hang-trace-lint exit-code gate rejected: {o:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    trace_dir: &Path,
    timeout_outcome: Option<&HangTimeoutOutcome>,
    success_outcome: Option<&HangEmptyOnSuccessOutcome>,
    exit_outcome: Option<&HangExitOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "trace_dir": trace_dir.display().to_string(),
            "timeout_dump": timeout_outcome.map(|o| format!("{o:?}")),
            "empty_on_success": success_outcome.map(|o| format!("{o:?}")),
            "exit_code": exit_outcome.map(|o| format!("{o:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("hang-trace-lint report for {}", trace_dir.display());
    if let Some(o) = timeout_outcome {
        println!("  timeout_dump    : {o:?}");
    }
    if let Some(o) = success_outcome {
        println!("  empty_on_success: {o:?}");
    }
    if let Some(o) = exit_outcome {
        println!("  exit_code       : {o:?}");
    }
}

#[cfg(test)]
mod cov_tests {
    use super::*;
    #[test]
    fn missing_trace_dir_is_file_not_found() {
        let err = run(
            Path::new("/no/such/tracedir"),
            HangMode::Timeout,
            2,
            Some(124),
            Some(124),
            false,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn missing_trace_dir_success_mode_is_file_not_found() {
        let err = run(
            Path::new("/no/such/tracedir2"),
            HangMode::Success,
            1,
            Some(0),
            Some(0),
            true,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
}
