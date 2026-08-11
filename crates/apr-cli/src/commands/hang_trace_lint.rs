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
    // `classify_timeout_dump` short-circuits to `Ok { ranks_seen: 0 }` at
    // world_size 0, so an unset `${WORLD_SIZE}` expanding to 0 turned the whole
    // timeout-dump gate into an unconditional pass whatever the directory held.
    // ddp-metrics-lint already rejects world_size 0; this matches it.
    if mode == HangMode::Timeout && world_size == 0 {
        return Err(CliError::ValidationFailed(
            "hang-trace-lint: --world-size 0 is not a world size — the timeout-dump gate would \
             accept any directory contents. Pass the rank count the run was launched with."
                .to_string(),
        ));
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

    /// At `--world-size 0` the timeout-dump gate reported
    /// `Ok { ranks_seen: 0 }` for three different directory states, each of
    /// which it correctly rejects at world_size 2.
    #[test]
    fn falsifier_world_size_zero_is_rejected_not_silently_passed() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("rank0.py.txt");
        std::fs::write(&good, "stack rank0").unwrap();
        std::fs::write(dir.path().join("rank1.py.txt"), "stack rank1").unwrap();

        // Control: the gate is live at a real world size.
        assert!(run(dir.path(), HangMode::Timeout, 2, None, None, false).is_ok());

        for state in ["populated", "truncated", "unrecognised-file"] {
            match state {
                "truncated" => std::fs::write(&good, "").unwrap(),
                "unrecognised-file" => {
                    std::fs::write(dir.path().join("random.txt"), "x").unwrap();
                }
                _ => {}
            }
            let err = run(dir.path(), HangMode::Timeout, 0, None, None, false).unwrap_err();
            match err {
                CliError::ValidationFailed(msg) => {
                    assert!(msg.contains("--world-size 0 is not a world size"), "{msg}");
                }
                other => panic!("{state}: expected ValidationFailed, got {other:?}"),
            }
            // …and at a real world size the same directory does fail.
            if state != "populated" {
                assert!(
                    run(dir.path(), HangMode::Timeout, 2, None, None, false).is_err(),
                    "{state}: control must fail at world_size 2"
                );
            }
        }
    }

    /// Success mode does not consult world_size, so it must stay reachable.
    #[test]
    fn success_mode_ignores_world_size_zero() {
        let dir = tempfile::tempdir().unwrap();
        assert!(run(dir.path(), HangMode::Success, 0, None, None, false).is_ok());
    }
}
