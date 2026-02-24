//! Qualify Command — Cross-subcommand smoke testing for model files.
//!
//! Runs every diagnostic CLI tool against a model file to verify no crashes.
//! Fills the gap between `apr qa` (inference gates) and unit tests (isolated logic).
//!
//! ## Tiers
//! - `smoke` (default): 11 in-process subcommand gates
//! - `standard`: smoke + contract audit via `pv`
//! - `full`: standard + playbook check via `apr-qa`

use crate::commands::{debug, explain, flow, hex, inspect, lint, tensors, tree, validate};
use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use serde::Serialize;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct GateResult {
    name: String,
    display_name: String,
    status: GateStatus,
    message: String,
    duration_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
enum GateStatus {
    Pass,
    Fail,
    Skip,
    Panic,
    Timeout,
}

impl GateResult {
    fn is_failure(&self) -> bool {
        matches!(
            self.status,
            GateStatus::Fail | GateStatus::Panic | GateStatus::Timeout
        )
    }

    fn is_skip(&self) -> bool {
        matches!(self.status, GateStatus::Skip)
    }
}

#[derive(Debug, Serialize)]
struct QualifyReport {
    model: String,
    tier: String,
    passed: bool,
    gates_executed: usize,
    gates_skipped: usize,
    gates_failed: usize,
    total_duration_ms: u64,
    gates: Vec<GateResult>,
}

// ============================================================================
// Gate Runners
// ============================================================================

/// Run an in-process gate with timeout, panic catch, and stdout suppression.
fn run_gate<F>(name: &str, display_name: &str, timeout_secs: u64, verbose: bool, f: F) -> GateResult
where
    F: FnOnce() -> Result<()> + Send + 'static,
{
    let start = Instant::now();
    let (tx, rx) = mpsc::channel();

    let _handle = std::thread::spawn(move || {
        // Scope the gag so fd 1 is restored BEFORE sending the result.
        // This prevents the main thread from printing into a gagged fd.
        let result = {
            let _stdout_gag = if verbose {
                None
            } else {
                gag::Gag::stdout().ok()
            };
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        };
        // Gag dropped here — fd 1 restored before main thread unblocks.
        let _ = tx.send(result);
    });

    let timeout = Duration::from_secs(timeout_secs);
    let (status, message) = match rx.recv_timeout(timeout) {
        Ok(Ok(Ok(()))) => (GateStatus::Pass, "OK".to_string()),
        Ok(Ok(Err(e))) => (GateStatus::Fail, format!("{e}")),
        Ok(Err(panic_info)) => {
            let msg = panic_info
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic_info.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            (GateStatus::Panic, format!("PANIC: {msg}"))
        }
        Err(_) => (
            GateStatus::Timeout,
            format!("Timed out after {timeout_secs}s"),
        ),
    };

    GateResult {
        name: name.to_string(),
        display_name: display_name.to_string(),
        status,
        message,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

/// Run an external tool gate (shell out). Skips gracefully if binary not on PATH.
fn run_external_gate(
    name: &str,
    display_name: &str,
    binary: &str,
    args: &[&str],
    timeout_secs: u64,
) -> GateResult {
    let which = std::process::Command::new("which")
        .arg(binary)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    if which.map_or(true, |s| !s.success()) {
        return GateResult {
            name: name.to_string(),
            display_name: display_name.to_string(),
            status: GateStatus::Skip,
            message: format!("{binary} not on PATH"),
            duration_ms: 0,
        };
    }

    let start = Instant::now();
    let result = std::process::Command::new(binary)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output();
    let duration_ms = start.elapsed().as_millis() as u64;

    if duration_ms > timeout_secs * 1000 {
        return GateResult {
            name: name.to_string(),
            display_name: display_name.to_string(),
            status: GateStatus::Timeout,
            message: format!("Timed out after {timeout_secs}s"),
            duration_ms,
        };
    }

    match result {
        Ok(out) if out.status.success() => GateResult {
            name: name.to_string(),
            display_name: display_name.to_string(),
            status: GateStatus::Pass,
            message: "OK".to_string(),
            duration_ms,
        },
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let msg = stderr.lines().next().unwrap_or("exit non-zero").to_string();
            GateResult {
                name: name.to_string(),
                display_name: display_name.to_string(),
                status: GateStatus::Fail,
                message: msg,
                duration_ms,
            }
        }
        Err(e) => GateResult {
            name: name.to_string(),
            display_name: display_name.to_string(),
            status: GateStatus::Fail,
            message: format!("{e}"),
            duration_ms,
        },
    }
}

// ============================================================================
// Summary Output
// ============================================================================

fn print_summary(gates: &[GateResult], passed: bool, total_duration: Duration) {
    output::header("Qualify Summary");

    let gate_rows: Vec<Vec<String>> = gates
        .iter()
        .map(|g| {
            let badge = match g.status {
                GateStatus::Pass => output::badge_pass("PASS"),
                GateStatus::Fail => output::badge_fail("FAIL"),
                GateStatus::Skip => output::badge_skip("SKIP"),
                GateStatus::Panic => output::badge_fail("PANIC"),
                GateStatus::Timeout => output::badge_warn("TMOUT"),
            };
            vec![
                g.display_name.clone(),
                badge,
                g.message.clone(),
                output::duration_fmt(g.duration_ms),
            ]
        })
        .collect();

    println!(
        "{}",
        output::table(&["Gate", "Status", "Message", "Duration"], &gate_rows)
    );

    println!();
    if passed {
        println!("  {}", output::badge_pass("ALL GATES PASSED"));
    } else {
        println!("  {}", output::badge_fail("GATES FAILED"));
        for gate in gates.iter().filter(|g| g.is_failure()) {
            println!("    {} {} — {}", "✗".red(), gate.display_name, gate.message);
        }
    }
    output::metric(
        "Total Duration",
        output::duration_fmt(total_duration.as_millis() as u64),
        "",
    );
}

// ============================================================================
// Smoke Gate Dispatcher
// ============================================================================

fn dispatch_smoke_gate(
    name: &str,
    display: &str,
    path: &Path,
    timeout: u64,
    verbose: bool,
) -> GateResult {
    match name {
        "inspect" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                inspect::run(&p, false, false, false, false)
            })
        }
        "validate" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                validate::run(&p, false, false, None, false)
            })
        }
        "validate_quality" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                validate::run(&p, true, false, None, false)
            })
        }
        "tensors" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                tensors::run(&p, false, None, false, 0)
            })
        }
        "lint" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                lint::run(&p, false)
            })
        }
        "debug" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                debug::run(&p, false, false, false, 256, false)
            })
        }
        "tree" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                tree::run(&p, None, tree::TreeFormat::Ascii, false, None)
            })
        }
        "hex" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                hex::run(&hex::HexOptions {
                    file: p,
                    header: true,
                    ..hex::HexOptions::default()
                })
            })
        }
        "flow" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                flow::run(&p, None, flow::FlowComponent::Full, false, false)
            })
        }
        "explain" => {
            let p = path.to_path_buf();
            run_gate(name, display, timeout, verbose, move || {
                let path_str = p.display().to_string();
                explain::run(Some(path_str), None, None)
            })
        }
        "check" => {
            #[cfg(feature = "inference")]
            {
                let p = path.to_path_buf();
                run_gate(name, display, timeout, verbose, move || {
                    crate::commands::check::run(&p, false, false)
                })
            }
            #[cfg(not(feature = "inference"))]
            {
                GateResult {
                    name: name.to_string(),
                    display_name: display.to_string(),
                    status: GateStatus::Skip,
                    message: "Requires inference feature".to_string(),
                    duration_ms: 0,
                }
            }
        }
        _ => GateResult {
            name: name.to_string(),
            display_name: display.to_string(),
            status: GateStatus::Skip,
            message: "Unknown gate".to_string(),
            duration_ms: 0,
        },
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Smoke gate definitions: (name, display_name)
const SMOKE_GATES: &[(&str, &str)] = &[
    ("inspect", "Inspect"),
    ("validate", "Validate"),
    ("validate_quality", "Validate (quality)"),
    ("tensors", "Tensors"),
    ("lint", "Lint"),
    ("debug", "Debug"),
    ("tree", "Tree"),
    ("hex", "Hex"),
    ("flow", "Flow"),
    ("explain", "Explain"),
    ("check", "Check (pipeline)"),
];

#[allow(clippy::too_many_arguments)]
pub fn run(
    file: &Path,
    tier: &str,
    timeout: u64,
    json: bool,
    verbose: bool,
    skip: Option<&[String]>,
) -> Result<()> {
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    let skip_set: HashSet<&str> = skip
        .map(|s| s.iter().map(String::as_str).collect())
        .unwrap_or_default();

    let total_start = Instant::now();
    let mut gates = Vec::new();

    if !json {
        output::header("Qualify");
        output::kv("Model", file.display());
        output::kv("Tier", tier);
        println!();
        // Flush before first gate gag to avoid buffered output loss
        let _ = std::io::stdout().flush();
    }

    // ── Phase 1: Smoke Gates (always) ──────────────────────────────────

    for &(name, display) in SMOKE_GATES {
        if skip_set.contains(name) {
            gates.push(GateResult {
                name: name.to_string(),
                display_name: display.to_string(),
                status: GateStatus::Skip,
                message: "Skipped by user".to_string(),
                duration_ms: 0,
            });
            continue;
        }

        let gate = dispatch_smoke_gate(name, display, file, timeout, verbose);

        if !json {
            let badge = match gate.status {
                GateStatus::Pass => output::badge_pass("PASS"),
                GateStatus::Fail => output::badge_fail("FAIL"),
                GateStatus::Skip => output::badge_skip("SKIP"),
                GateStatus::Panic => output::badge_fail("PANIC"),
                GateStatus::Timeout => output::badge_warn("TMOUT"),
            };
            println!(
                "  {badge} {display} ({})",
                output::duration_fmt(gate.duration_ms)
            );
            // Flush stdout before next gate's gag redirects fd 1
            let _ = std::io::stdout().flush();
        }

        gates.push(gate);
    }

    // ── Phase 2: Contract Audit (standard or full tier) ────────────────

    if (tier == "standard" || tier == "full") && !skip_set.contains("contract_audit") {
        let gate = run_external_gate(
            "contract_audit",
            "Contract Audit (pv)",
            "pv",
            &["audit", "contracts/aprender/tensor-layout-v1.yaml"],
            timeout,
        );
        if !json {
            let badge = match gate.status {
                GateStatus::Pass => output::badge_pass("PASS"),
                GateStatus::Skip => output::badge_skip("SKIP"),
                _ => output::badge_fail("FAIL"),
            };
            println!(
                "  {badge} {} ({})",
                gate.display_name,
                output::duration_fmt(gate.duration_ms)
            );
        }
        gates.push(gate);
    }

    // ── Phase 3: Playbook Check (full tier only) ───────────────────────

    if tier == "full" && !skip_set.contains("playbook_tools") {
        let file_str = file.display().to_string();
        let gate = run_external_gate(
            "playbook_tools",
            "Playbook (apr-qa)",
            "apr-qa",
            &["tools", &file_str, "--no-gpu"],
            timeout,
        );
        if !json {
            let badge = match gate.status {
                GateStatus::Pass => output::badge_pass("PASS"),
                GateStatus::Skip => output::badge_skip("SKIP"),
                _ => output::badge_fail("FAIL"),
            };
            println!(
                "  {badge} {} ({})",
                gate.display_name,
                output::duration_fmt(gate.duration_ms)
            );
        }
        gates.push(gate);
    }

    // ── Summary ────────────────────────────────────────────────────────

    let total_duration = total_start.elapsed();
    let gates_executed = gates.iter().filter(|g| !g.is_skip()).count();
    let gates_skipped = gates.iter().filter(|g| g.is_skip()).count();
    let gates_failed = gates.iter().filter(|g| g.is_failure()).count();
    let passed = gates_failed == 0;

    if json {
        let report = QualifyReport {
            model: file.display().to_string(),
            tier: tier.to_string(),
            passed,
            gates_executed,
            gates_skipped,
            gates_failed,
            total_duration_ms: total_duration.as_millis() as u64,
            gates,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
    } else {
        println!();
        print_summary(&gates, passed, total_duration);
    }

    if passed {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "{gates_failed} qualify gate(s) failed"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_status_serialization() {
        let json = serde_json::to_string(&GateStatus::Pass).expect("serialize");
        assert_eq!(json, "\"PASS\"");
        let json = serde_json::to_string(&GateStatus::Fail).expect("serialize");
        assert_eq!(json, "\"FAIL\"");
        let json = serde_json::to_string(&GateStatus::Skip).expect("serialize");
        assert_eq!(json, "\"SKIP\"");
        let json = serde_json::to_string(&GateStatus::Panic).expect("serialize");
        assert_eq!(json, "\"PANIC\"");
        let json = serde_json::to_string(&GateStatus::Timeout).expect("serialize");
        assert_eq!(json, "\"TIMEOUT\"");
    }

    #[test]
    fn test_gate_result_is_failure() {
        let pass = GateResult {
            name: "test".to_string(),
            display_name: "Test".to_string(),
            status: GateStatus::Pass,
            message: "OK".to_string(),
            duration_ms: 0,
        };
        assert!(!pass.is_failure());

        let fail = GateResult {
            name: "test".to_string(),
            display_name: "Test".to_string(),
            status: GateStatus::Fail,
            message: "bad".to_string(),
            duration_ms: 0,
        };
        assert!(fail.is_failure());

        let panic = GateResult {
            name: "test".to_string(),
            display_name: "Test".to_string(),
            status: GateStatus::Panic,
            message: "PANIC".to_string(),
            duration_ms: 0,
        };
        assert!(panic.is_failure());

        let timeout = GateResult {
            name: "test".to_string(),
            display_name: "Test".to_string(),
            status: GateStatus::Timeout,
            message: "timeout".to_string(),
            duration_ms: 0,
        };
        assert!(timeout.is_failure());
    }

    #[test]
    fn test_gate_result_is_skip() {
        let skip = GateResult {
            name: "test".to_string(),
            display_name: "Test".to_string(),
            status: GateStatus::Skip,
            message: "skipped".to_string(),
            duration_ms: 0,
        };
        assert!(skip.is_skip());

        let pass = GateResult {
            name: "test".to_string(),
            display_name: "Test".to_string(),
            status: GateStatus::Pass,
            message: "OK".to_string(),
            duration_ms: 0,
        };
        assert!(!pass.is_skip());
    }

    #[test]
    fn test_qualify_report_serialization() {
        let report = QualifyReport {
            model: "test.gguf".to_string(),
            tier: "smoke".to_string(),
            passed: true,
            gates_executed: 11,
            gates_skipped: 0,
            gates_failed: 0,
            total_duration_ms: 1234,
            gates: vec![GateResult {
                name: "smoke_inspect".to_string(),
                display_name: "Inspect".to_string(),
                status: GateStatus::Pass,
                message: "OK".to_string(),
                duration_ms: 42,
            }],
        };
        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"passed\": true"));
        assert!(json.contains("\"gates_executed\": 11"));
        assert!(json.contains("\"PASS\""));
    }

    #[test]
    fn test_run_nonexistent_file() {
        let result = run(
            Path::new("/nonexistent/model.gguf"),
            "smoke",
            30,
            true,
            false,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gate_with_panic() {
        let gate = run_gate("panic_test", "Panic Test", 5, false, || {
            panic!("intentional test panic");
        });
        assert_eq!(gate.status, GateStatus::Panic);
        assert!(gate.message.contains("intentional test panic"));
    }

    #[test]
    fn test_run_gate_with_error() {
        let gate = run_gate("error_test", "Error Test", 5, false, || {
            Err(CliError::ValidationFailed("test error".to_string()))
        });
        assert_eq!(gate.status, GateStatus::Fail);
        assert!(gate.message.contains("test error"));
    }

    #[test]
    fn test_run_gate_pass() {
        let gate = run_gate("ok_test", "OK Test", 5, false, || Ok(()));
        assert_eq!(gate.status, GateStatus::Pass);
        assert_eq!(gate.message, "OK");
    }

    #[test]
    fn test_run_external_gate_missing_binary() {
        let gate = run_external_gate(
            "missing",
            "Missing Tool",
            "nonexistent_binary_xyz_42",
            &["arg1"],
            5,
        );
        assert_eq!(gate.status, GateStatus::Skip);
        assert!(gate.message.contains("not on PATH"));
    }

    #[test]
    fn test_smoke_gates_list_has_11_entries() {
        assert_eq!(SMOKE_GATES.len(), 11);
    }

    #[test]
    fn test_skip_set_filters_gates() {
        let skip = vec!["inspect".to_string(), "lint".to_string()];
        let skip_set: HashSet<&str> = skip.iter().map(String::as_str).collect();
        assert!(skip_set.contains("inspect"));
        assert!(skip_set.contains("lint"));
        assert!(!skip_set.contains("validate"));
    }
}
