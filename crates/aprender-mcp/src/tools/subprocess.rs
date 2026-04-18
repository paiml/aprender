//! Shared subprocess wrapper for M2/M3 tools.
//!
//! Every M2 tool spawns `apr <subcommand> [...args] --json` and passes stdout
//! through to the MCP client verbatim. Non-zero exit maps to `isError: true`
//! with stderr attached. This module centralizes that pattern so each tool is
//! a thin definition + a list of CLI args.
//!
//! M3 (FALSIFY-MCP-006) adds [`run_apr_cancellable`], which polls a
//! [`std::sync::mpsc::Receiver`] between `try_wait` checks and escalates to
//! SIGTERM → (grace window) → SIGKILL on the spawned subprocess when a
//! cancellation is signalled. The non-cancellable [`run_apr`] is kept as a
//! thin wrapper for tools that don't support cancellation yet.

use crate::types::ToolCallResult;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

/// Default grace window between SIGTERM and SIGKILL for cancelled calls.
///
/// Per `docs/specifications/apr-mcp-server-spec.md`:
/// > `notifications/cancelled` from client → kill the spawned `apr`
/// > subprocess with SIGTERM (30s grace) → SIGKILL.
pub const CANCEL_GRACE_MS: u64 = 30_000;

/// Poll interval when waiting for subprocess exit / cancel signal.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Spawn `apr <args...>` and wait synchronously. Shorthand for the
/// non-cancellable path used by every tool except `apr.run`.
///
/// - Successful exit with non-empty stdout → `success(stdout)`
/// - Successful exit with empty stdout → `error("apr ... produced no output")`
/// - Non-zero exit → `error("apr ... failed (exit N): <stderr-or-stdout>")`
/// - Spawn failure → `error("Failed to spawn apr ...: <io-err>")`
#[must_use]
pub fn run_apr(args: &[&str]) -> ToolCallResult {
    let output = match Command::new("apr").args(args).output() {
        Ok(o) => o,
        Err(e) => {
            let cmd = format!("apr {}", args.join(" "));
            return ToolCallResult::error(format!("Failed to spawn `{cmd}`: {e}"));
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    if output.status.success() {
        if stdout.trim().is_empty() {
            let cmd = format!("apr {}", args.join(" "));
            ToolCallResult::error(format!("`{cmd}` produced no output"))
        } else {
            ToolCallResult::success(stdout)
        }
    } else {
        let code = output.status.code().unwrap_or(-1);
        let detail = if stderr.trim().is_empty() {
            stdout
        } else {
            stderr
        };
        let cmd = format!("apr {}", args.join(" "));
        ToolCallResult::error(format!("`{cmd}` failed (exit {code}): {detail}"))
    }
}

/// Spawn `apr <args...>` cancellable via `cancel_rx`.
///
/// On receipt of any value on `cancel_rx`, the subprocess is sent SIGTERM.
/// If it hasn't exited within `grace_ms` milliseconds, SIGKILL is sent. The
/// returned `ToolCallResult` carries whatever stdout was captured up to the
/// point of cancellation and has `is_error: Some(true)` with a message that
/// starts with `"Cancelled:"`.
///
/// Non-Unix targets do NOT support signalling; this function falls back to
/// `child.kill()` (equivalent to SIGKILL on Windows).
#[must_use]
pub fn run_apr_cancellable(
    args: &[&str],
    cancel_rx: &Receiver<()>,
    grace_ms: u64,
) -> ToolCallResult {
    spawn_cancellable("apr", args, cancel_rx, grace_ms)
}

/// Test-visible generic over the binary name. `run_apr_cancellable` is the
/// `"apr"`-bound wrapper clients should use in production code.
#[must_use]
pub fn spawn_cancellable(
    program: &str,
    args: &[&str],
    cancel_rx: &Receiver<()>,
    grace_ms: u64,
) -> ToolCallResult {
    let cmd_display = format!("{program} {}", args.join(" "));

    let mut child = match Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return ToolCallResult::error(format!("Failed to spawn `{cmd_display}`: {e}"));
        }
    };

    let pid = child.id();

    // Poll loop: check if the child exited, then check for a cancel signal.
    // Sleep `POLL_INTERVAL` between iterations. This keeps cancel latency
    // under ~10ms while the subprocess is alive.
    let wait_status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(e) => {
                return ToolCallResult::error(format!("Failed to poll `{cmd_display}`: {e}"));
            }
        }

        match cancel_rx.try_recv() {
            Ok(()) => break Err(CancelReason::Signalled),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                // Sender dropped without a cancel — treat as "no cancellation
                // will ever come" and just wait for natural exit.
            }
        }

        std::thread::sleep(POLL_INTERVAL);
    };

    match wait_status {
        Ok(status) => {
            // Natural exit: drain pipes and map to success/error.
            let stdout = drain(&mut child.stdout.take());
            let stderr = drain(&mut child.stderr.take());
            if status.success() {
                if stdout.trim().is_empty() {
                    ToolCallResult::error(format!("`{cmd_display}` produced no output"))
                } else {
                    ToolCallResult::success(stdout)
                }
            } else {
                let code = status.code().unwrap_or(-1);
                let detail = if stderr.trim().is_empty() {
                    stdout
                } else {
                    stderr
                };
                ToolCallResult::error(format!("`{cmd_display}` failed (exit {code}): {detail}"))
            }
        }
        Err(CancelReason::Signalled) => {
            // SIGTERM, grace window, then SIGKILL.
            send_sigterm(pid);
            let deadline = Instant::now() + Duration::from_millis(grace_ms);
            let mut escalated = false;
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => {}
                    Err(_) => break,
                }
                if Instant::now() >= deadline {
                    if !escalated {
                        // Best-effort kill (SIGKILL on Unix, TerminateProcess
                        // on Windows). Ignore errors — the process may have
                        // exited between try_wait and here.
                        let _ = child.kill();
                        escalated = true;
                    } else {
                        // Even SIGKILL hasn't reaped it — give up after a
                        // short extra window to avoid hanging the main
                        // thread forever. In practice SIGKILL is immediate.
                        break;
                    }
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            // Reap if still alive (keeps us from leaking a zombie).
            let _ = child.wait();

            let stdout = drain(&mut child.stdout.take());
            let preview = truncate_for_preview(&stdout);
            ToolCallResult::error(format!(
                "Cancelled: `{cmd_display}` terminated by notifications/cancelled; partial stdout: {preview}"
            ))
        }
    }
}

enum CancelReason {
    Signalled,
}

fn drain<R: Read>(reader: &mut Option<R>) -> String {
    let mut buf = String::new();
    if let Some(r) = reader.as_mut() {
        let _ = r.read_to_string(&mut buf);
    }
    buf
}

fn truncate_for_preview(s: &str) -> String {
    const MAX: usize = 512;
    if s.len() <= MAX {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(MAX).collect();
        format!("{truncated}… (truncated)")
    }
}

#[cfg(unix)]
fn send_sigterm(pid: u32) {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    // Cast is safe: u32 → i32 for PIDs in the valid pid_t range. Values
    // above i32::MAX would be invalid PIDs on every Unix we support, so
    // saturating is fine — `kill` will just fail with EINVAL and we move
    // on to the SIGKILL branch.
    #[allow(clippy::cast_possible_wrap)]
    let raw = pid as i32;
    let _ = kill(Pid::from_raw(raw), Signal::SIGTERM);
}

#[cfg(not(unix))]
fn send_sigterm(_pid: u32) {
    // Windows has no SIGTERM; we skip straight to the SIGKILL equivalent
    // (`child.kill()`) in the escalation path above.
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test helpers
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    /// Spawning `apr` with an unrecognised subcommand yields a tool error
    /// (non-zero exit), not a panic.
    #[test]
    fn spawn_failure_maps_to_tool_error() {
        let result = run_apr(&["this-subcommand-does-not-exist"]);
        assert_eq!(result.is_error, Some(true));
    }

    /// Cancellable path: a never-firing receiver lets the subprocess run to
    /// natural completion, producing identical behaviour to `run_apr`.
    #[test]
    fn cancellable_natural_exit_matches_run_apr() {
        let (_tx, rx) = mpsc::channel::<()>();
        let result = spawn_cancellable("echo", &["hello"], &rx, CANCEL_GRACE_MS);
        assert!(result.is_error.is_none(), "echo should succeed");
        assert!(result.content[0].text.contains("hello"));
    }

    /// Cancellable path: a disconnected receiver (sender dropped) is
    /// equivalent to "no cancellation will arrive" — behaviour should not
    /// change vs the never-firing channel.
    #[test]
    fn cancellable_disconnected_channel_is_noop() {
        let (tx, rx) = mpsc::channel::<()>();
        drop(tx);
        let result = spawn_cancellable("echo", &["world"], &rx, CANCEL_GRACE_MS);
        assert!(result.is_error.is_none());
        assert!(result.content[0].text.contains("world"));
    }

    /// Spawning a missing binary returns a spawn error without panic.
    #[test]
    fn cancellable_spawn_failure_maps_to_error() {
        let (_tx, rx) = mpsc::channel::<()>();
        let result = spawn_cancellable(
            "/this/binary/does/not/exist/apr-mcp-test",
            &[],
            &rx,
            CANCEL_GRACE_MS,
        );
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("Failed to spawn"));
    }

    /// FALSIFY-MCP-006 (unit-level): sending a cancel signal to a
    /// long-running `sleep 60` subprocess returns within the grace window
    /// (SIGTERM is immediate for `sleep`, so we see natural reap well
    /// before the SIGKILL escalation).
    #[test]
    #[cfg(unix)]
    fn cancellable_stops_long_running_subprocess_within_grace() {
        let (tx, rx) = mpsc::channel::<()>();

        // Fire the cancel shortly after spawn to give the subprocess time
        // to get into its sleep syscall.
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            let _ = tx.send(());
        });

        let t0 = Instant::now();
        // Grace of 2s: if SIGTERM fails for any reason we still fall back
        // to SIGKILL well before the test's own timeout.
        let result = spawn_cancellable("sleep", &["60"], &rx, 2_000);
        let elapsed = t0.elapsed();

        handle.join().expect("cancel-sender thread joins");

        assert_eq!(result.is_error, Some(true), "cancelled calls are errors");
        assert!(
            result.content[0].text.starts_with("Cancelled:"),
            "message should indicate cancellation, got: {}",
            result.content[0].text
        );
        // 100ms fire + ~immediate SIGTERM response + cleanup — well under
        // the 2s grace + 200ms test slack the spec calls for.
        assert!(
            elapsed < Duration::from_millis(2_500),
            "cancel should finish within grace + slack, took {elapsed:?}"
        );
        // And it must finish meaningfully faster than the underlying
        // `sleep 60` would have — this is the real falsification.
        assert!(
            elapsed < Duration::from_secs(5),
            "cancelled call must return far before sleep 60's natural exit"
        );
    }
}
