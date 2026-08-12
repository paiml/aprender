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
//!
//! #2418: the failure path used to pick *either* stderr *or* stdout — stdout
//! only when stderr was empty. `apr qa` writes its JSON gate report to stdout
//! and a one-line summary to stderr, so every failing QA run (the tool's
//! primary use case) reached the client as a single line with the >3 KB
//! report thrown away. [`failure_result`] now keeps the summary as the first
//! content block and attaches the report as a second one, so a failing gate
//! is as inspectable as a passing one.

use crate::apr_bin::apr_binary;
use crate::types::{ContentBlock, ToolCallResult};
use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Read};
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

/// Render one argv element so the echoed command can actually be re-run.
///
/// `format!("apr {}", args.join(" "))` produced `--prompt What is 2+2?`, which
/// is a different command from the one that ran (#2403). Anything outside the
/// POSIX-safe set is single-quoted, with embedded quotes escaped the shell way.
fn quote_arg(arg: &str) -> String {
    let safe = !arg.is_empty()
        && arg
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"@%+=:,./-_".contains(&b));
    if safe {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', r"'\''"))
    }
}

/// Build the `isError` result for a subprocess that exited non-zero.
///
/// The first content block is the one-line summary the client already relied
/// on. When the command wrote to BOTH streams the stdout payload is attached
/// as a second block instead of being discarded (#2418).
fn failure_result(cmd_display: &str, code: i32, stdout: &str, stderr: &str) -> ToolCallResult {
    let summary = if stderr.trim().is_empty() {
        stdout.to_string()
    } else {
        stderr.to_string()
    };
    let mut content = vec![ContentBlock::text(format!(
        "`{cmd_display}` failed (exit {code}): {summary}"
    ))];
    if !stderr.trim().is_empty() && !stdout.trim().is_empty() {
        content.push(ContentBlock::text(stdout.to_string()));
    }
    ToolCallResult {
        content,
        is_error: Some(true),
    }
}

/// Spawn `apr <args...>` and wait synchronously. Shorthand for the
/// non-cancellable path used by every tool except `apr.run`.
///
/// - Successful exit with non-empty stdout → `success(stdout)`
/// - Successful exit with empty stdout → `error("apr ... produced no output")`
/// - Non-zero exit → `error("apr ... failed (exit N): <stderr-or-stdout>")`
/// - Spawn failure → `error("Failed to spawn apr ...: <io-err>")`
#[must_use]
pub fn run_apr(args: &[&str]) -> ToolCallResult {
    run_program(apr_binary(), args)
}

/// Generic-over-program variant of [`run_apr`]. [`run_apr`] binds `program`
/// to [`crate::apr_bin::apr_binary`] — the running `apr` executable — so the
/// version the user launched is the version that answers.
#[must_use]
pub fn run_program<P: AsRef<OsStr>>(program: P, args: &[&str]) -> ToolCallResult {
    let program = program.as_ref();
    let cmd_display = display_cmd(program, args);
    let output = match Command::new(program).args(args).output() {
        Ok(o) => o,
        Err(e) => {
            return ToolCallResult::error(format!("Failed to spawn `{cmd_display}`: {e}"));
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    if output.status.success() {
        if stdout.trim().is_empty() {
            ToolCallResult::error(format!("`{cmd_display}` produced no output"))
        } else {
            ToolCallResult::success(stdout)
        }
    } else {
        let code = output.status.code().unwrap_or(-1);
        failure_result(&cmd_display, code, &stdout, &stderr)
    }
}

/// Render `program args...` for user-facing error messages, quoted so the echoed
/// command can actually be re-run.
///
/// This used to be `args.join(" ")`, which is the #2403 defect: `--prompt What is
/// 2+2?` is a DIFFERENT command from the one that ran, and a user copying it out
/// of an error message gets a different failure than the one being reported.
///
/// The merge that produced this file kept the `OsStr` signature (every call site
/// passes `apr_binary()`, an OsStr) but had dropped the quoting, leaving
/// `quote_arg` orphaned — clippy's dead-code error is what surfaced the lost fix.
fn display_cmd(program: &OsStr, args: &[&str]) -> String {
    let mut out = quote_arg(&program.to_string_lossy());
    for a in args {
        out.push(' ');
        out.push_str(&quote_arg(a));
    }
    out
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
    spawn_cancellable(apr_binary(), args, cancel_rx, grace_ms)
}

/// Generic over the binary. `run_apr_cancellable` binds `program` to
/// [`crate::apr_bin::apr_binary`] — the running `apr` executable — which is
/// what production code should use.
#[must_use]
pub fn spawn_cancellable<P: AsRef<OsStr>>(
    program: P,
    args: &[&str],
    cancel_rx: &Receiver<()>,
    grace_ms: u64,
) -> ToolCallResult {
    let program = program.as_ref();
    let cmd_display = display_cmd(program, args);

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
                failure_result(&cmd_display, code, &stdout, &stderr)
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

/// Spawn `apr <args...>` and stream stdout line-by-line to `on_line`.
///
/// FALSIFY-MCP-PROGRESS-001: this is the streaming variant used by tools that
/// emit `notifications/progress` (currently `apr.finetune`). Each line of
/// stdout (as written by `apr <cmd> --json`) is passed to `on_line`
/// synchronously before the next `read_line` — the caller is responsible for
/// emitting the notification.
///
/// Returns a `ToolCallResult` whose body is the concatenated stdout (the same
/// shape as [`run_apr`] would have produced), so callers can keep the
/// existing "final payload" semantics while layering progress on top.
#[must_use]
pub fn run_apr_streaming<F>(args: &[&str], on_line: F) -> ToolCallResult
where
    F: FnMut(&str),
{
    spawn_streaming(apr_binary(), args, on_line)
}

/// Generic-over-program variant of [`run_apr_streaming`]. Production callers
/// pass [`crate::apr_bin::apr_binary`]; tests use it to inject a mock.
#[must_use]
pub fn spawn_streaming<P: AsRef<OsStr>, F>(
    program: P,
    args: &[&str],
    mut on_line: F,
) -> ToolCallResult
where
    F: FnMut(&str),
{
    let program = program.as_ref();
    let cmd_display = display_cmd(program, args);

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

    // Take stdout so we can wrap it in a BufReader. Leaving stderr attached
    // to the child means we can drain it after wait() for the error path.
    let stdout_pipe = match child.stdout.take() {
        Some(p) => p,
        None => {
            let _ = child.wait();
            return ToolCallResult::error(format!("Failed to capture stdout of `{cmd_display}`"));
        }
    };

    let mut accumulated = String::new();
    let reader = BufReader::new(stdout_pipe);
    for line in reader.lines() {
        match line {
            Ok(text) => {
                on_line(&text);
                accumulated.push_str(&text);
                accumulated.push('\n');
            }
            Err(e) => {
                // Best-effort: surface the read error but still try to reap
                // the child so we don't leak a zombie.
                let _ = child.wait();
                return ToolCallResult::error(format!(
                    "Failed to read stdout of `{cmd_display}`: {e}"
                ));
            }
        }
    }

    // stdout closed → subprocess is either exited or about to. Wait for the
    // exit status so we can map success/failure correctly.
    let status = match child.wait() {
        Ok(s) => s,
        Err(e) => {
            return ToolCallResult::error(format!("Failed to reap `{cmd_display}`: {e}"));
        }
    };

    let stderr = drain(&mut child.stderr.take());

    if status.success() {
        if accumulated.trim().is_empty() {
            ToolCallResult::error(format!("`{cmd_display}` produced no output"))
        } else {
            ToolCallResult::success(accumulated)
        }
    } else {
        let code = status.code().unwrap_or(-1);
        let detail = if stderr.trim().is_empty() {
            accumulated
        } else {
            stderr
        };
        ToolCallResult::error(format!("`{cmd_display}` failed (exit {code}): {detail}"))
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test helpers
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    /// #2418 — a failing `apr qa` writes its JSON gate report to stdout and a
    /// one-line summary to stderr. The report is the whole point of the tool;
    /// it must survive the failure path, not be replaced by the summary.
    #[test]
    fn failure_keeps_the_stdout_report_when_stderr_also_spoke() {
        let report = r#"{"passed":false,"gates":[{"name":"ollama_parity","passed":false}]}"#;
        let result = failure_result(
            "apr qa m.gguf --json",
            5,
            report,
            "error: Validation failed",
        );

        assert_eq!(result.is_error, Some(true));
        let whole: String = result
            .content
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            whole.contains("ollama_parity"),
            "gate report must reach the client, got: {whole}"
        );
        assert!(
            whole.contains("failed (exit 5)"),
            "summary line must survive too, got: {whole}"
        );
    }

    /// End-to-end through a real subprocess that writes to BOTH streams and
    /// exits non-zero — the exact shape of a failing `apr qa`.
    ///
    /// The report is asserted on the SECOND content block specifically. The
    /// first block echoes the command, and the command here IS the script
    /// text, so an `any()` over all blocks passed even with the report
    /// discarded — that is how the first draft of this test survived its own
    /// mutation check.
    #[test]
    fn cancellable_failure_carries_both_streams() {
        let (_tx, rx) = mpsc::channel::<()>();
        let result = spawn_cancellable(
            "sh",
            &[
                "-c",
                "printf '{\"gates\":\"REP\"}\\nORT\\n'; echo SUMMARY >&2; exit 5",
            ],
            &rx,
            CANCEL_GRACE_MS,
        );
        assert_eq!(result.is_error, Some(true));
        assert!(
            result.content[0].text.contains("SUMMARY"),
            "stderr dropped: {}",
            result.content[0].text
        );
        assert_eq!(
            result.content.len(),
            2,
            "stdout report dropped, only got: {:?}",
            result.content
        );
        assert!(
            result.content[1].text.contains("{\"gates\":\"REP\"}\nORT"),
            "stdout report mangled: {}",
            result.content[1].text
        );
    }

    /// A failure with nothing on stderr still reports stdout in the summary,
    /// and does not emit a redundant duplicate block.
    #[test]
    fn failure_with_empty_stderr_reports_stdout_once() {
        let result = failure_result("apr qa m.gguf", 1, "only-stdout", "   \n");
        assert_eq!(result.content.len(), 1);
        assert!(result.content[0].text.contains("only-stdout"));
    }

    /// #2403 (secondary) — the echoed reproduction command must be
    /// copy-pasteable. `--prompt What is 2+2?` is a different command from the
    /// one that ran.
    #[test]
    fn echoed_command_is_shell_quoted() {
        let cmd = display_cmd(
            OsStr::new("apr"),
            &["run", "m.gguf", "--prompt", "What is 2+2?"],
        );
        assert_eq!(cmd, "apr run m.gguf --prompt 'What is 2+2?'");
    }

    /// Quoting must be idempotent for safe argv elements (no needless noise)
    /// and must survive an embedded single quote.
    #[test]
    fn quoting_leaves_safe_args_alone_and_escapes_quotes() {
        assert_eq!(quote_arg("--max-tokens"), "--max-tokens");
        assert_eq!(
            quote_arg("/home/noah/models/a.gguf"),
            "/home/noah/models/a.gguf"
        );
        assert_eq!(quote_arg(""), "''");
        assert_eq!(quote_arg("it's"), r"'it'\''s'");
    }

    /// Spawning `apr` with an unrecognised subcommand yields a tool error
    /// (non-zero exit), not a panic.
    #[test]
    fn spawn_failure_maps_to_tool_error() {
        let result = run_apr(&["this-subcommand-does-not-exist"]);
        assert_eq!(result.is_error, Some(true));
    }

    /// FALSIFIER (#2384): `run_apr` must execute the *resolved* `apr` binary,
    /// not a hard-coded `Command::new("apr")` that the OS looks up on `$PATH`.
    ///
    /// The field defect: `apr mcp` shipped in 0.63.0 returned results produced
    /// by a 0.60.0 binary that happened to be first on `$PATH`, while
    /// `apr.version` kept answering 0.63.0.
    ///
    /// Here we designate a specific binary via `$APR_BIN` and assert its
    /// marker output comes back as the tool payload. Before the fix, `run_apr`
    /// ignored resolution entirely and this returned whatever `apr` `$PATH`
    /// produced — never the marker.
    ///
    /// The shim mirrors the real CLI's exit semantics (unknown subcommand →
    /// exit 2) so it is compatible with `spawn_failure_maps_to_tool_error`
    /// should the two overlap while `$APR_BIN` is set.
    #[test]
    #[cfg(unix)]
    fn falsify_2384_run_apr_executes_the_resolved_binary() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        // Unique per process: a fixed path lets two concurrent runs of this
        // test binary delete each other's shim mid-flight.
        let dir =
            std::env::temp_dir().join(format!("aprender-mcp-2384-run-apr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir scratch");
        let shim = dir.join("apr");
        {
            let mut f = std::fs::File::create(&shim).expect("create shim");
            writeln!(f, "#!/bin/sh").expect("shebang");
            writeln!(f, "if [ \"$1\" = \"validate\" ]; then").expect("if");
            writeln!(f, "  echo '{{\"marker\":\"APR-BIN-RESOLVED-SHIM\"}}'").expect("body");
            writeln!(f, "  exit 0").expect("ok");
            writeln!(f, "fi").expect("fi");
            writeln!(f, "exit 2").expect("unknown subcommand");
            f.sync_all().expect("sync");
        }
        let mut perms = std::fs::metadata(&shim).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&shim, perms).expect("chmod");

        // Edition 2021 — `set_var` is safe here.
        std::env::set_var(crate::apr_bin::APR_BIN_ENV, &shim);
        let result = run_apr(&["validate", "/dev/null", "--json"]);
        std::env::remove_var(crate::apr_bin::APR_BIN_ENV);

        assert!(
            result.is_error.is_none(),
            "resolved shim should succeed, got: {}",
            result.content[0].text
        );
        assert!(
            result.content[0].text.contains("APR-BIN-RESOLVED-SHIM"),
            "run_apr must execute the resolved binary; got: {}",
            result.content[0].text
        );
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

    /// FALSIFY-MCP-PROGRESS-001 (unit): `spawn_streaming` fires the callback
    /// once per stdout line before returning the aggregated payload.
    #[test]
    fn streaming_invokes_callback_per_line() {
        let lines = std::sync::Mutex::new(Vec::<String>::new());
        let result = spawn_streaming("printf", &["line1\nline2\nline3\n"], |line| {
            lines
                .lock()
                .expect("test mutex not poisoned")
                .push(line.to_string());
        });
        assert!(result.is_error.is_none(), "printf should succeed");

        let captured = lines.lock().expect("mutex").clone();
        assert_eq!(captured, vec!["line1", "line2", "line3"]);
        assert!(result.content[0].text.contains("line1"));
        assert!(result.content[0].text.contains("line3"));
    }

    /// Spawn failure in the streaming path returns a tool error without
    /// invoking the callback.
    #[test]
    fn streaming_spawn_failure_does_not_call_callback() {
        let called = std::sync::Mutex::new(false);
        let result = spawn_streaming(
            "/this/binary/does/not/exist/apr-mcp-streaming-test",
            &[],
            |_| {
                *called.lock().expect("mutex") = true;
            },
        );
        assert_eq!(result.is_error, Some(true));
        assert!(!*called.lock().expect("mutex"));
        assert!(result.content[0].text.contains("Failed to spawn"));
    }

    /// Streaming path: non-zero exit surfaces as a tool error.
    #[test]
    #[cfg(unix)]
    fn streaming_nonzero_exit_is_error() {
        let result = spawn_streaming("sh", &["-c", "echo partial; exit 3"], |_| {});
        assert_eq!(result.is_error, Some(true));
        assert!(
            result.content[0].text.contains("exit 3"),
            "message should include exit code: {}",
            result.content[0].text
        );
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
