//! `apr.serve` — background subprocess wrapper over `apr serve run`.
//!
//! Unlike every other Phase-1 tool, this one does NOT wait for the subprocess
//! to exit: `apr serve run` is a long-running HTTP daemon. We spawn it, wait
//! for it to actually bind its port, then return `{pid, url, ready}` so the
//! MCP client can reach the daemon. The caller is responsible for killing the
//! pid out-of-band.
//!
//! DOGFOOD-0.63.0 (#2388) — two defects were fixed here at once:
//!
//! 1. The spawned argv was `apr serve <model> --port <n>`, which is not a
//!    valid CLI form: `apr serve` takes a `plan`/`run` subcommand, so every
//!    invocation died instantly with `error: unrecognized subcommand` (rc=2).
//!    The correct form is `apr serve run <model> --port <n>`; it is built by
//!    [`serve_argv`] and asserted by a unit test so it cannot silently drift.
//! 2. The `Child` handle was dropped without ever checking liveness, so
//!    `is_error` was left `None` and the tool reported `{pid, url}` — a
//!    success — for a process that was already a zombie, including for a
//!    model path that does not exist. Silent success on failure.
//!
//! [`spawn_and_confirm`] now polls until one of three terminal states:
//! the child exits (→ `isError` with the exit status and the tail of its
//! stderr), the port accepts a TCP connection (→ success, `ready: true`), or
//! the readiness window elapses with the child still alive (→ non-error, but
//! explicitly `ready: false` — a big model may still be loading; we report
//! what is true rather than claiming a listening server).
//!
//! M3 shipped `notifications/cancelled` → SIGTERM → SIGKILL for `apr.run`
//! only (see `server.rs::CancelHandle` docs: "Only `apr.run` currently
//! honours cancellation"). A lifecycle-tracked registry for `apr.serve` —
//! cancel token → SIGTERM the captured pid with 30s grace → SIGKILL — is a
//! post-M3 follow-up targeted at M5 alongside the pmcp dispatcher port (see
//! `docs/specifications/apr-mcp-server-spec.md` § Milestones → M5).
//! Until then, a daemon that is still alive when we return leaves a zombie on
//! Unix until the OS parent reaps it. A daemon that died inside the readiness
//! window IS reaped, by the `try_wait` that detects the death.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::types::{ContentBlock, InputSchema, ToolCallResult, ToolDefinition};
use std::io::Read;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.serve";

/// Default HTTP port when the caller omits `port`.
const DEFAULT_PORT: u16 = 8080;

/// How long to wait for the spawned daemon to bind its port before returning
/// a non-error `ready: false`. Generous enough for a multi-GB model load on
/// CPU; an argv/clap failure is detected in the first poll (~50ms).
const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// Poll interval for the readiness loop.
const READY_POLL: Duration = Duration::from_millis(50);

/// Per-probe TCP connect timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(200);

/// Readiness window when `port == 0` (OS-assigned): there is no port to probe,
/// so we can only prove the child did not die immediately.
const EPHEMERAL_PORT_WINDOW: Duration = Duration::from_secs(2);

/// Bytes of the child's stderr echoed back in a failure message.
const STDERR_TAIL_BYTES: usize = 2048;

/// The exact CLI form `apr.serve` shells out to.
///
/// `apr serve` is a command *group* — `plan` and `run` are its subcommands.
/// Omitting `run` makes the model path parse as a subcommand name and clap
/// exits 2 before the server is ever constructed (#2388).
#[must_use]
pub fn serve_argv(model_path: &str, port: u16) -> Vec<String> {
    vec![
        "serve".to_string(),
        "run".to_string(),
        model_path.to_string(),
        "--port".to_string(),
        port.to_string(),
    ]
}

/// Return the MCP tool definition for `apr.serve`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_SERVE_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn serve_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_SERVE_SCHEMA).expect(
        "FALSIFY-MCP-008: apr.serve codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_SERVE_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.serve` by spawning `apr serve run <model_path> --port <port>`
/// and confirming the daemon is alive before reporting success.
///
/// Returns `isError: true` when the child exits inside the readiness window —
/// which is what a bad argv, a missing model file, or an already-bound port
/// all look like. See this module's header for the terminal states.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let model_path = match crate::tools::args::require_str(args, "model_path") {
        Ok(p) => p,
        Err(e) => return e,
    };

    let port: u16 = match args.get("port") {
        None => DEFAULT_PORT,
        Some(v) => match v.as_u64().and_then(|n| u16::try_from(n).ok()) {
            Some(n) => n,
            None => {
                return ToolCallResult::error(format!(
                    "Invalid port: expected integer 0..=65535, got {v}"
                ));
            }
        },
    };

    spawn_and_confirm("apr", &serve_argv(model_path, port), port, READY_TIMEOUT)
}

/// Spawn `program <args...>` as a background HTTP daemon on `port` and wait
/// up to `ready_timeout` for it to prove it is running.
///
/// Generic over the program name so the readiness/liveness contract can be
/// exercised in unit tests without a built `apr` on `PATH` — the same shape
/// `subprocess::spawn_cancellable` uses.
#[must_use]
pub fn spawn_and_confirm(
    program: &str,
    args: &[String],
    port: u16,
    ready_timeout: Duration,
) -> ToolCallResult {
    let cmd_display = format!("{program} {}", args.join(" "));

    // The daemon outlives this call, so its stderr must not go to a pipe
    // nobody drains (a full pipe buffer would wedge the server). Route it to
    // a file we can read back if the child dies, and hand the path to the
    // caller when it lives.
    let log_path = stderr_log_path(port);
    let stderr = match std::fs::File::create(&log_path) {
        Ok(f) => Stdio::from(f),
        Err(_) => Stdio::null(),
    };

    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_file(&log_path);
            return ToolCallResult::error(format!("failed to spawn `{cmd_display}`: {e}"));
        }
    };

    let pid: u32 = child.id();
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    // Port 0 means "OS assigns" — there is nothing to probe, so the window
    // only proves the child survived startup.
    let window = if port == 0 {
        ready_timeout.min(EPHEMERAL_PORT_WINDOW)
    } else {
        ready_timeout
    };
    let deadline = Instant::now() + window;

    loop {
        // Liveness first: a dead child can never be ready, and checking it
        // before the connect probe stops an unrelated process that happens to
        // hold `port` from masquerading as our daemon.
        match child.try_wait() {
            Ok(Some(status)) => {
                let detail = read_stderr_tail(&log_path);
                let _ = std::fs::remove_file(&log_path);
                let code = status
                    .code()
                    .map_or_else(|| "signal".to_string(), |c| c.to_string());
                return ToolCallResult::error(format!(
                    "`{cmd_display}` exited immediately (status {code}) — no server is \
                     listening on port {port}{detail}"
                ));
            }
            Ok(None) => {}
            Err(e) => {
                let _ = std::fs::remove_file(&log_path);
                return ToolCallResult::error(format!("failed to poll `{cmd_display}`: {e}"));
            }
        }

        if port != 0 && TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).is_ok() {
            // Re-check liveness: the connect could have raced a dying child.
            if let Ok(Some(status)) = child.try_wait() {
                let detail = read_stderr_tail(&log_path);
                let _ = std::fs::remove_file(&log_path);
                let code = status
                    .code()
                    .map_or_else(|| "signal".to_string(), |c| c.to_string());
                return ToolCallResult::error(format!(
                    "`{cmd_display}` exited immediately (status {code}) — no server is \
                     listening on port {port}{detail}"
                ));
            }
            return running_result(pid, port, true, &log_path);
        }

        if Instant::now() >= deadline {
            return running_result(pid, port, false, &log_path);
        }
        std::thread::sleep(READY_POLL);
    }
}

/// Build the non-error payload for a child that is still running.
///
/// `ready` distinguishes "the port accepted a connection" from "still alive
/// but not listening yet" — the caller is told which, never told a listening
/// server exists when none does.
fn running_result(pid: u32, port: u16, ready: bool, log_path: &std::path::Path) -> ToolCallResult {
    let note = if ready {
        "server is accepting connections; kill pid via OS to stop"
    } else {
        "process is alive but has not bound the port yet (still loading?); kill pid via OS to stop"
    };
    let payload = serde_json::json!({
        "pid": pid,
        "url": format!("http://localhost:{port}"),
        "ready": ready,
        "stderr_log": log_path.display().to_string(),
        "note": note,
    });
    let text = serde_json::to_string(&payload)
        .unwrap_or_else(|_| format!("{{\"pid\":{pid},\"url\":\"http://localhost:{port}\"}}"));
    ToolCallResult {
        content: vec![ContentBlock::text(text)],
        is_error: None,
    }
}

/// Unique-per-call path for the daemon's stderr.
fn stderr_log_path(port: u16) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    std::env::temp_dir().join(format!(
        "apr-mcp-serve-{pid}-{port}-{nanos}.log",
        pid = std::process::id()
    ))
}

/// Last [`STDERR_TAIL_BYTES`] of the child's stderr, formatted for appending
/// to an error message. Empty string when there is nothing to report.
fn read_stderr_tail(log_path: &std::path::Path) -> String {
    let Ok(mut f) = std::fs::File::open(log_path) else {
        return String::new();
    };
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return String::new();
    }
    let start = buf.len().saturating_sub(STDERR_TAIL_BYTES);
    let tail = String::from_utf8_lossy(&buf[start..]).trim().to_string();
    if tail.is_empty() {
        String::new()
    } else {
        format!(": {tail}")
    }
}

/// HELIX-IDEA-002 — unified-signature shim for the inventory dispatcher.
pub fn dispatch(
    args: &serde_json::Value,
    _cancel: &std::sync::mpsc::Receiver<()>,
    _sink: Option<&crate::server::NotificationSink>,
    _token: Option<serde_json::Value>,
) -> ToolCallResult {
    call(args)
}

crate::register_mcp_tool!(
    name: NAME,
    definition: serve_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = serve_tool_definition();
        assert_eq!(def.name, "apr.serve");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "port"] {
            assert!(
                def.input_schema.properties.contains_key(field),
                "{field} property present"
            );
        }
    }

    /// Missing `model_path` must return `isError: true` with the offending
    /// field name — mirrors FALSIFY-MCP-VALIDATE-001. This is the only unit
    /// test we can run without spawning a real `apr serve` daemon.
    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(
            result.content[0].text.contains("model_path"),
            "error message must mention model_path, got: {}",
            result.content[0].text
        );
    }

    #[test]
    fn nonstring_model_path_returns_error() {
        let result = call(&serde_json::json!({ "model_path": 42 }));
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn out_of_range_port_returns_error() {
        let result = call(&serde_json::json!({
            "model_path": "/tmp/x.apr",
            "port": 99999
        }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("port"));
    }

    // ---- DOGFOOD-0.63.0 (#2388) falsifiers -------------------------------

    /// The argv defect: 0.63.0 built `apr serve <model> --port N`, so clap
    /// parsed the model path as a subcommand name and exited 2. `serve` is a
    /// command group; `run` is mandatory and must sit between them.
    #[test]
    fn serve_argv_places_run_between_serve_and_model_path() {
        let argv = serve_argv("/models/qwen.gguf", 18590);
        assert_eq!(
            argv,
            vec!["serve", "run", "/models/qwen.gguf", "--port", "18590"],
            "apr.serve must shell out to `apr serve run <model> --port <n>`"
        );
        // Position matters, not mere presence: the model path must never be
        // in the subcommand slot.
        assert_eq!(argv[1], "run");
        assert_ne!(argv[1], "/models/qwen.gguf");
    }

    /// Pick a port nothing is listening on. Binding and immediately dropping
    /// leaves the port free with high probability, which is enough for a test
    /// whose assertion is about the child, not the port.
    #[cfg(unix)]
    fn free_port() -> u16 {
        let l = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .expect("bind ephemeral port");
        l.local_addr().expect("local_addr").port()
    }

    #[cfg(unix)]
    fn payload_of(result: &ToolCallResult) -> serde_json::Value {
        serde_json::from_str(&result.content[0].text)
            .unwrap_or_else(|e| panic!("payload must be JSON: {e}; got {}", result.content[0].text))
    }

    /// The silent-success defect: 0.63.0 dropped the `Child` without checking
    /// liveness, so a subprocess that had already exited non-zero was still
    /// reported as `{pid, url}` with no `isError`.
    #[cfg(unix)]
    #[test]
    fn child_that_exits_immediately_is_reported_as_error() {
        let port = free_port();
        let result = spawn_and_confirm("false", &[], port, Duration::from_secs(5));
        assert_eq!(
            result.is_error,
            Some(true),
            "a child that exited must NOT be reported as a running server; got: {}",
            result.content[0].text
        );
        let msg = &result.content[0].text;
        assert!(
            msg.contains("exited immediately"),
            "error must say the child exited, got: {msg}"
        );
        assert!(
            msg.contains(&port.to_string()),
            "error must name the port that has no server, got: {msg}"
        );
    }

    /// A failing child's own diagnostics must survive into the MCP error —
    /// this is what turns "it didn't work" into "unrecognized subcommand".
    #[cfg(unix)]
    #[test]
    fn dead_child_error_carries_its_stderr_and_exit_status() {
        let port = free_port();
        let args = vec![
            "-c".to_string(),
            "echo 'unrecognized subcommand' >&2; exit 2".to_string(),
        ];
        let result = spawn_and_confirm("sh", &args, port, Duration::from_secs(5));
        assert_eq!(result.is_error, Some(true));
        let msg = &result.content[0].text;
        assert!(
            msg.contains("unrecognized subcommand"),
            "child stderr must be echoed back, got: {msg}"
        );
        assert!(
            msg.contains("status 2"),
            "exit status must be reported, got: {msg}"
        );
    }

    /// Success path: a live child plus a port that accepts connections is the
    /// only combination that may report `ready: true`.
    #[cfg(unix)]
    #[test]
    fn live_child_with_bound_port_reports_ready_true() {
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local_addr").port();
        let args = vec!["5".to_string()];
        let result = spawn_and_confirm("sleep", &args, port, Duration::from_secs(5));
        assert_eq!(result.is_error, None, "got: {}", result.content[0].text);
        let payload = payload_of(&result);
        assert_eq!(payload["ready"], serde_json::json!(true));
        assert_eq!(
            payload["url"],
            serde_json::json!(format!("http://localhost:{port}"))
        );
        assert!(payload["pid"].as_u64().is_some_and(|p| p > 0));
        // The log path handed to the client must actually exist.
        let log = payload["stderr_log"]
            .as_str()
            .expect("stderr_log is a path");
        assert!(
            std::path::Path::new(log).exists(),
            "stderr_log {log} must exist for a running daemon"
        );
        let _ = std::fs::remove_file(log);
        drop(listener);
    }

    /// A live child that never binds is neither a failure nor a running
    /// server: report it truthfully as `ready: false` rather than implying a
    /// reachable URL.
    #[cfg(unix)]
    #[test]
    fn live_child_that_never_binds_reports_ready_false() {
        let port = free_port();
        let args = vec!["5".to_string()];
        let result = spawn_and_confirm("sleep", &args, port, Duration::from_millis(300));
        assert_eq!(result.is_error, None);
        let payload = payload_of(&result);
        assert_eq!(
            payload["ready"],
            serde_json::json!(false),
            "must not claim readiness for a port nothing is listening on"
        );
        assert!(
            payload["note"]
                .as_str()
                .is_some_and(|n| n.contains("has not bound")),
            "note must explain the port is unbound, got: {}",
            payload["note"]
        );
        if let Some(log) = payload["stderr_log"].as_str() {
            let _ = std::fs::remove_file(log);
        }
    }

    /// Spawning a program that does not exist is an error, not a `{pid,url}`.
    #[cfg(unix)]
    #[test]
    fn unspawnable_program_reports_error() {
        let port = free_port();
        let result = spawn_and_confirm(
            "apr-does-not-exist-9c1f",
            &[],
            port,
            Duration::from_millis(200),
        );
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("failed to spawn"));
    }
}
