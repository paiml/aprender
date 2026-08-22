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
//! DOGFOOD-0.64.0 (#2606) — the published 0.63.0 binary still exhibited both
//! defects above (the fixes landed on `main` after the 0.63.0 cut), and the
//! sweep surfaced a third, narrower one plus a hole in how (1) was guarded:
//!
//! 3. `ready: false` — the "alive but not listening" branch — still emitted
//!    `"url": "http://localhost:<port>"`. A client that reads the field it
//!    asked for and ignores the annotation is told a URL exists for a port
//!    nothing is bound to. [`running_result`] now emits `url` **only** on the
//!    branch that observed a successful TCP connect with the child alive.
//! 4. The only guard on (1) was a unit test comparing [`serve_argv`]'s output
//!    to a hard-coded vector. That is tautological with the implementation:
//!    it re-states the argv rather than checking anything accepts it, so a
//!    rename of the `run` subcommand would keep it green while the tool broke
//!    exactly as 0.63.0 did. The argv is now fed to the CLI's own clap parser
//!    — the surface that emitted `rc=2, unrecognized subcommand` — by
//!    `apr-cli/src/lib_falsify_2606_mcp_serve_argv.rs`. `apr-cli` depends on
//!    this crate, so the check lives on that side of the edge.
//!
//! The generalizing invariant, the one that survives the next argv change:
//! **`apr.serve` reports a URL only for a port it watched accept a connection
//! while the child was still alive.** A pid proves a process was created, not
//! that it survived argv parsing.
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

use crate::tools::args::{self, try_arg};
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
    let model_path = try_arg!(args::required_str(args, "model_path"));

    let port: u16 = match try_arg!(args::opt_u64(args, "port")) {
        None => DEFAULT_PORT,
        Some(n) => match u16::try_from(n) {
            Ok(p) => p,
            Err(_) => {
                return ToolCallResult::error(format!(
                    "Invalid port: expected integer 0..=65535, got {n}"
                ));
            }
        },
    };

    // aprender#2563: a bare "apr" here is resolved through $PATH, so this tool
    // spawns whatever `apr` the user happens to have installed -- which during the
    // 0.63.0 dogfood was a 26-day-old 0.60.0. The MCP server then reports results
    // for code it is not running. apr_binary() is the resolver that already exists
    // for exactly this, and every other tool in this crate uses it.
    spawn_and_confirm(
        &crate::apr_bin::apr_binary().to_string_lossy(),
        &serve_argv(model_path, port),
        port,
        READY_TIMEOUT,
    )
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
///
/// #2606 — INVARIANT SERVE-URL-ONLY-WHEN-LISTENING. The `url` key is emitted
/// **only** when a TCP connect to `port` succeeded while the child was still
/// alive. `ready: false` is not a soft "probably fine": until #2606 this
/// branch still shipped `"url": "http://localhost:<port>"`, so an MCP client
/// that reads `url` (the field it asked for) and ignores `ready` was handed a
/// reachable-looking endpoint that nothing was listening on — the same
/// fabricated-success shape as the pre-#2388 `{pid, url}`, merely annotated.
/// Withholding the key makes the fabrication unrepresentable rather than
/// merely discouraged: there is no url to misread.
///
/// Still-alive-but-unbound stays a NON-error deliberately. A multi-GB model
/// load can outlast [`READY_TIMEOUT`], the child is a real process the caller
/// owns and must kill, and reporting `isError` for a healthy loader would be
/// its own false claim. The honest report is: pid yes, url no, ready false.
fn running_result(pid: u32, port: u16, ready: bool, log_path: &std::path::Path) -> ToolCallResult {
    let note = if ready {
        "server is accepting connections; kill pid via OS to stop"
    } else {
        "process is alive but has not bound the port yet (still loading?); no URL is \
         reported because nothing accepted a connection on the port; kill pid via OS to stop"
    };
    let mut payload = serde_json::json!({
        "pid": pid,
        "ready": ready,
        "port": port,
        "stderr_log": log_path.display().to_string(),
        "note": note,
    });
    if ready {
        // Only reachable after a successful connect + a live-child recheck.
        payload["url"] = serde_json::json!(format!("http://localhost:{port}"));
    }
    let text = serde_json::to_string(&payload)
        .unwrap_or_else(|_| format!("{{\"pid\":{pid},\"ready\":{ready},\"port\":{port}}}"));
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

    // ---- DOGFOOD-0.64.0 (#2606) falsifiers -------------------------------
    //
    // INVARIANT SERVE-URL-ONLY-WHEN-LISTENING: `apr.serve` may put a `url` in
    // its payload only on the branch that watched a TCP connect succeed while
    // the child was still alive. Every other terminal state — dead child,
    // unspawnable program, alive-but-unbound, ephemeral port — must carry NO
    // url. This is the property that survives the next argv change: it never
    // mentions `serve`, `run`, or any flag, so it keeps its teeth however the
    // CLI's subcommand names move.

    /// Assert the payload of a non-error result carries no reachable endpoint.
    #[cfg(unix)]
    fn assert_no_url(result: &ToolCallResult, ctx: &str) {
        let text = &result.content[0].text;
        assert!(
            !text.contains("http://"),
            "{ctx}: no URL may appear when nothing was observed listening, got: {text}"
        );
        if result.is_error.is_none() {
            let payload = payload_of(result);
            assert!(
                payload.get("url").is_none(),
                "{ctx}: `url` key must be absent, got: {payload}"
            );
        }
    }

    /// The #2606 defect proper: 0.63.0 (and the post-#2388 `ready:false`
    /// branch) handed back `http://localhost:<port>` for a port nothing was
    /// bound to. A client that reads `url` — the field it asked the tool for —
    /// gets a working-looking endpoint that refuses every connection.
    #[cfg(unix)]
    #[test]
    fn live_child_that_never_binds_reports_no_url() {
        let port = free_port();
        let args = vec!["5".to_string()];
        let result = spawn_and_confirm("sleep", &args, port, Duration::from_millis(300));
        assert_eq!(result.is_error, None, "a live child is not an error");
        assert_no_url(&result, "alive but never bound");
        let payload = payload_of(&result);
        // The port is still reported — the caller needs to know which one was
        // attempted — but as a scalar that cannot be dialled by mistake.
        assert_eq!(payload["port"], serde_json::json!(port));
        assert_eq!(payload["ready"], serde_json::json!(false));
        if let Some(log) = payload["stderr_log"].as_str() {
            let _ = std::fs::remove_file(log);
        }
    }

    /// A dead child must not leak a URL into its error text either.
    #[cfg(unix)]
    #[test]
    fn dead_child_reports_no_url() {
        let port = free_port();
        let result = spawn_and_confirm("false", &[], port, Duration::from_secs(5));
        assert_eq!(result.is_error, Some(true));
        assert_no_url(&result, "child exited immediately");
    }

    /// `port: 0` asks the OS to assign — there is no port to probe, so the
    /// tool can prove the child survived startup and nothing more. It must
    /// therefore never claim `ready`, and never emit `http://localhost:0`.
    #[cfg(unix)]
    #[test]
    fn ephemeral_port_reports_no_url_and_not_ready() {
        let args = vec!["5".to_string()];
        let result = spawn_and_confirm("sleep", &args, 0, Duration::from_millis(300));
        assert_eq!(result.is_error, None);
        let payload = payload_of(&result);
        assert_eq!(
            payload["ready"],
            serde_json::json!(false),
            "port 0 was never probed, so readiness was never observed"
        );
        assert_no_url(&result, "OS-assigned port");
        if let Some(log) = payload["stderr_log"].as_str() {
            let _ = std::fs::remove_file(log);
        }
    }

    /// `running_result` is a pure function of `(pid, port, ready)`, and the
    /// `url`⇔`ready` equivalence is small enough to check EXHAUSTIVELY rather
    /// than by sample: every one of the 65 536 ports, both readiness values.
    /// Spawn-based falsifiers can only ever sample a port; this closes the
    /// domain, so no port is a special case (0 and 65535 included).
    #[test]
    fn url_key_matches_ready_exhaustively_over_every_port() {
        let log = std::path::Path::new("/tmp/apr-mcp-serve-exhaustive.log");
        for port in 0..=u16::MAX {
            for ready in [false, true] {
                let result = running_result(4242, port, ready, log);
                assert_eq!(result.is_error, None, "port {port}: never an error here");
                let text = &result.content[0].text;
                let v: serde_json::Value =
                    serde_json::from_str(text).expect("running_result emits JSON");
                assert_eq!(
                    v.get("url").is_some(),
                    ready,
                    "port {port}, ready {ready}: `url` must be present iff ready"
                );
                assert_eq!(
                    text.contains("http://"),
                    ready,
                    "port {port}, ready {ready}: no endpoint may appear unless ready"
                );
                assert_eq!(v["ready"], serde_json::json!(ready));
                assert_eq!(v["port"], serde_json::json!(port));
                assert_eq!(v["pid"], serde_json::json!(4242));
            }
        }
    }

    /// The positive half of the invariant: the ONLY combination that may
    /// carry a url is live child + port that accepted a connection. Paired
    /// with the three negatives above this pins `url` to exactly one branch.
    #[cfg(unix)]
    #[test]
    fn url_appears_only_on_the_observed_listening_branch() {
        // Negative: same program, same timeout, port nothing is bound to.
        let dead_port = free_port();
        let unbound = spawn_and_confirm(
            "sleep",
            &["5".to_string()],
            dead_port,
            Duration::from_millis(300),
        );
        assert_no_url(&unbound, "control: unbound port");
        if let Some(log) = payload_of(&unbound)["stderr_log"].as_str() {
            let _ = std::fs::remove_file(log);
        }

        // Positive: identical call, except something IS listening.
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .expect("bind ephemeral port");
        let bound_port = listener.local_addr().expect("local_addr").port();
        let live = spawn_and_confirm(
            "sleep",
            &["5".to_string()],
            bound_port,
            Duration::from_millis(300),
        );
        assert_eq!(live.is_error, None, "got: {}", live.content[0].text);
        let payload = payload_of(&live);
        assert_eq!(payload["ready"], serde_json::json!(true));
        assert_eq!(
            payload["url"],
            serde_json::json!(format!("http://localhost:{bound_port}")),
            "the listening branch is the one branch that owes a url"
        );
        if let Some(log) = payload["stderr_log"].as_str() {
            let _ = std::fs::remove_file(log);
        }
        drop(listener);
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
