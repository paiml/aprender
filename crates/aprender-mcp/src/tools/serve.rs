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
//! DOGFOOD-0.64.0 (#2606, second pass) — an adversarial review reproduced the
//! ORIGINAL fabricated-URL shape against the fix above, and it was right to:
//!
//! 5. "The child is alive AND something accepted a TCP connect on that port"
//!    is not "**our child** is listening". Both conjuncts hold when any
//!    unrelated process happens to hold the port: the spawned child binds
//!    nothing, the probe succeeds against the stranger's socket, and the tool
//!    hands back `{"ready":true,"url":"http://localhost:<port>"}` for a server
//!    it did not start and cannot vouch for. The first fix made the *claim*
//!    narrower without making the *evidence* stronger.
//!
//!    Two things close it, and both are needed:
//!
//!    a. **Pre-flight** (portable). Before spawning, probe the port. If it
//!       already accepts connections, no later success on it can be
//!       attributed to a child that does not exist yet — and `apr serve run`
//!       could not have bound it anyway. Refuse, before spawning, naming the
//!       port.
//!    b. **Attribution** (Linux). When the probe succeeds, require that a
//!       LISTEN socket on that port is held by a file descriptor of the child
//!       or one of its descendants — [`crate::tools::port_owner`], via
//!       `/proc/net/tcp{,6}` ∩ `/proc/<pid>/fd`. A stranger that grabs the
//!       port *during* the readiness window is caught here, which pre-flight
//!       alone cannot do.
//!
//! The generalizing invariant, the one that survives the next argv change:
//!
//! **SERVE-URL-ONLY-WHEN-CHILD-OWNS-PORT** — `apr.serve` reports a URL only
//! when a listening socket on that port is held by the child it spawned (or a
//! descendant of it). It names no subcommand and no flag, so it keeps its
//! teeth however the CLI's surface moves. A pid proves a process was created,
//! not that it survived argv parsing; a TCP connect proves *somebody* is
//! listening, not that it is ours.
//!
//! **Where the guarantee is weaker, and the payload says so.** Socket→pid
//! attribution is not portable (see [`crate::tools::port_owner`] for why: no
//! `/proc` on macOS, and `libproc`/`lsof` are out of reach under
//! `unsafe_code = "forbid"` with no new dependencies). On a platform that
//! cannot attribute, the tool falls back to pre-flight alone — *the port was
//! proven closed immediately before spawn, and then began accepting
//! connections while our child was alive* — which is strictly weaker: a
//! process that grabs the port inside the readiness window would still be
//! mistaken for the child. That case is reported with
//! `"attribution": "unavailable"` rather than dressed up as the strong
//! guarantee, so a client can tell the two apart. On Linux — every CI runner
//! and every measured host in #2606 — the strong form applies and the payload
//! says `"attribution": "child"`.
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
use crate::tools::port_owner::{owner_of_listening_port, PortOwner};
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

/// What the readiness loop concluded about the port, and on what evidence.
///
/// This is the single gate on the `url` key (#2606). It exists as an enum
/// rather than a `bool` because "ready" was the bug: a boolean cannot
/// distinguish *our child is listening* from *somebody is listening*, and the
/// first fix for #2606 collapsed both into `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Listening {
    /// A LISTEN socket on the port is held by the spawned child or one of its
    /// descendants. The url is attributable. **The only strong branch.**
    ByChild,
    /// The port was proven closed immediately before spawn and then began
    /// accepting connections while the child was alive, on a platform that
    /// cannot map a socket to a pid. Reported with `attribution: unavailable`.
    UnattributedAfterPreflight,
    /// Something accepted a connection but the socket is NOT held by the child
    /// or any descendant — a stranger grabbed the port inside the readiness
    /// window. Never a url.
    ByForeignProcess,
    /// Nothing accepted a connection inside the readiness window.
    No,
}

impl Listening {
    /// Whether this branch may carry a `url`. The whole invariant, in one
    /// place: only evidence that ties the socket to *our* child qualifies.
    #[must_use]
    pub const fn is_attributable(self) -> bool {
        matches!(self, Self::ByChild | Self::UnattributedAfterPreflight)
    }

    /// Machine-readable evidence label for the payload.
    #[must_use]
    pub const fn attribution(self) -> &'static str {
        match self {
            Self::ByChild => "child",
            Self::UnattributedAfterPreflight => "unavailable",
            Self::ByForeignProcess => "foreign",
            Self::No => "none",
        }
    }
}

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
    spawn_and_confirm_with_env(program, args, port, ready_timeout, &[])
}

/// [`spawn_and_confirm`] with extra environment for the child.
///
/// The env hook exists so the readiness/attribution contract can be exercised
/// against a child that really does own a listening socket, without an extra
/// binary target (`cargo test --lib`, which CI runs, does not build `[[bin]]`
/// targets) and without mutating this process's environment.
#[must_use]
pub fn spawn_and_confirm_with_env(
    program: &str,
    args: &[String],
    port: u16,
    ready_timeout: Duration,
    envs: &[(&str, String)],
) -> ToolCallResult {
    let cmd_display = format!("{program} {}", args.join(" "));
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));

    if let Some(conflict) = preflight_conflict(port, &addr, &cmd_display) {
        return conflict;
    }

    // The daemon outlives this call, so its stderr must not go to a pipe
    // nobody drains (a full pipe buffer would wedge the server). Route it to
    // a file we can read back if the child dies, and hand the path to the
    // caller when it lives.
    let log_path = stderr_log_path(port);
    let stderr = match std::fs::File::create(&log_path) {
        Ok(f) => Stdio::from(f),
        Err(_) => Stdio::null(),
    };

    let mut command = Command::new(program);
    for (key, value) in envs {
        command.env(key, value);
    }
    let mut child = match command
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
    // Port 0 means "OS assigns" — there is nothing to probe, so the window
    // only proves the child survived startup.
    let window = if port == 0 {
        ready_timeout.min(EPHEMERAL_PORT_WINDOW)
    } else {
        ready_timeout
    };
    let deadline = Instant::now() + window;

    await_readiness(
        &mut child,
        pid,
        port,
        &addr,
        deadline,
        &cmd_display,
        &log_path,
    )
}

/// Poll a live child until it dies, until the port is proven to be its own, or
/// until the readiness window closes.
fn await_readiness(
    child: &mut std::process::Child,
    pid: u32,
    port: u16,
    addr: &SocketAddr,
    deadline: Instant,
    cmd_display: &str,
    log_path: &std::path::Path,
) -> ToolCallResult {
    loop {
        // Liveness first: a dead child can never be ready, and checking it
        // before the connect probe stops an unrelated process that happens to
        // hold `port` from masquerading as our daemon.
        match child.try_wait() {
            Ok(Some(status)) => return exited_error(cmd_display, status, port, log_path),
            Ok(None) => {}
            Err(e) => {
                let _ = std::fs::remove_file(log_path);
                return ToolCallResult::error(format!("failed to poll `{cmd_display}`: {e}"));
            }
        }

        if port != 0 && TcpStream::connect_timeout(addr, CONNECT_TIMEOUT).is_ok() {
            // Re-check liveness: the connect could have raced a dying child.
            if let Ok(Some(status)) = child.try_wait() {
                return exited_error(cmd_display, status, port, log_path);
            }
            // #2606 (second pass), (5b) ATTRIBUTION. A successful connect only
            // proves SOMEBODY is listening. Require the socket to be held by
            // our child (or a descendant) before naming a url after it.
            return running_result(pid, port, classify_listener(port, pid), log_path);
        }

        if Instant::now() >= deadline {
            return running_result(pid, port, Listening::No, log_path);
        }
        std::thread::sleep(READY_POLL);
    }
}

/// #2606 (second pass), (5a) PRE-FLIGHT — refuse a port that is already taken,
/// BEFORE spawning anything.
///
/// If the port already accepts connections there is no child yet to own it, so
/// every later probe on it is unattributable by construction — and
/// `apr serve run` could not have bound it anyway. This is the portable half of
/// the fix: on a platform with no socket→pid interface it is the only thing
/// standing between a client and a stranger's URL.
fn preflight_conflict(port: u16, addr: &SocketAddr, cmd_display: &str) -> Option<ToolCallResult> {
    if port == 0 || TcpStream::connect_timeout(addr, CONNECT_TIMEOUT).is_err() {
        return None;
    }
    // Name the squatter when the platform can: "port 8080 is busy" leaves the
    // caller stuck, "held by pid 4242 (apr)" does not.
    let holders = crate::tools::port_owner::listening_pids(port);
    let held_by = if holders.is_empty() {
        String::new()
    } else {
        let list: Vec<String> = holders
            .iter()
            .map(|(pid, comm)| format!("pid {pid} ({comm})"))
            .collect();
        format!(" (held by {})", list.join(", "))
    };
    Some(ToolCallResult::error(format!(
        "port {port} is already accepting connections BEFORE `{cmd_display}` was \
         spawned{held_by} — another process holds it. Refusing to start a server that \
         cannot bind it, and refusing to report a URL for a listener this tool did \
         not start."
    )))
}

/// The child died inside the readiness window: report the status and the tail
/// of its own stderr, which is what turns "it didn't work" into "unrecognized
/// subcommand".
fn exited_error(
    cmd_display: &str,
    status: std::process::ExitStatus,
    port: u16,
    log_path: &std::path::Path,
) -> ToolCallResult {
    let detail = read_stderr_tail(log_path);
    let _ = std::fs::remove_file(log_path);
    let code = status
        .code()
        .map_or_else(|| "signal".to_string(), |c| c.to_string());
    ToolCallResult::error(format!(
        "`{cmd_display}` exited immediately (status {code}) — no server is listening on \
         port {port}{detail}"
    ))
}

/// Turn a successful connect into the evidence class it actually supports.
///
/// Pure decision, kept out of the loop so it can be reasoned about (and
/// mutated) on its own: `Unknown` from a platform without socket→pid tables
/// must NOT become `ByChild`, and `Foreign` must NOT become a url.
fn classify_listener(port: u16, pid: u32) -> Listening {
    match owner_of_listening_port(port, pid) {
        PortOwner::Child => Listening::ByChild,
        PortOwner::Foreign => Listening::ByForeignProcess,
        // Pre-flight already proved the port was closed when we spawned, so
        // this is the best a non-attributing platform can honestly claim.
        PortOwner::Unknown => Listening::UnattributedAfterPreflight,
    }
}

/// Build the non-error payload for a child that is still running.
///
/// `listening` carries both the verdict and the EVIDENCE it rests on, and is
/// the sole gate on the `url` key.
///
/// #2606 — INVARIANT SERVE-URL-ONLY-WHEN-CHILD-OWNS-PORT. `url` is emitted
/// only for [`Listening::ByChild`] — a LISTEN socket on `port` held by the
/// spawned child or a descendant — or, on a platform that cannot map a socket
/// to a pid at all, for [`Listening::UnattributedAfterPreflight`], which is
/// flagged as such in the payload (`attribution: "unavailable"`) instead of
/// being passed off as the strong guarantee.
///
/// Two fabrications are excluded, and the second is the one the first #2606
/// fix missed. (i) Until #2388/#2606 the alive-but-unbound branch still
/// shipped `"url": "http://localhost:<port>"`, so a client reading the field
/// it asked for got an endpoint nothing was listening on. (ii) Gating on
/// "child alive AND something answered the port" still fabricated whenever an
/// unrelated process held the port: the url named a stranger's server. Both
/// are now unrepresentable — there is no url to misread on either branch.
///
/// Still-alive-but-unbound stays a NON-error deliberately. A multi-GB model
/// load can outlast [`READY_TIMEOUT`], the child is a real process the caller
/// owns and must kill, and reporting `isError` for a healthy loader would be
/// its own false claim. The honest report is: pid yes, url no, ready false.
fn running_result(
    pid: u32,
    port: u16,
    listening: Listening,
    log_path: &std::path::Path,
) -> ToolCallResult {
    let ready = listening.is_attributable();
    let note = match listening {
        Listening::ByChild => {
            "server is accepting connections and the listening socket is held by the \
             spawned process; kill pid via OS to stop"
        }
        Listening::UnattributedAfterPreflight => {
            "the port was closed before spawn and is now accepting connections while the \
             child is alive, but this platform cannot map a socket to a pid, so the \
             listener is NOT proven to be the spawned process; kill pid via OS to stop"
        }
        Listening::ByForeignProcess => {
            "a process OTHER than the one just spawned holds the listening socket on this \
             port; no URL is reported because it would name someone else's server; \
             kill pid via OS to stop"
        }
        Listening::No => {
            "process is alive but has not bound the port yet (still loading?); no URL is \
             reported because nothing accepted a connection on the port; kill pid via OS to stop"
        }
    };
    let mut payload = serde_json::json!({
        "pid": pid,
        "ready": ready,
        "port": port,
        "attribution": listening.attribution(),
        "stderr_log": log_path.display().to_string(),
        "note": note,
    });
    if ready {
        // Only reachable when the listening socket was tied to this child, or
        // (on a platform that cannot tie it) when pre-flight proved the port
        // was closed before this child existed.
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
    use crate::tools::port_owner::attribution_available;

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

    /// Pick a port nothing is listening on, and that nothing is likely to
    /// grab before the test spawns.
    ///
    /// Deliberately NOT `bind(0)`: that draws from the kernel's ephemeral
    /// range, so a port handed out here and released can be re-issued moments
    /// later to any *other* concurrent `bind(0)` — including one in a sibling
    /// test, or another crate's test binary on the same machine. Since #2606's
    /// second pass, a port that turns out to be held by a stranger is a hard
    /// error (that is the fix), so that race would surface as a flake in tests
    /// about something else entirely. Walking a private range above the
    /// ephemeral window, and proving each candidate binds, removes it.
    /// Low end of the private test range. Above the privileged ports, below
    /// the kernel's ephemeral window (32768–60999) so no `bind(0)` anywhere on
    /// the machine can be handed the same number.
    #[cfg(unix)]
    const TEST_PORT_BASE: u16 = 10_000;
    /// Width of the private range.
    #[cfg(unix)]
    const TEST_PORT_SPAN: u16 = 22_000;

    #[cfg(unix)]
    fn free_port() -> u16 {
        use std::sync::atomic::{AtomicU16, Ordering};
        static NEXT: AtomicU16 = AtomicU16::new(0);

        // Start each PROCESS at a different offset. Under nextest every test
        // is its own process, so a counter that always started at the same
        // number would hand the identical port to every test running in
        // parallel — and since #2606's second pass a port held by a stranger
        // is a hard error, that collision would surface as a flake in tests
        // about something else. The pid spreads them out; the bind check
        // below still proves each candidate is actually free.
        let seed = u16::try_from(std::process::id() % u32::from(TEST_PORT_SPAN))
            .unwrap_or_else(|_| unreachable!("modulo TEST_PORT_SPAN fits in u16"));
        let _ = NEXT.compare_exchange(0, seed.max(1), Ordering::Relaxed, Ordering::Relaxed);

        for _ in 0..256 {
            let offset = NEXT.fetch_add(1, Ordering::Relaxed) % TEST_PORT_SPAN;
            let port = TEST_PORT_BASE + offset;
            // A successful bind proves the port is free right now; dropping
            // the listener releases it for the test's own child to take.
            if std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).is_ok() {
                return port;
            }
        }
        panic!("no free port in the private test range");
    }

    /// Run `body` on a port nothing else holds, retrying if the machine
    /// disagrees.
    ///
    /// Between `free_port`'s bind check and the spawn there is a window in
    /// which any other process on the box can take the port; the tests below
    /// are about the tool's behaviour, not about winning that race. A
    /// pre-flight conflict therefore means the test's PRECONDITION failed, so
    /// re-draw a port — but only a bounded number of times, and only for that
    /// one distinctive message. A tool that refuses every port still fails,
    /// loudly: the retry cannot turn a real regression green.
    #[cfg(unix)]
    fn on_exclusive_port<F>(body: F) -> (u16, ToolCallResult)
    where
        F: Fn(u16) -> ToolCallResult,
    {
        let mut last = String::new();
        for _ in 0..8 {
            let port = free_port();
            let result = body(port);
            if !result.content[0]
                .text
                .contains("already accepting connections BEFORE")
            {
                return (port, result);
            }
            last.clone_from(&result.content[0].text);
        }
        panic!("8 ports in a row were reported as already held; last: {last}");
    }

    /// Environment key the re-entrant [`listener_helper`] reads its port from.
    const LISTEN_PORT_ENV: &str = "APR_MCP_SERVE_TEST_LISTEN_PORT";

    /// Ignored by every normal run; invoked ONLY as a child of
    /// `live_child_that_owns_the_socket_reports_ready_true_with_url`, where it
    /// plays the part of `apr serve run`: bind the port it is handed, hold it,
    /// and wait to be killed. Without a real listening child there is no
    /// honest way to exercise the one branch that may emit a url.
    #[cfg(unix)]
    #[test]
    #[ignore = "helper process for live_child_that_owns_the_socket_reports_ready_true_with_url"]
    fn listener_helper() {
        let Ok(raw) = std::env::var(LISTEN_PORT_ENV) else {
            // Run without the env var (e.g. `--include-ignored`): no-op.
            return;
        };
        let port: u16 = raw.parse().expect("port env var is a u16");
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
            .expect("helper binds the port it was handed");
        // Bounded, so a failed kill cannot leave the port held forever.
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(100));
        }
        drop(listener);
    }

    /// libtest's `--exact` takes the path WITHOUT the crate name.
    #[cfg(unix)]
    fn helper_test_path() -> String {
        let module = module_path!()
            .split_once("::")
            .map_or(module_path!(), |(_crate_name, rest)| rest);
        format!("{module}::listener_helper")
    }

    #[cfg(unix)]
    fn kill_pid(pid: u64) {
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
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
        let (port, result) =
            on_exclusive_port(|port| spawn_and_confirm("false", &[], port, Duration::from_secs(5)));
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
        let args = vec![
            "-c".to_string(),
            "echo 'unrecognized subcommand' >&2; exit 2".to_string(),
        ];
        let (_port, result) =
            on_exclusive_port(|port| spawn_and_confirm("sh", &args, port, Duration::from_secs(5)));
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

    /// Success path: a live child that OWNS a listening socket on the port.
    ///
    /// The child is this very test binary, re-invoked to run the `#[ignore]`d
    /// [`listener_helper`] below, which binds the port it is handed and holds
    /// it. That makes the listening socket genuinely the spawned process's —
    /// the one and only case that may carry a url. Nothing portable lets a
    /// shell builtin listen on a TCP port, and a helper `[[bin]]` would not be
    /// built by `cargo test --lib` (the form CI runs), so re-entering the test
    /// harness is what keeps this branch reachable from a `--lib` test.
    #[cfg(unix)]
    #[test]
    fn live_child_that_owns_the_socket_reports_ready_true_with_url() {
        let exe = std::env::current_exe().expect("current_exe");
        let args = vec![
            "--exact".to_string(),
            helper_test_path(),
            "--ignored".to_string(),
            "--nocapture".to_string(),
            "--test-threads".to_string(),
            "1".to_string(),
        ];
        // The helper reads its port from its own environment, set only for
        // the child — this process's environment is never mutated.
        let (port, result) = on_exclusive_port(|port| {
            spawn_and_confirm_with_env(
                &exe.to_string_lossy(),
                &args,
                port,
                Duration::from_secs(20),
                &[(LISTEN_PORT_ENV, port.to_string())],
            )
        });
        // Reap the helper BEFORE asserting: a panic here would otherwise leave
        // it holding the port, poisoning later runs on the same machine.
        if let Some(pid) = payload_of(&result)["pid"].as_u64() {
            kill_pid(pid);
        }
        assert_eq!(result.is_error, None, "got: {}", result.content[0].text);
        let payload = payload_of(&result);
        assert_eq!(payload["ready"], serde_json::json!(true));
        assert_eq!(
            payload["url"],
            serde_json::json!(format!("http://localhost:{port}")),
            "a child that owns the listening socket is the branch that owes a url"
        );
        if attribution_available() {
            assert_eq!(
                payload["attribution"],
                serde_json::json!("child"),
                "on Linux the url must rest on socket->pid attribution, not on the \
                 weaker pre-flight fallback"
            );
        }
        assert!(payload["pid"].as_u64().is_some_and(|pid| pid > 0));
        // The log path handed to the client must actually exist.
        let log = payload["stderr_log"]
            .as_str()
            .expect("stderr_log is a path");
        assert!(
            std::path::Path::new(log).exists(),
            "stderr_log {log} must exist for a running daemon"
        );
        let _ = std::fs::remove_file(log);
    }

    /// A live child that never binds is neither a failure nor a running
    /// server: report it truthfully as `ready: false` rather than implying a
    /// reachable URL.
    #[cfg(unix)]
    #[test]
    fn live_child_that_never_binds_reports_ready_false() {
        let args = vec!["5".to_string()];
        let (_port, result) = on_exclusive_port(|port| {
            spawn_and_confirm("sleep", &args, port, Duration::from_millis(300))
        });
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
    // INVARIANT SERVE-URL-ONLY-WHEN-CHILD-OWNS-PORT: `apr.serve` may put a
    // `url` in its payload only when a LISTEN socket on that port is held by
    // the child it spawned (or a descendant). Every other terminal state —
    // dead child, unspawnable program, alive-but-unbound, ephemeral port, and
    // *a stranger holding the port* — must carry NO url. This is the property
    // that survives the next argv change: it never mentions `serve`, `run`, or
    // any flag, so it keeps its teeth however the CLI's subcommand names move.
    //
    // The first #2606 fix stated the weaker "child alive AND something
    // accepted a connect", which an unrelated listener satisfies — see
    // `an_unrelated_listener_on_the_port_never_produces_a_url`. Where a
    // platform cannot attribute a socket to a pid, the tests below narrow to
    // what it does guarantee rather than assert a claim the code cannot make.

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
        let args = vec!["5".to_string()];
        let (port, result) = on_exclusive_port(|port| {
            spawn_and_confirm("sleep", &args, port, Duration::from_millis(300))
        });
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
        let (_port, result) =
            on_exclusive_port(|port| spawn_and_confirm("false", &[], port, Duration::from_secs(5)));
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

    /// `running_result` is a pure function of `(pid, port, listening)`, and
    /// the `url`⇔attribution equivalence is small enough to check
    /// EXHAUSTIVELY rather than by sample: every one of the 65 536 ports, all
    /// four evidence classes. Spawn-based falsifiers can only ever sample a
    /// port; this closes the domain, so no port is a special case (0 and 65535
    /// included).
    ///
    /// Crucially, `ByForeignProcess` — "the port answers, but the socket is
    /// somebody else's" — is in the domain. That is the state the first #2606
    /// fix collapsed into `ready: true`, and here it must carry no url at any
    /// port.
    #[test]
    fn url_key_matches_attribution_exhaustively_over_every_port() {
        let log = std::path::Path::new("/tmp/apr-mcp-serve-exhaustive.log");
        for port in 0..=u16::MAX {
            for listening in [
                Listening::No,
                Listening::ByForeignProcess,
                Listening::UnattributedAfterPreflight,
                Listening::ByChild,
            ] {
                let expect_url = matches!(
                    listening,
                    Listening::ByChild | Listening::UnattributedAfterPreflight
                );
                let result = running_result(4242, port, listening, log);
                assert_eq!(result.is_error, None, "port {port}: never an error here");
                let text = &result.content[0].text;
                let v: serde_json::Value =
                    serde_json::from_str(text).expect("running_result emits JSON");
                assert_eq!(
                    v.get("url").is_some(),
                    expect_url,
                    "port {port}, {listening:?}: `url` must be present iff the listener \
                     is attributable to the spawned child"
                );
                assert_eq!(
                    text.contains("http://"),
                    expect_url,
                    "port {port}, {listening:?}: no endpoint may appear otherwise"
                );
                assert_eq!(v["ready"], serde_json::json!(expect_url));
                assert_eq!(v["port"], serde_json::json!(port));
                assert_eq!(v["pid"], serde_json::json!(4242));
                assert_eq!(
                    v["attribution"],
                    serde_json::json!(listening.attribution()),
                    "port {port}, {listening:?}: the evidence class must be reported, so a \
                     client can tell the strong guarantee from the weak one"
                );
            }
        }
    }

    /// The evidence classes and the url gate must not drift apart: `ByChild`
    /// is the only class that is BOTH attributable and labelled `child`.
    #[test]
    fn only_child_owned_evidence_carries_the_strong_label() {
        assert!(Listening::ByChild.is_attributable());
        assert_eq!(Listening::ByChild.attribution(), "child");
        assert!(!Listening::ByForeignProcess.is_attributable());
        assert!(!Listening::No.is_attributable());
        // The weak branch may carry a url, but must never be mislabelled as
        // the strong one.
        assert!(Listening::UnattributedAfterPreflight.is_attributable());
        assert_ne!(
            Listening::UnattributedAfterPreflight.attribution(),
            "child",
            "the fallback must not masquerade as socket->pid attribution"
        );
    }

    /// `classify_listener` is the point where a platform verdict becomes a url
    /// decision. `Foreign` must never become a url, and `Unknown` must never
    /// become the strong `ByChild` claim.
    #[cfg(unix)]
    #[test]
    fn classify_listener_never_attributes_a_stranger_socket_to_the_child() {
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local_addr").port();
        let mut stranger = Command::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep");
        let verdict = classify_listener(port, stranger.id());
        let _ = stranger.kill();
        let _ = stranger.wait();
        drop(listener);

        if attribution_available() {
            assert_eq!(
                verdict,
                Listening::ByForeignProcess,
                "a socket held by an unrelated process must not be classified as the child's"
            );
            assert!(
                !verdict.is_attributable(),
                "and therefore must not be allowed to carry a url"
            );
        } else {
            assert_eq!(
                verdict,
                Listening::UnattributedAfterPreflight,
                "a platform that cannot attribute must say so, never claim ByChild"
            );
            assert_ne!(verdict, Listening::ByChild);
        }
    }

    /// **The #2606 second-pass falsifier — the verifier's own reproduction.**
    ///
    /// An unrelated process holds the port; the spawned child binds nothing
    /// and never will. The first #2606 fix reported
    /// `{"ready":true,"url":"http://localhost:<port>"}` here, because its gate
    /// was "child alive AND *something* accepted a connect" — a stranger's
    /// listener satisfies both conjuncts. This is byte-for-byte the
    /// fabricated-URL shape #2606 was filed about, merely reached by a
    /// different route.
    ///
    /// RED against the previous implementation; the whole point of the second
    /// pass.
    #[cfg(unix)]
    #[test]
    fn an_unrelated_listener_on_the_port_never_produces_a_url() {
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local_addr").port();

        // `sleep` stands in for a daemon that dies at clap, or hangs loading,
        // or otherwise never binds: exactly the 0.63.0 child.
        let result = spawn_and_confirm(
            "sleep",
            &["5".to_string()],
            port,
            Duration::from_millis(500),
        );

        assert_no_url(&result, "a stranger holds the port");
        assert_eq!(
            result.is_error,
            Some(true),
            "a port already held by someone else is a conflict the caller must be told \
             about, not a server; got: {}",
            result.content[0].text
        );
        assert!(
            result.content[0].text.contains(&port.to_string()),
            "the conflict must name the port, got: {}",
            result.content[0].text
        );
        drop(listener);
    }

    /// The same class, but with the stranger arriving *inside* the readiness
    /// window rather than before it — the case pre-flight alone cannot see.
    /// Only socket→pid attribution catches this, so on a platform without it
    /// the assertion narrows to what that platform actually guarantees.
    #[cfg(unix)]
    #[test]
    fn a_listener_that_appears_mid_window_is_not_attributed_to_the_child() {
        // The stranger must arrive AFTER pre-flight, which is the whole
        // point of this case. If the machine is loaded enough that it lands
        // first, pre-flight refuses and the attempt is re-drawn on a fresh
        // port rather than asserted on — the precondition failed, not the
        // tool. Eight refusals in a row still fail loudly.
        let mut attempt = 0;
        let (result, listener) = loop {
            attempt += 1;
            assert!(attempt <= 8, "pre-flight refused 8 ports in a row");
            let port = free_port();
            let handle = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).ok()
            });
            let result =
                spawn_and_confirm("sleep", &["5".to_string()], port, Duration::from_secs(3));
            let listener = handle.join().ok().flatten();
            if !result.content[0]
                .text
                .contains("already accepting connections BEFORE")
            {
                break (result, listener);
            }
            drop(listener);
        };

        let payload = payload_of(&result);
        if attribution_available() {
            assert_no_url(&result, "stranger bound the port mid-window");
            assert_eq!(
                payload["attribution"],
                serde_json::json!("foreign"),
                "the socket belongs to the test process, not the spawned child; got: {payload}"
            );
            assert_eq!(payload["ready"], serde_json::json!(false));
        } else {
            // Documented weaker guarantee: pre-flight passed and the port
            // later answered, so this platform cannot tell whose it is. It
            // must at least refuse the strong label.
            assert_ne!(
                payload["attribution"],
                serde_json::json!("child"),
                "a platform without attribution must not claim it; got: {payload}"
            );
        }
        if let Some(log) = payload["stderr_log"].as_str() {
            let _ = std::fs::remove_file(log);
        }
        drop(listener);
    }

    /// Spawning a program that does not exist is an error, not a `{pid,url}`.
    #[cfg(unix)]
    #[test]
    fn unspawnable_program_reports_error() {
        let (_port, result) = on_exclusive_port(|port| {
            spawn_and_confirm(
                "apr-does-not-exist-9c1f",
                &[],
                port,
                Duration::from_millis(200),
            )
        });
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("failed to spawn"));
    }
}
