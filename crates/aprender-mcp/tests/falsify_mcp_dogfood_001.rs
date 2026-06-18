//! FALSIFY-MCP-DOGFOOD-001 — End-to-end Claude Code dogfood conformance.
//!
//! Spec: `docs/specifications/apr-mcp-server-spec.md` M4 acceptance bullet
//! "Claude Code dogfood — 1 full session using only `apr.*` tools".
//!
//! Every other test in this crate exercises [`AprMcpServer::handle_request`]
//! in-process. That proves the dispatcher logic is correct but says nothing
//! about whether the shipped `apr mcp` *binary* — the executable a real MCP
//! client would actually launch — speaks the protocol end-to-end. This
//! falsifier closes that gap.
//!
//! # What this falsifier proves
//!
//! 1. The `apr` binary, launched as `apr mcp`, accepts JSON-RPC 2.0 messages
//!    on stdin and writes one-line JSON responses on stdout (the baseline
//!    transport contract — anything else means MCP clients can't connect).
//! 2. `initialize` returns `protocolVersion = "2024-11-05"` and
//!    `serverInfo.name = "aprender-mcp"`.
//! 3. `tools/list` returns the 9 registered Phase-1 tools with valid object
//!    schemas (one per `crates/aprender-mcp/src/tools/mod.rs`).
//! 4. `tools/call` works for every one of those 9 tools — either succeeding
//!    via a mock subprocess (for tools that shell out to `apr <cmd> --json`)
//!    or returning `isError:true` via the argument-validation branch (the
//!    same path a real client would hit on a malformed request). Either
//!    outcome confirms the protocol surface — the underlying CLI behaviour
//!    is covered by FALSIFY-MCP-003/-004.
//! 5. JSON-RPC ids round-trip unchanged.
//! 6. Unknown methods return code `-32601` (FALSIFY-MCP-METHODS).
//! 7. A request with `jsonrpc != "2.0"` returns code `-32600` from the
//!    binary surface, not just the in-process dispatcher (FALSIFY-MCP-005).
//!
//! # How it works
//!
//! - Locate the workspace-built `apr` binary via `assert_cmd::cargo_bin`.
//!   Cargo builds workspace binaries before running integration tests, so
//!   the binary is on disk when this test executes.
//! - Drop a mock `apr` shell shim into a tempdir and PREPEND it to the
//!   spawned process's `PATH`. The mock handles `validate`, `tensors`,
//!   `bench`, `qa`, `trace`, `run`, `serve`, `finetune` — every subcommand
//!   the 8 wrapper tools spawn via `Command::new("apr")` from inside the
//!   server. The real binary is only invoked once, at the top of the
//!   process tree, so there is no recursion.
//! - Spawn the binary with `apr mcp` and pipe one JSON-RPC request per line.
//! - Read responses on a bounded channel; fail loudly on a 2-second timeout.
//! - Close stdin → wait for clean exit → assert exit code 0.
//!
//! Pattern mirrors `tests/falsify_mcp_progress_001.rs` and
//! `tests/falsify_mcp_006.rs` — same mock-subprocess + PATH-override
//! discipline so this stays deterministic and CI-friendly. No `tokio`,
//! no `#[ignore]`, no real model required.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Names of every tool the M3 server registers via `AprMcpServer::tool_definitions`.
/// Kept in lock-step with `crates/aprender-mcp/src/server.rs`.
const EXPECTED_TOOLS: &[&str] = &[
    "apr.version",
    "apr.validate",
    "apr.tensors",
    "apr.bench",
    "apr.qa",
    "apr.trace",
    "apr.run",
    "apr.serve",
    "apr.finetune",
];

/// Hard cap on how long a single stdout read may block. Anything longer is
/// almost certainly a deadlock — fail loudly so CI surfaces it immediately.
const READ_TIMEOUT: Duration = Duration::from_secs(2);

/// Build `apr` on demand if `assert_cmd::cargo_bin` couldn't find it.
///
/// This happens when the test crate is exercised in isolation
/// (`cargo test -p aprender-mcp`) without a prior workspace build of the
/// root `aprender` package's `apr` binary. We invoke `cargo build --bin
/// apr -p aprender@<workspace-version>` and then re-resolve via
/// `cargo_bin`. The version qualifier is required because crates.io ships
/// older `aprender` packages that get pulled into the dependency graph,
/// making the bare `-p aprender` spec ambiguous.
///
/// Panics with a clear message if the build itself fails — that's a real
/// failure mode the test must surface, not paper over.
fn build_apr_binary() -> PathBuf {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    // env!() resolves at compile time so we always match the workspace
    // root's `aprender` version exactly, regardless of registry deps.
    let pkg_spec = format!("aprender@{}", env!("CARGO_PKG_VERSION"));
    let status = Command::new(&cargo)
        .args(["build", "--bin", "apr", "-p", &pkg_spec, "--quiet"])
        .status()
        .expect("invoke `cargo build --bin apr`");
    assert!(
        status.success(),
        "cargo build --bin apr -p {pkg_spec} failed with status {status:?}"
    );
    let path = assert_cmd::cargo::cargo_bin("apr");
    assert!(
        path.is_file(),
        "expected apr binary at {} after `cargo build`",
        path.display()
    );
    path
}

/// Tiny tempdir helper — same pattern as
/// `tests/falsify_mcp_progress_001.rs::tempdir_fallback`. Avoids pulling
/// `tempfile` into this crate for one test.
fn tempdir_fallback() -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("apr-mcp-falsify-dogfood-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}

/// Write a mock `apr` shim into `dir` and `chmod +x` it.
///
/// The shim recognises every subcommand the 8 Phase-1 MCP tools spawn —
/// `validate`, `tensors`, `bench`, `qa`, `trace`, `run`, `serve`,
/// `finetune` — and prints a deterministic JSON fixture on stdout, exit 0.
/// Unknown subcommands exit 2 so any new tool added without a matching
/// shim handler shows up as a clear failure rather than a hang.
///
/// Critically the shim does NOT recognise `mcp` — that subcommand is
/// reserved for the real binary at the top of the process tree. If the
/// real binary somehow re-spawns itself, we'd see exit 2 here instead of
/// infinite recursion.
fn write_mock_apr_shim(dir: &Path) {
    let path = dir.join("apr");
    {
        let mut f = std::fs::File::create(&path).expect("create mock apr shim");
        // POSIX shell — we know targets that run this test (Linux, macOS) ship
        // /bin/sh. The mock keeps to portable POSIX so it works on both.
        writeln!(f, "#!/bin/sh").expect("shebang");
        writeln!(f, "set -eu").expect("strict shell");
        // The dispatch is intentionally exhaustive on first arg. Each branch
        // emits the JSON shape the corresponding `tools/<name>.rs` wrapper
        // forwards to the MCP client verbatim. We don't go beyond the bare
        // minimum the wrapper / FALSIFY-MCP-003/-004 tests need; this test
        // only proves protocol round-trip, not CLI fidelity.
        writeln!(f, "case \"$1\" in").expect("case open");

        // apr.validate — wraps `apr validate <model> --json`.
        writeln!(f, "  validate)").expect("validate open");
        writeln!(
            f,
            "    printf '{{\"model\":\"mock\",\"valid\":true,\"gates\":[]}}\\n'"
        )
        .expect("validate body");
        writeln!(f, "    exit 0 ;;").expect("validate close");

        // apr.tensors — wraps `apr tensors <model> --json`.
        writeln!(f, "  tensors)").expect("tensors open");
        writeln!(
            f,
            "    printf '{{\"model\":\"mock\",\"tensors\":[{{\"name\":\"a\",\"shape\":[1]}}]}}\\n'"
        )
        .expect("tensors body");
        writeln!(f, "    exit 0 ;;").expect("tensors close");

        // apr.bench — wraps `apr bench <model> [...]`.
        writeln!(f, "  bench)").expect("bench open");
        writeln!(
            f,
            "    printf '{{\"model\":\"mock\",\"median_tps\":100.0,\"p50_ms\":10.0}}\\n'"
        )
        .expect("bench body");
        writeln!(f, "    exit 0 ;;").expect("bench close");

        // apr.qa — wraps `apr qa <model> --json`.
        writeln!(f, "  qa)").expect("qa open");
        writeln!(
            f,
            "    printf '{{\"model\":\"mock\",\"gates\":{{\"all\":\"PASS\"}}}}\\n'"
        )
        .expect("qa body");
        writeln!(f, "    exit 0 ;;").expect("qa close");

        // apr.trace — wraps `apr trace <model> --prompt X`.
        writeln!(f, "  trace)").expect("trace open");
        writeln!(f, "    printf '{{\"model\":\"mock\",\"layers\":[]}}\\n'").expect("trace body");
        writeln!(f, "    exit 0 ;;").expect("trace close");

        // apr.run — wraps `apr run <model> --json [...]`. Mirrors the schema
        // from FALSIFY-MCP-003: model+text+tokens+tok_per_sec+...
        writeln!(f, "  run)").expect("run open");
        writeln!(
            f,
            "    printf '{{\"model\":\"mock\",\"text\":\"ok\",\"tokens\":[1],\"tokens_generated\":1,\"max_tokens\":1,\"tok_per_sec\":1.0,\"inference_time_ms\":1.0,\"used_gpu\":false,\"cached\":false}}\\n'"
        )
        .expect("run body");
        writeln!(f, "    exit 0 ;;").expect("run close");

        // apr.serve — fire-and-forget. The wrapper only reads our pid; we
        // exit fast so the test doesn't leak a long-running child.
        writeln!(f, "  serve)").expect("serve open");
        writeln!(f, "    exit 0 ;;").expect("serve close");

        // apr.finetune — wraps `apr finetune <base> --json [...]`.
        writeln!(f, "  finetune)").expect("finetune open");
        writeln!(
            f,
            "    printf '{{\"event\":\"complete\",\"checkpoint\":\"/tmp/mock.apr\"}}\\n'"
        )
        .expect("finetune body");
        writeln!(f, "    exit 0 ;;").expect("finetune close");

        // Reject `mcp` explicitly so accidental re-entrancy is loud, not
        // infinite. Any other subcommand falls through to the catch-all.
        writeln!(f, "  mcp)").expect("mcp guard open");
        writeln!(f, "    echo \"mock apr: refusing to recurse into mcp\" >&2")
            .expect("mcp guard body");
        writeln!(f, "    exit 99 ;;").expect("mcp guard close");

        writeln!(f, "  *)").expect("default open");
        writeln!(f, "    echo \"mock apr: unknown subcommand $1\" >&2").expect("default body");
        writeln!(f, "    exit 2 ;;").expect("default close");

        writeln!(f, "esac").expect("case close");
        f.sync_all().expect("sync mock shim");
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path)
            .expect("stat mock apr")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod mock apr");
    }
}

/// Build a `$PATH` value with `mock_dir` prepended. Caller passes this into
/// `Command::env` so only the spawned process sees the override — the
/// parent test process keeps its own PATH untouched, which keeps tests
/// hermetic and avoids the `set_var` cross-test race.
fn path_with_mock_first(mock_dir: &Path) -> String {
    let existing = std::env::var("PATH").unwrap_or_default();
    if existing.is_empty() {
        mock_dir.display().to_string()
    } else {
        format!("{}:{existing}", mock_dir.display())
    }
}

/// Spawn one stdout reader thread that pushes each line through `tx`.
/// The thread exits cleanly on EOF (the server closes its stdout when we
/// close its stdin and it returns from `run_stdio`).
fn spawn_stdout_reader(stdout: ChildStdout, tx: mpsc::Sender<String>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(s) => {
                    if tx.send(s).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    })
}

/// Send one JSON-RPC request as a single newline-terminated line.
fn send(stdin: &mut ChildStdin, value: &serde_json::Value) {
    let line = serde_json::to_string(value).expect("serialize request");
    writeln!(stdin, "{line}").expect("write request line");
    stdin.flush().expect("flush stdin");
}

/// Read one response line within `READ_TIMEOUT` and parse it as JSON.
/// Panics with a helpful message on timeout — that's the loud-failure
/// behaviour the spec demands ("fail loudly if anything hangs").
fn recv(rx: &mpsc::Receiver<String>) -> serde_json::Value {
    let line = rx
        .recv_timeout(READ_TIMEOUT)
        .unwrap_or_else(|e| panic!("no response within {READ_TIMEOUT:?}: {e}"));
    serde_json::from_str(&line)
        .unwrap_or_else(|e| panic!("response line was not valid JSON: {e}; line was: {line}"))
}

/// Build a JSON-RPC 2.0 request with a numeric id.
fn request(id: u64, method: &str, params: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    })
}

/// Minimal valid arguments for each tool — enough to exercise either the
/// mock subprocess path or the argument-validation branch. We use
/// `/dev/null` as the model_path so even tools that read the file before
/// spawning a subprocess (none of the M3 wrappers do, but defensive) get
/// a real readable path.
fn minimal_args(tool: &str) -> serde_json::Value {
    match tool {
        "apr.version" => serde_json::json!({}),
        "apr.serve" => serde_json::json!({ "model_path": "/dev/null", "port": 18080 }),
        "apr.finetune" => serde_json::json!({ "base_model": "/dev/null" }),
        // Every other tool takes a single required `model_path`.
        _ => serde_json::json!({ "model_path": "/dev/null" }),
    }
}

/// FALSIFY-MCP-DOGFOOD-001 — full Claude-Code-style protocol session.
///
/// Single test (per spec) that walks the entire conversation a real MCP
/// client would have on first connection:
/// initialize → tools/list → tools/call × 9 → unknown method → bad
/// jsonrpc → close stdin → exit 0. The whole thing must finish well under
/// the 2s read timeout per message.
#[test]
#[cfg(unix)]
fn falsify_mcp_dogfood_001_full_client_session() {
    let session_start = Instant::now();

    // 1. Mock apr shim on a private PATH for the spawned process only.
    let tmp = tempdir_fallback();
    write_mock_apr_shim(&tmp);
    let path_value = path_with_mock_first(&tmp);

    // 2. Locate the real apr binary. assert_cmd::cargo::cargo_bin walks up
    //    from the test executable into the workspace target dir and looks
    //    for `apr`. If cargo hasn't built it yet (e.g. running
    //    `cargo test -p aprender-mcp` in isolation), fall back to invoking
    //    `cargo build` inline so the test is self-contained and CI doesn't
    //    have to remember an extra pre-step. The workspace member name for
    //    the root `apr` binary is `aprender` (per root Cargo.toml
    //    `[[bin]] name = "apr"`).
    let bin_path = {
        let candidate = assert_cmd::cargo::cargo_bin("apr");
        if candidate.is_file() {
            candidate
        } else {
            build_apr_binary()
        }
    };
    let mut cmd = Command::new(&bin_path);
    cmd.arg("mcp")
        .env("PATH", &path_value)
        // Keep the binary's stderr visible for postmortem if the test fails;
        // assert_cmd-style inheritance is fine here because we never assert
        // on stderr content.
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    let mut child = cmd.spawn().expect("spawn `apr mcp`");
    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");

    // 3. Reader thread + bounded-recv channel for hang detection.
    let (tx, rx) = mpsc::channel::<String>();
    let reader_handle = spawn_stdout_reader(stdout, tx);

    // 4. initialize.
    send(
        &mut stdin,
        &request(
            1,
            "initialize",
            serde_json::json!({ "protocolVersion": "2024-11-05" }),
        ),
    );
    let init = recv(&rx);
    assert_eq!(init["jsonrpc"], "2.0", "JSON-RPC version echoed");
    assert_eq!(init["id"], 1, "id round-trips on initialize");
    assert!(
        init.get("error").is_none(),
        "initialize must succeed, got error: {init:?}"
    );
    let result = &init["result"];
    assert_eq!(
        result["protocolVersion"], "2024-11-05",
        "MCP protocolVersion must match spec v2024-11-05"
    );
    assert_eq!(
        result["serverInfo"]["name"], "aprender-mcp",
        "serverInfo.name must be aprender-mcp"
    );
    assert!(
        result["capabilities"]["tools"].is_object(),
        "capabilities.tools must be present (spec v2024-11-05)"
    );

    // 5. tools/list — assert exactly 9 tools and every name is registered.
    send(&mut stdin, &request(2, "tools/list", serde_json::json!({})));
    let list = recv(&rx);
    assert_eq!(list["id"], 2);
    assert!(list.get("error").is_none(), "tools/list must succeed");
    let tools = list["result"]["tools"]
        .as_array()
        .expect("result.tools must be an array");
    assert_eq!(
        tools.len(),
        EXPECTED_TOOLS.len(),
        "expected exactly {} Phase-1 tools, got {}: {:?}",
        EXPECTED_TOOLS.len(),
        tools.len(),
        tools.iter().map(|t| &t["name"]).collect::<Vec<_>>()
    );
    let names: std::collections::BTreeSet<&str> =
        tools.iter().filter_map(|t| t["name"].as_str()).collect();
    for expected in EXPECTED_TOOLS {
        assert!(
            names.contains(expected),
            "tool {expected} missing from tools/list, got: {names:?}"
        );
    }
    for tool in tools {
        assert_eq!(
            tool["inputSchema"]["type"], "object",
            "every tool's inputSchema.type must be \"object\" per MCP spec; offender: {}",
            tool["name"]
        );
    }

    // 6. tools/call for every registered tool with minimal valid args.
    //    Each must round-trip with content[0] populated. isError may be
    //    true or false depending on whether the wrapper hit the mock or
    //    short-circuited on argument validation — both prove the
    //    protocol surface works (per task spec).
    let mut next_id: u64 = 100;
    for tool in EXPECTED_TOOLS {
        let id = next_id;
        next_id += 1;
        send(
            &mut stdin,
            &request(
                id,
                "tools/call",
                serde_json::json!({ "name": tool, "arguments": minimal_args(tool) }),
            ),
        );
        let resp = recv(&rx);
        assert_eq!(
            resp["id"], id,
            "id must round-trip for tools/call {tool}, got: {resp:?}"
        );
        assert!(
            resp.get("error").is_none(),
            "tools/call {tool} must not return a JSON-RPC error (tool errors live in result.isError), got: {resp:?}"
        );
        let result = &resp["result"];
        let content = result["content"].as_array().unwrap_or_else(|| {
            panic!("tools/call {tool} result.content must be an array; got: {result:?}")
        });
        assert!(
            !content.is_empty(),
            "tools/call {tool} result.content must have at least one block; got: {result:?}"
        );
        assert_eq!(
            content[0]["type"], "text",
            "tools/call {tool} content[0].type must be \"text\""
        );
        assert!(
            content[0]["text"].is_string(),
            "tools/call {tool} content[0].text must be a string"
        );
        // isError is optional in the MCP spec; if present it must be a
        // boolean. If it's true, the text must be a non-empty error blurb.
        if let Some(is_err) = result.get("isError").and_then(|v| v.as_bool()) {
            if is_err {
                let text = content[0]["text"].as_str().expect("error text is a string");
                assert!(
                    !text.is_empty(),
                    "tools/call {tool} returned isError:true but empty text"
                );
            }
        }
    }

    // 7. Unknown method → -32601.
    let unknown_id = next_id;
    next_id += 1;
    send(
        &mut stdin,
        &request(
            unknown_id,
            "this/method/does/not/exist",
            serde_json::json!({}),
        ),
    );
    let unknown = recv(&rx);
    assert_eq!(unknown["id"], unknown_id);
    assert!(
        unknown.get("result").is_none(),
        "unknown method must not return a result, got: {unknown:?}"
    );
    assert_eq!(
        unknown["error"]["code"], -32601,
        "unknown method must map to JSON-RPC -32601 Method Not Found"
    );

    // 8. Invalid jsonrpc field → -32600 (FALSIFY-MCP-005 at the binary
    //    surface, complementing the in-process gate in falsify_m1.rs).
    let bad_jsonrpc_id = next_id;
    let bad = serde_json::json!({
        "jsonrpc": "1.0",
        "id": bad_jsonrpc_id,
        "method": "initialize",
        "params": {}
    });
    send(&mut stdin, &bad);
    let bad_resp = recv(&rx);
    assert_eq!(
        bad_resp["id"], bad_jsonrpc_id,
        "id must round-trip even on Invalid Request"
    );
    assert!(
        bad_resp.get("result").is_none(),
        "invalid jsonrpc must not return a result, got: {bad_resp:?}"
    );
    assert_eq!(
        bad_resp["error"]["code"], -32600,
        "invalid jsonrpc field must map to JSON-RPC -32600 Invalid Request"
    );

    // 9. Shutdown — close stdin, wait for the server to drain and exit.
    drop(stdin);
    let exit = child.wait().expect("apr mcp must exit cleanly");
    assert!(
        exit.success(),
        "apr mcp must exit 0 on stdin EOF, got: {exit:?}"
    );

    // Reader thread should join now that stdout closed.
    reader_handle.join().expect("stdout reader joins cleanly");

    // 10. Whole-session budget. Spec asks for <2s; in practice this runs in
    //     well under 1s on any reasonable machine. Generous slack to absorb
    //     CI noise without masking real regressions.
    let elapsed = session_start.elapsed();
    assert!(
        elapsed < Duration::from_secs(10),
        "full dogfood session must complete in <10s (spec budget 2s + CI slack), took {elapsed:?}"
    );
}

/// Build a JSON-RPC 2.0 *notification* — a Request object with NO `id`
/// member. Per JSON-RPC 2.0 §4.1 this signifies the client's lack of
/// interest in a response, and "The Server MUST NOT reply to a
/// Notification."
fn notification(method: &str, params: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    })
}

/// FALSIFY-MCP-009 — the `apr mcp` binary MUST NOT reply to a JSON-RPC
/// notification, even when the notification's method is a normally-
/// id-bearing request method (`initialize`, an unknown method, etc.).
///
/// # The defect this falsifies
///
/// JSON-RPC 2.0 §4.1 defines a Notification purely by the *absence of an
/// `id`* — not by the method name. The pre-fix dispatcher only suppressed
/// responses for methods literally prefixed with `notifications/`; a
/// no-id `initialize` or no-id unknown method fell through to the inline
/// `handle_request` path and got a `{"jsonrpc":"2.0","id":null,...}`
/// response written to stdout. That stray response desyncs the client:
/// the next real request's reply is one slot behind, so the client
/// matches the wrong id.
///
/// # Why this assertion catches it without racing on "no output"
///
/// "Assert nothing was written" is inherently a timeout/race. Instead we
/// rely on *stream ordering*: send two notifications (no-id `initialize`,
/// no-id unknown method), then a real id-bearing request. Stdio responses
/// are written in order on a single mutex-guarded handle, so the FIRST
/// line we read after the notifications MUST be the real request's
/// response (`id == 7`). If the buggy server emitted an `id:null`
/// response for either notification, that line arrives first and the
/// assertion `resp["id"] == 7` fails deterministically — no timeout
/// needed.
#[test]
#[cfg(unix)]
fn falsify_mcp_009_no_reply_to_notification() {
    let tmp = tempdir_fallback();
    write_mock_apr_shim(&tmp);
    let path_value = path_with_mock_first(&tmp);

    let bin_path = {
        let candidate = assert_cmd::cargo::cargo_bin("apr");
        if candidate.is_file() {
            candidate
        } else {
            build_apr_binary()
        }
    };

    let mut cmd = Command::new(&bin_path);
    cmd.arg("mcp")
        .env("PATH", &path_value)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    let mut child = cmd.spawn().expect("spawn `apr mcp`");
    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");

    let (tx, rx) = mpsc::channel::<String>();
    let reader_handle = spawn_stdout_reader(stdout, tx);

    // Two notifications (no `id`): a normally-id-bearing request method and
    // an unknown method. A conformant server emits NOTHING for either.
    send(
        &mut stdin,
        &notification(
            "initialize",
            serde_json::json!({ "protocolVersion": "2024-11-05" }),
        ),
    );
    send(
        &mut stdin,
        &notification("this/method/does/not/exist", serde_json::json!({})),
    );

    // Now a real request. With a conformant server this is the only message
    // that produces output, so it must be the first (and only) line read.
    send(&mut stdin, &request(7, "tools/list", serde_json::json!({})));

    let resp = recv(&rx);
    assert_eq!(
        resp["id"], 7,
        "first response after two notifications must be the tools/list reply (id=7). \
         A different id (especially null) means the server illegally replied to a \
         notification and desynced the stream — JSON-RPC 2.0 §4.1 violation. got: {resp:?}"
    );
    assert!(
        resp.get("error").is_none(),
        "tools/list must succeed; got: {resp:?}"
    );
    assert!(
        resp["result"]["tools"].is_array(),
        "tools/list result.tools must be an array; got: {resp:?}"
    );

    // Belt-and-braces: drain stdin, let the server exit, and confirm NO
    // further response lines were buffered (e.g. a delayed id:null). Any
    // extra line here is also a §4.1 violation.
    drop(stdin);
    let exit = child.wait().expect("apr mcp must exit cleanly");
    assert!(
        exit.success(),
        "apr mcp must exit 0 on stdin EOF, got: {exit:?}"
    );
    reader_handle.join().expect("stdout reader joins cleanly");

    let mut extra = Vec::new();
    while let Ok(line) = rx.try_recv() {
        extra.push(line);
    }
    assert!(
        extra.is_empty(),
        "server emitted {} response line(s) beyond the single tools/list reply; \
         notifications must produce no response (JSON-RPC 2.0 §4.1). Extra lines: {extra:?}",
        extra.len()
    );
}
