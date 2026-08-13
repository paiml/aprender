//! FALSIFY-MCP-PIN-001..003 — every subprocess an MCP tool spawns must be the
//! `apr` resolved by [`crate::apr_bin::apr_binary`], never the bare name `apr`
//! handed to the OS for a `$PATH` search.
//!
//! #2384 pinned [`crate::tools::subprocess::run_apr`] and its cancellable
//! sibling, and the module docs of `apr_bin` claim the fix covers "all eight
//! subprocess tools". It did not. Three call sites kept the literal:
//!
//! | call site | reached when |
//! |-----------|--------------|
//! | `tools/serve.rs` `call` | always — `apr.serve` has no other path |
//! | `tools/run.rs` `call_with_sink` | client sends `_meta.progressToken` |
//! | `tools/finetune.rs` `call_with_sink` | client sends `_meta.progressToken` |
//!
//! Measured on the dev box before the fix, with the MCP server run from a
//! HEAD build (`apr 0.63.0 (9b19970db)`) and `/home/noah/.local/bin/apr`
//! (0.63.0's predecessor, `0.60.0`) first on `$PATH`:
//!
//! ```text
//! apr.serve {"model_path": "...q4_k_m.gguf", "port": 18077}
//!   -> {"pid":78614,"url":"http://localhost:18077","ready":true}
//! GET http://127.0.0.1:18077/health
//!   -> {"status":"ok","version":"0.60.0", ...}
//! ```
//!
//! `apr.version` answers `0.63.0` from in-process state, so the one tool a
//! client uses to establish provenance named a version that the daemon it had
//! just started was not running. That is the #2424 defect verbatim, on a
//! surface its fix never scanned.
//!
//! Each falsifier below is behavioural: `$APR_BIN` designates a shim that
//! prints a marker only it can print, and the assertion is that the marker
//! comes back through the tool's own result. Replacing `apr_binary()` with
//! `"apr"` at any of the three call sites turns the matching test RED —
//! the shim is never executed, so the marker cannot appear.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use crate::apr_bin::apr_bin_env_lock;
use crate::server::NotificationSink;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Marker the shim prints. Long and unique so no real `apr` can emit it.
pub(crate) const SHIM_MARKER: &str = "APR-BIN-PINNED-SHIM-7f31c0";

/// Per-process, per-call scratch dir — a fixed path would let two concurrent
/// runs of this test binary delete each other's shim mid-flight.
fn scratch_dir(name: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "aprender-mcp-spawn-pin-{name}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir scratch");
    dir
}

/// Write an executable `apr` shim that answers the three subcommands under
/// test and mimics the real CLI's exit-2-on-unknown-subcommand contract for
/// everything else (so a neighbouring test that spawns while `$APR_BIN` is
/// still set sees the behaviour it expects).
///
/// * `serve`    — marker on **stderr**, exit 1. `spawn_and_confirm` reports a
///                child that died with its stderr tail attached.
/// * `run`      — marker on **stdout**, exit 0 (the `--stream` NDJSON shape).
/// * `finetune` — marker on **stdout**, exit 0.
fn write_apr_shim(dir: &Path) -> PathBuf {
    let shim = dir.join("apr");
    {
        let mut f = std::fs::File::create(&shim).expect("create shim");
        writeln!(f, "#!/bin/sh").expect("shebang");
        writeln!(f, "case \"$1\" in").expect("case");
        writeln!(f, "  serve)    echo '{SHIM_MARKER}' >&2; exit 1 ;;").expect("serve arm");
        writeln!(
            f,
            "  run)      echo '{{\"marker\":\"{SHIM_MARKER}\"}}'; exit 0 ;;"
        )
        .expect("run arm");
        writeln!(
            f,
            "  finetune) echo '{{\"marker\":\"{SHIM_MARKER}\"}}'; exit 0 ;;"
        )
        .expect("finetune arm");
        writeln!(f, "  *)        exit 2 ;;").expect("default arm");
        writeln!(f, "esac").expect("esac");
        f.sync_all().expect("sync");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&shim).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&shim, perms).expect("chmod");
    }
    shim
}

/// Run `body` with `$APR_BIN` pointed at a freshly written shim, holding the
/// global lock for the duration and restoring the environment afterwards.
fn with_apr_shim<T>(name: &str, body: impl FnOnce() -> T) -> T {
    let _guard = apr_bin_env_lock();
    let dir = scratch_dir(name);
    let shim = write_apr_shim(&dir);

    // Edition 2021 — `set_var` is safe here, and the lock above is what keeps
    // it from racing the other tests in this module.
    std::env::set_var(crate::apr_bin::APR_BIN_ENV, &shim);
    let out = body();
    std::env::remove_var(crate::apr_bin::APR_BIN_ENV);
    let _ = std::fs::remove_dir_all(&dir);
    out
}

/// A port nothing is listening on, so `spawn_and_confirm`'s connect probe can
/// never succeed and the child's own exit is what decides the result.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

/// Collect every `notifications/progress` payload the tool emits.
fn capturing_sink() -> (
    NotificationSink,
    std::sync::Arc<Mutex<Vec<crate::types::JsonRpcNotification>>>,
) {
    let captured = std::sync::Arc::new(Mutex::new(Vec::new()));
    let clone = std::sync::Arc::clone(&captured);
    let sink: NotificationSink = Box::new(move |n| {
        clone
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(n);
    });
    (sink, captured)
}

/// FALSIFY-MCP-PIN-001: `apr.serve` must start the resolved `apr`.
///
/// The field failure this pins: the tool reported `ready: true` for a daemon
/// whose `/health` said `0.60.0` while the MCP server was `0.63.0`. Reverting
/// `serve.rs`'s `spawn_and_confirm(apr_binary(), ...)` to
/// `spawn_and_confirm("apr", ...)` makes the shim unreachable and this RED.
#[test]
#[cfg(unix)]
fn falsify_mcp_pin_001_serve_starts_the_resolved_binary() {
    let port = free_port();
    let result = with_apr_shim("serve", || {
        crate::tools::serve::call(&serde_json::json!({
            "model_path": "/nonexistent/model.gguf",
            "port": port,
        }))
    });

    assert_eq!(
        result.is_error,
        Some(true),
        "the shim exits 1, so apr.serve must report a dead daemon, got: {}",
        result.content[0].text
    );
    assert!(
        result.content[0].text.contains(SHIM_MARKER),
        "apr.serve must spawn the binary `apr_binary()` resolved, whose stderr \
         carries {SHIM_MARKER}; got: {}",
        result.content[0].text
    );
}

/// FALSIFY-MCP-PIN-002: `apr.run`'s **streaming** path must run the resolved
/// `apr`. The non-streaming path was pinned by #2384; opting into
/// `_meta.progressToken` used to switch binaries silently.
#[test]
#[cfg(unix)]
fn falsify_mcp_pin_002_run_streaming_uses_the_resolved_binary() {
    let (sink, captured) = capturing_sink();
    let token = serde_json::json!("tok-pin-002");
    let (_tx, rx) = std::sync::mpsc::channel::<()>();

    let result = with_apr_shim("run", || {
        crate::tools::run::call_with_sink(
            &serde_json::json!({ "model_path": "/nonexistent/model.gguf", "prompt": "hi" }),
            &rx,
            Some(&sink),
            Some(token.clone()),
        )
    });

    assert!(
        result.content[0].text.contains(SHIM_MARKER),
        "apr.run --stream must execute the resolved binary; got: {}",
        result.content[0].text
    );

    let notifs = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        notifs.len(),
        1,
        "the shim prints exactly one stdout line, so exactly one progress \
         notification must be forwarded"
    );
    assert_eq!(
        notifs[0].params["message"]["marker"],
        serde_json::json!(SHIM_MARKER),
        "the streamed payload must come from the resolved binary"
    );
}

/// FALSIFY-MCP-PIN-003: `apr.finetune`'s streaming path must run the resolved
/// `apr`. Same defect, same shape — training is the most expensive thing to
/// have silently executed by a different build.
#[test]
#[cfg(unix)]
fn falsify_mcp_pin_003_finetune_streaming_uses_the_resolved_binary() {
    let (sink, captured) = capturing_sink();
    let token = serde_json::json!("tok-pin-003");

    let result = with_apr_shim("finetune", || {
        crate::tools::finetune::call_with_sink(
            &serde_json::json!({ "base_model": "/nonexistent/model.gguf" }),
            Some(&sink),
            Some(token.clone()),
        )
    });

    assert!(
        result.content[0].text.contains(SHIM_MARKER),
        "apr.finetune streaming must execute the resolved binary; got: {}",
        result.content[0].text
    );

    let notifs = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        notifs.len(),
        1,
        "the shim prints exactly one stdout line, so exactly one progress \
         notification must be forwarded"
    );
    assert_eq!(
        notifs[0].params["message"]["marker"],
        serde_json::json!(SHIM_MARKER),
        "the streamed payload must come from the resolved binary"
    );
}

/// Control: the shim really is what `$APR_BIN` designates, and it really does
/// print the marker. Without this, a test above could go green for the wrong
/// reason (e.g. the marker leaking from somewhere else), and a shim that
/// never became executable would look like a passing pin.
#[test]
#[cfg(unix)]
fn shim_control_the_marker_only_exists_because_apr_bin_points_at_the_shim() {
    let resolved_without = crate::apr_bin::resolve(None, None);
    assert_eq!(
        resolved_without,
        PathBuf::from("apr"),
        "with no override and no apr-named exe, resolution falls back to $PATH"
    );

    with_apr_shim("control", || {
        let resolved = crate::apr_bin::apr_binary();
        assert_ne!(
            resolved,
            PathBuf::from("apr"),
            "$APR_BIN must win over the bare name"
        );
        let out = std::process::Command::new(&resolved)
            .arg("run")
            .output()
            .expect("shim is executable");
        assert!(
            String::from_utf8_lossy(&out.stdout).contains(SHIM_MARKER),
            "the shim must print the marker, else the pins above prove nothing"
        );
    });
}
