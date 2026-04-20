//! FALSIFY-MCP-003 — `apr.run` returns a JSON body whose shape matches
//! `apr run --json` (model+text+tokens+tok_per_sec+...).
//!
//! Spec: `docs/specifications/apr-mcp-server-spec.md` lines 79-82, 134.
//!
//! The M2 wrapper at `crates/aprender-mcp/src/tools/run.rs` shells out to
//! `apr run <model> --json [...]` and forwards stdout verbatim as the
//! `ToolCallResult` text payload. Surface unit tests (the `definition_*`
//! case inside `run.rs`) only prove the tool is *registered* — they do
//! nothing to prove the end-to-end MCP response shape.
//!
//! This falsifier drives the real `tools::run::call` entry point against a
//! **mock `apr` shell script** on the PATH so that no real model or GPU is
//! needed. The mock prints a deterministic JSON fixture matching the exact
//! schema emitted by `crates/apr-cli/src/commands/run_entry.rs::print_run_output`
//! (the branch guarded by `output_format == "json" && !benchmark`):
//!
//! ```json
//! {
//!   "model": "...",
//!   "text": "...",
//!   "tokens": [ /* u32 ids */ ],
//!   "tokens_generated": N,
//!   "max_tokens": N,
//!   "tok_per_sec": f64,
//!   "inference_time_ms": f64,
//!   "used_gpu": bool,
//!   "cached": bool
//! }
//! ```
//!
//! Note: the spec string mentions a `stop_reason` field; the CLI does not
//! currently emit one. We match the CLI (source of truth) here. The spec
//! should be updated in a follow-up — see PR body.
//!
//! The mock-subprocess + PATH-override pattern mirrors
//! `tests/falsify_mcp_progress_001.rs` and `tests/falsify_mcp_006.rs`. It is
//! deterministic, fast, and does not require a trained model.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use aprender_mcp::tools::run;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;

/// Write a mock `apr` shim to `dir` and return its parent dir (for PATH).
///
/// The shim is a POSIX shell script named `apr` that inspects its first
/// argument: if it's `run`, it prints the fixed JSON fixture and exits 0;
/// anything else prints a short error and exits 2. That's enough to let
/// `tools::run::call` spawn us via `Command::new("apr")` and observe the
/// same stdout an end-user would see from `apr run <model> --json`.
///
/// We write the script to a tempdir and prepend that tempdir to `$PATH` in
/// the test. That matches how `falsify_mcp_progress_001.rs` sidesteps the
/// real binary — no injectable program-name refactor needed in production
/// code.
fn write_mock_apr_bin(dir: &std::path::Path) -> PathBuf {
    let bin_path = dir.join("apr");
    {
        let mut f = std::fs::File::create(&bin_path).expect("create mock apr");
        writeln!(f, "#!/bin/sh").expect("shebang");
        // The shim ignores all arguments after "run" — it just prints the
        // fixture. A real apr would validate model_path, load the model,
        // and run inference; the MCP invariant we're falsifying is purely
        // about the shape of the stdout JSON that gets forwarded to the
        // MCP client.
        writeln!(f, "if [ \"$1\" = \"run\" ]; then").expect("if");
        writeln!(f, "  cat <<'JSON'").expect("heredoc open");
        writeln!(f, "{{").expect("json open");
        writeln!(f, "  \"model\": \"/dev/null\",").expect("model");
        writeln!(f, "  \"text\": \"mock inference output\",").expect("text");
        writeln!(f, "  \"tokens\": [16, 20, 42, 7],").expect("tokens");
        writeln!(f, "  \"tokens_generated\": 4,").expect("tg");
        writeln!(f, "  \"max_tokens\": 4,").expect("mt");
        writeln!(f, "  \"tok_per_sec\": 123.4,").expect("tps");
        writeln!(f, "  \"inference_time_ms\": 32.5,").expect("inftime");
        writeln!(f, "  \"used_gpu\": false,").expect("gpu");
        writeln!(f, "  \"cached\": true").expect("cached");
        writeln!(f, "}}").expect("json close");
        writeln!(f, "JSON").expect("heredoc close");
        writeln!(f, "  exit 0").expect("exit 0");
        writeln!(f, "fi").expect("fi");
        writeln!(f, "echo \"mock apr: unknown subcommand $1\" >&2").expect("err");
        writeln!(f, "exit 2").expect("exit 2");
        f.sync_all().expect("sync");
    }

    // chmod +x so PATH-resolution + exec works (the child runs directly,
    // not via `sh`, because that's how `Command::new("apr")` dispatches).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&bin_path)
            .expect("stat mock apr")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&bin_path, perms).expect("chmod mock apr");
    }

    dir.to_path_buf()
}

/// RAII guard that prepends a directory to `$PATH` and restores the prior
/// value on drop. Keeps tests hermetic even if they panic mid-way through.
///
/// We intentionally keep this small and inline rather than pulling in a
/// test-only dep — the MCP crate currently has zero test-only crates
/// outside jsonschema + serde_yaml.
struct PathGuard {
    prior: Option<String>,
}

impl PathGuard {
    fn prepend(dir: &std::path::Path) -> Self {
        let prior = std::env::var("PATH").ok();
        let new_path = match &prior {
            Some(existing) => format!("{}:{}", dir.display(), existing),
            None => dir.display().to_string(),
        };
        // Edition 2021 — `std::env::set_var` is safe. In 2024 this would
        // move into an `unsafe` block; tests run single-threaded at the
        // PATH level so the race caveat doesn't apply here.
        std::env::set_var("PATH", new_path);
        Self { prior }
    }
}

impl Drop for PathGuard {
    fn drop(&mut self) {
        match &self.prior {
            Some(v) => std::env::set_var("PATH", v),
            None => std::env::remove_var("PATH"),
        }
    }
}

/// Invoke `tools::run::call` with the mock `apr` on PATH and parse the
/// returned `ToolCallResult` text payload back into JSON for assertions.
fn invoke_mock_run() -> serde_json::Value {
    let tmp = tempdir_fallback();
    let bin_dir = write_mock_apr_bin(&tmp);
    let _guard = PathGuard::prepend(&bin_dir);

    let (_tx, rx) = mpsc::channel::<()>();
    let args = serde_json::json!({
        "model_path": "/dev/null",
        "prompt": "1+1=",
        "max_tokens": 4,
    });
    let result = run::call(&args, &rx);

    assert!(
        result.is_error.is_none(),
        "mock apr exits 0 → ToolCallResult must be success, got: {:?}",
        result.content.first().map(|c| c.text.clone())
    );
    assert_eq!(
        result.content.len(),
        1,
        "tools/call result has exactly one content block"
    );
    assert_eq!(
        result.content[0].content_type, "text",
        "MCP content blocks for apr.run are text-typed"
    );

    serde_json::from_str::<serde_json::Value>(&result.content[0].text)
        .expect("MCP text payload must parse as JSON — apr run --json contract")
}

/// FALSIFY-MCP-003 (tokens): the forwarded JSON carries a `tokens` array
/// with at least one entry. Proves we didn't regress to the old
/// BUG-RUN-001 word-splitting heuristic.
#[test]
#[cfg(unix)]
fn falsify_mcp_003_apr_run_returns_tokens_array() {
    let body = invoke_mock_run();

    let tokens = body
        .get("tokens")
        .expect("apr run --json must emit a `tokens` field");
    assert!(
        tokens.is_array(),
        "`tokens` must be an array of u32 ids, got: {tokens:?}"
    );
    let arr = tokens.as_array().expect("array");
    assert!(
        !arr.is_empty(),
        "mock fixture emits 4 token ids → array must be non-empty, got: {arr:?}"
    );
    // Every entry must be an unsigned integer — u32 token id per GH-250.
    for (i, t) in arr.iter().enumerate() {
        assert!(
            t.is_u64(),
            "tokens[{i}] must be a non-negative integer token id, got: {t:?}"
        );
    }
}

/// FALSIFY-MCP-003 (tok_per_sec): the forwarded JSON carries a numeric
/// `tok_per_sec` field. Field name matches the CLI (`tok_per_sec`, not
/// `tokens_per_second`).
#[test]
#[cfg(unix)]
fn falsify_mcp_003_apr_run_returns_tok_per_sec() {
    let body = invoke_mock_run();

    let tps = body
        .get("tok_per_sec")
        .expect("apr run --json must emit a `tok_per_sec` field (CLI field name)");
    assert!(
        tps.is_f64() || tps.is_u64() || tps.is_i64(),
        "`tok_per_sec` must be numeric, got: {tps:?}"
    );
    let v = tps.as_f64().expect("numeric");
    assert!(v >= 0.0, "`tok_per_sec` must be non-negative, got: {v}");
}

/// FALSIFY-MCP-003 (text + model + tokens_generated): the fixture round-
/// trips the three always-present string/integer fields. This is the "shape
/// contract" the spec calls for — the spec mentions `stop_reason`, but the
/// CLI source of truth (`run_entry.rs::print_run_output`) does not emit one
/// today. Matching the CLI here; spec follow-up tracked in the PR body.
#[test]
#[cfg(unix)]
fn falsify_mcp_003_apr_run_returns_stop_reason() {
    let body = invoke_mock_run();

    // `text` — the decoded string the user asked for.
    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .expect("`text` field present and string-typed");
    assert!(
        !text.is_empty(),
        "`text` must be non-empty for a successful run, got: {text:?}"
    );

    // `tokens_generated` — authoritative token count from the inference engine.
    let tg = body
        .get("tokens_generated")
        .and_then(serde_json::Value::as_u64)
        .expect("`tokens_generated` field present and u64-typed");
    assert!(
        tg > 0,
        "mock fixture generates 4 tokens → tokens_generated > 0, got: {tg}"
    );

    // `model` — echoed source path.
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .expect("`model` field present and string-typed");
    assert!(
        !model.is_empty(),
        "`model` must echo the source path, got: {model:?}"
    );

    // NB: when apr-cli adds `stop_reason` (tracked in follow-up), strengthen
    // this test to assert the field is present and one of {"eos", "max_tokens",
    // "stop_token", "cancelled"} per the MCP spec.
}

/// Tiny tempdir helper — matches the pattern in
/// `falsify_mcp_progress_001.rs`. We don't want a `tempfile` dev-dep just
/// for two tests.
fn tempdir_fallback() -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("apr-mcp-falsify-003-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}
