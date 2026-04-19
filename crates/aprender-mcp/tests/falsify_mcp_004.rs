//! FALSIFY-MCP-004 — `apr.qa` returns a JSON body whose shape matches the
//! `QaReport` serialized by `apr qa --json`.
//!
//! Spec: `docs/specifications/apr-mcp-server-spec.md` lines 79-82, 135.
//!
//! The M2 wrapper at `crates/aprender-mcp/src/tools/qa.rs` shells out to
//! `apr qa <model> --json [...]` and forwards stdout verbatim as the
//! `ToolCallResult` text payload. The existing surface test only proves
//! the tool definition is registered; this file proves the end-to-end MCP
//! response shape.
//!
//! Schema source of truth: `crates/apr-cli/src/commands/qa.rs::QaReport`
//! (Serialize + Deserialize). Fields:
//!
//! ```json
//! {
//!   "model": "...",
//!   "passed": bool,
//!   "gates": [
//!     { "name": "...", "passed": bool, "message": "...",
//!       "value": f64?, "threshold": f64?,
//!       "duration_ms": u64, "skipped": bool }, ...
//!   ],
//!   "gates_executed": N,
//!   "gates_skipped": N,
//!   "total_duration_ms": N,
//!   "timestamp": "...",
//!   "summary": "..."
//! }
//! ```
//!
//! Note: the spec phrasing is "8 gates × {pass, value, threshold}". The
//! actual CLI struct field is `passed` (not `pass`); we match the CLI.
//! Spec phrasing should be updated in a follow-up — see PR body.
//!
//! The mock-subprocess + PATH-override pattern mirrors
//! `tests/falsify_mcp_progress_001.rs` and `tests/falsify_mcp_006.rs`.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use aprender_mcp::tools::qa;
use std::io::Write;
use std::path::PathBuf;

/// Eight gate fixtures matching the real gate names emitted by
/// `apr qa`. The list of gates is stable across runs — dispatched in
/// `qa.rs::dispatch_gate` — and the spec asserts eight distinct gates.
const MOCK_GATES: &[&str] = &[
    "golden_output",
    "throughput",
    "ollama_parity",
    "gpu_speedup",
    "contract_density",
    "format_parity",
    "ptx_parity",
    "capability",
];

/// Write a mock `apr` shim to `dir` and return its parent dir (for PATH).
/// The shim prints a deterministic `QaReport` JSON fixture whose shape
/// matches the real Serialize output of `crates/apr-cli/src/commands/qa.rs`.
fn write_mock_apr_bin(dir: &std::path::Path) -> PathBuf {
    let bin_path = dir.join("apr");
    {
        let mut f = std::fs::File::create(&bin_path).expect("create mock apr");
        writeln!(f, "#!/bin/sh").expect("shebang");
        writeln!(f, "if [ \"$1\" = \"qa\" ]; then").expect("if");
        writeln!(f, "  cat <<'JSON'").expect("heredoc open");
        writeln!(f, "{{").expect("open");
        writeln!(f, "  \"model\": \"/dev/null\",").expect("model");
        writeln!(f, "  \"passed\": true,").expect("passed");
        writeln!(f, "  \"gates\": [").expect("gates open");
        for (i, name) in MOCK_GATES.iter().enumerate() {
            let comma = if i + 1 == MOCK_GATES.len() { "" } else { "," };
            // Mix value/threshold population across gates — some gates
            // have numeric measurements (throughput, gpu_speedup), others
            // are boolean (format_parity). Both must serialize cleanly.
            let (value, threshold) = match i % 3 {
                0 => ("100.0", "80.0"),
                1 => ("null", "null"),
                _ => ("1.0", "0.9"),
            };
            writeln!(f, "    {{").expect("gate open");
            writeln!(f, "      \"name\": \"{name}\",").expect("name");
            writeln!(f, "      \"passed\": true,").expect("gate passed");
            writeln!(f, "      \"message\": \"mock {name} pass\",").expect("msg");
            if value != "null" {
                writeln!(f, "      \"value\": {value},").expect("value");
            }
            if threshold != "null" {
                writeln!(f, "      \"threshold\": {threshold},").expect("threshold");
            }
            writeln!(f, "      \"duration_ms\": 10,").expect("dur");
            writeln!(f, "      \"skipped\": false").expect("skipped");
            writeln!(f, "    }}{comma}").expect("gate close");
        }
        writeln!(f, "  ],").expect("gates close");
        writeln!(f, "  \"gates_executed\": 8,").expect("exec");
        writeln!(f, "  \"gates_skipped\": 0,").expect("skip");
        writeln!(f, "  \"total_duration_ms\": 80,").expect("total");
        writeln!(f, "  \"timestamp\": \"2026-04-18T00:00:00Z\",").expect("ts");
        writeln!(f, "  \"summary\": \"8/8 gates pass\"").expect("summary");
        writeln!(f, "}}").expect("close");
        writeln!(f, "JSON").expect("heredoc close");
        writeln!(f, "  exit 0").expect("exit");
        writeln!(f, "fi").expect("fi");
        writeln!(f, "echo \"mock apr: unknown subcommand $1\" >&2").expect("err");
        writeln!(f, "exit 2").expect("exit 2");
        f.sync_all().expect("sync");
    }

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

/// RAII guard that prepends a directory to `$PATH` and restores on drop.
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
        // Edition 2021 — std::env::set_var is safe here (workspace
        // forbids `unsafe_code`). Tests are single-threaded wrt PATH.
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

/// Invoke `tools::qa::call` against the mock `apr` on PATH and parse the
/// returned `ToolCallResult` text payload back into JSON for assertions.
fn invoke_mock_qa() -> serde_json::Value {
    let tmp = tempdir_fallback();
    let bin_dir = write_mock_apr_bin(&tmp);
    let _guard = PathGuard::prepend(&bin_dir);

    let args = serde_json::json!({ "model_path": "/dev/null" });
    let result = qa::call(&args);

    assert!(
        result.is_error.is_none(),
        "mock apr qa exits 0 → ToolCallResult must be success, got: {:?}",
        result.content.first().map(|c| c.text.clone())
    );
    assert_eq!(result.content.len(), 1, "one content block");
    assert_eq!(
        result.content[0].content_type, "text",
        "MCP content blocks for apr.qa are text-typed"
    );

    serde_json::from_str::<serde_json::Value>(&result.content[0].text)
        .expect("MCP text payload must parse as JSON — apr qa --json contract")
}

/// FALSIFY-MCP-004 (gates shape): the forwarded JSON carries exactly 8
/// gate entries, each with `{name, passed, message, duration_ms, skipped}`
/// and the optional numeric `{value, threshold}` pair serialized when
/// present.
#[test]
#[cfg(unix)]
fn falsify_mcp_004_apr_qa_returns_eight_gates_with_pass_value_threshold() {
    let body = invoke_mock_qa();

    let gates = body
        .get("gates")
        .expect("apr qa --json must emit a `gates` array");
    let arr = gates
        .as_array()
        .unwrap_or_else(|| panic!("`gates` must be an array, got: {gates:?}"));
    assert_eq!(
        arr.len(),
        8,
        "apr qa ships exactly 8 gates (spec FALSIFY-MCP-004); got {}",
        arr.len()
    );

    let mut with_value_and_threshold = 0;
    for (i, gate) in arr.iter().enumerate() {
        // Required fields per `GateResult` serde schema.
        let name = gate
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("gates[{i}].name must be a string, got: {gate:?}"));
        assert!(!name.is_empty(), "gates[{i}].name must be non-empty");

        // `passed` — field name matches the CLI struct (not `pass`).
        let passed = gate.get("passed").and_then(serde_json::Value::as_bool);
        assert!(
            passed.is_some(),
            "gates[{i}].passed must be present and boolean (field name: passed, per QaReport serde), got: {gate:?}"
        );

        gate.get("message")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("gates[{i}].message must be a string, got: {gate:?}"));
        gate.get("duration_ms")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| panic!("gates[{i}].duration_ms must be u64, got: {gate:?}"));
        gate.get("skipped")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or_else(|| panic!("gates[{i}].skipped must be a boolean, got: {gate:?}"));

        // Optional numeric fields: `value` + `threshold`. When present,
        // they must be numeric. At least one gate in the fixture has both,
        // so we track it to fail loudly if the schema regresses to never
        // emitting them.
        let has_value = gate.get("value").is_some_and(|v| v.is_number());
        let has_threshold = gate.get("threshold").is_some_and(|v| v.is_number());
        if has_value && has_threshold {
            with_value_and_threshold += 1;
        }
        if let Some(v) = gate.get("value") {
            assert!(
                v.is_number(),
                "gates[{i}].value must be numeric when present, got: {v:?}"
            );
        }
        if let Some(t) = gate.get("threshold") {
            assert!(
                t.is_number(),
                "gates[{i}].threshold must be numeric when present, got: {t:?}"
            );
        }
    }
    assert!(
        with_value_and_threshold > 0,
        "at least one gate must emit both `value` and `threshold` — the spec's \
         `{{pass, value, threshold}}` triple would be meaningless otherwise"
    );
}

/// FALSIFY-MCP-004 (top-level summary): the forwarded JSON carries the
/// `QaReport` summary fields — top-level `passed`, `gates_executed`,
/// `gates_skipped`, `total_duration_ms`, `summary`, `timestamp`, `model`.
#[test]
#[cfg(unix)]
fn falsify_mcp_004_apr_qa_returns_summary_fields() {
    let body = invoke_mock_qa();

    // `passed` — top-level boolean aggregating all gates.
    let passed = body
        .get("passed")
        .and_then(serde_json::Value::as_bool)
        .expect("top-level `passed` must be present and boolean");
    assert!(
        passed,
        "mock fixture emits 8/8 passing → top-level passed: true"
    );

    // `gates_executed` / `gates_skipped` — denormalized gate counts
    // (GateResult::skipped is per-gate; these are the totals).
    let executed = body
        .get("gates_executed")
        .and_then(serde_json::Value::as_u64)
        .expect("`gates_executed` must be a u64");
    let skipped = body
        .get("gates_skipped")
        .and_then(serde_json::Value::as_u64)
        .expect("`gates_skipped` must be a u64");
    assert_eq!(
        executed + skipped,
        8,
        "executed + skipped must equal total gate count (8), got {executed} + {skipped}"
    );

    // `total_duration_ms` — monotone wall-clock measurement.
    body.get("total_duration_ms")
        .and_then(serde_json::Value::as_u64)
        .expect("`total_duration_ms` must be a u64");

    // `summary` — human-readable one-liner.
    let summary = body
        .get("summary")
        .and_then(|v| v.as_str())
        .expect("`summary` must be a string");
    assert!(!summary.is_empty(), "`summary` must be non-empty");

    // `timestamp` — ISO 8601 string (we don't parse it here, just assert
    // presence + string type; contract is enforced at the CLI side).
    body.get("timestamp")
        .and_then(|v| v.as_str())
        .expect("`timestamp` must be a string");

    // `model` — echoed source path.
    body.get("model")
        .and_then(|v| v.as_str())
        .expect("`model` must be a string");
}

/// Tiny tempdir helper — matches the pattern in
/// `falsify_mcp_progress_001.rs`.
fn tempdir_fallback() -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("apr-mcp-falsify-004-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}
