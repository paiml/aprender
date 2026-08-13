//! `apr.qa` — M2 tool. The 8-gate quality checklist; first stop for any model issue.
//!
//! Wraps `apr qa <model> --json [--assert-tps N] [--max-tokens N] [--iterations N]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::args::{self, try_arg};
use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.qa";

/// Return the MCP tool definition for `apr.qa`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_QA_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn qa_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_QA_SCHEMA).expect(
        "FALSIFY-MCP-008: apr.qa codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_QA_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Build the `apr qa ...` argv from `tools/call` arguments.
///
/// Separated from [`call`] so falsifiers can assert what actually reaches the
/// CLI rather than merely that the call returned something.
///
/// # Errors
/// Returns the client-facing message when an argument is present but not
/// usable at its declared type.
pub fn build_argv(args: &serde_json::Value) -> Result<Vec<String>, String> {
    let model_path = args::required_str(args, "model_path")?;

    let mut owned: Vec<String> = vec![
        "qa".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    // A wrong-typed assert_tps used to vanish here, disarming the throughput
    // gate without any diagnostic (#2403). `--flag=value` — see
    // [`args::flag`]: `assert_tps` is a `"type": "number"`, so a negative
    // value is schema-valid and the two-token form turned it into a clap
    // parse error instead of a threshold.
    if let Some(tps) = args::opt_f64(args, "assert_tps")? {
        owned.push(args::flag("assert-tps", tps));
    }
    if let Some(n) = args::opt_u64(args, "max_tokens")? {
        owned.push(args::flag("max-tokens", n));
    }
    if let Some(n) = args::opt_u64(args, "iterations")? {
        owned.push(args::flag("iterations", n));
    }
    Ok(owned)
}

/// Execute `apr.qa` by spawning `apr qa <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let owned = try_arg!(build_argv(args));
    let argv: Vec<&str> = owned.iter().map(String::as_str).collect();
    run_apr(&argv)
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
    definition: qa_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = qa_tool_definition();
        assert_eq!(def.name, "apr.qa");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "assert_tps", "max_tokens", "iterations"] {
            assert!(
                def.input_schema.properties.contains_key(field),
                "property {field} present"
            );
        }
    }

    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }

    /// #2403/#2418 — the sharpest case in the audit. `assert_tps` is a
    /// pass/fail threshold; sent as a JSON string it used to vanish from argv,
    /// turning a gate that should fail into one that cannot fail, silently.
    #[test]
    fn assert_tps_as_a_json_string_still_reaches_the_gate() {
        let argv = build_argv(&serde_json::json!({
            "model_path": "m.gguf",
            "assert_tps": "100000",
            "iterations": 1
        }))
        .expect("string 100000 is a usable number");
        assert_eq!(
            argv,
            vec![
                "qa",
                "m.gguf",
                "--json",
                "--assert-tps=100000",
                "--iterations=1"
            ],
            "throughput gate disarmed"
        );
    }

    /// A number still works exactly as before — the positive control the
    /// audit ran alongside the string case.
    #[test]
    fn assert_tps_as_a_number_reaches_the_gate() {
        let argv =
            build_argv(&serde_json::json!({ "model_path": "m.gguf", "assert_tps": 100_000 }))
                .expect("number is usable");
        assert_eq!(argv, vec!["qa", "m.gguf", "--json", "--assert-tps=100000"]);
    }

    /// A NEGATIVE `assert_tps` is a schema-valid `"type": "number"`, and under
    /// the two-token argv form it reached clap as a bare `-1` and died with
    /// `unexpected argument '-1' found` — a parser error standing in for what
    /// should be a threshold. The `=` form transmits it, so the CLI (not the
    /// argv encoding) decides what a negative threshold means.
    #[test]
    fn negative_assert_tps_survives_argv_encoding() {
        let argv = build_argv(&serde_json::json!({ "model_path": "m.gguf", "assert_tps": -1 }))
            .expect("-1 is a usable number");
        assert_eq!(argv, vec!["qa", "m.gguf", "--json", "--assert-tps=-1"]);
    }

    /// An assert_tps that cannot mean a threshold is an error the client can
    /// see, never a silently disarmed gate.
    #[test]
    fn unusable_assert_tps_is_an_error_not_a_dropped_flag() {
        let result = call(&serde_json::json!({ "model_path": "m.gguf", "assert_tps": "fast" }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("assert_tps"));
    }

    #[test]
    fn omitted_assert_tps_stays_omitted() {
        let argv = build_argv(&serde_json::json!({ "model_path": "m.gguf" })).expect("valid");
        assert_eq!(argv, vec!["qa", "m.gguf", "--json"]);
    }
}
