//! `apr.bench` — M2 tool. Benchmark inference throughput (tok/s, latency percentiles).
//!
//! Wraps `apr bench <model> --json [--iterations N] [--max-tokens N] [--prompt X]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::args::{self, try_arg};
use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.bench";

/// Return the MCP tool definition for `apr.bench`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_BENCH_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn bench_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_BENCH_SCHEMA).expect(
        "FALSIFY-MCP-008: apr.bench codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_BENCH_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Build the `apr bench ...` argv from `tools/call` arguments.
///
/// # Errors
/// Returns the client-facing message when an argument is present but not
/// usable at its declared type.
pub fn build_argv(args: &serde_json::Value) -> Result<Vec<String>, String> {
    let model_path = args::required_str(args, "model_path")?;

    let mut owned: Vec<String> = vec![
        "bench".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    // `--flag=value` — see [`args::flag`]: a bench prompt beginning with `-`
    // is ordinary user text that the two-token form makes untransmittable.
    if let Some(n) = args::opt_u64(args, "iterations")? {
        owned.push(args::flag("iterations", n));
    }
    if let Some(n) = args::opt_u64(args, "max_tokens")? {
        owned.push(args::flag("max-tokens", n));
    }
    let prompt = args::opt_str(args, "prompt")?.unwrap_or("");
    if !prompt.is_empty() {
        owned.push(args::flag("prompt", prompt));
    }
    Ok(owned)
}

/// Execute `apr.bench` by spawning `apr bench <model> --json [...flags]`.
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
    definition: bench_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = bench_tool_definition();
        assert_eq!(def.name, "apr.bench");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "iterations", "max_tokens", "prompt"] {
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

    /// #2403 — the audit asked for 1 iteration of 8 tokens as JSON strings and
    /// got 5 iterations of 32 tokens (the CLI defaults), reported back as if
    /// that is what it had asked for.
    #[test]
    fn string_iterations_and_max_tokens_reach_the_cli() {
        let argv = build_argv(&serde_json::json!({
            "model_path": "m.gguf",
            "iterations": "1",
            "max_tokens": "8"
        }))
        .expect("numeric strings are usable");
        assert_eq!(
            argv,
            vec![
                "bench",
                "m.gguf",
                "--json",
                "--iterations=1",
                "--max-tokens=8"
            ]
        );
    }

    #[test]
    fn unusable_iterations_is_an_error_not_a_dropped_flag() {
        let result = call(&serde_json::json!({ "model_path": "m.gguf", "iterations": "lots" }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("iterations"));
    }

    /// A benchmark prompt beginning with `-` is ordinary user text; the
    /// two-token argv form made it a clap parse error. Mirrors
    /// `run::tests::a_prompt_beginning_with_a_hyphen_survives_argv_encoding`.
    #[test]
    fn a_prompt_beginning_with_a_hyphen_survives_argv_encoding() {
        let argv =
            build_argv(&serde_json::json!({ "model_path": "m.gguf", "prompt": "-1 + 2 equals" }))
                .expect("any string is a usable prompt");
        assert_eq!(
            argv,
            vec!["bench", "m.gguf", "--json", "--prompt=-1 + 2 equals"]
        );
    }
}
