//! `apr.bench` — M2 tool. Benchmark inference throughput (tok/s, latency percentiles).
//!
//! Wraps `apr bench <model> --json [--iterations N] [--max-tokens N] [--prompt X]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

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

/// Execute `apr.bench` by spawning `apr bench <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let model_path = match crate::tools::args::require_str(args, "model_path") {
        Ok(p) => p,
        Err(e) => return e,
    };

    let mut owned: Vec<String> = vec![
        "bench".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    if let Some(n) = args.get("iterations").and_then(serde_json::Value::as_u64) {
        owned.push("--iterations".to_string());
        owned.push(n.to_string());
    }
    if let Some(n) = args.get("max_tokens").and_then(serde_json::Value::as_u64) {
        owned.push("--max-tokens".to_string());
        owned.push(n.to_string());
    }
    let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    if !prompt.is_empty() {
        owned.push("--prompt".to_string());
        owned.push(prompt.to_string());
    }

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
}
