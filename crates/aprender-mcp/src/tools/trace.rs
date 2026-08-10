//! `apr.trace` — M2 tool. Layer-by-layer tensor trace for debugging a model.
//!
//! Wraps `apr trace <model> --json [--layer <pat>]`.
//!
//! #2407: the tool used to advertise a `reference` argument in its
//! `inputSchema` and forward it as `apr trace --reference <path>`, which is a
//! stub: it printed `{"comparison": "reference comparison not yet
//! implemented"}` and exited 0. The wrapper drops stderr on success, so an
//! MCP client saw a plain success result for a comparison that never
//! happened. `reference` is no longer advertised, and supplying it now
//! returns `isError`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::args::{opt_str, reject_unsupported, require_str};
use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Why `reference` is refused (see module docs, #2407).
const REFERENCE_UNSUPPORTED: &str =
    "layer-by-layer comparison against a reference model is not implemented; \
     `apr trace --reference` returns a stub, so this tool would report success \
     for a comparison it never performed";

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.trace";

/// Return the MCP tool definition for `apr.trace`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_TRACE_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn trace_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_TRACE_SCHEMA).expect(
        "FALSIFY-MCP-008: apr.trace codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_TRACE_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.trace` by spawning `apr trace <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let model_path = match require_str(args, "model_path") {
        Ok(p) => p,
        Err(e) => return e,
    };
    if let Err(e) = reject_unsupported(args, "reference", REFERENCE_UNSUPPORTED) {
        return e;
    }
    let layer = match opt_str(args, "layer") {
        Ok(l) => l.unwrap_or(""),
        Err(e) => return e,
    };

    let mut owned: Vec<String> = vec![
        "trace".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    if !layer.is_empty() {
        owned.push("--layer".to_string());
        owned.push(layer.to_string());
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
    definition: trace_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    /// #2407: this test used to require `reference` to be an advertised
    /// property, which is what made the unimplemented option discoverable in
    /// the first place. It now asserts the opposite: a client must not be
    /// invited to pass an argument the tool cannot honour.
    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = trace_tool_definition();
        assert_eq!(def.name, "apr.trace");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "layer"] {
            assert!(
                def.input_schema.properties.contains_key(field),
                "property {field} present"
            );
        }
        assert!(
            !def.input_schema.properties.contains_key("reference"),
            "`reference` is not implemented and must not be advertised"
        );
    }

    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }

    /// #2407: supplying `reference` returned `isError: None` and a body of
    /// `{"comparison": "reference comparison not yet implemented"}` — a
    /// success result for an operation that did nothing, with the `layer`
    /// filter and the whole trace payload silently discarded.
    #[test]
    fn reference_argument_is_refused_instead_of_returning_a_stub_success() {
        let result = call(&serde_json::json!({
            "model_path": "/nonexistent/model.gguf",
            "reference": "/nonexistent/reference.apr",
        }));
        assert_eq!(
            result.is_error,
            Some(true),
            "an unimplemented comparison must not be reported as success"
        );
        let text = &result.content[0].text;
        assert!(
            text.contains("Unsupported argument: reference"),
            "the error must name the argument that was refused; got: {text}"
        );
        assert!(
            !text.contains("not yet implemented\"}"),
            "the stub body must not be relayed as the result; got: {text}"
        );
    }

    #[test]
    fn non_string_layer_is_rejected() {
        let result = call(&serde_json::json!({
            "model_path": "/nonexistent/model.gguf",
            "layer": 3,
        }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("Invalid layer"));
    }
}
