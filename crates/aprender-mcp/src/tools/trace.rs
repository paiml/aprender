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

use crate::tools::args::{self, try_arg};
use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

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

/// Build the `apr trace ...` argv from `tools/call` arguments.
///
/// # Errors
/// Returns the client-facing message when an argument is present but not
/// usable at its declared type.
pub fn build_argv(args: &serde_json::Value) -> Result<Vec<String>, String> {
    let model_path = args::required_str(args, "model_path")?;

    let mut owned: Vec<String> = vec![
        "trace".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    // `--flag=value` — see [`args::flag`].
    if let Some(pat) = args::opt_str(args, "layer")? {
        if !pat.is_empty() {
            owned.push(args::flag("layer", pat));
        }
    }
    if let Some(ref_path) = args::opt_str(args, "reference")? {
        if !ref_path.is_empty() {
            owned.push(args::flag("reference", ref_path));
        }
    }
    Ok(owned)
}

/// Execute `apr.trace` by spawning `apr trace <model> --json [...flags]`.
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

    /// #2403 — `{"layer": 7, "reference": 42}` dropped BOTH flags and traced
    /// the whole model against nothing, reported as a success.
    #[test]
    fn integer_layer_and_reference_are_errors_not_dropped_flags() {
        let result = call(&serde_json::json!({ "model_path": "m.gguf", "layer": 7 }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("layer"));

        let result = call(&serde_json::json!({ "model_path": "m.gguf", "reference": 42 }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("reference"));
    }

    #[test]
    fn string_layer_and_reference_reach_the_cli() {
        let argv = build_argv(&serde_json::json!({
            "model_path": "m.gguf",
            "layer": "blk.7",
            "reference": "ref.gguf"
        }))
        .expect("strings are usable");
        assert_eq!(
            argv,
            vec![
                "trace",
                "m.gguf",
                "--json",
                "--layer=blk.7",
                "--reference=ref.gguf"
            ]
        );
    }

    /// A layer pattern beginning with `-` reached clap as `--layer -norm` and
    /// died with `unexpected argument '-n' found`; the `=` form transmits it.
    #[test]
    fn a_layer_pattern_beginning_with_a_hyphen_survives_argv_encoding() {
        let argv = build_argv(&serde_json::json!({ "model_path": "m.gguf", "layer": "-norm" }))
            .expect("any string is a usable pattern");
        assert_eq!(argv, vec!["trace", "m.gguf", "--json", "--layer=-norm"]);
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
