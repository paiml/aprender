//! `apr.tensors` — M2 tool. List tensor names, shapes, and (optionally) stats.
//!
//! Wraps `apr tensors <model> --json [--stats] [--filter <pat>] [--limit <n>]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::args::{self, try_arg};
use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.tensors";

/// Return the MCP tool definition for `apr.tensors`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_TENSORS_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn tensors_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_TENSORS_SCHEMA)
        .expect(
            "FALSIFY-MCP-008: apr.tensors codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
        );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_TENSORS_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Build the `apr tensors ...` argv from `tools/call` arguments.
///
/// # Errors
/// Returns the client-facing message when an argument is present but not
/// usable at its declared type.
pub fn build_argv(args: &serde_json::Value) -> Result<Vec<String>, String> {
    let model_path = args::required_str(args, "model_path")?;

    let mut argv: Vec<String> = vec![
        "tensors".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];
    if args::opt_bool(args, "stats")?.unwrap_or(false) {
        argv.push("--stats".to_string());
    }
    // `--flag=value` — see [`args::flag`]. `{"filter": "-norm"}` used to reach
    // clap as `--filter -norm` and die with `unexpected argument '-n'`.
    let filter = args::opt_str(args, "filter")?.unwrap_or("");
    if !filter.is_empty() {
        argv.push(args::flag("filter", filter));
    }
    Ok(argv)
}

/// Execute `apr.tensors` by spawning `apr tensors <model> --json [...flags]`.
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
    definition: tensors_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = tensors_tool_definition();
        assert_eq!(def.name, "apr.tensors");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        assert!(def.input_schema.properties.contains_key("model_path"));
        assert!(def.input_schema.properties.contains_key("stats"));
        assert!(def.input_schema.properties.contains_key("filter"));
    }

    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }

    /// #2403 — `stats: "true"` used to drop `--stats`, so the caller got a
    /// bare listing back and no indication its request was ignored.
    #[test]
    fn string_stats_still_asks_for_stats() {
        let argv = build_argv(&serde_json::json!({ "model_path": "m.gguf", "stats": "true" }))
            .expect("\"true\" is a usable boolean");
        assert!(argv.contains(&"--stats".to_string()), "{argv:?}");
    }

    #[test]
    fn unusable_stats_is_an_error_not_a_dropped_flag() {
        let result = call(&serde_json::json!({ "model_path": "m.gguf", "stats": 1 }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("stats"));
    }

    /// #2419: `stats:"yes"` returned a success payload with no mean/std/min/max
    /// and no diagnostic — the caller believed statistics had been checked.
    #[test]
    fn non_boolean_stats_is_rejected_rather_than_read_as_false() {
        let result = call(&serde_json::json!({
            "model_path": "/nonexistent/model.gguf",
            "stats": "yes",
        }));
        assert_eq!(result.is_error, Some(true));
        // Substance, not exact wording: the message must name the argument,
        // say it was invalid, and quote what was actually received — so a
        // client can correct itself. The precise phrasing lives in args.rs and
        // is asserted there.
        let text = &result.content[0].text;
        assert!(text.contains("stats"), "must name the argument: {text}");
        assert!(
            text.contains("boolean"),
            "must state the expected type: {text}"
        );
        assert!(text.contains("yes"), "must quote what was received: {text}");
    }

    /// `{"filter": "-norm"}` reached clap as the two tokens `--filter -norm`
    /// and the call died with `unexpected argument '-n' found` (exit 2), so a
    /// perfectly ordinary substring pattern was untransmittable over MCP. The
    /// `=` form puts it in one token.
    ///
    /// Mutation-verified: reverting `build_argv` to the two-token push turns
    /// this RED. It exists because the first pass of this work had no such
    /// test here, and the tensors mutation survived while its five siblings
    /// went red.
    #[test]
    fn a_filter_beginning_with_a_hyphen_survives_argv_encoding() {
        let argv = build_argv(&serde_json::json!({ "model_path": "m.gguf", "filter": "-norm" }))
            .expect("any string is a usable filter");
        assert_eq!(argv, vec!["tensors", "m.gguf", "--json", "--filter=-norm"]);
    }

    /// Positive control for the pattern above: an ordinary filter is encoded
    /// the same way, so the `=` form is the rule and not a special case.
    #[test]
    fn an_ordinary_filter_uses_the_same_single_token_form() {
        let argv = build_argv(&serde_json::json!({ "model_path": "m.gguf", "filter": "norm" }))
            .expect("valid");
        assert_eq!(argv, vec!["tensors", "m.gguf", "--json", "--filter=norm"]);
    }

    #[test]
    fn non_string_filter_is_rejected() {
        let result = call(&serde_json::json!({
            "model_path": "/nonexistent/model.gguf",
            "filter": 7,
        }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("Invalid filter"));
    }
}
