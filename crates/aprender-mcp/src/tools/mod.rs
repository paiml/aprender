//! MCP tool implementations for aprender.
//!
//! Phase-1 surface (shipped M1–M3):
//! - M1 scaffold: `apr.version`.
//! - M2 subprocess wrappers around `apr <cmd> --json`: `apr.validate`,
//!   `apr.tensors`, `apr.bench`, `apr.qa`, `apr.trace`, `apr.run`, `apr.serve`.
//! - M3 streaming slice: `apr.finetune` (opt-in `notifications/progress`
//!   per non-empty stdout line when `params._meta.progressToken` is set —
//!   see FALSIFY-MCP-PROGRESS-001) + `notifications/cancelled` →
//!   SIGTERM→SIGKILL for `apr.run` (FALSIFY-MCP-006).

pub mod bench;
pub mod finetune;
pub mod qa;
pub mod registry;
pub mod run;
pub mod serve;
pub mod subprocess;
pub mod tensors;
pub mod trace;
pub mod validate;
pub mod version;

pub use registry::{DispatchFn, McpToolEntry, ToolIndex};

/// Extract a required string argument from a `tools/call` `arguments` object.
///
/// FALSIFY-MCP-013. Every wrapper used to do
/// `args.get(name).and_then(|v| v.as_str())` and report the `None` as
/// `"Missing required argument: <name>"`. `and_then(as_str)` collapses two
/// distinct failures into one, so a client that sent `{"model_path": 123}`
/// — present, but a number where the declared `inputSchema` says
/// `{"type":"string"}` — was told the key was missing. An LLM handed
/// "missing" retries by adding a key it already sent, and loops.
///
/// The absent case keeps its exact original wording so existing clients and
/// tests that match on it are unaffected; only the wrong-type case changes.
///
/// # Errors
/// Returns the client-facing message to put in a `ToolCallResult::error`.
pub fn require_str<'a>(args: &'a serde_json::Value, name: &str) -> Result<&'a str, String> {
    match args.get(name) {
        Some(serde_json::Value::String(s)) => Ok(s),
        Some(other) => Err(format!(
            "Argument {name} must be a string, got {}",
            crate::types::json_type_name(other)
        )),
        None => Err(format!("Missing required argument: {name}")),
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::require_str;

    /// FALSIFY-MCP-013: an argument supplied with the WRONG TYPE must be
    /// distinguishable from an argument that is absent. 0.63.0 reported
    /// `{"model_path": 123}` — present, but a number where the declared
    /// inputSchema says `{"type":"string"}` — with the byte-identical
    /// message it used for `{}`, so a client told "missing" retried by
    /// adding a key it had already sent.
    #[test]
    fn wrong_typed_argument_is_not_reported_as_missing() {
        let absent = require_str(&serde_json::json!({}), "model_path")
            .expect_err("absent argument must be an error");
        let wrong_type = require_str(&serde_json::json!({ "model_path": 123 }), "model_path")
            .expect_err("wrong-typed argument must be an error");

        assert_ne!(
            absent, wrong_type,
            "an argument that WAS sent must not report the same message as one that was not"
        );
        assert!(
            !wrong_type.contains("Missing"),
            "wrong-typed argument must not be described as missing, got: {wrong_type}"
        );
        assert!(
            wrong_type.contains("number"),
            "message must name the type actually received, got: {wrong_type}"
        );
        assert!(
            wrong_type.contains("model_path"),
            "message must name the offending argument, got: {wrong_type}"
        );
    }

    /// The absent-argument wording is unchanged, so clients and tests that
    /// match on it are unaffected by the fix.
    #[test]
    fn absent_argument_keeps_its_original_wording() {
        let err = require_str(&serde_json::json!({}), "base_model")
            .expect_err("absent argument must be an error");
        assert_eq!(err, "Missing required argument: base_model");
    }

    /// Every non-string JSON type is named, not just numbers.
    #[test]
    fn each_wrong_type_is_named() {
        for (value, expected) in [
            (serde_json::json!(1), "number"),
            (serde_json::json!(true), "boolean"),
            (serde_json::json!(null), "null"),
            (serde_json::json!([]), "array"),
            (serde_json::json!({}), "object"),
        ] {
            let args = serde_json::json!({ "model_path": value });
            let err = require_str(&args, "model_path").expect_err("must be an error");
            assert!(
                err.contains(expected),
                "expected the message to name `{expected}`, got: {err}"
            );
        }
    }

    #[test]
    fn string_argument_is_returned_verbatim() {
        let args = serde_json::json!({ "model_path": "/models/m.gguf" });
        assert_eq!(
            require_str(&args, "model_path").expect("string argument"),
            "/models/m.gguf"
        );
    }
}

pub use bench::bench_tool_definition;
pub use finetune::finetune_tool_definition;
pub use qa::qa_tool_definition;
pub use run::run_tool_definition;
pub use serve::serve_tool_definition;
pub use tensors::tensors_tool_definition;
pub use trace::trace_tool_definition;
pub use validate::validate_tool_definition;
pub use version::version_tool_definition;
