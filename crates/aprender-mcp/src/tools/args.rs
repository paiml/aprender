//! Typed argument extraction for the `apr.*` subprocess wrappers.
//!
//! #2419: every wrapper read its arguments with
//! `args.get("x").and_then(|v| v.as_str())`, which collapses "absent" and
//! "present but the wrong JSON type" into the same `None`. Three distinct
//! wrong answers followed from that one line:
//!
//! * `apr.validate {"model_path": 123}` → `isError: true, "Missing required
//!   argument: model_path"`. It is not missing. The caller is told to add an
//!   argument it already sent.
//! * `apr.tensors {"stats": "yes"}` → success, with no statistics in the
//!   report and nothing saying why (`"yes"` is not a JSON boolean, so
//!   `as_bool()` returned `None` and the default `false` won).
//! * `apr.run {"max_tokens": "eight"}` → success, 32 tokens generated. The
//!   requested limit was replaced by the default with no diagnostic.
//!
//! `apr.serve`'s port handling was the one place that got this right
//! (`"Invalid port: expected integer 0..=65535, got \"eighteen\""`); these
//! helpers generalise it so a declared argument is never silently dropped.
//!
//! Every helper returns `Err(ToolCallResult)` — an `isError: true` result
//! naming the argument, the expected type, and the value actually received —
//! so a call site reads `let p = match require_str(args, "model_path") { Ok(p)
//! => p, Err(e) => return e };`.

use crate::types::ToolCallResult;
use serde_json::Value;

/// Human name of a JSON value's type, for error messages.
fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Build the `isError` result for a present-but-wrong-typed argument.
fn type_error(name: &str, expected: &str, got: &Value) -> ToolCallResult {
    ToolCallResult::error(format!(
        "Invalid {name}: expected {expected}, got {} {got}",
        type_name(got)
    ))
}

/// Read a required string argument.
///
/// `Err` distinguishes the two failures the caller cares about: the argument
/// is absent, or it is present with the wrong type.
///
/// # Errors
/// Returns an `isError` [`ToolCallResult`] when `name` is absent or is not a
/// JSON string.
pub fn require_str<'a>(args: &'a Value, name: &str) -> Result<&'a str, ToolCallResult> {
    match args.get(name) {
        None | Some(Value::Null) => Err(ToolCallResult::error(format!(
            "Missing required argument: {name}"
        ))),
        Some(Value::String(s)) => Ok(s.as_str()),
        Some(other) => Err(type_error(name, "string", other)),
    }
}

/// Read an optional string argument.
///
/// # Errors
/// Returns an `isError` [`ToolCallResult`] when `name` is present and is not
/// a JSON string.
pub fn opt_str<'a>(args: &'a Value, name: &str) -> Result<Option<&'a str>, ToolCallResult> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(other) => Err(type_error(name, "string", other)),
    }
}

/// Read an optional boolean argument, defaulting to `false` when absent.
///
/// # Errors
/// Returns an `isError` [`ToolCallResult`] when `name` is present and is not
/// a JSON boolean — notably the string `"yes"`, which used to be read as
/// `false`.
pub fn opt_bool(args: &Value, name: &str) -> Result<bool, ToolCallResult> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(other) => Err(type_error(name, "boolean true or false", other)),
    }
}

/// Read an optional non-negative integer argument.
///
/// # Errors
/// Returns an `isError` [`ToolCallResult`] when `name` is present and is not
/// a non-negative JSON integer — including `"8"`, which used to fall back to
/// the tool's default.
pub fn opt_u64(args: &Value, name: &str) -> Result<Option<u64>, ToolCallResult> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) => Ok(Some(n)),
            None => Err(type_error(name, "a non-negative integer", v)),
        },
    }
}

/// Read an optional finite floating-point argument.
///
/// # Errors
/// Returns an `isError` [`ToolCallResult`] when `name` is present and is not
/// a JSON number.
pub fn opt_f64(args: &Value, name: &str) -> Result<Option<f64>, ToolCallResult> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(n) => Ok(Some(n)),
            None => Err(type_error(name, "a number", v)),
        },
    }
}

/// Reject an argument the tool does not implement.
///
/// #2407: `apr.trace` accepted a `reference` path, forwarded it to a CLI
/// stub, and returned the stub's output as a success. A tool that cannot
/// honour an argument must say so rather than appear to have used it.
///
/// # Errors
/// Returns an `isError` [`ToolCallResult`] when `name` is present and not
/// null.
pub fn reject_unsupported(args: &Value, name: &str, reason: &str) -> Result<(), ToolCallResult> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(()),
        Some(_) => Err(ToolCallResult::error(format!(
            "Unsupported argument: {name} — {reason}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    fn err_text(r: &ToolCallResult) -> &str {
        r.content[0].text.as_str()
    }

    #[test]
    fn require_str_accepts_a_string() {
        let args = serde_json::json!({ "model_path": "/m.gguf" });
        assert_eq!(
            require_str(&args, "model_path").ok(),
            Some("/m.gguf"),
            "a string argument must be read through"
        );
    }

    #[test]
    fn require_str_reports_absent_as_missing() {
        let args = serde_json::json!({});
        let err = require_str(&args, "model_path").expect_err("absent must fail");
        assert_eq!(err.is_error, Some(true));
        assert_eq!(err_text(&err), "Missing required argument: model_path");
    }

    #[test]
    fn require_str_reports_wrong_type_as_wrong_type_not_missing() {
        // #2419: a present-but-numeric model_path used to be reported as
        // "Missing required argument: model_path".
        let args = serde_json::json!({ "model_path": 123 });
        let err = require_str(&args, "model_path").expect_err("number must fail");
        assert_eq!(err.is_error, Some(true));
        assert!(
            !err_text(&err).contains("Missing"),
            "an argument that WAS sent must not be reported as missing; got: {}",
            err_text(&err)
        );
        assert_eq!(
            err_text(&err),
            "Invalid model_path: expected string, got number 123"
        );
    }

    #[test]
    fn opt_bool_defaults_false_when_absent_and_true_when_true() {
        assert_eq!(opt_bool(&serde_json::json!({}), "stats").ok(), Some(false));
        assert_eq!(
            opt_bool(&serde_json::json!({ "stats": true }), "stats").ok(),
            Some(true)
        );
    }

    #[test]
    fn opt_bool_rejects_a_string_instead_of_silently_defaulting_false() {
        // #2419: stats:"yes" produced a report with no statistics and no
        // diagnostic.
        let err = opt_bool(&serde_json::json!({ "stats": "yes" }), "stats")
            .expect_err("string must fail");
        assert_eq!(err.is_error, Some(true));
        assert_eq!(
            err_text(&err),
            "Invalid stats: expected boolean true or false, got string \"yes\""
        );
    }

    #[test]
    fn opt_u64_rejects_a_string_instead_of_silently_defaulting() {
        // #2419: max_tokens:"eight" silently became the default 32.
        let err = opt_u64(&serde_json::json!({ "max_tokens": "eight" }), "max_tokens")
            .expect_err("string must fail");
        assert_eq!(
            err_text(&err),
            "Invalid max_tokens: expected a non-negative integer, got string \"eight\""
        );
    }

    #[test]
    fn opt_u64_rejects_a_negative_number() {
        let err = opt_u64(&serde_json::json!({ "max_tokens": -1 }), "max_tokens")
            .expect_err("negative must fail");
        assert!(err_text(&err).contains("non-negative integer"));
    }

    #[test]
    fn opt_u64_accepts_an_integer() {
        assert_eq!(
            opt_u64(&serde_json::json!({ "max_tokens": 8 }), "max_tokens").ok(),
            Some(Some(8))
        );
    }

    #[test]
    fn opt_f64_rejects_a_string_and_accepts_a_number() {
        assert_eq!(
            opt_f64(&serde_json::json!({ "temperature": 0.5 }), "temperature").ok(),
            Some(Some(0.5))
        );
        let err = opt_f64(&serde_json::json!({ "temperature": "hot" }), "temperature")
            .expect_err("string must fail");
        assert!(err_text(&err).contains("expected a number"));
    }

    #[test]
    fn opt_str_rejects_a_non_string_and_passes_through_absent() {
        assert_eq!(opt_str(&serde_json::json!({}), "layer").ok(), Some(None));
        let err =
            opt_str(&serde_json::json!({ "layer": 7 }), "layer").expect_err("number must fail");
        assert!(err_text(&err).contains("expected string"));
    }

    #[test]
    fn reject_unsupported_passes_when_absent_and_fails_when_present() {
        assert!(reject_unsupported(&serde_json::json!({}), "reference", "nope").is_ok());
        let err = reject_unsupported(
            &serde_json::json!({ "reference": "/r.apr" }),
            "reference",
            "nope",
        )
        .expect_err("present must fail");
        assert_eq!(err.is_error, Some(true));
        assert!(err_text(&err).contains("Unsupported argument: reference"));
    }
}
