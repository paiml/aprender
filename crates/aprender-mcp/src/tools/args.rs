//! Shared `tools/call` argument extraction.
//!
//! Every subprocess wrapper needs the same thing: pull a required string out
//! of the `arguments` object or fail. Doing that with
//! `args.get(name).and_then(|v| v.as_str())` collapses two distinct client
//! mistakes into one message — an argument that was never sent and an
//! argument sent with the wrong JSON type both read as
//! "Missing required argument: model_path". A client (or an LLM) told the
//! argument is missing retries by adding a key it already sent, and loops.
//!
//! [`require_str`] keeps the two apart: absent says missing, present-but-not-a-
//! string names the type it actually received and the type the declared
//! `inputSchema` asked for.

use crate::types::ToolCallResult;

/// JSON type name as it appears in a JSON Schema `type` keyword.
///
/// Used in argument-validation messages so the text a client sees lines up
/// with the vocabulary of the `inputSchema` it was given by `tools/list`.
#[must_use]
pub fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Extract a required string argument from a `tools/call` `arguments` object.
///
/// # Errors
/// Returns a ready-to-send `isError: true` [`ToolCallResult`] when the
/// argument is absent (`Missing required argument: <name>`) or present with a
/// non-string JSON type (`Argument <name> must be a string, got <type>`). The
/// two messages are deliberately distinguishable — see the module header.
pub fn require_str<'a>(args: &'a serde_json::Value, name: &str) -> Result<&'a str, ToolCallResult> {
    match args.get(name) {
        Some(serde_json::Value::String(s)) => Ok(s.as_str()),
        Some(other) => Err(ToolCallResult::error(format!(
            "Argument {name} must be a string, got {}",
            json_type_name(other)
        ))),
        None => Err(ToolCallResult::error(format!(
            "Missing required argument: {name}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn present_string_is_returned() {
        let args = serde_json::json!({ "model_path": "/tmp/m.gguf" });
        assert_eq!(require_str(&args, "model_path").ok(), Some("/tmp/m.gguf"));
    }

    #[test]
    fn absent_argument_says_missing() {
        let args = serde_json::json!({});
        let err = require_str(&args, "model_path").expect_err("absent must fail");
        assert_eq!(err.is_error, Some(true));
        assert_eq!(err.content[0].text, "Missing required argument: model_path");
    }

    /// The defect this module exists for: a wrong-TYPE argument must not be
    /// reported as missing, because the client can see it sent the key.
    #[test]
    fn wrong_type_names_the_type_and_never_says_missing() {
        for (value, expected_type) in [
            (serde_json::json!(123), "number"),
            (serde_json::json!(true), "boolean"),
            (serde_json::json!(["/tmp/m.gguf"]), "array"),
            (serde_json::json!({ "path": "/tmp/m.gguf" }), "object"),
            (serde_json::json!(null), "null"),
        ] {
            let args = serde_json::json!({ "model_path": value });
            let err = require_str(&args, "model_path").expect_err("wrong type must fail");
            let text = &err.content[0].text;
            assert_eq!(
                text,
                &format!("Argument model_path must be a string, got {expected_type}"),
                "wrong-type message for {value}"
            );
            assert!(
                !text.contains("Missing"),
                "a supplied argument must never be reported as missing, got: {text}"
            );
        }
    }

    #[test]
    fn non_object_arguments_read_as_missing() {
        let args = serde_json::json!("notanobject");
        let err = require_str(&args, "model_path").expect_err("non-object must fail");
        assert_eq!(err.content[0].text, "Missing required argument: model_path");
    }

    #[test]
    fn json_type_name_covers_every_variant() {
        assert_eq!(json_type_name(&serde_json::json!(null)), "null");
        assert_eq!(json_type_name(&serde_json::json!(false)), "boolean");
        assert_eq!(json_type_name(&serde_json::json!(1.5)), "number");
        assert_eq!(json_type_name(&serde_json::json!("s")), "string");
        assert_eq!(json_type_name(&serde_json::json!([])), "array");
        assert_eq!(json_type_name(&serde_json::json!({})), "object");
    }
}
