//! Typed extraction of `tools/call` arguments.
//!
//! Every tool used to read its optional arguments with a bare
//! `args.get("max_tokens").and_then(Value::as_u64)`, so the `None` branch —
//! "the caller sent something, but not of the declared type" — was
//! indistinguishable from "the caller sent nothing" and the flag was simply
//! omitted from the spawned argv. That is a wrong-answer channel: the client
//! believes it asserted something the server never applied, and the result
//! JSON echoes the CLI's *default* so the drop is undetectable downstream.
//! `apr.qa`'s `assert_tps` is the sharpest case — passed as a JSON string it
//! disarmed the throughput gate entirely (#2403).
//!
//! The rules here are deliberately narrow:
//!
//! - absent or JSON `null` → `Ok(None)` (the argument is optional)
//! - the declared JSON type → `Ok(Some(v))`
//! - a **string that parses exactly** into the declared type → `Ok(Some(v))`.
//!   LLM clients routinely emit numbers as JSON strings, so `"8"` for an
//!   integer is the common path, not an exotic one — coercing it is what the
//!   caller meant and it is lossless.
//! - anything else → `Err(message)`, which the tool turns into an
//!   `isError: true` result. This matches the one field that already
//!   validated before this module existed, `apr.serve`'s `port`.
//!
//! Nothing is ever silently dropped.

use crate::types::ToolCallResult;
use serde_json::Value;

/// `Ok(None)` = absent, `Ok(Some(v))` = present and usable, `Err` = present
/// but not convertible to the declared type.
pub type ArgResult<T> = Result<Option<T>, String>;

/// Early-return an `isError` [`crate::types::ToolCallResult`] when an
/// argument is present with an unusable type.
macro_rules! try_arg {
    ($expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(msg) => return crate::types::ToolCallResult::error(msg),
        }
    };
}
pub(crate) use try_arg;

fn type_error(name: &str, expected: &str, value: &Value) -> String {
    format!("Invalid {name}: expected {expected}, got {value}")
}

/// Look the argument up, treating JSON `null` as absent.
fn lookup<'a>(args: &'a Value, name: &str) -> Option<&'a Value> {
    args.get(name).filter(|v| !v.is_null())
}

/// Extract a non-negative integer argument (`"type": "integer"`).
///
/// Accepts a JSON integer, a JSON float with no fractional part (`8.0`), or a
/// decimal string (`"8"`). Rejects negatives, fractions and anything else.
///
/// # Errors
/// Returns the client-facing message when the value is present but not an
/// integer.
pub fn opt_u64(args: &Value, name: &str) -> ArgResult<u64> {
    let Some(value) = lookup(args, name) else {
        return Ok(None);
    };
    if let Some(n) = value.as_u64() {
        return Ok(Some(n));
    }
    if let Some(f) = value.as_f64() {
        if f.is_finite() && f >= 0.0 && f.fract() == 0.0 {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            return Ok(Some(f as u64));
        }
    }
    if let Some(s) = value.as_str() {
        if let Ok(n) = s.trim().parse::<u64>() {
            return Ok(Some(n));
        }
    }
    Err(type_error(name, "integer", value))
}

/// Extract a floating-point argument (`"type": "number"`).
///
/// Accepts a JSON number or a numeric string (`"100000"`, `"0.7"`). Rejects
/// non-finite results and anything unparseable.
///
/// # Errors
/// Returns the client-facing message when the value is present but not a
/// number.
pub fn opt_f64(args: &Value, name: &str) -> ArgResult<f64> {
    let Some(value) = lookup(args, name) else {
        return Ok(None);
    };
    if let Some(f) = value.as_f64() {
        return Ok(Some(f));
    }
    if let Some(s) = value.as_str() {
        if let Ok(f) = s.trim().parse::<f64>() {
            if f.is_finite() {
                return Ok(Some(f));
            }
        }
    }
    Err(type_error(name, "number", value))
}

/// Extract a boolean argument (`"type": "boolean"`).
///
/// Accepts a JSON boolean or the strings `"true"` / `"false"` in any case.
/// Rejects `0` / `1` and every other spelling.
///
/// # Errors
/// Returns the client-facing message when the value is present but not a
/// boolean.
pub fn opt_bool(args: &Value, name: &str) -> ArgResult<bool> {
    let Some(value) = lookup(args, name) else {
        return Ok(None);
    };
    if let Some(b) = value.as_bool() {
        return Ok(Some(b));
    }
    if let Some(s) = value.as_str() {
        let t = s.trim();
        if t.eq_ignore_ascii_case("true") {
            return Ok(Some(true));
        }
        if t.eq_ignore_ascii_case("false") {
            return Ok(Some(false));
        }
    }
    Err(type_error(name, "boolean", value))
}

/// Extract a string argument (`"type": "string"`).
///
/// Strict: a number is not a path or a name pattern, so `{"reference": 42}`
/// is an error rather than a silently stringified `"42"`.
///
/// # Errors
/// Returns the client-facing message when the value is present but not a
/// string.
pub fn opt_str<'a>(args: &'a Value, name: &str) -> ArgResult<&'a str> {
    let Some(value) = lookup(args, name) else {
        return Ok(None);
    };
    match value.as_str() {
        Some(s) => Ok(Some(s)),
        None => Err(type_error(name, "string", value)),
    }
}

/// Extract a required string argument, reporting absence and wrong type
/// distinctly.
///
/// # Errors
/// Returns the client-facing message when the value is missing or not a
/// string.
pub fn required_str<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
    match lookup(args, name) {
        None => Err(format!("Missing required argument: {name}")),
        Some(v) => match v.as_str() {
            Some(s) => Ok(s),
            None => Err(type_error(name, "string", v)),
        },
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn absent_and_null_are_none_not_errors() {
        let args = json!({ "max_tokens": null });
        assert_eq!(opt_u64(&args, "max_tokens"), Ok(None));
        assert_eq!(opt_u64(&args, "iterations"), Ok(None));
        assert_eq!(opt_f64(&args, "assert_tps"), Ok(None));
        assert_eq!(opt_bool(&args, "stats"), Ok(None));
        assert_eq!(opt_str(&args, "prompt"), Ok(None));
    }

    #[test]
    fn correctly_typed_values_pass_through() {
        let args = json!({
            "max_tokens": 8,
            "assert_tps": 100_000.0,
            "stats": true,
            "filter": "attn_qkv",
        });
        assert_eq!(opt_u64(&args, "max_tokens"), Ok(Some(8)));
        assert_eq!(opt_f64(&args, "assert_tps"), Ok(Some(100_000.0)));
        assert_eq!(opt_bool(&args, "stats"), Ok(Some(true)));
        assert_eq!(opt_str(&args, "filter"), Ok(Some("attn_qkv")));
    }

    /// The #2403 repro: an LLM client emitting numbers as JSON strings must
    /// still reach the CLI, never be dropped.
    #[test]
    fn numeric_strings_are_coerced_not_dropped() {
        let args = json!({ "max_tokens": "8", "assert_tps": "100000", "stats": "true" });
        assert_eq!(opt_u64(&args, "max_tokens"), Ok(Some(8)));
        assert_eq!(opt_f64(&args, "assert_tps"), Ok(Some(100_000.0)));
        assert_eq!(opt_bool(&args, "stats"), Ok(Some(true)));
    }

    #[test]
    fn integral_float_is_accepted_for_integer() {
        assert_eq!(opt_u64(&json!({ "n": 8.0 }), "n"), Ok(Some(8)));
    }

    #[test]
    fn unusable_values_are_errors_not_silent_drops() {
        assert!(opt_u64(&json!({ "n": "eight" }), "n").is_err());
        assert!(opt_u64(&json!({ "n": -1 }), "n").is_err());
        assert!(opt_u64(&json!({ "n": 1.5 }), "n").is_err());
        assert!(opt_u64(&json!({ "n": true }), "n").is_err());
        assert!(opt_f64(&json!({ "n": "fast" }), "n").is_err());
        assert!(opt_bool(&json!({ "n": 1 }), "n").is_err());
        assert!(opt_bool(&json!({ "n": "yes" }), "n").is_err());
        assert!(opt_str(&json!({ "n": 42 }), "n").is_err());
    }

    #[test]
    fn error_message_names_the_field_the_type_and_the_value() {
        let err = opt_u64(&json!({ "max_tokens": "eight" }), "max_tokens")
            .expect_err("string 'eight' is not an integer");
        assert!(err.contains("max_tokens"), "{err}");
        assert!(err.contains("integer"), "{err}");
        assert!(err.contains("eight"), "{err}");
    }

    #[test]
    fn required_str_distinguishes_missing_from_wrong_type() {
        let missing = required_str(&json!({}), "model_path").expect_err("absent");
        assert!(missing.contains("Missing required argument"));
        assert!(missing.contains("model_path"));

        let wrong = required_str(&json!({ "model_path": 7 }), "model_path").expect_err("not a str");
        assert!(wrong.contains("model_path"));
        assert!(wrong.contains("string"));
    }
}

// ---------------------------------------------------------------------------
// Carried over from the transport-conformance work (#2434).
//
// `json_type_name` is called directly by server.rs, and `require_str` is the
// shape the already-merged tool wrappers use. `required_str` above draws the
// same absent-vs-wrong-type distinction and returns a plain String; this one
// returns a ready-to-send ToolCallResult, which is what a `call` entry point
// wants.
// ---------------------------------------------------------------------------

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
/// Tests for the carried-over `json_type_name` / `require_str` helpers.
mod require_str_tests {
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
