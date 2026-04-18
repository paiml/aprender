//! `apr.run` — M2 tool. Synchronous inference via subprocess wrapper.
//!
//! Wraps `apr run <model> --json [--prompt X] [--max-tokens N] [--temperature T] [--top-p P]`.
//!
//! M3 will extend this wrapper to stream `notifications/progress` per decoded
//! token (spec `docs/specifications/apr-mcp-server-spec.md` line 156). Until
//! then, the call blocks until decode completes and returns a single content
//! block of the subprocess stdout.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, PropertySchema, ToolCallResult, ToolDefinition};
use std::collections::HashMap;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.run";

/// Return the MCP tool definition for `apr.run`.
#[must_use]
pub fn run_tool_definition() -> ToolDefinition {
    let mut properties = HashMap::new();
    properties.insert(
        "model_path".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Path to the model file (.apr, .gguf, or .safetensors) or hf://org/repo"
                .to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "prompt".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Text prompt to generate from".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "max_tokens".to_string(),
        PropertySchema {
            prop_type: "integer".to_string(),
            description: "Maximum tokens to generate (default 32)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "temperature".to_string(),
        PropertySchema {
            prop_type: "number".to_string(),
            description: "Sampling temperature (0.0 = greedy argmax, >0 = stochastic)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "top_p".to_string(),
        PropertySchema {
            prop_type: "number".to_string(),
            description: "Top-p nucleus sampling threshold (omit to disable)".to_string(),
            r#enum: None,
        },
    );
    ToolDefinition {
        name: NAME.to_string(),
        description:
            "Run synchronous inference on a model. Wraps `apr run <model> --json` and returns tokens + tok/s + stop reason."
                .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["model_path".to_string()],
        },
    }
}

/// Execute `apr.run` by spawning `apr run <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: model_path");
    };

    let mut owned: Vec<String> = vec![
        "run".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    if let Some(prompt) = args.get("prompt").and_then(|v| v.as_str()) {
        if !prompt.is_empty() {
            owned.push("--prompt".to_string());
            owned.push(prompt.to_string());
        }
    }
    if let Some(n) = args.get("max_tokens").and_then(serde_json::Value::as_u64) {
        owned.push("--max-tokens".to_string());
        owned.push(n.to_string());
    }
    if let Some(t) = args.get("temperature").and_then(serde_json::Value::as_f64) {
        owned.push("--temperature".to_string());
        owned.push(t.to_string());
    }
    if let Some(p) = args.get("top_p").and_then(serde_json::Value::as_f64) {
        owned.push("--top-p".to_string());
        owned.push(p.to_string());
    }

    let argv: Vec<&str> = owned.iter().map(String::as_str).collect();
    run_apr(&argv)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = run_tool_definition();
        assert_eq!(def.name, "apr.run");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "prompt", "max_tokens", "temperature", "top_p"] {
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
