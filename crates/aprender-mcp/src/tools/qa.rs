//! `apr.qa` — M2 tool. The 8-gate quality checklist; first stop for any model issue.
//!
//! Wraps `apr qa <model> --json [--assert-tps N] [--max-tokens N] [--iterations N]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, PropertySchema, ToolCallResult, ToolDefinition};
use std::collections::HashMap;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.qa";

/// Return the MCP tool definition for `apr.qa`.
#[must_use]
pub fn qa_tool_definition() -> ToolDefinition {
    let mut properties = HashMap::new();
    properties.insert(
        "model_path".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Path to the model file (.apr, .gguf, or .safetensors)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "assert_tps".to_string(),
        PropertySchema {
            prop_type: "number".to_string(),
            description: "Minimum throughput threshold in tok/s — gate fails below this"
                .to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "max_tokens".to_string(),
        PropertySchema {
            prop_type: "integer".to_string(),
            description: "Maximum tokens to generate per iteration (default 32)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "iterations".to_string(),
        PropertySchema {
            prop_type: "integer".to_string(),
            description: "Benchmark iterations (default 10)".to_string(),
            r#enum: None,
        },
    );
    ToolDefinition {
        name: NAME.to_string(),
        description:
            "Run the 8-gate falsifiable QA checklist on a model. Wraps `apr qa <model> --json`."
                .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["model_path".to_string()],
        },
    }
}

/// Execute `apr.qa` by spawning `apr qa <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: model_path");
    };

    let mut owned: Vec<String> = vec![
        "qa".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    if let Some(tps) = args.get("assert_tps").and_then(serde_json::Value::as_f64) {
        owned.push("--assert-tps".to_string());
        owned.push(tps.to_string());
    }
    if let Some(n) = args.get("max_tokens").and_then(serde_json::Value::as_u64) {
        owned.push("--max-tokens".to_string());
        owned.push(n.to_string());
    }
    if let Some(n) = args.get("iterations").and_then(serde_json::Value::as_u64) {
        owned.push("--iterations".to_string());
        owned.push(n.to_string());
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
        let def = qa_tool_definition();
        assert_eq!(def.name, "apr.qa");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "assert_tps", "max_tokens", "iterations"] {
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
