//! `apr.bench` — M2 tool. Benchmark inference throughput (tok/s, latency percentiles).
//!
//! Wraps `apr bench <model> --json [--iterations N] [--max-tokens N] [--prompt X]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, PropertySchema, ToolCallResult, ToolDefinition};
use std::collections::HashMap;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.bench";

/// Return the MCP tool definition for `apr.bench`.
#[must_use]
pub fn bench_tool_definition() -> ToolDefinition {
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
        "iterations".to_string(),
        PropertySchema {
            prop_type: "integer".to_string(),
            description: "Measurement iterations (default 5)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "max_tokens".to_string(),
        PropertySchema {
            prop_type: "integer".to_string(),
            description: "Max tokens to generate per iteration (default 32)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "prompt".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Test prompt (default: model-specific)".to_string(),
            r#enum: None,
        },
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: "Benchmark model throughput and latency. Wraps `apr bench <model> --json`."
            .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["model_path".to_string()],
        },
    }
}

/// Execute `apr.bench` by spawning `apr bench <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: model_path");
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
