//! `apr.tensors` — M2 tool. List tensor names, shapes, and (optionally) stats.
//!
//! Wraps `apr tensors <model> --json [--stats] [--filter <pat>] [--limit <n>]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, PropertySchema, ToolCallResult, ToolDefinition};
use std::collections::HashMap;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.tensors";

/// Return the MCP tool definition for `apr.tensors`.
#[must_use]
pub fn tensors_tool_definition() -> ToolDefinition {
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
        "stats".to_string(),
        PropertySchema {
            prop_type: "boolean".to_string(),
            description: "Include tensor statistics (mean, std, min, max)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "filter".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Filter tensors by name pattern (substring match)".to_string(),
            r#enum: None,
        },
    );
    ToolDefinition {
        name: NAME.to_string(),
        description:
            "List tensors in a model with shapes and dtypes. Wraps `apr tensors <model> --json`."
                .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["model_path".to_string()],
        },
    }
}

/// Execute `apr.tensors` by spawning `apr tensors <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: model_path");
    };

    let mut argv: Vec<&str> = vec!["tensors", model_path, "--json"];
    if args
        .get("stats")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        argv.push("--stats");
    }
    let filter = args.get("filter").and_then(|v| v.as_str()).unwrap_or("");
    if !filter.is_empty() {
        argv.push("--filter");
        argv.push(filter);
    }

    run_apr(&argv)
}

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
}
