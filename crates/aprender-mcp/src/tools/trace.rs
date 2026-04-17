//! `apr.trace` — M2 tool. Layer-by-layer tensor trace for debugging a model.
//!
//! Wraps `apr trace <model> --json [--layer <pat>] [--reference <path>]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, PropertySchema, ToolCallResult, ToolDefinition};
use std::collections::HashMap;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.trace";

/// Return the MCP tool definition for `apr.trace`.
#[must_use]
pub fn trace_tool_definition() -> ToolDefinition {
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
        "layer".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Filter layers by name pattern (substring match)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "reference".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Reference model to diff against (path)".to_string(),
            r#enum: None,
        },
    );
    ToolDefinition {
        name: NAME.to_string(),
        description:
            "Layer-by-layer tensor trace with per-layer stats. Wraps `apr trace <model> --json`."
                .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["model_path".to_string()],
        },
    }
}

/// Execute `apr.trace` by spawning `apr trace <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: model_path");
    };

    let mut owned: Vec<String> = vec![
        "trace".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    if let Some(pat) = args.get("layer").and_then(|v| v.as_str()) {
        if !pat.is_empty() {
            owned.push("--layer".to_string());
            owned.push(pat.to_string());
        }
    }
    if let Some(ref_path) = args.get("reference").and_then(|v| v.as_str()) {
        if !ref_path.is_empty() {
            owned.push("--reference".to_string());
            owned.push(ref_path.to_string());
        }
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
        let def = trace_tool_definition();
        assert_eq!(def.name, "apr.trace");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "layer", "reference"] {
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
