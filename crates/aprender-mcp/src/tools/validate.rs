//! `apr.validate` — M2 subprocess wrapper over `apr validate <model> --json`.
//!
//! This is the first M2 tool and exercises the subprocess pattern that the
//! remaining 7 Phase-1 tools will follow: spawn `apr <subcommand> --json`,
//! capture stdout, pass through to the MCP client as a single text content
//! block. Non-zero exit maps to `isError: true` with stderr attached.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, PropertySchema, ToolCallResult, ToolDefinition};
use std::collections::HashMap;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.validate";

/// Return the MCP tool definition for `apr.validate`.
#[must_use]
pub fn validate_tool_definition() -> ToolDefinition {
    let mut properties = HashMap::new();
    properties.insert(
        "model_path".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Path to the model file (.apr, .gguf, or .safetensors)".to_string(),
            r#enum: None,
        },
    );
    ToolDefinition {
        name: NAME.to_string(),
        description:
            "Validate a model file's integrity and quality. Wraps `apr validate <model> --json`."
                .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["model_path".to_string()],
        },
    }
}

/// Execute `apr.validate` by spawning `apr validate <model_path> --json`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: model_path");
    };
    run_apr(&["validate", model_path, "--json"])
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = validate_tool_definition();
        assert_eq!(def.name, "apr.validate");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        assert!(def.input_schema.properties.contains_key("model_path"));
    }

    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }

    #[test]
    fn nonstring_model_path_returns_error() {
        let result = call(&serde_json::json!({ "model_path": 42 }));
        assert_eq!(result.is_error, Some(true));
    }
}
