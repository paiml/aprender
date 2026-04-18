//! apr.finetune — LoRA/full-finetune subprocess wrapper.
//!
//! M3 initial slice: **synchronous** — waits for finetuning to complete,
//! returns final JSON payload. Progress notifications via
//! `notifications/progress` per training step are deferred to a follow-up
//! M3 slice (same pattern as apr.run streaming upgrade).
//!
//! Wraps `apr finetune <base_model> --json [--data <path>] [--rank <N>]
//! [--epochs <N>] [--method <m>] [--output <path>]`.
//!
//! Note on argument names: the spec (`docs/specifications/apr-mcp-server-spec.md`
//! line 85) lists `base_model`, `dataset`, `lora_rank`, `epochs`. The actual
//! `apr finetune` CLI uses a positional `<FILE>` for the base model, `--data`
//! (not `--dataset`), and `--rank` (not `--lora-rank`). We keep the spec's
//! ergonomic MCP argument names (`base_model`, `dataset`, `lora_rank`) as the
//! schema surface but map them to the real CLI flags at dispatch time so LLM
//! callers aren't exposed to the flag-name mismatch.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, PropertySchema, ToolCallResult, ToolDefinition};
use std::collections::HashMap;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.finetune";

/// Return the MCP tool definition for `apr.finetune`.
#[must_use]
pub fn finetune_tool_definition() -> ToolDefinition {
    let mut properties = HashMap::new();
    properties.insert(
        "base_model".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description:
                "Path to the base model file (.apr, .gguf, or .safetensors) or hf://org/repo"
                    .to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "dataset".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Path to training data (JSONL). Mapped to `apr finetune --data <path>`."
                .to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "lora_rank".to_string(),
        PropertySchema {
            prop_type: "integer".to_string(),
            description: "LoRA rank. Mapped to `apr finetune --rank <N>`. Omit for auto."
                .to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "epochs".to_string(),
        PropertySchema {
            prop_type: "integer".to_string(),
            description: "Training epochs (default 3)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "method".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Fine-tuning method: auto, full, lora, qlora (default auto)".to_string(),
            r#enum: None,
        },
    );
    properties.insert(
        "output".to_string(),
        PropertySchema {
            prop_type: "string".to_string(),
            description: "Output path (adapter directory or merged model)".to_string(),
            r#enum: None,
        },
    );
    ToolDefinition {
        name: NAME.to_string(),
        description:
            "Fine-tune a base model with LoRA/QLoRA. Wraps `apr finetune <base_model> --json` and blocks until training completes. Progress streaming lands in a follow-up M3 slice."
                .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["base_model".to_string()],
        },
    }
}

/// Execute `apr.finetune` by spawning `apr finetune <base_model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let Some(base_model) = args.get("base_model").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: base_model");
    };

    let mut owned: Vec<String> = vec![
        "finetune".to_string(),
        base_model.to_string(),
        "--json".to_string(),
    ];

    if let Some(dataset) = args.get("dataset").and_then(|v| v.as_str()) {
        if !dataset.is_empty() {
            owned.push("--data".to_string());
            owned.push(dataset.to_string());
        }
    }
    if let Some(rank) = args.get("lora_rank").and_then(serde_json::Value::as_u64) {
        owned.push("--rank".to_string());
        owned.push(rank.to_string());
    }
    if let Some(epochs) = args.get("epochs").and_then(serde_json::Value::as_u64) {
        owned.push("--epochs".to_string());
        owned.push(epochs.to_string());
    }
    if let Some(method) = args.get("method").and_then(|v| v.as_str()) {
        if !method.is_empty() {
            owned.push("--method".to_string());
            owned.push(method.to_string());
        }
    }
    if let Some(output) = args.get("output").and_then(|v| v.as_str()) {
        if !output.is_empty() {
            owned.push("--output".to_string());
            owned.push(output.to_string());
        }
    }

    let argv: Vec<&str> = owned.iter().map(String::as_str).collect();
    run_apr(&argv)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn finetune_tool_definition_shape() {
        let def = finetune_tool_definition();
        assert_eq!(def.name, "apr.finetune");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["base_model".to_string()]);
        for field in [
            "base_model",
            "dataset",
            "lora_rank",
            "epochs",
            "method",
            "output",
        ] {
            assert!(
                def.input_schema.properties.contains_key(field),
                "property {field} present"
            );
        }
    }

    #[test]
    fn finetune_missing_base_model_is_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(
            result.content[0].text.contains("base_model"),
            "error message must mention base_model, got: {}",
            result.content[0].text
        );
    }

    #[test]
    fn finetune_nonstring_base_model_is_error() {
        let result = call(&serde_json::json!({ "base_model": 42 }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("base_model"));
    }
}
