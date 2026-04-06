//! Chat template formatting for local LLM inference.
//!
//! Different model families require different prompt formats.
//! Auto-detects the template from the model filename:
//! - Qwen (2.5, 3.x), DeepSeek, Yi → ChatML
//! - Llama → Llama 3.x
//! - Unknown → ChatML (most widely supported)
//!
//! Qwen3 uses ChatML with native `<tool_call>` support. Thinking mode
//! (`<think>...</think>`) is controlled by generation params, not template.
//! PMAT-179: Default model is Qwen3 1.7B (0.960 tool-calling score).
//!
//! See: apr-code.md §5.1

use super::{CompletionRequest, Message, ToolDefinition};

/// Chat template family, auto-detected from model filename.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChatTemplate {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>` (Qwen, Yi, Deepseek)
    ChatMl,
    /// Llama 3.x: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`
    Llama3,
    /// Generic: `<|system|>\ncontent\n<|end|>` (fallback)
    Generic,
}

impl ChatTemplate {
    /// Detect template from model filename.
    pub fn from_model_path(path: &std::path::Path) -> Self {
        let name = path.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();

        if name.contains("qwen") || name.contains("deepseek") || name.contains("yi-") {
            Self::ChatMl
        } else if name.contains("llama") {
            Self::Llama3
        } else {
            Self::ChatMl
        }
    }
}

/// Format messages using a specific chat template.
///
/// For local models, tool definitions from `request.tools` are injected
/// into the system prompt so the model knows what tools exist and how
/// to invoke them via `<tool_call>` blocks. API-based drivers handle
/// tools natively; local models need this explicit injection.
pub fn format_prompt_with_template(request: &CompletionRequest, template: ChatTemplate) -> String {
    // Build enriched system prompt with tool definitions
    let enriched_system = build_enriched_system(&request.system, &request.tools);
    let enriched_request = CompletionRequest {
        system: Some(enriched_system),
        model: request.model.clone(),
        messages: request.messages.clone(),
        tools: request.tools.clone(),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
    };

    match template {
        ChatTemplate::ChatMl => format_chatml(&enriched_request),
        ChatTemplate::Llama3 => format_llama3(&enriched_request),
        ChatTemplate::Generic => format_generic(&enriched_request),
    }
}

/// Build an enriched system prompt with tool definitions appended.
///
/// Local models need explicit tool definitions in text form — unlike
/// API models (Anthropic/OpenAI) which accept tools as structured params.
/// The format teaches the model to emit `<tool_call>` blocks that
/// `parse_tool_calls()` in realizar.rs can extract.
fn build_enriched_system(base_system: &Option<String>, tools: &[ToolDefinition]) -> String {
    let mut system = base_system.clone().unwrap_or_default();

    if tools.is_empty() {
        return system;
    }

    // Append tool definitions
    system.push_str("\n\n## Available Tools\n\n");
    system.push_str(
        "To use a tool, output a <tool_call> block with JSON inside. \
         You will receive the result in a <tool_result> block.\n\n",
    );
    system.push_str("Format:\n```\n<tool_call>\n{\"name\": \"tool_name\", \"input\": {\"param\": \"value\"}}\n</tool_call>\n```\n\n");

    for tool in tools {
        system.push_str(&format!("### {}\n{}\n", tool.name, tool.description));
        // Compact JSON schema — only include properties, not full schema boilerplate
        if let Some(props) = tool.input_schema.get("properties") {
            system.push_str(&format!("Parameters: {}\n\n", compact_schema(props)));
        } else {
            system.push('\n');
        }
    }

    system.push_str(
        "After receiving a <tool_result>, analyze it and either use another tool or respond to the user.\n",
    );

    system
}

/// Compact a JSON schema properties object into a readable summary.
fn compact_schema(props: &serde_json::Value) -> String {
    if let Some(obj) = props.as_object() {
        let params: Vec<String> = obj
            .iter()
            .map(|(k, v)| {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("string");
                let desc = v.get("description").and_then(|d| d.as_str()).unwrap_or("");
                if desc.is_empty() {
                    format!("{k}: {typ}")
                } else {
                    format!("{k} ({typ}): {desc}")
                }
            })
            .collect();
        format!("{{{}}}", params.join(", "))
    } else {
        props.to_string()
    }
}

/// ChatML format (Qwen, DeepSeek, Yi).
fn format_chatml(request: &CompletionRequest) -> String {
    let mut prompt = String::new();

    if let Some(ref system) = request.system {
        prompt.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    }

    for msg in &request.messages {
        match msg {
            Message::System(s) => {
                prompt.push_str(&format!("<|im_start|>system\n{s}<|im_end|>\n"));
            }
            Message::User(s) => {
                prompt.push_str(&format!("<|im_start|>user\n{s}<|im_end|>\n"));
            }
            Message::Assistant(s) => {
                prompt.push_str(&format!("<|im_start|>assistant\n{s}<|im_end|>\n"));
            }
            Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|im_start|>assistant\n<tool_call>\n{}\n</tool_call><|im_end|>\n",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            Message::ToolResult(result) => {
                prompt.push_str(&format!(
                    "<|im_start|>user\n<tool_result>{}</tool_result><|im_end|>\n",
                    result.content
                ));
            }
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Llama 3.x format.
fn format_llama3(request: &CompletionRequest) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|begin_of_text|>");

    if let Some(ref system) = request.system {
        prompt
            .push_str(&format!("<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"));
    }

    for msg in &request.messages {
        match msg {
            Message::System(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            Message::User(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            Message::Assistant(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n<tool_call>\n{}\n</tool_call><|eot_id|>",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            Message::ToolResult(result) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n<tool_result>{}</tool_result><|eot_id|>",
                    result.content
                ));
            }
        }
    }

    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Generic fallback format.
fn format_generic(request: &CompletionRequest) -> String {
    let mut prompt = String::new();

    if let Some(ref system) = request.system {
        prompt.push_str(&format!("<|system|>\n{system}\n<|end|>\n"));
    }

    for msg in &request.messages {
        match msg {
            Message::System(s) => {
                prompt.push_str(&format!("<|system|>\n{s}\n<|end|>\n"));
            }
            Message::User(s) => {
                prompt.push_str(&format!("<|user|>\n{s}\n<|end|>\n"));
            }
            Message::Assistant(s) => {
                prompt.push_str(&format!("<|assistant|>\n{s}\n<|end|>\n"));
            }
            Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|assistant|>\n<tool_call>\n{}\n</tool_call>\n<|end|>\n",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            Message::ToolResult(result) => {
                prompt.push_str(&format!(
                    "<|user|>\n<tool_result>{}</tool_result>\n<|end|>\n",
                    result.content
                ));
            }
        }
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::driver::ToolCall;

    fn sample_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "file_read".into(),
                description: "Read file contents".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    }
                }),
            },
            ToolDefinition {
                name: "shell".into(),
                description: "Execute shell command".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"}
                    }
                }),
            },
        ]
    }

    #[test]
    fn test_tool_definitions_injected_into_system() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![Message::User("Hello".into())],
            tools: sample_tools(),
            max_tokens: 100,
            temperature: 0.5,
            system: Some("You are helpful".into()),
        };
        let prompt = format_prompt_with_template(&request, ChatTemplate::ChatMl);
        assert!(prompt.contains("file_read"), "tool name missing");
        assert!(prompt.contains("Read file contents"), "tool description missing");
        assert!(prompt.contains("shell"), "second tool missing");
        assert!(prompt.contains("<tool_call>"), "tool call format missing");
        assert!(prompt.contains("tool_result"), "tool result format missing");
        assert!(prompt.contains("path (string): File path to read"), "schema missing");
    }

    #[test]
    fn test_no_tools_no_injection() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![Message::User("Hello".into())],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: Some("You are helpful".into()),
        };
        let prompt = format_prompt_with_template(&request, ChatTemplate::ChatMl);
        assert!(prompt.contains("You are helpful"));
        assert!(!prompt.contains("Available Tools"), "no tools = no injection");
    }

    #[test]
    fn test_compact_schema() {
        let props = serde_json::json!({
            "path": {"type": "string", "description": "File to read"},
            "limit": {"type": "integer"}
        });
        let result = compact_schema(&props);
        assert!(result.contains("path (string): File to read"));
        assert!(result.contains("limit: integer"));
    }

    #[test]
    fn test_format_prompt_chatml() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![Message::User("Hello".into())],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: Some("You are helpful".into()),
        };
        let prompt = format_chatml(&request);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are helpful"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_prompt_llama3() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![Message::User("Hello".into())],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: Some("Be helpful".into()),
        };
        let prompt = format_llama3(&request);
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("Be helpful"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_format_prompt_generic_fallback() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![Message::User("Hello".into())],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: Some("You are helpful".into()),
        };
        let prompt = format_generic(&request);
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_format_prompt_tool_messages() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![
                Message::AssistantToolUse(ToolCall {
                    id: "1".into(),
                    name: "rag".into(),
                    input: serde_json::json!({"query": "test"}),
                }),
                Message::ToolResult(crate::agent::driver::ToolResultMsg {
                    tool_use_id: "1".into(),
                    content: "result data".into(),
                    is_error: false,
                }),
            ],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: None,
        };
        for template in [ChatTemplate::ChatMl, ChatTemplate::Llama3, ChatTemplate::Generic] {
            let prompt = format_prompt_with_template(&request, template);
            assert!(prompt.contains("<tool_call>"), "missing tool_call in {template:?}");
            assert!(prompt.contains("<tool_result>"), "missing tool_result in {template:?}");
            assert!(prompt.contains("result data"), "missing result data in {template:?}");
        }
    }

    #[test]
    fn test_chat_template_detection() {
        use std::path::Path;
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("qwen2.5-coder-7b.gguf")),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("Qwen3-8B-Q4K.apr")),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("deepseek-coder-v2.gguf")),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("llama-3.2-3b.gguf")),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("Meta-Llama-3-8B.apr")),
            ChatTemplate::Llama3
        );
        assert_eq!(ChatTemplate::from_model_path(Path::new("yi-34b.gguf")), ChatTemplate::ChatMl);
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("custom-model.gguf")),
            ChatTemplate::ChatMl
        );
    }
}

#[cfg(test)]
#[path = "chat_template_contract_tests.rs"]
mod contract_tests;
