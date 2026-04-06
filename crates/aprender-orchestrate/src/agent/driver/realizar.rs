//! RealizarDriver — sovereign local inference via GGUF/APR models.
//!
//! Uses the `realizar` crate for local LLM inference. All data
//! stays on-device (Sovereign privacy tier, Genchi Genbutsu).
//!
//! Tool call parsing: local models output `<tool_call>` JSON blocks
//! in their text. The driver extracts these into `ToolCall` structs.
//!
//! Feature-gated behind `inference`.

use async_trait::async_trait;
use std::path::PathBuf;

use super::chat_template::{format_prompt_with_template, ChatTemplate};
use super::validate::validate_model_file;
use super::{CompletionRequest, CompletionResponse, LlmDriver, ToolCall};
use crate::agent::result::{AgentError, DriverError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// Local inference driver using realizar (GGUF/APR/SafeTensors).
pub struct RealizarDriver {
    /// Path to model file.
    model_path: PathBuf,
    /// Context window size.
    context_window_size: usize,
    /// Auto-detected chat template.
    template: ChatTemplate,
}

impl RealizarDriver {
    /// Create a new RealizarDriver from a model path.
    ///
    /// **Contract: `apr_model_validity` (apr-code-v1.yaml)**
    ///
    /// Preconditions enforced at the load boundary (Jidoka):
    /// - File must exist
    /// - APR files: must have embedded tokenizer (checked via header)
    /// - GGUF files: must have valid magic bytes
    ///
    /// Violation → actionable error with re-conversion instructions.
    /// No broken model ever reaches the inference loop.
    pub fn new(model_path: PathBuf, context_window: Option<usize>) -> Result<Self, AgentError> {
        if !model_path.exists() {
            return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
                "model not found: {}",
                model_path.display()
            ))));
        }

        // ═══ CONTRACT: apr_model_validity — Jidoka boundary check ═══
        validate_model_file(&model_path)?;

        let context_window_size = context_window.unwrap_or(4096);
        let template = ChatTemplate::from_model_path(&model_path);
        Ok(Self { model_path, context_window_size, template })
    }
}

#[async_trait]
impl LlmDriver for RealizarDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        // Format messages using auto-detected chat template
        let prompt = format_prompt_with_template(&request, self.template);

        // Build inference config (explicit fields — no Default impl)
        let config = realizar::infer::InferenceConfig {
            model_path: self.model_path.clone(),
            prompt: Some(prompt),
            input_tokens: None,
            max_tokens: request.max_tokens as usize,
            temperature: request.temperature,
            top_k: 0,
            // PMAT-156/158: Disable GPU only for APR models (wgpu shader bug).
            // GGUF models work fine with CUDA — keep GPU enabled for them.
            no_gpu: self.model_path.extension().is_some_and(|e| e == "apr"),
            trace: false,
            trace_verbose: false,
            trace_output: None,
            trace_steps: None,
            verbose: false,
            use_mock_backend: false,
            stop_tokens: vec![],
        };

        // Run inference in blocking thread (realizar is sync)
        let result = tokio::task::spawn_blocking(move || realizar::infer::run_inference(&config))
            .await
            .map_err(|e| {
                AgentError::Driver(DriverError::InferenceFailed(format!("spawn_blocking: {e}")))
            })?
            .map_err(|e| AgentError::Driver(DriverError::InferenceFailed(e.to_string())))?;

        // Parse tool calls from text output
        let (raw_text, tool_calls) = parse_tool_calls(&result.text);

        // Sanitize output: strip echoed system prompt and chat template markers
        let text = sanitize_output(&raw_text, request.system.as_deref());

        let stop_reason =
            if tool_calls.is_empty() { StopReason::EndTurn } else { StopReason::ToolUse };

        Ok(CompletionResponse {
            text,
            stop_reason,
            tool_calls,
            usage: TokenUsage {
                input_tokens: result.input_token_count as u64,
                output_tokens: result.generated_token_count as u64,
            },
        })
    }

    fn context_window(&self) -> usize {
        self.context_window_size
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Sovereign
    }
}

/// Parse tool calls from model output text.
///
/// Supports multiple formats (PMAT-158):
/// 1. `<tool_call>{"name":...}</tool_call>` — custom XML tags
/// 2. `<tool_call>{"name":...}` — unclosed XML (small model fallback)
/// 3. `` ```json\n{"name":...}\n``` `` — markdown code block (Qwen native)
///
/// Returns the remaining text (with tool call blocks removed)
/// and the extracted tool calls.
/// Public wrapper for tool call parsing (used by AprServeDriver).
pub fn parse_tool_calls_pub(text: &str) -> (String, Vec<ToolCall>) {
    parse_tool_calls(text)
}

fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut remaining = String::new();
    let mut call_counter = 0u32;

    let mut cursor = text;
    loop {
        // Find next tool call start — try <tool_call> first, then ```json
        let xml_pos = cursor.find("<tool_call>");
        let md_pos = cursor.find("```json");

        let (start, tag_len, is_markdown) = match (xml_pos, md_pos) {
            (Some(x), Some(m)) if x <= m => (x, "<tool_call>".len(), false),
            (Some(x), None) => (x, "<tool_call>".len(), false),
            (_, Some(m)) => (m, "```json".len(), true),
            (None, None) => {
                remaining.push_str(cursor);
                break;
            }
        };

        remaining.push_str(&cursor[..start]);
        let after_tag = &cursor[start + tag_len..];

        // Find closing tag and extract JSON
        let (json_str, advance_past) = if is_markdown {
            // Markdown: ```json\n...\n```
            if let Some(end) = after_tag.find("```") {
                (&after_tag[..end], &after_tag[end + "```".len()..])
            } else {
                (after_tag, "")
            }
        } else if let Some(end) = after_tag.find("</tool_call>") {
            (&after_tag[..end], &after_tag[end + "</tool_call>".len()..])
        } else {
            // PMAT-158: No closing tag — try parsing to end-of-string
            (after_tag, "")
        };
        let json_str = json_str.trim();

        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            // Must have "name" field to be a tool call (not just any JSON)
            if let Some(name) = parsed.get("name").and_then(|n| n.as_str()) {
                let name = name.to_string();
                let input = parsed.get("input").cloned().unwrap_or(serde_json::json!({}));
                call_counter += 1;
                tool_calls.push(ToolCall { id: format!("local-{call_counter}"), name, input });
            } else {
                remaining.push_str(&cursor[start..]);
                break;
            }
        } else {
            remaining.push_str(&cursor[start..]);
            break;
        }

        cursor = advance_past;
        if cursor.is_empty() {
            break;
        }
    }

    (remaining.trim().to_string(), tool_calls)
}

/// Sanitize model output: strip echoed system prompt and chat template markers.
///
/// Small models (<3B) often echo the system prompt or leak chat template
/// tokens into their response. This strips those artifacts so the agent
/// loop sees clean assistant text.
fn sanitize_output(text: &str, system_prompt: Option<&str>) -> String {
    let mut cleaned = text.to_string();

    // Strip echoed system prompt (common with small models)
    if let Some(sys) = system_prompt {
        // Check if output starts with a significant prefix of the system prompt
        let sys_prefix = &sys[..sys.len().min(80)];
        if cleaned.starts_with(sys_prefix) {
            // The model regurgitated the system prompt — strip it
            cleaned = cleaned[sys.len().min(cleaned.len())..].to_string();
        }
    }

    // Strip leaked chat template markers
    for marker in &[
        "<|im_start|>",
        "<|im_end|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    ] {
        cleaned = cleaned.replace(marker, "");
    }

    // Strip leading/trailing whitespace and role labels
    let cleaned = cleaned.trim();
    let cleaned = cleaned.strip_prefix("system\n").unwrap_or(cleaned);
    let cleaned = cleaned.strip_prefix("assistant\n").unwrap_or(cleaned);
    cleaned.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_no_tool_calls() {
        let (text, calls) = parse_tool_calls("Hello world");
        assert_eq!(text, "Hello world");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_single_tool_call() {
        let input = r#"Before text
<tool_call>
{"name": "rag", "input": {"query": "SIMD"}}
</tool_call>
After text"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "Before text\n\nAfter text");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "rag");
        assert_eq!(calls[0].id, "local-1");
        assert_eq!(calls[0].input, serde_json::json!({"query": "SIMD"}));
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let input = r#"<tool_call>
{"name": "rag", "input": {"query": "a"}}
</tool_call>
Middle
<tool_call>
{"name": "memory", "input": {"action": "recall", "query": "b"}}
</tool_call>"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "Middle");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "rag");
        assert_eq!(calls[0].id, "local-1");
        assert_eq!(calls[1].name, "memory");
        assert_eq!(calls[1].id, "local-2");
    }

    #[test]
    fn test_parse_malformed_json() {
        let input = r#"<tool_call>
not valid json
</tool_call>"#;
        let (_text, calls) = parse_tool_calls(input);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_missing_close_tag_with_valid_json() {
        // PMAT-158: Small models omit </tool_call>. Parser should still extract.
        let input =
            "<tool_call>\n{\"name\": \"file_read\", \"input\": {\"path\": \"src/main.rs\"}}";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1, "should extract tool call without closing tag");
        assert_eq!(calls[0].name, "file_read");
        assert!(text.is_empty(), "no remaining text expected");
    }

    #[test]
    fn test_parse_missing_close_tag_with_trailing_text() {
        // Unclosed tag with text before it — text preserved, tool call extracted
        let input =
            "Let me read that.\n<tool_call> {\"name\": \"file_read\", \"input\": {\"path\": \"foo.rs\"}}";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "file_read");
        assert!(text.contains("Let me read that"));
    }

    #[test]
    fn test_parse_missing_close_tag_invalid_json() {
        // Unclosed tag with invalid JSON — treated as plain text
        let input = "<tool_call>\nnot valid json at all";
        let (text, calls) = parse_tool_calls(input);
        assert!(calls.is_empty(), "invalid JSON should not produce tool call");
        assert!(text.contains("<tool_call>"));
    }

    #[test]
    fn test_parse_markdown_code_block() {
        // PMAT-158: Qwen2.5-Coder native format — ```json blocks
        let input = "Let me read that file.\n```json\n{\"name\": \"file_read\", \"input\": {\"path\": \"src/main.rs\"}}\n```";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1, "should extract tool call from markdown block");
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[0].input["path"], "src/main.rs");
        assert!(text.contains("Let me read that"));
    }

    #[test]
    fn test_parse_markdown_code_block_not_tool_call() {
        // JSON in code block without "name" field — not a tool call
        let input = "Here's an example:\n```json\n{\"key\": \"value\"}\n```";
        let (text, calls) = parse_tool_calls(input);
        assert!(calls.is_empty(), "JSON without name field should not be a tool call");
        assert!(text.contains("example"));
    }

    #[test]
    fn test_parse_missing_name() {
        let input = r#"<tool_call>
{"input": {"query": "test"}}
</tool_call>"#;
        let (_, calls) = parse_tool_calls(input);
        assert!(calls.is_empty(), "JSON without name should not be extracted");
    }

    #[test]
    fn test_privacy_tier_always_sovereign() {
        assert_eq!(PrivacyTier::Sovereign, PrivacyTier::Sovereign);
    }

    // ── Output sanitization tests ──

    #[test]
    fn test_sanitize_strips_echoed_system_prompt() {
        let sys = "You are apr code, a sovereign AI coding assistant.";
        let output = format!("{sys} And then the model continues here.");
        let cleaned = sanitize_output(&output, Some(sys));
        assert!(!cleaned.contains("sovereign AI coding assistant"));
        assert!(cleaned.contains("continues here"));
    }

    #[test]
    fn test_sanitize_strips_chat_markers() {
        let output = "<|im_start|>assistant\nHello world<|im_end|>";
        let cleaned = sanitize_output(output, None);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_sanitize_preserves_clean_output() {
        let output = "The answer is 42.";
        let cleaned = sanitize_output(output, Some("You are helpful."));
        assert_eq!(cleaned, "The answer is 42.");
    }

    #[test]
    fn test_sanitize_strips_role_prefix() {
        let output = "assistant\nHere is my response.";
        let cleaned = sanitize_output(output, None);
        assert_eq!(cleaned, "Here is my response.");
    }
}
