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
    ///
    /// **Superseded by [`format_prompt_for_model`] on the local-inference path
    /// (#3801).** This guessed from the FILE NAME and had no `Qwen3NoThink`, so
    /// every `qwen*` — including Qwen3 and Qwen3.5 — was rendered as plain
    /// ChatML, which leaves the model in thinking mode. `apr serve`, `apr run
    /// --chat`, `apr chat` and `apr qa`'s golden gate all take realizar's
    /// detector, keyed on `general.architecture`, and give those models the
    /// no-think template. This was the fourth mechanism answering that question,
    /// and the only one reading a filename.
    ///
    /// Kept only for the Llama-3 rendering realizar has no template for, and for
    /// callers that already hold a `ChatTemplate`.
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

/// Does this name denote Llama **3**, whose header format realizar cannot express?
///
/// realizar's detector maps every `llama` to `Llama2` (`[INST] … [/INST]`), and it
/// has no Llama-3 template at all. Delegating a Llama-3 model to it would be a
/// downgrade, so this is the ONE case that keeps the local renderer — stated
/// rather than silently forked.
fn is_llama3_name(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("llama-3") || n.contains("llama3") || n.contains("llama_3")
}

/// The architecture a GGUF declares, read from a bounded PREFIX of the file.
///
/// Never an mmap: `MappedGGUFModel::from_path` maps with `MAP_POPULATE` and
/// pre-faults every page (measured at 18 GB on a 30B MoE, #3817). GGUF puts the
/// header, metadata and tensor info before the tensor data, so a prefix carries
/// `general.architecture`. `None` for anything that is not a readable GGUF —
/// the caller then falls back to the file name, which is what this path used to
/// do for everything.
/// The GGUF header (metadata incl. `tokenizer.chat_template` and the vocabulary), parsed
/// from the smallest bounded prefix that holds it -- the same no-mmap rule as above.
#[cfg(feature = "inference")]
fn declared_header(path: &std::path::Path) -> Option<realizar::gguf::GGUFModel> {
    use std::io::Read;
    if !path.is_file() {
        return None;
    }
    for prefix in [1_usize << 20, 8 << 20, 64 << 20] {
        let file = std::fs::File::open(path).ok()?;
        let mut buf = Vec::new();
        if file.take(prefix as u64).read_to_end(&mut buf).is_err() {
            return None;
        }
        let read = buf.len();
        if let Ok(model) = realizar::gguf::GGUFModel::from_bytes(&buf) {
            return Some(model);
        }
        if read < prefix {
            return None;
        }
    }
    None
}

/// #3801: render the prompt the way PRODUCTION renders it for this model.
///
/// The selection comes from realizar's `detect_format_from_name` — the same
/// function `apr serve`, `apr run --chat`, `apr chat` and `apr qa` ask — keyed on
/// the GGUF's `general.architecture`, falling back to the file name only when the
/// file declares none. The rendering is delegated to realizar's own template, so
/// `apr code` cannot drift from the other verbs.
///
/// **The one exception, named:** a Llama-3 model keeps the local `format_llama3`,
/// because realizar maps every `llama` to `Llama2` and has no Llama-3 template.
/// Delegating there would replace `<|start_header_id|>` with `[INST]`.
///
/// Tool definitions are still injected into the system prompt here — that is this
/// crate's job, not the template's — and tool-use / tool-result turns are
/// flattened to the assistant and user turns they already rendered as.
#[cfg(feature = "inference")]
pub fn format_prompt_for_model(
    request: &CompletionRequest,
    model_path: &std::path::Path,
) -> String {
    let name = model_path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let header = declared_header(model_path);
    let msgs = realizar_messages(request);
    // #3990: the GGUF's OWN chat_template is the prompt format the model was trained on; it
    // outranks every detector below, the Llama-3 exception included.
    if let Some(h) = header.as_ref().filter(|h| h.metadata.contains_key("tokenizer.chat_template"))
    {
        // Thinking OFF, as every production verb renders it (#3801).
        match realizar::chat_template::render_official_for_model(h, &msgs, Some(false)) {
            Ok(prompt) => return prompt,
            Err(e) => eprintln!(
                "[#3990] WARNING: {}'s own chat_template failed to render ({e}); falling back to a \
                 hand-coded template, which is NOT the prompt format this model was trained on",
                model_path.display()
            ),
        }
    }
    if is_llama3_name(&name) {
        return format_prompt_with_template(request, ChatTemplate::Llama3);
    }
    let key = header.as_ref().and_then(|h| h.architecture().map(str::to_string)).unwrap_or(name);
    realizar::chat_template::format_messages(&msgs, Some(&key))
        .unwrap_or_else(|_| format_prompt_with_template(request, ChatTemplate::ChatMl))
}

/// The request as realizar chat messages: tool definitions injected into the system turn
/// (this crate's job, not the template's), tool-use / tool-result turns flattened to the
/// assistant and user turns they render as.
#[cfg(feature = "inference")]
fn realizar_messages(request: &CompletionRequest) -> Vec<realizar::chat_template::ChatMessage> {
    let enriched_system = build_enriched_system(&request.system, &request.tools);
    let mut msgs: Vec<realizar::chat_template::ChatMessage> = Vec::new();
    if !enriched_system.is_empty() {
        msgs.push(realizar::chat_template::ChatMessage::new("system", enriched_system));
    }
    for m in &request.messages {
        let (role, content) = match m {
            Message::System(s) => ("system", s.clone()),
            Message::User(s) => ("user", s.clone()),
            Message::Assistant(s) => ("assistant", s.clone()),
            Message::AssistantToolUse(call) => (
                "assistant",
                format!(
                    "<tool_call>\n{}\n</tool_call>",
                    serde_json::json!({"name": call.name, "input": call.input})
                ),
            ),
            Message::ToolResult(result) => {
                ("user", format!("<tool_result>{}</tool_result>", result.content))
            }
        };
        msgs.push(realizar::chat_template::ChatMessage::new(role, content));
    }
    msgs
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

// =============================================================================
// #3801: apr code asks the same detector as every other verb
// =============================================================================

#[cfg(all(test, feature = "inference"))]
mod one_detector_tests {
    use super::*;
    use crate::agent::driver::{CompletionRequest, Message};
    use std::path::Path;

    fn req(user: &str) -> CompletionRequest {
        CompletionRequest {
            system: None,
            model: String::new(),
            messages: vec![Message::User(user.to_string())],
            tools: Vec::new(),
            max_tokens: 64,
            temperature: 0.0,
        }
    }

    /// THE DEFECT. A Qwen3 model rendered plain ChatML, which leaves the model in
    /// thinking mode — `apr serve`, `apr run --chat`, `apr chat` and `apr qa` all
    /// give it the no-think template. The assistant turn must now end with the
    /// pre-closed think block.
    #[test]
    fn a_qwen3_model_gets_the_production_no_think_prompt() {
        let prompt =
            format_prompt_for_model(&req("What is 2+2?"), Path::new("Qwen3-1.7B-Q4_K_M.gguf"));
        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n</think>\n"),
            "expected the no-think prefill, got: {prompt:?}"
        );
        assert!(prompt.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>"), "{prompt}");
    }

    /// The old path's answer for the same file, kept as the negative control: a
    /// bare ChatML assistant turn is what left the model reasoning.
    #[test]
    fn the_old_filename_guess_produced_the_thinking_prompt() {
        let old = format_prompt_with_template(&req("What is 2+2?"), ChatTemplate::ChatMl);
        assert!(old.ends_with("<|im_start|>assistant\n"), "{old}");
        assert!(!old.contains("<think>"), "the old rendering left thinking ON: {old}");
    }

    /// A Qwen2 model keeps plain ChatML — the unification changes the models whose
    /// production template changed, and nothing else.
    #[test]
    fn a_qwen2_model_still_gets_plain_chatml() {
        let prompt = format_prompt_for_model(
            &req("Hi there"),
            Path::new("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        );
        assert!(prompt.ends_with("<|im_start|>assistant\n"), "{prompt}");
        assert!(!prompt.contains("<think>"), "{prompt}");
    }

    /// THE STATED EXCEPTION. realizar maps every `llama` to Llama2 (`[INST]`) and
    /// has no Llama-3 template, so a Llama-3 model keeps the local renderer. If
    /// this ever renders `[INST]`, the unification silently downgraded it.
    #[test]
    fn a_llama3_model_keeps_the_header_format_realizar_cannot_express() {
        let prompt =
            format_prompt_for_model(&req("Hi"), Path::new("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"));
        assert!(prompt.contains("<|start_header_id|>"), "{prompt}");
        assert!(!prompt.contains("[INST]"), "{prompt}");
        assert!(is_llama3_name("Meta-Llama-3-8B-Instruct"));
        assert!(is_llama3_name("llama3-8b"));
        assert!(!is_llama3_name("Llama-2-7b-chat"));
    }

    /// Tool definitions still reach the system turn: the template selection moved,
    /// this crate's job did not.
    #[test]
    fn tool_definitions_still_reach_the_system_turn() {
        let mut r = req("do the thing");
        r.tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: serde_json::json!({"properties": {"path": {"type": "string"}}}),
        }];
        let prompt = format_prompt_for_model(&r, Path::new("Qwen3-1.7B-Q4_K_M.gguf"));
        assert!(prompt.contains("## Available Tools"), "{prompt}");
        assert!(prompt.contains("read_file"), "{prompt}");
        assert!(prompt.starts_with("<|im_start|>system\n"), "{prompt}");
    }

    /// #3990 REAL MODELS: with the GGUF on disk, `apr code` renders the model's OWN
    /// chat_template (thinking off), byte-equal to llama.cpp's /apply-template on every
    /// thinking=false cell of the oracle. SKIP by name for a file not on this host.
    #[test]
    fn a_real_gguf_renders_its_own_template_equal_to_llama_cpp_3990() {
        // #4129: read at RUN time; an include_str! of a sibling crate's file cannot compile
        // from the published aprender-orchestrate tarball. In tree a missing oracle FAILS;
        // out of tree (no workspace contracts/) the test SKIPs by name.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        if !root.join("contracts").is_dir() {
            eprintln!(
                "SKIP a_real_gguf_renders_its_own_template_equal_to_llama_cpp_3990: out of tree \
                 (no workspace contracts/ beside this crate) - the #3990 oracle lives in \
                 aprender-serve's fixtures, which a published crate does not carry (#4129)"
            );
            return;
        }
        let oracle_path = root
            .join("crates/aprender-serve/src/fixtures/chat_template_3990/llama_cpp_df03399.json");
        let oracle = std::fs::read_to_string(&oracle_path)
            .unwrap_or_else(|e| panic!("in tree, {} must be readable: {e}", oracle_path.display()));
        let cells: Vec<serde_json::Value> = serde_json::from_str(&oracle).expect("oracle parses");
        let mut ran = 0usize;
        for c in cells.iter().filter(|c| c["thinking"] == false) {
            let path = Path::new(c["path"].as_str().expect("path"));
            if !path.exists() {
                eprintln!("SKIP: {} not on this host -- this cell did NOT run", path.display());
                continue;
            }
            let mut r = req("");
            r.messages.clear();
            for m in c["messages"].as_array().expect("messages") {
                let content = m["content"].as_str().expect("content").to_string();
                match m["role"].as_str().expect("role") {
                    "system" => r.system = Some(content),
                    _ => r.messages.push(Message::User(content)),
                }
            }
            let got = format_prompt_for_model(&r, path);
            assert_eq!(
                got,
                c["prompt"].as_str().expect("prompt"),
                "{} system={}",
                path.display(),
                c["system"]
            );
            ran += 1;
        }
        eprintln!("#3990 apr code: {ran}/8 real cells compared");
    }
}
