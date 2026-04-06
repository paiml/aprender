//! Contract: chat-template-v1 enforcement tests (PMAT-186)
//!
//! Falsification tests for batuta's chat template system.
//! Tests template detection, content preservation, format determinism,
//! tool injection, and APR/GGUF format support.

use super::*;
use crate::agent::driver::ToolDefinition;

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
fn falsify_ct_batuta_001_content_preserved_across_formats() {
    // FALSIFY-CT-BATUTA-001: User message content must appear in output
    // for ALL template formats — no content loss during formatting.
    let user_msg = "Fix the authentication bug in src/auth.rs line 42";
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![Message::User(user_msg.into())],
        tools: vec![],
        max_tokens: 100,
        temperature: 0.0,
        system: Some("You are a coding assistant.".into()),
    };
    for template in [ChatTemplate::ChatMl, ChatTemplate::Llama3, ChatTemplate::Generic] {
        let prompt = format_prompt_with_template(&request, template);
        assert!(
            prompt.contains(user_msg),
            "FALSIFY-CT-BATUTA-001: user content lost in {template:?}"
        );
        assert!(
            prompt.contains("You are a coding assistant."),
            "FALSIFY-CT-BATUTA-001: system content lost in {template:?}"
        );
    }
}

#[test]
fn falsify_ct_batuta_002_format_determinism() {
    // FALSIFY-CT-BATUTA-002: Same input always produces same output.
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::User("Hello".into()),
            Message::Assistant("Hi there".into()),
            Message::User("Follow up".into()),
        ],
        tools: vec![],
        max_tokens: 100,
        temperature: 0.0,
        system: Some("System".into()),
    };
    for template in [ChatTemplate::ChatMl, ChatTemplate::Llama3, ChatTemplate::Generic] {
        let a = format_prompt_with_template(&request, template);
        let b = format_prompt_with_template(&request, template);
        assert_eq!(a, b, "FALSIFY-CT-BATUTA-002: non-deterministic output for {template:?}");
    }
}

#[test]
fn falsify_ct_batuta_003_qwen_gets_chatml() {
    // FALSIFY-CT-BATUTA-003: All Qwen model filenames must resolve to ChatML.
    use std::path::Path;
    for name in
        &["qwen2.5-coder-1.5b-q4k.gguf", "qwen3-1.7b-q4k.apr", "Qwen3-8B-Q4K.gguf", "qwen2-7b.gguf"]
    {
        assert_eq!(
            ChatTemplate::from_model_path(Path::new(name)),
            ChatTemplate::ChatMl,
            "FALSIFY-CT-BATUTA-003: Qwen model '{name}' must get ChatML"
        );
    }
}

#[test]
fn falsify_ct_batuta_004_llama_gets_llama3() {
    // FALSIFY-CT-BATUTA-004: All Llama model filenames must resolve to Llama3.
    use std::path::Path;
    for name in &["llama-3.2-3b.gguf", "Meta-Llama-3-8B.apr", "LLAMA-2-7b.gguf"] {
        assert_eq!(
            ChatTemplate::from_model_path(Path::new(name)),
            ChatTemplate::Llama3,
            "FALSIFY-CT-BATUTA-004: Llama model '{name}' must get Llama3"
        );
    }
}

#[test]
fn falsify_ct_batuta_005_unknown_defaults_to_chatml() {
    // FALSIFY-CT-BATUTA-005: Unknown model names default to ChatML.
    use std::path::Path;
    for name in &["mystery-model.gguf", "custom.apr", "phi-3.5.gguf", "gemma-2b.gguf"] {
        assert_eq!(
            ChatTemplate::from_model_path(Path::new(name)),
            ChatTemplate::ChatMl,
            "FALSIFY-CT-BATUTA-005: Unknown model '{name}' must default to ChatML"
        );
    }
}

#[test]
fn falsify_ct_batuta_006_tool_injection_completeness() {
    // FALSIFY-CT-BATUTA-006: All registered tools must appear in enriched prompt.
    let tools = sample_tools();
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![Message::User("test".into())],
        tools: tools.clone(),
        max_tokens: 100,
        temperature: 0.0,
        system: Some("base".into()),
    };
    let prompt = format_prompt_with_template(&request, ChatTemplate::ChatMl);
    for tool in &tools {
        assert!(
            prompt.contains(&tool.name),
            "FALSIFY-CT-BATUTA-006: tool '{}' missing from enriched prompt",
            tool.name
        );
        assert!(
            prompt.contains(&tool.description),
            "FALSIFY-CT-BATUTA-006: description for '{}' missing",
            tool.name
        );
    }
}

#[test]
fn falsify_ct_batuta_007_chatml_ends_with_assistant_prefix() {
    // FALSIFY-CT-BATUTA-007: ChatML must end with assistant prefix for generation.
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![Message::User("test".into())],
        tools: vec![],
        max_tokens: 100,
        temperature: 0.0,
        system: None,
    };
    let prompt = format_prompt_with_template(&request, ChatTemplate::ChatMl);
    assert!(
        prompt.ends_with("<|im_start|>assistant\n"),
        "FALSIFY-CT-BATUTA-007: ChatML must end with assistant generation prefix"
    );
}

#[test]
fn falsify_ct_batuta_008_llama3_ends_with_assistant_prefix() {
    // FALSIFY-CT-BATUTA-008: Llama3 must end with assistant header for generation.
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![Message::User("test".into())],
        tools: vec![],
        max_tokens: 100,
        temperature: 0.0,
        system: None,
    };
    let prompt = format_prompt_with_template(&request, ChatTemplate::Llama3);
    assert!(
        prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "FALSIFY-CT-BATUTA-008: Llama3 must end with assistant header prefix"
    );
}

#[test]
fn falsify_ct_batuta_009_multi_turn_ordering() {
    // FALSIFY-CT-BATUTA-009: Multi-turn messages must appear in order.
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::User("first".into()),
            Message::Assistant("second".into()),
            Message::User("third".into()),
        ],
        tools: vec![],
        max_tokens: 100,
        temperature: 0.0,
        system: None,
    };
    let prompt = format_prompt_with_template(&request, ChatTemplate::ChatMl);
    let first_pos = prompt.find("first").expect("first missing");
    let second_pos = prompt.find("second").expect("second missing");
    let third_pos = prompt.find("third").expect("third missing");
    assert!(
        first_pos < second_pos && second_pos < third_pos,
        "FALSIFY-CT-BATUTA-009: messages must appear in order"
    );
}

#[test]
fn falsify_ct_batuta_010_apr_and_gguf_both_supported() {
    // FALSIFY-CT-BATUTA-010: Both APR and GGUF extensions produce valid templates.
    use std::path::Path;
    let apr_template = ChatTemplate::from_model_path(Path::new("qwen3-1.7b-q4k.apr"));
    let gguf_template = ChatTemplate::from_model_path(Path::new("qwen3-1.7b-q4k.gguf"));
    assert_eq!(
        apr_template, gguf_template,
        "FALSIFY-CT-BATUTA-010: APR and GGUF of same model must get same template"
    );
}
