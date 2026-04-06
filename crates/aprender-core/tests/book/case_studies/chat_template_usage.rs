#![allow(clippy::disallowed_methods)]
//! Chat Template Usage Tests
//!
//! These tests validate the chat template examples in the EXTREME TDD book.
//! Every code snippet in the chat template chapter must be tested here.

use aprender::text::chat_template::{
    auto_detect_template, create_template, detect_format_from_name, ChatMLTemplate, ChatMessage,
    ChatTemplateEngine, HuggingFaceTemplate, Llama2Template, MistralTemplate, SpecialTokens,
    TemplateFormat,
};

/// Book Example: Basic ChatML usage
#[test]
fn book_example_chatml_basic() {
    let template = ChatMLTemplate::new();
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Hello!"),
    ];

    let output = template.format_conversation(&messages).unwrap();

    assert!(output.contains("<|im_start|>system"));
    assert!(output.contains("<|im_start|>user"));
    assert!(output.contains("<|im_end|>"));
    assert!(output.ends_with("<|im_start|>assistant\n"));
}

/// Book Example: LLaMA2 with system prompt
#[test]
fn book_example_llama2_system_prompt() {
    let template = Llama2Template::new();
    let messages = vec![
        ChatMessage::system("You are a coding assistant."),
        ChatMessage::user("Write hello world"),
    ];

    let output = template.format_conversation(&messages).unwrap();

    assert!(output.starts_with("<s>"));
    assert!(output.contains("<<SYS>>"));
    assert!(output.contains("You are a coding assistant."));
    assert!(output.contains("<</SYS>>"));
    assert!(output.contains("[INST]"));
}

/// Book Example: Mistral (no system prompt)
#[test]
fn book_example_mistral_no_system() {
    let template = MistralTemplate::new();

    // Mistral doesn't support system prompts
    assert!(!template.supports_system_prompt());

    let messages = vec![
        ChatMessage::system("This will be ignored"),
        ChatMessage::user("Hello Mistral!"),
    ];

    let output = template.format_conversation(&messages).unwrap();

    // System prompt should NOT appear in output
    assert!(!output.contains("This will be ignored"));
    assert!(output.contains("[INST]"));
    assert!(output.contains("Hello Mistral!"));
}

/// Book Example: Auto-detection from model name
#[test]
fn book_example_auto_detection() {
    // TinyLlama -> LLaMA2 format
    assert_eq!(
        detect_format_from_name("TinyLlama-1.1B-Chat"),
        TemplateFormat::Llama2
    );

    // Qwen -> ChatML format
    assert_eq!(
        detect_format_from_name("Qwen2-0.5B-Instruct"),
        TemplateFormat::ChatML
    );

    // Mistral -> Mistral format
    assert_eq!(
        detect_format_from_name("Mistral-7B-Instruct"),
        TemplateFormat::Mistral
    );

    // Phi -> Phi format
    assert_eq!(detect_format_from_name("phi-2"), TemplateFormat::Phi);
}

/// Book Example: Create template from format enum
#[test]
fn book_example_create_from_enum() {
    let template = create_template(TemplateFormat::ChatML);
    assert_eq!(template.format(), TemplateFormat::ChatML);
    assert!(template.supports_system_prompt());

    let template = create_template(TemplateFormat::Mistral);
    assert_eq!(template.format(), TemplateFormat::Mistral);
    assert!(!template.supports_system_prompt());
}

/// Book Example: Multi-turn conversation
#[test]
fn book_example_multi_turn() {
    let template = ChatMLTemplate::new();
    let messages = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("What is 2+2?"),
        ChatMessage::assistant("4"),
        ChatMessage::user("And 3+3?"),
    ];

    let output = template.format_conversation(&messages).unwrap();

    // All messages should be in order
    let sys_pos = output.find("You are helpful.").unwrap();
    let user1_pos = output.find("What is 2+2?").unwrap();
    let asst_pos = output.find("4").unwrap();
    let user2_pos = output.find("And 3+3?").unwrap();

    assert!(sys_pos < user1_pos);
    assert!(user1_pos < asst_pos);
    assert!(asst_pos < user2_pos);
}

/// Book Example: Custom Jinja2 template
#[test]
fn book_example_custom_jinja2() {
    let template_str = r#"{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}"#;

    let template = HuggingFaceTemplate::new(
        template_str.to_string(),
        SpecialTokens::default(),
        TemplateFormat::Custom,
    )
    .expect("Template creation failed");

    let messages = vec![
        ChatMessage::user("Hello"),
        ChatMessage::assistant("Hi there"),
    ];

    let output = template.format_conversation(&messages).unwrap();
    assert!(output.contains("user: Hello"));
    assert!(output.contains("assistant: Hi there"));
}

/// Book Example: Auto-detect and create in one step
#[test]
fn book_example_auto_detect_template() {
    let template = auto_detect_template("tinyllama-chat");
    assert_eq!(template.format(), TemplateFormat::Llama2);

    let template = auto_detect_template("qwen2-instruct");
    assert_eq!(template.format(), TemplateFormat::ChatML);
}
