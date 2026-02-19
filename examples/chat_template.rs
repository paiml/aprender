#![allow(clippy::disallowed_methods)]
//! Chat Template Example
//!
//! Demonstrates the chat template system for formatting LLM conversations.
//!
//! This example shows:
//! - Auto-detection of template format from model name
//! - Manual template selection
//! - Multi-turn conversation formatting
//! - All supported template formats
//!
//! # Run
//!
//! ```bash
//! cargo run --example chat_template
//! ```

use aprender::text::chat_template::{
    auto_detect_template, create_template, detect_format_from_name, AlpacaTemplate, ChatMLTemplate,
    ChatMessage, ChatTemplateEngine, HuggingFaceTemplate, Llama2Template, MistralTemplate,
    PhiTemplate, RawTemplate, SpecialTokens, TemplateFormat,
};

fn main() {
    println!("=== Chat Template System Demo ===\n");

    // -------------------------------------------------------------------------
    // 1. Auto-detection from model name
    // -------------------------------------------------------------------------
    println!("1. Auto-Detection from Model Name\n");

    let models = [
        "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",
        "Qwen2-0.5B-Instruct",
        "Mistral-7B-Instruct-v0.2",
        "phi-2",
        "alpaca-7b",
        "unknown-model",
    ];

    for model in models {
        let format = detect_format_from_name(model);
        println!("  {} -> {:?}", model, format);
    }

    println!();

    // -------------------------------------------------------------------------
    // 2. ChatML Format (Qwen2, OpenHermes, Yi)
    // -------------------------------------------------------------------------
    println!("2. ChatML Format (Qwen2, OpenHermes, Yi)\n");

    let chatml = ChatMLTemplate::new();
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("What is 2+2?"),
        ChatMessage::assistant("4"),
        ChatMessage::user("And 3+3?"),
    ];

    let output = chatml
        .format_conversation(&messages)
        .expect("ChatML format failed");
    println!("Output:\n{}", output);
    println!();

    // -------------------------------------------------------------------------
    // 3. LLaMA2 Format (TinyLlama, Vicuna, LLaMA 2)
    // -------------------------------------------------------------------------
    println!("3. LLaMA2 Format (TinyLlama, Vicuna)\n");

    let llama2 = Llama2Template::new();
    let messages = vec![
        ChatMessage::system("You are a coding assistant."),
        ChatMessage::user("Write hello world in Python"),
    ];

    let output = llama2
        .format_conversation(&messages)
        .expect("LLaMA2 format failed");
    println!("Output:\n{}", output);
    println!();

    // -------------------------------------------------------------------------
    // 4. Mistral Format (no system prompt support)
    // -------------------------------------------------------------------------
    println!("4. Mistral Format (no system prompt)\n");

    let mistral = MistralTemplate::new();
    println!(
        "Supports system prompt: {}",
        mistral.supports_system_prompt()
    );

    let messages = vec![
        ChatMessage::system("This will be ignored"),
        ChatMessage::user("Hello Mistral!"),
    ];

    let output = mistral
        .format_conversation(&messages)
        .expect("Mistral format failed");
    println!("Output:\n{}", output);
    println!();

    // -------------------------------------------------------------------------
    // 5. Phi Format (Phi-2, Phi-3)
    // -------------------------------------------------------------------------
    println!("5. Phi Format (Phi-2, Phi-3)\n");

    let phi = PhiTemplate::new();
    let messages = vec![ChatMessage::user("Explain quantum computing")];

    let output = phi
        .format_conversation(&messages)
        .expect("Phi format failed");
    println!("Output:\n{}", output);
    println!();

    // -------------------------------------------------------------------------
    // 6. Alpaca Format
    // -------------------------------------------------------------------------
    println!("6. Alpaca Format\n");

    let alpaca = AlpacaTemplate::new();
    let messages = vec![
        ChatMessage::system("You are a helpful AI assistant."),
        ChatMessage::user("Summarize this text"),
    ];

    let output = alpaca
        .format_conversation(&messages)
        .expect("Alpaca format failed");
    println!("Output:\n{}", output);
    println!();

    // -------------------------------------------------------------------------
    // 7. Raw Format (fallback/passthrough)
    // -------------------------------------------------------------------------
    println!("7. Raw Format (fallback)\n");

    let raw = RawTemplate::new();
    let messages = vec![ChatMessage::user("Just pass this through")];

    let output = raw
        .format_conversation(&messages)
        .expect("Raw format failed");
    println!("Output:\n{}", output);
    println!();

    // -------------------------------------------------------------------------
    // 8. Custom HuggingFace Template (Jinja2)
    // -------------------------------------------------------------------------
    println!("8. Custom HuggingFace Template (Jinja2)\n");

    let template_str = r#"{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}Assistant:"#;

    let hf_template = HuggingFaceTemplate::new(
        template_str.to_string(),
        SpecialTokens::default(),
        TemplateFormat::Custom,
    )
    .expect("Failed to create HuggingFace template");

    let messages = vec![
        ChatMessage::user("Hello!"),
        ChatMessage::assistant("Hi there!"),
        ChatMessage::user("How are you?"),
    ];

    let output = hf_template
        .format_conversation(&messages)
        .expect("HF template failed");
    println!("Output:\n{}", output);
    println!();

    // -------------------------------------------------------------------------
    // 9. Auto-detect and create template
    // -------------------------------------------------------------------------
    println!("9. Auto-Detect and Create Template\n");

    let template = auto_detect_template("tinyllama-1.1b-chat");
    println!("Auto-detected format: {:?}", template.format());

    let messages = vec![
        ChatMessage::system("Be concise."),
        ChatMessage::user("What is Rust?"),
    ];

    let output = template
        .format_conversation(&messages)
        .expect("Auto-detect failed");
    println!("Output:\n{}", output);
    println!();

    // -------------------------------------------------------------------------
    // 10. Create template from enum
    // -------------------------------------------------------------------------
    println!("10. Create Template from Enum\n");

    for format in [
        TemplateFormat::ChatML,
        TemplateFormat::Llama2,
        TemplateFormat::Mistral,
        TemplateFormat::Phi,
        TemplateFormat::Alpaca,
        TemplateFormat::Raw,
    ] {
        let template = create_template(format);
        println!(
            "  {:?}: supports_system={}",
            format,
            template.supports_system_prompt()
        );
    }

    println!("\n=== Demo Complete ===");
}
