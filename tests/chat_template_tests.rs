#![allow(clippy::disallowed_methods)]
use aprender::text::chat_template::{
    ChatMessage, ChatTemplateEngine, HuggingFaceTemplate, SpecialTokens, TemplateFormat,
};

#[test]
fn test_chatml_template_rendering() {
    let template_str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    let special_tokens = SpecialTokens {
        im_start_token: Some("<|im_start|>".to_string()),
        im_end_token: Some("<|im_end|>".to_string()),
        ..Default::default()
    };

    let engine = HuggingFaceTemplate::new(
        template_str.to_string(),
        special_tokens,
        TemplateFormat::ChatML,
    )
    .expect("Failed to create engine");

    let messages = vec![
        ChatMessage::new("system", "You are a helpful assistant."),
        ChatMessage::new("user", "Hello"),
    ];

    let output = engine
        .format_conversation(&messages)
        .expect("Failed to render");

    let expected = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";

    assert_eq!(output, expected);
}

#[test]
fn test_llama2_template_rendering() {
    // Simplified LLaMA 2 template logic for testing
    let template_str = "{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST] {% elif message['role'] == 'assistant' %}{{ message['content'] }} {% endif %}{% endfor %}";

    let engine = HuggingFaceTemplate::new(
        template_str.to_string(),
        SpecialTokens::default(),
        TemplateFormat::Llama2,
    )
    .expect("Failed to create engine");

    let messages = vec![
        ChatMessage::new("user", "Hello"),
        ChatMessage::new("assistant", "Hi there"),
    ];

    let output = engine
        .format_conversation(&messages)
        .expect("Rendering failed");
    let expected = "[INST] Hello [/INST] Hi there ";

    assert_eq!(output, expected);
}
