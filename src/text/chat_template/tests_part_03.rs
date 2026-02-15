
/// Very long conversation (50+ turns)
#[test]
fn test_long_conversation() {
    let template = ChatMLTemplate::new();
    let mut messages = vec![ChatMessage::system("System")];

    for i in 0..50 {
        messages.push(ChatMessage::user(format!("User message {i}")));
        messages.push(ChatMessage::assistant(format!("Assistant response {i}")));
    }

    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
    let output = result.expect("format failed");
    // Should contain all messages
    assert!(output.contains("User message 49"));
    assert!(output.contains("Assistant response 49"));
}

/// Binary-like content doesn't crash
#[test]
fn test_binary_content_handling() {
    let template = ChatMLTemplate::new();
    let binary_content = String::from_utf8_lossy(&[0x00, 0x01, 0x02, 0xFF, 0xFE]).to_string();
    let messages = vec![ChatMessage::user(&binary_content)];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
}

/// RTL text preserved
#[test]
fn test_rtl_text_preserved() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("مرحبا بالعالم")]; // Arabic "Hello World"
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
    let output = result.expect("format failed");
    assert!(output.contains("مرحبا بالعالم"));
}

/// Mixed roles including custom
#[test]
fn test_custom_role() {
    let template = ChatMLTemplate::new();
    let messages = vec![
        ChatMessage::new("tool", "Function result: 42"),
        ChatMessage::user("What was the result?"),
    ];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
    let output = result.expect("format failed");
    assert!(output.contains("tool"));
    assert!(output.contains("Function result: 42"));
}

/// Newlines in content preserved
#[test]
fn test_multiline_content() {
    let template = ChatMLTemplate::new();
    let multiline = "Line 1\nLine 2\nLine 3";
    let messages = vec![ChatMessage::user(multiline)];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
    let output = result.expect("format failed");
    assert!(output.contains("Line 1\nLine 2\nLine 3"));
}

/// Format creation is idempotent
#[test]
fn test_format_creation_idempotent() {
    let t1 = create_template(TemplateFormat::ChatML);
    let t2 = create_template(TemplateFormat::ChatML);

    let messages = vec![ChatMessage::user("Test")];
    let o1 = t1.format_conversation(&messages).expect("format1");
    let o2 = t2.format_conversation(&messages).expect("format2");

    assert_eq!(o1, o2);
}

/// HuggingFaceTemplate Debug implementation
#[test]
fn test_hf_template_debug() {
    let json = r#"{
            "chat_template": "{{ message.content }}",
            "bos_token": "<s>"
        }"#;

    let template = HuggingFaceTemplate::from_json(json).expect("parse failed");
    let debug_str = format!("{:?}", template);
    assert!(debug_str.contains("HuggingFaceTemplate"));
    assert!(debug_str.contains("template_str"));
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

/// Test detect_format_from_tokens with ChatML tokens
#[test]
fn test_detect_format_from_tokens_chatml() {
    let tokens = SpecialTokens {
        im_start_token: Some("<|im_start|>".to_string()),
        im_end_token: Some("<|im_end|>".to_string()),
        ..Default::default()
    };
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::ChatML);
}

/// Test detect_format_from_tokens with only im_start
#[test]
fn test_detect_format_from_tokens_im_start_only() {
    let tokens = SpecialTokens {
        im_start_token: Some("<|im_start|>".to_string()),
        ..Default::default()
    };
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::ChatML);
}

/// Test detect_format_from_tokens with INST tokens
#[test]
fn test_detect_format_from_tokens_llama2() {
    let tokens = SpecialTokens {
        inst_start: Some("[INST]".to_string()),
        inst_end: Some("[/INST]".to_string()),
        ..Default::default()
    };
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::Llama2);
}

/// Test detect_format_from_tokens with no special tokens (Raw)
#[test]
fn test_detect_format_from_tokens_raw() {
    let tokens = SpecialTokens::default();
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::Raw);
}

/// Test Llama2Template format_message for unknown role
#[test]
fn test_llama2_format_message_unknown_role() {
    let template = Llama2Template::new();
    let result = template.format_message("tool", "tool output");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "tool output");
}

/// Test MistralTemplate format_message for system role
#[test]
fn test_mistral_format_message_system() {
    let template = MistralTemplate::new();
    let result = template.format_message("system", "system content");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "system content\n\n");
}

/// Test MistralTemplate format_message for unknown role
#[test]
fn test_mistral_format_message_unknown_role() {
    let template = MistralTemplate::new();
    let result = template.format_message("tool", "tool output");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "tool output");
}

/// Test PhiTemplate format_message for all roles
#[test]
fn test_phi_format_message_all_roles() {
    let template = PhiTemplate::new();

    let sys = template.format_message("system", "sys").unwrap();
    assert_eq!(sys, "sys\n");

    let user = template.format_message("user", "usr").unwrap();
    assert_eq!(user, "Instruct: usr\n");

    let asst = template.format_message("assistant", "asst").unwrap();
    assert_eq!(asst, "Output: asst\n");

    let unknown = template.format_message("tool", "tool").unwrap();
    assert_eq!(unknown, "tool");
}

/// Test AlpacaTemplate format_message for all roles
#[test]
fn test_alpaca_format_message_all_roles() {
    let template = AlpacaTemplate::new();

    let sys = template.format_message("system", "sys").unwrap();
    assert_eq!(sys, "sys\n\n");

    let user = template.format_message("user", "usr").unwrap();
    assert_eq!(user, "### Instruction:\nusr\n\n");

    let asst = template.format_message("assistant", "asst").unwrap();
    assert_eq!(asst, "### Response:\nasst\n\n");

    let unknown = template.format_message("tool", "tool").unwrap();
    assert_eq!(unknown, "tool");
}

/// Test RawTemplate format_message
#[test]
fn test_raw_format_message() {
    let template = RawTemplate::new();
    let result = template.format_message("any_role", "content");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "content");
}

/// Test ChatMLTemplate clone
#[test]
fn test_chatml_clone() {
    let template = ChatMLTemplate::new();
    let cloned = template.clone();
    let messages = vec![ChatMessage::user("test")];
    assert_eq!(
        template.format_conversation(&messages).unwrap(),
        cloned.format_conversation(&messages).unwrap()
    );
}

/// Test Llama2Template clone
#[test]
fn test_llama2_clone() {
    let template = Llama2Template::new();
    let cloned = template.clone();
    let messages = vec![ChatMessage::user("test")];
    assert_eq!(
        template.format_conversation(&messages).unwrap(),
        cloned.format_conversation(&messages).unwrap()
    );
}

/// Test MistralTemplate clone
#[test]
fn test_mistral_clone() {
    let template = MistralTemplate::new();
    let cloned = template.clone();
    let messages = vec![ChatMessage::user("test")];
    assert_eq!(
        template.format_conversation(&messages).unwrap(),
        cloned.format_conversation(&messages).unwrap()
    );
}

/// Test PhiTemplate clone
#[test]
fn test_phi_clone() {
    let template = PhiTemplate::new();
    let cloned = template.clone();
    let messages = vec![ChatMessage::user("test")];
    assert_eq!(
        template.format_conversation(&messages).unwrap(),
        cloned.format_conversation(&messages).unwrap()
    );
}

/// Test AlpacaTemplate clone
#[test]
fn test_alpaca_clone() {
    let template = AlpacaTemplate::new();
    let cloned = template.clone();
    let messages = vec![ChatMessage::user("test")];
    assert_eq!(
        template.format_conversation(&messages).unwrap(),
        cloned.format_conversation(&messages).unwrap()
    );
}

/// Test RawTemplate clone
#[test]
fn test_raw_clone() {
    let template = RawTemplate::new();
    let cloned = template.clone();
    let messages = vec![ChatMessage::user("test")];
    assert_eq!(
        template.format_conversation(&messages).unwrap(),
        cloned.format_conversation(&messages).unwrap()
    );
}

/// Test ChatMLTemplate with custom tokens
#[test]
fn test_chatml_with_custom_tokens() {
    let tokens = SpecialTokens {
        bos_token: Some("<bos>".to_string()),
        eos_token: Some("<eos>".to_string()),
        im_start_token: Some("<start>".to_string()),
        im_end_token: Some("<end>".to_string()),
        ..Default::default()
    };
    let template = ChatMLTemplate::with_tokens(tokens);
    // Check tokens are stored correctly
    assert_eq!(
        template.special_tokens().bos_token,
        Some("<bos>".to_string())
    );
}

/// Test TemplateFormat Debug
#[test]
fn test_template_format_debug() {
    let formats = [
        TemplateFormat::ChatML,
        TemplateFormat::Llama2,
        TemplateFormat::Mistral,
        TemplateFormat::Phi,
        TemplateFormat::Alpaca,
        TemplateFormat::Custom,
        TemplateFormat::Raw,
    ];
    for fmt in &formats {
        let debug = format!("{:?}", fmt);
        assert!(!debug.is_empty());
    }
}

/// Test TemplateFormat Copy
#[test]
fn test_template_format_copy() {
    let fmt = TemplateFormat::ChatML;
    let copied = fmt;
    assert_eq!(fmt, copied);
}

/// Test SpecialTokens Debug and Clone
#[test]
fn test_special_tokens_debug_clone() {
    let tokens = SpecialTokens {
        bos_token: Some("<s>".to_string()),
        ..Default::default()
    };
    let cloned = tokens.clone();
    assert_eq!(tokens.bos_token, cloned.bos_token);
    let debug = format!("{:?}", tokens);
    assert!(debug.contains("bos_token"));
}

/// Test ChatMessage Debug and Eq
#[test]
fn test_chat_message_debug_eq() {
    let msg1 = ChatMessage::user("hello");
    let msg2 = ChatMessage::user("hello");
    let msg3 = ChatMessage::user("world");

    assert_eq!(msg1, msg2);
    assert_ne!(msg1, msg3);

    let debug = format!("{:?}", msg1);
    assert!(debug.contains("user"));
}

/// Test HuggingFaceTemplate format_message
#[test]
fn test_hf_template_format_message() {
    let json = r#"{
            "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
            "bos_token": "<s>"
        }"#;
    let template = HuggingFaceTemplate::from_json(json).expect("parse failed");
    let result = template.format_message("user", "hello");
    assert!(result.is_ok());
}

/// Test HuggingFaceTemplate format and supports_system_prompt
#[test]
fn test_hf_template_accessors() {
    let json = r#"{
            "chat_template": "<|im_start|>{{ message.role }}",
            "bos_token": "<s>"
        }"#;
    let template = HuggingFaceTemplate::from_json(json).expect("parse failed");
    assert_eq!(template.format(), TemplateFormat::ChatML);
    assert!(template.supports_system_prompt());
}

/// Test HuggingFace template with Alpaca format detection
#[test]
fn test_hf_template_alpaca_detection() {
    let json = r####"{
            "chat_template": "### Instruction: {{ message.content }}",
            "bos_token": "<s>"
        }"####;
    let template = HuggingFaceTemplate::from_json(json).expect("parse failed");
    assert_eq!(template.format(), TemplateFormat::Alpaca);
}

/// Test create_template for Custom format
#[test]
fn test_create_template_custom() {
    let template = create_template(TemplateFormat::Custom);
    assert_eq!(template.format(), TemplateFormat::Raw); // Custom maps to Raw
}

/// Test ChatMLTemplate format_message
#[test]
fn test_chatml_format_message() {
    let template = ChatMLTemplate::new();
    let result = template.format_message("user", "hello");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "<|im_start|>user\nhello<|im_end|>\n");
}

/// Test Llama2 with system message merged into first user turn
#[test]
fn test_llama2_system_merged() {
    let template = Llama2Template::new();
    let messages = vec![
        ChatMessage::system("Be helpful"),
        ChatMessage::user("Hello"),
    ];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
    let output = result.unwrap();
    // System should be merged with first user message
    assert!(output.contains("<<SYS>>"));
    assert!(output.contains("Be helpful"));
    assert!(output.contains("[INST]"));
}

/// Test Llama2 multi-turn without system
#[test]
fn test_llama2_multi_turn_no_system() {
    let template = Llama2Template::new();
    let messages = vec![
        ChatMessage::user("First"),
        ChatMessage::assistant("Response 1"),
        ChatMessage::user("Second"),
    ];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(!output.contains("<<SYS>>"));
    assert!(output.contains("First"));
    assert!(output.contains("Response 1"));
    assert!(output.contains("Second"));
}
