use super::*;

/// Test Phi with system message
#[test]
fn test_phi_with_system() {
    let template = PhiTemplate::new();
    let messages = vec![
        ChatMessage::system("System instructions"),
        ChatMessage::user("Hello"),
    ];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("System instructions"));
    assert!(output.contains("Instruct: Hello"));
}

/// Test Alpaca with system message
#[test]
fn test_alpaca_with_system() {
    let template = AlpacaTemplate::new();
    let messages = vec![
        ChatMessage::system("System instructions"),
        ChatMessage::user("Hello"),
    ];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("System instructions"));
    assert!(output.contains("### Instruction:"));
}

/// Test detect_format_from_name for phi2 and phi3
#[test]
fn test_detect_phi_variants() {
    assert_eq!(detect_format_from_name("phi2"), TemplateFormat::Phi);
    assert_eq!(detect_format_from_name("phi3-mini"), TemplateFormat::Phi);
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Formatting never panics for arbitrary Unicode strings
        #[test]
        fn prop_format_never_panics(content in ".*") {
            let template = ChatMLTemplate::new();
            let messages = vec![ChatMessage::user(&content)];
            // Should not panic
            let _ = template.format_conversation(&messages);
        }

        /// Property: Output always contains the input content
        #[test]
        fn prop_output_contains_content(content in "[a-zA-Z0-9 ]{1,100}") {
            let template = ChatMLTemplate::new();
            let messages = vec![ChatMessage::user(&content)];
            let result = template.format_conversation(&messages);
            prop_assert!(result.is_ok());
            let output = result.unwrap();
            prop_assert!(output.contains(&content));
        }

        /// Property: Auto-detection is deterministic
        #[test]
        fn prop_detection_deterministic(name in "[a-zA-Z0-9_-]{1,50}") {
            let format1 = detect_format_from_name(&name);
            let format2 = detect_format_from_name(&name);
            prop_assert_eq!(format1, format2);
        }

        /// Property: Message order preserved in output
        #[test]
        fn prop_message_order_preserved(
            msg1 in "[a-z]{5,10}",
            msg2 in "[a-z]{5,10}",
            msg3 in "[a-z]{5,10}"
        ) {
            let template = ChatMLTemplate::new();
            let messages = vec![
                ChatMessage::user(&msg1),
                ChatMessage::assistant(&msg2),
                ChatMessage::user(&msg3),
            ];
            let result = template.format_conversation(&messages);
            prop_assert!(result.is_ok());
            let output = result.unwrap();

            let pos1 = output.find(&msg1);
            let pos2 = output.find(&msg2);
            let pos3 = output.find(&msg3);

            prop_assert!(pos1.is_some());
            prop_assert!(pos2.is_some());
            prop_assert!(pos3.is_some());
            prop_assert!(pos1.unwrap() < pos2.unwrap());
            prop_assert!(pos2.unwrap() < pos3.unwrap());
        }

        /// Property: Serde roundtrip preserves ChatMessage
        #[test]
        fn prop_message_serde_roundtrip(
            role in "(system|user|assistant)",
            content in ".*"
        ) {
            let msg = ChatMessage::new(&role, &content);
            let json = serde_json::to_string(&msg).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let restored: ChatMessage = serde_json::from_str(&json).map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(msg, restored);
        }

        /// Property: Template format enum is exhaustive in create_template
        #[test]
        fn prop_all_formats_creatable(format_idx in 0usize..7) {
            let formats = [
                TemplateFormat::ChatML,
                TemplateFormat::Llama2,
                TemplateFormat::Mistral,
                TemplateFormat::Phi,
                TemplateFormat::Alpaca,
                TemplateFormat::Custom,
                TemplateFormat::Raw,
            ];
            let format = formats[format_idx];
            let template = create_template(format);
            // Should not panic and should format
            let messages = vec![ChatMessage::user("test")];
            let result = template.format_conversation(&messages);
            prop_assert!(result.is_ok());
        }

        /// Property: Generation prompt is always appended for ChatML
        #[test]
        fn prop_chatml_generation_prompt(content in "[a-z]{1,50}") {
            let template = ChatMLTemplate::new();
            let messages = vec![ChatMessage::user(&content)];
            let result = template.format_conversation(&messages);
            prop_assert!(result.is_ok());
            let output = result.unwrap();
            prop_assert!(output.ends_with("<|im_start|>assistant\n"));
        }

        /// Property: LLaMA2 always starts with BOS token
        #[test]
        fn prop_llama2_bos_token(content in "[a-z]{1,50}") {
            let template = Llama2Template::new();
            let messages = vec![ChatMessage::user(&content)];
            let result = template.format_conversation(&messages);
            prop_assert!(result.is_ok());
            let output = result.unwrap();
            prop_assert!(output.starts_with("<s>"));
        }

        /// Property: Mistral never includes system prompt markers
        #[test]
        fn prop_mistral_no_system_markers(
            sys_content in "[a-z]{1,20}",
            user_content in "[a-z]{1,20}"
        ) {
            let template = MistralTemplate::new();
            let messages = vec![
                ChatMessage::system(&sys_content),
                ChatMessage::user(&user_content),
            ];
            let result = template.format_conversation(&messages);
            prop_assert!(result.is_ok());
            let output = result.unwrap();
            // Mistral doesn't support system prompts
            prop_assert!(!output.contains("<<SYS>>"));
            prop_assert!(!output.contains("<</SYS>>"));
        }
    }
}

// ============================================================================
// Additional Coverage Tests for 95% Target
// ============================================================================

#[test]
fn test_detect_format_from_tokens_chatml_im_start() {
    let tokens = SpecialTokens {
        im_start_token: Some("<|im_start|>".to_string()),
        ..Default::default()
    };
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::ChatML);
}

#[test]
fn test_detect_format_from_tokens_chatml_im_end() {
    let tokens = SpecialTokens {
        im_end_token: Some("<|im_end|>".to_string()),
        ..Default::default()
    };
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::ChatML);
}

#[test]
fn test_detect_format_from_tokens_llama2_inst_start() {
    let tokens = SpecialTokens {
        inst_start: Some("[INST]".to_string()),
        ..Default::default()
    };
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::Llama2);
}

#[test]
fn test_detect_format_from_tokens_llama2_inst_end() {
    let tokens = SpecialTokens {
        inst_end: Some("[/INST]".to_string()),
        ..Default::default()
    };
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::Llama2);
}

#[test]
fn test_detect_format_from_tokens_raw_empty() {
    let tokens = SpecialTokens::default();
    assert_eq!(detect_format_from_tokens(&tokens), TemplateFormat::Raw);
}

#[test]
fn test_auto_detect_template_qwen() {
    let template = auto_detect_template("Qwen2-0.5B-Instruct");
    assert_eq!(template.format(), TemplateFormat::ChatML);
}

#[test]
fn test_auto_detect_template_unknown() {
    let template = auto_detect_template("unknown-model-xyz");
    assert_eq!(template.format(), TemplateFormat::Raw);
}

#[test]
fn test_mistral_format_message_user() {
    let template = MistralTemplate::new();
    let result = template.format_message("user", "Hello!").unwrap();
    assert!(result.contains("[INST]"));
    assert!(result.contains("[/INST]"));
}

#[test]
fn test_mistral_format_message_assistant() {
    let template = MistralTemplate::new();
    let result = template.format_message("assistant", "Hi!").unwrap();
    assert!(result.contains("</s>"));
}

#[test]
fn test_mistral_format_message_system_coverage() {
    let template = MistralTemplate::new();
    let result = template
        .format_message("system", "You are helpful")
        .unwrap();
    assert!(result.contains("\n\n"));
}

#[test]
fn test_mistral_format_message_unknown_role_coverage() {
    let template = MistralTemplate::new();
    let result = template.format_message("custom", "content").unwrap();
    assert_eq!(result, "content");
}

#[test]
fn test_phi_format_message_user() {
    let template = PhiTemplate::new();
    let result = template.format_message("user", "Hello!").unwrap();
    assert!(result.contains("Instruct:"));
}

#[test]
fn test_phi_format_message_assistant() {
    let template = PhiTemplate::new();
    let result = template.format_message("assistant", "Hi!").unwrap();
    assert!(result.contains("Output:"));
}

#[test]
fn test_phi_format_message_system() {
    let template = PhiTemplate::new();
    let result = template.format_message("system", "System prompt").unwrap();
    assert!(result.contains("System prompt"));
}

#[test]
fn test_phi_format_message_unknown_role() {
    let template = PhiTemplate::new();
    let result = template.format_message("custom", "content").unwrap();
    assert_eq!(result, "content");
}

#[test]
fn test_llama2_format_message_user() {
    let template = Llama2Template::new();
    let result = template.format_message("user", "Hello!").unwrap();
    assert!(result.contains("[INST]"));
    assert!(result.contains("[/INST]"));
}

#[test]
fn test_llama2_format_message_assistant() {
    let template = Llama2Template::new();
    let result = template.format_message("assistant", "Hi!").unwrap();
    assert!(result.contains("</s>"));
}

#[test]
fn test_llama2_format_message_system() {
    let template = Llama2Template::new();
    let result = template.format_message("system", "System").unwrap();
    assert!(result.contains("<<SYS>>"));
    assert!(result.contains("<</SYS>>"));
}

#[test]
fn test_llama2_format_message_unknown_role_coverage() {
    let template = Llama2Template::new();
    let result = template.format_message("custom", "content").unwrap();
    assert_eq!(result, "content");
}

#[test]
fn test_alpaca_format_message_user() {
    let template = AlpacaTemplate::new();
    let result = template.format_message("user", "Hello!").unwrap();
    assert!(result.contains("### Instruction:"));
}

#[test]
fn test_alpaca_format_message_assistant() {
    let template = AlpacaTemplate::new();
    let result = template.format_message("assistant", "Response").unwrap();
    assert!(result.contains("### Response:"));
}

#[test]
fn test_alpaca_format_message_system() {
    let template = AlpacaTemplate::new();
    let result = template.format_message("system", "System").unwrap();
    assert!(result.contains("System"));
}

#[test]
fn test_alpaca_format_message_unknown_role() {
    let template = AlpacaTemplate::new();
    let result = template.format_message("custom", "content").unwrap();
    assert_eq!(result, "content");
}

#[test]
fn test_chatml_format_message_non_user() {
    let template = ChatMLTemplate::new();
    // Non-user role should not sanitize content
    let result = template
        .format_message("assistant", "<|im_start|>test")
        .unwrap();
    // Assistant messages are not sanitized
    assert!(result.contains("<|im_start|>test"));
}

#[test]
fn test_raw_template_format_message() {
    let template = RawTemplate::new();
    let result = template.format_message("user", "Hello").unwrap();
    assert_eq!(result, "Hello");
}

#[test]
fn test_raw_template_format_conversation() {
    let template = RawTemplate::new();
    let messages = vec![ChatMessage::user("Hello"), ChatMessage::assistant("Hi")];
    let result = template.format_conversation(&messages).unwrap();
    assert_eq!(result, "HelloHi");
}

#[test]
fn test_huggingface_template_debug() {
    let json = r#"{
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        "bos_token": "<s>"
    }"#;
    let template = HuggingFaceTemplate::from_json(json).unwrap();
    let debug = format!("{:?}", template);
    assert!(debug.contains("HuggingFaceTemplate"));
}

#[test]
fn test_huggingface_template_format_message() {
    let json = r#"{
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        "bos_token": "<s>",
        "eos_token": "</s>"
    }"#;
    let template = HuggingFaceTemplate::from_json(json).unwrap();
    let result = template.format_message("user", "Hello").unwrap();
    assert!(result.contains("Hello"));
}

#[test]
fn test_huggingface_template_detect_format_chatml() {
    let json = r#"{
        "chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}{{ m.content }}<|im_end|>{% endfor %}",
        "bos_token": "<s>"
    }"#;
    let template = HuggingFaceTemplate::from_json(json).unwrap();
    assert_eq!(template.format(), TemplateFormat::ChatML);
}

#[test]
fn test_huggingface_template_detect_format_llama2() {
    let json = r#"{
        "chat_template": "{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}",
        "bos_token": "<s>"
    }"#;
    let template = HuggingFaceTemplate::from_json(json).unwrap();
    assert_eq!(template.format(), TemplateFormat::Llama2);
}

#[test]
fn test_huggingface_template_detect_format_alpaca() {
    // Use string concatenation to avoid raw string issues with ### pattern
    let chat_template = String::new() + "##" + "# Instruction:{{ content }}" + "##" + "# Response:";
    let json = format!(
        r#"{{
        "chat_template": "{}",
        "bos_token": "<s>"
    }}"#,
        chat_template
    );
    let template = HuggingFaceTemplate::from_json(&json).unwrap();
    assert_eq!(template.format(), TemplateFormat::Alpaca);
}

#[test]
fn test_huggingface_template_detect_format_custom() {
    let json = r#"{
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        "bos_token": "<s>"
    }"#;
    let template = HuggingFaceTemplate::from_json(json).unwrap();
    assert_eq!(template.format(), TemplateFormat::Custom);
}
