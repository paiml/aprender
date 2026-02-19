//\! Chat Template Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

pub(crate) use super::*;

// ========================================================================
// Section 1: Template Loading & Parsing (CTL-01 to CTL-10)
// ========================================================================

/// CTL-01: tokenizer_config.json parsed correctly
#[test]
fn ctl_01_tokenizer_config_parsed() {
    let json = r#"{
            "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
            "bos_token": "<s>",
            "eos_token": "</s>"
        }"#;

    let template = HuggingFaceTemplate::from_json(json);
    assert!(template.is_ok(), "Failed to parse valid tokenizer config");
}

/// CTL-02: JSON parsing rejects invalid JSON
#[test]
fn ctl_02_invalid_json_rejected() {
    let json = "{ invalid json }";
    let result = HuggingFaceTemplate::from_json(json);
    assert!(result.is_err(), "Should reject invalid JSON");
}

/// CTL-03: chat_template field extracted
#[test]
fn ctl_03_chat_template_extracted() {
    let json = r#"{
            "chat_template": "test template",
            "bos_token": "<s>"
        }"#;

    let template = HuggingFaceTemplate::from_json(json).expect("Parse failed");
    assert!(!template.template_str.is_empty());
}

/// CTL-04: Missing chat_template produces error
#[test]
fn ctl_04_missing_template_error() {
    let json = r#"{"bos_token": "<s>"}"#;
    let result = HuggingFaceTemplate::from_json(json);
    assert!(result.is_err(), "Should error on missing chat_template");
}

/// CTL-05: Special tokens extracted
#[test]
fn ctl_05_special_tokens_extracted() {
    let json = r#"{
            "chat_template": "test",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>"
        }"#;

    let template = HuggingFaceTemplate::from_json(json).expect("Parse failed");
    assert_eq!(template.special_tokens.bos_token, Some("<s>".to_string()));
    assert_eq!(template.special_tokens.eos_token, Some("</s>".to_string()));
}

// ========================================================================
// Section 2: Special Token Handling (CTS-01 to CTS-10)
// ========================================================================

/// CTS-01: BOS token identified in ChatML
#[test]
fn cts_01_bos_token_chatml() {
    let template = ChatMLTemplate::new();
    assert!(template.special_tokens().bos_token.is_some());
}

/// CTS-02: EOS token identified in ChatML
#[test]
fn cts_02_eos_token_chatml() {
    let template = ChatMLTemplate::new();
    assert!(template.special_tokens().eos_token.is_some());
}

/// CTS-03: ChatML tokens recognized
#[test]
fn cts_03_chatml_tokens() {
    let template = ChatMLTemplate::new();
    assert_eq!(
        template.special_tokens().im_start_token,
        Some("<|im_start|>".to_string())
    );
    assert_eq!(
        template.special_tokens().im_end_token,
        Some("<|im_end|>".to_string())
    );
}

/// CTS-04: INST tokens recognized in LLaMA2
#[test]
fn cts_04_inst_tokens_llama2() {
    let template = Llama2Template::new();
    assert_eq!(
        template.special_tokens().inst_start,
        Some("[INST]".to_string())
    );
    assert_eq!(
        template.special_tokens().inst_end,
        Some("[/INST]".to_string())
    );
}

/// CTS-05: System tokens in LLaMA2
#[test]
fn cts_05_system_tokens_llama2() {
    let template = Llama2Template::new();
    assert_eq!(
        template.special_tokens().sys_start,
        Some("<<SYS>>".to_string())
    );
    assert_eq!(
        template.special_tokens().sys_end,
        Some("<</SYS>>".to_string())
    );
}

// ========================================================================
// Section 3: Format Auto-Detection (CTA-01 to CTA-10)
// ========================================================================

/// CTA-01: ChatML detected from model name
#[test]
fn cta_01_chatml_detected() {
    assert_eq!(
        detect_format_from_name("Qwen2-0.5B-Instruct"),
        TemplateFormat::ChatML
    );
    assert_eq!(
        detect_format_from_name("OpenHermes-2.5"),
        TemplateFormat::ChatML
    );
    assert_eq!(
        detect_format_from_name("Yi-6B-Chat"),
        TemplateFormat::ChatML
    );
}

/// CTA-02: LLaMA2 detected from model name
#[test]
fn cta_02_llama2_detected() {
    assert_eq!(
        detect_format_from_name("TinyLlama-1.1B-Chat"),
        TemplateFormat::Llama2
    );
    assert_eq!(
        detect_format_from_name("vicuna-7b-v1.5"),
        TemplateFormat::Llama2
    );
    assert_eq!(
        detect_format_from_name("Llama-2-7B-Chat"),
        TemplateFormat::Llama2
    );
}

/// CTA-03: Mistral detected from model name
#[test]
fn cta_03_mistral_detected() {
    assert_eq!(
        detect_format_from_name("Mistral-7B-Instruct"),
        TemplateFormat::Mistral
    );
    assert_eq!(
        detect_format_from_name("Mixtral-8x7B"),
        TemplateFormat::Mistral
    );
}

/// CTA-04: Phi detected from model name
#[test]
fn cta_04_phi_detected() {
    assert_eq!(detect_format_from_name("phi-2"), TemplateFormat::Phi);
    assert_eq!(detect_format_from_name("phi-3-mini"), TemplateFormat::Phi);
}

/// CTA-05: Alpaca detected from model name
#[test]
fn cta_05_alpaca_detected() {
    assert_eq!(detect_format_from_name("alpaca-7b"), TemplateFormat::Alpaca);
}

/// CTA-07: Raw fallback for unknown models
#[test]
fn cta_07_raw_fallback() {
    assert_eq!(
        detect_format_from_name("unknown-model"),
        TemplateFormat::Raw
    );
}

/// CTA-08: Detection is deterministic
#[test]
fn cta_08_detection_deterministic() {
    let name = "TinyLlama-1.1B-Chat";
    let format1 = detect_format_from_name(name);
    let format2 = detect_format_from_name(name);
    assert_eq!(format1, format2);
}

// ========================================================================
// Section 4: Multi-Turn Conversation (CTM-01 to CTM-10)
// ========================================================================

/// CTM-01: System prompt positioned first in ChatML
#[test]
fn ctm_01_system_first_chatml() {
    let template = ChatMLTemplate::new();
    let messages = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Hello!"),
    ];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");
    let sys_pos = output.find("system").expect("system not found");
    let user_pos = output.find("user").expect("user not found");
    assert!(sys_pos < user_pos, "System should come before user");
}

/// CTM-02: User/assistant alternation preserved
#[test]
fn ctm_02_alternation_preserved() {
    let template = ChatMLTemplate::new();
    let messages = vec![
        ChatMessage::user("Hi"),
        ChatMessage::assistant("Hello!"),
        ChatMessage::user("How are you?"),
    ];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");

    // Check order
    let pos1 = output.find("Hi").expect("Hi not found");
    let pos2 = output.find("Hello!").expect("Hello not found");
    let pos3 = output.find("How are you?").expect("How are you not found");

    assert!(pos1 < pos2 && pos2 < pos3, "Messages out of order");
}

/// CTM-04: Generation prompt appended
#[test]
fn ctm_04_generation_prompt_appended() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("Hello!")];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");
    assert!(
        output.ends_with("<|im_start|>assistant\n"),
        "Should end with assistant prompt"
    );
}

/// CTM-05: No system prompt handled
#[test]
fn ctm_05_no_system_handled() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("Hello!")];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "Should handle conversation without system");
}

/// CTM-07: Empty message handled
#[test]
fn ctm_07_empty_message_handled() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("")];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "Should handle empty content");
}

// ========================================================================
// Section 5: Model-Specific Validation (CTX-01 to CTX-10)
// ========================================================================

/// CTX-01: Qwen2 ChatML format correct
#[test]
fn ctx_01_qwen2_chatml_format() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("What is 2+2?")];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");

    assert!(output.contains("<|im_start|>user"));
    assert!(output.contains("What is 2+2?"));
    assert!(output.contains("<|im_end|>"));
}

/// CTX-02: TinyLlama LLaMA2 format correct
#[test]
fn ctx_02_tinyllama_llama2_format() {
    let template = Llama2Template::new();
    let messages = vec![ChatMessage::user("Hello!")];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");

    assert!(output.contains("<s>"));
    assert!(output.contains("[INST]"));
    assert!(output.contains("Hello!"));
    assert!(output.contains("[/INST]"));
}

/// CTX-03: Mistral format omits system
#[test]
fn ctx_03_mistral_no_system() {
    let template = MistralTemplate::new();
    assert!(!template.supports_system_prompt());

    let messages = vec![
        ChatMessage::system("System prompt"),
        ChatMessage::user("Hello!"),
    ];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");

    // System prompt should be ignored
    assert!(!output.contains("System prompt"));
}

/// CTX-04: Phi format correct
#[test]
fn ctx_04_phi_format() {
    let template = PhiTemplate::new();
    let messages = vec![ChatMessage::user("Hello!")];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");

    assert!(output.contains("Instruct: Hello!"));
    assert!(output.contains("Output:"));
}

/// CTX-05: Alpaca format correct
#[test]
fn ctx_05_alpaca_format() {
    let template = AlpacaTemplate::new();
    let messages = vec![ChatMessage::user("Hello!")];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");

    assert!(output.contains("### Instruction:"));
    assert!(output.contains("Hello!"));
    assert!(output.contains("### Response:"));
}

// ========================================================================
// Section 6: Edge Cases (CTE-01 to CTE-10)
// ========================================================================

/// CTE-01: Empty conversation handled
#[test]
fn cte_01_empty_conversation() {
    let template = ChatMLTemplate::new();
    let messages: Vec<ChatMessage> = vec![];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "Should handle empty conversation");
}

/// CTE-02: Unicode/emoji preserved
#[test]
fn cte_02_unicode_preserved() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("Hello! 你好 مرحبا 🎉")];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");

    assert!(output.contains("你好"));
    assert!(output.contains("مرحبا"));
    assert!(output.contains("🎉"));
}

/// CTE-03: Long content handled
#[test]
fn cte_03_long_content() {
    let template = ChatMLTemplate::new();
    let long_content = "x".repeat(10_000);
    let messages = vec![ChatMessage::user(&long_content)];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "Should handle long content");
}

/// CTE-07: Whitespace preserved
#[test]
fn cte_07_whitespace_preserved() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("  content with spaces  ")];
    let output = template
        .format_conversation(&messages)
        .expect("Format failed");
    assert!(output.contains("  content with spaces  "));
}

/// CTE-09: Nested quotes handled
#[test]
fn cte_09_nested_quotes() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user(r#"He said "hello""#)];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "Should handle nested quotes");
}

// ========================================================================
// Section 7: Performance (CTP-01 to CTP-10) - Basic checks
// ========================================================================

/// CTP-01: Template application under 100ms (sanity check)
#[test]
fn ctp_01_format_performance() {
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("Hello!")];

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = template.format_conversation(&messages);
    }
    let elapsed = start.elapsed();

    // 1000 iterations should complete in well under 1 second
    assert!(
        elapsed.as_millis() < 1000,
        "Formatting too slow: {:?}",
        elapsed
    );
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
#[path = "tests_part_04.rs"]
mod tests_part_04;
#[path = "tests_part_05.rs"]
mod tests_part_05;
#[path = "tests_part_06.rs"]
mod tests_part_06;
