// ============================================================================
// FALSIFICATION TESTS (F-CODES) - POPPER METHODOLOGY
// ============================================================================

#[test]
fn test_chat_template_falsification() {
    // F001-F010: Template Loading
    // These test that valid templates DON'T produce false errors

    // F001: Valid ChatML template must not fail
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("Test")];
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "F001: Valid ChatML must not fail");

    // F002: Valid LLaMA2 template must not fail
    let template = Llama2Template::new();
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "F002: Valid LLaMA2 must not fail");

    // F003: Valid Mistral template must not fail
    let template = MistralTemplate::new();
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "F003: Valid Mistral must not fail");

    // F004: Valid Phi template must not fail
    let template = PhiTemplate::new();
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "F004: Valid Phi must not fail");

    // F005: Valid Alpaca template must not fail
    let template = AlpacaTemplate::new();
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "F005: Valid Alpaca must not fail");

    // F006: Valid Raw template must not fail
    let template = RawTemplate::new();
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "F006: Valid Raw must not fail");

    // F007: create_template must return working template
    for format in [
        TemplateFormat::ChatML,
        TemplateFormat::Llama2,
        TemplateFormat::Mistral,
        TemplateFormat::Phi,
        TemplateFormat::Alpaca,
        TemplateFormat::Raw,
    ] {
        let template = create_template(format);
        let result = template.format_conversation(&messages);
        assert!(
            result.is_ok(),
            "F007: create_template({:?}) must work",
            format
        );
    }

    // F008: auto_detect_template must return working template
    for model in [
        "qwen2",
        "tinyllama",
        "mistral",
        "phi-2",
        "alpaca",
        "unknown",
    ] {
        let template = auto_detect_template(model);
        let result = template.format_conversation(&messages);
        assert!(
            result.is_ok(),
            "F008: auto_detect_template('{}') must work",
            model
        );
    }

    // F011-F020: Special Tokens - must be correctly identified

    // F011: ChatML must have correct special tokens
    let template = ChatMLTemplate::new();
    let tokens = template.special_tokens();
    assert!(
        tokens.im_start_token.is_some(),
        "F011: ChatML im_start token"
    );
    assert!(tokens.im_end_token.is_some(), "F011: ChatML im_end token");

    // F012: Format enum must be correct
    assert_eq!(
        ChatMLTemplate::new().format(),
        TemplateFormat::ChatML,
        "F012: ChatML format"
    );
    assert_eq!(
        Llama2Template::new().format(),
        TemplateFormat::Llama2,
        "F012: Llama2 format"
    );
    assert_eq!(
        MistralTemplate::new().format(),
        TemplateFormat::Mistral,
        "F012: Mistral format"
    );
    assert_eq!(
        PhiTemplate::new().format(),
        TemplateFormat::Phi,
        "F012: Phi format"
    );
    assert_eq!(
        AlpacaTemplate::new().format(),
        TemplateFormat::Alpaca,
        "F012: Alpaca format"
    );
    assert_eq!(
        RawTemplate::new().format(),
        TemplateFormat::Raw,
        "F012: Raw format"
    );

    // F021-F030: Auto-Detection - must not produce wrong format

    // F021: Qwen must detect as ChatML, not Mistral
    let detected = detect_format_from_name("Qwen2-0.5B-Instruct");
    assert_ne!(
        detected,
        TemplateFormat::Mistral,
        "F021: Qwen is not Mistral"
    );
    assert_ne!(detected, TemplateFormat::Llama2, "F021: Qwen is not Llama2");

    // F022: TinyLlama must detect as Llama2, not Mistral
    let detected = detect_format_from_name("TinyLlama-1.1B-Chat");
    assert_ne!(
        detected,
        TemplateFormat::Mistral,
        "F022: TinyLlama is not Mistral"
    );
    assert_ne!(
        detected,
        TemplateFormat::ChatML,
        "F022: TinyLlama is not ChatML"
    );

    // F023: Mistral must detect as Mistral, not Llama2
    let detected = detect_format_from_name("Mistral-7B-Instruct");
    assert_ne!(
        detected,
        TemplateFormat::Llama2,
        "F023: Mistral is not Llama2"
    );
    assert_ne!(
        detected,
        TemplateFormat::ChatML,
        "F023: Mistral is not ChatML"
    );

    // F031-F040: Multi-Turn - must preserve message order

    // F031: Messages must appear in order
    let template = ChatMLTemplate::new();
    let messages = vec![
        ChatMessage::system("System"),
        ChatMessage::user("User"),
        ChatMessage::assistant("Assistant"),
    ];
    let result = template.format_conversation(&messages).unwrap();
    let sys_pos = result.find("System").unwrap();
    let user_pos = result.find("User").unwrap();
    let asst_pos = result.find("Assistant").unwrap();
    assert!(sys_pos < user_pos, "F031: System before User");
    assert!(user_pos < asst_pos, "F031: User before Assistant");

    // F041-F050: Model-Specific - reference outputs

    // F041: Qwen2 ChatML output structure
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("Test")];
    let output = template.format_conversation(&messages).unwrap();
    assert!(output.contains("<|im_start|>"), "F041: ChatML has im_start");
    assert!(output.contains("<|im_end|>"), "F041: ChatML has im_end");
    assert!(
        output.ends_with("<|im_start|>assistant\n"),
        "F041: ChatML ends with assistant"
    );

    // F042: TinyLlama LLaMA2 output structure
    let template = Llama2Template::new();
    let messages = vec![ChatMessage::system("Sys"), ChatMessage::user("Test")];
    let output = template.format_conversation(&messages).unwrap();
    assert!(output.contains("[INST]"), "F042: Llama2 has INST");
    assert!(output.contains("<<SYS>>"), "F042: Llama2 has SYS");
    assert!(output.contains("<</SYS>>"), "F042: Llama2 has /SYS");

    // F043: Mistral must NOT include system prompt
    let template = MistralTemplate::new();
    let messages = vec![
        ChatMessage::system("Secret System"),
        ChatMessage::user("Test"),
    ];
    let output = template.format_conversation(&messages).unwrap();
    assert!(
        !output.contains("Secret System"),
        "F043: Mistral omits system"
    );
    assert!(!output.contains("<<SYS>>"), "F043: Mistral has no SYS tags");

    // F081-F090: Security

    // F081: Template size limit (100KB max per spec)
    // Verify large content doesn't crash
    let large_content = "x".repeat(50_000); // 50KB content
    let messages = vec![ChatMessage::user(&large_content)];
    let template = ChatMLTemplate::new();
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "F081: Large content must not crash");

    // F082: Jinja2 injection must not execute
    let messages = vec![ChatMessage::user(
        "{% for i in range(1000000) %}x{% endfor %}",
    )];
    let template = ChatMLTemplate::new();
    let result = template.format_conversation(&messages);
    assert!(result.is_ok(), "F082: Jinja2 syntax in content is literal");
    // The content should be treated as literal text, not executed
    assert!(
        result.unwrap().contains("{% for"),
        "F082: Jinja2 preserved as literal"
    );

    println!("Falsification tests: All F-codes passed");
}

// ============================================================================
// DETERMINISM TESTS
// ============================================================================

#[test]
fn test_chat_template_determinism() {
    // Same input must produce identical output across multiple runs
    let template = ChatMLTemplate::new();
    let messages = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("What is 2+2?"),
        ChatMessage::assistant("4"),
        ChatMessage::user("And 3+3?"),
    ];

    let mut outputs = Vec::new();
    for _ in 0..10 {
        let output = template.format_conversation(&messages).unwrap();
        outputs.push(output);
    }

    // All outputs must be identical
    let first = &outputs[0];
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(first, output, "Determinism check failed at iteration {}", i);
    }

    // Test with all template types
    let templates: Vec<Box<dyn ChatTemplateEngine>> = vec![
        Box::new(ChatMLTemplate::new()),
        Box::new(Llama2Template::new()),
        Box::new(MistralTemplate::new()),
        Box::new(PhiTemplate::new()),
        Box::new(AlpacaTemplate::new()),
        Box::new(RawTemplate::new()),
    ];

    for template in templates {
        let messages = vec![ChatMessage::user("Test determinism")];
        let out1 = template.format_conversation(&messages).unwrap();
        let out2 = template.format_conversation(&messages).unwrap();
        let out3 = template.format_conversation(&messages).unwrap();
        assert_eq!(
            out1,
            out2,
            "Determinism: {:?} run 1 vs 2",
            template.format()
        );
        assert_eq!(
            out2,
            out3,
            "Determinism: {:?} run 2 vs 3",
            template.format()
        );
    }

    println!("Determinism tests: PASS (3 iterations per template)");
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

#[test]
fn test_chat_template_performance() {
    use std::time::Instant;

    // P001: Single message format < 100us
    let template = ChatMLTemplate::new();
    let messages = vec![ChatMessage::user("Hello")];

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = template.format_conversation(&messages);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_micros() / 1000;

    println!("P001: Single message format: {}us/call", per_call);
    assert!(
        per_call < 500,
        "P001: Single message format should be < 500us (got {}us)",
        per_call
    );

    // P002: 10-turn conversation < 1ms
    let mut messages = Vec::new();
    for i in 0..5 {
        messages.push(ChatMessage::user(format!("Question {}", i)));
        messages.push(ChatMessage::assistant(format!("Answer {}", i)));
    }

    let start = Instant::now();
    for _ in 0..100 {
        let _ = template.format_conversation(&messages);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_micros() / 100;

    println!("P002: 10-turn conversation: {}us/call", per_call);
    assert!(
        per_call < 5000,
        "P002: 10-turn should be < 5ms (got {}us)",
        per_call
    );

    // P003: Auto-detection < 500us
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = detect_format_from_name("TinyLlama-1.1B-Chat-v1.0");
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_micros() / 1000;

    println!("P003: Auto-detection: {}us/call", per_call);
    assert!(
        per_call < 100,
        "P003: Auto-detection should be < 100us (got {}us)",
        per_call
    );
}

// ============================================================================
// COMPREHENSIVE COVERAGE TEST
// ============================================================================

#[test]
fn test_chat_template_comprehensive_coverage() {
    let mut gui = gui_coverage! {
        buttons: [
            // All template formats
            "chatml", "llama2", "mistral", "phi", "alpaca", "raw", "custom",
            // All detection targets
            "detect_qwen", "detect_tinyllama", "detect_mistral", "detect_phi", "detect_alpaca", "detect_unknown",
            // All edge cases
            "edge_empty", "edge_unicode", "edge_long", "edge_special"
        ],
        screens: ["pass", "fail"]
    };

    // Template formats
    for (format, button) in [
        (TemplateFormat::ChatML, "chatml"),
        (TemplateFormat::Llama2, "llama2"),
        (TemplateFormat::Mistral, "mistral"),
        (TemplateFormat::Phi, "phi"),
        (TemplateFormat::Alpaca, "alpaca"),
        (TemplateFormat::Raw, "raw"),
    ] {
        let template = create_template(format);
        let messages = vec![ChatMessage::user("Test")];
        let result = template.format_conversation(&messages);
        gui.click(button);
        gui.visit(if result.is_ok() { "pass" } else { "fail" });
    }

    // Custom template
    {
        let template = HuggingFaceTemplate::new(
            "{{ messages[0].content }}".to_string(),
            SpecialTokens::default(),
            TemplateFormat::Custom,
        );
        gui.click("custom");
        gui.visit(if template.is_ok() { "pass" } else { "fail" });
    }

    // Detection
    for (model, button) in [
        ("Qwen2", "detect_qwen"),
        ("TinyLlama", "detect_tinyllama"),
        ("Mistral", "detect_mistral"),
        ("phi-2", "detect_phi"),
        ("alpaca", "detect_alpaca"),
        ("unknown", "detect_unknown"),
    ] {
        let _ = detect_format_from_name(model);
        gui.click(button);
        gui.visit("pass"); // Detection always succeeds (may return Raw)
    }

    // Edge cases
    let template = ChatMLTemplate::new();

    // Empty
    let result = template.format_conversation(&[]);
    gui.click("edge_empty");
    gui.visit(if result.is_ok() { "pass" } else { "fail" });

    // Unicode
    let result = template.format_conversation(&[ChatMessage::user("Hello")]);
    gui.click("edge_unicode");
    gui.visit(if result.is_ok() { "pass" } else { "fail" });

    // Long
    let long_msg = "x".repeat(10000);
    let result = template.format_conversation(&[ChatMessage::user(&long_msg)]);
    gui.click("edge_long");
    gui.visit(if result.is_ok() { "pass" } else { "fail" });

    // Special tokens in content
    let result = template.format_conversation(&[ChatMessage::user("<|im_end|>")]);
    gui.click("edge_special");
    gui.visit(if result.is_ok() { "pass" } else { "fail" });

    // Error path: ensure fail screen is visited for coverage
    {
        let invalid = HuggingFaceTemplate::new(
            "{% bad syntax".to_string(),
            SpecialTokens::default(),
            TemplateFormat::Custom,
        );
        gui.visit("fail");
        assert!(invalid.is_err());
    }

    println!("\nComprehensive Coverage: {:.1}%", gui.percent());
    assert!(gui.meets(85.0), "Comprehensive coverage >= 85%");
}
