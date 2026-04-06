
    // ========================================================================
    // ChatMessage Tests
    // ========================================================================

    #[test]
    fn test_chat_message_new() {
        let msg = ChatMessage::new("user", "Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("sys");
        assert_eq!(sys.role, "system");

        let user = ChatMessage::user("usr");
        assert_eq!(user.role, "user");

        let asst = ChatMessage::assistant("asst");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_chat_message_serde_roundtrip() {
        let msg = ChatMessage::user("Hello, world!");
        let json = serde_json::to_string(&msg).expect("serialize");
        let restored: ChatMessage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(msg, restored);
    }

    // ========================================================================
    // Format Detection Tests
    // ========================================================================

    #[test]
    fn test_detect_chatml() {
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

    #[test]
    fn test_detect_zephyr() {
        assert_eq!(
            detect_format_from_name("TinyLlama-1.1B-Chat"),
            TemplateFormat::Zephyr
        );
        assert_eq!(
            detect_format_from_name("zephyr-7b-beta"),
            TemplateFormat::Zephyr
        );
        assert_eq!(
            detect_format_from_name("stablelm-3b-4e1t"),
            TemplateFormat::Zephyr
        );
    }

    #[test]
    fn test_detect_llama2() {
        assert_eq!(
            detect_format_from_name("vicuna-7b-v1.5"),
            TemplateFormat::Llama2
        );
        assert_eq!(
            detect_format_from_name("Llama-2-7B-Chat"),
            TemplateFormat::Llama2
        );
    }

    #[test]
    fn test_detect_mistral() {
        assert_eq!(
            detect_format_from_name("Mistral-7B-Instruct"),
            TemplateFormat::Mistral
        );
        assert_eq!(
            detect_format_from_name("Mixtral-8x7B"),
            TemplateFormat::Mistral
        );
    }

    #[test]
    fn test_detect_phi() {
        assert_eq!(detect_format_from_name("phi-2"), TemplateFormat::Phi);
        assert_eq!(detect_format_from_name("phi-3-mini"), TemplateFormat::Phi);
    }

    #[test]
    fn test_detect_alpaca() {
        assert_eq!(detect_format_from_name("alpaca-7b"), TemplateFormat::Alpaca);
    }

    #[test]
    fn test_detect_raw_fallback() {
        assert_eq!(
            detect_format_from_name("unknown-model"),
            TemplateFormat::Raw
        );
    }

    #[test]
    fn test_detection_deterministic() {
        let name = "TinyLlama-1.1B-Chat";
        let format1 = detect_format_from_name(name);
        let format2 = detect_format_from_name(name);
        assert_eq!(format1, format2);
    }

    #[test]
    fn test_detection_case_insensitive() {
        assert_eq!(
            detect_format_from_name("TINYLLAMA-1.1B-CHAT"),
            TemplateFormat::Zephyr
        );
        assert_eq!(
            detect_format_from_name("qwen2-0.5b-instruct"),
            TemplateFormat::ChatML
        );
    }

    // ========================================================================
    // ChatML Template Tests
    // ========================================================================

    #[test]
    fn test_chatml_format_message() {
        let template = ChatMLTemplate::new();
        let result = template
            .format_message("user", "Hello!")
            .expect("operation failed");
        assert_eq!(result, "<|im_start|>user\nHello!<|im_end|>\n");
    }

    #[test]
    fn test_chatml_format_conversation() {
        let template = ChatMLTemplate::new();
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        assert!(output.contains("<|im_start|>system"));
        assert!(output.contains("You are helpful."));
        assert!(output.contains("<|im_start|>user"));
        assert!(output.contains("Hello!"));
        assert!(output.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chatml_special_tokens() {
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

    // ========================================================================
    // LLaMA2 Template Tests
    // ========================================================================

    #[test]
    fn test_llama2_format_conversation() {
        let template = Llama2Template::new();
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        assert!(output.starts_with("<s>"));
        assert!(output.contains("[INST]"));
        assert!(output.contains("<<SYS>>"));
        assert!(output.contains("You are helpful."));
        assert!(output.contains("<</SYS>>"));
        assert!(output.contains("Hello!"));
        assert!(output.contains("[/INST]"));
    }

    #[test]
    fn test_llama2_multi_turn() {
        let template = Llama2Template::new();
        let messages = vec![
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        assert!(output.contains("Hello!"));
        assert!(output.contains("Hi there!"));
        assert!(output.contains("How are you?"));
        assert!(output.contains("</s>"));
    }

    // ========================================================================
    // Zephyr Template Tests (TinyLlama, Zephyr, StableLM)
    // ========================================================================

    #[test]
    fn test_zephyr_format_conversation() {
        let template = ZephyrTemplate::new();
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        // Zephyr format: <|role|>\ncontent</s>\n
        assert!(output.contains("<|system|>\n"));
        assert!(output.contains("You are helpful."));
        assert!(output.contains("</s>\n"));
        assert!(output.contains("<|user|>\n"));
        assert!(output.contains("Hello!"));
        assert!(output.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_zephyr_multi_turn() {
        let template = ZephyrTemplate::new();
        let messages = vec![
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        assert!(output.contains("<|user|>\nHello!</s>\n"));
        assert!(output.contains("<|assistant|>\nHi there!</s>\n"));
        assert!(output.contains("<|user|>\nHow are you?</s>\n"));
        assert!(output.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_zephyr_supports_system() {
        let template = ZephyrTemplate::new();
        assert!(template.supports_system_prompt());
    }

    // ========================================================================
    // Mistral Template Tests
    // ========================================================================

    #[test]
    fn test_mistral_no_system_prompt() {
        let template = MistralTemplate::new();
        assert!(!template.supports_system_prompt());

        let messages = vec![
            ChatMessage::system("System prompt"),
            ChatMessage::user("Hello!"),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        // System prompt should be ignored
        assert!(!output.contains("System prompt"));
        assert!(output.contains("[INST]"));
        assert!(output.contains("Hello!"));
    }

    // ========================================================================
    // Phi Template Tests
    // ========================================================================

    #[test]
    fn test_phi_format() {
        let template = PhiTemplate::new();
        let messages = vec![ChatMessage::user("Hello!")];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        assert!(output.contains("Instruct: Hello!"));
        assert!(output.ends_with("Output:"));
    }

    // ========================================================================
    // Alpaca Template Tests
    // ========================================================================

    #[test]
    fn test_alpaca_format() {
        let template = AlpacaTemplate::new();
        let messages = vec![ChatMessage::user("Hello!")];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        assert!(output.contains("### Instruction:"));
        assert!(output.contains("Hello!"));
        assert!(output.ends_with("### Response:\n"));
    }

    // ========================================================================
    // Raw Template Tests
    // ========================================================================

    #[test]
    fn test_raw_format() {
        let template = RawTemplate::new();
        let messages = vec![
            ChatMessage::system("System"),
            ChatMessage::user("User"),
            ChatMessage::assistant("Assistant"),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        assert_eq!(output, "SystemUserAssistant");
    }

    // ========================================================================
    // format_messages Tests
    // ========================================================================

    #[test]
    fn test_format_messages_with_model() {
        let messages = vec![ChatMessage::user("Hello!")];

        let output = format_messages(&messages, Some("Qwen2-0.5B")).expect("operation failed");
        assert!(output.contains("<|im_start|>"));

        // TinyLlama uses Zephyr format, NOT Llama2
        let output = format_messages(&messages, Some("TinyLlama")).expect("operation failed");
        assert!(output.contains("<|user|>"));
        assert!(output.contains("<|assistant|>"));
    }

    #[test]
    fn test_format_messages_without_model() {
        let messages = vec![ChatMessage::user("Hello!")];
        let output = format_messages(&messages, None).expect("operation failed");
        assert_eq!(output, "Hello!");
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_empty_conversation() {
        let template = ChatMLTemplate::new();
        let messages: Vec<ChatMessage> = vec![];
        let result = template.format_conversation(&messages);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unicode_preserved() {
        let template = ChatMLTemplate::new();
        let messages = vec![ChatMessage::user("Hello! 你好 مرحبا 🎉")];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");

        assert!(output.contains("你好"));
        assert!(output.contains("مرحبا"));
        assert!(output.contains("🎉"));
    }

    #[test]
    fn test_long_content() {
        let template = ChatMLTemplate::new();
        let long_content = "x".repeat(10_000);
        let messages = vec![ChatMessage::user(&long_content)];
        let result = template.format_conversation(&messages);
        assert!(result.is_ok());
    }

    #[test]
    fn test_whitespace_preserved() {
        let template = ChatMLTemplate::new();
        let messages = vec![ChatMessage::user("  content with spaces  ")];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");
        assert!(output.contains("  content with spaces  "));
    }

    #[test]
    fn test_multiline_content() {
        let template = ChatMLTemplate::new();
        let multiline = "Line 1\nLine 2\nLine 3";
        let messages = vec![ChatMessage::user(multiline)];
        let output = template
            .format_conversation(&messages)
            .expect("operation failed");
        assert!(output.contains("Line 1\nLine 2\nLine 3"));
    }

    #[test]
    fn test_custom_role() {
        let template = ChatMLTemplate::new();
        let messages = vec![
            ChatMessage::new("tool", "Function result: 42"),
            ChatMessage::user("What was the result?"),
        ];
        let result = template.format_conversation(&messages);
        assert!(result.is_ok());
        let output = result.expect("operation failed");
        assert!(output.contains("tool"));
        assert!(output.contains("Function result: 42"));
    }

    // ========================================================================
    // Security Tests
    // ========================================================================

    #[test]
    fn test_template_injection_prevention() {
        let template = ChatMLTemplate::new();
        let messages = vec![ChatMessage::user("{% for i in range(10) %}X{% endfor %}")];
        let result = template.format_conversation(&messages);
        assert!(result.is_ok());
        // Jinja syntax should appear as literal content
        let output = result.expect("operation failed");
        assert!(output.contains("{% for i in range(10) %}"));
    }
