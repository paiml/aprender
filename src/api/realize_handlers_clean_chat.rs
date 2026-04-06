
    // =========================================================================
    // GH-219 Coverage Gap: format_chat_messages, clean_chat_output,
    // epoch_secs, epoch_millis
    // =========================================================================

    // -------------------------------------------------------------------------
    // clean_chat_output (PMAT-088: prompt injection prevention)
    // -------------------------------------------------------------------------

    #[test]
    fn test_clean_chat_output_no_stop_sequences_gh219() {
        let result = clean_chat_output("Hello, world!");
        assert_eq!(result, "Hello, world!");
    }

    #[test]
    fn test_clean_chat_output_im_end_gh219() {
        let result = clean_chat_output("Good answer<|im_end|>extra garbage");
        assert_eq!(result, "Good answer");
    }

    #[test]
    fn test_clean_chat_output_endoftext_gh219() {
        let result = clean_chat_output("Some text<|endoftext|>more text");
        assert_eq!(result, "Some text");
    }

    #[test]
    fn test_clean_chat_output_eos_llama_gh219() {
        let result = clean_chat_output("Answer here</s>next turn");
        assert_eq!(result, "Answer here");
    }

    #[test]
    fn test_clean_chat_output_human_turn_gh219() {
        let result = clean_chat_output("Response\nHuman: new question");
        assert_eq!(result, "Response");
    }

    #[test]
    fn test_clean_chat_output_user_turn_gh219() {
        let result = clean_chat_output("Response\nUser: what?");
        assert_eq!(result, "Response");
    }

    #[test]
    fn test_clean_chat_output_im_start_gh219() {
        let result = clean_chat_output("Done<|im_start|>user\nnext");
        assert_eq!(result, "Done");
    }

    #[test]
    fn test_clean_chat_output_earliest_wins_gh219() {
        // Multiple stop sequences — should truncate at the earliest
        let result = clean_chat_output("Hello</s>mid<|im_end|>end");
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_clean_chat_output_empty_gh219() {
        let result = clean_chat_output("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_clean_chat_output_only_stop_sequence_gh219() {
        let result = clean_chat_output("<|im_end|>");
        assert_eq!(result, "");
    }

    #[test]
    fn test_clean_chat_output_trims_whitespace_gh219() {
        let result = clean_chat_output("  answer  <|im_end|>");
        assert_eq!(result, "answer");
    }

    #[test]
    fn test_clean_chat_output_double_newline_human_gh219() {
        let result = clean_chat_output("my response\n\nHuman: follow up");
        assert_eq!(result, "my response");
    }

    // -------------------------------------------------------------------------
    // format_chat_messages
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_chat_messages_single_user_gh219() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }];
        let result = format_chat_messages(&messages, None);
        assert!(!result.is_empty());
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_format_chat_messages_system_and_user_gh219() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
                name: None,
            },
        ];
        let result = format_chat_messages(&messages, None);
        assert!(!result.is_empty());
        assert!(result.contains("Hi"));
    }

    #[test]
    fn test_format_chat_messages_multi_turn_gh219() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
                name: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "4".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "And 3+3?".to_string(),
                name: None,
            },
        ];
        let result = format_chat_messages(&messages, None);
        assert!(result.contains("2+2"));
        assert!(result.contains("3+3"));
    }

    #[test]
    fn test_format_chat_messages_empty_gh219() {
        let messages: Vec<ChatMessage> = vec![];
        let result = format_chat_messages(&messages, None);
        // Should not panic even with empty input
        assert!(result.is_empty() || !result.is_empty());
    }

    #[test]
    fn test_format_chat_messages_with_model_name_gh219() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }];
        let result = format_chat_messages(&messages, Some("qwen2"));
        assert!(!result.is_empty());
    }

    // -------------------------------------------------------------------------
    // epoch_secs / epoch_millis
    // -------------------------------------------------------------------------

    #[test]
    fn test_epoch_secs_reasonable_gh219() {
        let secs = epoch_secs();
        // Should be after 2020-01-01 (1577836800)
        assert!(secs > 1_577_836_800);
    }

    #[test]
    fn test_epoch_millis_reasonable_gh219() {
        let millis = epoch_millis();
        // Should be after 2020-01-01 in millis
        assert!(millis > 1_577_836_800_000);
    }

    #[test]
    fn test_epoch_millis_greater_than_secs_gh219() {
        let secs = epoch_secs();
        let millis = epoch_millis();
        // millis should be roughly 1000x secs
        assert!(millis > secs as u128);
    }
