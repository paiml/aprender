
// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
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
            let output = result.expect("operation failed");
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
            let output = result.expect("operation failed");

            let pos1 = output.find(&msg1);
            let pos2 = output.find(&msg2);
            let pos3 = output.find(&msg3);

            prop_assert!(pos1.is_some());
            prop_assert!(pos2.is_some());
            prop_assert!(pos3.is_some());
            prop_assert!(pos1.expect("operation failed") < pos2.expect("operation failed"));
            prop_assert!(pos2.expect("operation failed") < pos3.expect("operation failed"));
        }

        /// Property: Serde roundtrip preserves ChatMessage
        #[test]
        fn prop_message_serde_roundtrip(
            role in "(system|user|assistant)",
            content in ".*"
        ) {
            let msg = ChatMessage::new(&role, &content);
            let json = serde_json::to_string(&msg).expect("invalid UTF-8");
            let restored: ChatMessage = serde_json::from_str(&json).expect("parse failed");
            prop_assert_eq!(msg, restored);
        }

        /// Property: Template format enum is exhaustive in create_template
        #[test]
        fn prop_all_formats_creatable(format_idx in 0usize..8) {
            let formats = [
                TemplateFormat::ChatML,
                TemplateFormat::Llama2,
                TemplateFormat::Zephyr,
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
            let output = result.expect("operation failed");
            prop_assert!(output.ends_with("<|im_start|>assistant\n"));
        }

        /// Property: LLaMA2 always starts with BOS token
        #[test]
        fn prop_llama2_bos_token(content in "[a-z]{1,50}") {
            let template = Llama2Template::new();
            let messages = vec![ChatMessage::user(&content)];
            let result = template.format_conversation(&messages);
            prop_assert!(result.is_ok());
            let output = result.expect("operation failed");
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
            let output = result.expect("operation failed");
            // Mistral doesn't support system prompts
            prop_assert!(!output.contains("<<SYS>>"));
            prop_assert!(!output.contains("<</SYS>>"));
        }
    }
}
