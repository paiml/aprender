//! Proptest falsification for chat template semantics contract
//!
//! Contract: contracts/chat-template-semantics-v1.yaml
//! Tests: FALSIFY-CT-001-prop through FALSIFY-CT-UTF8-prop
//!
//! Property-based testing for invariants that hold for ALL valid inputs.

use super::*;
use proptest::prelude::*;

proptest! {
    /// FALSIFY-CT-001-prop: Balanced delimiters in ChatML output.
    ///
    /// For ChatML format with arbitrary user content, the generation prompt
    /// appends one unpaired `<|im_start|>` for the assistant turn.
    /// Therefore: count(im_start) == count(im_end) + 1
    #[test]
    fn falsify_ct_001_prop_balanced_delimiters(content in ".*") {
        let template = ChatMLTemplate::new();
        let messages = vec![ChatMessage::user(content)];
        let output = template
            .format_conversation(&messages)
            .expect("format_conversation must not fail");

        let im_start_count = output.matches("<|im_start|>").count();
        let im_end_count = output.matches("<|im_end|>").count();

        prop_assert_eq!(
            im_start_count,
            im_end_count + 1,
            "im_start={} must equal im_end={} + 1 in output: {:?}",
            im_start_count,
            im_end_count,
            output
        );
    }

    /// FALSIFY-CT-002-prop: Sanitized output contains zero injection patterns.
    ///
    /// For arbitrary string input, `sanitize_user_content(input)` must return
    /// a string where `contains_injection_patterns()` is false.
    #[test]
    fn falsify_ct_002_prop_sanitize_removes_injections(input in ".*") {
        let sanitized = sanitize_user_content(&input);
        prop_assert!(
            !contains_injection_patterns(&sanitized),
            "Sanitized output still contains injection patterns: {:?}",
            sanitized
        );
    }

    /// FALSIFY-CT-003-prop: System content appears before first user content.
    ///
    /// For conversations with system + user messages, the system content
    /// appears before user content in the output for formats that support
    /// system prompts (ChatML, Llama2).
    ///
    /// Uses prefix markers to avoid false positives where short content
    /// matches inside template role labels (e.g., "r" matching in "user").
    #[test]
    fn falsify_ct_003_prop_system_before_user_chatml(
        sys_suffix in "[a-zA-Z0-9]{3,30}",
        user_suffix in "[a-zA-Z0-9]{3,30}"
    ) {
        // Prefix with unique markers so short strings don't match role labels
        let sys_content = format!("SYS_{sys_suffix}");
        let user_content = format!("USR_{user_suffix}");

        let template = ChatMLTemplate::new();
        let messages = vec![
            ChatMessage::system(sys_content.clone()),
            ChatMessage::user(user_content.clone()),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("format_conversation must not fail");

        let sys_pos = output.find(&sys_content);
        let user_pos = output.find(&user_content);

        prop_assert!(sys_pos.is_some(), "System content not found in output");
        prop_assert!(user_pos.is_some(), "User content not found in output");
        prop_assert!(
            sys_pos.expect("sys_pos checked above") < user_pos.expect("user_pos checked above"),
            "System content must appear before user content. sys_pos={:?}, user_pos={:?}",
            sys_pos,
            user_pos
        );
    }

    /// FALSIFY-CT-005-prop: Multi-turn message ordering preserved.
    ///
    /// For 3 messages with different content, their content appears in the
    /// same order in the formatted output.
    #[test]
    fn falsify_ct_005_prop_multi_turn_ordering(
        msg1 in "[a-zA-Z]{5,15}",
        msg2 in "[a-zA-Z]{5,15}",
        msg3 in "[a-zA-Z]{5,15}"
    ) {
        let template = ChatMLTemplate::new();
        let messages = vec![
            ChatMessage::user(msg1.clone()),
            ChatMessage::assistant(msg2.clone()),
            ChatMessage::user(msg3.clone()),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("format_conversation must not fail");

        let pos1 = output.find(&msg1);
        let pos2 = output.find(&msg2);
        let pos3 = output.find(&msg3);

        prop_assert!(pos1.is_some(), "msg1 not found in output");
        prop_assert!(pos2.is_some(), "msg2 not found in output");
        prop_assert!(pos3.is_some(), "msg3 not found in output");
        prop_assert!(
            pos1.expect("pos1 checked") < pos2.expect("pos2 checked"),
            "msg1 must appear before msg2"
        );
        prop_assert!(
            pos2.expect("pos2 checked") < pos3.expect("pos3 checked"),
            "msg2 must appear before msg3"
        );
    }

    /// FALSIFY-CT-006-prop: LLaMA2 system messages wrapped in <<SYS>> delimiters.
    ///
    /// For Llama2 format with a system message, the output must contain
    /// both `<<SYS>>` and `<</SYS>>`.
    #[test]
    fn falsify_ct_006_prop_llama2_system_wrapped(
        sys_content in "[a-zA-Z0-9 ]{1,100}",
        user_content in "[a-zA-Z0-9 ]{1,100}"
    ) {
        let template = Llama2Template::new();
        let messages = vec![
            ChatMessage::system(sys_content),
            ChatMessage::user(user_content),
        ];
        let output = template
            .format_conversation(&messages)
            .expect("format_conversation must not fail");

        prop_assert!(
            output.contains("<<SYS>>"),
            "LLaMA2 output missing <<SYS>> delimiter: {:?}",
            output
        );
        prop_assert!(
            output.contains("<</SYS>>"),
            "LLaMA2 output missing <</SYS>> delimiter: {:?}",
            output
        );
    }

    /// FALSIFY-CT-008-prop: detect_format_from_name never panics.
    ///
    /// Random strings must never cause `detect_format_from_name` to panic.
    #[test]
    fn falsify_ct_008_prop_detect_format_never_panics(name in ".*") {
        let _ = detect_format_from_name(&name);
    }

    /// FALSIFY-CT-011-prop: Unknown model names produce Raw format.
    ///
    /// Random strings that do not contain any known model keywords
    /// (qwen, openhermes, yi-, mistral, mixtral, llama, vicuna,
    /// tinyllama, phi-, phi2, phi3, alpaca) must produce `TemplateFormat::Raw`.
    #[test]
    fn falsify_ct_011_prop_unknown_names_produce_raw(name in "[0-9]{1,50}") {
        // Strategy: purely numeric strings cannot contain any model keyword
        let format = detect_format_from_name(&name);
        prop_assert_eq!(
            format,
            TemplateFormat::Raw,
            "Numeric-only name {:?} should produce Raw, got {:?}",
            name,
            format
        );
    }

    /// FALSIFY-CT-012-prop: Sanitization is idempotent.
    ///
    /// `sanitize_user_content(sanitize_user_content(x)) == sanitize_user_content(x)`
    #[test]
    fn falsify_ct_012_prop_sanitization_idempotent(input in ".*") {
        let once = sanitize_user_content(&input);
        let twice = sanitize_user_content(&once);
        prop_assert_eq!(
            once.clone(),
            twice.clone(),
            "Sanitization is not idempotent: once={:?}, twice={:?}",
            once,
            twice
        );
    }

    /// FALSIFY-CT-UTF8-prop: Valid UTF-8 preservation.
    ///
    /// Valid UTF-8 in -> valid UTF-8 out. The output of format_conversation
    /// should always be valid UTF-8 (which Rust strings guarantee), and the
    /// original Unicode content must be present in the output.
    /// Strategy uses emojis, CJK characters, and ZWJ sequences.
    #[test]
    fn falsify_ct_utf8_prop_unicode_preservation(
        content in prop::string::string_regex("[a-z\u{4e00}-\u{4e10}\u{1f600}-\u{1f610}\u{0410}-\u{042f} ]{1,50}").unwrap()
    ) {
        let template = ChatMLTemplate::new();
        let messages = vec![ChatMessage::user(content.clone())];
        let output = template
            .format_conversation(&messages)
            .expect("format_conversation must not fail");

        // Rust strings are always valid UTF-8 by construction, but verify
        // the content is preserved in the output
        prop_assert!(
            output.contains(&content),
            "Unicode content not preserved in output.\nContent: {:?}\nOutput: {:?}",
            content,
            output
        );
    }
}
