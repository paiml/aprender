
// ============================================================================
// Contract Tests: chat-template-v1.yaml (PMAT-187)
//
// Provable contract enforcement for chat template correctness.
// Motivated by PMAT-181/182/185 dogfood findings — three separate bugs
// shipped because no contract enforced template invariants.
// ============================================================================

#[cfg(test)]
mod contract_tests {
    use super::*;

    // ═══ FALSIFY-CT-001: a Qwen3 model does not think unless asked (#3755) ═══
    // PMAT-181's protection, derived: with no thinking choice, the prompt a Qwen3 /
    // Qwen3.5 model gets is ITS OWN template rendered with enable_thinking=false, byte
    // for byte what transformers renders (fixtures from the host inventory).

    fn ct_fixture(name: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/chat_templates")
            .join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    #[test]
    fn falsify_ct_001_qwen3_default_is_the_models_own_no_think_prompt() {
        let reference: serde_json::Value =
            serde_json::from_str(&ct_fixture("reference.json")).expect("reference.json");
        // Qwen3-8B, Qwen3-0.6B, Qwen3-1.7B, Qwen3.5 (0.8B-9B), Qwen3.5 (27B)
        for sha in ["57f1fd00f001", "5da44855ab7e", "8428c815ac94", "7f0e529032c2", "e60df41481b6"] {
            let template = EmbeddedChatTemplate::new(ct_fixture(&format!("{sha}.jinja")))
                .expect("inventory template loads");
            let prompt = format_chat_prompt(
                Some(&template),
                Some("qwen3"),
                &[ChatMessage::user("What is 2+2?")],
                None,
            )
            .expect("default prompt");
            assert!(!prompt.thinking, "{sha}: thinking must default OFF");
            assert_eq!(
                prompt.text,
                reference["cases"][format!("{sha}/single/off")].as_str().expect("case"),
                "{sha}: the default prompt must be the model's own no-think rendering"
            );
            // The model's scaffold, never the hand-typed single-newline one.
            assert!(prompt.text.ends_with("<think>\n\n</think>\n\n"), "{sha}");
        }
    }

    #[test]
    fn falsify_ct_001_no_template_fallback_types_no_scaffold() {
        // A file with no chat template: apr's family fallback, which types no think block.
        for name in ["qwen3", "Qwen3-8B-Q4_K_M", "qwen3moe", "Qwen2.5-Coder-1.5B", "qwen2"] {
            assert_eq!(detect_format_from_name(name), TemplateFormat::ChatML, "{name}");
            let text = auto_detect_template(name)
                .format_conversation(&[ChatMessage::user("hi")])
                .expect("format");
            assert!(!text.contains("<think>"), "{name}: {text:?}");
        }
    }

    #[test]
    fn falsify_ct_001_other_models_correct_template() {
        assert_eq!(
            detect_format_from_name("TinyLlama-1.1B-Chat"),
            TemplateFormat::Zephyr
        );
        assert_eq!(
            detect_format_from_name("Mistral-7B-Instruct"),
            TemplateFormat::Mistral
        );
        assert_eq!(
            detect_format_from_name("phi-2"),
            TemplateFormat::Phi
        );
        assert_eq!(
            detect_format_from_name("llama-3.2-3b"),
            TemplateFormat::Llama2
        );
    }

    // ═══ FALSIFY-CT-004: Template determinism ═══

    #[test]
    fn falsify_ct_004_embedded_template_deterministic() {
        let template =
            EmbeddedChatTemplate::new(ct_fixture("7f0e529032c2.jinja")).expect("loads");
        let messages = vec![ChatMessage::user("hello world")];
        let a = template.render(&messages, true, Some(false)).expect("render");
        let b = template.render(&messages, true, Some(false)).expect("render");
        assert_eq!(a, b, "render must be deterministic");
    }

    #[test]
    fn falsify_ct_004_chatml_deterministic() {
        let template = ChatMLTemplate::new();
        let messages = vec![
            ChatMessage::system("you are helpful"),
            ChatMessage::user("hi"),
        ];
        let a = template.format_conversation(&messages).unwrap();
        let b = template.format_conversation(&messages).unwrap();
        assert_eq!(a, b, "format_conversation must be deterministic");
    }

    // ═══ FALSIFY-CT-005: Template trait coverage ═══

    #[test]
    fn falsify_ct_005_all_impls_satisfy_trait() {
        // Compile-time + runtime check: every template must be constructible
        // and satisfy all ChatTemplateEngine methods.
        let templates: Vec<Box<dyn ChatTemplateEngine>> = vec![
            Box::new(ChatMLTemplate::new()),
            Box::new(Llama2Template::new()),
            Box::new(ZephyrTemplate::new()),
            Box::new(MistralTemplate::new()),
            Box::new(PhiTemplate::new()),
            Box::new(AlpacaTemplate::new()),
            Box::new(RawTemplate::new()),
        ];
        for t in &templates {
            // All 5 trait methods must be callable
            let _ = t.format();
            let _ = t.supports_system_prompt();
            let _ = t.special_tokens();
            let msg = t.format_message("user", "test");
            assert!(msg.is_ok(), "format_message failed for {:?}", t.format());
            let conv = t.format_conversation(&[ChatMessage::user("test")]);
            assert!(conv.is_ok(), "format_conversation failed for {:?}", t.format());
        }
        assert!(templates.len() >= 7, "expected at least 7 template impls");
    }

    // ═══ FALSIFY-CT-006: thinking ON and OFF are the model's own renderings ═══
    // Both modes come from the template (#3755), and a mode it cannot honour is
    // refused by name, never silently mapped (#3723).

    #[test]
    fn falsify_ct_006_both_modes_rendered_and_unhonourable_refused() {
        let reference: serde_json::Value =
            serde_json::from_str(&ct_fixture("reference.json")).expect("reference.json");
        let qwen3 = EmbeddedChatTemplate::new(ct_fixture("57f1fd00f001.jinja")).expect("loads");
        let msgs = [ChatMessage::user("What is 2+2?")];
        for (thinking, mode) in [(true, "on"), (false, "off")] {
            let prompt =
                format_chat_prompt(Some(&qwen3), None, &msgs, Some(thinking)).expect("honoured");
            assert_eq!(prompt.thinking, thinking);
            assert_eq!(
                prompt.text,
                reference["cases"][format!("57f1fd00f001/single/{mode}")].as_str().expect("case")
            );
        }
        // Qwen3-30B-A3B-Instruct-2507: think markers, no thinking mode.
        let instruct = EmbeddedChatTemplate::new(ct_fixture("40c21f34cf67.jinja")).expect("loads");
        let refused = format_chat_prompt(Some(&instruct), None, &msgs, Some(true))
            .expect_err("thinking ON on a non-thinking template");
        assert!(matches!(refused, RealizarError::ThinkingModeUnsupported { .. }), "{refused}");
        // No template at all: thinking cannot be switched on.
        assert!(format_chat_prompt(None, Some("qwen3"), &msgs, Some(true)).is_err());
    }

    // ═══ FALSIFY-CT-CREATE: create_template round-trip ═══

    #[test]
    fn falsify_ct_create_template_roundtrip() {
        // Every TemplateFormat variant must produce a working template
        let formats = [
            TemplateFormat::ChatML,
            TemplateFormat::Llama2,
            TemplateFormat::Zephyr,
            TemplateFormat::Mistral,
            TemplateFormat::Phi,
            TemplateFormat::Alpaca,
            TemplateFormat::Raw,
        ];
        for fmt in &formats {
            let template = create_template(*fmt);
            assert_eq!(template.format(), *fmt, "create_template({fmt:?}) returned wrong format");
        }
    }

    // ═══ FALSIFY-CT-AUTO: auto_detect_template consistency ═══

    #[test]
    fn falsify_ct_auto_detect_consistency() {
        // auto_detect_template must be consistent with detect_format_from_name + create_template
        for name in &["qwen3-1.7b", "qwen2-7b", "llama-3.2", "phi-3", "mistral-7b", "tinyllama"] {
            let format = detect_format_from_name(name);
            let template = auto_detect_template(name);
            assert_eq!(
                template.format(),
                format,
                "auto_detect_template({name}) format mismatch"
            );
        }
    }

    // ═══ FALSIFY-SRV-005: apr-serve-v1 contract (PMAT-188), derived (#3755) ═══
    // serve's Qwen3 prompt comes from the model's template, not from the name.

    #[test]
    fn falsify_srv_005_legacy_nothink_variant_builds_no_scaffold() {
        let text = create_template(TemplateFormat::Qwen3NoThink)
            .format_conversation(&[ChatMessage::user("hi")])
            .expect("format");
        assert!(!text.contains("<think>"), "{text:?}");
        assert_eq!(create_template(TemplateFormat::Qwen3NoThink).format(), TemplateFormat::ChatML);
    }

    // ═══ FALSIFY-CT-762: Llama2 system-message handling ═══
    // Adversarial chat-template bug-hunt (2026-06-14) confirmed Llama2Template dropped
    // system context when no user message followed, and overwrote multiple system messages.

    #[test]
    fn falsify_ct_762_llama2_system_only_not_dropped() {
        // A system-only request must NOT lose the system content (previously returned "<s>").
        let conv = Llama2Template::new()
            .format_conversation(&[ChatMessage::system("Be terse.")])
            .expect("format");
        assert!(
            conv.contains("Be terse."),
            "system-only prompt dropped the system content: {conv:?}"
        );
        assert!(conv.contains("<<SYS>>"), "system block markers missing: {conv:?}");
    }

    #[test]
    fn falsify_ct_762_llama2_multiple_system_messages_preserved() {
        // Multiple system messages must all survive (previously only the LAST was kept).
        let conv = Llama2Template::new()
            .format_conversation(&[
                ChatMessage::system("First rule."),
                ChatMessage::system("Second rule."),
                ChatMessage::user("hi"),
            ])
            .expect("format");
        assert!(conv.contains("First rule."), "first system message dropped: {conv:?}");
        assert!(conv.contains("Second rule."), "second system message dropped: {conv:?}");
    }

    #[test]
    fn falsify_ct_762_llama2_system_user_still_correct() {
        // Regression guard: the common [system, user] case stays well-formed.
        let conv = Llama2Template::new()
            .format_conversation(&[
                ChatMessage::system("Be helpful."),
                ChatMessage::user("2+2?"),
            ])
            .expect("format");
        assert!(conv.starts_with("<s>[INST] <<SYS>>\nBe helpful.\n<</SYS>>\n\n2+2? [/INST]"));
        // Exactly ONE leading <s> — no double-BOS (the system message must not trigger the
        // new-round <s> that belongs only after a completed assistant turn).
        assert_eq!(conv.matches("<s>").count(), 1, "double-BOS for [system,user]: {conv:?}");
    }

    // ═══ FALSIFY-CT-763: RawTemplate (unknown/default model fallback) separates turns ═══

    #[test]
    fn falsify_ct_763_rawtemplate_separates_messages() {
        // The previous `.collect::<String>()` produced "HelloWorld" (no separators), an
        // unparseable multi-turn prompt for unknown/"default"-named models. Each message's
        // content must be newline-delimited so turns are distinguishable.
        let conv = RawTemplate::new()
            .format_conversation(&[
                ChatMessage::user("Hello"),
                ChatMessage::assistant("World"),
                ChatMessage::user("Again"),
            ])
            .expect("format");
        assert!(
            !conv.contains("HelloWorld"),
            "RawTemplate merged messages with no separator: {conv:?}"
        );
        assert!(conv.contains("Hello\n"), "missing newline separator: {conv:?}");
        assert!(conv.contains("World\n"), "missing newline separator: {conv:?}");
        assert!(conv.contains("Again"), "dropped a message: {conv:?}");
    }

    #[test]
    fn falsify_ct_762_llama2_multiturn_round_separator_preserved() {
        // Regression the OTHER way: a genuine new round (after an assistant turn) MUST still
        // open with <s>, so multi-turn Llama-2 format is intact.
        let conv = Llama2Template::new()
            .format_conversation(&[
                ChatMessage::user("q1"),
                ChatMessage::assistant("a1"),
                ChatMessage::user("q2"),
            ])
            .expect("format");
        assert_eq!(conv.matches("<s>").count(), 2, "missing new-round <s>: {conv:?}");
        assert!(conv.contains("</s><s>[INST] q2 [/INST]"), "bad round boundary: {conv:?}");
    }
}
