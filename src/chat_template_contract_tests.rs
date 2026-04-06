
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

    // ═══ FALSIFY-CT-001: Qwen3 template selection ═══

    #[test]
    fn falsify_ct_001_qwen3_gets_nothink_template() {
        // Qwen3 models MUST get Qwen3NoThink, NEVER ChatML
        assert_eq!(
            detect_format_from_name("Qwen3-1.7B-Q4_K_M"),
            TemplateFormat::Qwen3NoThink
        );
        assert_eq!(
            detect_format_from_name("qwen3-0.6b"),
            TemplateFormat::Qwen3NoThink
        );
        assert_eq!(
            detect_format_from_name("Qwen3-8B-Instruct"),
            TemplateFormat::Qwen3NoThink
        );
        assert_eq!(
            detect_format_from_name("qwen3"),
            TemplateFormat::Qwen3NoThink
        );
    }

    #[test]
    fn falsify_ct_001_qwen2_gets_chatml_not_nothink() {
        // Qwen2 MUST get ChatML, NOT Qwen3NoThink
        assert_eq!(
            detect_format_from_name("Qwen2.5-Coder-1.5B"),
            TemplateFormat::ChatML
        );
        assert_eq!(
            detect_format_from_name("Qwen2-0.5B-Instruct"),
            TemplateFormat::ChatML
        );
        assert_ne!(
            detect_format_from_name("Qwen2.5-Coder-1.5B"),
            TemplateFormat::Qwen3NoThink
        );
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
    fn falsify_ct_004_qwen3_nothink_deterministic() {
        let template = Qwen3NoThinkTemplate::new();
        let messages = vec![ChatMessage::user("hello world")];
        let a = template.format_conversation(&messages).unwrap();
        let b = template.format_conversation(&messages).unwrap();
        assert_eq!(a, b, "format_conversation must be deterministic");
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
            Box::new(Qwen3NoThinkTemplate::new()),
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
        assert!(templates.len() >= 8, "expected at least 8 template impls");
    }

    // ═══ FALSIFY-CT-006: Qwen3NoThink pre-fills empty thinking block ═══

    #[test]
    fn falsify_ct_006_nothink_prefills_empty_block() {
        let template = Qwen3NoThinkTemplate::new();
        let messages = vec![ChatMessage::user("hello")];
        let output = template.format_conversation(&messages).unwrap();
        assert!(
            output.contains("<think>\n</think>"),
            "Qwen3NoThinkTemplate must pre-fill empty thinking block.\nGot: {output}"
        );
    }

    #[test]
    fn falsify_ct_006_nothink_ends_with_think_block() {
        let template = Qwen3NoThinkTemplate::new();
        let messages = vec![
            ChatMessage::system("you are a coding assistant"),
            ChatMessage::user("hello"),
        ];
        let output = template.format_conversation(&messages).unwrap();
        assert!(
            output.ends_with("<think>\n</think>\n"),
            "Thinking block must be at end of prompt.\nGot: ...{}", &output[output.len().saturating_sub(80)..]
        );
    }

    // ═══ FALSIFY-CT-CREATE: create_template round-trip ═══

    #[test]
    fn falsify_ct_create_template_roundtrip() {
        // Every TemplateFormat variant must produce a working template
        let formats = [
            TemplateFormat::ChatML,
            TemplateFormat::Qwen3NoThink,
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

    // ═══ FALSIFY-SRV-005: apr-serve-v1 contract (PMAT-188) ═══
    // Qwen3 architecture triggers NoThinkTemplate in serve context

    #[test]
    fn falsify_srv_005_qwen3_architecture_gets_nothink() {
        // Architecture string from GGUF metadata (general.architecture = "qwen3")
        assert_eq!(detect_format_from_name("qwen3"), TemplateFormat::Qwen3NoThink);
    }

    #[test]
    fn falsify_srv_005_qwen3_filename_gets_nothink() {
        // Filename pattern from model file (Qwen3-1.7B-Q4_K_M.gguf)
        assert_eq!(
            detect_format_from_name("Qwen3-1.7B-Q4_K_M"),
            TemplateFormat::Qwen3NoThink
        );
    }

    #[test]
    fn falsify_srv_005_qwen2_does_not_get_nothink() {
        // Qwen2 must get ChatML, NOT Qwen3NoThink
        assert_ne!(detect_format_from_name("qwen2"), TemplateFormat::Qwen3NoThink);
        assert_eq!(detect_format_from_name("qwen2"), TemplateFormat::ChatML);
    }
}
