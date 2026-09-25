// #3990: the serve path renders the GGUF's OWN chat_template. Included from
// realize_handlers.rs.

#[cfg(test)]
mod serve_official_chat_template_3990 {
    use super::{format_chat_messages, format_chat_messages_official, ChatMessage};
    use crate::gguf::test_factory::GGUFBuilder;
    use crate::gguf::GGUFModel;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.to_string(), content: content.to_string(), name: None, tool_calls: None, tool_call_id: None }
    }

    /// A GGUF header whose template uses bos, eos, the message content AND the generation
    /// prompt -- none of which any hand-coded family template would produce.
    fn gguf_with(template: Option<&str>) -> GGUFModel {
        let mut b = GGUFBuilder::new()
            .architecture("llama")
            .add_string_array("tokenizer.ggml.tokens", &["<unk>", "<BOS>", "<EOS>"])
            .add_u32("tokenizer.ggml.bos_token_id", 1)
            .add_u32("tokenizer.ggml.eos_token_id", 2);
        if let Some(t) = template {
            b = b.add_string("tokenizer.chat_template", t);
        }
        GGUFModel::from_bytes(&b.build()).expect("synthetic GGUF header parses")
    }

    const MARKER_TEMPLATE: &str =
        "{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}<GO>{% endif %}";

    #[test]
    fn the_served_prompt_is_the_ggufs_own_template_3990() {
        let g = gguf_with(Some(MARKER_TEMPLATE));
        let got = format_chat_messages_official(Some(&g), &[msg("system", "be terse"), msg("user", "hi")], Some("qwen2"));
        assert_eq!(got, "<BOS><system>be terse<EOS><user>hi<EOS><GO>");
    }

    #[test]
    fn no_gguf_or_no_template_keeps_the_legacy_formatter_3990() {
        let msgs = [msg("user", "hi")];
        let legacy = format_chat_messages(&msgs, Some("qwen2"));
        assert_eq!(format_chat_messages_official(None, &msgs, Some("qwen2")), legacy);
        assert_eq!(format_chat_messages_official(Some(&gguf_with(None)), &msgs, Some("qwen2")), legacy);
        assert!(legacy.contains("<|im_start|>user"), "the fallback is really the ChatML family template: {legacy:?}");
    }

    #[test]
    fn a_template_that_fails_to_render_falls_back_loudly_not_to_garbage_3990() {
        let g = gguf_with(Some("{{ raise_exception('unsupported role') }}"));
        let msgs = [msg("user", "hi")];
        assert_eq!(format_chat_messages_official(Some(&g), &msgs, Some("qwen2")), format_chat_messages(&msgs, Some("qwen2")));
    }

    /// REAL MODELS: the serve helper on a real GGUF equals llama.cpp's /apply-template byte
    /// for byte, on every thinking=false cell -- serve renders thinking OFF, production's
    /// default since #3801, so those are the cells that are the same request.
    #[test]
    fn the_served_prompt_equals_llama_cpp_on_real_ggufs_3990() {
        let cells: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../fixtures/chat_template_3990/llama_cpp_df03399.json")).expect("oracle parses");
        let mut ran = 0usize;
        for c in cells.iter().filter(|c| c["thinking"] == false) {
            let path = c["path"].as_str().expect("path");
            if !std::path::Path::new(path).exists() {
                eprintln!("SKIP: {path} not on this host -- this cell did NOT run");
                continue;
            }
            let mapped = crate::gguf::MappedGGUFModel::from_path(path).expect("map");
            let msgs: Vec<ChatMessage> = c["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .map(|m| msg(m["role"].as_str().expect("role"), m["content"].as_str().expect("content")))
                .collect();
            let got = format_chat_messages_official(Some(&mapped.model), &msgs, None);
            assert_eq!(got, c["prompt"].as_str().expect("prompt"), "{path} system={}", c["system"]);
            ran += 1;
        }
        eprintln!("#3990 serve: {ran}/8 real cells compared");
    }
}
