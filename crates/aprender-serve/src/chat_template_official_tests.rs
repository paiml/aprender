// #3990 acceptance: the rendered prompt must EQUAL llama.cpp's /apply-template output, for
// (Qwen2.5, Qwen3, Qwen3.5) x (system-less, with system) x (thinking off, on).
//
// The oracle is llama.cpp @ df03399 (the tree the gguf-py decoder proofs used), run
// CPU-only; its prompts and token ids are committed in fixtures/chat_template_3990/, and
// the three templates were extracted verbatim from the GGUFs' tokenizer.chat_template.
#[cfg(test)]
mod official_chat_template_3990 {
    use super::*;

    const ORACLE: &str = include_str!("fixtures/chat_template_3990/llama_cpp_df03399.json");
    const QWEN25: &str = include_str!("fixtures/chat_template_3990/qwen25.jinja");
    const QWEN3: &str = include_str!("fixtures/chat_template_3990/qwen3.jinja");
    const QWEN35: &str = include_str!("fixtures/chat_template_3990/qwen35.jinja");

    fn template_for(model: &str) -> &'static str {
        match model {
            "qwen25" => QWEN25,
            "qwen3" => QWEN3,
            "qwen35" => QWEN35,
            m => panic!("no fixture template for {m}"),
        }
    }

    fn cells() -> Vec<serde_json::Value> {
        let v: Vec<serde_json::Value> = serde_json::from_str(ORACLE).expect("oracle fixture parses");
        assert_eq!(v.len(), 12, "the matrix is 3 models x 2 system x 2 thinking");
        v
    }

    fn messages_of(cell: &serde_json::Value) -> Vec<ChatMessage> {
        cell["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|m| ChatMessage::new(m["role"].as_str().unwrap(), m["content"].as_str().unwrap()))
            .collect()
    }

    /// HERMETIC: needs no model file, so it runs in CI. Every one of the 12 cells must render
    /// BYTE-FOR-BYTE what llama.cpp renders. A mismatch names the cell and the first byte
    /// that differs, with both sides around it.
    #[test]
    fn the_official_renderer_equals_llama_cpp_on_every_cell_3990() {
        let mut bad = Vec::new();
        for c in cells() {
            let (model, sys, think) = (c["model"].as_str().unwrap(), c["system"].as_bool().unwrap(), c["thinking"].as_bool().unwrap());
            let want = c["prompt"].as_str().unwrap();
            let got = render_official(template_for(model), None, None, &messages_of(&c), true, Some(think))
                .unwrap_or_else(|e| panic!("{model} system={sys} thinking={think}: {e}"));
            if got != want {
                let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                let win = |s: &str| s.get(at.saturating_sub(24)..(at + 24).min(s.len())).unwrap_or("").to_string();
                bad.push(format!(
                    "\n  {model} system={sys} thinking={think}: first difference at byte {at} (apr {} bytes, llama.cpp {})\n      apr      ...{:?}...\n      llama.cpp ...{:?}...",
                    got.len(), want.len(), win(&got), win(&want)
                ));
            }
        }
        assert!(bad.is_empty(), "apr's render differs from llama.cpp's /apply-template:{}", bad.concat());
    }

    /// The three behaviours #3990 was filed for, asserted by name so a regression reads as
    /// what it is rather than as a byte offset.
    #[test]
    fn the_three_measured_defects_are_gone_3990() {
        let user = [ChatMessage::new("user", "hi")];
        // Qwen2.5 system-less keeps the template's default system prompt.
        let q25 = render_official(QWEN25, None, None, &user, true, None).unwrap();
        assert!(q25.contains("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."), "{q25:?}");
        // Qwen3.5 thinking OFF prefills with TWO newlines, as the template does.
        let off = render_official(QWEN35, None, None, &user, true, Some(false)).unwrap();
        assert!(off.ends_with("assistant\n<think>\n\n</think>\n\n"), "{off:?}");
        // Qwen3.5 thinking ON OPENS the block.
        let on = render_official(QWEN35, None, None, &user, true, Some(true)).unwrap();
        assert!(on.ends_with("assistant\n<think>\n"), "{on:?}");
    }

    /// REAL MODELS: the acceptance criterion as written -- token IDS. Render through the
    /// GGUF, tokenize with apr's own tokenizer, compare to llama.cpp's ids. Says SKIP, by
    /// name, for a model not on this host rather than passing.
    #[test]
    fn rendered_prompt_ids_equal_llama_cpp_on_every_cell_3990() {
        let mut ran = 0usize;
        let mut bad = Vec::new();
        for c in cells() {
            let path = c["path"].as_str().unwrap();
            if !std::path::Path::new(path).exists() {
                eprintln!("SKIP: {path} not on this host -- this cell's id check did NOT run");
                continue;
            }
            let (model, sys, think) = (c["model"].as_str().unwrap(), c["system"].as_bool().unwrap(), c["thinking"].as_bool().unwrap());
            let mapped = crate::gguf::MappedGGUFModel::from_path(path).expect("map");
            let prompt = render_official_for_model(&mapped.model, &messages_of(&c), Some(think))
                .unwrap_or_else(|e| panic!("{model}: {e}"));
            let got = mapped.model.encode(&prompt).expect("apr encodes the rendered prompt");
            let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| u32::try_from(v.as_u64().unwrap()).unwrap()).collect();
            if got != want {
                let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                bad.push(format!("\n  {model} system={sys} thinking={think}: {} vs {} ids, first differ at {at}: apr {:?} llama.cpp {:?}",
                    got.len(), want.len(), got.get(at), want.get(at)));
            }
            ran += 1;
        }
        eprintln!("#3990 ids: {ran}/12 cells compared");
        assert!(bad.is_empty(), "rendered prompt ids differ from llama.cpp:{}", bad.concat());
    }
}
