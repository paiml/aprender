// #3568: the constraint's vocabulary is built from the SAME per-token decoding `decode` uses.

fn gpt2_vocab_model(token_types: Option<&[i32]>) -> GGUFModel {
    // Qwen-style byte-level BPE: `Ġ` is a space, `Ċ` a newline, `<0x0A>` a byte token
    let tokens = ["<|endoftext|>", "{\"", "Ġverdict", "Ċ", "<0x0A>", "<|im_end|>", "<think>", "é"];
    let mut b = GGUFBuilder::new()
        .architecture("qwen2")
        .hidden_dim("qwen2", 32)
        .num_layers("qwen2", 1)
        .num_heads("qwen2", 1)
        .add_string("tokenizer.ggml.model", "gpt2")
        .add_string_array("tokenizer.ggml.tokens", &tokens)
        .add_u32("tokenizer.ggml.eos_token_id", 0);
    if let Some(t) = token_types {
        b = b.add_i32_array("tokenizer.ggml.token_type", t);
    }
    GGUFModel::from_bytes(&b.build()).expect("test model")
}

#[test]
fn constraint_vocab_bytes_are_exactly_what_decode_renders() {
    let model = gpt2_vocab_model(None);
    let v = model.constraint_vocab().expect("a vocabulary and an eos");
    assert_eq!(v.eos, 0);
    for (id, special) in v.special.iter().enumerate() {
        if *special {
            continue;
        }
        let decoded = model.decode(&[id as u32]);
        match std::str::from_utf8(&v.token_bytes[id]) {
            // whole UTF-8: decode renders exactly these bytes
            Ok(text) => assert_eq!(text, decoded, "token {id}"),
            // a partial UTF-8 byte: decode's final lossy step shows U+FFFD, the RAW byte is kept
            Err(_) => assert_eq!(decoded, "\u{FFFD}", "token {id}"),
        }
    }
    assert_eq!(v.token_bytes[2], b" verdict");
    assert_eq!(v.token_bytes[3], b"\n");
    assert_eq!(v.token_bytes[4], b"\n");
    // `é` in a byte-level vocabulary is the single byte 0xE9, the first half of nothing
    assert_eq!(v.token_bytes[7], vec![0xE9]);
}

#[test]
fn constraint_vocab_marks_control_tokens_special_from_token_type() {
    // 1 normal, 3 control, 4 user-defined, 5 unused
    let model = gpt2_vocab_model(Some(&[3, 1, 1, 1, 6, 3, 4, 5]));
    let v = model.constraint_vocab().expect("a vocabulary and an eos");
    assert_eq!(v.special, vec![true, false, false, false, false, true, false, true]);
    // a user-defined token like <think> is text; its bytes are its decoding
    assert!(!v.special[6]);
}

#[test]
fn constraint_vocab_falls_back_to_the_encode_rule_without_token_type() {
    let model = gpt2_vocab_model(None);
    let v = model.constraint_vocab().expect("a vocabulary and an eos");
    // GH-320's rule: `<|…|>` is special
    assert!(v.special[0] && v.special[5]);
    assert!(!v.special[6], "<think> is not a <|…|> token");
}

#[test]
fn constraint_vocab_needs_an_eos() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["a", "b"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test model");
    assert!(model.constraint_vocab().is_none(), "no eos id: no constraint vocabulary");
}

#[test]
fn constraint_vocab_maps_sentencepiece_boundaries_on_bytes_and_keeps_byte_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["</s>", "▁hello", "<0xE6>"])
        .add_u32("tokenizer.ggml.eos_token_id", 0)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test model");
    let v = model.constraint_vocab().expect("a vocabulary and an eos");
    assert_eq!(v.token_bytes[1], b" hello");
    // a lone byte of a multi-byte UTF-8 sequence must survive as that byte, never U+FFFD
    assert_eq!(v.token_bytes[2], vec![0xE6]);
}
