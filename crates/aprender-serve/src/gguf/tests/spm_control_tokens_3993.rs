// #3993: the SentencePiece (greedy) path of `GGUFModel::encode` recognised special tokens
// only by the `<|...|>` pattern (GH-320), so a llama-family control token such as `</s>`
// was matched as TEXT. On TinyLlama the rendered chat prompt fed the model `.</` `s` `>`
// where llama.cpp feeds `.` `</s>`. Special tokens come from `tokenizer.ggml.token_type`
// (UNKNOWN 2, CONTROL 3, USER_DEFINED 4), as `byte_level_bpe` and llama.cpp's
// `tokenizer_st_partition` already do.

/// A llama-style SPM vocabulary in which greedy text matching of `.</s>` would pick the
/// three pieces `.</` `s` `>` -- the shape measured on TinyLlama.
fn spm_vocab_model(token_types: Option<&[i32]>) -> GGUFModel {
    let tokens = [
        "<unk>",  // 0  UNKNOWN
        "<s>",    // 1  CONTROL
        "</s>",   // 2  CONTROL (eos)
        "▁Hi",    // 3
        ".",      // 4
        ".</",    // 5  the piece greedy matching wrongly chose
        "s",      // 6
        ">",      // 7
        "[INST]", // 8  USER_DEFINED
        "<|x|>",  // 9  NORMAL despite its <|...|> shape
        "▁",      // 10
        "[",      // 11
        "I",      // 12
        "N",      // 13
        "T",      // 14
        "]",      // 15
        "<",      // 16
        "|",      // 17
        "x",      // 18
    ];
    let mut b = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &tokens)
        .add_u32("tokenizer.ggml.eos_token_id", 2);
    if let Some(t) = token_types {
        b = b.add_i32_array("tokenizer.ggml.token_type", t);
    }
    GGUFModel::from_bytes(&b.build()).expect("test model")
}

const LLAMA_TYPES: [i32; 19] = [2, 3, 3, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

#[test]
fn control_eos_inside_text_encodes_as_the_eos_id_3993() {
    let model = spm_vocab_model(Some(&LLAMA_TYPES));
    let ids = model.encode("Hi.</s>").expect("vocabulary");
    // llama.cpp: `▁Hi` `.` `</s>`. Before #3993: `▁Hi` `.</` `s` `>`.
    assert_eq!(
        ids,
        vec![3, 4, 2],
        "`</s>` in text must be the CONTROL token, not `.</` `s` `>`"
    );
}

#[test]
fn user_defined_tokens_are_split_out_like_control_tokens_3993() {
    let model = spm_vocab_model(Some(&LLAMA_TYPES));
    let ids = model.encode("[INST]Hi").expect("vocabulary");
    assert_eq!(
        ids.first(),
        Some(&8),
        "USER_DEFINED `[INST]` is special: {ids:?}"
    );
    assert!(
        ids.contains(&3),
        "the text after it is still encoded: {ids:?}"
    );
}

/// Regression guard, not a must-RED (it passed before #3993): a file without a
/// token-type table keeps the GH-320 `<|...|>` convention. (A NORMAL-typed `<|x|>` still
/// encodes to id 9 through greedy text matching, so ids alone cannot show which path it took.)
#[test]
fn without_token_types_the_gh320_pattern_still_marks_specials_3993() {
    let ids = spm_vocab_model(None).encode("Hi<|x|>").expect("vocabulary");
    assert_eq!(
        ids,
        vec![3, 9],
        "fallback: `<|x|>` is special without token types"
    );
    // ...and without types, `</s>` is NOT special: that is the #3993 defect, confined to
    // files that carry no type table (every llama.cpp-produced GGUF carries one).
    let ids = spm_vocab_model(None).encode("Hi.</s>").expect("vocabulary");
    assert_ne!(ids, vec![3, 4, 2]);
}
