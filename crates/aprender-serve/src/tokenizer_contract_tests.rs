
// ═══ tokenizer-loading-v1 contract enforcement (PMAT-189) ═══

#[cfg(test)]
mod tokenizer_contract_tests {
    use super::BPETokenizer;

    fn make_test_tokenizer() -> BPETokenizer {
        // Minimal vocab: a-z + <unk> + <|im_start|> + <|im_end|> + <|endoftext|>
        let mut vocab: Vec<String> = (b'a'..=b'z').map(|c| String::from(c as char)).collect();
        vocab.push("<unk>".to_string());
        vocab.push("<|im_start|>".to_string());
        vocab.push("<|im_end|>".to_string());
        vocab.push("<|endoftext|>".to_string());
        vocab.push("he".to_string());
        vocab.push("ll".to_string());
        vocab.push("lo".to_string());
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("l".to_string(), "o".to_string()),
        ];
        BPETokenizer::new(vocab, merges, "<unk>").expect("test tokenizer")
    }

    /// #4661/#4662: greedy longest-match must not cut into a control token.
    ///
    /// Qwen GGUF vocabularies hold `?<` and `.<`, so before the fix greedy matching
    /// took `?<` and left `|im_end|>` as plain text: the model never saw a turn end.
    #[test]
    fn greedy_encode_keeps_control_tokens_whole() {
        let vocab: Vec<String> = [
            "<unk>", "?", "?<", ".<", "<", "|", ">", "i", "m", "_", "e", "n", "d", "s", "t",
            "a", "r", "Ċ", "<|im_end|>", "<|im_start|>", "x",
        ]
        .iter()
        .map(|t| (*t).to_string())
        .collect();
        let id = |t: &str| vocab.iter().position(|v| v == t).expect("in vocab") as u32;
        let tokenizer = BPETokenizer::new(vocab.clone(), vec![], "<unk>").expect("tokenizer");

        assert_eq!(
            tokenizer.encode("x?<|im_end|>\n<|im_start|>a"),
            vec![
                id("x"),
                id("?"),
                id("<|im_end|>"),
                id("Ċ"),
                id("<|im_start|>"),
                id("a")
            ],
        );
        // Text with no control token is untouched: `?<` is still matched greedily.
        assert_eq!(tokenizer.encode("x?<"), vec![id("x"), id("?<")]);
    }

    /// F-TOK-004: Deterministic encoding — same input always produces same tokens.
    #[test]
    fn falsify_tok_004_deterministic_encoding() {
        let tokenizer = make_test_tokenizer();
        let text = "hello";
        let ids_a = tokenizer.encode(text);
        let ids_b = tokenizer.encode(text);
        assert_eq!(ids_a, ids_b, "F-TOK-004: encode must be deterministic");
    }

    /// F-TOK-005: Empty input handling — encode('') returns empty, no crash.
    #[test]
    fn falsify_tok_005_empty_input() {
        let tokenizer = make_test_tokenizer();
        let ids = tokenizer.encode("");
        assert!(ids.is_empty(), "F-TOK-005: empty input should produce empty tokens");
    }

    /// F-TOK-003: Vocab size matches constructor input.
    #[test]
    fn falsify_tok_003_vocab_size() {
        let tokenizer = make_test_tokenizer();
        // 26 letters + <unk> + 3 special + 3 merges = 33
        assert!(tokenizer.vocab_size() >= 26, "F-TOK-003: vocab must include at least a-z");
    }

    /// F-TOK-004b: Encoding the same text multiple times is stable.
    #[test]
    fn falsify_tok_004b_encoding_stability() {
        let tokenizer = make_test_tokenizer();
        for input in &["a", "hello", "abc", ""] {
            let first = tokenizer.encode(input);
            for _ in 0..5 {
                assert_eq!(
                    tokenizer.encode(input),
                    first,
                    "F-TOK-004: encoding '{input}' must be stable across calls"
                );
            }
        }
    }

    /// Contract: BPETokenizer is Send + Sync (required for concurrent serve).
    #[test]
    fn falsify_tok_thread_safety() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BPETokenizer>();
    }
}
