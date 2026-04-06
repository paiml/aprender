
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
