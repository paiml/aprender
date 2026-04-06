
// ============================================================================
// Tests for APR Tokenizer (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SimpleTokenizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simple_tokenizer_new() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(1));
        assert_eq!(tokenizer.vocab_size(), 2);
        assert_eq!(tokenizer.bos_token_id, Some(0));
        assert_eq!(tokenizer.eos_token_id, Some(1));
    }

    #[test]
    fn test_simple_tokenizer_vocab_size() {
        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        assert_eq!(tokenizer.vocab_size(), 3);
    }

    #[test]
    fn test_simple_tokenizer_is_eos() {
        let vocab = vec!["<s>".to_string(), "</s>".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(1));
        assert!(tokenizer.is_eos(1));
        assert!(!tokenizer.is_eos(0));
        assert!(!tokenizer.is_eos(2)); // Out of range
    }

    #[test]
    fn test_simple_tokenizer_is_eos_none() {
        let vocab = vec!["hello".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        assert!(!tokenizer.is_eos(0));
        assert!(!tokenizer.is_eos(1));
    }

    #[test]
    fn test_simple_tokenizer_is_bos() {
        let vocab = vec!["<s>".to_string(), "</s>".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(1));
        assert!(tokenizer.is_bos(0));
        assert!(!tokenizer.is_bos(1));
    }

    #[test]
    fn test_simple_tokenizer_is_bos_none() {
        let vocab = vec!["hello".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        assert!(!tokenizer.is_bos(0));
    }

    // -------------------------------------------------------------------------
    // BpeTokenizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bpe_tokenizer_encode_empty() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let result = tokenizer.encode("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_encode_simple() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("i".to_string(), 1);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["h".to_string(), "i".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let result = tokenizer.encode("hi");
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_bpe_tokenizer_encode_with_space() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ġ".to_string(), 0); // Space
        token_to_id.insert("a".to_string(), 1);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["Ġ".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let result = tokenizer.encode(" a");
        assert_eq!(result, vec![0, 1]);
    }

    // -------------------------------------------------------------------------
    // bpe_encode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bpe_encode_empty() {
        let vocab: HashMap<String, u32> = HashMap::new();
        let merges: Vec<(String, String)> = vec![];
        let special: HashMap<String, u32> = HashMap::new();
        let result = bpe_encode("", &vocab, &merges, &special);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_simple_chars() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        let special: HashMap<String, u32> = HashMap::new();

        let result = bpe_encode("abc", &vocab, &[], &special);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_bpe_encode_with_merge() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("ab".to_string(), 2);
        let special: HashMap<String, u32> = HashMap::new();

        let merges = vec![("a".to_string(), "b".to_string())];
        let result = bpe_encode("ab", &vocab, &merges, &special);
        assert_eq!(result, vec![2]); // "ab" merged
    }

    #[test]
    fn test_bpe_encode_space_handling() {
        let mut vocab = HashMap::new();
        vocab.insert("Ġ".to_string(), 0); // Space becomes Ġ
        let special: HashMap<String, u32> = HashMap::new();

        let result = bpe_encode(" ", &vocab, &[], &special);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_newline_handling() {
        let mut vocab = HashMap::new();
        vocab.insert("Ċ".to_string(), 0); // Newline becomes Ċ
        let special: HashMap<String, u32> = HashMap::new();

        let result = bpe_encode("\n", &vocab, &[], &special);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_unknown_tokens() {
        let vocab: HashMap<String, u32> = HashMap::new();
        let special: HashMap<String, u32> = HashMap::new();
        let result = bpe_encode("xyz", &vocab, &[], &special);
        assert!(result.is_empty()); // Unknown tokens filtered out
    }

    // -------------------------------------------------------------------------
    // byte_to_bpe_char Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_byte_to_bpe_char_space() {
        assert_eq!(byte_to_bpe_char(b' '), "Ġ");
    }

    #[test]
    fn test_byte_to_bpe_char_newline() {
        assert_eq!(byte_to_bpe_char(b'\n'), "Ċ");
    }

    #[test]
    fn test_byte_to_bpe_char_tab() {
        assert_eq!(byte_to_bpe_char(b'\t'), "ĉ");
    }

    #[test]
    fn test_byte_to_bpe_char_ascii() {
        assert_eq!(byte_to_bpe_char(b'a'), "a");
        assert_eq!(byte_to_bpe_char(b'Z'), "Z");
        assert_eq!(byte_to_bpe_char(b'0'), "0");
        assert_eq!(byte_to_bpe_char(b'!'), "!");
    }

    #[test]
    fn test_byte_to_bpe_char_non_printable() {
        // Non-printable bytes get hex encoding
        assert_eq!(byte_to_bpe_char(0x00), "<0x00>");
        assert_eq!(byte_to_bpe_char(0x7F), "<0x7F>");
        assert_eq!(byte_to_bpe_char(0xFF), "<0xFF>");
    }

    // -------------------------------------------------------------------------
    // GH-189: Special Token Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bpe_tokenizer_with_special_tokens() {
        // GH-189: Verify special tokens are tokenized atomically
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<|im_start|>".to_string(), 100);
        token_to_id.insert("<|im_end|>".to_string(), 101);
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("i".to_string(), 1);

        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|im_start|>".to_string(), 100);
        special_tokens.insert("<|im_end|>".to_string(), 101);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec![
                "h".to_string(),
                "i".to_string(),
                // ... padding ...
            ],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens,
        };

        // Test that special tokens are kept atomic
        let result = tokenizer.encode("<|im_start|>hi<|im_end|>");
        assert_eq!(result.len(), 4); // <|im_start|>, h, i, <|im_end|>
        assert_eq!(result[0], 100); // <|im_start|>
        assert_eq!(result[3], 101); // <|im_end|>
    }

    #[test]
    fn test_bpe_encode_with_special_tokens() {
        // GH-189: Test bpe_encode directly with special tokens
        let mut vocab = HashMap::new();
        vocab.insert("<|im_start|>".to_string(), 100);
        vocab.insert("<|im_end|>".to_string(), 101);
        vocab.insert("H".to_string(), 0);
        vocab.insert("i".to_string(), 1);

        let mut special = HashMap::new();
        special.insert("<|im_start|>".to_string(), 100);
        special.insert("<|im_end|>".to_string(), 101);

        let result = bpe_encode("<|im_start|>Hi<|im_end|>", &vocab, &[], &special);
        // Should be: [100, 0, 1, 101] = <|im_start|>, H, i, <|im_end|>
        assert!(result.contains(&100)); // <|im_start|>
        assert!(result.contains(&101)); // <|im_end|>
        // Verify order: special token should be first
        assert_eq!(result[0], 100);
    }

    #[test]
    fn test_split_by_special_tokens() {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|im_start|>".to_string(), 100);
        special_tokens.insert("<|im_end|>".to_string(), 101);

        let text = "<|im_start|>user\nHello<|im_end|>";
        let segments = split_by_special_tokens(text, &special_tokens);

        // Should have 3 segments: Special(100), Regular("user\nHello"), Special(101)
        assert_eq!(segments.len(), 3);
        match &segments[0] {
            TextSegment::Special(id) => assert_eq!(*id, 100),
            TextSegment::Regular(_) => panic!("Expected Special segment"),
        }
        match &segments[1] {
            TextSegment::Regular(s) => assert_eq!(s, "user\nHello"),
            TextSegment::Special(_) => panic!("Expected Regular segment"),
        }
        match &segments[2] {
            TextSegment::Special(id) => assert_eq!(*id, 101),
            TextSegment::Regular(_) => panic!("Expected Special segment"),
        }
    }
}
