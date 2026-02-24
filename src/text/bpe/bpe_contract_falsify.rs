//! BPE Tokenizer Contract Falsification Tests
//!
//! Popperian falsification of Sennrich et al. (2016) BPE claims:
//!   - Encode is deterministic (same input → same token IDs)
//!   - Encode-decode roundtrip preserves text (for byte-level BPE)
//!   - Merge priority ordering: lower rank merges applied first
//!   - Special tokens are never split by BPE merges
//!   - Byte-level coverage: all 256 byte values have encoder mappings
//!   - Token IDs are bounded within vocab
//!   - Empty input/output is preserved
//!
//! Five-Whys (PMAT-347):
//!   Why #1: BPE module has 40+ unit tests but zero FALSIFY-BPE-* contract tests
//!   Why #2: existing tests verify implementation details, not Sennrich (2016) claims
//!   Why #3: BPE was implemented before DbC methodology
//!   Why #4: no provable-contract YAML defines BPE tokenizer invariants
//!   Why #5: contract YAMLs focused on embedding layer, not the tokenizer feeding it
//!
//! References:
//!   - Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
//!   - docs/specifications/nlp-models-techniques-spec.md §2.1.1
//!   - src/text/bpe/qwen2bpe_tokenizer.rs (BpeTokenizer::encode/decode)
//!   - src/text/bpe/qwen2.rs (bytes_to_unicode, Qwen2BpeTokenizer)

use super::*;

// ============================================================================
// FALSIFY-BPE-001: Encode determinism
// Contract: same input always produces same token IDs
// ============================================================================

#[test]
fn falsify_bpe_001_encode_determinism() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let input = "fix: null pointer dereference in parse_expr()";

    let t1 = tokenizer.encode(input);
    let t2 = tokenizer.encode(input);

    assert_eq!(t1, t2, "FALSIFIED BPE-001: encoder is non-deterministic");
}

#[test]
fn falsify_bpe_001_encode_determinism_unicode() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let input = "日本語テスト 🦀 Rust";

    let t1 = tokenizer.encode(input);
    let t2 = tokenizer.encode(input);

    assert_eq!(
        t1, t2,
        "FALSIFIED BPE-001: encoder is non-deterministic for unicode input"
    );
}

#[test]
fn falsify_bpe_001_encode_determinism_qwen2() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let input = "<|im_start|>user\nHello world<|im_end|>";

    let t1 = tokenizer.encode(input);
    let t2 = tokenizer.encode(input);

    assert_eq!(
        t1, t2,
        "FALSIFIED BPE-001: Qwen2 encoder is non-deterministic"
    );
}

// ============================================================================
// FALSIFY-BPE-002: Encode-decode roundtrip
// Contract: decode(encode(text)) recovers original text for ASCII
// ============================================================================

#[test]
fn falsify_bpe_002_roundtrip_ascii() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let input = "Hello world";

    let encoded = tokenizer.encode(input);
    let decoded = tokenizer.decode(&encoded);

    // GPT-2 byte-level BPE may add prefix space
    assert!(
        decoded.contains("Hello") && decoded.contains("world"),
        "FALSIFIED BPE-002: roundtrip lost content. Input: '{input}', decoded: '{decoded}'"
    );
}

#[test]
fn falsify_bpe_002_roundtrip_preserves_all_ascii_printable() {
    let tokenizer = BpeTokenizer::gpt2_base();

    // All printable ASCII characters
    for c in 33..=126u8 {
        let input = String::from(c as char);
        let encoded = tokenizer.encode(&input);
        let decoded = tokenizer.decode(&encoded);

        assert!(
            decoded.contains(c as char),
            "FALSIFIED BPE-002: roundtrip lost ASCII char {} (0x{:02x}). Encoded: {:?}, decoded: '{}'",
            c as char, c, encoded, decoded
        );
    }
}

// ============================================================================
// FALSIFY-BPE-003: Merge priority ordering
// Contract: lower rank merges are applied before higher rank merges
// ============================================================================

#[test]
fn falsify_bpe_003_merge_priority_ordering() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    // rank 0 = highest priority
    tokenizer.add_merge("a", "b"); // rank 0
    tokenizer.add_merge("c", "d"); // rank 1
    tokenizer.add_merge("ab", "cd"); // rank 2

    let tokens = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
    ];
    let result = tokenizer.bpe(&tokens);

    // With proper priority: ab + cd first, then abcd
    assert_eq!(
        result,
        vec!["abcd".to_string()],
        "FALSIFIED BPE-003: merge priority violated. Got: {:?}",
        result
    );
}

#[test]
fn falsify_bpe_003_lower_rank_wins() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    // If both "a"+"b" and "b"+"c" are possible, rank 0 wins
    tokenizer.add_merge("a", "b"); // rank 0 (higher priority)
    tokenizer.add_merge("b", "c"); // rank 1 (lower priority)

    let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let result = tokenizer.bpe(&tokens);

    // "a"+"b" should merge first (rank 0), leaving ["ab", "c"]
    // "b"+"c" at rank 1 can't fire (no standalone "b" left)
    assert_eq!(
        result,
        vec!["ab".to_string(), "c".to_string()],
        "FALSIFIED BPE-003: lower rank merge did not take priority. Got: {:?}",
        result
    );
}

// ============================================================================
// FALSIFY-BPE-004: Special token isolation
// Contract: special tokens are never split by BPE merges
// ============================================================================

#[test]
fn falsify_bpe_004_special_tokens_not_split() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let text = "<|endoftext|>Hello";

    let tokens = tokenizer.encode(text);

    // The special token ID (50256) must appear as a single token
    assert!(
        tokens.contains(&50256),
        "FALSIFIED BPE-004: <|endoftext|> was split instead of encoding as single token. Got: {:?}",
        tokens
    );
}

#[test]
fn falsify_bpe_004_qwen2_special_tokens_isolated() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let text = "<|im_start|>user\nHello<|im_end|>";

    let tokens = tokenizer.encode(text);

    assert!(
        tokens.contains(&Qwen2BpeTokenizer::IM_START_ID),
        "FALSIFIED BPE-004: <|im_start|> was split. Got: {:?}",
        tokens
    );
    assert!(
        tokens.contains(&Qwen2BpeTokenizer::IM_END_ID),
        "FALSIFIED BPE-004: <|im_end|> was split. Got: {:?}",
        tokens
    );
}

// ============================================================================
// FALSIFY-BPE-005: Byte-level coverage
// Contract: bytes_to_unicode maps all 256 byte values to unique chars
// ============================================================================

#[test]
fn falsify_bpe_005_byte_encoder_covers_all_256() {
    let (encoder, _decoder) = bytes_to_unicode();

    assert_eq!(
        encoder.len(),
        256,
        "FALSIFIED BPE-005: byte encoder covers only {} of 256 byte values",
        encoder.len()
    );
}

#[test]
fn falsify_bpe_005_byte_encoder_decoder_bijective() {
    let (encoder, decoder) = bytes_to_unicode();

    // Every byte must have a unique char mapping
    let unique_chars: std::collections::HashSet<char> = encoder.values().copied().collect();
    assert_eq!(
        unique_chars.len(),
        256,
        "FALSIFIED BPE-005: byte encoder maps multiple bytes to same char (not bijective)"
    );

    // Decoder must be exact inverse
    for (&byte_val, &char_val) in &encoder {
        assert_eq!(
            decoder.get(&char_val),
            Some(&byte_val),
            "FALSIFIED BPE-005: decoder({:?}) != {}, encoder/decoder not inverse",
            char_val,
            byte_val
        );
    }
}

// ============================================================================
// FALSIFY-BPE-006: Token ID bounds
// Contract: all encoded token IDs are valid within the tokenizer's vocab
// ============================================================================

#[test]
fn falsify_bpe_006_token_ids_in_vocab() {
    let tokenizer = BpeTokenizer::gpt2_base();

    let texts = [
        "Hello world",
        "fix: null pointer",
        "fn main() { println!(\"hello\"); }",
        "The quick brown fox jumps over the lazy dog",
    ];

    for text in &texts {
        let tokens = tokenizer.encode(text);
        for &id in &tokens {
            // Every ID must map back to a token
            assert!(
                tokenizer.id_to_token(id).is_some(),
                "FALSIFIED BPE-006: token ID {} from encoding '{}' has no vocab entry",
                id,
                text
            );
        }
    }
}

#[test]
fn falsify_bpe_006_qwen2_special_ids_in_range() {
    // All Qwen2 special token IDs must be < vocab_size
    let vocab_size = Qwen2BpeTokenizer::new().vocab_size();

    assert!(
        (Qwen2BpeTokenizer::IM_START_ID as usize) < vocab_size,
        "FALSIFIED BPE-006: IM_START_ID {} >= vocab_size {}",
        Qwen2BpeTokenizer::IM_START_ID,
        vocab_size
    );
    assert!(
        (Qwen2BpeTokenizer::IM_END_ID as usize) < vocab_size,
        "FALSIFIED BPE-006: IM_END_ID {} >= vocab_size {}",
        Qwen2BpeTokenizer::IM_END_ID,
        vocab_size
    );
    assert!(
        (Qwen2BpeTokenizer::ENDOFTEXT_ID as usize) < vocab_size,
        "FALSIFIED BPE-006: ENDOFTEXT_ID {} >= vocab_size {}",
        Qwen2BpeTokenizer::ENDOFTEXT_ID,
        vocab_size
    );
}

// ============================================================================
// FALSIFY-BPE-007: Empty input/output preservation
// Contract: encode("") → [] and decode([]) → ""
// ============================================================================

#[test]
fn falsify_bpe_007_encode_empty() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let tokens = tokenizer.encode("");

    assert!(
        tokens.is_empty(),
        "FALSIFIED BPE-007: encode('') should return [], got {:?}",
        tokens
    );
}

#[test]
fn falsify_bpe_007_decode_empty() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let text = tokenizer.decode(&[]);

    assert!(
        text.is_empty(),
        "FALSIFIED BPE-007: decode([]) should return '', got '{}'",
        text
    );
}

#[test]
fn falsify_bpe_007_qwen2_encode_empty() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let tokens = tokenizer.encode("");

    assert!(
        tokens.is_empty(),
        "FALSIFIED BPE-007: Qwen2 encode('') should return [], got {:?}",
        tokens
    );
}

#[test]
fn falsify_bpe_007_qwen2_decode_empty() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let text = tokenizer.decode(&[]);

    assert!(
        text.is_empty(),
        "FALSIFIED BPE-007: Qwen2 decode([]) should return '', got '{}'",
        text
    );
}

// ============================================================================
// FALSIFY-BPE-008: BPE merge monotonicity
// Contract: applying merges never increases token count
// ============================================================================

#[test]
fn falsify_bpe_008_merge_never_increases_count() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_merge("a", "b");
    tokenizer.add_merge("ab", "c");
    tokenizer.add_merge("abc", "d");

    let inputs: Vec<Vec<String>> = vec![
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
        vec!["x".into(), "y".into()],
        vec!["a".into(), "b".into()],
        vec!["a".into()],
    ];

    for input in &inputs {
        let result = tokenizer.bpe(input);
        assert!(
            result.len() <= input.len(),
            "FALSIFIED BPE-008: merge increased token count from {} to {} for {:?}",
            input.len(),
            result.len(),
            input
        );
    }
}

// ============================================================================
// FALSIFY-BPE-009: BPE idempotence
// Contract: bpe(bpe(tokens)) == bpe(tokens) (merges fully applied in one pass)
// ============================================================================

#[test]
fn falsify_bpe_009_bpe_idempotent() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_merge("a", "b");
    tokenizer.add_merge("ab", "c");

    let input = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let once = tokenizer.bpe(&input);
    let twice = tokenizer.bpe(&once);

    assert_eq!(
        once, twice,
        "FALSIFIED BPE-009: bpe is not idempotent. Once: {:?}, twice: {:?}",
        once, twice
    );
}
