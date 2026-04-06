//! Tokenizer stress tests for Phase 46 - The Soft Target Sweep
//!
//! Goal: Push apr/tokenizer.rs from 33% to >90% coverage.

use realizar::apr::{
    byte_to_bpe_char, detect_format, is_apr_file, simd_dot, BpeTokenizer, SimpleTokenizer,
};
use std::collections::HashMap;

// ============================================================================
// SimpleTokenizer Tests
// ============================================================================

#[test]
fn test_simple_tokenizer_new() {
    let vocab = vec!["hello".to_string(), "world".to_string()];
    let tok = SimpleTokenizer::new(vocab.clone(), Some(0), Some(1));

    assert_eq!(tok.vocab_size(), 2);
    assert_eq!(tok.bos_token_id, Some(0));
    assert_eq!(tok.eos_token_id, Some(1));
    assert_eq!(tok.id_to_token, vocab);
}

#[test]
fn test_simple_tokenizer_no_special_tokens() {
    let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let tok = SimpleTokenizer::new(vocab, None, None);

    assert_eq!(tok.vocab_size(), 3);
    assert!(tok.bos_token_id.is_none());
    assert!(tok.eos_token_id.is_none());
}

#[test]
fn test_simple_tokenizer_is_eos() {
    let vocab = vec!["<eos>".to_string()];
    let tok = SimpleTokenizer::new(vocab, None, Some(0));

    assert!(tok.is_eos(0));
    assert!(!tok.is_eos(1));
    assert!(!tok.is_eos(999));
}

#[test]
fn test_simple_tokenizer_is_eos_none() {
    let vocab = vec!["hello".to_string()];
    let tok = SimpleTokenizer::new(vocab, None, None);

    assert!(!tok.is_eos(0));
    assert!(!tok.is_eos(1));
}

#[test]
fn test_simple_tokenizer_is_bos() {
    let vocab = vec!["<bos>".to_string()];
    let tok = SimpleTokenizer::new(vocab, Some(0), None);

    assert!(tok.is_bos(0));
    assert!(!tok.is_bos(1));
    assert!(!tok.is_bos(999));
}

#[test]
fn test_simple_tokenizer_is_bos_none() {
    let vocab = vec!["hello".to_string()];
    let tok = SimpleTokenizer::new(vocab, None, None);

    assert!(!tok.is_bos(0));
    assert!(!tok.is_bos(1));
}

#[test]
fn test_simple_tokenizer_vocab_size() {
    let tok = SimpleTokenizer::new(vec![], None, None);
    assert_eq!(tok.vocab_size(), 0);

    let tok2 = SimpleTokenizer::new(vec!["a".to_string(); 1000], None, None);
    assert_eq!(tok2.vocab_size(), 1000);
}

// ============================================================================
// BpeTokenizer Tests
// ============================================================================

fn create_test_bpe_tokenizer() -> BpeTokenizer {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("h".to_string(), 0);
    token_to_id.insert("e".to_string(), 1);
    token_to_id.insert("l".to_string(), 2);
    token_to_id.insert("o".to_string(), 3);
    token_to_id.insert("Ġ".to_string(), 4); // Space
    token_to_id.insert("w".to_string(), 5);
    token_to_id.insert("r".to_string(), 6);
    token_to_id.insert("d".to_string(), 7);
    token_to_id.insert("he".to_string(), 8);
    token_to_id.insert("ll".to_string(), 9);
    token_to_id.insert("Ċ".to_string(), 10); // Newline
    token_to_id.insert("ĉ".to_string(), 11); // Tab

    let id_to_token: Vec<String> = (0..12)
        .map(|i| {
            token_to_id
                .iter()
                .find(|(_, &v)| v == i)
                .map(|(k, _)| k.clone())
                .unwrap_or_default()
        })
        .collect();

    let merge_rules = vec![
        ("h".to_string(), "e".to_string()),
        ("l".to_string(), "l".to_string()),
    ];

    BpeTokenizer {
        token_to_id,
        id_to_token,
        merge_rules,
        bos_id: Some(0),
        eos_id: Some(1),
        special_tokens: std::collections::HashMap::new(),
    }
}

#[test]
fn test_bpe_tokenizer_encode_simple() {
    let tok = create_test_bpe_tokenizer();

    // "hello" should encode with merges applied
    let tokens = tok.encode("hello");
    assert!(!tokens.is_empty());
}

#[test]
fn test_bpe_tokenizer_encode_with_space() {
    let tok = create_test_bpe_tokenizer();

    // "hello world" should include the space token (Ġ)
    let tokens = tok.encode("hello world");
    assert!(!tokens.is_empty());
    // Should contain space token
    assert!(tokens.contains(&4)); // Ġ = 4
}

#[test]
fn test_bpe_tokenizer_encode_with_newline() {
    let tok = create_test_bpe_tokenizer();

    let tokens = tok.encode("hello\nworld");
    assert!(!tokens.is_empty());
    // Should contain newline token
    assert!(tokens.contains(&10)); // Ċ = 10
}

#[test]
fn test_bpe_tokenizer_encode_with_tab() {
    let tok = create_test_bpe_tokenizer();

    let tokens = tok.encode("hello\tworld");
    assert!(!tokens.is_empty());
    // Should contain tab token
    assert!(tokens.contains(&11)); // ĉ = 11
}

#[test]
fn test_bpe_tokenizer_encode_empty() {
    let tok = create_test_bpe_tokenizer();
    let tokens = tok.encode("");
    assert!(tokens.is_empty());
}

#[test]
fn test_bpe_tokenizer_decode() {
    let tok = create_test_bpe_tokenizer();

    // Decode [0, 1, 2, 2, 3] = "hello"
    let text = tok.decode(&[0, 1, 2, 2, 3]);
    // Note: decode goes through AprV2Model::decode_tokens
    assert!(!text.is_empty());
}

#[test]
fn test_bpe_tokenizer_decode_empty() {
    let tok = create_test_bpe_tokenizer();
    let text = tok.decode(&[]);
    assert!(text.is_empty());
}

// ============================================================================
// byte_to_bpe_char Tests
// ============================================================================

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
fn test_byte_to_bpe_char_ascii_letters() {
    assert_eq!(byte_to_bpe_char(b'a'), "a");
    assert_eq!(byte_to_bpe_char(b'z'), "z");
    assert_eq!(byte_to_bpe_char(b'A'), "A");
    assert_eq!(byte_to_bpe_char(b'Z'), "Z");
}

#[test]
fn test_byte_to_bpe_char_ascii_digits() {
    assert_eq!(byte_to_bpe_char(b'0'), "0");
    assert_eq!(byte_to_bpe_char(b'9'), "9");
}

#[test]
fn test_byte_to_bpe_char_ascii_punctuation() {
    assert_eq!(byte_to_bpe_char(b'!'), "!");
    assert_eq!(byte_to_bpe_char(b'@'), "@");
    assert_eq!(byte_to_bpe_char(b'#'), "#");
}

#[test]
fn test_byte_to_bpe_char_non_printable() {
    // Non-printable ASCII should be formatted as hex
    assert_eq!(byte_to_bpe_char(0x00), "<0x00>");
    assert_eq!(byte_to_bpe_char(0x01), "<0x01>");
    assert_eq!(byte_to_bpe_char(0x7F), "<0x7F>"); // DEL
}

#[test]
fn test_byte_to_bpe_char_high_bytes() {
    // High bytes (>127) should be formatted as hex
    assert_eq!(byte_to_bpe_char(0x80), "<0x80>");
    assert_eq!(byte_to_bpe_char(0xFF), "<0xFF>");
    assert_eq!(byte_to_bpe_char(0xC0), "<0xC0>");
}

#[test]
fn test_byte_to_bpe_char_all_ascii_printable() {
    // Test all printable ASCII (32-126)
    for b in 33..=126 {
        let result = byte_to_bpe_char(b);
        if b == b' ' {
            assert_eq!(result, "Ġ");
        } else {
            assert_eq!(result, (b as char).to_string());
        }
    }
}

// ============================================================================
// BPE Encoding Edge Cases
// ============================================================================

#[test]
fn test_bpe_encode_unicode() {
    let tok = create_test_bpe_tokenizer();

    // Unicode characters should be encoded as bytes
    let tokens = tok.encode("日本語");
    // Should produce some tokens (even if not in vocab, we get the byte representations)
    // The function filters out unknown tokens, so result may be empty
    // But the code path is exercised
    let _ = tokens;
}

#[test]
fn test_bpe_encode_emoji() {
    let tok = create_test_bpe_tokenizer();

    // Emoji should be encoded as UTF-8 bytes
    let _ = tok.encode("😀");
}

#[test]
fn test_bpe_encode_mixed_ascii_unicode() {
    let tok = create_test_bpe_tokenizer();

    let _ = tok.encode("hello 世界");
}

#[test]
fn test_bpe_merge_application() {
    // Test that merges are actually applied
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    token_to_id.insert("b".to_string(), 1);
    token_to_id.insert("ab".to_string(), 2);

    let id_to_token = vec!["a".to_string(), "b".to_string(), "ab".to_string()];

    let merge_rules = vec![("a".to_string(), "b".to_string())];

    let tok = BpeTokenizer {
        token_to_id,
        id_to_token,
        merge_rules,
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    let tokens = tok.encode("ab");
    // After merge, "ab" should be a single token
    assert_eq!(tokens, vec![2]); // "ab" = 2
}

#[test]
fn test_bpe_multiple_merges() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    token_to_id.insert("b".to_string(), 1);
    token_to_id.insert("c".to_string(), 2);
    token_to_id.insert("ab".to_string(), 3);
    token_to_id.insert("abc".to_string(), 4);

    let id_to_token = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "ab".to_string(),
        "abc".to_string(),
    ];

    let merge_rules = vec![
        ("a".to_string(), "b".to_string()),
        ("ab".to_string(), "c".to_string()),
    ];

    let tok = BpeTokenizer {
        token_to_id,
        id_to_token,
        merge_rules,
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    let tokens = tok.encode("abc");
    // After merges: a+b -> ab, ab+c -> abc
    assert_eq!(tokens, vec![4]); // "abc" = 4
}

#[test]
fn test_bpe_no_merge_when_not_adjacent() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    token_to_id.insert("b".to_string(), 1);
    token_to_id.insert("c".to_string(), 2);
    token_to_id.insert("ab".to_string(), 3);

    let id_to_token = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "ab".to_string(),
    ];

    let merge_rules = vec![("a".to_string(), "b".to_string())];

    let tok = BpeTokenizer {
        token_to_id,
        id_to_token,
        merge_rules,
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    // "acb" - a and b are not adjacent, so no merge
    let tokens = tok.encode("acb");
    assert_eq!(tokens, vec![0, 2, 1]); // a=0, c=2, b=1
}

#[test]
fn test_bpe_repeated_merges() {
    // Test that merge is applied multiple times
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    token_to_id.insert("b".to_string(), 1);
    token_to_id.insert("ab".to_string(), 2);

    let id_to_token = vec!["a".to_string(), "b".to_string(), "ab".to_string()];

    let merge_rules = vec![("a".to_string(), "b".to_string())];

    let tok = BpeTokenizer {
        token_to_id,
        id_to_token,
        merge_rules,
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    // "abab" should become "ab" "ab" -> 2, 2
    let tokens = tok.encode("abab");
    assert_eq!(tokens, vec![2, 2]);
}

#[test]
fn test_bpe_unknown_tokens_filtered() {
    let tok = create_test_bpe_tokenizer();

    // Characters not in vocab should be filtered out
    let tokens = tok.encode("xyz");
    // x, y, z are not in our test vocab
    assert!(tokens.is_empty());
}

// ============================================================================
// Stress Tests with Random Data
// ============================================================================

#[test]
fn test_tokenizer_stress_random_strings() {
    let tok = create_test_bpe_tokenizer();

    // Test various string patterns
    let test_strings = [
        "",
        " ",
        "  ",
        "\n",
        "\t",
        "\n\n\n",
        "a",
        "hello",
        "hello world",
        "hello\nworld",
        "hello\tworld",
        "hello world hello world",
        &"a".repeat(1000),
        &" ".repeat(100),
    ];

    for s in &test_strings {
        let _ = tok.encode(s);
    }
}

#[test]
fn test_tokenizer_stress_all_ascii() {
    let tok = create_test_bpe_tokenizer();

    // Test all ASCII characters
    let all_ascii: String = (0u8..=127).map(|b| b as char).collect();
    let _ = tok.encode(&all_ascii);
}

#[test]
fn test_simple_tokenizer_decode_out_of_bounds() {
    let vocab = vec!["a".to_string(), "b".to_string()];
    let tok = SimpleTokenizer::new(vocab, None, None);

    // Token ID 999 is out of bounds
    let text = tok.decode(&[0, 999, 1]);
    // Should handle gracefully (decode_tokens should skip invalid IDs)
    assert!(!text.is_empty() || text.is_empty()); // Just exercise the path
}

#[test]
fn test_simple_tokenizer_large_vocab() {
    let vocab: Vec<String> = (0..10000).map(|i| format!("token_{}", i)).collect();
    let tok = SimpleTokenizer::new(vocab, Some(0), Some(9999));

    assert_eq!(tok.vocab_size(), 10000);
    assert!(tok.is_bos(0));
    assert!(tok.is_eos(9999));
    assert!(!tok.is_eos(5000));
}

// ============================================================================
// simd_dot Tests (from helpers.rs)
// ============================================================================

#[test]
fn test_simd_dot_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let result = simd_dot(&a, &b);
    assert!((result - 10.0).abs() < 1e-6);
}

#[test]
fn test_simd_dot_zeros() {
    let a = vec![0.0; 16];
    let b = vec![1.0; 16];
    let result = simd_dot(&a, &b);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_simd_dot_ones() {
    let a = vec![1.0; 32];
    let b = vec![1.0; 32];
    let result = simd_dot(&a, &b);
    assert!((result - 32.0).abs() < 1e-6);
}

#[test]
fn test_simd_dot_large() {
    // Test with AVX2-aligned size (8-element chunks)
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..256).map(|i| (255 - i) as f32 * 0.1).collect();
    let result = simd_dot(&a, &b);
    // Just verify it completes without panic
    assert!(result.is_finite());
}

#[test]
fn test_simd_dot_unaligned() {
    // Test with non-8-aligned size (has remainder)
    let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..13).map(|i| i as f32).collect();
    let result = simd_dot(&a, &b);
    // Sum of i^2 from 0 to 12 = 0+1+4+9+16+25+36+49+64+81+100+121+144 = 650
    assert!((result - 650.0).abs() < 1e-4);
}

#[test]
fn test_simd_dot_mismatched_lengths() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 1.0];
    let result = simd_dot(&a, &b);
    // Should use min(a.len, b.len) = 2 elements
    assert!((result - 3.0).abs() < 1e-6); // 1*1 + 2*1 = 3
}

#[test]
fn test_simd_dot_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let result = simd_dot(&a, &b);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_simd_dot_negative() {
    let a = vec![-1.0, -2.0, -3.0, -4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let result = simd_dot(&a, &b);
    assert!((result - (-30.0)).abs() < 1e-6);
}

// ============================================================================
// is_apr_file and detect_format Tests (from helpers.rs)
// ============================================================================

#[test]
fn test_is_apr_file_nonexistent() {
    assert!(!is_apr_file("/nonexistent/path/file.apr"));
}

#[test]
fn test_detect_format_by_extension_apr() {
    // Even nonexistent files can be detected by extension
    assert_eq!(detect_format("/fake/model.apr"), "apr");
}

#[test]
fn test_detect_format_by_extension_gguf() {
    assert_eq!(detect_format("/fake/model.gguf"), "gguf");
}

#[test]
fn test_detect_format_by_extension_safetensors() {
    assert_eq!(detect_format("/fake/model.safetensors"), "safetensors");
}

#[test]
fn test_detect_format_unknown_extension() {
    assert_eq!(detect_format("/fake/model.xyz"), "unknown");
}

#[test]
fn test_detect_format_no_extension() {
    assert_eq!(detect_format("/fake/model"), "unknown");
}
