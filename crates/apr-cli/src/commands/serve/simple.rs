
/// SafeTensors generate handler
#[cfg(feature = "inference")]
pub(crate) async fn safetensors_generate_handler(
    axum::extract::State(state): axum::extract::State<SafeTensorsState>,
    axum::Json(request): axum::Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let prompt = request.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
    let max_tokens = request
        .get("max_tokens")
        .and_then(|m| m.as_u64())
        .unwrap_or(32) as usize;

    let transformer = match &state.transformer {
        Some(t) => t.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({"error": "Inference not available"})),
            )
                .into_response();
        }
    };

    // Encode prompt using BPE tokenizer (PMAT-093)
    let input_ids = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info.tokenizer.encode(prompt)
    } else {
        prompt.chars().map(|c| c as u32).collect()
    };

    // PMAT-103 FIX: Use generate_with_cache for O(n) generation
    // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
    // generate_with_cache() uses KV cache for incremental generation = O(n)
    let start = Instant::now();
    let temperature = request
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0) as f32;
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: false,
        stop_tokens: vec![],
    };
    let output_ids = {
        // PMAT-189: Handle transformer lock poisoning gracefully
        let t = match transformer.lock() {
            Ok(guard) => guard,
            Err(_poisoned) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": "Transformer state corrupted (lock poisoned). Please restart the server."
                    })),
                )
                    .into_response();
            }
        };
        match t.generate_with_cache(&input_ids, &gen_config) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": format!("Generation failed: {e}")})),
                )
                    .into_response();
            }
        }
    };
    let elapsed = start.elapsed();

    // Decode using BPE tokenizer (PMAT-093)
    let new_tokens = &output_ids[input_ids.len()..];
    let output_text = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info
            .tokenizer
            .decode(new_tokens)
            .unwrap_or_else(|_| simple_decode(new_tokens, &tok_info.vocab))
    } else {
        new_tokens
            .iter()
            .map(|&id| char::from_u32(id.min(127)).unwrap_or('?'))
            .collect()
    };

    let tokens_generated = new_tokens.len();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    axum::Json(serde_json::json!({
        "text": output_text,
        "tokens_generated": tokens_generated,
        "latency_ms": elapsed.as_millis(),
        "tok_per_sec": tok_per_sec
    }))
    .into_response()
}

/// Simple tokenization using greedy longest match
pub(crate) fn simple_encode(text: &str, vocab: &[String]) -> Vec<u32> {
    let mut tokens = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Find longest matching token
        let mut best_match = None;
        let mut best_len = 0;

        for (id, token) in vocab.iter().enumerate() {
            if remaining.starts_with(token) && token.len() > best_len {
                best_match = Some(id as u32);
                best_len = token.len();
            }
        }

        if let Some(id) = best_match {
            tokens.push(id);
            remaining = &remaining[best_len..];
        } else {
            // Skip unknown character
            let char_len = remaining.chars().next().map_or(1, char::len_utf8);
            remaining = &remaining[char_len..];
        }
    }

    tokens
}

/// Simple decode using vocab lookup
pub(crate) fn simple_decode(token_ids: &[u32], vocab: &[String]) -> String {
    token_ids
        .iter()
        .map(|&id| {
            vocab
                .get(id as usize)
                .map_or("?".to_string(), |s| s.clone())
        })
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Helper: build a small vocab for testing
    // ========================================================================

    /// Build a minimal vocabulary: ["h", "e", "l", "lo", "hello", " ", "world"]
    fn test_vocab() -> Vec<String> {
        vec![
            "h".to_string(),     // 0
            "e".to_string(),     // 1
            "l".to_string(),     // 2
            "lo".to_string(),    // 3
            "hello".to_string(), // 4
            " ".to_string(),     // 5
            "world".to_string(), // 6
        ]
    }

    // ========================================================================
    // A. simple_encode — normal cases
    // ========================================================================

    /// Greedy longest match should prefer "hello" (id=4) over "h"+"e"+"l"+"lo"
    #[test]
    fn test_simple_encode_greedy_longest_match() {
        let vocab = test_vocab();
        let tokens = simple_encode("hello", &vocab);
        // "hello" is the longest match starting at position 0
        assert_eq!(tokens, vec![4]);
    }

    /// Encodes "hello world" as ["hello", " ", "world"]
    #[test]
    fn test_simple_encode_multi_token() {
        let vocab = test_vocab();
        let tokens = simple_encode("hello world", &vocab);
        assert_eq!(tokens, vec![4, 5, 6]);
    }

    /// Single-character tokens when no longer match exists
    #[test]
    fn test_simple_encode_single_char_fallback() {
        let vocab = test_vocab();
        // "hel" -> "h"(0) + "e"(1) + "l"(2)
        let tokens = simple_encode("hel", &vocab);
        assert_eq!(tokens, vec![0, 1, 2]);
    }

    /// "lo" should match id=3 (length 2) rather than just "l" (length 1)
    #[test]
    fn test_simple_encode_prefers_longer_token() {
        let vocab = test_vocab();
        let tokens = simple_encode("lo", &vocab);
        assert_eq!(tokens, vec![3]);
    }

    // ========================================================================
    // B. simple_encode — edge cases
    // ========================================================================

    /// Empty input produces empty output
    #[test]
    fn test_simple_encode_empty_input() {
        let vocab = test_vocab();
        let tokens = simple_encode("", &vocab);
        assert!(tokens.is_empty());
    }

    /// Empty vocab: all characters are unknown and skipped
    #[test]
    fn test_simple_encode_empty_vocab() {
        let vocab: Vec<String> = vec![];
        let tokens = simple_encode("abc", &vocab);
        // Every character is unknown and skipped
        assert!(tokens.is_empty());
    }

    /// Characters not in vocab are silently skipped
    #[test]
    fn test_simple_encode_unknown_chars_skipped() {
        let vocab = test_vocab();
        // 'x', 'y', 'z' are not in the vocab
        let tokens = simple_encode("xyz", &vocab);
        assert!(tokens.is_empty());
    }

    /// Mix of known and unknown characters: only known ones produce tokens
    #[test]
    fn test_simple_encode_mixed_known_unknown() {
        let vocab = test_vocab();
        // "xhey" -> skip 'x', then "h"(0), "e"(1), skip 'y'
        let tokens = simple_encode("xhey", &vocab);
        assert_eq!(tokens, vec![0, 1]);
    }

    /// Unicode characters not in vocab are properly skipped (multi-byte)
    #[test]
    fn test_simple_encode_unicode_skipped() {
        let vocab = test_vocab();
        let tokens = simple_encode("h\u{00e9}llo", &vocab);
        // 'h'(0), skip U+00E9 (e-acute, 2 bytes), "l"(2), "lo"... but 'l' consumed first
        // After 'h', next is U+00E9 which is not in vocab -> skip
        // Then 'l' (2), then 'lo' (3) -- wait, after 'l' is consumed, 'l' then 'o'
        // Actually: h, skip(e-acute), l, l, o
        // "l"(2), "lo"? no -- after first 'l' is consumed, remaining is "lo" -> "lo"(3)
        assert_eq!(tokens, vec![0, 2, 3]);
    }

    /// Vocab with multi-byte unicode tokens
    #[test]
    fn test_simple_encode_unicode_in_vocab() {
        let vocab = vec![
            "\u{00e9}".to_string(), // 0: e-acute
            "caf".to_string(),      // 1
            "cafe".to_string(),     // 2 -- won't match "caf\u{00e9}"
        ];
        // "caf\u{00e9}" -> "caf"(1) + "\u{00e9}"(0)
        let tokens = simple_encode("caf\u{00e9}", &vocab);
        assert_eq!(tokens, vec![1, 0]);
    }

    /// Repeated tokens
    #[test]
    fn test_simple_encode_repeated_tokens() {
        let vocab = vec!["a".to_string(), "aa".to_string()];
        // "aaaa" -> greedy: "aa"(1), "aa"(1)
        let tokens = simple_encode("aaaa", &vocab);
        assert_eq!(tokens, vec![1, 1]);
    }

    /// Odd-length repeated: "aaa" -> "aa"(1), "a"(0)
    #[test]
    fn test_simple_encode_odd_repeated() {
        let vocab = vec!["a".to_string(), "aa".to_string()];
        let tokens = simple_encode("aaa", &vocab);
        assert_eq!(tokens, vec![1, 0]);
    }

    // ========================================================================
    // C. simple_decode — normal cases
    // ========================================================================

    /// Decode single token
    #[test]
    fn test_simple_decode_single_token() {
        let vocab = test_vocab();
        let text = simple_decode(&[4], &vocab);
        assert_eq!(text, "hello");
    }

    /// Decode multiple tokens
    #[test]
    fn test_simple_decode_multiple_tokens() {
        let vocab = test_vocab();
        let text = simple_decode(&[4, 5, 6], &vocab);
        assert_eq!(text, "hello world");
    }

    /// Decode concatenates tokens without separators
    #[test]
    fn test_simple_decode_concatenation() {
        let vocab = test_vocab();
        // [0, 1, 2, 3] -> "h" + "e" + "l" + "lo" = "hello"
        let text = simple_decode(&[0, 1, 2, 3], &vocab);
        assert_eq!(text, "hello");
    }

    // ========================================================================
    // D. simple_decode — edge cases
    // ========================================================================

    /// Empty token list produces empty string
    #[test]
    fn test_simple_decode_empty_input() {
        let vocab = test_vocab();
        let text = simple_decode(&[], &vocab);
        assert_eq!(text, "");
    }

    /// Out-of-range token IDs produce "?"
    #[test]
    fn test_simple_decode_out_of_range() {
        let vocab = test_vocab();
        let text = simple_decode(&[99], &vocab);
        assert_eq!(text, "?");
    }

    /// Mix of valid and out-of-range IDs
    #[test]
    fn test_simple_decode_mixed_valid_invalid() {
        let vocab = test_vocab();
        let text = simple_decode(&[4, 100, 6], &vocab);
        assert_eq!(text, "hello?world");
    }

    /// Empty vocab means all tokens are out-of-range
    #[test]
    fn test_simple_decode_empty_vocab() {
        let vocab: Vec<String> = vec![];
        let text = simple_decode(&[0, 1, 2], &vocab);
        assert_eq!(text, "???");
    }

    /// Token ID u32::MAX is out-of-range
    #[test]
    fn test_simple_decode_max_token_id() {
        let vocab = test_vocab();
        let text = simple_decode(&[u32::MAX], &vocab);
        assert_eq!(text, "?");
    }

    /// Decode with empty-string tokens in vocab
    #[test]
    fn test_simple_decode_empty_string_token() {
        let vocab = vec!["".to_string(), "a".to_string()];
        let text = simple_decode(&[0, 1, 0], &vocab);
        // "" + "a" + "" = "a"
        assert_eq!(text, "a");
    }

    // ========================================================================
    // E. Round-trip consistency: encode then decode
    // ========================================================================

    /// Encoding then decoding should recover the original text when all
    /// characters exist in the vocab
    #[test]
    fn test_roundtrip_encode_decode() {
        let vocab = test_vocab();
        let original = "hello world";
        let tokens = simple_encode(original, &vocab);
        let recovered = simple_decode(&tokens, &vocab);
        assert_eq!(recovered, original);
    }

    /// Round-trip with single-char vocab covering ASCII subset
    #[test]
    fn test_roundtrip_single_char_vocab() {
        let vocab: Vec<String> = (b'a'..=b'z').map(|c| String::from(c as char)).collect();
        let original = "hello";
        let tokens = simple_encode(original, &vocab);
        let recovered = simple_decode(&tokens, &vocab);
        assert_eq!(recovered, original);
    }

    /// Round-trip fails gracefully when text has characters not in vocab:
    /// encode drops them, so decode produces a shorter string
    #[test]
    fn test_roundtrip_with_unknown_chars() {
        let vocab = test_vocab();
        let original = "hello!world";
        let tokens = simple_encode(original, &vocab);
        let recovered = simple_decode(&tokens, &vocab);
        // '!' is not in vocab, so it's dropped by encode
        assert_eq!(recovered, "helloworld");
    }

    // ========================================================================
    // F. Encode stability: same input always produces same output
    // ========================================================================

    #[test]
    fn test_encode_deterministic() {
        let vocab = test_vocab();
        let tokens1 = simple_encode("hello world", &vocab);
        let tokens2 = simple_encode("hello world", &vocab);
        assert_eq!(tokens1, tokens2);
    }

    // ========================================================================
    // G. Vocab ordering: first match at same length should win (by iteration)
    // ========================================================================

    /// When two vocab entries have the same length and both match at the
    /// same position, the one with the higher index wins (because the
    /// loop updates best_match on `>` not `>=`)
    #[test]
    fn test_encode_same_length_first_wins() {
        // Both "ab" entries have length 2, but `>` means only strictly longer
        // replaces, so id=0 ("ab" first) is kept over id=1 ("ab" second).
        // However since they're identical strings, both match and the last
        // one checked does NOT replace (same length, not strictly greater).
        let vocab = vec!["ab".to_string(), "ab".to_string()];
        let tokens = simple_encode("ab", &vocab);
        // First match (id=0) wins because second match has same length (not >)
        assert_eq!(tokens, vec![0]);
    }
}
