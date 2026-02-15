//! Unit tests for parse_tokenizer_json (pure JSON parsing, no filesystem I/O).
//!
//! Extracted from load_tokenizer_from_json to enable deterministic testing
//! of the tokenizer parsing logic with synthetic JSON inputs.

use super::super::parse_tokenizer_json;

// ============================================================================
// 1. Basic vocab extraction from model.vocab
// ============================================================================

#[test]
fn test_basic_vocab_extraction() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"hello": 0, "world": 1, "<unk>": 2}
        }
    });

    let result = parse_tokenizer_json(&json, None);
    assert!(result.is_some(), "should parse valid tokenizer JSON");

    let tok = result.expect("already checked is_some");
    assert_eq!(tok.vocabulary.len(), 3);
    assert_eq!(tok.vocabulary[0], "hello");
    assert_eq!(tok.vocabulary[1], "world");
    assert_eq!(tok.vocabulary[2], "<unk>");
}

#[test]
fn test_vocab_ids_are_respected() {
    // Non-contiguous IDs: 0, 5, 10
    let json = serde_json::json!({
        "model": {
            "vocab": {"alpha": 0, "beta": 5, "gamma": 10}
        }
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    // Final size = max_id + 1 = 11
    assert_eq!(tok.vocabulary.len(), 11);
    assert_eq!(tok.vocabulary[0], "alpha");
    assert_eq!(tok.vocabulary[5], "beta");
    assert_eq!(tok.vocabulary[10], "gamma");
    // Gaps filled with <unk>
    assert_eq!(tok.vocabulary[1], "<unk>");
    assert_eq!(tok.vocabulary[7], "<unk>");
}

// ============================================================================
// 2. Added tokens merged into vocabulary
// ============================================================================

#[test]
fn test_added_tokens_merged() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"hello": 0, "world": 1}
        },
        "added_tokens": [
            {"content": "<s>", "id": 2},
            {"content": "</s>", "id": 3}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(tok.vocabulary.len(), 4);
    assert_eq!(tok.vocabulary[0], "hello");
    assert_eq!(tok.vocabulary[1], "world");
    assert_eq!(tok.vocabulary[2], "<s>");
    assert_eq!(tok.vocabulary[3], "</s>");
}

#[test]
fn test_added_tokens_override_base_vocab() {
    // added_tokens should overwrite base vocab at the same ID
    let json = serde_json::json!({
        "model": {
            "vocab": {"placeholder": 5, "hello": 0}
        },
        "added_tokens": [
            {"content": "<special>", "id": 5}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(
        tok.vocabulary[5], "<special>",
        "added_tokens should override base vocab"
    );
    assert_eq!(tok.vocabulary[0], "hello");
}

// ============================================================================
// 3. BOS/EOS token detection from config.json
// ============================================================================

#[test]
fn test_bos_eos_from_config_json() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0, "b": 1, "<bos>": 2, "<eos>": 3}
        }
    });

    let config = serde_json::json!({
        "bos_token_id": 2,
        "eos_token_id": 3
    });

    let tok = parse_tokenizer_json(&json, Some(&config)).expect("should parse");
    assert_eq!(tok.bos_token_id, Some(2));
    assert_eq!(tok.eos_token_id, Some(3));
}

#[test]
fn test_config_json_bos_eos_takes_priority_over_added_tokens() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0}
        },
        "added_tokens": [
            {"content": "<s>", "id": 10},
            {"content": "</s>", "id": 11}
        ]
    });

    let config = serde_json::json!({
        "bos_token_id": 99,
        "eos_token_id": 100
    });

    let tok = parse_tokenizer_json(&json, Some(&config)).expect("should parse");
    // config.json values should win over added_tokens fallback
    assert_eq!(tok.bos_token_id, Some(99));
    assert_eq!(tok.eos_token_id, Some(100));
}

// ============================================================================
// 4. BOS/EOS fallback from added_tokens patterns
// ============================================================================

#[test]
fn test_bos_fallback_from_added_tokens_s_tag() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"x": 0}
        },
        "added_tokens": [
            {"content": "<s>", "id": 1}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(tok.bos_token_id, Some(1), "<s> should be detected as BOS");
}

#[test]
fn test_eos_fallback_from_added_tokens_s_close_tag() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"x": 0}
        },
        "added_tokens": [
            {"content": "</s>", "id": 2}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(tok.eos_token_id, Some(2), "</s> should be detected as EOS");
}

#[test]
fn test_eos_fallback_from_eot_id_pattern() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"x": 0}
        },
        "added_tokens": [
            {"content": "<|eot_id|>", "id": 42}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(
        tok.eos_token_id,
        Some(42),
        "<|eot_id|> should be detected as EOS"
    );
}

#[test]
fn test_bos_fallback_from_startoftext_pattern() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"x": 0}
        },
        "added_tokens": [
            {"content": "<|startoftext|>", "id": 7}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(
        tok.bos_token_id,
        Some(7),
        "<|startoftext|> should be detected as BOS"
    );
}

#[test]
fn test_bos_fallback_from_content_containing_bos() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"x": 0}
        },
        "added_tokens": [
            {"content": "<|bos_token|>", "id": 55}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(
        tok.bos_token_id,
        Some(55),
        "token containing 'bos' should be detected as BOS"
    );
}

#[test]
fn test_eos_fallback_from_content_containing_eos() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"x": 0}
        },
        "added_tokens": [
            {"content": "<|eos_token|>", "id": 56}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(
        tok.eos_token_id,
        Some(56),
        "token containing 'eos' should be detected as EOS"
    );
}

#[test]
fn test_no_bos_eos_without_matching_patterns() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"hello": 0, "world": 1}
        },
        "added_tokens": [
            {"content": "<pad>", "id": 2},
            {"content": "<mask>", "id": 3}
        ]
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(tok.bos_token_id, None, "no BOS pattern matched");
    assert_eq!(tok.eos_token_id, None, "no EOS pattern matched");
}

// ============================================================================
// 5. BPE merge rules extraction
// ============================================================================

#[test]
fn test_bpe_merges_extracted() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
            "merges": ["h e", "l l", "he llo"]
        }
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(tok.merges.len(), 3);
    assert_eq!(tok.merges[0], "h e");
    assert_eq!(tok.merges[1], "l l");
    assert_eq!(tok.merges[2], "he llo");
}

#[test]
fn test_no_merges_field_yields_empty_vec() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0, "b": 1}
        }
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert!(
        tok.merges.is_empty(),
        "missing merges field should yield empty vec"
    );
}

#[test]
fn test_empty_merges_array() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0},
            "merges": []
        }
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert!(
        tok.merges.is_empty(),
        "empty merges array should yield empty vec"
    );
}

// ============================================================================
// 6. Missing model.vocab returns None
// ============================================================================

#[test]
fn test_missing_model_key_returns_none() {
    let json = serde_json::json!({
        "not_model": {}
    });

    assert!(
        parse_tokenizer_json(&json, None).is_none(),
        "missing 'model' key should return None"
    );
}

#[test]
fn test_missing_vocab_key_returns_none() {
    let json = serde_json::json!({
        "model": {
            "merges": ["a b"]
        }
    });

    assert!(
        parse_tokenizer_json(&json, None).is_none(),
        "missing 'vocab' key should return None"
    );
}

#[test]
fn test_vocab_not_object_returns_none() {
    let json = serde_json::json!({
        "model": {
            "vocab": "not_an_object"
        }
    });

    assert!(
        parse_tokenizer_json(&json, None).is_none(),
        "non-object vocab should return None"
    );
}

// ============================================================================
// 7. Empty vocabulary returns None
// ============================================================================

#[test]
fn test_empty_vocab_object_produces_single_unk() {
    let json = serde_json::json!({
        "model": {
            "vocab": {}
        }
    });

    // Empty vocab_map => token_to_id is empty => max_id defaults to 0,
    // expected_vocab_size defaults to 0 => final_size = max(0, 0+1) = 1
    // So vocabulary has 1 element "<unk>", which is NOT empty.
    // The function returns Some in this case since there IS a vocabulary
    // (just all <unk>). This matches the production behavior.
    let result = parse_tokenizer_json(&json, None);
    assert!(
        result.is_some(),
        "empty vocab still yields single <unk> entry"
    );
    let tok = result.expect("checked above");
    assert_eq!(tok.vocabulary.len(), 1);
    assert_eq!(tok.vocabulary[0], "<unk>");
}

// ============================================================================
// 8. Model type extraction
// ============================================================================

#[test]
fn test_model_type_extracted() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0},
            "type": "BPE"
        }
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(tok.model_type.as_deref(), Some("BPE"));
}

#[test]
fn test_model_type_none_when_missing() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0}
        }
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(tok.model_type, None);
}

#[test]
fn test_model_type_unigram() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0},
            "type": "Unigram"
        }
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(tok.model_type.as_deref(), Some("Unigram"));
}

include!("tokenizer_parse_part_02.rs");
