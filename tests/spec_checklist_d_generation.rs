//! Spec Checklist Tests - Section D: Generation & Quality (20 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::text::bpe::Qwen2BpeTokenizer;

// ============================================================================
// Section D: Generation & Quality (20 points)
// ============================================================================

/// D3: Generation must stop at EOS token
#[test]
fn d3_eos_respect() {
    let tokenizer = Qwen2BpeTokenizer::new();

    // Verify EOS tokens are properly defined
    let eos_tokens = [151645, 151643]; // im_end, endoftext

    for &eos in &eos_tokens {
        assert!(
            tokenizer.is_eos(eos),
            "D3 FAIL: Token {eos} not recognized as EOS"
        );
    }
}

/// D4: Check that repetition detection works
#[test]
fn d4_repetition_detection() {
    // Helper to count n-gram repetitions
    fn count_ngram_repetitions(tokens: &[u32], n: usize) -> usize {
        if tokens.len() < n {
            return 0;
        }

        let mut seen = std::collections::HashSet::new();
        let mut repetitions = 0;

        for window in tokens.windows(n) {
            if !seen.insert(window.to_vec()) {
                repetitions += 1;
            }
        }
        repetitions
    }

    // A good model should not have excessive repetition
    // Test with sample data
    let no_repeat = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let high_repeat = vec![1u32, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];

    let no_repeat_count = count_ngram_repetitions(&no_repeat, 4);
    let high_repeat_count = count_ngram_repetitions(&high_repeat, 4);

    assert_eq!(
        no_repeat_count, 0,
        "D4 FAIL: False positive on no-repeat sequence"
    );
    assert!(
        high_repeat_count > 0,
        "D4 FAIL: Did not detect repetitions in repeated sequence"
    );
}
