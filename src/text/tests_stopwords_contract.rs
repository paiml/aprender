// =========================================================================
// FALSIFY-SW: Stopwords contract (aprender text)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-SW-* tests for stopwords
//   Why 2: stopwords tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for stopwords yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Stopword filtering was "obviously correct" (set membership)
//
// References:
//   - van Rijsbergen (1979) "Information Retrieval"
// =========================================================================

use super::*;

/// FALSIFY-SW-001: Filtering removes stop words and keeps content words
#[test]
fn falsify_sw_001_removes_stop_words() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["the", "quick", "brown", "fox"];
    let filtered = filter.filter(&tokens).expect("filter succeeds");

    assert!(
        !filtered.iter().any(|w| *w == "the"),
        "FALSIFIED SW-001: 'the' was not removed"
    );
    assert!(
        filtered.iter().any(|w| *w == "quick"),
        "FALSIFIED SW-001: 'quick' was incorrectly removed"
    );
}

/// FALSIFY-SW-002: Filtering is case-insensitive
#[test]
fn falsify_sw_002_case_insensitive() {
    let filter = StopWordsFilter::english();

    assert!(
        filter.is_stop_word("THE"),
        "FALSIFIED SW-002: 'THE' not recognized as stop word"
    );
    assert!(
        filter.is_stop_word("The"),
        "FALSIFIED SW-002: 'The' not recognized as stop word"
    );
    assert!(
        filter.is_stop_word("the"),
        "FALSIFIED SW-002: 'the' not recognized as stop word"
    );
}

/// FALSIFY-SW-003: Filtered output is subset of input
#[test]
fn falsify_sw_003_output_subset_of_input() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["is", "this", "a", "test", "of", "the", "system"];
    let filtered = filter.filter(&tokens).expect("filter succeeds");

    for word in &filtered {
        assert!(
            tokens.iter().any(|t| *t == word.as_str()),
            "FALSIFIED SW-003: filtered contains '{}' not in input",
            word
        );
    }
}

/// FALSIFY-SW-004: Filtered output length <= input length
#[test]
fn falsify_sw_004_output_not_longer() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["the", "cat", "is", "on", "the", "mat"];
    let filtered = filter.filter(&tokens).expect("filter succeeds");

    assert!(
        filtered.len() <= tokens.len(),
        "FALSIFIED SW-004: filtered len={} > input len={}",
        filtered.len(),
        tokens.len()
    );
}

mod sw_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-SW-004-prop: Filtered output <= input length
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_sw_004_prop_output_not_longer(
            seed in 0..500u32,
        ) {
            let words_pool = ["the", "is", "a", "cat", "dog", "run", "fast", "on", "in", "big"];
            let n = ((seed % 6) + 3) as usize;
            let tokens: Vec<&str> = (0..n).map(|i| words_pool[(i + seed as usize) % words_pool.len()]).collect();

            let filter = StopWordsFilter::english();
            let filtered = filter.filter(&tokens).expect("filter");
            prop_assert!(
                filtered.len() <= tokens.len(),
                "FALSIFIED SW-004-prop: filtered len {} > input len {}",
                filtered.len(), tokens.len()
            );
        }
    }

    /// FALSIFY-SW-003-prop: Filtered output is subset of input
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_sw_003_prop_subset(
            seed in 0..500u32,
        ) {
            let words_pool = ["the", "is", "a", "cat", "dog", "run", "fast", "on", "in", "big"];
            let n = ((seed % 6) + 3) as usize;
            let tokens: Vec<&str> = (0..n).map(|i| words_pool[(i + seed as usize) % words_pool.len()]).collect();

            let filter = StopWordsFilter::english();
            let filtered = filter.filter(&tokens).expect("filter");
            for word in &filtered {
                prop_assert!(
                    tokens.iter().any(|t| *t == word.as_str()),
                    "FALSIFIED SW-003-prop: '{}' not in input",
                    word
                );
            }
        }
    }
}
