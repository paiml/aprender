// =========================================================================
// FALSIFY-ST: Stemming contract (aprender text)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-ST-* tests for stemming
//   Why 2: stemming tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for stemming yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Porter stemmer was "obviously correct" (textbook algorithm)
//
// References:
//   - Porter (1980) "An algorithm for suffix stripping"
// =========================================================================

use super::*;

/// FALSIFY-ST-001: Stemming is idempotent: stem(stem(w)) = stem(w)
#[test]
fn falsify_st_001_idempotent() {
    let stemmer = PorterStemmer::new();
    let words = ["running", "studies", "easily", "caresses", "relational"];

    for word in &words {
        let once = stemmer.stem(word).expect("stem succeeds");
        let twice = stemmer.stem(&once).expect("stem succeeds");
        assert_eq!(
            once, twice,
            "FALSIFIED ST-001: stem({word})={once}, stem({once})={twice} — not idempotent"
        );
    }
}

/// FALSIFY-ST-002: Stem output is never longer than input
#[test]
fn falsify_st_002_output_not_longer() {
    let stemmer = PorterStemmer::new();
    let words = [
        "running",
        "studies",
        "flies",
        "processing",
        "generalization",
    ];

    for word in &words {
        let stemmed = stemmer.stem(word).expect("stem succeeds");
        assert!(
            stemmed.len() <= word.len(),
            "FALSIFIED ST-002: stem({word})={stemmed} is longer ({} > {})",
            stemmed.len(),
            word.len()
        );
    }
}

/// FALSIFY-ST-003: Stem output is non-empty for non-empty input
#[test]
fn falsify_st_003_nonempty_output() {
    let stemmer = PorterStemmer::new();
    let words = ["a", "the", "running", "x"];

    for word in &words {
        let stemmed = stemmer.stem(word).expect("stem succeeds");
        assert!(
            !stemmed.is_empty(),
            "FALSIFIED ST-003: stem({word}) returned empty string"
        );
    }
}

mod st_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-ST-001-prop: Stemming is idempotent for random words
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_st_001_prop_idempotent(
            seed in 0..500u32,
        ) {
            let words = ["running", "studies", "easily", "caresses", "relational",
                         "flying", "happiness", "dogs", "played", "faster"];
            let word = words[seed as usize % words.len()];

            let stemmer = PorterStemmer::new();
            let once = stemmer.stem(word).expect("stem 1");
            let twice = stemmer.stem(&once).expect("stem 2");
            prop_assert_eq!(
                once, twice,
                "FALSIFIED ST-001-prop: not idempotent for '{}'",
                word
            );
        }
    }

    /// FALSIFY-ST-002-prop: Stem output not longer than input
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_st_002_prop_not_longer(
            seed in 0..500u32,
        ) {
            let words = ["running", "studies", "easily", "caresses", "relational",
                         "generalization", "processing", "flies", "playing", "connected"];
            let word = words[seed as usize % words.len()];

            let stemmer = PorterStemmer::new();
            let stemmed = stemmer.stem(word).expect("stem");
            prop_assert!(
                stemmed.len() <= word.len(),
                "FALSIFIED ST-002-prop: stem('{}')='{}' is longer",
                word, stemmed
            );
        }
    }
}
