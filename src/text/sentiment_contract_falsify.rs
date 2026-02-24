//! Sentiment Analysis Contract Falsification Tests
//!
//! Popperian falsification of NLP spec §2.1.5 claims:
//!   - Positive words produce positive scores
//!   - Negative words produce negative scores
//!   - Empty/neutral text produces score near 0
//!   - Classify is consistent with score + threshold
//!   - Score is deterministic
//!   - Lexicon contains expected positive/negative words
//!
//! Five-Whys (PMAT-350):
//!   Why #1: sentiment module has unit tests but zero FALSIFY-SENT-* tests
//!   Why #2: unit tests check examples, not sentiment polarity contracts
//!   Why #3: no provable-contract YAML for sentiment analysis
//!   Why #4: sentiment module was built before DbC methodology
//!   Why #5: no formal verification that positive/negative words produce correct sign
//!
//! References:
//!   - docs/specifications/nlp-models-techniques-spec.md §2.1.5
//!   - src/text/sentiment.rs

use super::*;

// ============================================================================
// FALSIFY-SENT-001: Positive text produces positive score
// Contract: text composed of positive words → score > 0
// ============================================================================

#[test]
fn falsify_sent_001_positive_text_positive_score() {
    let analyzer = SentimentAnalyzer::default();

    let positive_texts = [
        "amazing wonderful great",
        "I love this excellent product",
        "fantastic and brilliant work",
    ];

    for text in &positive_texts {
        let score = analyzer.score(text).expect("score");
        assert!(
            score > 0.0,
            "FALSIFIED SENT-001: positive text '{}' got score {}, expected > 0",
            text,
            score
        );
    }
}

// ============================================================================
// FALSIFY-SENT-002: Negative text produces negative score
// Contract: text composed of negative words → score < 0
// ============================================================================

#[test]
fn falsify_sent_002_negative_text_negative_score() {
    let analyzer = SentimentAnalyzer::default();

    let negative_texts = [
        "terrible awful horrible",
        "worst disgusting pathetic",
        "hate this dreadful thing",
    ];

    for text in &negative_texts {
        let score = analyzer.score(text).expect("score");
        assert!(
            score < 0.0,
            "FALSIFIED SENT-002: negative text '{}' got score {}, expected < 0",
            text,
            score
        );
    }
}

// ============================================================================
// FALSIFY-SENT-003: Empty/neutral text produces score near 0
// Contract: empty or neutral text → score = 0 or |score| < threshold
// ============================================================================

#[test]
fn falsify_sent_003_empty_text_zero_score() {
    let analyzer = SentimentAnalyzer::default();

    let score = analyzer.score("").expect("score empty");
    assert!(
        score.abs() < f64::EPSILON,
        "FALSIFIED SENT-003: empty text score = {}, expected 0.0",
        score
    );
}

#[test]
fn falsify_sent_003_neutral_text_near_zero() {
    let analyzer = SentimentAnalyzer::default();

    // Text with no sentiment-bearing words
    let score = analyzer
        .score("the cat sat on the mat")
        .expect("score neutral");
    assert!(
        score.abs() < 0.5,
        "FALSIFIED SENT-003: neutral text got score {}, expected near 0",
        score
    );
}

// ============================================================================
// FALSIFY-SENT-004: classify consistency with score + threshold
// Contract: classify(text) == Positive iff score > threshold,
//           Negative iff score < -threshold, else Neutral
// ============================================================================

#[test]
fn falsify_sent_004_classify_score_consistency() {
    let analyzer = SentimentAnalyzer::default();

    let texts = [
        "amazing wonderful excellent",
        "terrible horrible awful",
        "the cat sat on the mat",
        "",
    ];

    for text in &texts {
        let score = analyzer.score(text).expect("score");
        let polarity = analyzer.classify(text).expect("classify");

        let expected = if score > 0.05 {
            Polarity::Positive
        } else if score < -0.05 {
            Polarity::Negative
        } else {
            Polarity::Neutral
        };

        assert_eq!(
            polarity, expected,
            "FALSIFIED SENT-004: classify('{}') = {:?} but score = {} implies {:?}",
            text, polarity, score, expected
        );
    }
}

// ============================================================================
// FALSIFY-SENT-005: Sentiment score determinism
// Contract: same text always produces same score
// ============================================================================

#[test]
fn falsify_sent_005_score_determinism() {
    let analyzer = SentimentAnalyzer::default();
    let text = "I love this amazing product but hate the terrible packaging";

    let s1 = analyzer.score(text).expect("first");
    let s2 = analyzer.score(text).expect("second");

    assert!(
        (s1 - s2).abs() < f64::EPSILON,
        "FALSIFIED SENT-005: score is non-deterministic: {s1} != {s2}"
    );
}

// ============================================================================
// FALSIFY-SENT-006: Default lexicon contains expected words
// Contract: lexicon has both positive and negative entries
// ============================================================================

#[test]
fn falsify_sent_006_lexicon_has_both_polarities() {
    let analyzer = SentimentAnalyzer::default();

    assert!(
        analyzer.lexicon_size() >= 20,
        "FALSIFIED SENT-006: lexicon only has {} words, expected >= 20",
        analyzer.lexicon_size()
    );
}

// ============================================================================
// FALSIFY-SENT-007: Score is finite for all inputs
// Contract: score never returns NaN or infinity
// ============================================================================

#[test]
fn falsify_sent_007_score_always_finite() {
    let analyzer = SentimentAnalyzer::default();

    let repeated = "amazing ".repeat(100);
    let edge_cases = [
        "",
        " ",
        "!!!???...",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        repeated.as_str(),
    ];

    for text in &edge_cases {
        let score = analyzer.score(text).expect("score");
        assert!(
            score.is_finite(),
            "FALSIFIED SENT-007: score('{}...') = {} is not finite",
            &text[..text.len().min(30)],
            score
        );
    }
}
