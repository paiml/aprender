pub(crate) use super::*;

#[test]
fn test_positive_sentiment() {
    let analyzer = SentimentAnalyzer::new();
    let score = analyzer
        .score("This is great and wonderful")
        .expect("score should succeed");
    assert!(score > 0.0);
}

#[test]
fn test_negative_sentiment() {
    let analyzer = SentimentAnalyzer::new();
    let score = analyzer
        .score("This is terrible and awful")
        .expect("score should succeed");
    assert!(score < 0.0);
}

#[test]
fn test_neutral_sentiment() {
    let analyzer = SentimentAnalyzer::new();
    let score = analyzer
        .score("The product arrived on time")
        .expect("score should succeed");
    // Should be near zero (neutral)
    assert!(score.abs() < 0.5);
}

#[test]
fn test_polarity_classification() {
    let analyzer = SentimentAnalyzer::new();

    assert_eq!(
        analyzer
            .classify("amazing product")
            .expect("classify should succeed"),
        Polarity::Positive
    );
    assert_eq!(
        analyzer
            .classify("terrible experience")
            .expect("classify should succeed"),
        Polarity::Negative
    );
    assert_eq!(
        analyzer
            .classify("the item")
            .expect("classify should succeed"),
        Polarity::Neutral
    );
}

#[test]
fn test_empty_text() {
    let analyzer = SentimentAnalyzer::new();
    let score = analyzer.score("").expect("score should succeed");
    assert_eq!(score, 0.0);
}

#[test]
fn test_lexicon_size() {
    let analyzer = SentimentAnalyzer::new();
    assert!(analyzer.lexicon_size() > 50);
}

// ====================================================================
// Additional coverage tests for uncovered branches
// ====================================================================

#[test]
fn test_with_custom_lexicon() {
    let mut lexicon = HashMap::new();
    lexicon.insert("awesome".to_string(), 5.0);
    lexicon.insert("horrible".to_string(), -5.0);

    let analyzer = SentimentAnalyzer::with_lexicon(lexicon);
    assert_eq!(analyzer.lexicon_size(), 2);

    let score = analyzer.score("awesome").expect("score should succeed");
    assert!(score > 0.0);

    let score = analyzer.score("horrible").expect("score should succeed");
    assert!(score < 0.0);

    // Word not in custom lexicon should score 0
    let score = analyzer.score("banana").expect("score should succeed");
    assert_eq!(score, 0.0);
}

#[test]
fn test_with_neutral_threshold() {
    let analyzer = SentimentAnalyzer::new().with_neutral_threshold(10.0);

    // Even strong sentiment words should classify as neutral with high threshold
    let polarity = analyzer
        .classify("amazing")
        .expect("classify should succeed");
    assert_eq!(polarity, Polarity::Neutral);
}

#[test]
fn test_only_punctuation_text() {
    let analyzer = SentimentAnalyzer::new();
    // All characters are punctuation/whitespace, so tokens vec will be empty
    let score = analyzer.score("!!! ??? ...").expect("score should succeed");
    assert_eq!(score, 0.0);
}

#[test]
fn test_only_whitespace_text() {
    let analyzer = SentimentAnalyzer::new();
    let score = analyzer.score("   \t  \n  ").expect("score should succeed");
    assert_eq!(score, 0.0);
}

#[test]
fn test_classify_neutral_boundary() {
    // Use a custom lexicon with a very small score near the threshold boundary
    let mut lexicon = HashMap::new();
    lexicon.insert("mildly".to_string(), 0.01);

    let analyzer = SentimentAnalyzer::with_lexicon(lexicon).with_neutral_threshold(0.05);

    // Score is 0.01 / 1 = 0.01, which is below 0.05 threshold
    let polarity = analyzer
        .classify("mildly")
        .expect("classify should succeed");
    assert_eq!(polarity, Polarity::Neutral);
}

#[test]
fn test_polarity_debug() {
    let debug_pos = format!("{:?}", Polarity::Positive);
    let debug_neg = format!("{:?}", Polarity::Negative);
    let debug_neu = format!("{:?}", Polarity::Neutral);

    assert!(debug_pos.contains("Positive"));
    assert!(debug_neg.contains("Negative"));
    assert!(debug_neu.contains("Neutral"));
}

#[test]
fn test_polarity_clone_copy_eq() {
    let p1 = Polarity::Positive;
    let p2 = p1; // Copy
    let p3 = p1.clone(); // Clone
    assert_eq!(p1, p2);
    assert_eq!(p1, p3);
    assert_ne!(Polarity::Positive, Polarity::Negative);
}

#[test]
fn test_analyzer_debug() {
    let analyzer = SentimentAnalyzer::new();
    let debug = format!("{:?}", analyzer);
    assert!(debug.contains("SentimentAnalyzer"));
}

#[test]
fn test_default_trait_impl() {
    let a1 = SentimentAnalyzer::new();
    let a2 = SentimentAnalyzer::default();
    // Both should have the same lexicon size
    assert_eq!(a1.lexicon_size(), a2.lexicon_size());
}

#[test]
fn test_mixed_sentiment_text() {
    let analyzer = SentimentAnalyzer::new();
    // Text with both positive and negative words
    let score = analyzer
        .score("great terrible")
        .expect("score should succeed");
    // great=2.0, terrible=-3.0, normalized = -1.0/2 = -0.5
    assert!(score < 0.0);
}

#[test]
fn test_case_insensitivity() {
    let analyzer = SentimentAnalyzer::new();
    let score_lower = analyzer.score("great").expect("score should succeed");
    let score_upper = analyzer.score("GREAT").expect("score should succeed");
    assert_eq!(score_lower, score_upper);
}

#[test]
fn test_score_with_unicode_text() {
    let analyzer = SentimentAnalyzer::new();
    // Unicode text not in lexicon should yield 0
    let score = analyzer
        .score("こんにちは 世界")
        .expect("score should succeed");
    assert_eq!(score, 0.0);
}

#[test]
fn test_empty_custom_lexicon() {
    let analyzer = SentimentAnalyzer::with_lexicon(HashMap::new());
    assert_eq!(analyzer.lexicon_size(), 0);

    let score = analyzer
        .score("anything here")
        .expect("score should succeed");
    assert_eq!(score, 0.0);
}
