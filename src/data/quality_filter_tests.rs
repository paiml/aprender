//! Tests for data quality filtering (GH-453)

use super::*;

// ── Length Scoring ──────────────────────────────────────────

#[test]
fn test_score_length_within_range() {
    let config = QualityConfig::default();
    let report = score_quality("This is a normal length sentence for testing.", &config);
    let length_score = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Length)
        .unwrap();
    assert!((length_score.score - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_score_length_too_short() {
    let config = QualityConfig::default().with_length_bounds(100, 10000);
    let report = score_quality("Short", &config);
    let length_score = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Length)
        .unwrap();
    assert!(length_score.score < 1.0);
    assert!(!length_score.passed);
}

#[test]
fn test_score_length_too_long() {
    let config = QualityConfig::default().with_length_bounds(1, 10);
    let report = score_quality("This is a very long text that exceeds the limit", &config);
    let length_score = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Length)
        .unwrap();
    assert!(length_score.score < 1.0);
}

// ── Diversity Scoring ──────────────────────────────────────

#[test]
fn test_score_diversity_high() {
    let config = QualityConfig::default();
    let report = score_quality(
        "The quick brown fox jumps over the lazy dog near a peaceful river.",
        &config,
    );
    let div = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Diversity)
        .unwrap();
    assert!(div.score > 0.5);
}

#[test]
fn test_score_diversity_low() {
    let config = QualityConfig::default().with_min_diversity(0.8);
    let report = score_quality("the the the the the the the the the the", &config);
    let div = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Diversity)
        .unwrap();
    assert!(div.score < 0.5);
    assert!(!div.passed);
}

#[test]
fn test_score_diversity_empty() {
    let config = QualityConfig::default();
    let report = score_quality("", &config);
    let div = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Diversity)
        .unwrap();
    assert!((div.score - 0.0).abs() < f32::EPSILON);
}

// ── Repetition Scoring ─────────────────────────────────────

#[test]
fn test_score_repetition_no_repeats() {
    let config = QualityConfig::default();
    let report = score_quality(
        "Each word here is completely unique and different from the rest.",
        &config,
    );
    let rep = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Repetition)
        .unwrap();
    assert!(rep.score > 0.8);
}

#[test]
fn test_score_repetition_high_repeats() {
    let config = QualityConfig::default();
    let report = score_quality(
        "the cat sat the cat sat the cat sat the cat sat the cat sat",
        &config,
    );
    let rep = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Repetition)
        .unwrap();
    assert!(rep.score < 0.8);
}

#[test]
fn test_score_repetition_short_text() {
    let config = QualityConfig::default();
    let report = score_quality("Hi there", &config);
    let rep = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Repetition)
        .unwrap();
    assert!((rep.score - 1.0).abs() < f32::EPSILON);
}

// ── Structure Scoring ──────────────────────────────────────

#[test]
fn test_score_structure_good() {
    let config = QualityConfig::default();
    let report = score_quality(
        "This is a well-structured sentence, with proper punctuation.",
        &config,
    );
    let st = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Structure)
        .unwrap();
    assert!(st.score >= 0.8);
}

#[test]
fn test_score_structure_poor() {
    let config = QualityConfig::default();
    let report = score_quality("no cap no punct", &config);
    let st = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Structure)
        .unwrap();
    assert!(st.score < 0.5);
}

// ── Language Scoring ───────────────────────────────────────

#[test]
fn test_score_language_ascii() {
    let config = QualityConfig::default();
    let report = score_quality("Pure ASCII text with no special characters.", &config);
    let lang = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Language)
        .unwrap();
    assert!((lang.score - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_score_language_mixed() {
    let config = QualityConfig::default();
    let report = score_quality("Mixed: こんにちは world", &config);
    let lang = report
        .scores
        .iter()
        .find(|s| s.dimension == QualityDimension::Language)
        .unwrap();
    assert!(lang.score < 1.0);
}

// ── Overall Quality ────────────────────────────────────────

#[test]
fn test_quality_good_text() {
    let config = QualityConfig::default();
    let report = score_quality(
        "Machine learning involves training models on data, evaluating their performance, and iterating on improvements.",
        &config,
    );
    assert!(report.passed);
    assert!(report.aggregate > 0.5);
}

#[test]
fn test_quality_poor_text() {
    let config = QualityConfig::default().with_min_quality(0.8);
    let report = score_quality("bad", &config);
    assert!(!report.passed);
}

// ── Batch Filtering ────────────────────────────────────────

#[test]
fn test_filter_by_quality() {
    let config = QualityConfig::default();
    let texts = vec![
        "This is a well-written, properly structured sentence.".to_string(),
        "x".to_string(),
        "Another high-quality sample with good structure and diversity.".to_string(),
    ];
    let (passed, rejected) = filter_by_quality(&texts, &config);
    assert_eq!(passed.len() + rejected, texts.len());
    assert!(rejected >= 1); // "x" should be rejected
}

#[test]
fn test_filter_by_quality_empty() {
    let config = QualityConfig::default();
    let (passed, rejected) = filter_by_quality(&[], &config);
    assert!(passed.is_empty());
    assert_eq!(rejected, 0);
}

// ── JSONL Filtering ────────────────────────────────────────

#[test]
fn test_filter_jsonl_quality() {
    let config = QualityConfig::default();
    let input = r#"{"text": "This is a good quality training sample, with proper structure."}
{"text": "x"}
{"text": "Another well-written sample for the dataset, covering diverse topics."}"#;

    let (output, stats) = filter_jsonl_quality(input, &config);
    assert_eq!(stats.total, 3);
    assert!(stats.passed >= 1);
    assert!(stats.rejected >= 1);
    assert!(!output.is_empty());
}

#[test]
fn test_filter_jsonl_invalid_json() {
    let config = QualityConfig::default();
    let input = "not json";
    let (output, stats) = filter_jsonl_quality(input, &config);
    assert!(output.is_empty());
    assert_eq!(stats.rejected, 1);
}

#[test]
fn test_filter_stats_pass_rate() {
    let stats = FilterStats {
        total: 10,
        passed: 7,
        rejected: 3,
        rejection_reasons: vec![],
    };
    assert!((stats.pass_rate() - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_filter_stats_pass_rate_zero() {
    let stats = FilterStats::default();
    assert!((stats.pass_rate() - 0.0).abs() < f32::EPSILON);
}

// ── Config ─────────────────────────────────────────────────

#[test]
fn test_config_default() {
    let config = QualityConfig::default();
    assert!((config.min_quality - 0.5).abs() < f32::EPSILON);
    assert_eq!(config.min_length, 10);
    assert_eq!(config.max_length, 100_000);
}

#[test]
fn test_config_with_min_quality_clamped() {
    let config = QualityConfig::default().with_min_quality(1.5);
    assert!((config.min_quality - 1.0).abs() < f32::EPSILON);

    let config = QualityConfig::default().with_min_quality(-0.5);
    assert!((config.min_quality - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_config_with_length_bounds() {
    let config = QualityConfig::default().with_length_bounds(50, 5000);
    assert_eq!(config.min_length, 50);
    assert_eq!(config.max_length, 5000);
}

#[test]
fn test_config_with_length_bounds_inverted() {
    let config = QualityConfig::default().with_length_bounds(100, 10);
    assert_eq!(config.max_length, 100); // max >= min enforced
}

// ── Falsification Tests ────────────────────────────────────

/// FALSIFY-PII-004: Quality filter rejects garbage
#[test]
fn falsify_quality_rejects_garbage() {
    let config = QualityConfig::default().with_min_quality(0.5);
    let garbage_texts = ["", "x", "aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa"];
    for text in &garbage_texts {
        let report = score_quality(text, &config);
        assert!(!report.passed, "Garbage should be rejected: {text:?}");
    }
}

/// FALSIFY-PII-005: Quality filter passes good text
#[test]
fn falsify_quality_passes_good() {
    let config = QualityConfig::default();
    let good_texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Implement a binary search tree with insert, delete, and search operations.",
        "Machine learning pipelines require careful data preprocessing and feature engineering.",
    ];
    for text in &good_texts {
        let report = score_quality(text, &config);
        assert!(
            report.passed,
            "Good text should pass: {text:?} (score={:.2})",
            report.aggregate
        );
    }
}

/// FALSIFY-EVOL-003: Filter + PII pipeline compatible
#[test]
fn falsify_quality_and_pii_composable() {
    let config = QualityConfig::default();
    let input = r#"{"text": "Contact user@example.com for high-quality ML training data."}"#;

    // Quality filter first
    let (filtered, stats) = filter_jsonl_quality(input, &config);
    assert_eq!(stats.passed, 1);

    // Then PII filter
    let (pii_filtered, pii_count) = crate::data::pii::filter_pii_jsonl(&filtered);
    assert_eq!(pii_count, 1);
    assert!(pii_filtered.contains("[EMAIL]"));
}
