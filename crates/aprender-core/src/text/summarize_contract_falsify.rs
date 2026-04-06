//! Text Summarization Contract Falsification Tests
//!
//! Popperian falsification of NLP spec §2.1.8 claims:
//!   - Summary length ≤ max_sentences
//!   - Summary sentences are subsets of original text
//!   - Summarization is deterministic (same input → same output)
//!   - Empty input produces empty summary
//!   - Short text (≤ max_sentences) returns all sentences
//!
//! Five-Whys (PMAT-353):
//!   Why #1: summarize module has unit tests but zero FALSIFY-SUM-* tests
//!   Why #2: unit tests check examples, not extractive summarization contracts
//!   Why #3: no provable-contract YAML for text summarization
//!   Why #4: summarize module was built before DbC methodology
//!   Why #5: no systematic verification that summary is a true subset of input
//!
//! References:
//!   - Mihalcea & Tarau (2004). TextRank: Bringing Order into Text.
//!   - docs/specifications/nlp-models-techniques-spec.md §2.1.8
//!   - src/text/summarize.rs

use super::*;

// ============================================================================
// FALSIFY-SUM-001: Summary length bound
// Contract: summary.len() ≤ max_sentences
// ============================================================================

#[test]
fn falsify_sum_001_summary_length_bound() {
    let text = "First sentence about machine learning. \
                Second sentence about deep learning. \
                Third sentence about neural networks. \
                Fourth sentence about transformers. \
                Fifth sentence about attention mechanisms.";

    for max in 1..=3 {
        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, max);
        let summary = summarizer.summarize(text).expect("summarize");

        assert!(
            summary.len() <= max,
            "FALSIFIED SUM-001: summary has {} sentences, max was {}",
            summary.len(),
            max
        );
    }
}

// ============================================================================
// FALSIFY-SUM-002: Summary is subset of original
// Contract: every summary sentence appears in the original text (extractive)
// ============================================================================

#[test]
fn falsify_sum_002_extractive_subset_tfidf() {
    let text = "Machine learning trains models. \
                Deep learning uses neural networks. \
                Transformers use attention mechanisms. \
                GPT generates text tokens.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
    let summary = summarizer.summarize(text).expect("summarize");

    for sent in &summary {
        assert!(
            text.contains(sent.trim()),
            "FALSIFIED SUM-002: summary sentence '{}' not found in original text",
            sent
        );
    }
}

#[test]
fn falsify_sum_002_extractive_subset_textrank() {
    let text = "Machine learning trains models. \
                Deep learning uses neural networks. \
                Transformers use attention mechanisms. \
                GPT generates text tokens.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2);
    let summary = summarizer.summarize(text).expect("summarize");

    for sent in &summary {
        assert!(
            text.contains(sent.trim()),
            "FALSIFIED SUM-002: TextRank sentence '{}' not found in original text",
            sent
        );
    }
}

// ============================================================================
// FALSIFY-SUM-003: Empty input
// Contract: summarize("") returns empty summary
// ============================================================================

#[test]
fn falsify_sum_003_empty_input() {
    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
    let summary = summarizer.summarize("").expect("summarize empty");

    assert!(
        summary.is_empty(),
        "FALSIFIED SUM-003: empty input produced {} sentences",
        summary.len()
    );
}

// ============================================================================
// FALSIFY-SUM-004: Short text passthrough
// Contract: text with ≤ max_sentences sentences returns all sentences
// ============================================================================

#[test]
fn falsify_sum_004_short_text_passthrough() {
    let text = "First sentence. Second sentence.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 5); // max=5, but only 2 sentences
    let summary = summarizer.summarize(text).expect("summarize");

    assert_eq!(
        summary.len(),
        2,
        "FALSIFIED SUM-004: short text with 2 sentences returned {} sentences (max=5)",
        summary.len()
    );
}

// ============================================================================
// FALSIFY-SUM-005: Summarization determinism
// Contract: same input + same method → same summary
// ============================================================================

#[test]
fn falsify_sum_005_determinism() {
    let text = "First sentence about machine learning. \
                Second sentence about deep learning. \
                Third sentence about neural networks. \
                Fourth sentence about transformers.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);

    let s1 = summarizer.summarize(text).expect("first");
    let s2 = summarizer.summarize(text).expect("second");

    assert_eq!(
        s1, s2,
        "FALSIFIED SUM-005: summarization is non-deterministic"
    );
}

// ============================================================================
// FALSIFY-SUM-006: All methods produce valid summaries
// Contract: TextRank, TfIdf, and Hybrid all produce non-empty summaries
// ============================================================================

#[test]
fn falsify_sum_006_all_methods_produce_output() {
    let text = "Machine learning is powerful. \
                It solves complex problems. \
                Deep learning extends machine learning. \
                Neural networks are the foundation.";

    let methods = [
        SummarizationMethod::TfIdf,
        SummarizationMethod::TextRank,
        SummarizationMethod::Hybrid,
    ];

    for method in &methods {
        let summarizer = TextSummarizer::new(*method, 2);
        let summary = summarizer.summarize(text).expect("summarize");

        assert!(
            !summary.is_empty(),
            "FALSIFIED SUM-006: {:?} produced empty summary for non-empty text",
            method
        );
    }
}
