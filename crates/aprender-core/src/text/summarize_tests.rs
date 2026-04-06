pub(crate) use super::*;

#[test]
fn test_tfidf_summarization() {
    let text = "Machine learning is great. It solves many problems. \
                Deep learning is a subset of machine learning. \
                Neural networks are powerful tools.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
    let summary = summarizer.summarize(text).expect("should succeed");

    assert_eq!(summary.len(), 2);
}

#[test]
fn test_textrank_summarization() {
    let text = "First sentence about AI. Second sentence about ML. \
                Third sentence about AI and ML. Fourth unrelated sentence.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2);
    let summary = summarizer.summarize(text).expect("should succeed");

    assert_eq!(summary.len(), 2);
}

#[test]
fn test_hybrid_summarization() {
    let text = "Alpha beta gamma. Delta epsilon zeta. \
                Eta theta iota. Kappa lambda mu.";

    let summarizer = TextSummarizer::new(SummarizationMethod::Hybrid, 2);
    let summary = summarizer.summarize(text).expect("should succeed");

    assert_eq!(summary.len(), 2);
}

#[test]
fn test_empty_text() {
    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
    let summary = summarizer.summarize("").expect("should succeed");

    assert_eq!(summary.len(), 0);
}

#[test]
fn test_fewer_sentences_than_max() {
    let text = "Only one sentence here.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 5);
    let summary = summarizer.summarize(text).expect("should succeed");

    assert_eq!(summary.len(), 1);
}

#[test]
fn test_sentence_order_preserved() {
    let text = "First. Second. Third. Fourth. Fifth.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 3);
    let summary = summarizer.summarize(text).expect("should succeed");

    // Check that sentences appear in original order
    for i in 0..summary.len().saturating_sub(1) {
        let idx1 = text.find(&summary[i]).expect("sentence should exist");
        let idx2 = text.find(&summary[i + 1]).expect("sentence should exist");
        assert!(idx1 < idx2, "Sentences should maintain original order");
    }
}

// ====================================================================
// Additional coverage tests for uncovered branches
// ====================================================================

#[test]
fn test_with_damping_factor() {
    let text = "First sentence about topic. Second sentence about topic. \
                Third sentence different. Fourth sentence topic again.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2).with_damping_factor(0.5);
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 2);
}

#[test]
fn test_with_max_iterations() {
    let text = "Alpha beta gamma. Delta epsilon zeta. \
                Eta theta iota. Kappa lambda mu.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2).with_max_iterations(5);
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 2);
}

#[test]
fn test_normalize_equal_scores() {
    // When all scores are the same, normalize should return 0.5 for each
    let scores = vec![3.0, 3.0, 3.0, 3.0];
    let normalized = TextSummarizer::normalize(&scores);
    for val in &normalized {
        assert!(
            (*val - 0.5).abs() < 1e-10,
            "Equal scores should normalize to 0.5"
        );
    }
}

#[test]
fn test_normalize_empty() {
    let scores: Vec<f64> = Vec::new();
    let normalized = TextSummarizer::normalize(&scores);
    assert!(normalized.is_empty());
}

#[test]
fn test_normalize_two_values() {
    let scores = vec![0.0, 1.0];
    let normalized = TextSummarizer::normalize(&scores);
    assert!((normalized[0] - 0.0).abs() < 1e-10);
    assert!((normalized[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_summarization_method_debug() {
    let debug_tr = format!("{:?}", SummarizationMethod::TextRank);
    let debug_tf = format!("{:?}", SummarizationMethod::TfIdf);
    let debug_hy = format!("{:?}", SummarizationMethod::Hybrid);

    assert!(debug_tr.contains("TextRank"));
    assert!(debug_tf.contains("TfIdf"));
    assert!(debug_hy.contains("Hybrid"));
}

#[test]
fn test_summarization_method_clone_copy_eq() {
    let m1 = SummarizationMethod::TextRank;
    let m2 = m1; // Copy
    let m3 = m1.clone(); // Clone
    assert_eq!(m1, m2);
    assert_eq!(m1, m3);
    assert_ne!(SummarizationMethod::TextRank, SummarizationMethod::TfIdf);
}

#[test]
fn test_summarizer_debug() {
    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
    let debug = format!("{:?}", summarizer);
    assert!(debug.contains("TextSummarizer"));
}

#[test]
fn test_textrank_early_convergence() {
    // With very few iterations and high convergence threshold,
    // TextRank should converge quickly
    let text = "Same sentence. Same sentence. Same sentence. Same sentence.";

    let summarizer =
        TextSummarizer::new(SummarizationMethod::TextRank, 2).with_max_iterations(1000);
    let summary = summarizer.summarize(text).expect("should succeed");
    // All sentences are identical, so any 2 can be selected
    assert_eq!(summary.len(), 2);
}

#[test]
fn test_tfidf_with_empty_sentence_tokens() {
    // A sentence that becomes empty after tokenization (only punctuation)
    let text = "Hello world. !!! ???. This is a sentence. Another one here.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 2);
}

#[test]
fn test_hybrid_with_many_sentences() {
    let text = "Sentence one about AI. Sentence two about ML. \
                Sentence three about AI and ML. Sentence four about data. \
                Sentence five about AI models. Sentence six conclusion.";

    let summarizer = TextSummarizer::new(SummarizationMethod::Hybrid, 3);
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 3);
}

#[test]
fn test_split_sentences_with_exclamation_and_question() {
    let text = "First! Second? Third.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 3);
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 3);
}

#[test]
fn test_two_sentence_input_one_output() {
    let text = "First sentence. Second sentence.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 1);
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 1);
}

#[test]
fn test_select_top_sentences_ordering() {
    let scores = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let top2 = TextSummarizer::select_top_sentences(&scores, 2);
    // Should select indices of highest scores: index 3 (0.9) and index 1 (0.5)
    assert!(top2.contains(&3));
    assert!(top2.contains(&1));
}

#[test]
fn test_textrank_no_overlap_sentences() {
    // Sentences with no word overlap (similarity matrix is all zeros)
    let text = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 2);
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 2);
}

#[test]
fn test_exactly_max_sentences() {
    // Exactly max_sentences count -- should return all
    let text = "First. Second.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 2);
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 2);
}

#[test]
fn test_builder_chain() {
    let summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 3)
        .with_damping_factor(0.75)
        .with_max_iterations(50);

    let text = "One about topic. Two about topic. Three topic. Four topic. Five topic.";
    let summary = summarizer.summarize(text).expect("should succeed");
    assert_eq!(summary.len(), 3);
}
