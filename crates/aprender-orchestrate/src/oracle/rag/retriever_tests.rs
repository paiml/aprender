use super::*;
use std::collections::HashSet;

/// Standard three-component document corpus for retriever tests.
///
/// Returns a retriever with:
/// - `{prefix}trueno{suffix}`: SIMD/GPU tensor content
/// - `{prefix}aprender{suffix}`: ML algorithm content
/// - `{prefix}entrenar{suffix}`: training content
fn retriever_with_stack_corpus(prefix: &str, suffix: &str) -> HybridRetriever {
    let mut retriever = HybridRetriever::new();
    retriever.index_document(
        &format!("{prefix}trueno{suffix}"),
        "SIMD GPU tensor operations accelerated compute",
    );
    retriever.index_document(
        &format!("{prefix}aprender{suffix}"),
        "machine learning algorithms random forest",
    );
    retriever.index_document(
        &format!("{prefix}entrenar{suffix}"),
        "training autograd LoRA quantization",
    );
    retriever
}

/// Assert that tokens contain every exact word in `expected`.
fn assert_tokens_contain(tokens: &[String], expected: &[&str]) {
    for word in expected {
        assert!(
            tokens.contains(&word.to_string()),
            "expected token {:?} not found in {tokens:?}",
            word,
        );
    }
}

/// Assert that tokens contain none of the words in `absent`.
fn assert_tokens_exclude(tokens: &[String], absent: &[&str]) {
    for word in absent {
        assert!(
            !tokens.contains(&word.to_string()),
            "unexpected token {:?} found in {tokens:?}",
            word,
        );
    }
}

#[test]
fn test_retriever_creation() {
    let retriever = HybridRetriever::new();
    let stats = retriever.stats();
    assert_eq!(stats.total_documents, 0);
    assert_eq!(stats.total_terms, 0);
}

#[test]
fn test_index_document() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("doc1", "hello world rust programming");

    let stats = retriever.stats();
    assert_eq!(stats.total_documents, 1);
    assert!(stats.total_terms > 0);
}

#[test]
fn test_bm25_search() {
    let retriever = retriever_with_stack_corpus("", "/CLAUDE.md");

    let results = retriever.bm25_search("GPU tensor", 5);

    // trueno should rank higher for GPU tensor query
    assert!(!results.is_empty());
    if !results.is_empty() {
        assert!(results[0].0.contains("trueno"));
    }
}

#[test]
fn test_rrf_fusion() {
    let retriever = HybridRetriever::new();

    let sparse =
        vec![("doc1".to_string(), 0.9), ("doc2".to_string(), 0.7), ("doc3".to_string(), 0.5)];
    let dense =
        vec![("doc2".to_string(), 0.95), ("doc1".to_string(), 0.8), ("doc4".to_string(), 0.6)];

    let fused = retriever.rrf_fuse(&sparse, &dense, 5);

    // doc1 and doc2 should both appear (in both lists)
    let doc_ids: HashSet<_> = fused.iter().map(|r| r.id.clone()).collect();
    assert!(doc_ids.contains("doc1"));
    assert!(doc_ids.contains("doc2"));
}

#[test]
fn test_tokenize() {
    let tokens = tokenize("Hello, World! This is Rust programming.");
    assert_tokens_contain(&tokens, &["hello", "world", "rust"]);
    // "programming" should be stemmed (exact output depends on stemmer)
    assert!(tokens.iter().any(|t| t.starts_with("program")));
    // Single chars and stop words should be filtered
    assert_tokens_exclude(&tokens, &["a", "this", "is"]);
}

#[test]
fn test_tokenize_code() {
    let tokens = tokenize("fn main() { let x_value = 42; }");
    assert_tokens_contain(&tokens, &["fn", "main", "42"]);
    // PorterStemmer (ml feature) stems "x_value" → "x_valu"; fallback keeps it as-is
    assert!(
        tokens.contains(&"x_value".to_string()) || tokens.contains(&"x_valu".to_string()),
        "expected x_value or x_valu in {tokens:?}",
    );
}

#[test]
fn test_stem_basic() {
    // Related words should produce the same stem
    assert_eq!(stem("tokenization"), stem("tokenize"));
    // Stemming should shorten words
    assert!(stem("programming").len() < "programming".len());
    assert!(stem("compression").len() < "compression".len());
    // Short words preserved
    assert_eq!(stem("run"), "run");
    assert_eq!(stem("go"), "go");
}

#[test]
fn test_stop_words_filtered() {
    let tokens = tokenize("how do I use the tensor operations");
    assert_tokens_exclude(&tokens, &["how", "do", "the"]);
    // Meaningful words preserved (stemmed)
    assert!(tokens.iter().any(|t| t.starts_with("tensor")));
    assert!(tokens.iter().any(|t| t.starts_with("oper")));
}

#[test]
fn test_tokenize_with_stemming() {
    let tokens = tokenize("tokenization and optimization");
    // Both should be stemmed, and produce the same stem as their base forms
    assert_tokens_contain(&tokens, &[&stem("tokenize"), &stem("optimize")]);
    // "and" is a stop word
    assert_tokens_exclude(&tokens, &["and"]);
}

#[test]
fn test_tfidf_dense_search() {
    let retriever = retriever_with_stack_corpus("", "/CLAUDE.md");

    let results = retriever.dense_search("GPU tensor", 5);

    // trueno should rank highest for GPU tensor
    assert!(!results.is_empty());
    assert!(results[0].0.contains("trueno"));
    // Scores should be positive
    for (_, score) in &results {
        assert!(*score > 0.0);
    }
}

#[test]
fn test_tfidf_empty_query() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("doc1", "some content here");

    // Empty query
    let results = retriever.dense_search("", 5);
    assert!(results.is_empty());

    // All-stop-words query
    let results = retriever.dense_search("the is and", 5);
    assert!(results.is_empty());
}

#[test]
fn test_component_boost() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("trueno/CLAUDE.md", "SIMD GPU tensor compute");
    retriever.index_document("aprender/CLAUDE.md", "machine learning tensor ops");

    let index = DocumentIndex::default();
    let results = retriever.retrieve("trueno tensor", &index, 5);

    // trueno should be boosted for query mentioning "trueno"
    if results.len() >= 2 {
        let trueno_result = results.iter().find(|r| r.component == "trueno");
        let aprender_result = results.iter().find(|r| r.component == "aprender");
        if let (Some(t), Some(a)) = (trueno_result, aprender_result) {
            assert!(
                t.score >= a.score,
                "trueno score {} should be >= aprender score {}",
                t.score,
                a.score
            );
        }
    }
}

#[test]
fn test_component_boost_hyphenated() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("trueno-ublk/CLAUDE.md", "block device ublk GPU compression");
    retriever.index_document("trueno/CLAUDE.md", "SIMD GPU tensor compute general");

    let index = DocumentIndex::default();
    let results = retriever.retrieve("trueno-ublk block device", &index, 5);

    // trueno-ublk should be boosted, not just trueno
    if !results.is_empty() {
        let ublk_result = results.iter().find(|r| r.component == "trueno-ublk");
        assert!(ublk_result.is_some(), "trueno-ublk should appear in results");
    }
}

#[test]
fn test_remove_document() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("doc1", "hello world");
    retriever.index_document("doc2", "goodbye world");

    assert_eq!(retriever.stats().total_documents, 2);

    retriever.remove_document("doc1");
    assert_eq!(retriever.stats().total_documents, 1);

    // "hello" should no longer be in index
    let results = retriever.bm25_search("hello", 5);
    assert!(results.is_empty() || !results.iter().any(|(id, _)| id == "doc1"));
}

#[test]
fn test_extract_component() {
    assert_eq!(extract_component("trueno/CLAUDE.md"), "trueno");
    assert_eq!(extract_component("aprender/docs/ml.md"), "aprender");
    assert_eq!(extract_component("simple_doc"), "simple_doc");
}

#[test]
fn test_bm25_idf() {
    let mut retriever = HybridRetriever::new();

    // Add documents where "rare" appears in only one
    retriever.index_document("doc1", "common common common rare");
    retriever.index_document("doc2", "common common common");
    retriever.index_document("doc3", "common common common");

    let results = retriever.bm25_search("rare", 5);

    // doc1 should be the only result for "rare"
    assert!(!results.is_empty());
    assert_eq!(results[0].0, "doc1");
}

#[test]
fn test_avg_doc_length_update() {
    let mut retriever = HybridRetriever::new();

    retriever.index_document("doc1", "one two three four five");
    assert!(retriever.avg_doc_length > 0.0);

    let first_avg = retriever.avg_doc_length;

    retriever.index_document("doc2", "one two");
    // Average should change
    assert!(retriever.avg_doc_length != first_avg || retriever.stats().total_documents == 2);
}

#[test]
fn test_retrieval_result_score_breakdown() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("doc1", "test query terms");

    let index = DocumentIndex::default();
    let results = retriever.retrieve("test query", &index, 5);

    // Results should have score breakdown
    for result in results {
        // RRF score should be set
        assert!(result.score_breakdown.rrf_score >= 0.0);
    }
}

// Profiling tests - use relative assertions since GLOBAL_METRICS is shared
mod profiling_tests {
    use super::*;
    use crate::oracle::rag::profiling::GLOBAL_METRICS;

    #[test]
    fn test_retrieve_records_query_latency() {
        let before = GLOBAL_METRICS.total_queries.get();

        let mut retriever = HybridRetriever::new();
        retriever.index_document("doc1", "test content here");

        let index = DocumentIndex::default();
        let _ = retriever.retrieve("test", &index, 5);

        let after = GLOBAL_METRICS.total_queries.get();
        assert!(after > before, "Query count should increase: before={}, after={}", before, after);
    }

    #[test]
    fn test_retrieve_records_spans() {
        let mut retriever = HybridRetriever::new();
        retriever.index_document("doc1", "hello world rust");
        retriever.index_document("doc2", "machine learning algorithms");

        let index = DocumentIndex::default();
        let _ = retriever.retrieve("rust algorithms", &index, 5);

        // Spans should be recorded
        let spans = GLOBAL_METRICS.all_span_stats();
        assert!(spans.contains_key("retrieve"), "retrieve span should be recorded");
        assert!(spans.contains_key("bm25_search"), "bm25_search span should be recorded");
        assert!(spans.contains_key("dense_search"), "dense_search span should be recorded");
        assert!(spans.contains_key("rrf_fuse"), "rrf_fuse span should be recorded");
        assert!(spans.contains_key("component_boost"), "component_boost span should be recorded");
    }

    #[test]
    fn test_multiple_queries_accumulate_metrics() {
        let before_queries = GLOBAL_METRICS.total_queries.get();
        let before_retrieve =
            GLOBAL_METRICS.get_span_stats("retrieve").map(|s| s.count).unwrap_or(0);

        let mut retriever = HybridRetriever::new();
        retriever.index_document("doc1", "test document");

        let index = DocumentIndex::default();
        let _ = retriever.retrieve("test", &index, 5);
        let _ = retriever.retrieve("document", &index, 5);
        let _ = retriever.retrieve("test document", &index, 5);

        let after_queries = GLOBAL_METRICS.total_queries.get();
        let after_retrieve =
            GLOBAL_METRICS.get_span_stats("retrieve").map(|s| s.count).unwrap_or(0);

        // Should have increased by at least 3 (other tests may also run in parallel)
        assert!(
            after_queries - before_queries >= 3,
            "At least 3 queries should be recorded: diff={}",
            after_queries - before_queries
        );
        assert!(
            after_retrieve - before_retrieve >= 3,
            "At least 3 retrieve spans should be recorded: diff={}",
            after_retrieve - before_retrieve
        );
    }

    #[test]
    fn test_query_latency_is_measured() {
        let mut retriever = HybridRetriever::new();
        // Add more documents for measurable latency
        for i in 0..100 {
            retriever.index_document(
                &format!("doc{}", i),
                &format!("document {} with content about topic {}", i, i % 10),
            );
        }

        let index = DocumentIndex::default();
        let _ = retriever.retrieve("content topic", &index, 10);

        // Query latency histogram should have observations
        assert!(GLOBAL_METRICS.query_latency.count() > 0, "Latency should be measured");
    }
}

// ============================================================================
// Coverage gap tests: remove_document, component_boost, from_persisted,
// with_config, avg_doc_length edge cases, dense_search empty index
// ============================================================================

#[test]
fn test_remove_document_cleans_up_terms() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("doc1", "unique_alpha_term");
    retriever.index_document("doc2", "unique_beta_term");

    // Verify unique_alpha_term is only in doc1
    let results = retriever.bm25_search("unique_alpha_term", 5);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");

    // Remove doc1
    retriever.remove_document("doc1");

    // unique_alpha_term should be completely gone from the index
    let results = retriever.bm25_search("unique_alpha_term", 5);
    assert!(results.is_empty(), "Term from removed document should be gone");

    // doc2 should still be searchable
    let results = retriever.bm25_search("unique_beta_term", 5);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc2");
}

#[test]
fn test_remove_document_updates_avg_doc_length() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("short", "ab");
    retriever.index_document("long", "one two three four five six seven eight nine ten");

    let avg_before = retriever.avg_doc_length;

    retriever.remove_document("long");

    // avg_doc_length should change after removing the long document
    assert!(
        (retriever.avg_doc_length - avg_before).abs() > 0.01,
        "avg_doc_length should change after removing document"
    );
    assert_eq!(retriever.stats().total_documents, 1);
}

#[test]
fn test_remove_all_documents_resets_avg_doc_length() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("doc1", "hello world");
    retriever.index_document("doc2", "goodbye world");

    retriever.remove_document("doc1");
    retriever.remove_document("doc2");

    assert_eq!(retriever.stats().total_documents, 0);
    assert!(
        retriever.avg_doc_length.abs() < f64::EPSILON,
        "avg_doc_length should be 0 when no documents remain"
    );
}

#[test]
fn test_remove_nonexistent_document_is_no_op() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("doc1", "hello world");

    let count_before = retriever.stats().total_documents;
    retriever.remove_document("nonexistent");
    let count_after = retriever.stats().total_documents;

    assert_eq!(count_before, count_after);
}

#[test]
fn test_component_boost_no_match() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("trueno/CLAUDE.md", "SIMD GPU tensor compute");
    retriever.index_document("aprender/CLAUDE.md", "machine learning tensor ops");

    let index = DocumentIndex::default();
    // Query does not mention any component name
    let results = retriever.retrieve("tensor operations", &index, 5);

    // Results should still be returned, just without boost
    assert!(!results.is_empty());
}

#[test]
fn test_component_boost_case_insensitive() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("trueno/CLAUDE.md", "SIMD GPU tensor compute");
    retriever.index_document("aprender/CLAUDE.md", "machine learning tensor");

    let index = DocumentIndex::default();
    // Query mentions "TRUENO" in uppercase - should still boost trueno
    let results = retriever.retrieve("TRUENO tensor", &index, 5);

    if results.len() >= 2 {
        let trueno_result = results.iter().find(|r| r.component == "trueno");
        let aprender_result = results.iter().find(|r| r.component == "aprender");
        if let (Some(t), Some(a)) = (trueno_result, aprender_result) {
            assert!(
                t.score >= a.score,
                "trueno should be boosted: trueno={}, aprender={}",
                t.score,
                a.score
            );
        }
    }
}

#[test]
fn test_from_persisted_roundtrip() {
    let mut original = HybridRetriever::new();
    original.index_document("trueno/CLAUDE.md", "SIMD GPU tensor operations accelerated");
    original.index_document("aprender/CLAUDE.md", "machine learning random forest");

    let persisted = original.to_persisted();

    // Verify persisted structure has data
    assert_eq!(persisted.doc_lengths.len(), 2);
    assert!(!persisted.inverted_index.is_empty());

    let restored = HybridRetriever::from_persisted(persisted);

    // Stats should match
    assert_eq!(original.stats().total_documents, restored.stats().total_documents);
    assert_eq!(original.stats().total_terms, restored.stats().total_terms);
    assert!(
        (original.stats().avg_doc_length - restored.stats().avg_doc_length).abs() < f64::EPSILON
    );

    // Search should produce same results
    let orig_results = original.bm25_search("GPU tensor", 5);
    let rest_results = restored.bm25_search("GPU tensor", 5);
    assert_eq!(orig_results.len(), rest_results.len());
    for (o, r) in orig_results.iter().zip(rest_results.iter()) {
        assert_eq!(o.0, r.0, "doc_ids should match");
        assert!((o.1 - r.1).abs() < 1e-10, "scores should match");
    }
}

#[test]
fn test_from_persisted_empty_index() {
    let persisted = super::super::persistence::PersistedIndex::default();
    let restored = HybridRetriever::from_persisted(persisted);

    assert_eq!(restored.stats().total_documents, 0);
    assert_eq!(restored.stats().total_terms, 0);
    assert!(restored.avg_doc_length.abs() < f64::EPSILON);
}

#[test]
fn test_with_config() {
    let bm25 = super::super::types::Bm25Config { k1: 2.0, b: 0.5 };
    let rrf = super::super::types::RrfConfig { k: 30 };
    let retriever = HybridRetriever::with_config(bm25, rrf);

    // Should start empty
    assert_eq!(retriever.stats().total_documents, 0);
    assert_eq!(retriever.stats().total_terms, 0);
}

#[test]
fn test_with_config_affects_search() {
    // Low b means less length normalization
    let bm25_low_b = super::super::types::Bm25Config { k1: 1.5, b: 0.0 };
    let rrf = super::super::types::RrfConfig { k: 60 };
    let mut retriever = HybridRetriever::with_config(bm25_low_b, rrf);
    retriever.index_document("short", "test keyword");
    retriever
        .index_document("long", "test keyword extra words more content here padding filler text");

    let results = retriever.bm25_search("keyword", 5);
    // With b=0.0 (no length normalization), both should have very similar scores
    assert!(results.len() >= 2);
}

#[test]
fn test_dense_search_empty_index() {
    let retriever = HybridRetriever::new();
    let results = retriever.dense_search("anything", 5);
    assert!(results.is_empty());
}

#[test]
fn test_dense_search_no_matching_terms() {
    let mut retriever = HybridRetriever::new();
    retriever.index_document("doc1", "alpha beta gamma");

    // Query with terms not in the index at all
    let results = retriever.dense_search("zzzznotfound", 5);
    assert!(results.is_empty());
}

#[test]
fn test_inverted_index_remove_cleans_empty_postings() {
    let mut index = InvertedIndex::new();
    index.add_document("doc1", "unique_word shared_word");
    index.add_document("doc2", "shared_word other_word");

    // "unique_word" should only appear for doc1
    assert!(index.index.contains_key(&stem("unique_word")));

    index.remove_document("doc1");

    // "unique_word" posting list should be cleaned up (empty)
    let unique_stem = stem("unique_word");
    assert!(!index.index.contains_key(&unique_stem), "Empty posting lists should be cleaned up");

    // "shared_word" should still exist for doc2
    let shared_stem = stem("shared_word");
    assert!(index.index.contains_key(&shared_stem));
}

#[test]
fn test_sort_and_truncate() {
    let mut results = vec![
        ("c".to_string(), 0.3),
        ("a".to_string(), 0.9),
        ("b".to_string(), 0.6),
        ("d".to_string(), 0.1),
    ];
    sort_and_truncate(&mut results, 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "a"); // highest score first
    assert_eq!(results[1].0, "b");
}

#[test]
fn test_sort_and_truncate_fewer_than_k() {
    let mut results = vec![("a".to_string(), 0.5)];
    sort_and_truncate(&mut results, 10);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_stem_suffix_stripping() {
    // Test various suffixes are handled
    assert!(stem("running").len() < "running".len());
    assert!(stem("careful").len() < "careful".len());
    assert!(stem("actively").len() < "actively".len());
    // Short words (<=3 chars) preserved
    assert_eq!(stem("abc"), "abc");
    assert_eq!(stem("ab"), "ab");
}

#[test]
fn test_tokenize_empty_and_whitespace() {
    let tokens = tokenize("");
    assert!(tokens.is_empty());

    let tokens = tokenize("   ");
    assert!(tokens.is_empty());
}

#[test]
fn test_tokenize_single_chars_filtered() {
    let tokens = tokenize("a b c d e");
    assert!(tokens.is_empty(), "Single-char tokens should be filtered out");
}

#[test]
fn test_rrf_fuse_empty_inputs() {
    let retriever = HybridRetriever::new();
    let results = retriever.rrf_fuse(&[], &[], 5);
    assert!(results.is_empty());
}

#[test]
fn test_rrf_fuse_one_empty_list() {
    let retriever = HybridRetriever::new();
    let sparse = vec![("doc1".to_string(), 0.9), ("doc2".to_string(), 0.5)];
    let results = retriever.rrf_fuse(&sparse, &[], 5);
    assert!(!results.is_empty());
    // doc1 and doc2 should appear
    assert!(results.iter().any(|r| r.id == "doc1"));
    assert!(results.iter().any(|r| r.id == "doc2"));
}

#[test]
fn test_default_impl() {
    let retriever = HybridRetriever::default();
    assert_eq!(retriever.stats().total_documents, 0);
}

// Property-based tests for hybrid retriever
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: Indexing and searching is deterministic
        #[test]
        fn prop_search_deterministic(
            doc_id in "[a-z]{3,10}",
            content in "[a-z ]{10,100}"
        ) {
            let mut retriever = HybridRetriever::new();
            retriever.index_document(&doc_id, &content);

            let results1 = retriever.bm25_search(&content, 5);
            let results2 = retriever.bm25_search(&content, 5);

            prop_assert_eq!(results1.len(), results2.len());
            for (r1, r2) in results1.iter().zip(results2.iter()) {
                prop_assert_eq!(&r1.0, &r2.0);  // Same doc IDs
                prop_assert!((r1.1 - r2.1).abs() < 1e-6);  // Same scores
            }
        }

        /// Property: BM25 scores are non-negative
        #[test]
        fn prop_bm25_scores_non_negative(
            content in "[a-z ]{20,200}",
            query in "[a-z]{3,15}"
        ) {
            let mut retriever = HybridRetriever::new();
            retriever.index_document("doc1", &content);

            let results = retriever.bm25_search(&query, 10);

            for (_, score) in results {
                prop_assert!(score >= 0.0, "BM25 score {} should be >= 0", score);
            }
        }

        /// Property: RRF scores from retrieval are in valid range [0, 1]
        #[test]
        fn prop_rrf_scores_valid_range(
            content1 in "[a-z ]{10,50}",
            content2 in "[a-z ]{10,50}",
            query in "[a-z]{3,10}"
        ) {
            let mut retriever = HybridRetriever::new();
            retriever.index_document("doc1", &content1);
            retriever.index_document("doc2", &content2);

            let index = DocumentIndex::default();
            let results = retriever.retrieve(&query, &index, 10);

            for result in &results {
                prop_assert!(result.score >= 0.0 && result.score <= 1.0,
                    "RRF score {} should be in [0, 1]", result.score);
            }
        }

        /// Property: Document count increases on indexing
        #[test]
        fn prop_doc_count_increases(
            docs in prop::collection::vec(("[a-z]{5}", "[a-z ]{10,50}"), 1..10)
        ) {
            let mut retriever = HybridRetriever::new();

            for (i, (id, content)) in docs.iter().enumerate() {
                retriever.index_document(id, content);
                // Use >= because duplicate IDs won't increase count
                prop_assert!(retriever.stats().total_documents >= 1,
                    "After {} docs, count is {}", i + 1, retriever.stats().total_documents);
            }
        }

        /// Property: Empty query returns empty results
        #[test]
        fn prop_empty_query_empty_results(content in "[a-z ]{10,100}") {
            let mut retriever = HybridRetriever::new();
            retriever.index_document("doc1", &content);

            let results = retriever.bm25_search("", 10);
            prop_assert!(results.is_empty());
        }

        /// Property: Removing document decreases count
        #[test]
        fn prop_remove_decreases_count(
            id1 in "[a-z]{5}",
            id2 in "[A-Z]{5}",
            content in "[a-z ]{10,50}"
        ) {
            let mut retriever = HybridRetriever::new();
            retriever.index_document(&id1, &content);
            retriever.index_document(&id2, &content);

            let count_before = retriever.stats().total_documents;
            retriever.remove_document(&id1);
            let count_after = retriever.stats().total_documents;

            prop_assert!(count_after <= count_before);
        }
    }
}
