proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // NLP: Summarization properties
    #[test]
    fn summarization_respects_max_sentences(
        n_sentences in 5usize..15,
        max_summary in 1usize..5
    ) {
        use aprender::text::summarize::{TextSummarizer, SummarizationMethod};

        // Create text with n_sentences
        let sentences: Vec<String> = (0..n_sentences)
            .map(|i| format!("Sentence number {i} with some content here"))
            .collect();
        let text = sentences.join(". ");

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, max_summary);
        let summary = summarizer.summarize(&text).expect("Should succeed");

        // Summary should have at most max_summary sentences
        prop_assert!(summary.len() <= max_summary.min(n_sentences));
    }

    #[test]
    fn summarization_preserves_sentence_order(
        n_sentences in 5usize..10
    ) {
        use aprender::text::summarize::{TextSummarizer, SummarizationMethod};

        // Create text with numbered sentences
        let sentences: Vec<String> = (0..n_sentences)
            .map(|i| format!("Important sentence number {i} with content"))
            .collect();
        let text = sentences.join(". ");

        let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 3);
        let summary = summarizer.summarize(&text).expect("Should succeed");

        // Sentences in summary should maintain original order
        for i in 0..summary.len().saturating_sub(1) {
            // Each sentence should appear before the next one in original text
            let pos1 = text.find(&summary[i]);
            let pos2 = text.find(&summary[i + 1]);
            if let (Some(p1), Some(p2)) = (pos1, pos2) {
                prop_assert!(p1 < p2);
            }
        }
    }

    #[test]
    fn summarization_empty_text_returns_empty(
        method in prop::sample::select(vec![
            aprender::text::summarize::SummarizationMethod::TfIdf,
            aprender::text::summarize::SummarizationMethod::TextRank,
            aprender::text::summarize::SummarizationMethod::Hybrid,
        ])
    ) {
        use aprender::text::summarize::TextSummarizer;

        let summarizer = TextSummarizer::new(method, 3);
        let summary = summarizer.summarize("").expect("Should succeed");

        prop_assert!(summary.is_empty());
    }

    // HNSW property tests
    #[test]
    fn hnsw_search_returns_k_results(
        vectors in proptest::collection::vec(vector_f64_strategy(5), 10..20),
        k in 1usize..5
    ) {
        use aprender::index::hnsw::HNSWIndex;

        let mut index = HNSWIndex::new(8, 100, 0.0);

        // Add vectors to index
        for (i, vec) in vectors.iter().enumerate() {
            index.add(format!("item{i}"), vec.clone());
        }

        // Search with first vector as query
        let query = &vectors[0];
        let results = index.search(query, k);

        // Should return at most k results
        prop_assert!(results.len() <= k);
        // Should return at least 1 result (since index is not empty)
        prop_assert!(!results.is_empty());
    }

    #[test]
    fn hnsw_distances_are_non_negative(
        vectors in proptest::collection::vec(vector_f64_strategy(5), 5..10),
        k in 1usize..3
    ) {
        use aprender::index::hnsw::HNSWIndex;

        let mut index = HNSWIndex::new(8, 100, 0.0);

        for (i, vec) in vectors.iter().enumerate() {
            index.add(format!("item{i}"), vec.clone());
        }

        let query = &vectors[0];
        let results = index.search(query, k);

        // All distances should be non-negative
        for (_, dist) in results {
            prop_assert!(dist >= 0.0, "Distance should be non-negative");
        }
    }

    #[test]
    fn hnsw_search_is_deterministic(
        vectors in proptest::collection::vec(vector_f64_strategy(5), 5..10)
    ) {
        use aprender::index::hnsw::HNSWIndex;

        // Note: HNSW construction uses randomness, so this test verifies
        // that search results are consistent for a given index, not that
        // the index construction is deterministic

        let mut index = HNSWIndex::new(8, 100, 42.0);

        for (i, vec) in vectors.iter().enumerate() {
            index.add(format!("item{i}"), vec.clone());
        }

        let query = &vectors[0];
        let results1 = index.search(query, 3);
        let results2 = index.search(query, 3);

        // Same query on same index should return same results
        prop_assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            prop_assert_eq!(&r1.0, &r2.0, "Item IDs should match");
            prop_assert!((r1.1 - r2.1).abs() < 1e-10, "Distances should match");
        }
    }

    #[test]
    fn hnsw_cosine_distance_bounds(
        vectors in proptest::collection::vec(vector_f64_strategy(5), 5..10)
    ) {
        use aprender::index::hnsw::HNSWIndex;

        let mut index = HNSWIndex::new(8, 100, 0.0);

        // Add all vectors (handle zeros by checking distances are valid)
        for (i, vec) in vectors.iter().enumerate() {
            index.add(format!("item{i}"), vec.clone());
        }

        let query = &vectors[0];
        let results = index.search(query, 3);

        // Cosine distance should be in [0, 2] or infinity for zero vectors
        // 0: identical direction, 1: orthogonal, 2: opposite direction
        for (_, dist) in results {
            // Distance should be non-negative and either finite in [0,2] or infinity
            prop_assert!(dist >= 0.0, "Distance should be non-negative, got {}", dist);
            if dist.is_finite() {
                prop_assert!(dist <= 2.0, "Finite cosine distance should be <= 2, got {}", dist);
            }
        }
    }

    // Incremental IDF property tests
    #[test]
    fn idf_monotonicity(
        terms1 in proptest::collection::vec("[a-z]{3,8}", 1..10),
        terms2 in proptest::collection::vec("[a-z]{3,8}", 1..10)
    ) {
        use aprender::text::incremental_idf::IncrementalIDF;

        let mut idf = IncrementalIDF::new(0.95);

        // Add first document
        let refs1: Vec<&str> = terms1.iter().map(String::as_str).collect();
        idf.update(&refs1);

        // Add second document
        let refs2: Vec<&str> = terms2.iter().map(String::as_str).collect();
        idf.update(&refs2);

        // Terms appearing in both documents should have lower IDF than unique terms
        let common_terms: Vec<_> = terms1.iter()
            .filter(|t| terms2.contains(t))
            .collect();

        let unique_to_1: Vec<_> = terms1.iter()
            .filter(|t| !terms2.contains(t))
            .collect();

        // If we have both common and unique terms
        if !common_terms.is_empty() && !unique_to_1.is_empty() {
            let common_idf = idf.idf(common_terms[0]);
            let unique_idf = idf.idf(unique_to_1[0]);

            // Unique term (appears in fewer docs) should have higher IDF
            prop_assert!(unique_idf >= common_idf,
                "Unique term IDF ({}) should be >= common term IDF ({})",
                unique_idf, common_idf);
        }
    }

    #[test]
    fn idf_decay_reduces_frequency(
        terms in proptest::collection::vec("[a-z]{3,8}", 1..5),
        n_updates in 5usize..15
    ) {
        use aprender::text::incremental_idf::IncrementalIDF;

        let mut idf = IncrementalIDF::new(0.9); // Stronger decay for testing

        // Add initial document with terms
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        idf.update(&refs);

        let initial_total = idf.total_docs();

        // Add many documents without the original terms
        for _ in 0..n_updates {
            idf.update(&["newterm"]);
        }

        let final_total = idf.total_docs();

        // Total docs should have increased
        prop_assert!(final_total > initial_total,
            "Total docs should increase: {} -> {}", initial_total, final_total);

        // With decay, total shouldn't grow linearly
        prop_assert!(final_total < initial_total + n_updates as f64,
            "Decay should prevent linear growth: {} < {}",
            final_total, initial_total + n_updates as f64);
    }

    #[test]
    fn idf_all_values_positive(
        documents in proptest::collection::vec(
            proptest::collection::vec("[a-z]{3,8}", 1..10),
            1..10
        )
    ) {
        use aprender::text::incremental_idf::IncrementalIDF;

        let mut idf = IncrementalIDF::new(0.95);

        for doc in &documents {
            let refs: Vec<&str> = doc.iter().map(String::as_str).collect();
            idf.update(&refs);
        }

        // All IDF values should be positive
        for term in idf.terms().keys() {
            let idf_val = idf.idf(term);
            prop_assert!(idf_val > 0.0, "IDF for '{}' should be positive, got {}", term, idf_val);
        }
    }

    // Content-Based Recommender property tests
    #[test]
    fn recommender_returns_at_most_k_results(
        items in proptest::collection::vec(
            proptest::collection::vec("[a-z]{3,8}", 2..10),
            5..15
        ),
        k in 1usize..5
    ) {
        use aprender::recommend::ContentRecommender;

        let mut rec = ContentRecommender::new(8, 100, 0.95);

        // Add items
        for (i, words) in items.iter().enumerate() {
            let content = words.join(" ");
            rec.add_item(format!("item{i}"), content);
        }

        // Get recommendations for first item
        if let Ok(results) = rec.recommend("item0", k) {
            // Should return at most k results (excluding query item)
            prop_assert!(results.len() <= k, "Should return at most {} results, got {}", k, results.len());
        }
    }

    #[test]
    fn recommender_size_increases_with_items(
        items in proptest::collection::vec("[a-z]{3,8}", 1..10)
    ) {
        use aprender::recommend::ContentRecommender;

        let mut rec = ContentRecommender::new(8, 100, 0.95);

        for (i, word) in items.iter().enumerate() {
            let initial_len = rec.len();
            rec.add_item(format!("item{i}"), word.clone());
            let new_len = rec.len();

            // Size should increase by exactly 1
            prop_assert_eq!(new_len, initial_len + 1, "Size should increase by 1");
        }

        // Final size should match number of items
        prop_assert_eq!(rec.len(), items.len());
    }

    #[test]
    fn recommender_handles_empty_content(
        num_items in 1usize..10
    ) {
        use aprender::recommend::ContentRecommender;

        let mut rec = ContentRecommender::new(8, 100, 0.95);

        // Add items with empty content
        for i in 0..num_items {
            rec.add_item(format!("item{i}"), "");
        }

        // Should not panic
        prop_assert_eq!(rec.len(), num_items);

        // Recommending should work (even if results are meaningless)
        let result = rec.recommend("item0", 2);
        prop_assert!(result.is_ok(), "Should handle empty content without error");
    }
}
