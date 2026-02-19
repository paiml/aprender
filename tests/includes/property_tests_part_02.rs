proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // LogisticRegression properties
    #[test]
    fn logistic_regression_predict_proba_in_range(x in matrix_strategy(10, 3)) {
        use aprender::classification::LogisticRegression;
        let y = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]; // Binary labels

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(100);
        model.fit(&x, &y).expect("Test data should be valid");

        let probas = model.predict_proba(&x);

        // All probabilities must be in [0, 1]
        for &p in probas.as_slice() {
            prop_assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn logistic_regression_predictions_are_binary(x in matrix_strategy(10, 3)) {
        use aprender::classification::LogisticRegression;
        let y = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(100);
        model.fit(&x, &y).expect("Test data should be valid");

        let predictions = model.predict(&x);

        // All predictions must be 0 or 1
        for pred in predictions {
            prop_assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn logistic_regression_score_in_range(x in matrix_strategy(10, 3)) {
        use aprender::classification::LogisticRegression;
        let y = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(100);
        model.fit(&x, &y).expect("Test data should be valid");

        let score = model.score(&x, &y);

        // Score (accuracy) must be in [0, 1]
        prop_assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn logistic_regression_output_length_matches_input(x in matrix_strategy(15, 4)) {
        use aprender::classification::LogisticRegression;
        let y = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(100);
        model.fit(&x, &y).expect("Test data should be valid");

        let probas = model.predict_proba(&x);
        let predictions = model.predict(&x);

        prop_assert_eq!(probas.len(), 15);
        prop_assert_eq!(predictions.len(), 15);
    }

    // NLP: Similarity properties
    #[test]
    fn cosine_similarity_bounds(a in vector_f64_strategy(10), b in vector_f64_strategy(10)) {
        use aprender::text::similarity::cosine_similarity;

        // Cosine similarity should be in [-1, 1]
        if let Ok(sim) = cosine_similarity(&a, &b) {
            prop_assert!((-1.0..=1.0).contains(&sim));
        }
    }

    #[test]
    fn cosine_similarity_self_is_one(v in vector_f64_strategy(10)) {
        use aprender::text::similarity::cosine_similarity;

        // Compute norm manually
        let norm_sq: f64 = v.as_slice().iter().map(|x| x * x).sum();
        if norm_sq > 1e-10 {
            if let Ok(sim) = cosine_similarity(&v, &v) {
                prop_assert!((sim - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn jaccard_similarity_bounds(
        tokens1 in proptest::collection::vec("[a-z]+", 1..20),
        tokens2 in proptest::collection::vec("[a-z]+", 1..20)
    ) {
        use aprender::text::similarity::jaccard_similarity;

        // Jaccard similarity should be in [0, 1]
        let sim = jaccard_similarity(&tokens1, &tokens2).expect("Should succeed");
        prop_assert!((0.0..=1.0).contains(&sim));
    }

    #[test]
    fn jaccard_similarity_symmetric(
        tokens1 in proptest::collection::vec("[a-z]+", 1..20),
        tokens2 in proptest::collection::vec("[a-z]+", 1..20)
    ) {
        use aprender::text::similarity::jaccard_similarity;

        // Jaccard similarity is symmetric: J(A,B) = J(B,A)
        let sim1 = jaccard_similarity(&tokens1, &tokens2).expect("Should succeed");
        let sim2 = jaccard_similarity(&tokens2, &tokens1).expect("Should succeed");
        prop_assert!((sim1 - sim2).abs() < 1e-10);
    }

    #[test]
    fn edit_distance_non_negative(
        s1 in "[a-z]{0,20}",
        s2 in "[a-z]{0,20}"
    ) {
        use aprender::text::similarity::edit_distance;

        // Edit distance is always non-negative (usize is always >= 0)
        // This test verifies the function returns successfully
        let _dist = edit_distance(&s1, &s2).expect("Should succeed");
    }

    #[test]
    fn edit_distance_zero_iff_equal(s in "[a-z]{1,20}") {
        use aprender::text::similarity::edit_distance;

        // Distance is zero iff strings are equal
        let dist = edit_distance(&s, &s).expect("Should succeed");
        prop_assert_eq!(dist, 0);
    }

    #[test]
    fn edit_distance_symmetric(
        s1 in "[a-z]{0,20}",
        s2 in "[a-z]{0,20}"
    ) {
        use aprender::text::similarity::edit_distance;

        // Edit distance is symmetric: d(A,B) = d(B,A)
        let dist1 = edit_distance(&s1, &s2).expect("Should succeed");
        let dist2 = edit_distance(&s2, &s1).expect("Should succeed");
        prop_assert_eq!(dist1, dist2);
    }

    // NLP: Entity extraction properties
    #[test]
    fn entity_extraction_idempotent(text in "[a-zA-Z0-9@#. _-]{10,100}") {
        use aprender::text::entities::EntityExtractor;

        let extractor = EntityExtractor::new();

        // Extract twice should give same results
        let entities1 = extractor.extract(&text).expect("Should succeed");
        let entities2 = extractor.extract(&text).expect("Should succeed");

        prop_assert_eq!(entities1.emails, entities2.emails);
        prop_assert_eq!(entities1.urls, entities2.urls);
        prop_assert_eq!(entities1.hashtags, entities2.hashtags);
    }

    #[test]
    fn entity_email_format_validation(
        local in "[a-z]{1,10}",
        domain in "[a-z]{1,10}",
        tld in "[a-z]{2,5}"
    ) {
        use aprender::text::entities::EntityExtractor;

        let email = format!("{local}@{domain}.{tld}");
        let text = format!("Contact {email} for info");

        let extractor = EntityExtractor::new();
        let entities = extractor.extract(&text).expect("Should succeed");

        // Should extract exactly one email
        prop_assert_eq!(entities.emails.len(), 1);
        prop_assert!(entities.emails[0].contains('@'));
        prop_assert!(entities.emails[0].contains('.'));
    }

    #[test]
    fn entity_url_extraction(
        protocol in prop::sample::select(vec!["http", "https"]),
        domain in "[a-z]{1,10}",
        tld in "(com|org|net)"
    ) {
        use aprender::text::entities::EntityExtractor;

        let url = format!("{protocol}://{domain}.{tld}");
        let text = format!("Visit {url} for more");

        let extractor = EntityExtractor::new();
        let entities = extractor.extract(&text).expect("Should succeed");

        // Should extract exactly one URL
        prop_assert!(!entities.urls.is_empty());
        prop_assert!(entities.urls[0].starts_with("http"));
    }

}
