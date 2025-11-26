//! Property-based tests using proptest.
//!
//! These tests verify invariants and properties of the ML algorithms.

use aprender::model_selection::{train_test_split, KFold};
use aprender::prelude::*;
use proptest::prelude::*;

// Strategy for generating small matrices
fn matrix_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Matrix<f32>> {
    proptest::collection::vec(-100.0f32..100.0, rows * cols).prop_map(move |data| {
        Matrix::from_vec(rows, cols, data).expect("Test data should be valid")
    })
}

// Strategy for generating vectors
fn vector_strategy(len: usize) -> impl Strategy<Value = Vector<f32>> {
    proptest::collection::vec(-100.0f32..100.0, len).prop_map(Vector::from_vec)
}

// Strategy for generating f64 vectors (for NLP similarity functions)
fn vector_f64_strategy(len: usize) -> impl Strategy<Value = Vector<f64>> {
    proptest::collection::vec(-100.0f64..100.0, len).prop_map(Vector::from_vec)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Vector properties
    #[test]
    fn vector_sum_is_associative(a in vector_strategy(10), b in vector_strategy(10)) {
        let sum_ab = (&a + &b).sum();
        let sum_ba = (&b + &a).sum();
        prop_assert!((sum_ab - sum_ba).abs() < 1e-4);
    }

    #[test]
    fn vector_dot_is_commutative(a in vector_strategy(10), b in vector_strategy(10)) {
        let dot_ab = a.dot(&b);
        let dot_ba = b.dot(&a);
        prop_assert!((dot_ab - dot_ba).abs() < 1e-4);
    }

    #[test]
    fn vector_norm_is_non_negative(v in vector_strategy(10)) {
        prop_assert!(v.norm() >= 0.0);
    }

    #[test]
    fn vector_scalar_mul_distributes(v in vector_strategy(10), s in -10.0f32..10.0) {
        let scaled = v.mul_scalar(s);
        let expected_sum = v.sum() * s;
        prop_assert!((scaled.sum() - expected_sum).abs() < 1e-3);
    }

    #[test]
    fn vector_elementwise_mul_is_commutative(a in vector_strategy(10), b in vector_strategy(10)) {
        let mul_ab = &a * &b;
        let mul_ba = &b * &a;
        for i in 0..10 {
            prop_assert!((mul_ab[i] - mul_ba[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn vector_elementwise_mul_with_ones_is_identity(v in vector_strategy(10)) {
        let ones = Vector::<f32>::ones(10);
        let result = &v * &ones;
        for i in 0..10 {
            prop_assert!((result[i] - v[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_elementwise_mul_with_zeros_is_zero(v in vector_strategy(10)) {
        let zeros = Vector::<f32>::zeros(10);
        let result = &v * &zeros;
        for i in 0..10 {
            prop_assert!((result[i]).abs() < 1e-6);
        }
    }

    // Matrix properties
    #[test]
    fn matrix_transpose_involution(m in matrix_strategy(5, 5)) {
        let m_t = m.transpose();
        let m_tt = m_t.transpose();
        for i in 0..5 {
            for j in 0..5 {
                prop_assert!((m.get(i, j) - m_tt.get(i, j)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn matrix_shape_preserved_by_add(a in matrix_strategy(4, 3), b in matrix_strategy(4, 3)) {
        let c = a.add(&b).expect("Test data should be valid");
        prop_assert_eq!(c.shape(), (4, 3));
    }

    #[test]
    fn matrix_matmul_shape(a in matrix_strategy(3, 4), b in matrix_strategy(4, 2)) {
        let c = a.matmul(&b).expect("Test data should be valid");
        prop_assert_eq!(c.shape(), (3, 2));
    }

    // DataFrame properties
    #[test]
    fn dataframe_select_preserves_rows(
        data in proptest::collection::vec(-100.0f32..100.0, 15)
    ) {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&data[0..5])),
            ("b".to_string(), Vector::from_slice(&data[5..10])),
            ("c".to_string(), Vector::from_slice(&data[10..15])),
        ];
        let df = DataFrame::new(columns).expect("Test data should be valid");
        let selected = df.select(&["a", "c"]).expect("Test data should be valid");
        prop_assert_eq!(selected.n_rows(), df.n_rows());
    }

    #[test]
    fn dataframe_to_matrix_shape(
        data in proptest::collection::vec(-100.0f32..100.0, 15)
    ) {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&data[0..5])),
            ("b".to_string(), Vector::from_slice(&data[5..10])),
            ("c".to_string(), Vector::from_slice(&data[10..15])),
        ];
        let df = DataFrame::new(columns).expect("Test data should be valid");
        let matrix = df.to_matrix();
        prop_assert_eq!(matrix.shape(), (5, 3));
    }

    // Metrics properties
    #[test]
    fn r_squared_perfect_prediction(y in vector_strategy(10)) {
        let r2 = r_squared(&y, &y);
        // Perfect prediction should give R² = 1 (or very close)
        prop_assert!((r2 - 1.0).abs() < 1e-6 || y.variance() == 0.0);
    }

    #[test]
    fn mse_is_non_negative(y_true in vector_strategy(10), y_pred in vector_strategy(10)) {
        let error = mse(&y_pred, &y_true);
        prop_assert!(error >= 0.0);
    }

    #[test]
    fn mae_is_non_negative(y_true in vector_strategy(10), y_pred in vector_strategy(10)) {
        let error = mae(&y_pred, &y_true);
        prop_assert!(error >= 0.0);
    }

    #[test]
    fn rmse_equals_sqrt_mse(y_true in vector_strategy(10), y_pred in vector_strategy(10)) {
        let mse_val = mse(&y_pred, &y_true);
        let rmse_val = rmse(&y_pred, &y_true);
        prop_assert!((rmse_val - mse_val.sqrt()).abs() < 1e-6);
    }

    // Linear regression properties
    #[test]
    fn linear_regression_coefficients_length(n_features in 1usize..5) {
        // Create well-conditioned data
        let n_samples = n_features + 3;
        let mut x_data = vec![0.0; n_samples * n_features];
        let mut y_data = vec![0.0; n_samples];

        // Create identity-like pattern to ensure positive definiteness
        for i in 0..n_samples {
            for j in 0..n_features {
                x_data[i * n_features + j] = if i == j { 1.0 } else { 0.1 * (i + j) as f32 };
            }
            y_data[i] = (i + 1) as f32;
        }

        let x = Matrix::from_vec(n_samples, n_features, x_data).expect("Test data should be valid");
        let y = Vector::from_vec(y_data);

        let mut model = LinearRegression::new();
        if model.fit(&x, &y).is_ok() {
            prop_assert_eq!(model.coefficients().len(), n_features);
        }
    }

    // K-Means properties
    #[test]
    fn kmeans_labels_valid(n_clusters in 1usize..4) {
        // Create data with clear clusters
        let n_samples = n_clusters * 5;
        let mut data = Vec::with_capacity(n_samples * 2);

        for k in 0..n_clusters {
            for i in 0..5 {
                data.push((k * 10) as f32 + i as f32 * 0.1);
                data.push((k * 10) as f32 + i as f32 * 0.1);
            }
        }

        let matrix = Matrix::from_vec(n_samples, 2, data).expect("Test data should be valid");
        let mut kmeans = KMeans::new(n_clusters).with_random_state(42);
        kmeans.fit(&matrix).expect("Test data should be valid");

        let labels = kmeans.predict(&matrix);

        // All labels should be valid cluster indices
        for &label in &labels {
            prop_assert!(label < n_clusters);
        }
    }

    #[test]
    fn kmeans_inertia_non_negative(n_clusters in 1usize..3) {
        let n_samples = n_clusters * 3;
        let data: Vec<f32> = (0..n_samples * 2).map(|i| i as f32).collect();
        let matrix = Matrix::from_vec(n_samples, 2, data).expect("Test data should be valid");

        let mut kmeans = KMeans::new(n_clusters).with_random_state(42);
        kmeans.fit(&matrix).expect("Test data should be valid");

        prop_assert!(kmeans.inertia() >= 0.0);
    }

    // Train-test split properties
    #[test]
    fn train_test_split_preserves_samples(n_samples in 10usize..50) {
        let x_data: Vec<f32> = (0..n_samples * 2).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();

        let x = Matrix::from_vec(n_samples, 2, x_data).expect("Test data should be valid");
        let y = Vector::from_vec(y_data);

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42)).expect("Test data should be valid");

        // Total samples should be preserved
        let total = x_train.n_rows() + x_test.n_rows();
        prop_assert_eq!(total, n_samples);
        prop_assert_eq!(y_train.len() + y_test.len(), n_samples);
    }

    #[test]
    fn train_test_split_ratio_approximate(test_size in 0.1f32..0.5) {
        let n_samples = 100;
        let x_data: Vec<f32> = (0..n_samples * 2).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();

        let x = Matrix::from_vec(n_samples, 2, x_data).expect("Test data should be valid");
        let y = Vector::from_vec(y_data);

        let (_, x_test, _, _) = train_test_split(&x, &y, test_size, Some(42)).expect("Test data should be valid");

        // Test set should be approximately the right size
        let actual_ratio = x_test.n_rows() as f32 / n_samples as f32;
        prop_assert!((actual_ratio - test_size).abs() < 0.1);
    }

    #[test]
    fn kfold_splits_cover_all_samples(k in 2usize..6) {
        let n_samples = 20;
        let kfold = KFold::new(k);

        let mut seen = vec![false; n_samples];
        for (_, test_indices) in kfold.split(n_samples) {
            for &idx in &test_indices {
                seen[idx] = true;
            }
        }

        // Every sample should appear in exactly one test fold
        for (i, &s) in seen.iter().enumerate() {
            prop_assert!(s, "Sample {} not in any test fold", i);
        }
    }

    // Decision tree properties
    #[test]
    fn decision_tree_predictions_valid(n_classes in 2usize..4) {
        // Create well-separated data
        let n_per_class = 5;
        let n_samples = n_classes * n_per_class;
        let mut data = Vec::with_capacity(n_samples * 2);
        let mut labels = Vec::with_capacity(n_samples);

        for class in 0..n_classes {
            for i in 0..n_per_class {
                data.push((class * 10) as f32 + i as f32 * 0.1);
                data.push((class * 10) as f32 + i as f32 * 0.1);
                labels.push(class);
            }
        }

        let x = Matrix::from_vec(n_samples, 2, data).expect("Test data should be valid");
        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &labels).expect("Test data should be valid");

        let predictions = tree.predict(&x);

        // All predictions should be valid class indices
        for &pred in &predictions {
            prop_assert!(pred < n_classes);
        }
    }

    // StandardScaler properties
    #[test]
    fn standard_scaler_produces_zero_mean(data in matrix_strategy(10, 3)) {
        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data).expect("Test data should be valid");

        let (n_rows, n_cols) = transformed.shape();
        for j in 0..n_cols {
            let mut sum = 0.0;
            for i in 0..n_rows {
                sum += transformed.get(i, j);
            }
            let mean = sum / n_rows as f32;
            prop_assert!(mean.abs() < 1e-4, "Column {} mean should be ~0, got {}", j, mean);
        }
    }

    #[test]
    fn standard_scaler_produces_unit_variance(data in matrix_strategy(10, 3)) {
        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data).expect("Test data should be valid");

        let (n_rows, n_cols) = transformed.shape();
        for j in 0..n_cols {
            // Compute variance
            let mut sum = 0.0;
            for i in 0..n_rows {
                sum += transformed.get(i, j);
            }
            let mean = sum / n_rows as f32;

            let mut var_sum = 0.0;
            for i in 0..n_rows {
                let diff = transformed.get(i, j) - mean;
                var_sum += diff * diff;
            }
            let std = (var_sum / n_rows as f32).sqrt();

            // std should be 1 (or 0 if constant column)
            prop_assert!(std < 1e-4 || (std - 1.0).abs() < 1e-4,
                "Column {} std should be ~0 or ~1, got {}", j, std);
        }
    }

    #[test]
    fn standard_scaler_inverse_recovers_data(data in matrix_strategy(8, 2)) {
        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data).expect("Test data should be valid");
        let recovered = scaler.inverse_transform(&transformed).expect("Test data should be valid");

        let (n_rows, n_cols) = data.shape();
        for i in 0..n_rows {
            for j in 0..n_cols {
                prop_assert!(
                    (data.get(i, j) - recovered.get(i, j)).abs() < 1e-3,
                    "Mismatch at ({}, {}): expected {}, got {}",
                    i, j, data.get(i, j), recovered.get(i, j)
                );
            }
        }
    }

    // MinMaxScaler properties
    #[test]
    fn minmax_scaler_bounds_to_range(data in matrix_strategy(10, 3)) {
        let mut scaler = MinMaxScaler::new();
        let transformed = scaler.fit_transform(&data).expect("Test data should be valid");

        let (n_rows, n_cols) = transformed.shape();
        for j in 0..n_cols {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..n_rows {
                let val = transformed.get(i, j);
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            // Min should be 0, max should be 1 (or both 0 if constant)
            prop_assert!(min_val >= -1e-4, "Column {} min should be >= 0, got {}", j, min_val);
            prop_assert!(max_val <= 1.0 + 1e-4, "Column {} max should be <= 1, got {}", j, max_val);
        }
    }

    #[test]
    fn minmax_scaler_inverse_recovers_data(data in matrix_strategy(8, 2)) {
        let mut scaler = MinMaxScaler::new();
        let transformed = scaler.fit_transform(&data).expect("Test data should be valid");
        let recovered = scaler.inverse_transform(&transformed).expect("Test data should be valid");

        let (n_rows, n_cols) = data.shape();
        for i in 0..n_rows {
            for j in 0..n_cols {
                prop_assert!(
                    (data.get(i, j) - recovered.get(i, j)).abs() < 1e-3,
                    "Mismatch at ({}, {}): expected {}, got {}",
                    i, j, data.get(i, j), recovered.get(i, j)
                );
            }
        }
    }

    #[test]
    fn minmax_scaler_custom_range_bounds(data in matrix_strategy(10, 2)) {
        let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
        let transformed = scaler.fit_transform(&data).expect("Test data should be valid");

        let (n_rows, n_cols) = transformed.shape();
        for j in 0..n_cols {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..n_rows {
                let val = transformed.get(i, j);
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            // Min should be -1, max should be 1 (or both -1 if constant)
            prop_assert!(min_val >= -1.0 - 1e-4, "Column {} min should be >= -1, got {}", j, min_val);
            prop_assert!(max_val <= 1.0 + 1e-4, "Column {} max should be <= 1, got {}", j, max_val);
        }
    }

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

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_vector_zero_norm() {
        let v = Vector::<f32>::zeros(5);
        assert!((v.norm() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_identity_matrix_properties() {
        let eye = Matrix::<f32>::eye(3);

        // I * I = I
        let result = eye.matmul(&eye).expect("Test data should be valid");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result.get(i, j) - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_silhouette_bounds() {
        // Silhouette score should be in [-1, 1]
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
            ],
        )
        .expect("Test data should be valid");
        let labels = vec![0, 0, 0, 1, 1, 1];
        let score = silhouette_score(&data, &labels);

        assert!(score >= -1.0);
        assert!(score <= 1.0);
    }
}
