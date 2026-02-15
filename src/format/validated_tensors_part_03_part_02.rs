
    // FALSIFY-001: Embedding density check
    #[test]
    fn falsify_001_embedding_rejects_all_zeros() {
        let bad_data = vec![0.0f32; 100 * 64]; // 100% zeros
        let result = ValidatedEmbedding::new(bad_data, 100, 64);
        assert!(result.is_err(), "Should reject 100% zeros");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("DENSITY"),
            "Error should mention density: {}",
            err.message
        );
    }

    #[test]
    fn falsify_001_embedding_rejects_mostly_zeros() {
        // Simulate PMAT-234: 94.5% zeros
        let vocab_size = 1000;
        let hidden_dim = 64;
        let mut data = vec![0.0f32; vocab_size * hidden_dim];
        // Only last 5.5% non-zero
        for i in (945 * hidden_dim)..(vocab_size * hidden_dim) {
            data[i] = 0.1;
        }
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject 94.5% zeros");
    }

    #[test]
    fn falsify_001_embedding_accepts_good_data() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(
            result.is_ok(),
            "Should accept good data: {:?}",
            result.err()
        );
    }

    // FALSIFY-002: Inf rejection (Gate 4 — F-DATA-QUALITY-002)
    // §18.3: This test was missing, causing a gap in FALSIFY numbering.
    #[test]
    fn falsify_002_embedding_rejects_inf() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[7] = f32::INFINITY;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject Inf");
        assert!(result.unwrap_err().message.contains("Inf"));
    }

    #[test]
    fn falsify_002_embedding_rejects_neg_inf() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[3] = f32::NEG_INFINITY;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject -Inf");
        assert!(result.unwrap_err().message.contains("Inf"));
    }

    #[test]
    fn falsify_002_weight_rejects_inf() {
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        data[50] = f32::INFINITY;
        let result = ValidatedWeight::new(data, 10, 10, "test_weight");
        assert!(result.is_err(), "Should reject Inf in weight");
    }

    // FALSIFY-003: NaN rejection
    #[test]
    fn falsify_003_embedding_rejects_nan() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        data[5] = f32::NAN;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject NaN");
        assert!(result.unwrap_err().message.contains("NaN"));
    }

    #[test]
    fn falsify_003_weight_rejects_nan() {
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        data[50] = f32::NAN;
        let result = ValidatedWeight::new(data, 10, 10, "test_weight");
        assert!(result.is_err(), "Should reject NaN");
    }

    // FALSIFY-004: Spot check catches offset bugs
    #[test]
    fn falsify_004_spot_check_catches_offset_bug() {
        // Token at 10% of vocab has zero embedding
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Zero out token at 10% (token 10)
        let token_10_start = 10 * hidden_dim;
        for i in token_10_start..(token_10_start + hidden_dim) {
            data[i] = 0.0;
        }

        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should catch zero token at 10%");
        assert!(result.unwrap_err().rule_id == "F-DATA-QUALITY-004");
    }

    // FALSIFY-005: Shape validation
    #[test]
    fn falsify_005_rejects_wrong_shape() {
        let data = vec![0.1f32; 1000];
        let result = ValidatedEmbedding::new(data, 100, 64); // expects 6400
        assert!(result.is_err(), "Should reject wrong shape");
    }

    // Weight-specific tests
    #[test]
    fn weight_rejects_all_zeros() {
        let data = vec![0.0f32; 100];
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_err());
    }

    #[test]
    fn weight_accepts_good_data() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_ok());
    }

    // Vector tests
    #[test]
    fn vector_rejects_wrong_length() {
        let data = vec![0.1f32; 50];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_err());
    }

    #[test]
    fn vector_accepts_good_data() {
        let data = vec![1.0f32; 100];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_ok());
    }

    // PMAT-248: PhantomData<Layout> enforcement tests

    #[test]
    fn pmat_248_validated_weight_is_row_major_by_default() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let weight: ValidatedWeight = ValidatedWeight::new(data, 10, 10, "test").unwrap();
        // This compiles because ValidatedWeight == ValidatedWeight<RowMajor>
        let _explicit: ValidatedWeight<RowMajor> = weight;
    }

    #[test]
    fn pmat_248_row_major_marker_is_zero_sized() {
        assert_eq!(std::mem::size_of::<RowMajor>(), 0);
        assert_eq!(
            std::mem::size_of::<PhantomData<RowMajor>>(),
            0,
            "PhantomData<RowMajor> must be zero-sized"
        );
    }

    #[test]
    fn pmat_248_phantom_data_does_not_increase_struct_size() {
        // ValidatedWeight with PhantomData should have same layout as without
        // (PhantomData is ZST, compiler optimizes it away)
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let weight = ValidatedWeight::new(data, 10, 10, "test").unwrap();
        // Ensure all fields are accessible (compile-time check)
        let _ = weight.data();
        let _ = weight.out_dim();
        let _ = weight.in_dim();
        let _ = weight.name();
        let _ = weight.stats();
    }

    // ================================================================
    // ValidatedEmbedding - Inf rejection
    // ================================================================

    #[test]
    fn embedding_rejects_inf_values() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[7] = f32::INFINITY;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject Inf values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Inf"),
            "Error should mention Inf: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedEmbedding - L2 norm ~0
    // ================================================================

    #[test]
    fn embedding_rejects_near_zero_l2_norm() {
        let vocab_size = 10;
        let hidden_dim = 8;
        // Values above the zero threshold (1e-10) but producing negligible L2 norm (< 1e-6).
        // With 80 elements at 1e-8 each: L2 = sqrt(80 * (1e-8)^2) = sqrt(80)*1e-8 ~ 8.9e-8.
        // Also make values vary slightly to pass the constant check.
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| 1e-8 + (i as f32) * 1e-12)
            .collect();
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject near-zero L2 norm");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("L2 norm"),
            "Error should mention L2 norm: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedEmbedding - Constant values
    // ================================================================

    #[test]
    fn embedding_rejects_constant_values() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let data = vec![0.5f32; vocab_size * hidden_dim];
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject constant values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("constant"),
            "Error should mention constant: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedEmbedding accessors
    // ================================================================

    #[test]
    fn embedding_accessors_return_correct_values() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let original_data = data.clone();
        let emb = ValidatedEmbedding::new(data, vocab_size, hidden_dim)
            .expect("good data should be accepted");

        assert_eq!(emb.vocab_size(), vocab_size);
        assert_eq!(emb.hidden_dim(), hidden_dim);
        assert_eq!(emb.data().len(), vocab_size * hidden_dim);
        assert_eq!(emb.data(), original_data.as_slice());

        let stats = emb.stats();
        assert_eq!(stats.len, vocab_size * hidden_dim);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);

        // into_inner consumes
        let inner = emb.into_inner();
        assert_eq!(inner.len(), vocab_size * hidden_dim);
        assert_eq!(inner, original_data);
    }

    // ================================================================
    // ValidatedWeight - Inf rejection
    // ================================================================

    #[test]
    fn weight_rejects_inf_values() {
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        data[42] = f32::INFINITY;
        let result = ValidatedWeight::new(data, 10, 10, "test_weight");
        assert!(result.is_err(), "Should reject Inf values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Inf"),
            "Error should mention Inf: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedWeight - L2 norm ~0
    // ================================================================

    #[test]
    fn weight_rejects_near_zero_l2_norm() {
        // Values above the zero threshold (1e-10) but producing negligible L2 norm (< 1e-6).
        // With 100 elements at 1e-8: L2 = sqrt(100 * (1e-8)^2) = 10 * 1e-8 = 1e-7.
        let data: Vec<f32> = (0..100).map(|i| 1e-8 + (i as f32) * 1e-12).collect();
        let result = ValidatedWeight::new(data, 10, 10, "test_weight");
        assert!(result.is_err(), "Should reject near-zero L2 norm");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("L2 norm"),
            "Error should mention L2 norm: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedWeight accessors
    // ================================================================

    #[test]
    fn weight_accessors_return_correct_values() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let original_data = data.clone();
        let weight =
            ValidatedWeight::new(data, 10, 10, "my_weight").expect("good data should be accepted");

        assert_eq!(weight.out_dim(), 10);
        assert_eq!(weight.in_dim(), 10);
        assert_eq!(weight.name(), "my_weight");
        assert_eq!(weight.data().len(), 100);
        assert_eq!(weight.data(), original_data.as_slice());

        let stats = weight.stats();
        assert_eq!(stats.len, 100);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);

        let inner = weight.into_inner();
        assert_eq!(inner, original_data);
    }

    // ================================================================
    // ValidatedVector - NaN rejection
    // ================================================================

    #[test]
    fn vector_rejects_nan_values() {
        let mut data = vec![1.0f32; 50];
        data[25] = f32::NAN;
        let result = ValidatedVector::new(data, 50, "test_vec");
        assert!(result.is_err(), "Should reject NaN values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("NaN"),
            "Error should mention NaN: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedVector - Inf rejection
    // ================================================================

    #[test]
    fn vector_rejects_inf_values() {
        let mut data = vec![1.0f32; 50];
        data[10] = f32::NEG_INFINITY;
        let result = ValidatedVector::new(data, 50, "test_vec");
        assert!(result.is_err(), "Should reject Inf values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Inf"),
            "Error should mention Inf: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedVector accessors
    // ================================================================

    #[test]
    fn vector_accessors_return_correct_values() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let original_data = data.clone();
        let vec = ValidatedVector::new(data, 5, "my_vector").expect("good data should be accepted");

        assert_eq!(vec.name(), "my_vector");
        assert_eq!(vec.data().len(), 5);
        assert_eq!(vec.data(), original_data.as_slice());

        let stats = vec.stats();
        assert_eq!(stats.len, 5);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);

        let inner = vec.into_inner();
        assert_eq!(inner, original_data);
    }

    // ================================================================
    // TensorStats::compute edge cases
    // ================================================================

    #[test]
    fn tensor_stats_compute_empty_data() {
        let stats = TensorStats::compute(&[]);
        assert_eq!(stats.len, 0);
        assert_eq!(stats.zero_count, 0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.l2_norm, 0.0);
    }

    #[test]
    fn tensor_stats_compute_all_nan() {
        let stats = TensorStats::compute(&[f32::NAN, f32::NAN, f32::NAN]);
        assert_eq!(stats.len, 3);
        assert_eq!(stats.nan_count, 3);
        assert_eq!(stats.inf_count, 0);
        assert_eq!(stats.zero_count, 0);
        // min/max should be 0.0 since no valid values found
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }
