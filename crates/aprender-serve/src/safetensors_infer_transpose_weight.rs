
    // =========================================================================
    // GH-219 Coverage Gap: transpose_weight, concat_qkv_transposed,
    // has_tensor_with_fallback_generic, get_tensor_with_fallback_generic
    // =========================================================================

    // -------------------------------------------------------------------------
    // transpose_weight (PMAT-095: no-op pass-through)
    // -------------------------------------------------------------------------

    #[test]
    fn test_transpose_weight_identity_gh219() {
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = SafetensorsToAprConverter::transpose_weight(&weight, 2, 3);
        assert_eq!(result, weight);
    }

    #[test]
    fn test_transpose_weight_empty_gh219() {
        let result = SafetensorsToAprConverter::transpose_weight(&[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_transpose_weight_single_element_gh219() {
        let weight = vec![42.0];
        let result = SafetensorsToAprConverter::transpose_weight(&weight, 1, 1);
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_transpose_weight_preserves_data_gh219() {
        // Verify that it copies, not moves
        let weight: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let result = SafetensorsToAprConverter::transpose_weight(&weight, 10, 10);
        assert_eq!(result.len(), 100);
        assert!((result[50] - 5.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // concat_qkv_transposed
    // -------------------------------------------------------------------------

    #[test]
    fn test_concat_qkv_transposed_basic_gh219() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let v = vec![5.0, 6.0];
        let result = SafetensorsToAprConverter::concat_qkv_transposed(&q, &k, &v, 2, 2);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_qkv_transposed_different_sizes_gh219() {
        // Q: hidden_dim x hidden_dim (4 elements for 2x2)
        let q = vec![1.0, 2.0, 3.0, 4.0];
        // K: kv_dim x hidden_dim (2 elements for 1x2)
        let k = vec![5.0, 6.0];
        // V: kv_dim x hidden_dim (2 elements for 1x2)
        let v = vec![7.0, 8.0];
        let result = SafetensorsToAprConverter::concat_qkv_transposed(&q, &k, &v, 2, 1);
        assert_eq!(result.len(), 8);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_qkv_transposed_empty_gh219() {
        let result =
            SafetensorsToAprConverter::concat_qkv_transposed(&[], &[], &[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_concat_qkv_transposed_capacity_gh219() {
        let q = vec![0.0; 100];
        let k = vec![0.0; 50];
        let v = vec![0.0; 50];
        let result = SafetensorsToAprConverter::concat_qkv_transposed(&q, &k, &v, 10, 5);
        assert_eq!(result.len(), 200);
    }
