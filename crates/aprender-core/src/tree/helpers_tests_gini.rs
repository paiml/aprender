
    // ========================================================================
    // Gini Impurity Tests
    // ========================================================================

    #[test]
    fn test_gini_impurity_empty() {
        assert!((gini_impurity(&[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gini_impurity_pure_single_class() {
        // All same class -> Gini = 0
        let labels = vec![0, 0, 0, 0];
        assert!((gini_impurity(&labels) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gini_impurity_two_classes_balanced() {
        // 50/50 split -> Gini = 0.5
        let labels = vec![0, 1, 0, 1];
        assert!((gini_impurity(&labels) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_gini_impurity_two_classes_unbalanced() {
        // 3/4 class 0, 1/4 class 1 -> Gini = 1 - (0.75^2 + 0.25^2) = 1 - 0.625 = 0.375
        let labels = vec![0, 0, 0, 1];
        assert!((gini_impurity(&labels) - 0.375).abs() < 1e-7);
    }

    #[test]
    fn test_gini_impurity_three_classes_uniform() {
        // Three classes equally distributed -> Gini = 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
        let labels = vec![0, 1, 2, 0, 1, 2];
        let expected = 1.0 - 3.0 * (1.0_f32 / 3.0).powi(2);
        assert!((gini_impurity(&labels) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gini_impurity_single_sample() {
        let labels = vec![5];
        assert!((gini_impurity(&labels) - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // Gini Split Tests
    // ========================================================================

    #[test]
    fn test_gini_split_empty_both() {
        assert!((gini_split(&[], &[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gini_split_perfect_split() {
        // Perfect split: all 0s on left, all 1s on right -> weighted Gini = 0
        let left = vec![0, 0, 0];
        let right = vec![1, 1, 1];
        assert!((gini_split(&left, &right) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gini_split_worst_case() {
        // Both sides have same distribution -> split doesn't help
        let left = vec![0, 1];
        let right = vec![0, 1];
        assert!((gini_split(&left, &right) - 0.5).abs() < 1e-7);
    }

    // ========================================================================
    // Sorted Unique Values Tests
    // ========================================================================

    #[test]
    fn test_get_sorted_unique_values_basic() {
        let values = vec![3.0, 1.0, 2.0, 1.0, 3.0];
        let unique = get_sorted_unique_values(&values);
        assert_eq!(unique, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_get_sorted_unique_values_single() {
        let values = vec![5.0];
        let unique = get_sorted_unique_values(&values);
        assert_eq!(unique, vec![5.0]);
    }

    #[test]
    fn test_get_sorted_unique_values_all_same() {
        let values = vec![2.0, 2.0, 2.0];
        let unique = get_sorted_unique_values(&values);
        assert_eq!(unique, vec![2.0]);
    }

    #[test]
    fn test_get_sorted_unique_values_already_sorted() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let unique = get_sorted_unique_values(&values);
        assert_eq!(unique, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ========================================================================
    // Split Labels by Threshold Tests
    // ========================================================================

    #[test]
    fn test_split_labels_by_threshold_normal() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0, 0, 1, 1];
        let result = split_labels_by_threshold(&x, &y, 2.5);
        let (left, right) = result.expect("split should produce non-empty partitions");
        assert_eq!(left, vec![0, 0]);
        assert_eq!(right, vec![1, 1]);
    }

    #[test]
    fn test_split_labels_by_threshold_all_left() {
        // All values <= threshold -> right is empty -> None
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0, 1, 0];
        assert!(split_labels_by_threshold(&x, &y, 10.0).is_none());
    }

    #[test]
    fn test_split_labels_by_threshold_all_right() {
        // All values > threshold -> left is empty -> None
        let x = vec![5.0, 6.0, 7.0];
        let y = vec![0, 1, 0];
        assert!(split_labels_by_threshold(&x, &y, 0.0).is_none());
    }

    #[test]
    fn test_split_labels_by_threshold_boundary() {
        // Exact threshold value goes to left (<=)
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0, 1, 2];
        let (left, right) = split_labels_by_threshold(&x, &y, 2.0).expect("split should succeed");
        assert_eq!(left, vec![0, 1]);
        assert_eq!(right, vec![2]);
    }

    // ========================================================================
    // Calculate Information Gain Tests
    // ========================================================================

    #[test]
    fn test_calculate_information_gain_perfect_split() {
        let current = gini_impurity(&[0, 0, 1, 1]);
        let gain = calculate_information_gain(current, &[0, 0], &[1, 1]);
        // Perfect split: gain equals the full impurity
        assert!((gain - current).abs() < 1e-7);
    }

    #[test]
    fn test_calculate_information_gain_no_improvement() {
        let current = gini_impurity(&[0, 1, 0, 1]);
        let gain = calculate_information_gain(current, &[0, 1], &[0, 1]);
        // Same distribution on both sides -> zero gain
        assert!(gain.abs() < 1e-7);
    }

    // ========================================================================
    // Find Best Split for Feature Tests
    // ========================================================================

    #[test]
    fn test_find_best_split_for_feature_too_few_samples() {
        let x = vec![1.0];
        let y = vec![0];
        assert!(find_best_split_for_feature(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_split_for_feature_all_same_value() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![0, 1, 0];
        assert!(find_best_split_for_feature(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_split_for_feature_clear_split() {
        // Feature perfectly separates classes
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0, 0, 1, 1];
        let result = find_best_split_for_feature(&x, &y);
        let (threshold, gain) = result.expect("should find a split");
        assert!((threshold - 2.5).abs() < 1e-7);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_split_for_feature_pure_labels() {
        // All same label -> no gain possible
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0, 0, 0];
        assert!(find_best_split_for_feature(&x, &y).is_none());
    }

    // ========================================================================
    // Find Best Split (all features) Tests
    // ========================================================================

    #[test]
    fn test_find_best_split_single_sample() {
        let x = Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("matrix creation");
        let y = vec![0];
        assert!(find_best_split(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_split_separable() {
        // Feature 0 separates perfectly, feature 1 does not
        let x = Matrix::from_vec(4, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0])
            .expect("matrix creation");
        let y = vec![0, 0, 1, 1];
        let (feat, threshold, gain) = find_best_split(&x, &y).expect("should find split");
        assert_eq!(feat, 0);
        assert!((threshold - 2.5).abs() < 1e-7);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_split_pure_labels() {
        let x =
            Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("matrix creation");
        let y = vec![0, 0, 0];
        assert!(find_best_split(&x, &y).is_none());
    }

    // ========================================================================
    // Majority Class Tests
    // ========================================================================

    #[test]
    fn test_majority_class_single() {
        assert_eq!(majority_class(&[7]), 7);
    }

    #[test]
    fn test_majority_class_clear_winner() {
        assert_eq!(majority_class(&[0, 1, 1, 1, 0]), 1);
    }

    #[test]
    fn test_majority_class_all_same() {
        assert_eq!(majority_class(&[3, 3, 3]), 3);
    }

    // ========================================================================
    // Split Data by Indices Tests
    // ========================================================================

    #[test]
    fn test_split_data_by_indices_basic() {
        let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("matrix creation");
        let y = vec![0, 1, 2, 3];
        let indices = vec![0, 2];
        let (subset_x, subset_y) = split_data_by_indices(&x, &y, &indices);
        assert_eq!(subset_x.shape(), (2, 2));
        assert_eq!(subset_y, vec![0, 2]);
        assert!((subset_x.get(0, 0) - 1.0).abs() < 1e-7);
        assert!((subset_x.get(0, 1) - 2.0).abs() < 1e-7);
        assert!((subset_x.get(1, 0) - 5.0).abs() < 1e-7);
        assert!((subset_x.get(1, 1) - 6.0).abs() < 1e-7);
    }

    // ========================================================================
    // Check Stopping Criteria Tests
    // ========================================================================

    #[test]
    fn test_check_stopping_criteria_pure_node() {
        let y = vec![1, 1, 1];
        let result = check_stopping_criteria(&y, 0, None);
        let node = result.expect("pure node should trigger stop");
        match node {
            TreeNode::Leaf(leaf) => {
                assert_eq!(leaf.class_label, 1);
                assert_eq!(leaf.n_samples, 3);
            }
            TreeNode::Node(_) => panic!("expected leaf, got internal node"),
        }
    }

    #[test]
    fn test_check_stopping_criteria_max_depth() {
        let y = vec![0, 1, 1];
        let result = check_stopping_criteria(&y, 5, Some(5));
        let node = result.expect("max depth should trigger stop");
        match node {
            TreeNode::Leaf(leaf) => {
                assert_eq!(leaf.class_label, 1); // majority
                assert_eq!(leaf.n_samples, 3);
            }
            TreeNode::Node(_) => panic!("expected leaf, got internal node"),
        }
    }

    #[test]
    fn test_check_stopping_criteria_no_stop() {
        let y = vec![0, 1, 1];
        let result = check_stopping_criteria(&y, 2, Some(5));
        assert!(result.is_none());
    }

    #[test]
    fn test_check_stopping_criteria_no_max_depth() {
        let y = vec![0, 1, 1];
        let result = check_stopping_criteria(&y, 100, None);
        assert!(result.is_none());
    }

    // ========================================================================
    // Split Indices by Threshold Tests
    // ========================================================================

    #[test]
    fn test_split_indices_by_threshold_normal() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 3.0, 2.0, 4.0]).expect("matrix creation");
        let result = split_indices_by_threshold(&x, 0, 2.5, 4);
        let (left, right) = result.expect("should split successfully");
        assert_eq!(left, vec![0, 2]);
        assert_eq!(right, vec![1, 3]);
    }

    #[test]
    fn test_split_indices_by_threshold_all_left() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        assert!(split_indices_by_threshold(&x, 0, 10.0, 3).is_none());
    }

    #[test]
    fn test_split_indices_by_threshold_all_right() {
        let x = Matrix::from_vec(3, 1, vec![5.0, 6.0, 7.0]).expect("matrix creation");
        assert!(split_indices_by_threshold(&x, 0, 0.0, 3).is_none());
    }

    // ========================================================================
    // Flatten / Reconstruct Tree Tests
    // ========================================================================

    #[test]
    fn test_flatten_and_reconstruct_leaf() {
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 2,
            n_samples: 10,
        });

        let mut features = Vec::new();
        let mut thresholds = Vec::new();
        let mut classes = Vec::new();
        let mut samples = Vec::new();
        let mut left_children = Vec::new();
        let mut right_children = Vec::new();

        let root_idx = flatten_tree_node(
            &leaf,
            &mut features,
            &mut thresholds,
            &mut classes,
            &mut samples,
            &mut left_children,
            &mut right_children,
        );

        assert_eq!(root_idx, 0);
        assert_eq!(features.len(), 1);
        assert!((features[0] - (-1.0)).abs() < 1e-7);
        assert!((classes[0] - 2.0).abs() < 1e-7);
        assert!((samples[0] - 10.0).abs() < 1e-7);

        // Reconstruct
        let reconstructed = reconstruct_tree_node(
            root_idx,
            &features,
            &thresholds,
            &classes,
            &samples,
            &left_children,
            &right_children,
        );
        match reconstructed {
            TreeNode::Leaf(l) => {
                assert_eq!(l.class_label, 2);
                assert_eq!(l.n_samples, 10);
            }
            TreeNode::Node(_) => panic!("expected leaf"),
        }
    }
