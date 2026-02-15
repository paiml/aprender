
    #[test]
    fn test_flatten_and_reconstruct_tree() {
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 2.5,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 3,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 2,
            })),
        });

        let mut features = Vec::new();
        let mut thresholds = Vec::new();
        let mut classes = Vec::new();
        let mut samples = Vec::new();
        let mut left_children = Vec::new();
        let mut right_children = Vec::new();

        let root_idx = flatten_tree_node(
            &tree,
            &mut features,
            &mut thresholds,
            &mut classes,
            &mut samples,
            &mut left_children,
            &mut right_children,
        );

        assert_eq!(root_idx, 0);
        assert_eq!(features.len(), 3); // root + 2 leaves

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
            TreeNode::Node(n) => {
                assert_eq!(n.feature_idx, 0);
                assert!((n.threshold - 2.5).abs() < 1e-7);
                match n.left.as_ref() {
                    TreeNode::Leaf(l) => assert_eq!(l.class_label, 0),
                    _ => panic!("expected left leaf"),
                }
                match n.right.as_ref() {
                    TreeNode::Leaf(l) => assert_eq!(l.class_label, 1),
                    _ => panic!("expected right leaf"),
                }
            }
            TreeNode::Leaf(_) => panic!("expected node, got leaf"),
        }
    }

    // ========================================================================
    // Build Tree Tests
    // ========================================================================

    #[test]
    fn test_build_tree_pure_data() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        let y = vec![0, 0, 0];
        let tree = build_tree(&x, &y, 0, None);
        match tree {
            TreeNode::Leaf(leaf) => {
                assert_eq!(leaf.class_label, 0);
                assert_eq!(leaf.n_samples, 3);
            }
            TreeNode::Node(_) => panic!("pure data should produce a leaf"),
        }
    }

    #[test]
    fn test_build_tree_max_depth_zero() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = vec![0, 0, 1, 1];
        let tree = build_tree(&x, &y, 0, Some(0));
        match tree {
            TreeNode::Leaf(leaf) => {
                // Majority class with depth=0
                assert_eq!(leaf.n_samples, 4);
            }
            TreeNode::Node(_) => panic!("max_depth=0 should produce a leaf"),
        }
    }

    #[test]
    fn test_build_tree_separable() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = vec![0, 0, 1, 1];
        let tree = build_tree(&x, &y, 0, Some(5));
        match &tree {
            TreeNode::Node(n) => {
                assert_eq!(n.feature_idx, 0);
                // Threshold should be 2.5 (midpoint between 2 and 3)
                assert!((n.threshold - 2.5).abs() < 1e-7);
            }
            TreeNode::Leaf(_) => panic!("separable data should produce a split"),
        }
    }

    #[test]
    fn test_build_tree_depth_limited() {
        // Three classes in feature space, max_depth=1 forces a single split
        let x =
            Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("matrix creation");
        let y = vec![0, 0, 1, 1, 2, 2];
        let tree = build_tree(&x, &y, 0, Some(1));
        match &tree {
            TreeNode::Node(n) => {
                // Both children must be leaves since depth=1
                assert!(matches!(n.left.as_ref(), TreeNode::Leaf(_)));
                assert!(matches!(n.right.as_ref(), TreeNode::Leaf(_)));
            }
            TreeNode::Leaf(_) => panic!("mixed labels should produce a split"),
        }
    }

    // ========================================================================
    // Mean and Variance Tests
    // ========================================================================

    #[test]
    fn test_mean_f32_empty() {
        assert!((mean_f32(&[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_mean_f32_single() {
        assert!((mean_f32(&[5.0]) - 5.0).abs() < 1e-7);
    }

    #[test]
    fn test_mean_f32_multiple() {
        assert!((mean_f32(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-7);
    }

    #[test]
    fn test_variance_f32_empty() {
        assert!((variance_f32(&[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_variance_f32_single() {
        assert!((variance_f32(&[42.0]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_variance_f32_uniform() {
        // Var([1,2,3,4]) = mean of squared deviations from mean(2.5)
        // = ((1.5^2 + 0.5^2 + 0.5^2 + 1.5^2)/4) = (2.25+0.25+0.25+2.25)/4 = 5.0/4 = 1.25
        assert!((variance_f32(&[1.0, 2.0, 3.0, 4.0]) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn test_variance_f32_all_same() {
        assert!((variance_f32(&[3.0, 3.0, 3.0]) - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // Compute MSE Tests
    // ========================================================================

    #[test]
    fn test_compute_mse_empty_both() {
        assert!((compute_mse(&[], &[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_compute_mse_perfect_split() {
        // Left: all same, right: all same -> MSE = 0
        let left = vec![1.0, 1.0, 1.0];
        let right = vec![5.0, 5.0, 5.0];
        assert!((compute_mse(&left, &right) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_compute_mse_mixed() {
        let left = vec![1.0, 3.0]; // variance = 1.0
        let right = vec![5.0, 7.0]; // variance = 1.0
                                    // MSE = (2/4)*1.0 + (2/4)*1.0 = 1.0
        assert!((compute_mse(&left, &right) - 1.0).abs() < 1e-6);
    }

    // ========================================================================
    // Get Unique Feature Values Tests
    // ========================================================================

    #[test]
    fn test_get_unique_feature_values_basic() {
        let x = Matrix::from_vec(4, 2, vec![3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0])
            .expect("matrix creation");
        let values = get_unique_feature_values(&x, 0, 4);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_get_unique_feature_values_second_column() {
        let x =
            Matrix::from_vec(3, 2, vec![0.0, 5.0, 0.0, 3.0, 0.0, 5.0]).expect("matrix creation");
        let values = get_unique_feature_values(&x, 1, 3);
        assert_eq!(values, vec![3.0, 5.0]);
    }

    // ========================================================================
    // Split by Threshold (Regression) Tests
    // ========================================================================

    #[test]
    fn test_split_by_threshold_regression() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let (left, right) = split_by_threshold(&x, &y, 0, 2.5);
        assert_eq!(left, vec![10.0, 20.0]);
        assert_eq!(right, vec![30.0, 40.0]);
    }

    #[test]
    fn test_split_by_threshold_all_left() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        let y = vec![10.0, 20.0, 30.0];
        let (left, right) = split_by_threshold(&x, &y, 0, 100.0);
        assert_eq!(left.len(), 3);
        assert!(right.is_empty());
    }

    // ========================================================================
    // Evaluate Split Gain Tests
    // ========================================================================

    #[test]
    fn test_evaluate_split_gain_empty_left() {
        assert!(evaluate_split_gain(&[], &[1.0, 2.0], 1.0).is_none());
    }

    #[test]
    fn test_evaluate_split_gain_empty_right() {
        assert!(evaluate_split_gain(&[1.0, 2.0], &[], 1.0).is_none());
    }

    #[test]
    fn test_evaluate_split_gain_positive() {
        let current_variance = variance_f32(&[1.0, 1.0, 5.0, 5.0]);
        let gain = evaluate_split_gain(&[1.0, 1.0], &[5.0, 5.0], current_variance);
        let g = gain.expect("should have positive gain");
        assert!(g > 0.0);
        // Perfect split removes all variance
        assert!((g - current_variance).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_split_gain_no_gain() {
        // Same distribution on both sides
        let current_variance = variance_f32(&[1.0, 5.0, 1.0, 5.0]);
        let result = evaluate_split_gain(&[1.0, 5.0], &[1.0, 5.0], current_variance);
        assert!(result.is_none());
    }

    // ========================================================================
    // Find Best Regression Split Tests
    // ========================================================================

    #[test]
    fn test_find_best_regression_split_single_sample() {
        let x = Matrix::from_vec(1, 1, vec![1.0]).expect("matrix creation");
        let y = vec![5.0];
        assert!(find_best_regression_split(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_regression_split_separable() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = vec![1.0, 1.0, 10.0, 10.0];
        let result = find_best_regression_split(&x, &y);
        let (feat, threshold, gain) = result.expect("should find a split");
        assert_eq!(feat, 0);
        assert!((threshold - 2.5).abs() < 1e-7);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_regression_split_constant_y() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        let y = vec![5.0, 5.0, 5.0];
        assert!(find_best_regression_split(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_regression_split_for_feature_single_unique_value() {
        let x = Matrix::from_vec(3, 1, vec![2.0, 2.0, 2.0]).expect("matrix creation");
        let y = vec![1.0, 2.0, 3.0];
        let current_var = variance_f32(&y);
        let result = find_best_regression_split_for_feature(&x, &y, 0, 3, current_var);
        assert!(result.is_none());
    }

    // ========================================================================
    // Split Regression Data by Indices Tests
    // ========================================================================

    #[test]
    fn test_split_regression_data_by_indices() {
        let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("matrix creation");
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let indices = vec![1, 3];
        let (subset_x, subset_y) = split_regression_data_by_indices(&x, &y, &indices);
        assert_eq!(subset_x.shape(), (2, 2));
        assert_eq!(subset_y, vec![20.0, 40.0]);
        assert!((subset_x.get(0, 0) - 3.0).abs() < 1e-7);
        assert!((subset_x.get(0, 1) - 4.0).abs() < 1e-7);
        assert!((subset_x.get(1, 0) - 7.0).abs() < 1e-7);
        assert!((subset_x.get(1, 1) - 8.0).abs() < 1e-7);
    }

    // ========================================================================
    // Make Regression Leaf Tests
    // ========================================================================

    #[test]
    fn test_make_regression_leaf() {
        let y = vec![2.0, 4.0, 6.0];
        let leaf = make_regression_leaf(&y, 3);
        match leaf {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - 4.0).abs() < 1e-7);
                assert_eq!(l.n_samples, 3);
            }
            RegressionTreeNode::Node(_) => panic!("expected leaf"),
        }
    }

    #[test]
    fn test_make_regression_leaf_empty() {
        let leaf = make_regression_leaf(&[], 0);
        match leaf {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - 0.0).abs() < 1e-7);
                assert_eq!(l.n_samples, 0);
            }
            RegressionTreeNode::Node(_) => panic!("expected leaf"),
        }
    }

    // ========================================================================
    // At Max Depth Tests
    // ========================================================================

    #[test]
    fn test_at_max_depth_none() {
        assert!(!at_max_depth(100, None));
    }

    #[test]
    fn test_at_max_depth_not_reached() {
        assert!(!at_max_depth(3, Some(5)));
    }

    #[test]
    fn test_at_max_depth_exactly_reached() {
        assert!(at_max_depth(5, Some(5)));
    }

    #[test]
    fn test_at_max_depth_exceeded() {
        assert!(at_max_depth(10, Some(5)));
    }

    // ========================================================================
    // Partition by Threshold Tests
    // ========================================================================

    #[test]
    fn test_partition_by_threshold() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 5.0, 2.0, 4.0, 3.0]).expect("matrix creation");
        let (left, right) = partition_by_threshold(&x, 5, 0, 3.0);
        assert_eq!(left, vec![0, 2, 4]);
        assert_eq!(right, vec![1, 3]);
    }

    #[test]
    fn test_partition_by_threshold_all_left() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        let (left, right) = partition_by_threshold(&x, 3, 0, 10.0);
        assert_eq!(left.len(), 3);
        assert!(right.is_empty());
    }

    // ========================================================================
    // Build Regression Tree Tests
    // ========================================================================

    #[test]
    fn test_build_regression_tree_constant_target() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![5.0, 5.0, 5.0, 5.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(5), 2, 1);
        match tree {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - 5.0).abs() < 1e-7);
            }
            RegressionTreeNode::Node(_) => panic!("constant target should produce leaf"),
        }
    }

    #[test]
    fn test_build_regression_tree_max_depth_zero() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![1.0, 1.0, 10.0, 10.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(0), 2, 1);
        match tree {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - mean_f32(&[1.0, 1.0, 10.0, 10.0])).abs() < 1e-6);
            }
            RegressionTreeNode::Node(_) => panic!("max_depth=0 should produce leaf"),
        }
    }

    #[test]
    fn test_build_regression_tree_separable() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![1.0, 1.0, 10.0, 10.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(5), 2, 1);
        match &tree {
            RegressionTreeNode::Node(n) => {
                assert_eq!(n.feature_idx, 0);
                assert!((n.threshold - 2.5).abs() < 1e-7);
            }
            RegressionTreeNode::Leaf(_) => panic!("separable data should produce split"),
        }
    }
