
    #[test]
    fn test_build_regression_tree_min_samples_split() {
        // min_samples_split=10 means we never split with 4 samples
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![1.0, 1.0, 10.0, 10.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(5), 10, 1);
        match tree {
            RegressionTreeNode::Leaf(_) => {} // expected
            RegressionTreeNode::Node(_) => {
                panic!("min_samples_split=10 should prevent splitting 4 samples")
            }
        }
    }

    #[test]
    fn test_build_regression_tree_min_samples_leaf() {
        // min_samples_leaf=3 with 4 samples: 2/2 split disallowed, must be leaf
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![1.0, 1.0, 10.0, 10.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(5), 2, 3);
        match tree {
            RegressionTreeNode::Leaf(_) => {} // expected
            RegressionTreeNode::Node(_) => {
                panic!("min_samples_leaf=3 should prevent 2/2 split")
            }
        }
    }

    // ========================================================================
    // Feature Importance Tests
    // ========================================================================

    #[test]
    fn test_count_tree_samples_leaf() {
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 42,
        });
        assert_eq!(count_tree_samples(&leaf), 42);
    }

    #[test]
    fn test_count_tree_samples_tree() {
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 1.0,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 10,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 20,
            })),
        });
        assert_eq!(count_tree_samples(&tree), 30);
    }

    #[test]
    fn test_compute_tree_feature_importances_leaf_only() {
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 10,
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&leaf, &mut importances);
        assert_eq!(importances, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_tree_feature_importances_single_split() {
        let tree = TreeNode::Node(Node {
            feature_idx: 1,
            threshold: 2.0,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 5,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 3,
            })),
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&tree, &mut importances);
        // Feature 1 used at root with 8 total samples
        assert!((importances[0] - 0.0).abs() < 1e-7);
        assert!((importances[1] - 8.0).abs() < 1e-7);
        assert!((importances[2] - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // Regression Feature Importance Tests
    // ========================================================================

    #[test]
    fn test_count_regression_tree_samples_leaf() {
        let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 3.5,
            n_samples: 15,
        });
        assert_eq!(count_regression_tree_samples(&leaf), 15);
    }

    #[test]
    fn test_count_regression_tree_samples_tree() {
        let tree = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 1.0,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 1.0,
                n_samples: 7,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 5.0,
                n_samples: 13,
            })),
        });
        assert_eq!(count_regression_tree_samples(&tree), 20);
    }

    #[test]
    fn test_compute_regression_tree_feature_importances() {
        let tree = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 2,
            threshold: 3.0,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 1.0,
                n_samples: 4,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 9.0,
                n_samples: 6,
            })),
        });
        let mut importances = vec![0.0; 4];
        compute_regression_tree_feature_importances(&tree, &mut importances);
        assert!((importances[2] - 10.0).abs() < 1e-7);
        assert!((importances[0] - 0.0).abs() < 1e-7);
        assert!((importances[1] - 0.0).abs() < 1e-7);
        assert!((importances[3] - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // Bootstrap Sample Tests
    // ========================================================================

    #[test]
    fn test_bootstrap_sample_deterministic() {
        let sample1 = bootstrap_sample(10, Some(42));
        let sample2 = bootstrap_sample(10, Some(42));
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_bootstrap_sample_length() {
        let sample = bootstrap_sample(20, Some(1));
        assert_eq!(sample.len(), 20);
    }

    #[test]
    fn test_bootstrap_sample_range() {
        let n = 10;
        let sample = bootstrap_sample(n, Some(99));
        for &idx in &sample {
            assert!(idx < n, "index {idx} should be less than {n}");
        }
    }

    #[test]
    fn test_bootstrap_sample_without_seed() {
        let sample = bootstrap_sample(10, None);
        assert_eq!(sample.len(), 10);
        for &idx in &sample {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_bootstrap_sample_different_seeds() {
        let sample1 = bootstrap_sample(100, Some(1));
        let sample2 = bootstrap_sample(100, Some(2));
        // Very unlikely to be identical with different seeds
        assert_ne!(sample1, sample2);
    }

    // ========================================================================
    // Multi-Feature Regression Split Tests
    // ========================================================================

    #[test]
    fn test_find_best_regression_split_multi_feature() {
        // Feature 1 is the informative one, feature 0 is noise
        #[rustfmt::skip]
        let x = Matrix::from_vec(6, 2, vec![
            5.0, 1.0,
            5.0, 2.0,
            5.0, 3.0,
            5.0, 4.0,
            5.0, 5.0,
            5.0, 6.0,
        ]).expect("matrix creation");
        let y = vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0];
        let (feat, _threshold, gain) =
            find_best_regression_split(&x, &y).expect("should find split");
        assert_eq!(feat, 1); // should pick the informative feature
        assert!(gain > 0.0);
    }

    // ========================================================================
    // Deep Tree Flatten/Reconstruct Test
    // ========================================================================

    #[test]
    fn test_flatten_reconstruct_deep_tree() {
        // Three levels deep
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 5.0,
            left: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 2.0,
                left: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 0,
                    n_samples: 2,
                })),
                right: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 1,
                    n_samples: 3,
                })),
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 2,
                n_samples: 5,
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

        // 5 nodes total: 2 internal + 3 leaves
        assert_eq!(features.len(), 5);

        let reconstructed = reconstruct_tree_node(
            root_idx,
            &features,
            &thresholds,
            &classes,
            &samples,
            &left_children,
            &right_children,
        );

        // Verify structure
        match &reconstructed {
            TreeNode::Node(root) => {
                assert_eq!(root.feature_idx, 0);
                assert!((root.threshold - 5.0).abs() < 1e-7);
                match root.left.as_ref() {
                    TreeNode::Node(left) => {
                        assert_eq!(left.feature_idx, 1);
                        assert!((left.threshold - 2.0).abs() < 1e-7);
                        match left.left.as_ref() {
                            TreeNode::Leaf(ll) => assert_eq!(ll.class_label, 0),
                            _ => panic!("expected leaf"),
                        }
                        match left.right.as_ref() {
                            TreeNode::Leaf(lr) => assert_eq!(lr.class_label, 1),
                            _ => panic!("expected leaf"),
                        }
                    }
                    _ => panic!("expected node"),
                }
                match root.right.as_ref() {
                    TreeNode::Leaf(r) => assert_eq!(r.class_label, 2),
                    _ => panic!("expected leaf"),
                }
            }
            _ => panic!("expected node at root"),
        }
    }

    // ========================================================================
    // Feature Importance with Nested Tree
    // ========================================================================

    #[test]
    fn test_compute_tree_feature_importances_nested() {
        // Root splits on feature 0, left child splits on feature 1
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 5.0,
            left: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 2.0,
                left: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 0,
                    n_samples: 2,
                })),
                right: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 1,
                    n_samples: 3,
                })),
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 2,
                n_samples: 5,
            })),
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&tree, &mut importances);
        // feature 0 at root: total samples = 2+3+5 = 10
        assert!((importances[0] - 10.0).abs() < 1e-7);
        // feature 1 at left subtree: total samples = 2+3 = 5
        assert!((importances[1] - 5.0).abs() < 1e-7);
        assert!((importances[2] - 0.0).abs() < 1e-7);
    }
