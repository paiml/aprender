
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
            impurity: 0.0,
        });
        assert_eq!(count_tree_samples(&leaf), 42);
    }

    #[test]
    fn test_count_tree_samples_tree() {
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 1.0,
            impurity: 0.0,
            n_node_samples: 0,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 10,
                impurity: 0.0,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 20,
                impurity: 0.0,
            })),
        });
        assert_eq!(count_tree_samples(&tree), 30);
    }

    #[test]
    fn test_compute_tree_feature_importances_leaf_only() {
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 10,
            impurity: 0.0,
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&leaf, &mut importances);
        assert_eq!(importances, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_tree_feature_importances_single_split() {
        // MDI: feature 1 split over 8 samples, root gini 0.5, pure children.
        // importance = 8*0.5 - 5*0.0 - 3*0.0 = 4.0
        let tree = TreeNode::Node(Node {
            feature_idx: 1,
            threshold: 2.0,
            impurity: 0.5,
            n_node_samples: 8,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 5,
                impurity: 0.0,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 3,
                impurity: 0.0,
            })),
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&tree, &mut importances);
        assert!((importances[0] - 0.0).abs() < 1e-7);
        assert!((importances[1] - 4.0).abs() < 1e-7);
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
            impurity: 0.0,
        });
        assert_eq!(count_regression_tree_samples(&leaf), 15);
    }

    #[test]
    fn test_count_regression_tree_samples_tree() {
        let tree = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 1.0,
            impurity: 0.0,
            n_node_samples: 0,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 1.0,
                n_samples: 7,
                impurity: 0.0,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 5.0,
                n_samples: 13,
                impurity: 0.0,
            })),
        });
        assert_eq!(count_regression_tree_samples(&tree), 20);
    }

    #[test]
    fn test_compute_regression_tree_feature_importances() {
        // MDI: feature 2 split over 10 samples, root variance 2.0, pure leaves.
        // importance = 10*2.0 - 4*0.0 - 6*0.0 = 20.0
        let tree = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 2,
            threshold: 3.0,
            impurity: 2.0,
            n_node_samples: 10,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 1.0,
                n_samples: 4,
                impurity: 0.0,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 9.0,
                n_samples: 6,
                impurity: 0.0,
            })),
        });
        let mut importances = vec![0.0; 4];
        compute_regression_tree_feature_importances(&tree, &mut importances);
        assert!((importances[2] - 20.0).abs() < 1e-7);
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
            impurity: 0.0,
            n_node_samples: 0,
            left: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 2.0,
                impurity: 0.0,
                n_node_samples: 0,
                left: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 0,
                    n_samples: 2,
                    impurity: 0.0,
                })),
                right: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 1,
                    n_samples: 3,
                    impurity: 0.0,
                })),
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 2,
                n_samples: 5,
                impurity: 0.0,
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
        // Root splits on feature 0 (n=10, gini 0.6), left child splits on
        // feature 1 (n=5, gini 0.48). Children pure. MDI per split:
        //   feature 0 = 10*0.6 - 5*0.48 - 5*0.0 = 3.6
        //   feature 1 =  5*0.48 - 2*0.0  - 3*0.0 = 2.4
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 5.0,
            impurity: 0.6,
            n_node_samples: 10,
            left: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 2.0,
                impurity: 0.48,
                n_node_samples: 5,
                left: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 0,
                    n_samples: 2,
                    impurity: 0.0,
                })),
                right: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 1,
                    n_samples: 3,
                    impurity: 0.0,
                })),
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 2,
                n_samples: 5,
                impurity: 0.0,
            })),
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&tree, &mut importances);
        assert!((importances[0] - 3.6).abs() < 1e-6);
        assert!((importances[1] - 2.4).abs() < 1e-6);
        assert!((importances[2] - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // PMAT-851 FALSIFIER: MDI feature-importance must reflect impurity decrease,
    // not raw split sample-count.
    //
    // Repro tree (regression):
    //   root: split feature 0 over 20 samples
    //     left  -> leaf value=1.0 n=10 (var 0)
    //     right -> internal split feature 1 over 10 samples (var ~2500 -> 0)
    //                left  -> leaf value=0.0 n=5
    //                right -> leaf value=100.0 n=5
    //
    // Feature 1 fully separates a high-variance subset (variance 2500 -> 0), so
    // its impurity decrease (25000) dwarfs feature 0's (root variance is small).
    // The OLD count-only code attributed [20.0, 10.0] -> feature 0 ranked higher.
    // Correct MDI ranks feature 1 ABOVE feature 0.
    // ========================================================================
    #[test]
    fn test_regression_mdi_outranks_by_variance_decrease_not_count() {
        // ten 1.0, five 0.0, five 100.0
        let y_root: Vec<f32> = std::iter::repeat_n(1.0_f32, 10)
            .chain(std::iter::repeat_n(0.0_f32, 5))
            .chain(std::iter::repeat_n(100.0_f32, 5))
            .collect();
        let root_var = variance_f32(&y_root);
        let right_var = variance_f32(&[0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0]);

        let tree = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 0.5,
            impurity: root_var,
            n_node_samples: 20,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 1.0,
                n_samples: 10,
                impurity: 0.0,
            })),
            right: Box::new(RegressionTreeNode::Node(RegressionNode {
                feature_idx: 1,
                threshold: 50.0,
                impurity: right_var,
                n_node_samples: 10,
                left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                    value: 0.0,
                    n_samples: 5,
                    impurity: 0.0,
                })),
                right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                    value: 100.0,
                    n_samples: 5,
                    impurity: 0.0,
                })),
            })),
        });

        let mut importances = vec![0.0_f32; 2];
        compute_regression_tree_feature_importances(&tree, &mut importances);

        // FALSIFIER: feature 1's variance drop must outrank feature 0.
        // (Old count-only code yields [20.0, 10.0] -> feature 0 wins, FAILS this.)
        assert!(
            importances[1] > importances[0],
            "MDI must rank feature 1 > feature 0; got {importances:?}"
        );

        // Weighted-decrease formula check:
        //   feature 0 = 20*root_var - 10*0 - 10*right_var
        //   feature 1 = 10*right_var - 5*0 - 5*0
        let expected_f0 = 20.0 * root_var - 10.0 * right_var;
        let expected_f1 = 10.0 * right_var;
        assert!(
            (importances[0] - expected_f0).abs() < 1e-1,
            "feature 0 MDI mismatch: got {}, want {expected_f0}",
            importances[0]
        );
        assert!(
            (importances[1] - expected_f1).abs() < 1e-1,
            "feature 1 MDI mismatch: got {}, want {expected_f1}",
            importances[1]
        );
    }

    // PMAT-851 classification analog: gini impurity decrease, not raw count.
    #[test]
    fn test_classification_mdi_uses_gini_decrease_not_count() {
        // Root split (feature 0) over 12 samples: gini 0.5. Left child (feature 1)
        // splits a 6-sample mixed subset (gini 0.5 -> pure). Right leaf pure.
        //   feature 0 = 12*0.5 - 6*0.0 - 6*0.5 = 6 - 3 = 3.0
        //   feature 1 =  6*0.5 - 3*0.0 - 3*0.0 = 3.0
        // Count-only code would give feature 0 = 12 (the whole subtree) > feature 1 = 6.
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 0.5,
            impurity: 0.5,
            n_node_samples: 12,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 6,
                impurity: 0.0,
            })),
            right: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 0.5,
                impurity: 0.5,
                n_node_samples: 6,
                left: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 1,
                    n_samples: 3,
                    impurity: 0.0,
                })),
                right: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 2,
                    n_samples: 3,
                    impurity: 0.0,
                })),
            })),
        });
        let mut importances = vec![0.0_f32; 2];
        compute_tree_feature_importances(&tree, &mut importances);
        assert!((importances[0] - 3.0).abs() < 1e-6, "got {importances:?}");
        assert!((importances[1] - 3.0).abs() < 1e-6, "got {importances:?}");
    }
