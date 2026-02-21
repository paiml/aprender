
#[test]
fn test_gradient_boosting_classifier_fit_mismatch() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        .expect("Matrix creation should succeed");
    let y = vec![0, 1]; // Mismatched length

    let mut gb = GradientBoostingClassifier::new();
    let result = gb.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_gradient_boosting_classifier_fit_empty() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
    let y: Vec<usize> = vec![];

    let mut gb = GradientBoostingClassifier::new();
    let result = gb.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_decision_tree_classifier_score() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    )
    .expect("Matrix creation should succeed");
    let y = vec![0, 0, 1, 1, 0, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("fit should succeed");

    let score = tree.score(&x, &y);
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_random_forest_classifier_score() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    )
    .expect("Matrix creation should succeed");
    let y = vec![0, 0, 1, 1, 0, 1];

    let mut rf = RandomForestClassifier::new(10).with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let score = rf.score(&x, &y);
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_random_forest_regressor_score_r2() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut rf = RandomForestRegressor::new(10).with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let score = rf.score(&x, &y);
    assert!(score > 0.0, "R² should be positive, got {score}");
}

#[test]
fn test_leaf_struct_fields() {
    let leaf = Leaf {
        class_label: 2,
        n_samples: 50,
    };
    assert_eq!(leaf.class_label, 2);
    assert_eq!(leaf.n_samples, 50);
}

#[test]
fn test_regression_leaf_struct_fields() {
    let leaf = RegressionLeaf {
        value: 3.14,
        n_samples: 25,
    };
    assert!((leaf.value - 3.14).abs() < 1e-6);
    assert_eq!(leaf.n_samples, 25);
}

#[test]
fn test_node_struct_fields() {
    let left = TreeNode::Leaf(Leaf {
        class_label: 0,
        n_samples: 5,
    });
    let right = TreeNode::Leaf(Leaf {
        class_label: 1,
        n_samples: 5,
    });

    let node = Node {
        feature_idx: 2,
        threshold: 1.5,
        left: Box::new(left),
        right: Box::new(right),
    };

    assert_eq!(node.feature_idx, 2);
    assert!((node.threshold - 1.5).abs() < 1e-6);
}

#[test]
fn test_regression_node_struct_fields() {
    let left = RegressionTreeNode::Leaf(RegressionLeaf {
        value: 1.0,
        n_samples: 5,
    });
    let right = RegressionTreeNode::Leaf(RegressionLeaf {
        value: 2.0,
        n_samples: 5,
    });

    let node = RegressionNode {
        feature_idx: 1,
        threshold: 0.75,
        left: Box::new(left),
        right: Box::new(right),
    };

    assert_eq!(node.feature_idx, 1);
    assert!((node.threshold - 0.75).abs() < 1e-6);
}

#[test]
fn test_tree_node_variants() {
    let leaf_node = TreeNode::Leaf(Leaf {
        class_label: 0,
        n_samples: 10,
    });
    assert_eq!(leaf_node.depth(), 0);

    let internal_node = TreeNode::Node(Node {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 5,
        })),
        right: Box::new(TreeNode::Leaf(Leaf {
            class_label: 1,
            n_samples: 5,
        })),
    });
    assert_eq!(internal_node.depth(), 1);
}

#[test]
fn test_regression_tree_node_variants() {
    let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
        value: 5.0,
        n_samples: 10,
    });
    assert_eq!(leaf.depth(), 0);

    let internal = RegressionTreeNode::Node(RegressionNode {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
            value: 3.0,
            n_samples: 5,
        })),
        right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
            value: 7.0,
            n_samples: 5,
        })),
    });
    assert_eq!(internal.depth(), 1);
}

// =========================================================================
// Coverage boost tests: Helper functions and edge cases
// =========================================================================

#[test]
fn test_mean_f32_empty() {
    let empty: Vec<f32> = vec![];
    assert_eq!(mean_f32(&empty), 0.0);
}

#[test]
fn test_mean_f32_single() {
    let single = vec![42.5];
    assert!((mean_f32(&single) - 42.5).abs() < 1e-6);
}

#[test]
fn test_mean_f32_multiple() {
    let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((mean_f32(&vals) - 3.0).abs() < 1e-6);
}

#[test]
fn test_variance_f32_empty() {
    let empty: Vec<f32> = vec![];
    assert_eq!(variance_f32(&empty), 0.0);
}

#[test]
fn test_variance_f32_single() {
    let single = vec![5.0];
    assert_eq!(variance_f32(&single), 0.0);
}

#[test]
fn test_variance_f32_uniform() {
    let uniform = vec![3.0, 3.0, 3.0, 3.0];
    assert!(variance_f32(&uniform) < 1e-6);
}

#[test]
fn test_variance_f32_variable() {
    // Values: 1, 2, 3 -> mean = 2, variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    let vals = vec![1.0, 2.0, 3.0];
    let expected = 2.0 / 3.0;
    assert!((variance_f32(&vals) - expected).abs() < 1e-6);
}

#[test]
fn test_compute_mse_empty() {
    let left: Vec<f32> = vec![];
    let right: Vec<f32> = vec![];
    assert_eq!(compute_mse(&left, &right), 0.0);
}

#[test]
fn test_compute_mse_pure_splits() {
    // Both partitions have constant values -> MSE = 0
    let left = vec![5.0, 5.0, 5.0];
    let right = vec![10.0, 10.0];
    assert!(compute_mse(&left, &right) < 1e-6);
}

#[test]
fn test_at_max_depth_none() {
    // No max depth -> never at max
    assert!(!at_max_depth(0, None));
    assert!(!at_max_depth(100, None));
    assert!(!at_max_depth(1000, None));
}

#[test]
fn test_at_max_depth_with_limit() {
    assert!(at_max_depth(5, Some(5)));
    assert!(at_max_depth(6, Some(5)));
    assert!(!at_max_depth(4, Some(5)));
    assert!(!at_max_depth(0, Some(5)));
}

#[test]
fn test_get_unique_feature_values() {
    let x = Matrix::from_vec(
        5,
        2,
        vec![
            1.0, 10.0, 2.0, 20.0, 1.0, 30.0, // duplicate for feature 0
            3.0, 20.0, // duplicate for feature 1
            2.0, 40.0, // duplicate for feature 0
        ],
    )
    .expect("Matrix creation should succeed");

    let unique_feat0 = get_unique_feature_values(&x, 0, 5);
    assert_eq!(unique_feat0, vec![1.0, 2.0, 3.0]);

    let unique_feat1 = get_unique_feature_values(&x, 1, 5);
    assert_eq!(unique_feat1, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_split_by_threshold() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Matrix creation should succeed");
    let y = vec![10.0, 20.0, 30.0, 40.0];

    let (left, right) = split_by_threshold(&x, &y, 0, 2.5);
    assert_eq!(left, vec![10.0, 20.0]);
    assert_eq!(right, vec![30.0, 40.0]);
}

#[test]
fn test_evaluate_split_gain_empty() {
    let left: Vec<f32> = vec![];
    let right = vec![1.0, 2.0];
    assert!(evaluate_split_gain(&left, &right, 1.0).is_none());
}

#[test]
fn test_evaluate_split_gain_positive() {
    // High variance before split, low variance after
    let left = vec![1.0, 1.0, 1.0];
    let right = vec![10.0, 10.0, 10.0];
    let current_variance = 20.0; // Arbitrary high variance
    let gain = evaluate_split_gain(&left, &right, current_variance);
    assert!(gain.is_some());
    assert!(gain.expect("Expected some gain") > 0.0);
}

#[test]
fn test_partition_by_threshold() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 5.0, 2.0, 8.0, 3.0])
        .expect("Matrix creation should succeed");
    let (left, right) = partition_by_threshold(&x, 5, 0, 3.5);
    assert_eq!(left, vec![0, 2, 4]); // indices with values <= 3.5
    assert_eq!(right, vec![1, 3]); // indices with values > 3.5
}

#[test]
fn test_gini_split_empty_partitions() {
    let left: Vec<usize> = vec![];
    let right: Vec<usize> = vec![];
    assert_eq!(gini_split(&left, &right), 0.0);
}

#[test]
fn test_get_sorted_unique_values() {
    let x = vec![3.0, 1.0, 2.0, 1.0, 3.0];
    let unique = get_sorted_unique_values(&x);
    assert_eq!(unique, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_split_labels_by_threshold_empty_result() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![0, 1, 2];
    // Threshold that puts all on one side
    assert!(split_labels_by_threshold(&x, &y, 10.0).is_none());
    assert!(split_labels_by_threshold(&x, &y, 0.0).is_none());
}

#[test]
fn test_split_labels_by_threshold_valid() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![0, 0, 1, 1];
    let result = split_labels_by_threshold(&x, &y, 2.5);
    assert!(result.is_some());
    let (left, right) = result.expect("Expected valid split");
    assert_eq!(left, vec![0, 0]);
    assert_eq!(right, vec![1, 1]);
}

#[test]
fn test_calculate_information_gain() {
    let left = vec![0, 0, 0];
    let right = vec![1, 1, 1];
    // Pure split should have maximum gain from impure parent
    let gain = calculate_information_gain(0.5, &left, &right);
    assert!((gain - 0.5).abs() < 1e-6);
}

#[test]
fn test_split_data_by_indices() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .expect("Matrix creation should succeed");
    let y = vec![10, 20, 30, 40];
    let indices = vec![1, 3];

    let (subset_x, subset_y) = split_data_by_indices(&x, &y, &indices);
    assert_eq!(subset_y, vec![20, 40]);
    assert_eq!(subset_x.shape(), (2, 2));
}

#[test]
fn test_check_stopping_criteria_pure_node() {
    let y = vec![5, 5, 5, 5]; // All same class
    let result = check_stopping_criteria(&y, 0, None);
    assert!(result.is_some());
    if let Some(TreeNode::Leaf(leaf)) = result {
        assert_eq!(leaf.class_label, 5);
        assert_eq!(leaf.n_samples, 4);
    } else {
        panic!("Expected leaf node");
    }
}

#[test]
fn test_check_stopping_criteria_max_depth() {
    let y = vec![0, 1, 0, 1]; // Mixed classes
    let result = check_stopping_criteria(&y, 5, Some(5));
    assert!(result.is_some());
}

#[test]
fn test_check_stopping_criteria_continue() {
    let y = vec![0, 1, 0, 1];
    let result = check_stopping_criteria(&y, 2, Some(5));
    assert!(result.is_none()); // Should continue splitting
}

#[test]
fn test_split_indices_by_threshold_empty() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Matrix creation should succeed");
    // All values <= threshold
    assert!(split_indices_by_threshold(&x, 0, 10.0, 3).is_none());
    // All values > threshold
    assert!(split_indices_by_threshold(&x, 0, 0.0, 3).is_none());
}

#[test]
fn test_count_tree_samples() {
    let leaf = TreeNode::Leaf(Leaf {
        class_label: 0,
        n_samples: 42,
    });
    assert_eq!(count_tree_samples(&leaf), 42);

    let node = TreeNode::Node(Node {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 10,
        })),
        right: Box::new(TreeNode::Leaf(Leaf {
            class_label: 1,
            n_samples: 20,
        })),
    });
    assert_eq!(count_tree_samples(&node), 30);
}
