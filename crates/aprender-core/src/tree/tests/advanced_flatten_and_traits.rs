
// =========================================================================
// Coverage boost: Tree flattening and reconstruction
// =========================================================================

#[test]
fn test_flatten_and_reconstruct_tree() {
    let original = TreeNode::Node(Node {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 5,
        })),
        right: Box::new(TreeNode::Node(Node {
            feature_idx: 1,
            threshold: 0.3,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 3,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 2,
                n_samples: 2,
            })),
        })),
    });

    let mut features = Vec::new();
    let mut thresholds = Vec::new();
    let mut classes = Vec::new();
    let mut samples = Vec::new();
    let mut left_children = Vec::new();
    let mut right_children = Vec::new();

    flatten_tree_node(
        &original,
        &mut features,
        &mut thresholds,
        &mut classes,
        &mut samples,
        &mut left_children,
        &mut right_children,
    );

    let reconstructed = reconstruct_tree_node(
        0,
        &features,
        &thresholds,
        &classes,
        &samples,
        &left_children,
        &right_children,
    );

    assert_eq!(original.depth(), reconstructed.depth());
}

// =========================================================================
// Coverage boost: Feature importance computation
// =========================================================================

#[test]
fn test_compute_tree_feature_importances() {
    let tree = TreeNode::Node(Node {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 10,
        })),
        right: Box::new(TreeNode::Node(Node {
            feature_idx: 1,
            threshold: 0.3,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 5,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 2,
                n_samples: 5,
            })),
        })),
    });

    let mut importances = vec![0.0; 2];
    compute_tree_feature_importances(&tree, &mut importances);

    // Feature 0 splits 20 samples, feature 1 splits 10 samples
    assert!(importances[0] > 0.0);
    assert!(importances[1] > 0.0);
}

#[test]
fn test_compute_regression_tree_feature_importances() {
    let tree = RegressionTreeNode::Node(RegressionNode {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
            value: 1.0,
            n_samples: 10,
        })),
        right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
            value: 2.0,
            n_samples: 10,
        })),
    });

    let mut importances = vec![0.0; 2];
    compute_regression_tree_feature_importances(&tree, &mut importances);

    assert!(importances[0] > 0.0);
    assert_eq!(importances[1], 0.0); // Feature 1 not used
}

// =========================================================================
// Coverage boost: Regression tree building edge cases
// =========================================================================

#[test]
fn test_split_regression_data_by_indices() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Matrix creation should succeed");
    let y = vec![10.0, 20.0, 30.0, 40.0];
    let indices = vec![0, 2];

    let (subset_x, subset_y) = split_regression_data_by_indices(&x, &y, &indices);
    assert_eq!(subset_y, vec![10.0, 30.0]);
    assert_eq!(subset_x.shape(), (2, 2));
}

#[test]
fn test_make_regression_leaf() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let leaf = make_regression_leaf(&y, 5);
    match leaf {
        RegressionTreeNode::Leaf(l) => {
            assert!((l.value - 3.0).abs() < 1e-6);
            assert_eq!(l.n_samples, 5);
        }
        _ => panic!("Expected leaf"),
    }
}

#[test]
fn test_find_best_regression_split_for_feature() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).expect("Matrix creation should succeed");
    let y = vec![1.0, 2.0, 10.0, 11.0];
    let variance = variance_f32(&y);

    let result = find_best_regression_split_for_feature(&x, &y, 0, 4, variance);
    assert!(result.is_some());
    let (threshold, gain) = result.expect("Expected split");
    assert!(threshold > 2.0 && threshold < 5.0);
    assert!(gain > 0.0);
}

#[test]
fn test_find_best_regression_split_no_samples() {
    let x = Matrix::from_vec(1, 1, vec![1.0]).expect("Matrix creation should succeed");
    let y = vec![1.0];

    let result = find_best_regression_split(&x, &y);
    assert!(result.is_none());
}

// =========================================================================
// Coverage boost: Estimator trait implementation
// =========================================================================

#[test]
fn test_estimator_trait_decision_tree() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).expect("Matrix creation should succeed");
    let y = vec![0, 0, 1, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(3);

    // Using Estimator trait methods
    tree.fit(&x, &y).expect("Fit should succeed");
    let score = tree.score(&x, &y);
    assert!(score > 0.9);
}
