//! Tests for decision tree algorithms.

use crate::primitives::{Matrix, Vector};
use crate::traits::Estimator;
use crate::tree::helpers::{
    bootstrap_sample, build_tree, find_best_split, find_best_split_for_feature, gini_impurity,
    gini_split, majority_class,
};
use crate::tree::*;

// RED Phase: Write failing tests first

#[test]
fn test_leaf_creation() {
    let leaf = Leaf {
        class_label: 1,
        n_samples: 10,
    };
    assert_eq!(leaf.class_label, 1);
    assert_eq!(leaf.n_samples, 10);
}

#[test]
fn test_node_creation() {
    let left = TreeNode::Leaf(Leaf {
        class_label: 0,
        n_samples: 5,
    });
    let right = TreeNode::Leaf(Leaf {
        class_label: 1,
        n_samples: 5,
    });

    let node = Node {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(left),
        right: Box::new(right),
    };

    assert_eq!(node.feature_idx, 0);
    assert!((node.threshold - 0.5).abs() < 1e-6);
}

#[test]
fn test_tree_depth() {
    // Leaf has depth 0
    let leaf = TreeNode::Leaf(Leaf {
        class_label: 0,
        n_samples: 1,
    });
    assert_eq!(leaf.depth(), 0);

    // Tree with one split has depth 1
    let tree = TreeNode::Node(Node {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 1,
        })),
        right: Box::new(TreeNode::Leaf(Leaf {
            class_label: 1,
            n_samples: 1,
        })),
    });
    assert_eq!(tree.depth(), 1);

    // Tree with nested splits has depth 2
    let deep_tree = TreeNode::Node(Node {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(tree),
        right: Box::new(TreeNode::Leaf(Leaf {
            class_label: 1,
            n_samples: 1,
        })),
    });
    assert_eq!(deep_tree.depth(), 2);
}

#[test]
fn test_decision_tree_creation() {
    let tree = DecisionTreeClassifier::new();
    assert!(tree.tree.is_none());
    assert!(tree.max_depth.is_none());
}

#[test]
fn test_decision_tree_with_max_depth() {
    let tree = DecisionTreeClassifier::new().with_max_depth(5);
    assert_eq!(tree.max_depth, Some(5));
}

#[test]
fn test_decision_tree_default() {
    let tree = DecisionTreeClassifier::default();
    assert!(tree.tree.is_none());
    assert!(tree.max_depth.is_none());
}

// RED-GREEN-REFACTOR Cycle 2: Gini Impurity

#[test]
fn test_gini_impurity_pure() {
    // Pure node (all same class) should have Gini = 0.0
    let pure = vec![0, 0, 0, 0, 0];
    assert!((gini_impurity(&pure) - 0.0).abs() < 1e-6);

    let pure_ones = vec![1, 1, 1];
    assert!((gini_impurity(&pure_ones) - 0.0).abs() < 1e-6);
}

#[test]
fn test_gini_impurity_empty() {
    // Empty set should have Gini = 0.0
    let empty: Vec<usize> = vec![];
    assert!((gini_impurity(&empty) - 0.0).abs() < 1e-6);
}

#[test]
fn test_gini_impurity_binary_50_50() {
    // 50/50 binary split should have Gini = 0.5
    let mixed = vec![0, 1, 0, 1];
    assert!((gini_impurity(&mixed) - 0.5).abs() < 1e-6);
}

#[test]
fn test_gini_impurity_three_class_even() {
    // Three classes evenly distributed: Gini = 1 - 3*(1/3)² = 1 - 1/3 = 0.6667
    let three_class = vec![0, 1, 2, 0, 1, 2];
    assert!((gini_impurity(&three_class) - 0.6667).abs() < 1e-4);
}

#[test]
fn test_gini_impurity_bounds() {
    // Gini impurity should always be in [0, 1]
    let labels_sets = vec![
        vec![0, 0, 0],
        vec![0, 1],
        vec![0, 1, 2],
        vec![0, 0, 1, 1, 2, 2],
        vec![0, 0, 0, 1],
    ];

    for labels in labels_sets {
        let gini = gini_impurity(&labels);
        assert!(gini >= 0.0, "Gini should be >= 0, got {gini}");
        assert!(gini <= 1.0, "Gini should be <= 1, got {gini}");
    }
}

#[test]
fn test_gini_split_calculation() {
    // Test weighted Gini for a split
    let left = vec![0, 0, 0]; // Pure, Gini = 0.0
    let right = vec![1, 1, 1]; // Pure, Gini = 0.0
                               // Weighted: (3/6)*0.0 + (3/6)*0.0 = 0.0
    assert!((gini_split(&left, &right) - 0.0).abs() < 1e-6);
}

#[test]
fn test_gini_split_mixed() {
    // Left: [0, 0, 1] - Gini = 1 - (2/3)² - (1/3)² = 1 - 4/9 - 1/9 = 4/9 ≈ 0.4444
    // Right: [1, 1] - Gini = 0.0
    // Weighted: (3/5)*0.4444 + (2/5)*0.0 ≈ 0.2667
    let left = vec![0, 0, 1];
    let right = vec![1, 1];
    let expected = (3.0 / 5.0) * (4.0 / 9.0);
    assert!((gini_split(&left, &right) - expected).abs() < 1e-4);
}

// RED-GREEN-REFACTOR Cycle 3: Best Split Finding

#[test]
fn test_find_best_split_simple() {
    // Simple linearly separable data
    // x: [1.0, 2.0, 5.0, 6.0]
    // y: [0, 0, 1, 1]
    // Best split should be around 3.5 (midpoint between 2 and 5)
    let x = vec![1.0, 2.0, 5.0, 6.0];
    let y = vec![0, 0, 1, 1];

    let result = find_best_split_for_feature(&x, &y);
    assert!(result.is_some());

    let (threshold, gain) = result.expect("should have valid result");
    // Threshold should be between 2.0 and 5.0
    assert!(threshold > 2.0 && threshold < 5.0);
    // Gain should be positive (we're reducing impurity)
    assert!(gain > 0.0);
}

#[test]
fn test_find_best_split_no_gain() {
    // All same label - no possible gain
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![0, 0, 0, 0];

    let result = find_best_split_for_feature(&x, &y);
    // Should return None since there's no gain possible
    assert!(result.is_none());
}

#[test]
fn test_find_best_split_too_small() {
    // Only one sample
    let x = vec![1.0];
    let y = vec![0];

    let result = find_best_split_for_feature(&x, &y);
    assert!(result.is_none());
}

#[test]
fn test_find_best_split_gain_is_positive() {
    // Property: Gain should always be >= 0
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![0, 1, 0, 1];

    if let Some((_threshold, gain)) = find_best_split_for_feature(&x, &y) {
        assert!(gain >= 0.0);
    }
}

#[test]
fn test_find_best_split_across_features() {
    use crate::primitives::Matrix;

    // 2D data with 2 features
    // Feature 0: [1.0, 1.0, 5.0, 5.0]
    // Feature 1: [1.0, 2.0, 5.0, 6.0]
    // Labels:    [  0,   0,   1,   1]
    // Both features separate perfectly, should choose one
    let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let result = find_best_split(&x, &y);
    assert!(result.is_some());

    let (feature_idx, _threshold, gain) = result.expect("should have valid result");
    // Should choose one of the features
    assert!(feature_idx < 2);
    // Should have positive gain
    assert!(gain > 0.0);
}

#[test]
fn test_find_best_split_perfect_separation() {
    use crate::primitives::Matrix;

    // Perfectly separable data on feature 0
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let result = find_best_split(&x, &y);
    assert!(result.is_some());

    let (feature_idx, threshold, gain) = result.expect("should have valid result");
    assert_eq!(feature_idx, 0); // Should choose the only feature
    assert!(threshold > 2.0 && threshold < 5.0);
    // Gain should be maximum (from mixed to pure)
    assert!(gain > 0.4); // At least some significant gain
}

// RED-GREEN-REFACTOR Cycle 4: Tree Building

#[test]
fn test_majority_class_simple() {
    let labels = vec![0, 0, 1, 0, 1];
    // Class 0 appears 3 times, class 1 appears 2 times
    assert_eq!(majority_class(&labels), 0);
}

#[test]
fn test_majority_class_tie() {
    let labels = vec![0, 1, 0, 1];
    // Tie - either 0 or 1 is acceptable
    let result = majority_class(&labels);
    assert!(result == 0 || result == 1);
}

#[test]
fn test_majority_class_single() {
    let labels = vec![5];
    assert_eq!(majority_class(&labels), 5);
}

#[test]
fn test_build_tree_pure_leaf() {
    use crate::primitives::Matrix;

    // All same label - should create a leaf immediately
    let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0];

    let tree = build_tree(&x, &y, 0, None);

    // Should be a leaf with class 0
    match tree {
        TreeNode::Leaf(leaf) => {
            assert_eq!(leaf.class_label, 0);
            assert_eq!(leaf.n_samples, 3);
        }
        TreeNode::Node(_) => panic!("Expected Leaf node for pure data"),
    }
}

#[test]
fn test_build_tree_max_depth_zero() {
    use crate::primitives::Matrix;

    // Mixed labels but max_depth=0, should create leaf with majority
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let tree = build_tree(&x, &y, 0, Some(0));

    // Should be a leaf (can't split at depth 0)
    match tree {
        TreeNode::Leaf(leaf) => {
            assert!(leaf.class_label == 0 || leaf.class_label == 1);
            assert_eq!(leaf.n_samples, 4);
        }
        TreeNode::Node(_) => panic!("Expected Leaf node at max depth"),
    }
}

#[test]
fn test_build_tree_simple_split() {
    use crate::primitives::Matrix;

    // Simple binary split
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let tree = build_tree(&x, &y, 0, Some(5));

    // Should be a node (not pure, depth allows split)
    match tree {
        TreeNode::Node(node) => {
            assert_eq!(node.feature_idx, 0); // Only one feature
            assert!(node.threshold > 2.0 && node.threshold < 5.0);
        }
        TreeNode::Leaf(_) => panic!("Expected Node for splittable data"),
    }
}

#[test]
fn test_build_tree_depth_tracking() {
    use crate::primitives::Matrix;

    // Build tree and verify depth is respected
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let tree = build_tree(&x, &y, 0, Some(1));

    // Depth should be <= 1
    assert!(tree.depth() <= 1);
}

// RED-GREEN-REFACTOR Cycle 5: fit/predict/score

#[test]
fn test_fit_simple() {
    use crate::primitives::Matrix;

    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    let result = tree.fit(&x, &y);

    assert!(result.is_ok());
    assert!(tree.tree.is_some()); // Tree should be built
}

#[test]
fn test_predict_perfect_classification() {
    use crate::primitives::Matrix;

    // Perfectly separable data
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);
    assert_eq!(predictions, vec![0, 0, 1, 1]);
}

#[test]
fn test_predict_single_sample() {
    use crate::primitives::Matrix;

    let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y_train = vec![0, 0, 1, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x_train, &y_train).expect("fit should succeed");

    // Test single sample prediction
    let x_test =
        Matrix::from_vec(1, 1, vec![1.5]).expect("Matrix creation should succeed in tests");
    let predictions = tree.predict(&x_test);
    assert_eq!(predictions.len(), 1);
    assert_eq!(predictions[0], 0); // Should be class 0 (closer to 1.0, 2.0)
}

#[test]
fn test_score_perfect() {
    use crate::primitives::Matrix;

    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("fit should succeed");

    let accuracy = tree.score(&x, &y);
    assert!((accuracy - 1.0).abs() < 1e-6); // Perfect classification
}

include!("core_part_02.rs");
include!("core_part_03.rs");
include!("core_part_04.rs");
