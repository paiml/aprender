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

#[test]
fn test_score_partial() {
    use crate::primitives::Matrix;

    // Train on simple data
    let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y_train = vec![0, 0, 1, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(1);
    tree.fit(&x_train, &y_train).expect("fit should succeed");

    // Score should be between 0 and 1
    let accuracy = tree.score(&x_train, &y_train);
    assert!((0.0..=1.0).contains(&accuracy));
}

#[test]
fn test_multiclass_classification() {
    use crate::primitives::Matrix;

    // 3-class problem
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);
    assert_eq!(predictions.len(), 6);
    // Should classify perfectly
    assert_eq!(predictions, vec![0, 0, 1, 1, 2, 2]);
}

#[test]
fn test_save_load() {
    use crate::primitives::Matrix;
    use std::fs;
    use std::path::Path;

    // Train a tree
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("fit should succeed");

    // Save model
    let path = Path::new("/tmp/test_decision_tree.bin");
    tree.save(path).expect("Failed to save model");

    // Load model
    let loaded = DecisionTreeClassifier::load(path).expect("Failed to load model");

    // Verify predictions match
    let original_pred = tree.predict(&x);
    let loaded_pred = loaded.predict(&x);
    assert_eq!(original_pred, loaded_pred);

    // Verify accuracy matches
    let original_score = tree.score(&x, &y);
    let loaded_score = loaded.score(&x, &y);
    assert!((original_score - loaded_score).abs() < 1e-6);

    // Cleanup
    fs::remove_file(path).ok();
}

// Random Forest Tests

#[test]
fn test_bootstrap_sample_size() {
    let indices = bootstrap_sample(100, Some(42));
    assert_eq!(
        indices.len(),
        100,
        "Bootstrap sample should have same size as original"
    );
}

#[test]
fn test_bootstrap_sample_reproducible() {
    let indices1 = bootstrap_sample(50, Some(42));
    let indices2 = bootstrap_sample(50, Some(42));
    assert_eq!(
        indices1, indices2,
        "Same seed should give same bootstrap sample"
    );
}

#[test]
fn test_random_forest_creation() {
    let rf = RandomForestClassifier::new(10);
    assert_eq!(rf.n_estimators, 10);
}

#[test]
fn test_random_forest_builder() {
    let rf = RandomForestClassifier::new(5)
        .with_max_depth(3)
        .with_random_state(42);
    assert_eq!(rf.n_estimators, 5);
    assert_eq!(rf.max_depth, Some(3));
    assert_eq!(rf.random_state, Some(42));
}

#[test]
fn test_random_forest_fit_basic() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2];

    let mut rf = RandomForestClassifier::new(3)
        .with_max_depth(3)
        .with_random_state(42);

    rf.fit(&x, &y).expect("Fit should succeed");

    // Should have trained the correct number of trees
    assert_eq!(rf.trees.len(), 3, "Should have 3 trees");
}

#[test]
fn test_random_forest_predict() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2];

    let mut rf = RandomForestClassifier::new(5)
        .with_max_depth(5)
        .with_random_state(42);

    rf.fit(&x, &y).expect("fit should succeed");
    let predictions = rf.predict(&x);

    assert_eq!(predictions.len(), 6, "Should predict for all samples");

    // Perfect separation - should get perfect accuracy
    let score = rf.score(&x, &y);
    assert!(
        score > 0.8,
        "Random Forest should achieve >80% accuracy on simple data"
    );
}

#[test]
fn test_random_forest_reproducible() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, // class 0
            0.5, 0.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            10.0, 10.0, // class 2
            10.5, 10.5, // class 2
            1.0, 1.0, // class 0
            6.0, 6.0, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2, 0, 1];

    let mut rf1 = RandomForestClassifier::new(5).with_random_state(42);
    rf1.fit(&x, &y).expect("fit should succeed");
    let pred1 = rf1.predict(&x);

    let mut rf2 = RandomForestClassifier::new(5).with_random_state(42);
    rf2.fit(&x, &y).expect("fit should succeed");
    let pred2 = rf2.predict(&x);

    assert_eq!(
        pred1, pred2,
        "Same random state should give same predictions"
    );
}

// ===== Gradient Boosting Tests =====

#[test]
fn test_gradient_boosting_new() {
    let gbm = GradientBoostingClassifier::new();
    assert_eq!(gbm.configured_n_estimators(), 100);
    assert!((gbm.learning_rate() - 0.1).abs() < 1e-6);
    assert_eq!(gbm.max_depth(), 3);
    assert_eq!(gbm.n_estimators(), 0); // No estimators before fit
}

#[test]
fn test_gradient_boosting_builder() {
    let gbm = GradientBoostingClassifier::new()
        .with_n_estimators(50)
        .with_learning_rate(0.05)
        .with_max_depth(5);

    assert_eq!(gbm.configured_n_estimators(), 50);
    assert!((gbm.learning_rate() - 0.05).abs() < 1e-6);
    assert_eq!(gbm.max_depth(), 5);
}

#[test]
fn test_gradient_boosting_fit_simple() {
    // Simple linearly separable data
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 1
            1.0, 1.0, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_learning_rate(0.1)
        .with_max_depth(2);

    let result = gbm.fit(&x, &y);
    assert!(result.is_ok());
    assert!(gbm.n_estimators() > 0); // Should have fitted some trees
}

#[test]
fn test_gradient_boosting_predict_simple() {
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 1
            1.0, 1.0, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .with_learning_rate(0.1)
        .with_max_depth(2);

    gbm.fit(&x, &y).expect("fit should succeed");
    let predictions = gbm.predict(&x).expect("predict should succeed");

    assert_eq!(predictions.len(), 4);

    // GBM should classify correctly with enough iterations
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| *pred == *true_label)
        .count();

    // Should get at least 3 out of 4 correct
    assert!(correct >= 3);
}

#[test]
fn test_gradient_boosting_predict_proba() {
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 1
            1.0, 1.0, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .with_learning_rate(0.1)
        .with_max_depth(2);

    gbm.fit(&x, &y).expect("fit should succeed");
    let probas = gbm.predict_proba(&x).expect("predict_proba should succeed");

    assert_eq!(probas.len(), 4);

    // Each sample should have 2 probabilities (class 0 and class 1)
    for probs in &probas {
        assert_eq!(probs.len(), 2);
        // Probabilities should sum to ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        // Each probability should be between 0 and 1
        for &p in probs {
            assert!((0.0..=1.0).contains(&p));
        }
    }
}

#[test]
fn test_gradient_boosting_predict_untrained() {
    let gbm = GradientBoostingClassifier::new();
    let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation should succeed in tests");

    let result = gbm.predict(&x);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail when predicting with untrained model"),
        "Model not trained yet"
    );
}

#[test]
fn test_gradient_boosting_empty_data() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed in tests");
    let y = vec![];

    let mut gbm = GradientBoostingClassifier::new();
    let result = gbm.fit(&x, &y);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with empty data"),
        "Cannot fit with 0 samples"
    );
}

#[test]
fn test_gradient_boosting_mismatched_samples() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1]; // Wrong length

    let mut gbm = GradientBoostingClassifier::new();
    let result = gbm.fit(&x, &y);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with mismatched sample counts"),
        "x and y must have the same number of samples"
    );
}

#[test]
fn test_gradient_boosting_learning_rate_effect() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, // class 0
            0.1, 0.1, // class 0
            0.0, 0.2, // class 0
            1.0, 1.0, // class 1
            0.9, 0.9, // class 1
            1.0, 0.8, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0, 1, 1, 1];

    // High learning rate
    let mut gbm_high_lr = GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_learning_rate(0.5);
    gbm_high_lr.fit(&x, &y).expect("fit should succeed");
    let pred_high = gbm_high_lr.predict(&x).expect("predict should succeed");

    // Low learning rate
    let mut gbm_low_lr = GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_learning_rate(0.01);
    gbm_low_lr.fit(&x, &y).expect("fit should succeed");
    let pred_low = gbm_low_lr.predict(&x).expect("predict should succeed");

    // Both should make predictions
    assert_eq!(pred_high.len(), 6);
    assert_eq!(pred_low.len(), 6);
}

#[test]
fn test_gradient_boosting_n_estimators_effect() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2, // class 0
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0, 1, 1, 1];

    // Few estimators
    let mut gbm_few = GradientBoostingClassifier::new()
        .with_n_estimators(5)
        .with_learning_rate(0.1);
    gbm_few.fit(&x, &y).expect("fit should succeed");

    // Many estimators
    let mut gbm_many = GradientBoostingClassifier::new()
        .with_n_estimators(50)
        .with_learning_rate(0.1);
    gbm_many.fit(&x, &y).expect("fit should succeed");

    // More estimators should generally lead to more trees (up to limit)
    assert!(gbm_many.n_estimators() >= gbm_few.n_estimators());
}

#[test]
fn test_gradient_boosting_max_depth_effect() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2, // class 0
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0, 1, 1, 1];

    // Shallow trees
    let mut gbm_shallow = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .with_max_depth(1);
    gbm_shallow.fit(&x, &y).expect("fit should succeed");
    let pred_shallow = gbm_shallow.predict(&x).expect("predict should succeed");

    // Deeper trees
    let mut gbm_deep = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .with_max_depth(5);
    gbm_deep.fit(&x, &y).expect("fit should succeed");
    let pred_deep = gbm_deep.predict(&x).expect("predict should succeed");

    // Both should make predictions
    assert_eq!(pred_shallow.len(), 6);
    assert_eq!(pred_deep.len(), 6);
}

#[test]
fn test_gradient_boosting_binary_classification() {
    // More realistic binary classification problem
    let x = Matrix::from_vec(
        10,
        2,
        vec![
            // Class 0 (bottom-left cluster)
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.0, 0.1, 0.2, // Class 1 (top-right cluster)
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8, 0.8, 1.0, 0.9, 1.1,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(30)
        .with_learning_rate(0.1)
        .with_max_depth(3);

    gbm.fit(&x, &y).expect("fit should succeed");
    let predictions = gbm.predict(&x).expect("predict should succeed");

    // Should achieve reasonable accuracy
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| *pred == *true_label)
        .count();

    // Should get at least 7 out of 10 correct for well-separated clusters
    assert!(
        correct >= 7,
        "Expected at least 7/10 correct, got {correct}/10"
    );
}

#[test]
fn test_gradient_boosting_default() {
    let gbm1 = GradientBoostingClassifier::new();
    let gbm2 = GradientBoostingClassifier::default();

    assert_eq!(
        gbm1.configured_n_estimators(),
        gbm2.configured_n_estimators()
    );
    assert!((gbm1.learning_rate() - gbm2.learning_rate()).abs() < 1e-6);
    assert_eq!(gbm1.max_depth(), gbm2.max_depth());
}

// ========================================================================
// Decision Tree Regression Tests (RED Phase - Issue #29)
// ========================================================================

#[test]
fn test_regression_tree_creation() {
    let tree = DecisionTreeRegressor::new();
    assert!(tree.tree.is_none());
    assert!(tree.max_depth.is_none());
}

#[test]
fn test_regression_tree_with_max_depth() {
    let tree = DecisionTreeRegressor::new().with_max_depth(5);
    assert_eq!(tree.max_depth, Some(5));
}

#[test]
fn test_regression_tree_fit_simple_linear() {
    // Simple linear relationship: y = 2x + 1
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);

    // Tree should learn the linear pattern reasonably well
    let pred_slice = predictions.as_slice();
    let y_slice = y.as_slice();
    for i in 0..predictions.len() {
        assert!(
            (pred_slice[i] - y_slice[i]).abs() < 2.0,
            "Prediction {} too far from true value {}",
            pred_slice[i],
            y_slice[i]
        );
    }
}

#[test]
fn test_regression_tree_predict_nonlinear() {
    // Quadratic relationship: y = x^2
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);

    // Should capture quadratic pattern with enough depth
    let mut mse_sum = 0.0_f32;
    let pred_slice = predictions.as_slice();
    let y_slice = y.as_slice();
    for i in 0..predictions.len() {
        mse_sum += (pred_slice[i] - y_slice[i]).powi(2);
    }
    let mse = mse_sum / predictions.len() as f32;

    assert!(mse < 50.0, "MSE {mse} too high for quadratic fit");
}

#[test]
fn test_regression_tree_score() {
    // Perfect predictions should give R² = 1.0
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
    tree.fit(&x, &y).expect("fit should succeed");

    let r2 = tree.score(&x, &y);

    // R² should be high for training data
    assert!(r2 > 0.5, "R² score {r2} too low");
    assert!(r2 <= 1.0, "R² score {r2} exceeds maximum");
}

#[test]
fn test_regression_tree_max_depth_limits_complexity() {
    let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

    // Shallow tree
    let mut tree_shallow = DecisionTreeRegressor::new().with_max_depth(1);
    tree_shallow.fit(&x, &y).expect("fit should succeed");
    let depth_shallow = tree_shallow
        .tree
        .as_ref()
        .expect("tree should exist after fit")
        .depth();
    assert!(
        depth_shallow <= 1,
        "Shallow tree depth {depth_shallow} exceeds max"
    );

    // Deep tree
    let mut tree_deep = DecisionTreeRegressor::new().with_max_depth(5);
    tree_deep.fit(&x, &y).expect("fit should succeed");
    let depth_deep = tree_deep
        .tree
        .as_ref()
        .expect("tree should exist after fit")
        .depth();
    assert!(depth_deep <= 5, "Deep tree depth {depth_deep} exceeds max");

    // Deeper tree should fit better
    let r2_shallow = tree_shallow.score(&x, &y);
    let r2_deep = tree_deep.score(&x, &y);
    assert!(
        r2_deep >= r2_shallow,
        "Deeper tree R²={r2_deep} should be >= shallow tree R²={r2_shallow}"
    );
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_regression_tree_predict_before_fit_panics() {
    let tree = DecisionTreeRegressor::new();
    let x =
        Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Matrix creation should succeed in tests");
    let _ = tree.predict(&x); // Should panic
}

#[test]
fn test_regression_tree_multidimensional_features() {
    // 2D features: y = x1 + 2*x2
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // y = 3
            2.0, 1.0, // y = 4
            1.0, 2.0, // y = 5
            2.0, 2.0, // y = 6
            3.0, 1.0, // y = 5
            1.0, 3.0, // y = 7
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[3.0, 4.0, 5.0, 6.0, 5.0, 7.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
    tree.fit(&x, &y).expect("fit should succeed");

    let r2 = tree.score(&x, &y);
    assert!(r2 > 0.5, "R² score {r2} too low for 2D features");
}

#[test]
fn test_regression_tree_constant_target() {
    // All y values the same
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);

    let mut tree = DecisionTreeRegressor::new();
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);

    // Should predict the constant value
    for &pred in predictions.as_slice() {
        assert!(
            (pred - 5.0).abs() < 1e-5,
            "Prediction {pred} should be 5.0 for constant target"
        );
    }
}

#[test]
fn test_regression_tree_single_sample() {
    let x = Matrix::from_vec(1, 1, vec![5.0]).expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[10.0]);

    let mut tree = DecisionTreeRegressor::new();
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);
    assert_eq!(predictions.len(), 1);
    assert!((predictions[0] - 10.0).abs() < 1e-5);
}

#[test]
fn test_regression_tree_fit_validation() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong size

    let mut tree = DecisionTreeRegressor::new();
    let result = tree.fit(&x, &y);

    assert!(result.is_err(), "Should error on mismatched dimensions");
}

#[test]
fn test_regression_tree_zero_samples() {
    let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[]);

    let mut tree = DecisionTreeRegressor::new();
    let result = tree.fit(&x, &y);

    assert!(result.is_err(), "Should error on zero samples");
}

#[test]
fn test_regression_tree_min_samples_split() {
    let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);

    // Tree with min_samples_split=4 should not split nodes with fewer samples
    let mut tree = DecisionTreeRegressor::new()
        .with_max_depth(5)
        .with_min_samples_split(4);

    tree.fit(&x, &y).expect("fit should succeed");

    // Should still fit successfully
    let r2 = tree.score(&x, &y);
    assert!(r2 > 0.0, "Tree with min_samples_split should still fit");
}

#[test]
fn test_regression_tree_min_samples_leaf() {
    let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

    // Tree with min_samples_leaf=3 should ensure leaves have at least 3 samples
    let mut tree = DecisionTreeRegressor::new()
        .with_max_depth(5)
        .with_min_samples_leaf(3);

    tree.fit(&x, &y).expect("fit should succeed");

    // Should fit without error
    let predictions = tree.predict(&x);
    assert_eq!(predictions.len(), 8);
}

#[test]
fn test_regression_tree_default() {
    let tree1 = DecisionTreeRegressor::new();
    let tree2 = DecisionTreeRegressor::default();

    assert_eq!(tree1.max_depth, tree2.max_depth);
    assert_eq!(tree1.tree.is_none(), tree2.tree.is_none());
}

#[test]
fn test_regression_tree_comparison_with_linear_regression() {
    // On perfectly linear data, both should perform well
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    // Train tree
    let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
    tree.fit(&x, &y).expect("fit should succeed");
    let tree_r2 = tree.score(&x, &y);

    // Train linear model
    let mut lr = crate::linear_model::LinearRegression::new();
    lr.fit(&x, &y).expect("fit should succeed");
    let lr_r2 = lr.score(&x, &y);

    // Both should achieve high R² on linear data
    assert!(tree_r2 > 0.9, "Tree R² {tree_r2} too low on linear data");
    assert!(lr_r2 > 0.99, "Linear regression R² {lr_r2} too low");
}

// ===================================================================
// Random Forest Regression Tests
// ===================================================================

#[test]
fn test_random_forest_regressor_creation() {
    let rf = RandomForestRegressor::new(10);
    assert_eq!(rf.n_estimators, 10);
    assert!(rf.trees.is_empty());
    assert!(rf.max_depth.is_none());
}

#[test]
fn test_random_forest_regressor_with_max_depth() {
    let rf = RandomForestRegressor::new(5).with_max_depth(3);
    assert_eq!(rf.max_depth, Some(3));
}

#[test]
fn test_random_forest_regressor_fit_simple_linear() {
    // Simple linear data: y = 2x + 1
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);

    let mut rf = RandomForestRegressor::new(10).with_max_depth(5);
    rf.fit(&x, &y).expect("fit should succeed");

    // Should have trained 10 trees
    assert_eq!(rf.trees.len(), 10);

    // Should make reasonable predictions
    let _predictions = rf.predict(&x);
    let r2 = rf.score(&x, &y);
    assert!(r2 > 0.8, "R² should be high on training data: {r2}");
}

#[test]
fn test_random_forest_regressor_predict_nonlinear() {
    // Non-linear data: y = x²
    let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

    // Train RF (with fixed random state for reproducibility)
    let mut rf = RandomForestRegressor::new(20)
        .with_max_depth(4)
        .with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let predictions = rf.predict(&x);
    assert_eq!(predictions.len(), 8);

    // Check predictions are reasonable (allow some error due to averaging and small dataset)
    let pred_slice = predictions.as_slice();
    let y_slice = y.as_slice();
    for i in 0..8 {
        let error = (pred_slice[i] - y_slice[i]).abs();
        assert!(
            error <= 12.0,
            "Prediction {} too far from true value {}: error {}",
            pred_slice[i],
            y_slice[i],
            error
        );
    }
}

#[test]
fn test_random_forest_regressor_score() {
    let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

    let mut rf = RandomForestRegressor::new(15).with_max_depth(3);
    rf.fit(&x, &y).expect("fit should succeed");

    let r2 = rf.score(&x, &y);
    // R² should be positive and high for this simple linear pattern
    assert!(r2 > 0.7, "R² score {r2} should be high");
    assert!(r2 <= 1.0, "R² score {r2} should be <= 1.0");
}

#[test]
fn test_random_forest_regressor_n_estimators_effect() {
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]);

    // Few trees
    let mut rf_few = RandomForestRegressor::new(5).with_max_depth(4);
    rf_few.fit(&x, &y).expect("fit should succeed");
    let r2_few = rf_few.score(&x, &y);

    // Many trees
    let mut rf_many = RandomForestRegressor::new(30).with_max_depth(4);
    rf_many.fit(&x, &y).expect("fit should succeed");
    let r2_many = rf_many.score(&x, &y);

    // More trees should generally give same or better performance
    // (at least not significantly worse)
    assert!(
        r2_many >= r2_few - 0.1,
        "More trees should not hurt performance"
    );
}

#[test]
fn test_random_forest_regressor_vs_single_tree() {
    // Random forest should generalize better than single tree
    let x = Matrix::from_vec(
        15,
        1,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[
        2.1, 4.2, 8.9, 16.1, 24.8, 36.2, 49.1, 63.8, 81.2, 100.1, 120.9, 144.2, 169.1, 195.8, 225.0,
    ]);

    // Single tree with high depth (prone to overfitting)
    let mut single_tree = DecisionTreeRegressor::new().with_max_depth(10);
    single_tree.fit(&x, &y).expect("fit should succeed");
    let single_r2 = single_tree.score(&x, &y);

    // Random forest with moderate depth
    let mut rf = RandomForestRegressor::new(20).with_max_depth(6);
    rf.fit(&x, &y).expect("fit should succeed");
    let rf_r2 = rf.score(&x, &y);

    // Both should fit well, but RF typically more stable
    assert!(single_r2 > 0.8, "Single tree R²: {single_r2}");
    assert!(rf_r2 > 0.8, "Random forest R²: {rf_r2}");
}

#[test]
fn test_random_forest_regressor_multidimensional() {
    // 2D features: [x1, x2], y = x1 + 2*x2
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 1.0, 4.0, 2.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 6.0, 8.0]);

    let mut rf = RandomForestRegressor::new(15).with_max_depth(5);
    rf.fit(&x, &y).expect("fit should succeed");

    let predictions = rf.predict(&x);
    assert_eq!(predictions.len(), 8);

    let r2 = rf.score(&x, &y);
    assert!(r2 > 0.6, "R² on 2D data should be reasonable: {r2}");
}

#[test]
fn test_random_forest_regressor_constant_target() {
    // All samples have same target value
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0, 7.0]);

    let mut rf = RandomForestRegressor::new(10).with_max_depth(3);
    rf.fit(&x, &y).expect("fit should succeed");

    let predictions = rf.predict(&x);
    for &pred in predictions.as_slice() {
        assert!(
            (pred - 7.0).abs() < 1e-5,
            "Prediction {pred} should be ~7.0 for constant target"
        );
    }
}

#[test]
fn test_random_forest_regressor_single_sample() {
    let x =
        Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[5.0]);

    let mut rf = RandomForestRegressor::new(5).with_max_depth(2);
    rf.fit(&x, &y).expect("fit should succeed");

    let predictions = rf.predict(&x);
    assert_eq!(predictions.len(), 1);
    assert!(
        (predictions.as_slice()[0] - 5.0).abs() < 1e-5,
        "Single sample prediction should be exact"
    );
}

#[test]
fn test_random_forest_regressor_random_state() {
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

    // Train two forests with same random state
    let mut rf1 = RandomForestRegressor::new(10)
        .with_max_depth(4)
        .with_random_state(42);
    rf1.fit(&x, &y).expect("fit should succeed");
    let pred1 = rf1.predict(&x);

    let mut rf2 = RandomForestRegressor::new(10)
        .with_max_depth(4)
        .with_random_state(42);
    rf2.fit(&x, &y).expect("fit should succeed");
    let pred2 = rf2.predict(&x);

    // Predictions should be identical
    for (p1, p2) in pred1.as_slice().iter().zip(pred2.as_slice().iter()) {
        assert!(
            (p1 - p2).abs() < 1e-10,
            "Predictions with same random_state should match: {p1} vs {p2}"
        );
    }
}

#[test]
fn test_random_forest_regressor_validation_errors() {
    // Mismatched dimensions
    let x = Matrix::from_vec(
        5,
        2,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]); // Wrong size

    let mut rf = RandomForestRegressor::new(5);
    let result = rf.fit(&x, &y);
    assert!(result.is_err(), "Should error on mismatched dimensions");

    // Zero samples
    let x_empty = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed in tests");
    let y_empty = Vector::from_slice(&[]);
    let mut rf_empty = RandomForestRegressor::new(5);
    let result_empty = rf_empty.fit(&x_empty, &y_empty);
    assert!(result_empty.is_err(), "Should error on zero samples");
}

#[test]
#[should_panic(expected = "Cannot predict with an unfitted Random Forest")]
fn test_random_forest_regressor_predict_before_fit() {
    let rf = RandomForestRegressor::new(5);
    let x =
        Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Matrix creation should succeed in tests");
    let _ = rf.predict(&x); // Should panic
}

#[test]
fn test_random_forest_regressor_default() {
    let rf1 = RandomForestRegressor::new(10);
    let rf2 = RandomForestRegressor::default();

    assert_eq!(rf1.n_estimators, rf2.n_estimators);
    assert_eq!(rf1.max_depth, rf2.max_depth);
}

#[test]
fn test_random_forest_regressor_comparison_with_linear_regression() {
    // On non-linear data, RF should significantly outperform linear regression
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]); // y = x²

    // Train RF (with fixed random state for reproducibility)
    let mut rf = RandomForestRegressor::new(30)
        .with_max_depth(5)
        .with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");
    let rf_r2 = rf.score(&x, &y);

    // Train linear regression
    let mut lr = crate::linear_model::LinearRegression::new();
    lr.fit(&x, &y).expect("fit should succeed");
    let lr_r2 = lr.score(&x, &y);

    // RF should handle non-linearity better
    assert!(
        rf_r2 > 0.9,
        "Random forest R² {rf_r2} should be high on quadratic data"
    );
    assert!(
        lr_r2 < 0.98,
        "Linear regression R² {lr_r2} should be lower on non-linear data"
    );
    assert!(
        rf_r2 > lr_r2,
        "RF R² {rf_r2} should exceed linear R² {lr_r2} on non-linear data"
    );
}

#[test]
fn test_random_forest_regressor_max_depth_effect() {
    let x = Matrix::from_vec(
        12,
        1,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[
        1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0,
    ]);

    // Shallow trees (with fixed random state for reproducibility)
    let mut rf_shallow = RandomForestRegressor::new(15)
        .with_max_depth(2)
        .with_random_state(42);
    rf_shallow.fit(&x, &y).expect("fit should succeed");
    let r2_shallow = rf_shallow.score(&x, &y);

    // Deep trees (with fixed random state for reproducibility)
    let mut rf_deep = RandomForestRegressor::new(15)
        .with_max_depth(8)
        .with_random_state(42);
    rf_deep.fit(&x, &y).expect("fit should succeed");
    let r2_deep = rf_deep.score(&x, &y);

    // Deeper trees should capture at least as much complexity as shallow trees
    // Note: On small datasets, the difference may be minimal due to variance
    assert!(
        r2_deep >= r2_shallow - 0.01,
        "Deeper trees R² {r2_deep} should be at least as good as shallow trees R² {r2_shallow}"
    );
}
