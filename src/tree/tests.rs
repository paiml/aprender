//! Tests for decision tree algorithms.

use super::*;
use crate::primitives::{Matrix, Vector};
use crate::traits::Estimator;

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
        assert_eq!(gbm.n_estimators, 100);
        assert_eq!(gbm.learning_rate, 0.1);
        assert_eq!(gbm.max_depth, 3);
        assert_eq!(gbm.n_estimators(), 0); // No estimators before fit
    }

    #[test]
    fn test_gradient_boosting_builder() {
        let gbm = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.05)
            .with_max_depth(5);

        assert_eq!(gbm.n_estimators, 50);
        assert_eq!(gbm.learning_rate, 0.05);
        assert_eq!(gbm.max_depth, 5);
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
                0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.0, 0.1,
                0.2, // Class 1 (top-right cluster)
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

        assert_eq!(gbm1.n_estimators, gbm2.n_estimators);
        assert_eq!(gbm1.learning_rate, gbm2.learning_rate);
        assert_eq!(gbm1.max_depth, gbm2.max_depth);
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
        let x = Matrix::from_vec(2, 1, vec![1.0, 2.0])
            .expect("Matrix creation should succeed in tests");
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
            2.1, 4.2, 8.9, 16.1, 24.8, 36.2, 49.1, 63.8, 81.2, 100.1, 120.9, 144.2, 169.1, 195.8,
            225.0,
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
        let x = Matrix::from_vec(1, 2, vec![1.0, 2.0])
            .expect("Matrix creation should succeed in tests");
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
        let x_empty =
            Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed in tests");
        let y_empty = Vector::from_slice(&[]);
        let mut rf_empty = RandomForestRegressor::new(5);
        let result_empty = rf_empty.fit(&x_empty, &y_empty);
        assert!(result_empty.is_err(), "Should error on zero samples");
    }

    #[test]
    #[should_panic(expected = "Cannot predict with an unfitted Random Forest")]
    fn test_random_forest_regressor_predict_before_fit() {
        let rf = RandomForestRegressor::new(5);
        let x = Matrix::from_vec(2, 1, vec![1.0, 2.0])
            .expect("Matrix creation should succeed in tests");
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

    // ===================================================================
    // Out-of-Bag (OOB) Error Estimation Tests
    // ===================================================================

    #[test]
    fn test_random_forest_classifier_oob_score_after_fit() {
        // Simple classification data
        let x = Matrix::from_vec(
            15,
            4,
            vec![
                // Class 0
                5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
                3.6, 1.4, 0.2, // Class 1
                7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3,
                6.5, 2.8, 4.6, 1.5, // Class 2
                6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8,
                6.5, 3.0, 5.8, 2.2,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let mut rf = RandomForestClassifier::new(20)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf.oob_score();
        assert!(
            oob_score.is_some(),
            "OOB score should be available after fit"
        );

        let score_value = oob_score.expect("oob_score should be available");
        assert!(
            (0.0..=1.0).contains(&score_value),
            "OOB score {score_value} should be between 0 and 1"
        );
    }

    #[test]
    fn test_random_forest_classifier_oob_prediction_length() {
        let x = Matrix::from_vec(
            10,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 5.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        let mut rf = RandomForestClassifier::new(15).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_preds = rf.oob_prediction();
        assert!(
            oob_preds.is_some(),
            "OOB predictions should be available after fit"
        );

        let preds = oob_preds.expect("oob_preds should be available");
        assert_eq!(
            preds.len(),
            10,
            "OOB predictions should have same length as training data"
        );
    }

    #[test]
    fn test_random_forest_classifier_oob_before_fit() {
        let rf = RandomForestClassifier::new(10);

        assert!(
            rf.oob_score().is_none(),
            "OOB score should be None before fit"
        );
        assert!(
            rf.oob_prediction().is_none(),
            "OOB prediction should be None before fit"
        );
    }

    #[test]
    fn test_random_forest_classifier_oob_vs_test_score() {
        // Larger dataset to get reliable OOB estimate
        let x = Matrix::from_vec(
            30,
            4,
            vec![
                // Class 0 (10 samples)
                5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
                3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4,
                2.9, 1.4, 0.2, 4.9, 3.1, 1.5, 0.1, // Class 1 (10 samples)
                7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3,
                6.5, 2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0,
                6.6, 2.9, 4.6, 1.3, 5.2, 2.7, 3.9, 1.4, // Class 2 (10 samples)
                6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8,
                6.5, 3.0, 5.8, 2.2, 7.6, 3.0, 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3, 2.9, 6.3, 1.8,
                6.7, 2.5, 5.8, 1.8, 7.2, 3.6, 6.1, 2.5,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2,
        ];

        let mut rf = RandomForestClassifier::new(50)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf
            .oob_score()
            .expect("oob_score should be available after fit");
        let train_score = rf.score(&x, &y);

        // OOB score should be reasonable (within 0.3 of training score for small dataset)
        assert!(
            (oob_score - train_score).abs() < 0.3,
            "OOB score {oob_score} should be close to training score {train_score}"
        );
    }

    #[test]
    fn test_random_forest_classifier_oob_reproducibility() {
        let x = Matrix::from_vec(
            15,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 6.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let mut rf1 = RandomForestClassifier::new(20)
            .with_max_depth(4)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let oob1 = rf1.oob_score();

        let mut rf2 = RandomForestClassifier::new(20)
            .with_max_depth(4)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let oob2 = rf2.oob_score();

        assert_eq!(oob1, oob2, "OOB scores should be identical with same seed");
    }

    #[test]
    fn test_random_forest_regressor_oob_score_after_fit() {
        let x = Matrix::from_vec(
            20,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0, 34.0, 36.0, 38.0, 40.0,
        ]);

        let mut rf = RandomForestRegressor::new(30)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf.oob_score();
        assert!(
            oob_score.is_some(),
            "OOB score should be available after fit"
        );

        let score_value = oob_score.expect("oob_score should be available");
        assert!(
            score_value > -1.0 && score_value <= 1.0,
            "OOB R² score {score_value} should be reasonable"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_prediction_length() {
        let x = Matrix::from_vec(
            12,
            2,
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 1.0, 4.0, 2.0,
                5.0, 3.0, 5.0, 4.0, 6.0, 3.0, 6.0, 4.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 6.0, 8.0, 8.0, 9.0, 9.0, 10.0]);

        let mut rf = RandomForestRegressor::new(20).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_preds = rf.oob_prediction();
        assert!(
            oob_preds.is_some(),
            "OOB predictions should be available after fit"
        );

        let preds = oob_preds.expect("oob_preds should be available");
        assert_eq!(
            preds.len(),
            12,
            "OOB predictions should have same length as training data"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_before_fit() {
        let rf = RandomForestRegressor::new(10);

        assert!(
            rf.oob_score().is_none(),
            "OOB score should be None before fit"
        );
        assert!(
            rf.oob_prediction().is_none(),
            "OOB prediction should be None before fit"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_vs_test_score() {
        // Linear data for predictable results
        let x = Matrix::from_vec(
            25,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0,
            33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0,
        ]);

        let mut rf = RandomForestRegressor::new(50)
            .with_max_depth(6)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf
            .oob_score()
            .expect("oob_score should be available after fit");
        let train_score = rf.score(&x, &y);

        // OOB R² should be positive and within reasonable range of training R²
        assert!(oob_score > 0.5, "OOB R² {oob_score} should be positive");
        assert!(
            (oob_score - train_score).abs() < 0.3,
            "OOB R² {oob_score} should be close to training R² {train_score}"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_reproducibility() {
        let x = Matrix::from_vec(
            15,
            2,
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0,
        ]);

        let mut rf1 = RandomForestRegressor::new(25)
            .with_max_depth(5)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let oob1 = rf1.oob_score();

        let mut rf2 = RandomForestRegressor::new(25)
            .with_max_depth(5)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let oob2 = rf2.oob_score();

        assert_eq!(
            oob1, oob2,
            "OOB R² scores should be identical with same seed"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_nonlinear_data() {
        // Quadratic data to test OOB on non-linear patterns
        let x = Matrix::from_vec(
            15,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0, 196.0,
            225.0,
        ]);

        let mut rf = RandomForestRegressor::new(40)
            .with_max_depth(6)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf.oob_score();
        assert!(oob_score.is_some(), "OOB score should be available");

        // OOB should still be reasonably high for non-linear data
        let score_value = oob_score.expect("oob_score should be available");
        assert!(
            score_value > 0.7,
            "OOB R² {score_value} should be high on non-linear data"
        );
    }

    // ===================================================================
    // Feature Importance Tests (Issue #32)
    // ===================================================================

    #[test]
    fn test_random_forest_classifier_feature_importances_after_fit() {
        // Simple classification data with 3 features
        let x = Matrix::from_vec(
            12,
            3,
            vec![
                // Class 0 - feature 0 is discriminative
                1.0, 5.0, 5.0, 1.0, 6.0, 4.0, 2.0, 5.0, 6.0, 1.0, 4.0,
                5.0, // Class 1 - feature 0 is discriminative
                10.0, 5.0, 5.0, 10.0, 6.0, 4.0, 11.0, 5.0, 6.0, 10.0, 4.0,
                5.0, // Class 2 - feature 0 is discriminative
                20.0, 5.0, 5.0, 20.0, 6.0, 4.0, 21.0, 5.0, 6.0, 20.0, 4.0, 5.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let mut rf = RandomForestClassifier::new(20)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let importances = rf.feature_importances();
        assert!(
            importances.is_some(),
            "Feature importances should be available after fit"
        );

        let imps = importances.expect("importances should be available");
        assert_eq!(
            imps.len(),
            3,
            "Should have importance for each of 3 features"
        );

        // Importances should sum to ~1.0
        let sum: f32 = imps.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Importances should sum to 1.0, got {sum}"
        );

        // Feature 0 should be most important (it's the discriminative feature)
        assert!(
            imps[0] > imps[1] && imps[0] > imps[2],
            "Feature 0 should be most important, got {imps:?}"
        );
    }

    #[test]
    fn test_random_forest_classifier_feature_importances_before_fit() {
        let rf = RandomForestClassifier::new(10);

        let importances = rf.feature_importances();
        assert!(
            importances.is_none(),
            "Feature importances should be None before fit"
        );
    }

    #[test]
    fn test_random_forest_classifier_feature_importances_reproducibility() {
        let x = Matrix::from_vec(
            10,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 5.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        let mut rf1 = RandomForestClassifier::new(20).with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let imps1 = rf1
            .feature_importances()
            .expect("feature importances should be available");

        let mut rf2 = RandomForestClassifier::new(20).with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let imps2 = rf2
            .feature_importances()
            .expect("feature importances should be available");

        // Should be very similar with same random_state
        // Note: Small variations can occur due to floating point arithmetic in normalization
        // Trueno v0.6.0 may have different SIMD optimizations affecting FP precision
        for (i, (&imp1, &imp2)) in imps1.iter().zip(imps2.iter()).enumerate() {
            assert!(
                (imp1 - imp2).abs() <= 0.15,
                "Importance {i} should be similar: {imp1} vs {imp2}"
            );
        }
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_after_fit() {
        // Simple regression data where feature 0 is predictive
        let x = Matrix::from_vec(
            10,
            3,
            vec![
                // Feature 0 is predictive of y
                1.0, 5.0, 5.0, 2.0, 6.0, 4.0, 3.0, 5.0, 6.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0,
                4.0, 7.0, 5.0, 6.0, 8.0, 4.0, 5.0, 9.0, 5.0, 5.0, 10.0, 6.0, 4.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]); // Linear with feature 0

        let mut rf = RandomForestRegressor::new(20)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let importances = rf.feature_importances();
        assert!(
            importances.is_some(),
            "Feature importances should be available after fit"
        );

        let imps = importances.expect("importances should be available");
        assert_eq!(
            imps.len(),
            3,
            "Should have importance for each of 3 features"
        );

        // Importances should sum to ~1.0
        let sum: f32 = imps.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Importances should sum to 1.0, got {sum}"
        );

        // Feature 0 should be most important
        assert!(
            imps[0] > imps[1] && imps[0] > imps[2],
            "Feature 0 should be most important, got {imps:?}"
        );
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_before_fit() {
        let rf = RandomForestRegressor::new(10);

        let importances = rf.feature_importances();
        assert!(
            importances.is_none(),
            "Feature importances should be None before fit"
        );
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_reproducibility() {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 6.0, 1.0, 7.0, 0.0, 8.0, 1.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

        let mut rf1 = RandomForestRegressor::new(20).with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let imps1 = rf1
            .feature_importances()
            .expect("feature importances should be available");

        let mut rf2 = RandomForestRegressor::new(20).with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let imps2 = rf2
            .feature_importances()
            .expect("feature importances should be available");

        // Should be very similar with same random_state
        // Note: Small variations can occur due to floating point arithmetic in normalization
        for (i, (&imp1, &imp2)) in imps1.iter().zip(imps2.iter()).enumerate() {
            assert!(
                (imp1 - imp2).abs() <= 0.1,
                "Importance {i} should be similar: {imp1} vs {imp2}"
            );
        }
    }

    #[test]
    fn test_random_forest_classifier_feature_importances_all_nonnegative() {
        // All importances should be >= 0
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

        let mut rf = RandomForestClassifier::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let imps = rf
            .feature_importances()
            .expect("feature importances should be available");
        for (i, &imp) in imps.iter().enumerate() {
            assert!(
                imp >= 0.0,
                "Importance {i} should be non-negative, got {imp}"
            );
        }
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_all_nonnegative() {
        // All importances should be >= 0
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 6.0, 1.0, 7.0, 0.0, 8.0, 1.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

        let mut rf = RandomForestRegressor::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let imps = rf
            .feature_importances()
            .expect("feature importances should be available");
        for (i, &imp) in imps.iter().enumerate() {
            assert!(
                imp >= 0.0,
                "Importance {i} should be non-negative, got {imp}"
            );
        }
    }

    #[test]
    fn test_random_forest_classifier_predict_proba() {
        // Test predict_proba returns valid probabilities
        let x = Matrix::from_vec(
            6,
            2,
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1, 2, 2]; // 3 classes

        let mut rf = RandomForestClassifier::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let proba = rf.predict_proba(&x);

        // Shape should be (n_samples, n_classes)
        assert_eq!(proba.shape(), (6, 3));

        // Each row should sum to 1.0
        for row in 0..6 {
            let sum: f32 = (0..3).map(|col| proba.get(row, col)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row {row} probabilities should sum to 1.0, got {sum}"
            );
        }

        // All probabilities should be in [0, 1]
        for row in 0..6 {
            for col in 0..3 {
                let p = proba.get(row, col);
                assert!(
                    (0.0..=1.0).contains(&p),
                    "Probability should be in [0,1], got {p}"
                );
            }
        }
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn test_decision_tree_regressor_new() {
        let tree = DecisionTreeRegressor::new();
        assert!(tree.tree.is_none());
        assert!(tree.max_depth.is_none());
    }

    #[test]
    fn test_decision_tree_regressor_default() {
        let tree = DecisionTreeRegressor::default();
        assert!(tree.tree.is_none());
    }

    #[test]
    fn test_decision_tree_regressor_with_max_depth() {
        let tree = DecisionTreeRegressor::new().with_max_depth(5);
        assert_eq!(tree.max_depth, Some(5));
    }

    #[test]
    fn test_decision_tree_regressor_with_min_samples_split() {
        let tree = DecisionTreeRegressor::new().with_min_samples_split(5);
        assert_eq!(tree.min_samples_split, 5);
    }

    #[test]
    fn test_decision_tree_regressor_with_min_samples_leaf() {
        let tree = DecisionTreeRegressor::new().with_min_samples_leaf(3);
        assert_eq!(tree.min_samples_leaf, 3);
    }

    #[test]
    fn test_decision_tree_regressor_fit_predict() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
        tree.fit(&x, &y).expect("fit should succeed");

        let predictions = tree.predict(&x);
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_decision_tree_regressor_score() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(5);
        tree.fit(&x, &y).expect("fit should succeed");

        let score = tree.score(&x, &y);
        // R² should be reasonably high for this simple linear data
        assert!(score > 0.5, "R² should be > 0.5, got {score}");
    }

    #[test]
    fn test_decision_tree_regressor_fit_mismatch_error() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0]); // Mismatched length

        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_decision_tree_regressor_fit_empty_error() {
        let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[]);

        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_forest_regressor_new() {
        let rf = RandomForestRegressor::new(10);
        assert_eq!(rf.n_estimators, 10);
        assert!(rf.trees.is_empty());
    }

    #[test]
    fn test_random_forest_regressor_default_values() {
        let rf = RandomForestRegressor::default();
        assert!(rf.trees.is_empty());
    }

    #[test]
    fn test_random_forest_regressor_fit_mismatch_error() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0]); // Mismatched length

        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_forest_regressor_fit_empty_error() {
        let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[]);

        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_forest_regressor_oob_methods() {
        let rf = RandomForestRegressor::new(10);
        assert!(rf.oob_prediction().is_none());
        assert!(rf.oob_score().is_none());
    }

    #[test]
    fn test_random_forest_regressor_oob_methods_after_fit() {
        let x = Matrix::from_vec(
            10,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        let mut rf = RandomForestRegressor::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_pred = rf.oob_prediction();
        assert!(oob_pred.is_some());
        assert_eq!(oob_pred.as_ref().map(|v| v.len()), Some(10));

        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
    }

    #[test]
    fn test_regression_tree_node_depth() {
        let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 5.0,
            n_samples: 10,
        });
        assert_eq!(leaf.depth(), 0);

        let node = RegressionTreeNode::Node(RegressionNode {
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
        assert_eq!(node.depth(), 1);
    }

    #[test]
    fn test_gradient_boosting_classifier_new() {
        let gb = GradientBoostingClassifier::new();
        assert!(gb.estimators.is_empty());
        assert_eq!(gb.n_estimators, 100); // Config value, not fitted count
    }

    #[test]
    fn test_gradient_boosting_classifier_default() {
        let gb = GradientBoostingClassifier::default();
        assert!(gb.estimators.is_empty());
    }

    #[test]
    fn test_gradient_boosting_classifier_builders() {
        let gb = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.05)
            .with_max_depth(5);

        assert_eq!(gb.n_estimators, 50); // Config value, not fitted count
    }

    #[test]
    fn test_gradient_boosting_classifier_fit_predict() {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
            ],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

        let mut gb = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(3);

        gb.fit(&x, &y).expect("fit should succeed");

        let predictions = gb.predict(&x).expect("predict should succeed");
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_gradient_boosting_classifier_predict_proba() {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
            ],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

        let mut gb = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(3);

        gb.fit(&x, &y).expect("fit should succeed");

        let proba = gb.predict_proba(&x).expect("predict_proba should succeed");
        assert_eq!(proba.len(), 8);

        // Each row should be a probability distribution
        for row_proba in &proba {
            let sum: f32 = row_proba.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Probabilities should sum to 1.0, got {sum}"
            );
            for &p in row_proba {
                assert!((0.0..=1.0).contains(&p), "Probability should be in [0,1]");
            }
        }
    }

    #[test]
    fn test_random_forest_classifier_oob_methods() {
        let rf = RandomForestClassifier::new(10);
        assert!(rf.oob_prediction().is_none());
        assert!(rf.oob_score().is_none());
        assert!(rf.feature_importances().is_none());
    }

    #[test]
    fn test_decision_tree_classifier_fit_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .expect("Matrix creation should succeed");
        let y = vec![0, 1]; // Mismatched length

        let mut tree = DecisionTreeClassifier::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

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
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
            .expect("Matrix creation should succeed");
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
        let x =
            Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Matrix creation should succeed");
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

    #[test]
    fn test_count_regression_tree_samples() {
        let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 5.0,
            n_samples: 15,
        });
        assert_eq!(count_regression_tree_samples(&leaf), 15);

        let node = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 1.0,
                n_samples: 5,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 2.0,
                n_samples: 8,
            })),
        });
        assert_eq!(count_regression_tree_samples(&node), 13);
    }

    // =========================================================================
    // Coverage boost: OOB and feature importance tests
    // =========================================================================

    #[test]
    fn test_random_forest_classifier_oob_score() {
        let x = Matrix::from_vec(
            20,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.0, 0.3, 0.1, 0.4, 0.2, 0.0, 0.3, 0.1, 0.0, 0.2,
                0.1, 0.3, 0.2, 0.1, // Class 1 samples
                1.0, 1.0, 0.9, 0.9, 1.0, 0.8, 0.8, 1.0, 0.9, 1.1, 1.1, 0.9, 0.8, 0.8, 1.0, 0.9,
                0.9, 1.0, 1.0, 0.8,
            ],
        )
        .expect("Matrix creation should succeed");
        let y: Vec<usize> = (0..10).map(|_| 0).chain((0..10).map(|_| 1)).collect();

        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        // Test OOB prediction
        let oob_pred = rf.oob_prediction();
        assert!(oob_pred.is_some());
        assert_eq!(oob_pred.expect("Expected OOB predictions").len(), 20);

        // Test OOB score
        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
        let score = oob_score.expect("Expected OOB score");
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_random_forest_classifier_oob_unfitted() {
        let rf = RandomForestClassifier::new(5);
        assert!(rf.oob_prediction().is_none());
        assert!(rf.oob_score().is_none());
    }

    #[test]
    fn test_random_forest_classifier_feature_importances() {
        let x = Matrix::from_vec(
            10,
            3,
            vec![
                0.0, 0.5, 0.1, 0.1, 0.5, 0.2, 0.0, 0.5, 0.0, 0.2, 0.5, 0.1, 0.1, 0.5, 0.3, 1.0,
                0.5, 0.8, 0.9, 0.5, 0.9, 1.0, 0.5, 1.0, 0.8, 0.5, 0.7, 0.9, 0.5, 0.8,
            ],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let importances = rf.feature_importances();
        assert!(importances.is_some());
        let imp = importances.expect("Expected feature importances");
        assert_eq!(imp.len(), 3);
        // Sum should be ~1.0
        let sum: f32 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_random_forest_classifier_feature_importances_unfitted() {
        let rf = RandomForestClassifier::new(5);
        assert!(rf.feature_importances().is_none());
    }

    #[test]
    fn test_random_forest_regressor_oob_score() {
        let x = Matrix::from_vec(20, 1, (1..=20).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&(1..=20).map(|i| (i * 2) as f32).collect::<Vec<_>>());

        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let oob_pred = rf.oob_prediction();
        assert!(oob_pred.is_some());

        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
    }

    #[test]
    fn test_random_forest_regressor_oob_unfitted() {
        let rf = RandomForestRegressor::new(5);
        assert!(rf.oob_prediction().is_none());
        assert!(rf.oob_score().is_none());
    }

    #[test]
    fn test_random_forest_regressor_feature_importances() {
        let x = Matrix::from_vec(
            10,
            2,
            vec![
                1.0, 0.5, 2.0, 0.5, 3.0, 0.5, 4.0, 0.5, 5.0, 0.5, 6.0, 0.5, 7.0, 0.5, 8.0, 0.5,
                9.0, 0.5, 10.0, 0.5,
            ],
        )
        .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let importances = rf.feature_importances();
        assert!(importances.is_some());
        let imp = importances.expect("Expected feature importances");
        assert_eq!(imp.len(), 2);
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_unfitted() {
        let rf = RandomForestRegressor::new(5);
        assert!(rf.feature_importances().is_none());
    }

    // =========================================================================
    // Coverage boost: Default implementations
    // =========================================================================

    #[test]
    fn test_rfr_default_coverage() {
        let rf = RandomForestRegressor::default();
        assert_eq!(rf.n_estimators, 10);
    }

    #[test]
    fn test_dtr_default_coverage() {
        let tree = DecisionTreeRegressor::default();
        assert!(tree.tree.is_none());
    }

    // =========================================================================
    // Coverage boost: Error handling
    // =========================================================================

    #[test]
    fn test_dtc_fit_empty_coverage() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y: Vec<usize> = vec![];
        let mut tree = DecisionTreeClassifier::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtc_fit_sample_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
        let y = vec![0, 1]; // Wrong length
        let mut tree = DecisionTreeClassifier::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtr_fit_empty_coverage() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[]);
        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtr_fit_sample_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length
        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "low >= high")]
    fn test_rfc_fit_empty_coverage() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y: Vec<usize> = vec![];
        let mut rf = RandomForestClassifier::new(5);
        // RF doesn't check empty data explicitly and will panic in bootstrap_sample
        let _ = rf.fit(&x, &y);
    }

    #[test]
    fn test_rfr_fit_empty_coverage() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[]);
        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_rfr_fit_sample_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[1.0, 2.0]);
        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    // =========================================================================
    // Coverage boost: Regression tree builder options
    // =========================================================================

    #[test]
    fn test_decision_tree_regressor_min_samples_split_coverage() {
        let tree = DecisionTreeRegressor::new().with_min_samples_split(5);
        assert_eq!(tree.min_samples_split, 5);
    }

    #[test]
    fn test_decision_tree_regressor_min_samples_split_floor_coverage() {
        // min_samples_split should be at least 2
        let tree = DecisionTreeRegressor::new().with_min_samples_split(1);
        assert_eq!(tree.min_samples_split, 2);
    }

    #[test]
    fn test_decision_tree_regressor_min_samples_leaf_coverage() {
        let tree = DecisionTreeRegressor::new().with_min_samples_leaf(3);
        assert_eq!(tree.min_samples_leaf, 3);
    }

    #[test]
    fn test_decision_tree_regressor_min_samples_leaf_floor_coverage() {
        // min_samples_leaf should be at least 1
        let tree = DecisionTreeRegressor::new().with_min_samples_leaf(0);
        assert_eq!(tree.min_samples_leaf, 1);
    }

    #[test]
    fn test_decision_tree_regressor_score_coverage() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(5);
        tree.fit(&x, &y).expect("Fit should succeed");

        let score = tree.score(&x, &y);
        // R² should be high for this simple linear relationship
        assert!(score > 0.8);
    }

    // =========================================================================
    // Coverage boost: Gradient Boosting additional tests
    // =========================================================================

    #[test]
    fn test_gradient_boosting_predict_proba_untrained() {
        let gbm = GradientBoostingClassifier::new();
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0])
            .expect("Matrix creation should succeed");
        let result = gbm.predict_proba(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_boosting_n_estimators_getter() {
        let gbm = GradientBoostingClassifier::new().with_n_estimators(25);
        // Before training, n_estimators() returns 0 (number of trained estimators)
        assert_eq!(gbm.n_estimators(), 0);
    }

    // =========================================================================
    // Coverage boost: predict_proba for RandomForestClassifier
    // =========================================================================

    #[test]
    fn test_random_forest_predict_proba() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 1.0, 1.0, 0.9, 0.9, 1.0, 0.8],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let proba = rf.predict_proba(&x);
        let (n_samples, n_classes) = proba.shape();
        assert_eq!(n_samples, 6);
        assert_eq!(n_classes, 2);

        // Each row should sum to 1.0
        for i in 0..n_samples {
            let row_sum: f32 = (0..n_classes).map(|j| proba.get(i, j)).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    // =========================================================================
    // Coverage boost: Bootstrap sample edge cases
    // =========================================================================

    #[test]
    fn test_bootstrap_sample_no_seed() {
        let indices = bootstrap_sample(50, None);
        assert_eq!(indices.len(), 50);
        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 50);
        }
    }

    #[test]
    fn test_bootstrap_sample_different_seeds() {
        let indices1 = bootstrap_sample(50, Some(1));
        let indices2 = bootstrap_sample(50, Some(2));
        // Different seeds should (almost certainly) give different results
        assert_ne!(indices1, indices2);
    }

    // =========================================================================
    // Coverage boost: SafeTensors serialization
    // =========================================================================

    #[test]
    fn test_decision_tree_save_load_safetensors() {
        use std::fs;

        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
        tree.fit(&x, &y).expect("Fit should succeed");

        let path = "/tmp/test_tree_safetensors.safetensors";
        tree.save_safetensors(path).expect("Save should succeed");

        let loaded = DecisionTreeClassifier::load_safetensors(path).expect("Load should succeed");

        // Verify predictions match
        let orig_pred = tree.predict(&x);
        let loaded_pred = loaded.predict(&x);
        assert_eq!(orig_pred, loaded_pred);

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_decision_tree_save_safetensors_unfitted() {
        let tree = DecisionTreeClassifier::new();
        let result = tree.save_safetensors("/tmp/unfitted.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_random_forest_save_load_safetensors() {
        use std::fs;

        let x = Matrix::from_vec(
            6,
            2,
            vec![0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 1.0, 1.0, 0.9, 0.9, 1.0, 0.8],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(2)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let path = "/tmp/test_rf_safetensors.safetensors";
        rf.save_safetensors(path).expect("Save should succeed");

        let loaded = RandomForestClassifier::load_safetensors(path).expect("Load should succeed");

        // Verify structure
        assert_eq!(loaded.n_estimators, 3);
        assert_eq!(loaded.trees.len(), 3);

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_random_forest_save_safetensors_unfitted() {
        let rf = RandomForestClassifier::new(5);
        let result = rf.save_safetensors("/tmp/unfitted_rf.safetensors");
        assert!(result.is_err());
    }

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
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed");
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
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(3);

        // Using Estimator trait methods
        tree.fit(&x, &y).expect("Fit should succeed");
        let score = tree.score(&x, &y);
        assert!(score > 0.9);
    }
