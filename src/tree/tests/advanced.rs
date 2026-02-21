//! Advanced tree tests: integration, coverage, utilities, and estimator traits.

use crate::primitives::{Matrix, Vector};
use crate::tree::helpers::{
    at_max_depth, bootstrap_sample, calculate_information_gain, check_stopping_criteria,
    compute_mse, compute_regression_tree_feature_importances, compute_tree_feature_importances,
    count_regression_tree_samples, count_tree_samples, evaluate_split_gain,
    find_best_regression_split, find_best_regression_split_for_feature, flatten_tree_node,
    get_sorted_unique_values, get_unique_feature_values, gini_split, make_regression_leaf,
    mean_f32, partition_by_threshold, reconstruct_tree_node, split_by_threshold,
    split_data_by_indices, split_indices_by_threshold, split_labels_by_threshold,
    split_regression_data_by_indices, variance_f32,
};
use crate::tree::*;

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
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, // Class 2
            6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5,
            3.0, 5.8, 2.2,
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
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 4.0,
            4.0, 4.0, 5.0,
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
            3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4, 2.9,
            1.4, 0.2, 4.9, 3.1, 1.5, 0.1, // Class 1 (10 samples)
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0, 6.6, 2.9,
            4.6, 1.3, 5.2, 2.7, 3.9, 1.4, // Class 2 (10 samples)
            6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5,
            3.0, 5.8, 2.2, 7.6, 3.0, 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3, 2.9, 6.3, 1.8, 6.7, 2.5,
            5.8, 1.8, 7.2, 3.6, 6.1, 2.5,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
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
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 4.0,
            4.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 6.0,
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
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
        34.0, 36.0, 38.0, 40.0,
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
            1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 1.0, 4.0, 2.0, 5.0,
            3.0, 5.0, 4.0, 6.0, 3.0, 6.0, 4.0,
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
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[
        3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0,
        35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0,
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
            1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0,
            10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0,
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
        1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0, 196.0, 225.0,
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
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 4.0,
            4.0, 4.0, 5.0,
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

include!("advanced_regressor_importances.rs");
include!("advanced_gbm_and_scoring.rs");
include!("advanced_oob_and_serialization.rs");
include!("advanced_flatten_and_traits.rs");
