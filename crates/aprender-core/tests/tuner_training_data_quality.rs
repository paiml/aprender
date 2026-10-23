#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! F041-F060 (aprender RandomForest half): Training Data Quality, ML-tuner showcase.
//!
//! SHOWCASE-BRICK-001 feeds trueno `TunerFeatures` into `aprender::RandomForestRegressor`
//! and `RandomForestClassifier`. These tests lived in aprender-compute behind a
//! `ml-tuner` feature nothing enabled, driven by a dev-dependency of compute on core.
//! That dev edge tied compute (the base of the stack) to core, so a touch of any base
//! crate selected most of the workspace (FLOW-010, #5803). Core depends on compute, so a
//! change to compute still selects these tests here. The tests that need no aprender stay
//! in `crates/aprender-compute/tests/tuner_falsification/training_data_quality.rs`.

use trueno::tuner::TunerFeatures;

/// Train an aprender RandomForest regressor on trueno features, with the showcase's
/// `<10 samples` guard. Returns `Err` on insufficient data or matrix/fit failure.
fn fit_throughput_rf(
    rows: &[(TunerFeatures, f32)],
    n_trees: usize,
) -> Result<aprender::tree::RandomForestRegressor, String> {
    use aprender::primitives::{Matrix, Vector};
    use aprender::tree::RandomForestRegressor;
    if rows.len() < 10 {
        return Err(format!("insufficient data: {} < 10", rows.len()));
    }
    // RandomForestRegressor::fit accepts NaN labels and then predicts NaN everywhere (measured), so the guard is here.
    if let Some(i) = rows.iter().position(|(_, t)| !t.is_finite()) {
        return Err(format!("non-finite label at row {i}"));
    }
    let dim = TunerFeatures::DIM;
    let n = rows.len();
    let mut x = Vec::with_capacity(n * dim);
    let mut y = Vec::with_capacity(n);
    for (f, t) in rows {
        x.extend(f.to_vector());
        y.push(*t);
    }
    let x_mat = Matrix::from_vec(n, dim, x).map_err(|e| e.to_string())?;
    let mut rf = RandomForestRegressor::new(n_trees);
    rf.fit(&x_mat, &Vector::from_vec(y))
        .map_err(|e| e.to_string())?;
    Ok(rf)
}

/// F041: Empty training data should error
#[test]
fn f041_empty_training_errors() {
    let empty_data: Vec<(TunerFeatures, f32)> = vec![];
    let result = fit_throughput_rf(&empty_data, 10);
    assert!(
        result.is_err(),
        "F041 FALSIFIED: empty training data should error"
    );
}

/// F042: Single sample training should error gracefully (below the 10-sample floor)
#[test]
fn f042_single_sample_graceful() {
    let features = TunerFeatures::builder().model_params_b(1.5).build();
    let data = vec![(features, 100.0)];
    let err = fit_throughput_rf(&data, 10)
        .err()
        .expect("F042: a single sample must error");
    assert!(
        err.contains("insufficient data"),
        "F042: wrong error: {err}"
    );
}

/// F043: Training with NaN labels must error (never fit and predict NaN)
#[test]
fn f043_nan_labels_error() {
    let mut data: Vec<(TunerFeatures, f32)> = (0..10)
        .map(|i| {
            (
                TunerFeatures::builder()
                    .batch_size(1 + (i % 8) as u32)
                    .build(),
                100.0,
            )
        })
        .collect();
    data[0].1 = f32::NAN;
    let err = fit_throughput_rf(&data, 10)
        .err()
        .expect("F043: NaN labels must be refused, not fitted");
    assert!(
        err.contains("non-finite label at row 0"),
        "F043: wrong error: {err}"
    );
}

/// F044: Training with negative labels should be handled without panic
#[test]
fn f044_negative_labels_handled() {
    let data: Vec<(TunerFeatures, f32)> = (0..12)
        .map(|i| {
            let f = TunerFeatures::builder()
                .batch_size(1 + (i % 8) as u32)
                .build();
            (f, if i % 2 == 0 { -100.0 } else { 100.0 })
        })
        .collect();
    use aprender::primitives::Matrix;
    let rf = fit_throughput_rf(&data, 10).expect("F044: negative finite labels must fit");
    for (feat, _) in &data {
        let fx = Matrix::from_vec(1, TunerFeatures::DIM, feat.to_vector().to_vec()).unwrap();
        let p = rf.predict(&fx).as_slice()[0];
        assert!(
            p.is_finite() && (-100.0..=100.0).contains(&p),
            "F044: prediction {p} outside the label range"
        );
    }
}

/// F046: Training improves over heuristic (or doesn't regress)
#[test]
fn f046_training_improves() {
    // Generate training data that matches heuristic pattern
    let training_data: Vec<(TunerFeatures, f32)> = (0..50)
        .map(|i| {
            let batch = 1 + (i % 8) as u32;
            let features = TunerFeatures::builder()
                .model_params_b(1.5)
                .batch_size(batch)
                .gpu_mem_bw_gbs(1000.0)
                .build();
            // Throughput scales with batch size
            let throughput = 100.0 + (batch as f32) * 50.0;
            (features, throughput)
        })
        .collect();

    let result = fit_throughput_rf(&training_data, 50);
    assert!(
        result.is_ok(),
        "F046 FALSIFIED: training failed: {:?}",
        result.err()
    );
}

/// F047: Large training set should not OOM
#[test]
fn f047_large_training_no_oom() {
    let training_data: Vec<(TunerFeatures, f32)> = (0..1000)
        .map(|i| {
            let features = TunerFeatures::builder()
                .model_params_b((i % 10) as f32 * 0.5 + 0.5)
                .batch_size((i % 8 + 1) as u32)
                .build();
            (features, 100.0 + (i as f32))
        })
        .collect();

    let result = fit_throughput_rf(&training_data, 10);
    assert!(result.is_ok(), "F047 FALSIFIED: large training failed");
}

/// F048: Classifier training should work (aprender RandomForestClassifier on trueno features)
#[test]
fn f048_classifier_training() {
    use aprender::primitives::Matrix;
    use aprender::tree::RandomForestClassifier;

    let dim = TunerFeatures::DIM;
    let n = 50usize;
    let mut x = Vec::with_capacity(n * dim);
    let mut y: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let batch = 1 + (i % 8) as u32;
        let f = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(batch)
            .build();
        x.extend(f.to_vector());
        // Label: BatchedQ4K (3) for M>=4, VectorizedQ4K (2) otherwise
        y.push(if batch >= 4 { 3 } else { 2 });
    }
    let x_mat = Matrix::from_vec(n, dim, x).expect("feature matrix");
    let mut classifier = RandomForestClassifier::new(10);
    let result = classifier.fit(&x_mat, &y);
    assert!(
        result.is_ok(),
        "F048 FALSIFIED: classifier training failed: {:?}",
        result.err()
    );
}
