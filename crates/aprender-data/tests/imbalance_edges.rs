//! Public-API edge cases for class-imbalance helpers (#5578).
//! Contract: FALSIFY-BIN-ALIMENTAR-007.

use std::sync::Arc;

use alimentar::{
    resample, sqrt_inverse_weights, ArrowDataset, Dataset, ImbalanceDetector, ResampleStrategy,
};
use arrow::{
    array::{ArrayRef, Float64Array, Int64Array, StringArray},
    datatypes::{Field, Schema},
    record_batch::RecordBatch,
};

fn one_col(array: ArrayRef) -> ArrowDataset {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "label",
        array.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![array]).expect("batch");
    ArrowDataset::from_batch(batch).expect("dataset")
}

#[test]
fn weights_degenerate_inputs_are_empty() {
    assert!(sqrt_inverse_weights(&[]).is_empty());
    assert!(sqrt_inverse_weights(&[0, 0, 0]).is_empty());
}

#[test]
fn weights_balanced_is_uniform() {
    let w = sqrt_inverse_weights(&[10, 10, 10, 10]);
    assert_eq!(w.len(), 4);
    for x in &w {
        assert!((x - 1.0).abs() < 1e-6, "balanced weight {x} != 1.0");
    }
}

#[test]
fn weights_zero_count_class_gets_zero() {
    let w = sqrt_inverse_weights(&[4, 0, 4]);
    assert!(w[1].abs() < f32::EPSILON, "empty class weight {}", w[1]);
    assert!((w[0] - 1.5).abs() < 1e-6 && (w[2] - 1.5).abs() < 1e-6);
}

#[test]
fn weights_exact_sqrt_ratio() {
    let w = sqrt_inverse_weights(&[1, 4]);
    assert!((w[0] / w[1] - 2.0).abs() < 1e-5);
    assert!((w.iter().sum::<f32>() - 2.0).abs() < 1e-5);
}

#[test]
fn detector_rejects_unsupported_column_type() {
    let ds = one_col(Arc::new(Float64Array::from(vec![1.0, 2.0])));
    let err = ImbalanceDetector::new("label").analyze(&ds).unwrap_err();
    assert!(err.to_string().contains("Unsupported"), "got: {err}");
}

#[test]
fn detector_all_null_labels_is_error() {
    let ds = one_col(Arc::new(StringArray::from(vec![None::<&str>, None])));
    let err = ImbalanceDetector::new("label").analyze(&ds).unwrap_err();
    assert!(err.to_string().contains("No valid values"), "got: {err}");
}

#[test]
fn detector_counts_int64_labels_and_skips_nulls() {
    let ds = one_col(Arc::new(Int64Array::from(vec![
        Some(1),
        Some(1),
        Some(1),
        Some(2),
        None,
    ])));
    let report = ImbalanceDetector::new("label").analyze(&ds).unwrap();
    assert_eq!(report.distribution.total, 4);
    assert_eq!(report.distribution.num_classes, 2);
}

#[test]
fn resample_missing_column_is_error() {
    let ds = one_col(Arc::new(StringArray::from(vec!["a", "b"])));
    let err = resample(&ds, "nope", ResampleStrategy::Oversample, 1).unwrap_err();
    assert!(err.to_string().contains("nope"), "got: {err}");
}

#[test]
fn resample_rejects_float_labels() {
    let ds = one_col(Arc::new(Float64Array::from(vec![1.0, 2.0])));
    assert!(resample(&ds, "label", ResampleStrategy::Oversample, 1).is_err());
}

#[test]
fn resample_skips_null_labels_when_balancing() {
    let ds = one_col(Arc::new(StringArray::from(vec![
        Some("a"),
        None,
        Some("a"),
        Some("b"),
    ])));
    let out = resample(&ds, "label", ResampleStrategy::Oversample, 3).unwrap();
    assert_eq!(out.len(), 4); // 2 classes x max count 2; the null row is dropped
}

#[test]
fn resample_oversample_balances_int64_exactly() {
    let ds = one_col(Arc::new(Int64Array::from(vec![0, 0, 0, 0, 0, 1])));
    let out = resample(&ds, "label", ResampleStrategy::Oversample, 7).unwrap();
    assert_eq!(out.len(), 10);
    let report = ImbalanceDetector::new("label").analyze(&out).unwrap();
    assert_eq!(report.distribution.num_classes, 2);
    assert!((report.metrics.imbalance_ratio - 1.0).abs() < 1e-9);
}

#[test]
fn resample_undersample_shrinks_to_minority() {
    let ds = one_col(Arc::new(Int64Array::from(vec![0, 0, 0, 0, 1, 1])));
    let out = resample(&ds, "label", ResampleStrategy::Undersample, 5).unwrap();
    assert_eq!(out.len(), 4);
}
