//! Public-API edge cases for the data quality checker (coverage ticket, Rule 7).
//! Contract: FALSIFY-BIN-ALIMENTAR-008.

use std::sync::Arc;

use alimentar::quality::{
    ColumnQuality, NumericStats, QualityChecker, QualityIssue, QualityReport,
};
use alimentar::ArrowDataset;
use arrow::{
    array::{ArrayRef, BooleanArray, Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn ds(cols: Vec<(&str, ArrayRef)>) -> ArrowDataset {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        cols.into_iter().map(|(_, a)| a).collect(),
    )
    .expect("batch");
    ArrowDataset::from_batch(batch).expect("dataset")
}

fn issue_cols(r: &QualityReport, pick: fn(&QualityIssue) -> bool) -> Vec<&str> {
    r.issues
        .iter()
        .filter(|i| pick(i))
        .filter_map(|i| i.column())
        .collect()
}

#[test]
fn severity_table_is_pinned() {
    let c = || "c".to_string();
    assert_eq!(QualityIssue::EmptySchema.severity(), 5);
    assert_eq!(QualityIssue::EmptyDataset.severity(), 5);
    assert_eq!(
        QualityIssue::ConstantColumn {
            column: c(),
            value: "x".into()
        }
        .severity(),
        4
    );
    let nulls = |r| QualityIssue::HighNullRatio {
        column: c(),
        null_ratio: r,
        threshold: 0.1,
    };
    assert_eq!(nulls(0.9).severity(), 4);
    assert_eq!(nulls(0.5).severity(), 3);
    let outl = |r| QualityIssue::OutliersDetected {
        column: c(),
        outlier_count: 1,
        outlier_ratio: r,
    };
    assert_eq!(outl(0.2).severity(), 3);
    assert_eq!(outl(0.1).severity(), 2);
    assert_eq!(
        QualityIssue::DuplicateRows {
            duplicate_count: 1,
            duplicate_ratio: 0.5
        }
        .severity(),
        2
    );
    assert_eq!(
        QualityIssue::HighDuplicateRatio {
            column: c(),
            duplicate_ratio: 0.9,
            threshold: 0.5
        }
        .severity(),
        2
    );
    assert_eq!(
        QualityIssue::LowCardinality {
            column: c(),
            unique_count: 1,
            total_count: 9
        }
        .severity(),
        1
    );
}

#[test]
fn column_accessor_only_for_column_issues() {
    assert_eq!(QualityIssue::EmptySchema.column(), None);
    assert_eq!(QualityIssue::EmptyDataset.column(), None);
    assert_eq!(
        QualityIssue::DuplicateRows {
            duplicate_count: 1,
            duplicate_ratio: 0.1
        }
        .column(),
        None
    );
    let i = QualityIssue::LowCardinality {
        column: "k".into(),
        unique_count: 1,
        total_count: 2,
    };
    assert_eq!(i.column(), Some("k"));
}

#[test]
fn numeric_stats_iqr_and_fences() {
    let s = NumericStats {
        min: 0.0,
        max: 100.0,
        mean: 5.0,
        std_dev: 1.0,
        q1: 2.0,
        median: 4.0,
        q3: 6.0,
    };
    assert!((s.iqr() - 4.0).abs() < 1e-12);
    assert!((s.outlier_lower_bound() - (-4.0)).abs() < 1e-12);
    assert!((s.outlier_upper_bound() - 12.0).abs() < 1e-12);
}

#[test]
fn column_quality_constant_and_mostly_null_boundaries() {
    let mk = |unique, total, null_ratio| ColumnQuality {
        name: "c".into(),
        total_count: total,
        null_count: 0,
        null_ratio,
        unique_count: unique,
        unique_ratio: 0.0,
        duplicate_count: 0,
        duplicate_ratio: 0.0,
        outlier_count: None,
        numeric_stats: None,
    };
    assert!(mk(1, 5, 0.0).is_constant());
    assert!(mk(0, 5, 0.0).is_constant());
    assert!(!mk(2, 5, 0.0).is_constant());
    assert!(!mk(0, 0, 0.0).is_constant());
    assert!(mk(2, 5, 0.3).is_mostly_null(0.3)); // >= is inclusive
    assert!(!mk(2, 5, 0.29).is_mostly_null(0.3));
}

#[test]
fn empty_schema_and_empty_dataset_reports() {
    let empty_schema = ArrowDataset::from_batch(RecordBatch::new_empty(Arc::new(Schema::empty())))
        .expect("dataset");
    let r = QualityChecker::new().check(&empty_schema).expect("check");
    assert_eq!(r.issues, vec![QualityIssue::EmptySchema]);
    assert_eq!(r.max_severity(), 5);
    assert!(r.has_issues());

    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let no_rows = ArrowDataset::from_batch(RecordBatch::new_empty(schema)).expect("dataset");
    let r = QualityChecker::new().check(&no_rows).expect("check");
    assert_eq!(r.issues, vec![QualityIssue::EmptyDataset]);
    assert_eq!(r.column_count, 1);
    assert_eq!(r.row_count, 0);
}

#[test]
fn clean_dataset_has_no_issues_and_zero_severity() {
    let d = ds(vec![(
        "id",
        Arc::new(Int32Array::from((0..20).collect::<Vec<_>>())),
    )]);
    let r = QualityChecker::new()
        .with_outlier_check(false)
        .check(&d)
        .expect("check");
    assert!(!r.has_issues(), "{:?}", r.issues);
    assert_eq!(r.max_severity(), 0);
    assert!(r.problematic_columns().is_empty());
    assert_eq!(r.columns["id"].unique_count, 20);
}

#[test]
fn null_heavy_constant_and_low_cardinality_columns_are_flagged() {
    let d = ds(vec![
        (
            "nulls",
            Arc::new(StringArray::from(vec![
                Some("a"),
                None,
                None,
                None,
                Some("b"),
            ])) as ArrayRef,
        ),
        ("flag", Arc::new(BooleanArray::from(vec![true; 5]))),
        ("id", Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))),
    ]);
    let r = QualityChecker::new().check(&d).expect("check");
    assert_eq!(
        issue_cols(&r, |i| matches!(i, QualityIssue::HighNullRatio { .. })),
        vec!["nulls"]
    );
    assert_eq!(
        issue_cols(&r, |i| matches!(i, QualityIssue::ConstantColumn { .. })),
        vec!["flag"]
    );
    assert_eq!(
        issue_cols(&r, |i| matches!(i, QualityIssue::LowCardinality { .. })),
        vec!["flag"]
    );
    let mut cols = r.problematic_columns();
    cols.sort_unstable();
    assert_eq!(cols, vec!["flag", "nulls"]);
    assert_eq!(r.column_issues("id").len(), 0);
    assert!(r.column_issues("flag").len() >= 2);
    assert_eq!(r.max_severity(), 4);
    let c = r.issues.iter().find_map(|i| match i {
        QualityIssue::ConstantColumn { value, .. } => Some(value.clone()),
        _ => None,
    });
    assert_eq!(c.as_deref(), Some("true"));
}

#[test]
fn threshold_builders_change_what_is_flagged() {
    let d = ds(vec![(
        "v",
        Arc::new(StringArray::from(vec![
            Some("a"),
            None,
            Some("b"),
            Some("c"),
        ])) as ArrayRef,
    )]);
    // 25% null: flagged at the 0.1 default, tolerated at 0.5.
    let strict = QualityChecker::new().check(&d).expect("check");
    assert!(strict
        .issues
        .iter()
        .any(|i| matches!(i, QualityIssue::HighNullRatio { .. })));
    let lax = QualityChecker::new()
        .max_null_ratio(0.5)
        .check(&d)
        .expect("check");
    assert!(!lax
        .issues
        .iter()
        .any(|i| matches!(i, QualityIssue::HighNullRatio { .. })));
    // min_cardinality above the distinct count flags LowCardinality.
    let card = QualityChecker::new()
        .max_null_ratio(0.5)
        .min_cardinality(10)
        .check(&d)
        .expect("check");
    assert!(card
        .issues
        .iter()
        .any(|i| matches!(i, QualityIssue::LowCardinality { .. })));
}

#[test]
fn duplicate_rows_flagged_and_switchable() {
    let d = ds(vec![
        (
            "a",
            Arc::new(Int32Array::from(vec![1, 1, 1, 2])) as ArrayRef,
        ),
        ("b", Arc::new(StringArray::from(vec!["x", "x", "x", "y"]))),
    ]);
    let on = QualityChecker::new().check(&d).expect("check");
    assert_eq!(on.duplicate_row_count, 2);
    assert!(on.issues.iter().any(|i| matches!(
        i,
        QualityIssue::DuplicateRows {
            duplicate_count: 2,
            ..
        }
    )));
    let off = QualityChecker::new()
        .with_duplicate_check(false)
        .check(&d)
        .expect("check");
    assert_eq!(off.duplicate_row_count, 0);
    assert!(!off
        .issues
        .iter()
        .any(|i| matches!(i, QualityIssue::DuplicateRows { .. })));
}

#[test]
fn outlier_detected_and_disabled() {
    let mut v: Vec<f64> = (0..30).map(f64::from).collect();
    v.extend([10_000.0, 20_000.0, -9_000.0]);
    let d = ds(vec![("x", Arc::new(Float64Array::from(v)) as ArrayRef)]);
    let on = QualityChecker::new()
        .max_outlier_ratio(0.01)
        .check(&d)
        .expect("check");
    let stats = on.columns["x"]
        .numeric_stats
        .as_ref()
        .expect("numeric stats");
    assert!(stats.max >= 20_000.0 && stats.min <= -9_000.0);
    assert!(on.columns["x"].outlier_count.unwrap_or(0) >= 3);
    assert!(on
        .issues
        .iter()
        .any(|i| matches!(i, QualityIssue::OutliersDetected { .. })));
    let off = QualityChecker::new()
        .with_outlier_check(false)
        .check(&d)
        .expect("check");
    assert!(off.columns["x"].outlier_count.is_none());
    assert!(!off
        .issues
        .iter()
        .any(|i| matches!(i, QualityIssue::OutliersDetected { .. })));
}

#[test]
fn score_drops_with_issues() {
    let good = ds(vec![(
        "id",
        Arc::new(Int32Array::from((0..20).collect::<Vec<_>>())) as ArrayRef,
    )]);
    let bad = ds(vec![(
        "flag",
        Arc::new(BooleanArray::from(vec![true; 20])) as ArrayRef,
    )]);
    let g = QualityChecker::new()
        .with_outlier_check(false)
        .check(&good)
        .expect("check");
    let b = QualityChecker::new()
        .with_outlier_check(false)
        .check(&bad)
        .expect("check");
    assert!(g.score > b.score, "good {} <= bad {}", g.score, b.score);
}
