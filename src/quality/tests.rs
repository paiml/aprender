//! Tests for the quality module.

use std::sync::Arc;

use arrow::{
    array::{Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

use super::*;
use crate::dataset::ArrowDataset;

// ========== QualityIssue tests ==========

#[test]
fn test_issue_severity() {
    assert_eq!(QualityIssue::EmptySchema.severity(), 5);
    assert_eq!(QualityIssue::EmptyDataset.severity(), 5);

    let constant = QualityIssue::ConstantColumn {
        column: "x".to_string(),
        value: "1".to_string(),
    };
    assert_eq!(constant.severity(), 4);

    let high_null = QualityIssue::HighNullRatio {
        column: "x".to_string(),
        null_ratio: 0.6,
        threshold: 0.1,
    };
    assert_eq!(high_null.severity(), 4);

    let low_null = QualityIssue::HighNullRatio {
        column: "x".to_string(),
        null_ratio: 0.3,
        threshold: 0.1,
    };
    assert_eq!(low_null.severity(), 3);
}

#[test]
fn test_issue_column() {
    let issue = QualityIssue::HighNullRatio {
        column: "test".to_string(),
        null_ratio: 0.5,
        threshold: 0.1,
    };
    assert_eq!(issue.column(), Some("test"));

    assert_eq!(QualityIssue::EmptySchema.column(), None);
}

// ========== ColumnQuality tests ==========

#[test]
fn test_column_quality_is_constant() {
    let mut quality = ColumnQuality {
        name: "test".to_string(),
        total_count: 100,
        null_count: 0,
        null_ratio: 0.0,
        unique_count: 1,
        unique_ratio: 0.01,
        duplicate_count: 99,
        duplicate_ratio: 0.99,
        outlier_count: None,
        numeric_stats: None,
    };

    assert!(quality.is_constant());

    quality.unique_count = 5;
    assert!(!quality.is_constant());
}

#[test]
fn test_column_quality_mostly_null() {
    let quality = ColumnQuality {
        name: "test".to_string(),
        total_count: 100,
        null_count: 80,
        null_ratio: 0.8,
        unique_count: 5,
        unique_ratio: 0.25,
        duplicate_count: 15,
        duplicate_ratio: 0.75,
        outlier_count: None,
        numeric_stats: None,
    };

    assert!(quality.is_mostly_null(0.5));
    assert!(!quality.is_mostly_null(0.9));
}

// ========== NumericStats tests ==========

#[test]
fn test_numeric_stats_iqr() {
    let stats = NumericStats {
        min: 0.0,
        max: 100.0,
        mean: 50.0,
        std_dev: 25.0,
        q1: 25.0,
        median: 50.0,
        q3: 75.0,
    };

    assert!((stats.iqr() - 50.0).abs() < 0.01);
    assert!((stats.outlier_lower_bound() - (-50.0)).abs() < 0.01);
    assert!((stats.outlier_upper_bound() - 150.0).abs() < 0.01);
}

// ========== QualityReport tests ==========

#[test]
fn test_report_has_issues() {
    let report = QualityReport {
        row_count: 100,
        column_count: 2,
        columns: std::collections::HashMap::new(),
        issues: vec![],
        score: 100.0,
        duplicate_row_count: 0,
    };
    assert!(!report.has_issues());

    let report_with_issues = QualityReport {
        row_count: 100,
        column_count: 2,
        columns: std::collections::HashMap::new(),
        issues: vec![QualityIssue::EmptySchema],
        score: 50.0,
        duplicate_row_count: 0,
    };
    assert!(report_with_issues.has_issues());
}

#[test]
fn test_report_max_severity() {
    let report = QualityReport {
        row_count: 100,
        column_count: 2,
        columns: std::collections::HashMap::new(),
        issues: vec![
            QualityIssue::LowCardinality {
                column: "x".to_string(),
                unique_count: 1,
                total_count: 100,
            },
            QualityIssue::ConstantColumn {
                column: "y".to_string(),
                value: "1".to_string(),
            },
        ],
        score: 80.0,
        duplicate_row_count: 0,
    };

    assert_eq!(report.max_severity(), 4);
}

// ========== QualityChecker tests ==========

fn make_dataset(col1: Vec<Option<&str>>, col2: Vec<Option<i32>>) -> ArrowDataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("value", DataType::Int32, true),
    ]));

    let names: Vec<Option<&str>> = col1;
    let values: Vec<Option<i32>> = col2;

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(Int32Array::from(values)),
        ],
    )
    .expect("batch");

    ArrowDataset::from_batch(batch).expect("dataset")
}

fn make_float_dataset(values: Vec<Option<f64>>) -> ArrowDataset {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        true,
    )]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Float64Array::from(values))],
    )
    .expect("batch");

    ArrowDataset::from_batch(batch).expect("dataset")
}

#[test]
fn test_checker_new() {
    let checker = QualityChecker::new();
    assert!((checker.thresholds.max_null_ratio - 0.1).abs() < 0.01);
}

#[test]
fn test_checker_builder() {
    let checker = QualityChecker::new()
        .max_null_ratio(0.2)
        .max_duplicate_ratio(0.3)
        .min_cardinality(5);

    assert!((checker.thresholds.max_null_ratio - 0.2).abs() < 0.01);
    assert!((checker.thresholds.max_duplicate_ratio - 0.3).abs() < 0.01);
    assert_eq!(checker.thresholds.min_cardinality, 5);
}

#[test]
fn test_checker_clean_data() {
    let dataset = make_dataset(
        vec![Some("a"), Some("b"), Some("c"), Some("d")],
        vec![Some(1), Some(2), Some(3), Some(4)],
    );

    let checker = QualityChecker::new();
    let report = checker.check(&dataset).expect("check");

    assert_eq!(report.row_count, 4);
    assert_eq!(report.column_count, 2);
    assert!(report.score > 80.0);
}

#[test]
fn test_checker_detects_nulls() {
    let dataset = make_dataset(
        vec![Some("a"), None, None, None, None],
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
    );

    let checker = QualityChecker::new().max_null_ratio(0.5);
    let report = checker.check(&dataset).expect("check");

    let null_issues: Vec<_> = report
        .issues
        .iter()
        .filter(|i| matches!(i, QualityIssue::HighNullRatio { .. }))
        .collect();

    assert_eq!(null_issues.len(), 1);
}

#[test]
fn test_checker_detects_constant() {
    let dataset = make_dataset(
        vec![Some("same"), Some("same"), Some("same"), Some("same")],
        vec![Some(1), Some(2), Some(3), Some(4)],
    );

    let checker = QualityChecker::new();
    let report = checker.check(&dataset).expect("check");

    let constant_issues: Vec<_> = report
        .issues
        .iter()
        .filter(|i| matches!(i, QualityIssue::ConstantColumn { .. }))
        .collect();

    assert_eq!(constant_issues.len(), 1);
}

#[test]
fn test_checker_detects_duplicates() {
    let dataset = make_dataset(
        vec![Some("a"), Some("a"), Some("a"), Some("b")],
        vec![Some(1), Some(1), Some(1), Some(2)],
    );

    let checker = QualityChecker::new().max_duplicate_ratio(0.01);
    let report = checker.check(&dataset).expect("check");

    // Should detect duplicate rows
    assert!(report.duplicate_row_count > 0);
}

#[test]
fn test_checker_detects_outliers() {
    // Create dataset with clear outliers
    let mut values: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64)).collect();
    values.push(Some(10000.0)); // outlier
    values.push(Some(-10000.0)); // outlier

    let dataset = make_float_dataset(values);

    let checker = QualityChecker::new().max_outlier_ratio(0.01);
    let report = checker.check(&dataset).expect("check");

    let outlier_issues: Vec<_> = report
        .issues
        .iter()
        .filter(|i| matches!(i, QualityIssue::OutliersDetected { .. }))
        .collect();

    assert!(!outlier_issues.is_empty());
}

#[test]
fn test_checker_empty_dataset() {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(Vec::<i32>::new()))],
    )
    .expect("batch");
    let dataset = ArrowDataset::from_batch(batch).expect("dataset");

    let checker = QualityChecker::new();
    let report = checker.check(&dataset).expect("check");

    assert!(report.issues.contains(&QualityIssue::EmptyDataset));
    assert_eq!(report.score, 0.0);
}

#[test]
fn test_checker_score_decreases_with_issues() {
    let clean_dataset = make_dataset(
        vec![Some("a"), Some("b"), Some("c"), Some("d")],
        vec![Some(1), Some(2), Some(3), Some(4)],
    );

    let dirty_dataset = make_dataset(
        vec![Some("same"), Some("same"), None, None],
        vec![Some(1), Some(1), None, None],
    );

    let checker = QualityChecker::new();
    let clean_report = checker.check(&clean_dataset).expect("check");
    let dirty_report = checker.check(&dirty_dataset).expect("check");

    assert!(clean_report.score > dirty_report.score);
}

#[test]
fn test_checker_column_issues() {
    let dataset = make_dataset(
        vec![None, None, None, None],
        vec![Some(1), Some(2), Some(3), Some(4)],
    );

    let checker = QualityChecker::new();
    let report = checker.check(&dataset).expect("check");

    let name_issues = report.column_issues("name");
    assert!(!name_issues.is_empty());

    let value_issues = report.column_issues("value");
    // value column should have fewer issues
    assert!(value_issues.len() < name_issues.len());
}

#[test]
fn test_checker_problematic_columns() {
    let dataset = make_dataset(
        vec![None, None, None, None],
        vec![Some(1), Some(1), Some(1), Some(1)],
    );

    let checker = QualityChecker::new();
    let report = checker.check(&dataset).expect("check");

    let problematic = report.problematic_columns();
    assert!(problematic.contains(&"name"));
    assert!(problematic.contains(&"value"));
}

#[test]
fn test_checker_disable_outliers() {
    let mut values: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64)).collect();
    values.push(Some(10000.0));

    let dataset = make_float_dataset(values);

    let checker = QualityChecker::new()
        .with_outlier_check(false)
        .max_outlier_ratio(0.001);
    let report = checker.check(&dataset).expect("check");

    let outlier_issues: Vec<_> = report
        .issues
        .iter()
        .filter(|i| matches!(i, QualityIssue::OutliersDetected { .. }))
        .collect();

    assert!(outlier_issues.is_empty());
}

// ========== 100-Point Quality Scoring System Tests (GH-6) ==========

#[test]
fn test_severity_weights() {
    assert!((Severity::Critical.weight() - 2.0).abs() < 0.01);
    assert!((Severity::High.weight() - 1.5).abs() < 0.01);
    assert!((Severity::Medium.weight() - 1.0).abs() < 0.01);
    assert!((Severity::Low.weight() - 0.5).abs() < 0.01);
}

#[test]
fn test_severity_base_points() {
    assert!((Severity::Critical.base_points() - 2.0).abs() < 0.01);
    assert!((Severity::High.base_points() - 1.5).abs() < 0.01);
    assert!((Severity::Medium.base_points() - 1.0).abs() < 0.01);
    assert!((Severity::Low.base_points() - 0.5).abs() < 0.01);
}

#[test]
fn test_severity_display() {
    assert_eq!(format!("{}", Severity::Critical), "Critical");
    assert_eq!(format!("{}", Severity::High), "High");
    assert_eq!(format!("{}", Severity::Medium), "Medium");
    assert_eq!(format!("{}", Severity::Low), "Low");
}

#[test]
fn test_letter_grade_from_score() {
    assert_eq!(LetterGrade::from_score(100.0), LetterGrade::A);
    assert_eq!(LetterGrade::from_score(95.0), LetterGrade::A);
    assert_eq!(LetterGrade::from_score(94.9), LetterGrade::B);
    assert_eq!(LetterGrade::from_score(85.0), LetterGrade::B);
    assert_eq!(LetterGrade::from_score(84.9), LetterGrade::C);
    assert_eq!(LetterGrade::from_score(70.0), LetterGrade::C);
    assert_eq!(LetterGrade::from_score(69.9), LetterGrade::D);
    assert_eq!(LetterGrade::from_score(50.0), LetterGrade::D);
    assert_eq!(LetterGrade::from_score(49.9), LetterGrade::F);
    assert_eq!(LetterGrade::from_score(0.0), LetterGrade::F);
}

#[test]
fn test_letter_grade_publication_decision() {
    assert_eq!(LetterGrade::A.publication_decision(), "Publish immediately");
    assert_eq!(
        LetterGrade::B.publication_decision(),
        "Publish with documented caveats"
    );
    assert_eq!(
        LetterGrade::C.publication_decision(),
        "Remediation required before publication"
    );
    assert_eq!(LetterGrade::D.publication_decision(), "Major rework needed");
    assert_eq!(LetterGrade::F.publication_decision(), "Do not publish");
}

#[test]
fn test_letter_grade_is_publishable() {
    assert!(LetterGrade::A.is_publishable());
    assert!(LetterGrade::B.is_publishable());
    assert!(!LetterGrade::C.is_publishable());
    assert!(!LetterGrade::D.is_publishable());
    assert!(!LetterGrade::F.is_publishable());
}

#[test]
fn test_letter_grade_display() {
    assert_eq!(format!("{}", LetterGrade::A), "A");
    assert_eq!(format!("{}", LetterGrade::B), "B");
    assert_eq!(format!("{}", LetterGrade::C), "C");
    assert_eq!(format!("{}", LetterGrade::D), "D");
    assert_eq!(format!("{}", LetterGrade::F), "F");
}

#[test]
fn test_checklist_item_new() {
    let item = ChecklistItem::new(1, "Schema version documented", Severity::Critical, true);
    assert_eq!(item.id, 1);
    assert_eq!(item.description, "Schema version documented");
    assert!(item.passed);
    assert_eq!(item.severity, Severity::Critical);
    assert!(item.suggestion.is_none());
}

#[test]
fn test_checklist_item_with_suggestion() {
    let item = ChecklistItem::new(1, "Schema version documented", Severity::Critical, false)
        .with_suggestion("Add schema_version field to metadata");
    assert!(item.suggestion.is_some());
    assert_eq!(
        item.suggestion.unwrap(),
        "Add schema_version field to metadata"
    );
}

#[test]
fn test_checklist_item_points() {
    let passed_critical = ChecklistItem::new(1, "Test", Severity::Critical, true);
    assert!((passed_critical.points_earned() - 2.0).abs() < 0.01);
    assert!((passed_critical.max_points() - 2.0).abs() < 0.01);

    let failed_critical = ChecklistItem::new(2, "Test", Severity::Critical, false);
    assert!((failed_critical.points_earned() - 0.0).abs() < 0.01);
    assert!((failed_critical.max_points() - 2.0).abs() < 0.01);

    let passed_low = ChecklistItem::new(3, "Test", Severity::Low, true);
    assert!((passed_low.points_earned() - 0.5).abs() < 0.01);
}

#[test]
fn test_quality_score_perfect() {
    let checklist = vec![
        ChecklistItem::new(1, "Critical check", Severity::Critical, true),
        ChecklistItem::new(2, "High check", Severity::High, true),
        ChecklistItem::new(3, "Medium check", Severity::Medium, true),
        ChecklistItem::new(4, "Low check", Severity::Low, true),
    ];
    let score = QualityScore::from_checklist(checklist);

    // Total max points: 2.0 + 1.5 + 1.0 + 0.5 = 5.0
    // Total earned: 5.0
    // Score: 100%
    assert!((score.score - 100.0).abs() < 0.01);
    assert_eq!(score.grade, LetterGrade::A);
    assert!(score.grade.is_publishable());
    assert!(!score.has_critical_failures());
}

#[test]
fn test_quality_score_with_critical_failure() {
    let checklist = vec![
        ChecklistItem::new(1, "Critical check", Severity::Critical, false),
        ChecklistItem::new(2, "High check", Severity::High, true),
        ChecklistItem::new(3, "Medium check", Severity::Medium, true),
        ChecklistItem::new(4, "Low check", Severity::Low, true),
    ];
    let score = QualityScore::from_checklist(checklist);

    // Total max: 5.0, Earned: 3.0, Score: 60%
    assert!((score.score - 60.0).abs() < 0.01);
    assert_eq!(score.grade, LetterGrade::D);
    assert!(score.has_critical_failures());
    assert!(!score.grade.is_publishable());
}

#[test]
fn test_quality_score_failed_items() {
    let checklist = vec![
        ChecklistItem::new(1, "Critical check", Severity::Critical, false),
        ChecklistItem::new(2, "High check", Severity::High, true),
        ChecklistItem::new(3, "Medium check", Severity::Medium, false),
    ];
    let score = QualityScore::from_checklist(checklist);

    let failed = score.failed_items();
    assert_eq!(failed.len(), 2);
    assert_eq!(failed[0].id, 1);
    assert_eq!(failed[1].id, 3);

    let critical = score.critical_failures();
    assert_eq!(critical.len(), 1);
    assert_eq!(critical[0].id, 1);
}

#[test]
fn test_quality_score_severity_breakdown() {
    let checklist = vec![
        ChecklistItem::new(1, "C1", Severity::Critical, true),
        ChecklistItem::new(2, "C2", Severity::Critical, false),
        ChecklistItem::new(3, "H1", Severity::High, true),
    ];
    let score = QualityScore::from_checklist(checklist);

    let critical_stats = score.severity_breakdown.get(&Severity::Critical).unwrap();
    assert_eq!(critical_stats.total, 2);
    assert_eq!(critical_stats.passed, 1);
    assert_eq!(critical_stats.failed, 1);

    let high_stats = score.severity_breakdown.get(&Severity::High).unwrap();
    assert_eq!(high_stats.total, 1);
    assert_eq!(high_stats.passed, 1);
}

#[test]
fn test_quality_score_badge_url() {
    let checklist = vec![ChecklistItem::new(1, "Test", Severity::Critical, true)];
    let score = QualityScore::from_checklist(checklist);

    let badge = score.badge_url();
    assert!(badge.contains("shields.io"));
    assert!(badge.contains("data_quality"));
    assert!(badge.contains("brightgreen")); // Grade A
}

#[test]
fn test_quality_score_badge_colors() {
    // Test each grade gets correct color
    let grades_colors = vec![
        (100.0, "brightgreen"), // A
        (90.0, "green"),        // B
        (75.0, "yellow"),       // C
        (55.0, "orange"),       // D
        (30.0, "red"),          // F
    ];

    for (target_score, expected_color) in grades_colors {
        // Create checklist that produces approximately the target score
        let target: f64 = target_score;
        #[allow(clippy::cast_sign_loss)] // target is always positive (30.0-100.0)
        let passed = (target / 100.0 * 10.0).round() as usize;
        let failed = 10 - passed;
        let mut checklist: Vec<ChecklistItem> = (0..passed)
            .map(|i| ChecklistItem::new(i as u8, "Test", Severity::Medium, true))
            .collect();
        checklist.extend(
            (0..failed)
                .map(|i| ChecklistItem::new((passed + i) as u8, "Test", Severity::Medium, false)),
        );

        let score = QualityScore::from_checklist(checklist);
        let badge = score.badge_url();
        assert!(
            badge.contains(expected_color),
            "Score {:.0} should have color {} but badge was {}",
            score.score,
            expected_color,
            badge
        );
    }
}

#[test]
fn test_quality_score_json_output() {
    let checklist = vec![
        ChecklistItem::new(1, "Schema check", Severity::Critical, true),
        ChecklistItem::new(2, "Column check", Severity::High, false)
            .with_suggestion("Add missing columns"),
    ];
    let score = QualityScore::from_checklist(checklist);

    let json = score.to_json();
    assert!(json.contains("\"score\":"));
    assert!(json.contains("\"grade\":"));
    assert!(json.contains("\"is_publishable\":"));
    assert!(json.contains("\"failed_items\":"));
    assert!(json.contains("\"badge_url\":"));
    assert!(json.contains("Add missing columns"));
}

#[test]
fn test_quality_score_empty_checklist() {
    let checklist: Vec<ChecklistItem> = vec![];
    let score = QualityScore::from_checklist(checklist);

    // Empty checklist = 100% (nothing to fail)
    assert!((score.score - 100.0).abs() < 0.01);
    assert_eq!(score.grade, LetterGrade::A);
}

// ========== Quality Profile Tests (GH-10) ==========

#[test]
fn test_quality_profile_default() {
    let profile = QualityProfile::default();
    assert_eq!(profile.name, "default");
    assert!(profile.expected_constant_columns.is_empty());
    assert!(profile.nullable_columns.is_empty());
    assert!((profile.max_null_ratio - 0.1).abs() < 0.001);
}

#[test]
fn test_quality_profile_doctest_corpus() {
    let profile = QualityProfile::doctest_corpus();
    assert_eq!(profile.name, "doctest-corpus");
    assert!(profile.is_expected_constant("source"));
    assert!(profile.is_expected_constant("version"));
    assert!(!profile.is_expected_constant("function"));
    assert!(profile.is_nullable("signature"));
    assert!(!profile.is_nullable("input"));
}

#[test]
fn test_quality_profile_ml_training() {
    let profile = QualityProfile::ml_training();
    assert_eq!(profile.name, "ml-training");
    assert!((profile.max_null_ratio - 0.0).abs() < 0.001);
    assert!((profile.max_duplicate_ratio - 0.8).abs() < 0.001);
}

#[test]
fn test_quality_profile_time_series() {
    let profile = QualityProfile::time_series();
    assert_eq!(profile.name, "time-series");
    assert!((profile.max_duplicate_row_ratio - 0.0).abs() < 0.001);
}

#[test]
fn test_quality_profile_by_name() {
    assert!(QualityProfile::by_name("default").is_some());
    assert!(QualityProfile::by_name("doctest-corpus").is_some());
    assert!(QualityProfile::by_name("doctest").is_some());
    assert!(QualityProfile::by_name("ml-training").is_some());
    assert!(QualityProfile::by_name("ml").is_some());
    assert!(QualityProfile::by_name("time-series").is_some());
    assert!(QualityProfile::by_name("timeseries").is_some());
    assert!(QualityProfile::by_name("nonexistent").is_none());
}

#[test]
fn test_quality_profile_available_profiles() {
    let profiles = QualityProfile::available_profiles();
    assert!(profiles.contains(&"default"));
    assert!(profiles.contains(&"doctest-corpus"));
    assert!(profiles.contains(&"ml-training"));
    assert!(profiles.contains(&"time-series"));
}

#[test]
fn test_quality_profile_builders() {
    let profile = QualityProfile::new("custom")
        .with_description("Custom profile")
        .with_expected_constant("id")
        .with_nullable("optional_field")
        .with_max_null_ratio(0.2)
        .with_max_duplicate_ratio(0.6);

    assert_eq!(profile.name, "custom");
    assert_eq!(profile.description, "Custom profile");
    assert!(profile.is_expected_constant("id"));
    assert!(profile.is_nullable("optional_field"));
    assert!((profile.max_null_ratio - 0.2).abs() < 0.001);
    assert!((profile.max_duplicate_ratio - 0.6).abs() < 0.001);
}

#[test]
fn test_quality_profile_null_threshold_for() {
    let profile = QualityProfile::doctest_corpus();

    // Nullable columns get 100% threshold
    assert!((profile.null_threshold_for("signature") - 1.0).abs() < 0.001);

    // Non-nullable columns get profile threshold
    assert!((profile.null_threshold_for("input") - profile.max_null_ratio).abs() < 0.001);
}

#[test]
fn test_quality_profile_clone() {
    let profile = QualityProfile::doctest_corpus();
    let cloned = profile.clone();
    assert_eq!(profile.name, cloned.name);
    assert_eq!(
        profile.expected_constant_columns,
        cloned.expected_constant_columns
    );
}

#[test]
fn test_quality_profile_debug() {
    let profile = QualityProfile::default();
    let debug = format!("{:?}", profile);
    assert!(debug.contains("QualityProfile"));
    assert!(debug.contains("default"));
}

// ========== GH-013: Nested Arrow Types Support ==========

/// Helper to create a dataset with List<Struct> columns (like doctest corpus)
fn make_nested_dataset() -> ArrowDataset {
    use arrow::{
        array::{ArrayRef, ListArray, StructArray},
        buffer::OffsetBuffer,
    };

    // Schema with nested types
    let func_struct = DataType::Struct(
        vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("line", DataType::Int32, false),
        ]
        .into(),
    );
    let func_list = DataType::List(Arc::new(Field::new("element", func_struct, true)));

    let schema = Arc::new(Schema::new(vec![
        Field::new("module", DataType::Utf8, false),
        Field::new("functions", func_list, true),
        Field::new("coverage", DataType::Float32, false),
    ]));

    // Build the values array for the list (all function structs combined)
    let all_names = StringArray::from(vec!["foo", "bar", "baz", "qux", "quux"]);
    let all_lines = Int32Array::from(vec![10, 20, 30, 40, 50]);
    let all_structs = StructArray::try_new(
        vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("line", DataType::Int32, false),
        ]
        .into(),
        vec![
            Arc::new(all_names) as ArrayRef,
            Arc::new(all_lines) as ArrayRef,
        ],
        None,
    )
    .expect("all_structs");

    // Create list array with offsets [0, 2, 3, 5] for 3 rows
    let offsets = OffsetBuffer::new(vec![0i32, 2, 3, 5].into());
    let list_field = Arc::new(Field::new(
        "element",
        DataType::Struct(
            vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("line", DataType::Int32, false),
            ]
            .into(),
        ),
        true,
    ));
    let functions_array =
        ListArray::try_new(list_field, offsets, Arc::new(all_structs), None).expect("list");

    let modules = StringArray::from(vec!["mod_a.py", "mod_b.py", "mod_c.py"]);
    let coverages = arrow::array::Float32Array::from(vec![95.0f32, 98.5, 100.0]);

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(modules),
            Arc::new(functions_array),
            Arc::new(coverages),
        ],
    )
    .expect("batch");

    ArrowDataset::from_batch(batch).expect("dataset")
}

#[test]
fn test_gh013_nested_types_not_constant() {
    // GH-013: List<Struct> columns should NOT be flagged as constant
    // when they contain different values
    let dataset = make_nested_dataset();

    let checker = QualityChecker::new();
    let report = checker.check(&dataset).expect("check");

    // The functions column has 3 different list values, should NOT be constant
    let functions_col = report.columns.get("functions").expect("functions column");

    // Before fix: unique_count would be 1 (all "?")
    // After fix: unique_count should be 3 (different lists)
    assert!(
        functions_col.unique_count > 1,
        "List<Struct> column should have >1 unique values, got {}",
        functions_col.unique_count
    );
    assert!(
        !functions_col.is_constant(),
        "List<Struct> column should not be flagged as constant"
    );

    // Verify no ConstantColumn issue for functions
    let constant_issues: Vec<_> = report
        .issues
        .iter()
        .filter(
            |i| matches!(i, QualityIssue::ConstantColumn { column, .. } if column == "functions"),
        )
        .collect();
    assert!(
        constant_issues.is_empty(),
        "functions column should not have ConstantColumn issue"
    );
}

#[test]
fn test_gh013_float32_supported() {
    // Float32 columns should be properly analyzed
    let schema = Arc::new(Schema::new(vec![Field::new(
        "score",
        DataType::Float32,
        false,
    )]));

    let scores = arrow::array::Float32Array::from(vec![0.5f32, 0.75, 0.9, 0.85]);

    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(scores)]).expect("batch");

    let dataset = ArrowDataset::from_batch(batch).expect("dataset");
    let checker = QualityChecker::new();
    let report = checker.check(&dataset).expect("check");

    let score_col = report.columns.get("score").expect("score column");

    // Should have 4 unique values, not be constant
    assert_eq!(score_col.unique_count, 4);
    assert!(!score_col.is_constant());
}

#[test]
fn test_gh013_boolean_supported() {
    // Boolean columns should be properly analyzed
    let schema = Arc::new(Schema::new(vec![Field::new(
        "flag",
        DataType::Boolean,
        false,
    )]));

    let flags = arrow::array::BooleanArray::from(vec![true, false, true, false]);

    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(flags)]).expect("batch");

    let dataset = ArrowDataset::from_batch(batch).expect("dataset");
    let checker = QualityChecker::new();
    let report = checker.check(&dataset).expect("check");

    let flag_col = report.columns.get("flag").expect("flag column");

    // Should have 2 unique values (true, false)
    assert_eq!(flag_col.unique_count, 2);
    assert!(!flag_col.is_constant());
}
