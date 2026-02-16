pub(crate) use super::*;

#[test]
fn test_commit_features_default() {
    let f = CommitFeatures::default();
    assert_eq!(f.defect_category, 0);
    assert!((f.files_changed - 0.0).abs() < f32::EPSILON);
    assert_eq!(f.hour_of_day, 12);
}

#[test]
fn test_commit_features_to_vec() {
    let f = CommitFeatures {
        defect_category: 1,
        files_changed: 5.0,
        lines_added: 100.0,
        lines_deleted: 20.0,
        complexity_delta: 8.0,
        timestamp: 1700000000.0,
        hour_of_day: 14,
        day_of_week: 2,
    };

    let v = f.to_vec();
    assert_eq!(v.len(), 8);
    assert!((v[0] - 1.0).abs() < f32::EPSILON);
    assert!((v[1] - 5.0).abs() < f32::EPSILON);
    assert!((v[2] - 100.0).abs() < f32::EPSILON);
}

#[test]
fn test_commit_features_from_vec() {
    let v = vec![2.0, 3.0, 50.0, 10.0, 4.0, 1700000000.0, 10.0, 5.0];
    let f = CommitFeatures::from_vec(&v);

    assert_eq!(f.defect_category, 2);
    assert!((f.files_changed - 3.0).abs() < f32::EPSILON);
    assert!((f.lines_added - 50.0).abs() < f32::EPSILON);
    assert_eq!(f.hour_of_day, 10);
    assert_eq!(f.day_of_week, 5);
}

#[test]
fn test_commit_features_churn() {
    let f = CommitFeatures {
        lines_added: 100.0,
        lines_deleted: 50.0,
        ..Default::default()
    };
    assert!((f.churn() - 150.0).abs() < f32::EPSILON);
    assert!((f.net_change() - 50.0).abs() < f32::EPSILON);
}

#[test]
fn test_commit_features_is_fix() {
    let fix = CommitFeatures {
        defect_category: 1,
        ..Default::default()
    };
    let clean = CommitFeatures::default();

    assert!(fix.is_fix());
    assert!(!clean.is_fix());
}

#[test]
fn test_commit_diff_builder() {
    let diff = CommitDiff::new()
        .with_files_changed(3)
        .with_lines_added(100)
        .with_lines_deleted(50)
        .with_timestamp(1700000000)
        .with_message("fix: resolve bug");

    assert_eq!(diff.files_changed, 3);
    assert_eq!(diff.lines_added, 100);
    assert_eq!(diff.lines_deleted, 50);
    assert_eq!(diff.timestamp, 1700000000);
    assert_eq!(diff.message, "fix: resolve bug");
}

#[test]
fn test_extractor_new() {
    let extractor = CodeFeatureExtractor::new();
    assert!(extractor.bug_keywords.contains("fix"));
    assert!(extractor.security_keywords.contains("vulnerability"));
    assert!(extractor.perf_keywords.contains("optimize"));
}

#[test]
fn test_extractor_extract_basic() {
    let extractor = CodeFeatureExtractor::new();
    let diff = CommitDiff {
        files_changed: 3,
        lines_added: 150,
        lines_deleted: 50,
        timestamp: 1700000000,
        message: "Add new feature".to_string(),
    };

    let features = extractor.extract(&diff);

    assert!((features.files_changed - 3.0).abs() < f32::EPSILON);
    assert!((features.lines_added - 150.0).abs() < f32::EPSILON);
    assert!((features.lines_deleted - 50.0).abs() < f32::EPSILON);
    assert!((features.timestamp - 1700000000.0).abs() < f64::EPSILON);
}

#[test]
fn test_extractor_classify_bug() {
    let extractor = CodeFeatureExtractor::new();
    let diff = CommitDiff {
        message: "fix: resolve memory leak bug".to_string(),
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    assert_eq!(features.defect_category, 1); // Bug
}

#[test]
fn test_extractor_classify_security() {
    let extractor = CodeFeatureExtractor::new();
    let diff = CommitDiff {
        message: "security: patch CVE-2024-1234 vulnerability".to_string(),
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    assert_eq!(features.defect_category, 2); // Security
}

#[test]
fn test_extractor_classify_performance() {
    let extractor = CodeFeatureExtractor::new();
    let diff = CommitDiff {
        message: "perf: optimize database queries".to_string(),
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    assert_eq!(features.defect_category, 3); // Performance
}

#[test]
fn test_extractor_classify_refactor() {
    let extractor = CodeFeatureExtractor::new();
    let diff = CommitDiff {
        message: "refactor: clean up legacy code".to_string(),
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    assert_eq!(features.defect_category, 4); // Refactor
}

#[test]
fn test_extractor_classify_clean() {
    let extractor = CodeFeatureExtractor::new();
    let diff = CommitDiff {
        message: "Add new dashboard component".to_string(),
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    assert_eq!(features.defect_category, 0); // Clean
}

#[test]
fn test_extractor_complexity_delta() {
    let extractor = CodeFeatureExtractor::new().with_complexity_factor(10.0);
    let diff = CommitDiff {
        lines_added: 100,
        lines_deleted: 20,
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    // (100 - 20) / 10 = 8.0
    assert!((features.complexity_delta - 8.0).abs() < f32::EPSILON);
}

#[test]
fn test_extractor_time_features() {
    let extractor = CodeFeatureExtractor::new();
    // 1700000000 = Tuesday, November 14, 2023 22:13:20 UTC
    let diff = CommitDiff {
        timestamp: 1700000000,
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    assert_eq!(features.hour_of_day, 22);
    assert_eq!(features.day_of_week, 2); // Tuesday
}

#[test]
fn test_extractor_batch() {
    let extractor = CodeFeatureExtractor::new();
    let diffs = vec![
        CommitDiff {
            files_changed: 1,
            ..Default::default()
        },
        CommitDiff {
            files_changed: 2,
            ..Default::default()
        },
        CommitDiff {
            files_changed: 3,
            ..Default::default()
        },
    ];

    let features = extractor.extract_batch(&diffs);
    assert_eq!(features.len(), 3);
    assert!((features[0].files_changed - 1.0).abs() < f32::EPSILON);
    assert!((features[1].files_changed - 2.0).abs() < f32::EPSILON);
    assert!((features[2].files_changed - 3.0).abs() < f32::EPSILON);
}

#[test]
fn test_feature_stats_from_features() {
    let features = vec![
        CommitFeatures {
            files_changed: 5.0,
            lines_added: 100.0,
            lines_deleted: 20.0,
            complexity_delta: 8.0,
            ..Default::default()
        },
        CommitFeatures {
            files_changed: 10.0,
            lines_added: 200.0,
            lines_deleted: 50.0,
            complexity_delta: -5.0,
            ..Default::default()
        },
    ];

    let stats = FeatureStats::from_features(&features);
    assert!((stats.files_changed_max - 10.0).abs() < f32::EPSILON);
    assert!((stats.lines_added_max - 200.0).abs() < f32::EPSILON);
    assert!((stats.lines_deleted_max - 50.0).abs() < f32::EPSILON);
    assert!((stats.complexity_max - 8.0).abs() < f32::EPSILON);
}

#[test]
fn test_extractor_normalize() {
    let extractor = CodeFeatureExtractor::new();
    let features = CommitFeatures {
        files_changed: 5.0,
        lines_added: 100.0,
        lines_deleted: 25.0,
        complexity_delta: 4.0,
        ..Default::default()
    };
    let stats = FeatureStats {
        files_changed_max: 10.0,
        lines_added_max: 200.0,
        lines_deleted_max: 50.0,
        complexity_max: 8.0,
    };

    let normalized = extractor.normalize(&features, &stats);
    assert!((normalized.files_changed - 0.5).abs() < f32::EPSILON);
    assert!((normalized.lines_added - 0.5).abs() < f32::EPSILON);
    assert!((normalized.lines_deleted - 0.5).abs() < f32::EPSILON);
    assert!((normalized.complexity_delta - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_extractor_normalize_zero_max() {
    let extractor = CodeFeatureExtractor::new();
    let features = CommitFeatures {
        files_changed: 5.0,
        ..Default::default()
    };
    let stats = FeatureStats::default(); // All zeros

    let normalized = extractor.normalize(&features, &stats);
    assert!((normalized.files_changed - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_extractor_add_keywords() {
    let mut extractor = CodeFeatureExtractor::new();
    extractor.add_bug_keywords(&["glitch", "oops"]);
    extractor.add_security_keywords(&["hack"]);

    assert!(extractor.bug_keywords.contains("glitch"));
    assert!(extractor.bug_keywords.contains("oops"));
    assert!(extractor.security_keywords.contains("hack"));
}

#[test]
fn test_security_priority_over_bug() {
    let extractor = CodeFeatureExtractor::new();
    // Message contains both security and bug keywords
    let diff = CommitDiff {
        message: "fix security vulnerability bug".to_string(),
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    // Security should take priority
    assert_eq!(features.defect_category, 2);
}

#[test]
fn test_epoch_thursday() {
    let extractor = CodeFeatureExtractor::new();
    // Unix epoch (0) was Thursday, January 1, 1970
    let diff = CommitDiff {
        timestamp: 0,
        ..Default::default()
    };

    let features = extractor.extract(&diff);
    assert_eq!(features.day_of_week, 4); // Thursday
    assert_eq!(features.hour_of_day, 0);
}
