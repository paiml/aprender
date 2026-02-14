use super::*;

#[test]
fn test_qa_checklist_default() {
    let checklist = QaChecklist::default();
    assert_eq!(QaChecklist::max_score(), 100);
    assert!(checklist.model_path.as_os_str().is_empty());
    assert!(checklist.test_data.is_none());
    assert_eq!(checklist.max_turns, 5);
}

#[test]
fn test_qa_checklist_builder() {
    let checklist = QaChecklist::new(PathBuf::from("model.apr"))
        .with_test_data(PathBuf::from("test.ald"))
        .with_protected_attrs(vec!["gender".to_string(), "age".to_string()])
        .with_latency_sla(Duration::from_millis(50))
        .with_memory_budget(256 * 1024 * 1024);

    assert_eq!(checklist.model_path, PathBuf::from("model.apr"));
    assert_eq!(checklist.test_data, Some(PathBuf::from("test.ald")));
    assert_eq!(checklist.protected_attrs.len(), 2);
    assert_eq!(checklist.latency_sla, Duration::from_millis(50));
    assert_eq!(checklist.memory_budget, 256 * 1024 * 1024);
}

#[test]
fn test_category_points_sum_to_100() {
    let points = QaChecklist::category_points();
    let total: u8 = points.values().sum();
    assert_eq!(total, 100, "Category points must sum to 100");
}

#[test]
fn test_qa_report_creation() {
    let mut report = QaReport::new("test-model".to_string());
    assert_eq!(report.model_id, "test-model");
    assert_eq!(report.total_score, 0);
    assert!(!report.passed);

    // Add a category score
    let mut score = CategoryScore::new(20);
    score.add_result(TestResult::pass("test1", Duration::from_millis(10)));
    score.add_result(TestResult::pass("test2", Duration::from_millis(10)));
    score.finalize();

    report.add_category(QaCategory::Robustness, score);
    assert!(report.total_score > 0);
}

#[test]
fn test_qa_report_blockers() {
    let mut report = QaReport::new("test-model".to_string());

    // Add passing scores
    let mut score = CategoryScore::new(100);
    for i in 0..10 {
        score.add_result(TestResult::pass(
            format!("test{i}"),
            Duration::from_millis(10),
        ));
    }
    score.finalize();
    report.add_category(QaCategory::Robustness, score);

    assert!(report.passed);

    // Add a blocker
    report.add_blocker(QaIssue::blocker(
        QaCategory::Fairness,
        "Disparate impact ratio < 0.8",
        "Retrain with balanced dataset",
    ));

    assert!(!report.passed, "Report should fail with blockers");
}

#[test]
fn test_category_score_pass_rate() {
    let mut score = CategoryScore::new(20);
    score.add_result(TestResult::pass("t1", Duration::ZERO));
    score.add_result(TestResult::pass("t2", Duration::ZERO));
    score.add_result(TestResult::fail("t3", "failed", Duration::ZERO));
    score.add_result(TestResult::fail("t4", "failed", Duration::ZERO));
    score.finalize();

    assert_eq!(score.pass_rate(), 50.0);
    assert_eq!(score.points_earned, 10); // 50% of 20
}

#[test]
fn test_all_categories() {
    let categories = QaCategory::all();
    assert_eq!(categories.len(), 8);

    // Verify all have names
    for cat in categories {
        assert!(!cat.name().is_empty());
    }
}

#[test]
fn test_severity_blocking() {
    assert!(Severity::Blocker.is_blocking());
    assert!(!Severity::Critical.is_blocking());
    assert!(!Severity::Warning.is_blocking());
    assert!(!Severity::Info.is_blocking());

    assert!(Severity::Blocker.requires_review());
    assert!(Severity::Critical.requires_review());
    assert!(!Severity::Warning.requires_review());
    assert!(!Severity::Info.requires_review());
}

#[test]
fn test_jidoka_stop() {
    let stop = JidokaStop::QualityGateFailed {
        score: 75,
        threshold: 80,
    };
    assert!(stop.requires_human_review());
    assert!(stop.description().contains("75"));
    assert!(stop.description().contains("80"));
}

#[test]
fn test_production_ready() {
    let mut report = QaReport::new("model".to_string());

    // Add full score
    let mut score = CategoryScore::new(100);
    for i in 0..10 {
        score.add_result(TestResult::pass(format!("t{i}"), Duration::ZERO));
    }
    score.finalize();
    report.add_category(QaCategory::Robustness, score);

    assert!(report.is_production_ready());
}

#[test]
fn test_qa_issue_creation() {
    let blocker = QaIssue::blocker(
        QaCategory::Privacy,
        "Membership inference AUC > 0.6",
        "Add differential privacy",
    );
    assert_eq!(blocker.severity, Severity::Blocker);

    let warning = QaIssue::warning(
        QaCategory::Latency,
        "P99 latency > SLA",
        "Optimize inference path",
    );
    assert_eq!(warning.severity, Severity::Warning);
}
