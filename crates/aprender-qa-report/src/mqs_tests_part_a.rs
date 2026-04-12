/// Create a default test scenario for MQS test helpers
fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

/// Create a corroborated evidence entry for the given gate ID
fn test_evidence_passed(gate_id: &str) -> Evidence {
    Evidence::corroborated(gate_id, test_scenario(), "4", 100)
}

/// Create a falsified evidence entry for the given gate ID
fn test_evidence_failed(gate_id: &str) -> Evidence {
    Evidence::falsified(gate_id, test_scenario(), "Wrong answer", "5", 100)
}

/// Verify GatewayResult::passed creates a passing result with no failure reason
#[test]
fn test_gateway_result_passed() {
    let result = GatewayResult::passed("G1", "Model loads");
    assert!(result.passed);
    assert!(result.failure_reason.is_none());
}

/// Verify GatewayResult::failed creates a failing result with reason
#[test]
fn test_gateway_result_failed() {
    let result = GatewayResult::failed("G1", "Model loads", "OOM");
    assert!(!result.passed);
    assert_eq!(result.failure_reason, Some("OOM".to_string()));
}

/// Verify CategoryScores::total sums all six category scores
#[test]
fn test_category_scores_total() {
    let scores = CategoryScores {
        qual: 150,
        perf: 100,
        stab: 150,
        comp: 100,
        edge: 100,
        regr: 100,
    };
    assert_eq!(scores.total(), 700);
}

/// Verify CategoryScores::MAX_TOTAL equals 1000
#[test]
fn test_category_scores_max() {
    assert_eq!(CategoryScores::MAX_TOTAL, 1000);
}

/// Verify all-passing evidence yields grade A+ with score 1000
#[test]
fn test_mqs_calculator_all_pass() {
    let calculator = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add passing evidence for each category
    for i in 0..10 {
        collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-PERF-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-STAB-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-COMP-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-EDGE-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-REGR-{i:03}")));
    }

    let score = calculator
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    assert!(score.gateways_passed);
    assert_eq!(score.raw_score, 1000);
    assert!(score.normalized_score > 99.0);
    assert_eq!(score.grade, "A+");
}

/// Verify crash evidence triggers gateway failure with grade F
#[test]
fn test_mqs_calculator_gateway_failure() {
    let calculator = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add a crash (fails G3 gateway)
    collector.add(Evidence::crashed(
        "F-QUAL-001",
        test_scenario(),
        "SIGSEGV",
        -11,
        0,
    ));

    let score = calculator
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    assert!(!score.gateways_passed);
    assert_eq!(score.raw_score, 0);
    assert_eq!(score.normalized_score, 0.0);
    assert_eq!(score.grade, "F");
}

/// Verify timeout evidence applies TIMEOUT penalty to score
#[test]
fn test_mqs_calculator_with_penalties() {
    let calculator = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add mostly passing tests
    for i in 0..50 {
        collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
    }

    // Add some timeouts (but not crashes to keep gateways passing)
    for i in 0..5 {
        collector.add(Evidence::timeout(
            &format!("F-PERF-{i:03}"),
            test_scenario(),
            30000,
        ));
    }

    let score = calculator
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    // Should have timeout penalty
    assert!(score.total_penalty > 0);
    assert!(score.penalties.iter().any(|p| p.code == "TIMEOUT"));
}

/// Verify extract_category parses category from gate ID format
#[test]
fn test_extract_category() {
    assert_eq!(MqsCalculator::extract_category("F-QUAL-001"), "QUAL");
    assert_eq!(MqsCalculator::extract_category("F-PERF-042"), "PERF");
    assert_eq!(MqsCalculator::extract_category("UNKNOWN"), "QUAL");
}

/// Verify proportional_score computes correct ratios
#[test]
fn test_proportional_score() {
    assert_eq!(MqsCalculator::proportional_score(10, 10, 200), 200);
    assert_eq!(MqsCalculator::proportional_score(5, 10, 200), 100);
    assert_eq!(MqsCalculator::proportional_score(0, 10, 200), 0);
    // Untested categories score zero (Popper: untested ≠ qualified)
    assert_eq!(MqsCalculator::proportional_score(0, 0, 200), 0);
}

/// Verify calculate_grade maps scores to correct letter grades
#[test]
fn test_grade_calculation() {
    assert_eq!(MqsCalculator::calculate_grade(100.0), "A+");
    assert_eq!(MqsCalculator::calculate_grade(97.0), "A+");
    assert_eq!(MqsCalculator::calculate_grade(93.0), "A");
    assert_eq!(MqsCalculator::calculate_grade(90.0), "A-");
    assert_eq!(MqsCalculator::calculate_grade(83.0), "B");
    assert_eq!(MqsCalculator::calculate_grade(73.0), "C");
    assert_eq!(MqsCalculator::calculate_grade(50.0), "F");
}

/// Verify qualifies returns true for passing gateways with C grade
#[test]
fn test_mqs_score_qualifies() {
    let score = MqsScore {
        model_id: "test".to_string(),
        raw_score: 800,
        normalized_score: 75.0,
        grade: "C".to_string(),
        gateways: vec![],
        gateways_passed: true,
        categories: CategoryScores::default(),
        total_tests: 100,
        tests_passed: 80,
        tests_failed: 20,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };

    assert!(score.qualifies());
    assert!(!score.is_production_ready());
}

/// Verify normalization provides diminishing returns with perfect at 100
#[test]
fn test_normalize_score_scaling() {
    let calc = MqsCalculator::new();

    // Test that normalization provides diminishing returns
    let low = calc.normalize_score(200, 200);
    let mid = calc.normalize_score(500, 500);
    let high = calc.normalize_score(900, 900);
    let perfect = calc.normalize_score(1000, 1000);

    // Each increment should be harder
    assert!(low < mid);
    assert!(mid < high);
    assert!(high < perfect);

    // Perfect score should be 100
    assert!((perfect - 100.0).abs() < 0.01);
}

/// Verify calculate_grade covers all twelve grade levels from A+ to F
#[test]
fn test_grade_all_levels() {
    assert_eq!(MqsCalculator::calculate_grade(98.0), "A+");
    assert_eq!(MqsCalculator::calculate_grade(95.0), "A");
    assert_eq!(MqsCalculator::calculate_grade(91.0), "A-");
    assert_eq!(MqsCalculator::calculate_grade(88.0), "B+");
    assert_eq!(MqsCalculator::calculate_grade(85.0), "B");
    assert_eq!(MqsCalculator::calculate_grade(81.0), "B-");
    assert_eq!(MqsCalculator::calculate_grade(78.0), "C+");
    assert_eq!(MqsCalculator::calculate_grade(75.0), "C");
    assert_eq!(MqsCalculator::calculate_grade(71.0), "C-");
    assert_eq!(MqsCalculator::calculate_grade(68.0), "D+");
    assert_eq!(MqsCalculator::calculate_grade(65.0), "D");
    assert_eq!(MqsCalculator::calculate_grade(61.0), "D-");
    assert_eq!(MqsCalculator::calculate_grade(55.0), "F");
}

/// Verify is_production_ready returns true for A grade with passing gateways
#[test]
fn test_mqs_score_is_production_ready() {
    let score = MqsScore {
        model_id: "test".to_string(),
        raw_score: 950,
        normalized_score: 95.0,
        grade: "A".to_string(),
        gateways: vec![],
        gateways_passed: true,
        categories: CategoryScores::default(),
        total_tests: 100,
        tests_passed: 95,
        tests_failed: 5,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };
    assert!(score.is_production_ready());
}

/// Verify qualifies returns false for low-scoring model
#[test]
fn test_mqs_score_not_qualifies() {
    let score = MqsScore {
        model_id: "test".to_string(),
        raw_score: 500,
        normalized_score: 50.0,
        grade: "F".to_string(),
        gateways: vec![],
        gateways_passed: true,
        categories: CategoryScores::default(),
        total_tests: 100,
        tests_passed: 50,
        tests_failed: 50,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };
    assert!(!score.qualifies());
}

/// Verify qualifies returns false when gateways are not passed
#[test]
fn test_mqs_score_gateway_failed_not_qualifies() {
    let score = MqsScore {
        model_id: "test".to_string(),
        raw_score: 900,
        normalized_score: 90.0,
        grade: "A-".to_string(),
        gateways: vec![],
        gateways_passed: false,
        categories: CategoryScores::default(),
        total_tests: 100,
        tests_passed: 90,
        tests_failed: 10,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };
    assert!(!score.qualifies());
}

/// Verify default CategoryScores total is zero
#[test]
fn test_category_scores_default() {
    let scores = CategoryScores::default();
    assert_eq!(scores.total(), 0);
}

/// Verify breakdown returns correct (score, max) pairs per category
#[test]
fn test_category_scores_breakdown() {
    let scores = CategoryScores {
        qual: 180,
        perf: 150,
        stab: 160,
        comp: 140,
        edge: 130,
        regr: 120,
    };
    let breakdown = scores.breakdown();
    assert_eq!(breakdown.get("QUAL"), Some(&(180, 200)));
    assert_eq!(breakdown.get("PERF"), Some(&(150, 150)));
    assert_eq!(breakdown.get("STAB"), Some(&(160, 200)));
    assert_eq!(breakdown.get("COMP"), Some(&(140, 150)));
    assert_eq!(breakdown.get("EDGE"), Some(&(130, 150)));
    assert_eq!(breakdown.get("REGR"), Some(&(120, 150)));
}

/// Verify Penalty clone preserves all fields
#[test]
fn test_penalty_clone() {
    let penalty = Penalty {
        code: "TEST".to_string(),
        description: "Test penalty".to_string(),
        points: 10,
    };
    let cloned = penalty.clone();
    assert_eq!(cloned.code, penalty.code);
    assert_eq!(cloned.points, penalty.points);
}

/// Verify GatewayResult clone preserves id and passed fields
#[test]
fn test_gateway_result_clone() {
    let result = GatewayResult::passed("G1", "Test");
    let cloned = result.clone();
    assert_eq!(cloned.id, result.id);
    assert_eq!(cloned.passed, result.passed);
}

/// Verify MqsScore serializes to JSON with expected fields
#[test]
fn test_mqs_score_serialize() {
    let score = MqsScore {
        model_id: "test".to_string(),
        raw_score: 800,
        normalized_score: 80.0,
        grade: "B".to_string(),
        gateways: vec![],
        gateways_passed: true,
        categories: CategoryScores::default(),
        total_tests: 100,
        tests_passed: 80,
        tests_failed: 20,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };
    let json = serde_json::to_string(&score).expect("serialize");
    assert!(json.contains("test"));
    assert!(json.contains("800"));
}

/// Verify extract_category returns STAB for stability gate IDs
#[test]
fn test_extract_category_stab() {
    assert_eq!(MqsCalculator::extract_category("F-STAB-001"), "STAB");
}

/// Verify extract_category returns COMP for compliance gate IDs
#[test]
fn test_extract_category_comp() {
    assert_eq!(MqsCalculator::extract_category("F-COMP-001"), "COMP");
}

/// Verify extract_category returns EDGE for edge-case gate IDs
#[test]
fn test_extract_category_edge() {
    assert_eq!(MqsCalculator::extract_category("F-EDGE-001"), "EDGE");
}

/// Verify extract_category returns REGR for regression gate IDs
#[test]
fn test_extract_category_regr() {
    assert_eq!(MqsCalculator::extract_category("F-REGR-001"), "REGR");
}

/// Verify normalize_score returns 0.0 for zero raw score
#[test]
fn test_normalize_score_zero() {
    let calc = MqsCalculator::new();
    let score = calc.normalize_score(0, 0);
    assert_eq!(score, 0.0);
}

/// Verify check_gateways returns all five gateway results (G0-G4)
#[test]
fn test_mqs_calculator_check_gateways() {
    let calc = MqsCalculator::new();
    let collector = EvidenceCollector::new();

    let gateways = calc.check_gateways(collector.all());
    // Should have 5 gateways (G0-G4)
    assert_eq!(gateways.len(), 5);
}

/// Verify default MqsCalculator is constructible
#[test]
fn test_mqs_calculator_default() {
    let calc = MqsCalculator::default();
    let score = calc
        .calculate("test/model", &EvidenceCollector::new())
        .expect("Calculation failed");
    // Empty evidence = untested = F grade
    assert_eq!(score.grade, "F");
}

/// Verify MqsCalculator Debug format contains struct name
#[test]
fn test_mqs_calculator_debug() {
    let calc = MqsCalculator::new();
    let debug_str = format!("{calc:?}");
    assert!(debug_str.contains("MqsCalculator"));
}
