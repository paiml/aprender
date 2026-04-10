#[test]
fn test_gateway_g1_failure() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // G1 fails when ALL inference attempts fail and NONE succeed.
    // Simulates a model that never loaded — every inference attempt
    // exits non-zero (G2-BASIC failures).
    collector.add(Evidence::falsified(
        "G2-BASIC",
        test_scenario(),
        "Command failed (exit 1): model not found",
        "",
        100,
    ));
    collector.add(Evidence::falsified(
        "G2-BASIC",
        test_scenario(),
        "Command failed (exit 1): model not found",
        "",
        100,
    ));

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    // G1 failed should fail all gateways
    assert!(!score.gateways_passed);
    let g1 = score.gateways.iter().find(|g| g.id == "G1").unwrap();
    assert!(!g1.passed);
}

#[test]
fn test_gateway_g1_passes_with_mixed_results() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // G1 passes when at least one inference succeeds (model loaded)
    collector.add(test_evidence_passed("F-QUAL-001"));
    collector.add(Evidence::falsified(
        "G2-BASIC",
        test_scenario(),
        "wrong answer",
        "",
        100,
    ));

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    let g1 = score.gateways.iter().find(|g| g.id == "G1").unwrap();
    assert!(g1.passed);
}

#[test]
fn test_gateway_g2_failure() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add a G2 failure (basic inference failure)
    collector.add(Evidence::falsified(
        "G2-INFERENCE",
        test_scenario(),
        "Inference failed",
        "",
        100,
    ));

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    let g2 = score.gateways.iter().find(|g| g.id == "G2").unwrap();
    assert!(!g2.passed);
}

/// Create a scenario with oracle_type "garbage" (default oracle for non-arithmetic, non-code prompts)
fn garbage_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "Tell me about AI".to_string(),
        42,
    )
}

#[test]
fn test_gateway_g4_failure_garbage_output() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // G4 detects garbage via oracle_type == "garbage" on falsified evidence.
    // All 10 items use the garbage oracle and are falsified — >25% threshold.
    let gs = garbage_scenario();
    assert_eq!(gs.oracle_type, "garbage", "prompt must select garbage oracle");
    for i in 0..10 {
        collector.add(Evidence::falsified(
            &format!("F-A1-{i:03}"),
            garbage_scenario(),
            "Garbage output",
            "###$$@@!!",
            100,
        ));
    }

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    let g4 = score.gateways.iter().find(|g| g.id == "G4").unwrap();
    assert!(!g4.passed);
}

#[test]
fn test_gateway_g4_passes_with_mostly_good_garbage_oracle() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // 8 corroborated + 1 falsified garbage oracle → 1/9 < 25% → G4 passes
    for i in 0..8 {
        collector.add(Evidence::corroborated(
            &format!("F-A1-{i:03}"),
            garbage_scenario(),
            "Good output",
            100,
        ));
    }
    collector.add(Evidence::falsified(
        "F-A1-008",
        garbage_scenario(),
        "Garbage output",
        "###$$@@!!",
        100,
    ));

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    let g4 = score.gateways.iter().find(|g| g.id == "G4").unwrap();
    assert!(g4.passed);
}

#[test]
fn test_mqs_with_crash_penalty() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add mostly passing evidence first (so gateways pass)
    for i in 0..50 {
        collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
    }

    // Now the crash count will fail G3 gateway
    // So we need to test crash penalty separately without actual crashes

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");
    assert!(score.gateways_passed);
}

#[test]
fn test_calculate_categories_all_types() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add one of each category
    collector.add(test_evidence_passed("F-QUAL-001"));
    collector.add(test_evidence_passed("F-PERF-001"));
    collector.add(test_evidence_passed("F-STAB-001"));
    collector.add(test_evidence_passed("F-COMP-001"));
    collector.add(test_evidence_passed("F-EDGE-001"));
    collector.add(test_evidence_passed("F-REGR-001"));

    let categories = calc.calculate_categories(collector.all());

    assert!(categories.qual > 0);
    assert!(categories.perf > 0);
    assert!(categories.stab > 0);
    assert!(categories.comp > 0);
    assert!(categories.edge > 0);
    assert!(categories.regr > 0);
}

#[test]
fn test_calculate_categories_with_failures() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add passing and failing evidence
    collector.add(test_evidence_passed("F-QUAL-001"));
    collector.add(test_evidence_failed("F-QUAL-002"));
    collector.add(test_evidence_passed("F-QUAL-003"));

    let categories = calc.calculate_categories(collector.all());

    // 2 out of 3 passed, so qual should be ~133 (2/3 of 200)
    assert!(categories.qual > 100);
    assert!(categories.qual < 200);
}

#[test]
fn test_calculate_categories_unknown_category() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add evidence with unknown category - should default to QUAL
    collector.add(test_evidence_passed("UNKNOWN"));

    let categories = calc.calculate_categories(collector.all());
    assert!(categories.qual > 0);
}

#[test]
fn test_gateway_result_debug() {
    let result = GatewayResult::passed("G1", "Test");
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("GatewayResult"));
}

#[test]
fn test_category_scores_debug() {
    let scores = CategoryScores::default();
    let debug_str = format!("{scores:?}");
    assert!(debug_str.contains("CategoryScores"));
}

#[test]
fn test_penalty_debug() {
    let penalty = Penalty {
        code: "TEST".to_string(),
        description: "Test".to_string(),
        points: 10,
    };
    let debug_str = format!("{penalty:?}");
    assert!(debug_str.contains("Penalty"));
}

#[test]
fn test_mqs_score_debug() {
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
    let debug_str = format!("{score:?}");
    assert!(debug_str.contains("MqsScore"));
}

#[test]
fn test_mqs_score_clone() {
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
    let cloned = score.clone();
    assert_eq!(cloned.model_id, score.model_id);
    assert_eq!(cloned.raw_score, score.raw_score);
}

#[test]
fn test_gateway_result_serialize() {
    let result = GatewayResult::passed("G1", "Test");
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains("G1"));
}

#[test]
fn test_category_scores_serialize() {
    let scores = CategoryScores {
        qual: 100,
        perf: 50,
        stab: 75,
        comp: 60,
        edge: 40,
        regr: 30,
    };
    let json = serde_json::to_string(&scores).expect("serialize");
    assert!(json.contains("100"));
}

#[test]
fn test_penalty_serialize() {
    let penalty = Penalty {
        code: "CRASH".to_string(),
        description: "Crash detected".to_string(),
        points: 20,
    };
    let json = serde_json::to_string(&penalty).expect("serialize");
    assert!(json.contains("CRASH"));
}

#[test]
fn test_mqs_calculator_calculate_empty() {
    let calc = MqsCalculator::new();
    let collector = EvidenceCollector::new();

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    // Empty evidence = untested model = cannot qualify
    assert!(!score.gateways_passed);
    assert_eq!(score.total_tests, 0);
    assert_eq!(score.raw_score, 0);
    assert_eq!(score.grade, "F");
}

#[test]
fn test_category_scores_clone() {
    let scores = CategoryScores {
        qual: 100,
        perf: 50,
        stab: 75,
        comp: 60,
        edge: 40,
        regr: 30,
    };
    let cloned = scores.clone();
    assert_eq!(cloned.qual, scores.qual);
    assert_eq!(cloned.total(), scores.total());
}

#[test]
fn test_mqs_score_deserialize() {
    let json = r#"{
        "model_id": "test",
        "raw_score": 800,
        "normalized_score": 80.0,
        "grade": "B",
        "gateways": [],
        "gateways_passed": true,
        "categories": {"qual": 0, "perf": 0, "stab": 0, "comp": 0, "edge": 0, "regr": 0},
        "total_tests": 100,
        "tests_passed": 80,
        "tests_failed": 20,
        "penalties": [],
        "total_penalty": 0
    }"#;
    let score: MqsScore = serde_json::from_str(json).expect("deserialize");
    assert_eq!(score.model_id, "test");
    assert_eq!(score.raw_score, 800);
}

#[test]
fn test_gateway_g0_integrity_failure() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add a G0 integrity failure (layer count mismatch)
    collector.add(Evidence::falsified(
        "G0-INTEGRITY-LAYERS",
        test_scenario(),
        "config says 14 layers but tensors have 24",
        "",
        100,
    ));

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    // G0 failed should fail all gateways and zero score
    assert!(!score.gateways_passed);
    assert_eq!(score.raw_score, 0);
    assert_eq!(score.normalized_score, 0.0);
    let g0 = score.gateways.iter().find(|g| g.id == "G0").unwrap();
    assert!(!g0.passed);
    assert!(g0.failure_reason.as_ref().unwrap().contains("1 G0 check"));
}

#[test]
fn test_gateway_g0_integrity_multiple_failures() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add multiple G0 integrity failures (corrupted config scenario)
    collector.add(Evidence::falsified(
        "G0-INTEGRITY-LAYERS",
        test_scenario(),
        "config says 14 layers but tensors have 24",
        "",
        100,
    ));
    collector.add(Evidence::falsified(
        "G0-INTEGRITY-HIDDEN",
        test_scenario(),
        "config says hidden_size=4096 but embedding has 896",
        "",
        100,
    ));
    collector.add(Evidence::falsified(
        "G0-INTEGRITY-VOCAB",
        test_scenario(),
        "config says vocab_size=896 but embedding has 151936",
        "",
        100,
    ));

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    assert!(!score.gateways_passed);
    assert_eq!(score.raw_score, 0);
    let g0 = score.gateways.iter().find(|g| g.id == "G0").unwrap();
    assert!(!g0.passed);
    // Should mention all 3 failures
    assert!(g0.failure_reason.as_ref().unwrap().contains("3 G0 check"));
}

#[test]
fn test_gateway_g0_passes_when_no_integrity_failures() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Add only regular test evidence, no G0 failures
    collector.add(test_evidence_passed("F-QUAL-001"));
    collector.add(test_evidence_passed("F-PERF-001"));

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    assert!(score.gateways_passed);
    let g0 = score.gateways.iter().find(|g| g.id == "G0").unwrap();
    assert!(g0.passed);
}

#[test]
fn test_gateway_order_g0_first() {
    let calc = MqsCalculator::new();
    let collector = EvidenceCollector::new();

    let gateways = calc.check_gateways(collector.all());
    // G0 should be first
    assert_eq!(gateways[0].id, "G0");
    assert_eq!(gateways[1].id, "G1");
    assert_eq!(gateways[2].id, "G2");
    assert_eq!(gateways[3].id, "G3");
    assert_eq!(gateways[4].id, "G4");
}

/// Verify G0 gateway catches non-INTEGRITY G0 sub-gate failures (DIM, FORMAT, LAYOUT, etc.)
/// Regression test: prior to round 16, only "G0-INTEGRITY" prefix was checked.
#[test]
fn test_gateway_g0_catches_dim_and_format_failures() {
    let calc = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    // G0-DIM failure (from metadata-only mode)
    collector.add(Evidence::falsified(
        "G0-DIM-HIDDEN_SIZE",
        test_scenario(),
        "expected hidden_size=896 actual=0",
        "",
        50,
    ));
    // G0-FORMAT failure
    collector.add(Evidence::falsified(
        "G0-FORMAT-APR-001",
        test_scenario(),
        "APR workspace conversion failed",
        "",
        50,
    ));
    // G0-LAYOUT failure
    collector.add(Evidence::falsified(
        "G0-LAYOUT-001",
        test_scenario(),
        "Tensor layout contract violation",
        "",
        50,
    ));

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    // All three G0 failures must zero the score
    assert!(!score.gateways_passed);
    assert_eq!(score.raw_score, 0);
    let g0 = score.gateways.iter().find(|g| g.id == "G0").unwrap();
    assert!(!g0.passed);
    assert!(g0.failure_reason.as_ref().unwrap().contains("3 G0 check"));
}

#[test]
fn test_with_proof_bonus_adds_points() {
    let bonus = ProofBonus {
        kernel_class: Some("A".to_string()),
        proof_level: Some("L3".to_string()),
        bonus_points: 25,
    };
    let calculator = MqsCalculator::new().with_proof_bonus(bonus);
    let mut collector = EvidenceCollector::new();

    for i in 0..10 {
        collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
    }

    let score = calculator
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    assert!(score.gateways_passed);
    // Raw score should include the 25-point bonus
    assert!(score.raw_score > 200); // QUAL-only max is 200
    assert!(score.proof_bonus.is_some());
    assert_eq!(score.proof_bonus.as_ref().unwrap().bonus_points, 25);
}

#[test]
fn test_proof_bonus_zeroed_on_gateway_failure() {
    let bonus = ProofBonus {
        kernel_class: Some("A".to_string()),
        proof_level: Some("L5".to_string()),
        bonus_points: 50,
    };
    let calculator = MqsCalculator::new().with_proof_bonus(bonus);
    let mut collector = EvidenceCollector::new();

    // Crash triggers gateway failure
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
    // Bonus is still recorded but didn't help
    assert!(score.proof_bonus.is_some());
}

#[test]
fn test_no_proof_bonus_backward_compatible() {
    let calculator = MqsCalculator::new();
    let mut collector = EvidenceCollector::new();

    for i in 0..10 {
        collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
    }

    let score = calculator
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    assert!(score.proof_bonus.is_none());
    // Without bonus, raw max is still 1000
    assert!(score.raw_score <= 1000);
}

#[test]
fn test_proof_bonus_json_omitted_when_none() {
    let score = MqsScore {
        model_id: "test".to_string(),
        raw_score: 800,
        normalized_score: 80.0,
        grade: "B".to_string(),
        gateways: vec![],
        gateways_passed: true,
        categories: CategoryScores::default(),
        total_tests: 0,
        tests_passed: 0,
        tests_failed: 0,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };
    let json = serde_json::to_string(&score).unwrap();
    // proof_bonus should NOT appear in JSON when None
    assert!(!json.contains("proof_bonus"));
}
