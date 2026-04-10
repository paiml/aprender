// ── FALSIFY-MQS-RISK: Risk tier derivation ───────────────────────────────
//
// Prediction: risk_tier() is deterministically derived from gateways_passed,
// normalized_score, and total_penalty.

/// Helper to build an MqsScore with given parameters for risk tier testing
fn risk_tier_score(gateways_passed: bool, normalized_score: f64, total_penalty: u32) -> MqsScore {
    MqsScore {
        model_id: "test".to_string(),
        raw_score: 0,
        normalized_score,
        grade: String::new(),
        gateways: vec![],
        gateways_passed,
        categories: CategoryScores::default(),
        total_tests: 0,
        tests_passed: 0,
        tests_failed: 0,
        penalties: vec![],
        total_penalty,
        proof_bonus: None,
    }
}

/// Verify risk_tier returns BLOCKED when gateways fail
#[test]
fn test_risk_tier_blocked() {
    let score = risk_tier_score(false, 99.0, 0);
    assert_eq!(score.risk_tier(), "BLOCKED");
}

/// Verify risk_tier returns MINIMAL for high score with zero penalty
#[test]
fn test_risk_tier_minimal() {
    let score = risk_tier_score(true, 96.0, 0);
    assert_eq!(score.risk_tier(), "MINIMAL");
}

/// Verify risk_tier returns LOW for score >= 90 with small penalty
#[test]
fn test_risk_tier_low() {
    let score = risk_tier_score(true, 91.0, 15);
    assert_eq!(score.risk_tier(), "LOW");
}

/// Verify risk_tier returns MODERATE for score >= 80 with moderate penalty
#[test]
fn test_risk_tier_moderate() {
    let score = risk_tier_score(true, 85.0, 30);
    assert_eq!(score.risk_tier(), "MODERATE");
}

/// Verify risk_tier returns ELEVATED for score >= 70 with larger penalty
#[test]
fn test_risk_tier_elevated() {
    let score = risk_tier_score(true, 72.0, 80);
    assert_eq!(score.risk_tier(), "ELEVATED");
}

/// Verify risk_tier returns HIGH for score >= 60
#[test]
fn test_risk_tier_high() {
    let score = risk_tier_score(true, 62.0, 200);
    assert_eq!(score.risk_tier(), "HIGH");
}

/// Verify risk_tier returns VERY HIGH for score >= 40
#[test]
fn test_risk_tier_very_high() {
    let score = risk_tier_score(true, 45.0, 500);
    assert_eq!(score.risk_tier(), "VERY HIGH");
}

/// Verify risk_tier returns CRITICAL for score < 40
#[test]
fn test_risk_tier_critical() {
    let score = risk_tier_score(true, 30.0, 0);
    assert_eq!(score.risk_tier(), "CRITICAL");
}

/// Verify high score with high penalty drops to worse tier
#[test]
fn test_risk_tier_penalty_pushes_tier_down() {
    // Score >= 95 but penalty > 0 should NOT be MINIMAL
    let score = risk_tier_score(true, 96.0, 10);
    assert_ne!(score.risk_tier(), "MINIMAL");
    // Should fall to LOW (score >= 90, penalty <= 20)
    assert_eq!(score.risk_tier(), "LOW");
}

/// Verify is_production_ready returns false when gateways fail despite high score
#[test]
fn test_is_production_ready_gateway_failed() {
    let score = risk_tier_score(false, 99.0, 0);
    assert!(!score.is_production_ready());
}

// ── FALSIFY-MQS-CATEGORY: Prefix-based category extraction ──────────────
//
// Prediction: extract_category resolves gate IDs via PREFIX_MAP before
// falling back to the F-{CATEGORY}-xxx pattern.

/// Verify F-CONV-RT prefix maps to REGR category
#[test]
fn test_extract_category_conv_rt_prefix() {
    assert_eq!(MqsCalculator::extract_category("F-CONV-RT-001"), "REGR");
}

/// Verify F-CONV-IDEM prefix maps to REGR category
#[test]
fn test_extract_category_conv_idem_prefix() {
    assert_eq!(MqsCalculator::extract_category("F-CONV-IDEM-001"), "REGR");
}

/// Verify F-CONV-COM prefix maps to REGR category
#[test]
fn test_extract_category_conv_com_prefix() {
    assert_eq!(MqsCalculator::extract_category("F-CONV-COM-001"), "REGR");
}

/// Verify F-CONV prefix (without sub-prefix) maps to COMP category
#[test]
fn test_extract_category_conv_prefix() {
    assert_eq!(MqsCalculator::extract_category("F-CONV-001"), "COMP");
}

/// Verify F-CONV-BE prefix maps to COMP (caught by F-CONV before F-CONV-BE)
#[test]
fn test_extract_category_conv_be_prefix() {
    assert_eq!(MqsCalculator::extract_category("F-CONV-BE-001"), "COMP");
}

/// Verify F-CONTRACT prefix maps to COMP category
#[test]
fn test_extract_category_contract_prefix() {
    assert_eq!(MqsCalculator::extract_category("F-CONTRACT-001"), "COMP");
}

/// Verify G0- prefix maps to STAB category
#[test]
fn test_extract_category_g0_prefix() {
    assert_eq!(MqsCalculator::extract_category("G0-INTEGRITY-LAYERS"), "STAB");
}

// ── Normalization edge cases ─────────────────────────────────────────────

/// Verify normalize_score_with_max uses expanded denominator for proof bonus
#[test]
fn test_normalize_with_proof_bonus_denominator() {
    let calc = MqsCalculator::new();
    // Same raw score with different max_possible should give different normalized
    let without_bonus = calc.normalize_score_with_max(800, 800, 1000);
    let with_bonus = calc.normalize_score_with_max(800, 800, 1050);
    // Expanded denominator means lower normalized score for same raw
    assert!(with_bonus < without_bonus);
}

/// Verify full MQS calculation with proof bonus expands denominator
#[test]
fn test_calculate_with_proof_bonus_full() {
    let bonus = ProofBonus {
        kernel_class: Some("A".to_string()),
        proof_level: Some("L5".to_string()),
        bonus_points: 50,
    };
    let calc = MqsCalculator::new().with_proof_bonus(bonus);
    let mut collector = EvidenceCollector::new();

    for i in 0..10 {
        collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-PERF-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-STAB-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-COMP-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-EDGE-{i:03}")));
        collector.add(test_evidence_passed(&format!("F-REGR-{i:03}")));
    }

    let score = calc
        .calculate("test/model", &collector)
        .expect("Calculation failed");

    // Raw score should be 1000 + 50 bonus = 1050
    assert_eq!(score.raw_score, 1050);
    // Normalized with expanded denominator (1050/1050)
    assert!(score.normalized_score > 99.0);
}

/// Verify serve battery gate IDs map to correct MQS categories
#[test]
fn test_extract_category_serve_battery() {
    // Basic inference → QUAL
    assert_eq!(MqsCalculator::extract_category("F-A5-001"), "QUAL");
    assert_eq!(MqsCalculator::extract_category("F-A6-001"), "QUAL");
    // Streaming → COMP
    assert_eq!(MqsCalculator::extract_category("F-A5-STREAM-001"), "COMP");
    assert_eq!(MqsCalculator::extract_category("F-A6-CSTREAM-001"), "COMP");
    // Error handling → STAB
    assert_eq!(MqsCalculator::extract_category("F-A5-ERR-001"), "STAB");
    assert_eq!(MqsCalculator::extract_category("F-A1-ERR-001"), "STAB");
    // Performance → PERF
    assert_eq!(MqsCalculator::extract_category("F-A5-METRICS-001"), "PERF");
    assert_eq!(MqsCalculator::extract_category("F-A5-PERF-001"), "PERF");
    // API endpoints → COMP
    assert_eq!(MqsCalculator::extract_category("F-A5-INFO-001"), "COMP");
    assert_eq!(MqsCalculator::extract_category("F-A5-MODELS-001"), "COMP");
    assert_eq!(MqsCalculator::extract_category("F-A5-TMPL-001"), "COMP");
    // Character edge cases → EDGE
    assert_eq!(MqsCalculator::extract_category("F-A5-CHARS-001"), "EDGE");
    // Non-serve battery should not match
    assert_eq!(MqsCalculator::extract_category("F-A7-ERR-001"), "QUAL");
}

/// Verify HF parity and layout gate IDs map correctly
#[test]
fn test_extract_category_hf_parity_and_layout() {
    assert_eq!(MqsCalculator::extract_category("F-HF-PARITY-001"), "QUAL");
    assert_eq!(MqsCalculator::extract_category("F-HF-PARITY-004"), "QUAL");
    assert_eq!(MqsCalculator::extract_category("F-LAYOUT-002"), "STAB");
}

/// Bug #62: Verify previously-unmapped gate IDs now map to correct categories.
/// These all defaulted to QUAL before the PREFIX_MAP additions.
#[test]
fn test_extract_category_previously_unmapped() {
    // Integrity checks → STAB
    assert_eq!(MqsCalculator::extract_category("F-INT-001"), "STAB");
    assert_eq!(MqsCalculator::extract_category("F-INT-005"), "STAB");
    // Security checks → STAB
    assert_eq!(MqsCalculator::extract_category("F-SEC-003"), "STAB");
    assert_eq!(MqsCalculator::extract_category("F-SEC-PATH-001"), "STAB");
    assert_eq!(MqsCalculator::extract_category("F-SEC-INJECT-001"), "STAB");
    // Numerical stability → STAB
    assert_eq!(MqsCalculator::extract_category("F-NUM-001"), "STAB");
    assert_eq!(MqsCalculator::extract_category("F-NUM-004"), "STAB");
    // Performance profiling → PERF
    assert_eq!(MqsCalculator::extract_category("F-PROFILE-001"), "PERF");
    assert_eq!(MqsCalculator::extract_category("F-PROFILE-CI-001"), "PERF");
    // Golden rule (conversion) → REGR
    assert_eq!(MqsCalculator::extract_category("F-GOLDEN-RULE-001"), "REGR");
    assert_eq!(MqsCalculator::extract_category("F-GOLDEN-RULE-003"), "REGR");
}
