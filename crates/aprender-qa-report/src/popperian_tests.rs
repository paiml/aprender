use super::*;

use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use apr_qa_runner::Evidence;

/// Create a default QA scenario for Popperian tests
fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test prompt".to_string(),
        42,
    )
}

/// Verify 100% corroboration produces strongly corroborated score
#[test]
fn test_popperian_all_corroborated() {
    let calculator = PopperianCalculator::new();
    let mut collector = EvidenceCollector::new();

    for i in 0..100 {
        collector.add(Evidence::corroborated(
            &format!("F-QUAL-{i:03}"),
            test_scenario(),
            "correct output",
            100,
        ));
    }

    let score = calculator.calculate("test/model", &collector);

    assert_eq!(score.corroborated, 100);
    assert_eq!(score.falsified, 0);
    assert!((score.corroboration_ratio - 1.0).abs() < 0.001);
    assert!(score.is_strongly_corroborated());
}

/// Verify mixed results produce correct corroboration ratio
#[test]
fn test_popperian_with_falsifications() {
    let calculator = PopperianCalculator::new();
    let mut collector = EvidenceCollector::new();

    // 90 corroborated
    for i in 0..90 {
        collector.add(Evidence::corroborated(
            &format!("F-QUAL-{i:03}"),
            test_scenario(),
            "correct",
            100,
        ));
    }

    // 10 falsified
    for i in 90..100 {
        collector.add(Evidence::falsified(
            &format!("F-QUAL-{i:03}"),
            test_scenario(),
            "wrong answer",
            "garbage",
            100,
        ));
    }

    let score = calculator.calculate("test/model", &collector);

    assert_eq!(score.corroborated, 90);
    assert_eq!(score.falsified, 10);
    assert!((score.corroboration_ratio - 0.9).abs() < 0.001);
    assert!(!score.is_strongly_corroborated());
}

/// Verify crash evidence triggers black swan detection
#[test]
fn test_popperian_black_swan_detection() {
    let calculator = PopperianCalculator::new();
    let mut collector = EvidenceCollector::new();

    // Normal passes
    for i in 0..95 {
        collector.add(Evidence::corroborated(
            &format!("F-QUAL-{i:03}"),
            test_scenario(),
            "ok",
            100,
        ));
    }

    // One crash (black swan)
    collector.add(Evidence::crashed(
        "F-QUAL-099",
        test_scenario(),
        "SIGSEGV",
        -11,
        0,
    ));

    let score = calculator.calculate("test/model", &collector);

    assert!(score.has_black_swans());
    assert_eq!(score.black_swan_count, 1);
    assert!(!score.is_strongly_corroborated());
}

/// Verify severity determination for all gate ID patterns
#[test]
fn test_severity_determination() {
    assert_eq!(PopperianCalculator::determine_severity("G1-LOAD"), 5);
    assert_eq!(PopperianCalculator::determine_severity("F-QUAL-P0-001"), 5);
    assert_eq!(PopperianCalculator::determine_severity("F-QUAL-P1-001"), 4);
    assert_eq!(PopperianCalculator::determine_severity("F-QUAL-P2-001"), 3);
    assert_eq!(PopperianCalculator::determine_severity("F-EDGE-001"), 3);
    assert_eq!(PopperianCalculator::determine_severity("F-PERF-001"), 2);
    assert_eq!(PopperianCalculator::determine_severity("F-OTHER-001"), 1);
}

/// Verify gate_to_hypothesis produces meaningful descriptions
#[test]
fn test_gate_to_hypothesis() {
    assert!(PopperianCalculator::gate_to_hypothesis("F-QUAL-001").contains("valid output"));
    assert!(PopperianCalculator::gate_to_hypothesis("F-PERF-001").contains("performance"));
    assert!(PopperianCalculator::gate_to_hypothesis("F-STAB-001").contains("stable"));
}

/// Verify falsification_summary text for different result states
#[test]
fn test_falsification_summary() {
    let score = PopperianScore {
        model_id: "test".to_string(),
        hypotheses_tested: 100,
        corroborated: 100,
        falsified: 0,
        inconclusive: 0,
        corroboration_ratio: 1.0,
        severity_weighted_score: 1.0,
        confidence_level: 0.95,
        reproducibility_index: 1.0,
        black_swan_count: 0,
        falsifications: vec![],
    };

    assert!(score
        .falsification_summary()
        .contains("strongly corroborated"));

    let score_with_failures = PopperianScore {
        falsified: 5,
        hypotheses_tested: 100,
        ..score
    };

    assert!(score_with_failures
        .falsification_summary()
        .contains("5 of 100"));
}

/// Verify confidence increases with larger sample sizes
#[test]
fn test_confidence_calculation() {
    let calculator = PopperianCalculator::new();

    // Small sample = lower confidence
    let small_conf = calculator.calculate_confidence(10, 0.9);
    // Large sample = higher confidence
    let large_conf = calculator.calculate_confidence(1000, 0.9);

    assert!(large_conf > small_conf);
}

/// Verify reproducibility index is 1.0 when no failures exist
#[test]
fn test_reproducibility_no_failures() {
    let calculator = PopperianCalculator::new();
    let empty: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    let index = calculator.calculate_reproducibility(&empty, 100);
    assert!((index - 1.0).abs() < 0.001);
}

/// Verify reproducibility is 1.0 for consistently reproducing failures
#[test]
fn test_reproducibility_with_consistent_failures() {
    let calculator = PopperianCalculator::new();
    let mut failures = std::collections::HashMap::new();
    failures.insert("F-001".to_string(), 5); // Consistent
    failures.insert("F-002".to_string(), 3); // Consistent

    let index = calculator.calculate_reproducibility(&failures, 100);
    // All failures are consistent (appeared more than once)
    assert!((index - 1.0).abs() < 0.001);
}

/// Verify reproducibility is 0.0 for sporadic single-occurrence failures
#[test]
fn test_reproducibility_with_sporadic_failures() {
    let calculator = PopperianCalculator::new();
    let mut failures = std::collections::HashMap::new();
    failures.insert("F-001".to_string(), 1); // Sporadic
    failures.insert("F-002".to_string(), 1); // Sporadic

    let index = calculator.calculate_reproducibility(&failures, 100);
    // No consistent failures
    assert!((index - 0.0).abs() < 0.001);
}

/// Verify reproducibility defaults to 1.0 for zero total tests
#[test]
fn test_reproducibility_zero_total() {
    let calculator = PopperianCalculator::new();
    let failures = std::collections::HashMap::new();

    let index = calculator.calculate_reproducibility(&failures, 0);
    assert!((index - 1.0).abs() < 0.001);
}

/// Verify confidence is 0.0 for zero sample size
#[test]
fn test_confidence_zero_samples() {
    let calculator = PopperianCalculator::new();
    let conf = calculator.calculate_confidence(0, 0.9);
    assert!((conf - 0.0).abs() < 0.001);
}

/// Verify COMP gate maps to compatible hypothesis
#[test]
fn test_gate_to_hypothesis_comp() {
    assert!(PopperianCalculator::gate_to_hypothesis("F-COMP-001").contains("compatible"));
}

/// Verify EDGE gate maps to edge cases hypothesis
#[test]
fn test_gate_to_hypothesis_edge() {
    assert!(PopperianCalculator::gate_to_hypothesis("F-EDGE-001").contains("edge cases"));
}

/// Verify REGR gate maps to consistent hypothesis
#[test]
fn test_gate_to_hypothesis_regr() {
    assert!(PopperianCalculator::gate_to_hypothesis("F-REGR-001").contains("consistent"));
}

/// Verify unknown gate includes gate ID in hypothesis text
#[test]
fn test_gate_to_hypothesis_unknown() {
    let result = PopperianCalculator::gate_to_hypothesis("F-UNKNOWN-001");
    assert!(result.contains("F-UNKNOWN-001"));
}

/// Verify has_black_swans returns true when count > 0
#[test]
fn test_popperian_score_has_black_swans() {
    let score = PopperianScore {
        model_id: "test".to_string(),
        hypotheses_tested: 100,
        corroborated: 99,
        falsified: 1,
        inconclusive: 0,
        corroboration_ratio: 0.99,
        severity_weighted_score: 0.99,
        confidence_level: 0.95,
        reproducibility_index: 1.0,
        black_swan_count: 1,
        falsifications: vec![],
    };
    assert!(score.has_black_swans());
}

/// Verify has_black_swans returns false when count is 0
#[test]
fn test_popperian_score_no_black_swans() {
    let score = PopperianScore {
        model_id: "test".to_string(),
        hypotheses_tested: 100,
        corroborated: 90,
        falsified: 10,
        inconclusive: 0,
        corroboration_ratio: 0.9,
        severity_weighted_score: 0.9,
        confidence_level: 0.9,
        reproducibility_index: 1.0,
        black_swan_count: 0,
        falsifications: vec![],
    };
    assert!(!score.has_black_swans());
}

/// Verify STAB gate severity is 3
#[test]
fn test_severity_stab() {
    assert_eq!(PopperianCalculator::determine_severity("F-STAB-001"), 3);
}

/// Verify FalsificationDetail clone preserves gate_id
#[test]
fn test_falsification_detail_clone() {
    let detail = FalsificationDetail {
        gate_id: "F-001".to_string(),
        hypothesis: "test".to_string(),
        evidence: "failed".to_string(),
        occurrence_count: 1,
        severity: 3,
        is_black_swan: false,
    };
    let cloned = detail.clone();
    assert_eq!(cloned.gate_id, detail.gate_id);
}

/// Verify PopperianScore JSON serialization
#[test]
fn test_popperian_score_serialize() {
    let score = PopperianScore {
        model_id: "test".to_string(),
        hypotheses_tested: 100,
        corroborated: 100,
        falsified: 0,
        inconclusive: 0,
        corroboration_ratio: 1.0,
        severity_weighted_score: 1.0,
        confidence_level: 0.95,
        reproducibility_index: 1.0,
        black_swan_count: 0,
        falsifications: vec![],
    };
    let json = serde_json::to_string(&score).expect("serialize");
    assert!(json.contains("test"));
}

/// Verify timeout evidence is counted as inconclusive, not falsified
#[test]
fn test_popperian_with_timeout() {
    let calculator = PopperianCalculator::new();
    let mut collector = EvidenceCollector::new();

    collector.add(Evidence::timeout("F-PERF-001", test_scenario(), 30000));

    let score = calculator.calculate("test/model", &collector);
    // Timeout is treated as inconclusive, not falsified
    assert_eq!(score.inconclusive, 1);
    assert_eq!(score.falsified, 0);
}

/// Verify G0-G4 gateway prefixes map to specific meaningful hypotheses
#[test]
fn test_gate_to_hypothesis_gateways() {
    let g0 = PopperianCalculator::gate_to_hypothesis("G0-DIM-CONFIG");
    assert!(
        g0.contains("consistent") || g0.contains("metadata") || g0.contains("layout"),
        "G0 hypothesis should mention metadata/layout consistency, got: {g0}"
    );

    let g1 = PopperianCalculator::gate_to_hypothesis("G1-LOAD");
    assert!(
        g1.contains("load") || g1.contains("timeout"),
        "G1 hypothesis should mention loading, got: {g1}"
    );

    let g2 = PopperianCalculator::gate_to_hypothesis("G2-BASIC");
    assert!(
        g2.contains("inference") || g2.contains("output"),
        "G2 hypothesis should mention inference, got: {g2}"
    );

    let g3 = PopperianCalculator::gate_to_hypothesis("G3-STABLE");
    assert!(
        g3.contains("crash") || g3.contains("panic"),
        "G3 hypothesis should mention crashes/panics, got: {g3}"
    );

    let g4 = PopperianCalculator::gate_to_hypothesis("G4-VALID");
    assert!(
        g4.contains("garbage") || g4.contains("output"),
        "G4 hypothesis should mention garbage detection, got: {g4}"
    );
}

/// Verify G0 sub-gates (G0-DIM-*) also get the gateway hypothesis
#[test]
fn test_gate_to_hypothesis_g0_dim_variants() {
    let tokenizer = PopperianCalculator::gate_to_hypothesis("G0-DIM-TOKENIZER_EXISTS");
    let eos = PopperianCalculator::gate_to_hypothesis("G0-DIM-EOS_TOKEN_VALID");
    // Both G0-DIM-* sub-gates should get the G0 hypothesis (StartsWith "G0-")
    assert_eq!(
        tokenizer, eos,
        "All G0- sub-gates should share the same hypothesis"
    );
    assert!(
        !tokenizer.contains("G0-DIM"),
        "G0 hypothesis should not fall back to generic 'Hypothesis for ...' form"
    );
}
