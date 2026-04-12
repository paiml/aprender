use super::*;
use crate::mqs::{CategoryScores, GatewayResult, Penalty};
use crate::popperian::FalsificationDetail;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

/// Create a default QA scenario for markdown tests
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

/// Create a sample MQS score for markdown rendering tests
fn test_mqs_score() -> MqsScore {
    MqsScore {
        model_id: "qwen2.5-coder-7b".to_string(),
        raw_score: 847,
        normalized_score: 84.7,
        grade: "B+".to_string(),
        gateways: vec![
            GatewayResult::passed("G1", "Model loads successfully"),
            GatewayResult::passed("G2", "Basic inference works"),
            GatewayResult::passed("G3", "No crashes"),
            GatewayResult::passed("G4", "Output is not garbage"),
        ],
        gateways_passed: true,
        categories: CategoryScores {
            qual: 180,
            perf: 120,
            stab: 180,
            comp: 130,
            edge: 120,
            regr: 117,
        },
        total_tests: 100,
        tests_passed: 85,
        tests_failed: 15,
        penalties: vec![Penalty {
            code: "TIMEOUT".to_string(),
            description: "3 timeouts detected".to_string(),
            points: 30,
        }],
        total_penalty: 30,
        proof_bonus: None,
    }
}

/// Create a sample Popperian score for markdown rendering tests
fn test_popperian_score() -> PopperianScore {
    PopperianScore {
        model_id: "qwen2.5-coder-7b".to_string(),
        hypotheses_tested: 100,
        corroborated: 85,
        falsified: 15,
        inconclusive: 0,
        corroboration_ratio: 0.85,
        severity_weighted_score: 0.82,
        confidence_level: 0.95,
        reproducibility_index: 0.98,
        black_swan_count: 0,
        falsifications: vec![FalsificationDetail {
            gate_id: "F-PERF-042".to_string(),
            hypothesis: "Inference completes under 100ms".to_string(),
            evidence: "Actual: 142ms".to_string(),
            severity: 3,
            is_black_swan: false,
            occurrence_count: 2,
        }],
    }
}

/// Verify basic RAG markdown contains model name and summary
#[test]
fn test_generate_rag_markdown_basic() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("# Model Qualification: qwen2.5-coder-7b"));
    assert!(md.contains("## Summary"));
    assert!(md.contains("847/1000"));
    assert!(md.contains("B+"));
}

/// Verify RAG markdown contains gateway check sections
#[test]
fn test_generate_rag_markdown_contains_gateways() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("## Gateway Checks"));
    assert!(md.contains("G1"));
    assert!(md.contains("G2"));
    assert!(md.contains("G3"));
    assert!(md.contains("G4"));
    assert!(md.contains("✓ PASS"));
}

/// Verify RAG markdown contains category score sections
#[test]
fn test_generate_rag_markdown_contains_categories() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("## Category Scores"));
    assert!(md.contains("QUAL"));
    assert!(md.contains("PERF"));
    assert!(md.contains("STAB"));
}

/// Verify RAG markdown contains falsification details
#[test]
fn test_generate_rag_markdown_contains_falsifications() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("## Falsifications"));
    assert!(md.contains("F-PERF-042"));
    assert!(md.contains("**Severity**: 3/5"));
}

/// Verify RAG markdown contains penalty section
#[test]
fn test_generate_rag_markdown_contains_penalties() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("## Penalties Applied"));
    assert!(md.contains("TIMEOUT"));
    assert!(md.contains("-30"));
}

/// Verify RAG markdown contains Popperian analysis section
#[test]
fn test_generate_rag_markdown_contains_popperian() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("## Popperian Analysis"));
    assert!(md.contains("Hypotheses Tested"));
    assert!(md.contains("Corroboration Rate"));
}

/// Verify RAG markdown contains metadata section
#[test]
fn test_generate_rag_markdown_contains_metadata() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("## Metadata"));
    assert!(md.contains("Production Ready"));
}

/// Verify RAG markdown renders evidence details by category
#[test]
fn test_generate_rag_markdown_with_evidence() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let mut collector = EvidenceCollector::new();

    collector.add(Evidence::corroborated(
        "F-QUAL-001",
        test_scenario(),
        "4",
        100,
    ));
    collector.add(Evidence::falsified(
        "F-QUAL-002",
        test_scenario(),
        "Wrong answer",
        "5",
        200,
    ));

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("## Test Results by Category"));
    assert!(md.contains("QUAL Tests"));
    assert!(md.contains("F-QUAL-002"));
}

/// Verify index entry contains model name, score, and qualification status
#[test]
fn test_generate_index_entry() {
    let mqs = test_mqs_score();
    let entry = generate_index_entry(&mqs);

    assert!(entry.contains("qwen2.5-coder-7b"));
    assert!(entry.contains("847/1000"));
    assert!(entry.contains("B+"));
    assert!(entry.contains("QUALIFIED (Conditional)"));
}

/// Verify CERTIFIED status for high-score models
#[test]
fn test_qualification_status_certified() {
    let mut mqs = test_mqs_score();
    mqs.normalized_score = 95.0;
    mqs.gateways_passed = true;

    assert_eq!(qualification_status(&mqs), "CERTIFIED");
}

/// Verify all qualification status tiers map to correct labels
#[test]
fn test_qualification_status_tiers() {
    // Each (score, gateways_passed, expected_status)
    let cases: &[(f64, bool, &str)] = &[
        (95.0, true, "CERTIFIED"),
        (87.0, true, "CERTIFIED (Conditional)"),
        (84.7, true, "QUALIFIED (Conditional)"),
        (75.0, true, "PROVISIONAL"),
        (65.0, true, "UNDER REVIEW"),
        (50.0, true, "NEEDS IMPROVEMENT"),
        (40.0, true, "REJECTED"),
        (95.0, false, "REJECTED (Gateway Failure)"),
    ];
    for &(score, gw, expected) in cases {
        let mut mqs = test_mqs_score();
        mqs.normalized_score = score;
        mqs.gateways_passed = gw;
        assert_eq!(
            qualification_status(&mqs),
            expected,
            "score={score}, gw={gw}"
        );
    }
}

/// Verify category extraction from gate IDs matches MQS calculator logic
#[test]
fn test_extract_category() {
    // Standard F-{CATEGORY}-xxx pattern
    assert_eq!(extract_category("F-QUAL-001"), "QUAL");
    assert_eq!(extract_category("F-PERF-042"), "PERF");
    assert_eq!(extract_category("F-STAB-100"), "STAB");
    assert_eq!(extract_category("F-COMP-001"), "COMP");
    assert_eq!(extract_category("F-EDGE-001"), "EDGE");
    assert_eq!(extract_category("F-REGR-001"), "REGR");
    // Prefix-mapped gate IDs
    assert_eq!(extract_category("G0-INTEGRITY"), "STAB");
    assert_eq!(extract_category("G0-DIM-001"), "STAB");
    assert_eq!(extract_category("F-CONV-RT-001"), "REGR");
    assert_eq!(extract_category("F-CONV-IDEM-001"), "REGR");
    assert_eq!(extract_category("F-CONV-COM-001"), "REGR");
    assert_eq!(extract_category("F-CONV-001"), "COMP");
    assert_eq!(extract_category("F-CONTRACT-001"), "COMP");
    // Modality gate IDs (F-A1 through F-A6): quality by default, suffix-mapped otherwise
    assert_eq!(extract_category("F-A1-001"), "QUAL");
    assert_eq!(extract_category("F-A5-COMP-001"), "COMP"); // completions endpoint → API compatibility
    assert_eq!(extract_category("F-A5-CHAT-001"), "COMP"); // chat completions → API compatibility
    assert_eq!(extract_category("F-A5-STREAM-001"), "COMP"); // streaming → API compatibility
    assert_eq!(extract_category("F-A5-ERR-001"), "STAB"); // error handling → stability
    assert_eq!(extract_category("F-A5-METRICS-001"), "PERF"); // metrics → performance
    assert_eq!(extract_category("F-A5-CHARS-001"), "EDGE"); // character edge cases → edge
                                                            // Unknown/malformed gate IDs default to QUAL
    assert_eq!(extract_category("UNKNOWN"), "QUAL");
}

/// Verify evidence detail markdown contains gate ID and outcome
#[test]
fn test_generate_evidence_detail() {
    let evidence = Evidence::corroborated("F-QUAL-001", test_scenario(), "output text", 150);
    let md = generate_evidence_detail(&evidence);

    assert!(md.contains("### F-QUAL-001"));
    assert!(md.contains("Outcome"));
    assert!(md.contains("Corroborated"));
    assert!(md.contains("150ms"));
}

/// Verify evidence detail includes performance metrics when present
#[test]
fn test_generate_evidence_detail_with_metrics() {
    let mut evidence = Evidence::corroborated("F-PERF-001", test_scenario(), "output", 100);
    evidence.metrics.tokens_per_second = Some(150.5);
    evidence.metrics.time_to_first_token_ms = Some(25.3);
    evidence.metrics.memory_peak_mb = Some(4096);

    let md = generate_evidence_detail(&evidence);

    assert!(md.contains("**Tokens/sec**: 150.5"));
    assert!(md.contains("**Time to First Token**: 25.3ms"));
    assert!(md.contains("**Peak Memory**: 4096 MB"));
}

/// Verify penalties section is omitted when no penalties exist
#[test]
fn test_generate_rag_markdown_no_penalties() {
    let mut mqs = test_mqs_score();
    mqs.penalties.clear();
    mqs.total_penalty = 0;

    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    // Should not contain penalties section if no penalties
    assert!(!md.contains("## Penalties Applied"));
}

/// Verify falsifications section is omitted when none exist
#[test]
fn test_generate_rag_markdown_no_falsifications() {
    let mqs = test_mqs_score();
    let mut popperian = test_popperian_score();
    popperian.falsifications.clear();

    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    // Should still have the section header but no falsifications listed
    // Actually, with empty falsifications, the section is skipped
    assert!(!md.contains("### 1:"));
}

/// Verify markdown renders gateway failure with FAIL marker
#[test]
fn test_generate_rag_markdown_gateway_failure() {
    let mut mqs = test_mqs_score();
    mqs.gateways = vec![
        GatewayResult::passed("G1", "Model loads successfully"),
        GatewayResult::failed("G2", "Basic inference works", "Inference failed"),
        GatewayResult::passed("G3", "No crashes"),
        GatewayResult::passed("G4", "Output is not garbage"),
    ];
    mqs.gateways_passed = false;

    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("✗ FAIL"));
    assert!(md.contains("Inference failed"));
}

/// Verify markdown renders black swan events with count and flag
#[test]
fn test_generate_rag_markdown_black_swan() {
    let mqs = test_mqs_score();
    let mut popperian = test_popperian_score();
    popperian.black_swan_count = 2;
    popperian.falsifications = vec![FalsificationDetail {
        gate_id: "F-STAB-001".to_string(),
        hypothesis: "Model does not crash".to_string(),
        evidence: "SIGSEGV".to_string(),
        severity: 5,
        is_black_swan: true,
        occurrence_count: 1,
    }];

    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("Black Swans**: 2"));
    assert!(md.contains("Black Swan**: Yes"));
}

/// Verify markdown renders evidence for all six test categories
#[test]
fn test_generate_rag_markdown_multiple_categories() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let mut collector = EvidenceCollector::new();

    collector.add(Evidence::corroborated(
        "F-QUAL-001",
        test_scenario(),
        "ok",
        100,
    ));
    collector.add(Evidence::corroborated(
        "F-PERF-001",
        test_scenario(),
        "ok",
        100,
    ));
    collector.add(Evidence::falsified(
        "F-STAB-001",
        test_scenario(),
        "fail",
        "",
        100,
    ));
    collector.add(Evidence::corroborated(
        "F-COMP-001",
        test_scenario(),
        "ok",
        100,
    ));
    collector.add(Evidence::corroborated(
        "F-EDGE-001",
        test_scenario(),
        "ok",
        100,
    ));
    collector.add(Evidence::corroborated(
        "F-REGR-001",
        test_scenario(),
        "ok",
        100,
    ));

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("### QUAL Tests"));
    assert!(md.contains("### PERF Tests"));
    assert!(md.contains("### STAB Tests"));
    assert!(md.contains("### COMP Tests"));
    assert!(md.contains("### EDGE Tests"));
    assert!(md.contains("### REGR Tests"));
}

/// Verify markdown truncates failure lists beyond 10 items
#[test]
fn test_generate_rag_markdown_many_failures() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let mut collector = EvidenceCollector::new();

    // Add more than 10 failures to test truncation
    for i in 0..15 {
        collector.add(Evidence::falsified(
            &format!("F-QUAL-{:03}", i),
            test_scenario(),
            &format!("Failure {}", i),
            "",
            100,
        ));
    }

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    assert!(md.contains("... and 5 more failures"));
}

/// Verify falsified evidence detail contains reason text
#[test]
fn test_generate_evidence_detail_falsified() {
    let evidence = Evidence::falsified(
        "F-EDGE-001",
        test_scenario(),
        "Empty input caused crash",
        "",
        50,
    );
    let md = generate_evidence_detail(&evidence);

    assert!(md.contains("Falsified"));
    assert!(md.contains("Empty input caused crash"));
}

/// Verify timeout evidence detail shows duration
#[test]
fn test_generate_evidence_detail_timeout() {
    let evidence = Evidence::timeout("F-PERF-001", test_scenario(), 30000);
    let md = generate_evidence_detail(&evidence);

    assert!(md.contains("Timeout"));
    assert!(md.contains("30000ms"));
}

/// Verify crashed evidence detail shows exit code
#[test]
fn test_generate_evidence_detail_crashed() {
    let evidence = Evidence::crashed("F-STAB-001", test_scenario(), "SIGSEGV", -11, 100);
    let md = generate_evidence_detail(&evidence);

    assert!(md.contains("Crashed"));
    assert!(md.contains("-11"));
}

/// Verify long output is truncated in evidence detail
#[test]
fn test_generate_evidence_detail_with_output() {
    let long_output = "a".repeat(300);
    let evidence = Evidence::corroborated("F-QUAL-001", test_scenario(), &long_output, 100);
    let md = generate_evidence_detail(&evidence);

    // Should truncate to 200 chars
    assert!(md.contains("Output Preview"));
    assert!(!md.contains(&long_output)); // Full output should not be present
}

/// Verify markdown uses RAG-friendly header hierarchy
#[test]
fn test_markdown_uses_correct_headers() {
    let mqs = test_mqs_score();
    let popperian = test_popperian_score();
    let collector = EvidenceCollector::new();

    let md = generate_rag_markdown(&mqs, &popperian, &collector);

    // Check for RAG-friendly headers
    assert!(md.contains("# Model Qualification:"));
    assert!(md.contains("## Summary"));
    assert!(md.contains("## Gateway Checks"));
    assert!(md.contains("## Category Scores"));
    assert!(md.contains("## Popperian Analysis"));
    assert!(md.contains("## Metadata"));
}
