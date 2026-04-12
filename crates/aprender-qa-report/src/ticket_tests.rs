use super::*;

use crate::popperian::FalsificationDetail;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

/// Build a test QaScenario with fixed model and arithmetic prompt
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

/// Verify TicketPriority Display outputs formatted priority label
#[test]
fn test_ticket_priority_display() {
    assert_eq!(TicketPriority::P0.to_string(), "P0-Critical");
    assert_eq!(TicketPriority::P1.to_string(), "P1-High");
    assert_eq!(TicketPriority::P2.to_string(), "P2-Medium");
    assert_eq!(TicketPriority::P3.to_string(), "P3-Low");
}

/// Verify TicketCategory Display outputs lowercase category name
#[test]
fn test_ticket_category_display() {
    assert_eq!(TicketCategory::Bug.to_string(), "bug");
    assert_eq!(TicketCategory::Crash.to_string(), "crash");
    assert_eq!(TicketCategory::Performance.to_string(), "performance");
}

/// Verify generate_from_evidence creates P0 crash ticket with black swan flag
#[test]
fn test_generate_from_evidence() {
    let generator = TicketGenerator::new("paiml/aprender");
    let evidence = vec![Evidence::crashed(
        "F-QUAL-001",
        test_scenario(),
        "SIGSEGV",
        -11,
        0,
    )];

    let tickets = generator.generate_from_evidence(&evidence);

    assert_eq!(tickets.len(), 1);
    assert!(tickets[0].title.contains("Crash"));
    assert_eq!(tickets[0].priority, TicketPriority::P0);
    assert!(tickets[0].is_black_swan);
}

/// Verify generate_from_popperian creates tickets from falsification details
#[test]
fn test_generate_from_popperian() {
    let generator = TicketGenerator::new("paiml/aprender");
    let popperian = PopperianScore {
        model_id: "test/model".to_string(),
        hypotheses_tested: 100,
        corroborated: 95,
        falsified: 5,
        inconclusive: 0,
        corroboration_ratio: 0.95,
        severity_weighted_score: 0.93,
        confidence_level: 0.92,
        reproducibility_index: 0.85,
        black_swan_count: 1,
        falsifications: vec![FalsificationDetail {
            gate_id: "F-STAB-001".to_string(),
            hypothesis: "Model is stable".to_string(),
            evidence: "Crash detected".to_string(),
            severity: 5,
            is_black_swan: true,
            occurrence_count: 1,
        }],
    };

    let tickets = generator.generate_from_popperian(&popperian);

    assert_eq!(tickets.len(), 1);
    assert!(tickets[0].title.contains("F-STAB-001"));
    assert_eq!(tickets[0].priority, TicketPriority::P0);
    assert!(tickets[0].is_black_swan);
}

/// Verify min_occurrences filter suppresses tickets below threshold
#[test]
fn test_min_occurrences_filter() {
    let generator = TicketGenerator::new("paiml/aprender").with_min_occurrences(3);
    let evidence = vec![
        Evidence::falsified("F-QUAL-001", test_scenario(), "wrong", "5", 100),
        Evidence::falsified("F-QUAL-001", test_scenario(), "wrong", "5", 100),
    ];

    let tickets = generator.generate_from_evidence(&evidence);

    // Should be empty because we need 3 occurrences
    assert!(tickets.is_empty());
}

/// Verify black_swans_only filter excludes non-black-swan failures
#[test]
fn test_black_swans_only_filter() {
    let generator = TicketGenerator::new("paiml/aprender").black_swans_only();
    let evidence = vec![Evidence::falsified(
        "F-QUAL-001",
        test_scenario(),
        "wrong",
        "5",
        100,
    )];

    let tickets = generator.generate_from_evidence(&evidence);

    // Should be empty because it's not a black swan
    assert!(tickets.is_empty());
}

/// Verify to_gh_command generates valid gh issue create command string
#[test]
fn test_gh_command_generation() {
    let ticket = UpstreamTicket {
        title: "Test ticket".to_string(),
        body: "Test body\nLine 2".to_string(),
        priority: TicketPriority::P1,
        category: TicketCategory::Bug,
        labels: vec!["bug".to_string(), "qa-automated".to_string()],
        gate_id: "F-TEST-001".to_string(),
        model_id: "test/model".to_string(),
        is_black_swan: false,
        upstream_fixture: None,
        pygmy_builder: None,
    };

    let cmd = ticket.to_gh_command("paiml/aprender");

    assert!(cmd.contains("gh issue create"));
    assert!(cmd.contains("paiml/aprender"));
    assert!(cmd.contains("Test ticket"));
}

/// Verify determine_category maps gate_id prefixes to correct categories
#[test]
fn test_category_determination() {
    let generator = TicketGenerator::new("test");

    assert_eq!(
        generator.determine_category("F-PERF-001"),
        TicketCategory::Performance
    );
    assert_eq!(
        generator.determine_category("F-STAB-001"),
        TicketCategory::Crash
    );
    assert_eq!(
        generator.determine_category("F-COMP-001"),
        TicketCategory::Compatibility
    );
    assert_eq!(
        generator.determine_category("F-EDGE-001"),
        TicketCategory::EdgeCase
    );
    assert_eq!(
        generator.determine_category("F-REGR-001"),
        TicketCategory::Regression
    );
    assert_eq!(
        generator.determine_category("F-QUAL-001"),
        TicketCategory::Bug
    );
}

/// Verify determine_category maps CRASH gate prefix to Crash category
#[test]
fn test_category_crash_detection() {
    let generator = TicketGenerator::new("test");
    assert_eq!(
        generator.determine_category("F-CRASH-001"),
        TicketCategory::Crash
    );
}

/// Verify TicketGenerator stores and returns the repository name
#[test]
fn test_generator_repo() {
    let generator = TicketGenerator::new("owner/repo");
    assert_eq!(generator.repo(), "owner/repo");
}

/// Verify determine_priority assigns correct priorities for crash, black swan, and gate patterns
#[test]
fn test_priority_determination() {
    let generator = TicketGenerator::new("test");

    // Crash is always P0
    let crash_evidence = Evidence::crashed("F-QUAL-001", test_scenario(), "err", -1, 0);
    assert_eq!(
        generator.determine_priority(&crash_evidence, false),
        TicketPriority::P0
    );

    // Black swan is P0
    let regular_evidence = Evidence::falsified("F-QUAL-001", test_scenario(), "bad", "5", 100);
    assert_eq!(
        generator.determine_priority(&regular_evidence, true),
        TicketPriority::P0
    );

    // P0 gate ID
    let p0_evidence = Evidence::falsified("F-QUAL-P0-001", test_scenario(), "bad", "5", 100);
    assert_eq!(
        generator.determine_priority(&p0_evidence, false),
        TicketPriority::P0
    );

    // P1 gate ID
    let p1_evidence = Evidence::falsified("F-QUAL-P1-001", test_scenario(), "bad", "5", 100);
    assert_eq!(
        generator.determine_priority(&p1_evidence, false),
        TicketPriority::P1
    );

    // P2 gate ID
    let p2_evidence = Evidence::falsified("F-QUAL-P2-001", test_scenario(), "bad", "5", 100);
    assert_eq!(
        generator.determine_priority(&p2_evidence, false),
        TicketPriority::P2
    );

    // Default is P3
    let default_evidence = Evidence::falsified("F-QUAL-001", test_scenario(), "bad", "5", 100);
    assert_eq!(
        generator.determine_priority(&default_evidence, false),
        TicketPriority::P3
    );
}

/// Verify gateway gates (G1-LOAD) are classified as P0
#[test]
fn test_gateway_gate_is_p0() {
    let generator = TicketGenerator::new("test");
    let evidence = Evidence::falsified("G1-LOAD", test_scenario(), "failed", "", 100);
    assert_eq!(
        generator.determine_priority(&evidence, false),
        TicketPriority::P0
    );
}

/// Verify TicketCategory equality comparisons
#[test]
fn test_ticket_category_eq() {
    assert_eq!(TicketCategory::Bug, TicketCategory::Bug);
    assert_ne!(TicketCategory::Bug, TicketCategory::Crash);
}

/// Verify TicketPriority equality comparisons
#[test]
fn test_ticket_priority_eq() {
    assert_eq!(TicketPriority::P0, TicketPriority::P0);
    assert_ne!(TicketPriority::P0, TicketPriority::P1);
}

/// Verify UpstreamTicket clone preserves all fields
#[test]
fn test_upstream_ticket_clone() {
    let ticket = UpstreamTicket {
        title: "Test".to_string(),
        body: "Body".to_string(),
        priority: TicketPriority::P1,
        category: TicketCategory::Bug,
        labels: vec!["label".to_string()],
        gate_id: "F-001".to_string(),
        model_id: "test/model".to_string(),
        is_black_swan: false,
        upstream_fixture: None,
        pygmy_builder: None,
    };
    let cloned = ticket.clone();
    assert_eq!(cloned.title, ticket.title);
}

/// Verify generate_from_evidence creates timeout ticket from timeout evidence
#[test]
fn test_generate_from_evidence_with_timeout() {
    let generator = TicketGenerator::new("paiml/aprender");
    let evidence = vec![Evidence::timeout("F-PERF-001", test_scenario(), 30000)];

    let tickets = generator.generate_from_evidence(&evidence);

    assert_eq!(tickets.len(), 1);
    assert!(tickets[0].title.contains("Timeout"));
}

/// Verify generate_from_evidence creates assertion ticket from falsified evidence
#[test]
fn test_generate_from_evidence_falsified() {
    let generator = TicketGenerator::new("paiml/aprender");
    let evidence = vec![Evidence::falsified(
        "F-QUAL-001",
        test_scenario(),
        "Wrong answer",
        "5",
        100,
    )];

    let tickets = generator.generate_from_evidence(&evidence);

    assert_eq!(tickets.len(), 1);
    assert!(tickets[0].title.contains("Assertion"));
}

/// Verify generate_from_evidence deduplicates multiple failures with same gate_id
#[test]
fn test_generate_deduplication() {
    let generator = TicketGenerator::new("test");
    let evidence = vec![
        Evidence::falsified("F-QUAL-001", test_scenario(), "err1", "out", 100),
        Evidence::falsified("F-QUAL-001", test_scenario(), "err2", "out", 100),
        Evidence::falsified("F-QUAL-001", test_scenario(), "err3", "out", 100),
    ];

    let tickets = generator.generate_from_evidence(&evidence);

    // Should be deduplicated to 1 ticket
    assert_eq!(tickets.len(), 1);
}

/// Verify generated tickets include modality and backend labels
#[test]
fn test_ticket_labels_include_modality() {
    let generator = TicketGenerator::new("test");
    let evidence = vec![Evidence::crashed("F-001", test_scenario(), "err", -1, 0)];

    let tickets = generator.generate_from_evidence(&evidence);

    assert!(tickets[0].labels.iter().any(|l| l.contains("modality:")));
    assert!(tickets[0].labels.iter().any(|l| l.contains("backend:")));
}

/// Verify black swan crashes get the black-swan label
#[test]
fn test_black_swan_label_added() {
    let generator = TicketGenerator::new("test");
    let evidence = vec![Evidence::crashed(
        "F-001",
        test_scenario(),
        "SIGSEGV",
        -11,
        0,
    )];

    let tickets = generator.generate_from_evidence(&evidence);

    assert!(tickets[0].is_black_swan);
    assert!(tickets[0].labels.contains(&"black-swan".to_string()));
}

/// Create a falsified evidence with attached stderr for structured ticket tests
fn falsified_with_stderr(gate_id: &str, stderr: &str) -> Evidence {
    let mut ev = Evidence::falsified(gate_id, test_scenario(), "failure", "N/A", 100);
    ev.stderr = Some(stderr.to_string());
    ev
}

/// Verify structured tickets group same-cause failures into one ticket
#[test]
fn test_structured_tickets_same_cause_dedup() {
    let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

    // 12 failures with the same stderr pattern → all classify as same root cause
    let evidence: Vec<Evidence> = (0..12)
        .map(|i| {
            falsified_with_stderr(
                "F-CONV-001",
                &format!("tensor name mismatch: layer.{i}.weight"),
            )
        })
        .collect();

    let tickets = generate_structured_tickets(&evidence, &defect_map);

    // Should be 1 ticket (12 same-cause failures → 1 grouped ticket)
    assert_eq!(tickets.len(), 1);
    assert!(tickets[0].title.contains("12 occurrences"));
}

/// Verify structured tickets create separate tickets for distinct root causes
#[test]
fn test_structured_tickets_two_causes() {
    let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

    let mut evidence = Vec::new();
    // 3 tensor name mismatches
    for _ in 0..3 {
        evidence.push(falsified_with_stderr(
            "F-CONV-001",
            "tensor name mismatch: layer.0.weight",
        ));
    }
    // 2 missing artifact failures
    for _ in 0..2 {
        evidence.push(falsified_with_stderr(
            "F-CONV-002",
            "file not found: model.safetensors",
        ));
    }

    let tickets = generate_structured_tickets(&evidence, &defect_map);

    // Should be 2 tickets (2 different root causes)
    assert_eq!(tickets.len(), 2);
}

/// Verify structured tickets include upstream fixture and pygmy builder references
#[test]
fn test_structured_tickets_fixture_in_body() {
    let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

    let evidence = vec![falsified_with_stderr(
        "F-CONV-001",
        "tensor name mismatch: layer.0.weight",
    )];

    let tickets = generate_structured_tickets(&evidence, &defect_map);

    assert_eq!(tickets.len(), 1);
    assert!(tickets[0].upstream_fixture.is_some());
    assert!(tickets[0].pygmy_builder.is_some());
    assert_eq!(
        tickets[0].upstream_fixture.as_deref(),
        Some("fixtures/tensor_name_mismatch.py")
    );
}

/// Verify structured tickets returns empty vec when all evidence is corroborated
#[test]
fn test_structured_tickets_no_failures() {
    let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

    let evidence = vec![Evidence::corroborated(
        "F-CONV-001",
        test_scenario(),
        "4",
        100,
    )];

    let tickets = generate_structured_tickets(&evidence, &defect_map);
    assert!(tickets.is_empty());
}

/// Verify structured tickets omit fixture for unknown error patterns
#[test]
fn test_structured_tickets_unknown_cause_no_fixture() {
    let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

    // Stderr that doesn't match any known pattern
    let evidence = vec![Evidence::falsified(
        "F-CONV-001",
        test_scenario(),
        "some unknown error xyz",
        "N/A",
        100,
    )];

    let tickets = generate_structured_tickets(&evidence, &defect_map);

    assert_eq!(tickets.len(), 1);
    // Unknown cause → no fixture mapping
    assert!(tickets[0].upstream_fixture.is_none());
    assert!(tickets[0].pygmy_builder.is_none());
}

/// Verify structured tickets include failure-type and qa-automated labels
#[test]
fn test_structured_tickets_labels() {
    let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

    let evidence = vec![Evidence::falsified(
        "F-CONV-001",
        test_scenario(),
        "tensor name mismatch: layer.0.weight",
        "N/A",
        100,
    )];

    let tickets = generate_structured_tickets(&evidence, &defect_map);

    assert!(!tickets.is_empty());
    assert!(
        tickets[0]
            .labels
            .iter()
            .any(|l| l.starts_with("failure-type:"))
    );
    assert!(tickets[0].labels.contains(&"qa-automated".to_string()));
}

/// Verify generate_from_popperian assigns P1 for severity 4
#[test]
fn test_popperian_severity_4_p1() {
    let generator = TicketGenerator::new("test");
    let popperian = PopperianScore {
        model_id: "test/model".to_string(),
        hypotheses_tested: 10,
        corroborated: 9,
        falsified: 1,
        inconclusive: 0,
        corroboration_ratio: 0.9,
        severity_weighted_score: 0.9,
        confidence_level: 0.9,
        reproducibility_index: 0.8,
        black_swan_count: 0,
        falsifications: vec![FalsificationDetail {
            gate_id: "F-QUAL-001".to_string(),
            hypothesis: "Quality holds".to_string(),
            evidence: "Failure".to_string(),
            severity: 4,
            is_black_swan: false,
            occurrence_count: 1,
        }],
    };

    let tickets = generator.generate_from_popperian(&popperian);
    assert_eq!(tickets.len(), 1);
    assert_eq!(tickets[0].priority, TicketPriority::P1);
    assert!(!tickets[0].is_black_swan);
}

/// Verify generate_from_popperian assigns P2 for severity 3
#[test]
fn test_popperian_severity_3_p2() {
    let generator = TicketGenerator::new("test");
    let popperian = PopperianScore {
        model_id: "test/model".to_string(),
        hypotheses_tested: 10,
        corroborated: 9,
        falsified: 1,
        inconclusive: 0,
        corroboration_ratio: 0.9,
        severity_weighted_score: 0.9,
        confidence_level: 0.9,
        reproducibility_index: 0.8,
        black_swan_count: 0,
        falsifications: vec![FalsificationDetail {
            gate_id: "F-EDGE-001".to_string(),
            hypothesis: "Edge case holds".to_string(),
            evidence: "Minor failure".to_string(),
            severity: 3,
            is_black_swan: false,
            occurrence_count: 1,
        }],
    };

    let tickets = generator.generate_from_popperian(&popperian);
    assert_eq!(tickets.len(), 1);
    assert_eq!(tickets[0].priority, TicketPriority::P2);
}

/// Verify generate_from_popperian assigns P3 for severity < 3
#[test]
fn test_popperian_severity_low_p3() {
    let generator = TicketGenerator::new("test");
    let popperian = PopperianScore {
        model_id: "test/model".to_string(),
        hypotheses_tested: 10,
        corroborated: 9,
        falsified: 1,
        inconclusive: 0,
        corroboration_ratio: 0.9,
        severity_weighted_score: 0.9,
        confidence_level: 0.9,
        reproducibility_index: 0.8,
        black_swan_count: 0,
        falsifications: vec![FalsificationDetail {
            gate_id: "F-QUAL-001".to_string(),
            hypothesis: "Quality holds".to_string(),
            evidence: "Minor issue".to_string(),
            severity: 2,
            is_black_swan: false,
            occurrence_count: 1,
        }],
    };

    let tickets = generator.generate_from_popperian(&popperian);
    assert_eq!(tickets.len(), 1);
    assert_eq!(tickets[0].priority, TicketPriority::P3);
}

/// Verify generate_from_popperian respects black_swans_only filter
#[test]
fn test_popperian_black_swans_only_filter() {
    let generator = TicketGenerator::new("test").black_swans_only();
    let popperian = PopperianScore {
        model_id: "test/model".to_string(),
        hypotheses_tested: 10,
        corroborated: 8,
        falsified: 2,
        inconclusive: 0,
        corroboration_ratio: 0.8,
        severity_weighted_score: 0.8,
        confidence_level: 0.8,
        reproducibility_index: 0.8,
        black_swan_count: 0,
        falsifications: vec![FalsificationDetail {
            gate_id: "F-QUAL-001".to_string(),
            hypothesis: "Quality holds".to_string(),
            evidence: "Failure".to_string(),
            severity: 4,
            is_black_swan: false,
            occurrence_count: 1,
        }],
    };

    let tickets = generator.generate_from_popperian(&popperian);
    assert!(tickets.is_empty(), "Non-black-swan should be filtered");
}

/// Verify generate_from_popperian respects min_occurrences filter
#[test]
fn test_popperian_min_occurrences_filter() {
    let generator = TicketGenerator::new("test").with_min_occurrences(5);
    let popperian = PopperianScore {
        model_id: "test/model".to_string(),
        hypotheses_tested: 10,
        corroborated: 8,
        falsified: 2,
        inconclusive: 0,
        corroboration_ratio: 0.8,
        severity_weighted_score: 0.8,
        confidence_level: 0.8,
        reproducibility_index: 0.8,
        black_swan_count: 0,
        falsifications: vec![FalsificationDetail {
            gate_id: "F-QUAL-001".to_string(),
            hypothesis: "Quality holds".to_string(),
            evidence: "Failure".to_string(),
            severity: 4,
            is_black_swan: false,
            occurrence_count: 2, // Below min_occurrences of 5
        }],
    };

    let tickets = generator.generate_from_popperian(&popperian);
    assert!(tickets.is_empty(), "Below min_occurrences should be filtered");
}

/// Verify generate_from_evidence produces empty result for all-pass evidence
#[test]
fn test_generate_from_evidence_all_pass() {
    let generator = TicketGenerator::new("test");
    let evidence = vec![
        Evidence::corroborated("F-001", test_scenario(), "ok", 100),
        Evidence::corroborated("F-002", test_scenario(), "ok", 100),
    ];

    let tickets = generator.generate_from_evidence(&evidence);
    assert!(tickets.is_empty(), "All-pass evidence should produce no tickets");
}
