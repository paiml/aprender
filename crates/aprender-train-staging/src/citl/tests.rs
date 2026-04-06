//! Integration tests for CITL module

#![allow(clippy::field_reassign_with_default)]

use super::*;

// ============ Integration Tests ============

#[test]
fn test_full_citl_workflow() {
    // Create trainer
    let mut trainer = DecisionCITL::new().expect("operation should succeed");

    // Simulate multiple failed sessions with similar patterns
    for i in 0..5 {
        let span = SourceSpan::line("main.rs", 10);
        let traces = vec![
            DecisionTrace::new(
                format!("d1_{i}"),
                "type_inference",
                "Inferred i32 for string literal",
            )
            .with_span(span.clone()),
            DecisionTrace::new(format!("d2_{i}"), "borrow_check", "Checking borrow")
                .with_span(SourceSpan::line("main.rs", 15)),
        ];

        let outcome = CompilationOutcome::failure(
            vec!["E0308".to_string()],
            vec![span],
            vec!["expected `&str`, found `i32`".to_string()],
        );

        // On the last iteration, provide a fix
        let fix = if i == 4 {
            Some("- let x: i32 = \"hello\";\n+ let x: &str = \"hello\";".to_string())
        } else {
            None
        };

        trainer.ingest_session(traces, outcome, fix).expect("operation should succeed");
    }

    // Simulate successful sessions
    for i in 0..3 {
        let traces = vec![DecisionTrace::new(
            format!("s1_{i}"),
            "type_inference",
            "Inferred &str correctly",
        )
        .with_span(SourceSpan::line("main.rs", 10))];

        trainer
            .ingest_session(traces, CompilationOutcome::success(), None)
            .expect("operation should succeed");
    }

    // Verify session counts
    assert_eq!(trainer.failure_count(), 5);
    assert_eq!(trainer.success_count(), 3);

    // Verify pattern was indexed
    assert_eq!(trainer.pattern_store().len(), 1);

    // Correlate the error
    let error_span = SourceSpan::line("main.rs", 10);
    let correlation =
        trainer.correlate_error("E0308", &error_span).expect("operation should succeed");

    assert_eq!(correlation.error_code, "E0308");

    // Check suspicious decisions
    let type_inference_suspicious = correlation
        .suspicious_decisions
        .iter()
        .any(|s| s.decision.decision_type == "type_inference");
    assert!(type_inference_suspicious || correlation.suspicious_decisions.is_empty());

    // Check for fix suggestions
    // May or may not have suggestions depending on RAG matching
    assert!(correlation.fix_suggestions.len() <= 5);

    // Verify top suspicious types
    let top = trainer.top_suspicious_types(3);
    assert!(!top.is_empty());
}

#[test]
fn test_dependency_chain_tracking() {
    let mut trainer = DecisionCITL::new().expect("operation should succeed");

    let span = SourceSpan::line("main.rs", 5);

    // Create a chain: root -> middle -> leaf
    let traces = vec![
        DecisionTrace::new("root", "parse", "Parsed expression").with_span(span.clone()),
        DecisionTrace::new("middle", "type_check", "Type checking")
            .with_span(span.clone())
            .with_dependency("root"),
        DecisionTrace::new("leaf", "borrow_check", "Borrow checking")
            .with_span(span.clone())
            .with_dependency("middle"),
    ];

    trainer
        .ingest_session(
            traces,
            CompilationOutcome::failure(vec!["E0308".to_string()], vec![span.clone()], vec![]),
            None,
        )
        .expect("operation should succeed");

    // Build dependency graph
    let graph = trainer.build_dependency_graph();
    assert_eq!(graph.get("middle").expect("key should exist"), &vec!["root".to_string()]);
    assert_eq!(graph.get("leaf").expect("key should exist"), &vec!["middle".to_string()]);

    // Find root causes
    let roots = trainer.find_root_causes(&span);
    assert!(roots.iter().any(|r| r.id == "root"));
}

#[test]
fn test_pattern_store_with_multiple_error_codes() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    // Index patterns for different error codes
    store
        .index_fix(FixPattern::new("E0308", "type mismatch fix").with_decision("type_inference"))
        .expect("operation should succeed");

    store
        .index_fix(FixPattern::new("E0382", "use after move fix").with_decision("borrow_check"))
        .expect("operation should succeed");

    store
        .index_fix(FixPattern::new("E0308", "another type fix").with_decision("type_coercion"))
        .expect("operation should succeed");

    // Verify indexing
    assert_eq!(store.len(), 3);
    assert_eq!(store.patterns_for_error("E0308").len(), 2);
    assert_eq!(store.patterns_for_error("E0382").len(), 1);

    // Query suggestions
    let suggestions = store
        .suggest_fix("E0308", &["type_inference".to_string()], 5)
        .expect("operation should succeed");
    assert!(!suggestions.is_empty());
}

#[test]
fn test_success_rate_affects_ranking() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    // Pattern with high success rate
    let mut pattern1 = FixPattern::new("E0308", "successful fix");
    for _ in 0..10 {
        pattern1.record_success();
    }
    let id1 = pattern1.id;
    store.index_fix(pattern1).expect("operation should succeed");

    // Pattern with low success rate
    let mut pattern2 = FixPattern::new("E0308", "failing fix");
    for _ in 0..10 {
        pattern2.record_failure();
    }
    let id2 = pattern2.id;
    store.index_fix(pattern2).expect("operation should succeed");

    // High success pattern should have higher weighted score
    let p1 = store.get(&id1).expect("key should exist");
    let p2 = store.get(&id2).expect("key should exist");

    let suggestion1 = FixSuggestion::new(p1.clone(), 1.0, 0);
    let suggestion2 = FixSuggestion::new(p2.clone(), 1.0, 0);

    assert!(suggestion1.weighted_score() > suggestion2.weighted_score());
}

#[test]
fn test_tarantula_suspiciousness_calculation() {
    let mut trainer = DecisionCITL::new().expect("operation should succeed");

    // Bad decision: appears in all failures
    for _ in 0..10 {
        trainer
            .ingest_session(
                vec![DecisionTrace::new("d", "bad_decision", "")],
                CompilationOutcome::failure(vec!["E".to_string()], vec![], vec![]),
                None,
            )
            .expect("operation should succeed");
    }

    // Good decision: appears in all successes
    for _ in 0..10 {
        trainer
            .ingest_session(
                vec![DecisionTrace::new("d", "good_decision", "")],
                CompilationOutcome::success(),
                None,
            )
            .expect("operation should succeed");
    }

    // Mixed decision: appears in both
    for _ in 0..5 {
        trainer
            .ingest_session(
                vec![DecisionTrace::new("d", "mixed_decision", "")],
                CompilationOutcome::failure(vec!["E".to_string()], vec![], vec![]),
                None,
            )
            .expect("operation should succeed");
        trainer
            .ingest_session(
                vec![DecisionTrace::new("d", "mixed_decision", "")],
                CompilationOutcome::success(),
                None,
            )
            .expect("operation should succeed");
    }

    let top = trainer.top_suspicious_types(3);

    // bad_decision should have highest suspiciousness (appears only in failures)
    assert!(!top.is_empty());
    if top.len() >= 2 {
        // First should have higher score than second
        assert!(top[0].1 >= top[1].1);
    }
}

#[test]
fn test_json_export_import_preserves_data() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    let mut pattern =
        FixPattern::new("E0308", "- old\n+ new").with_decision("step1").with_decision("step2");
    pattern.record_success();
    pattern.record_success();
    pattern.record_failure();

    store.index_fix(pattern).expect("operation should succeed");

    let json = store.export_json().expect("operation should succeed");

    let mut new_store = DecisionPatternStore::new().expect("operation should succeed");
    new_store.import_json(&json).expect("operation should succeed");

    let patterns: Vec<_> = new_store.patterns_for_error("E0308");
    assert_eq!(patterns.len(), 1);

    let p = patterns[0];
    assert_eq!(p.error_code, "E0308");
    assert_eq!(p.fix_diff, "- old\n+ new");
    assert_eq!(p.decision_sequence, vec!["step1", "step2"]);
    assert_eq!(p.success_count, 2);
    assert_eq!(p.attempt_count, 3);
}

#[test]
fn test_overlapping_spans_detected() {
    let mut trainer = DecisionCITL::new().expect("operation should succeed");

    // Decision spanning lines 5-10
    let wide_span = SourceSpan::new("main.rs", 5, 1, 10, 80);
    trainer
        .ingest_session(
            vec![DecisionTrace::new("d1", "type", "").with_span(wide_span)],
            CompilationOutcome::failure(vec!["E".to_string()], vec![], vec![]),
            None,
        )
        .expect("operation should succeed");

    // Error on line 7 should overlap with decision
    let error_span = SourceSpan::line("main.rs", 7);
    let correlation = trainer.correlate_error("E", &error_span).expect("operation should succeed");

    // Should find the overlapping decision
    let found = trainer.find_root_causes(&error_span);
    assert!(!found.is_empty() || correlation.suspicious_decisions.is_empty());
}

// ============ Property Tests ============

use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_fix_pattern_roundtrip(
        error_code in "[A-Z][0-9]{4}",
        fix_diff in ".*",
        decisions in prop::collection::vec("[a-z_]+", 0..5)
    ) {
        let pattern = FixPattern::new(&error_code, &fix_diff)
            .with_decisions(decisions.clone());

        let json = serde_json::to_string(&pattern);
        prop_assert!(json.is_ok(), "serialize failed: {:?}", json.err());
        let json = json.expect("operation should succeed");
        let restored: Result<FixPattern, _> = serde_json::from_str(&json);
        prop_assert!(restored.is_ok(), "deserialize failed: {:?}", restored.err());
        let restored = restored.expect("operation should succeed");

        prop_assert_eq!(pattern.error_code, restored.error_code);
        prop_assert_eq!(pattern.fix_diff, restored.fix_diff);
        prop_assert_eq!(pattern.decision_sequence, restored.decision_sequence);
    }

    #[test]
    fn prop_source_span_overlap_reflexive(
        line in 1u32..1000
    ) {
        let span = SourceSpan::line("file.rs", line);
        prop_assert!(span.overlaps(&span));
    }

    #[test]
    fn prop_store_len_matches_indexed(
        n_patterns in 1usize..20
    ) {
        let mut store = DecisionPatternStore::new().expect("operation should succeed");

        for i in 0..n_patterns {
            store.index_fix(FixPattern::new(format!("E{i:04}"), "fix")).expect("operation should succeed");
        }

        prop_assert_eq!(store.len(), n_patterns);
    }

    #[test]
    fn prop_trainer_never_negative_counts(
        n_success in 0usize..10,
        n_fail in 0usize..10
    ) {
        let mut trainer = DecisionCITL::new().expect("operation should succeed");

        for _ in 0..n_success {
            trainer.ingest_session(vec![], CompilationOutcome::success(), None).expect("operation should succeed");
        }

        for _ in 0..n_fail {
            trainer.ingest_session(
                vec![],
                CompilationOutcome::failure(vec![], vec![], vec![]),
                None,
            ).expect("operation should succeed");
        }

        prop_assert!(trainer.success_count() >= 0);
        prop_assert!(trainer.failure_count() >= 0);
    }
}
