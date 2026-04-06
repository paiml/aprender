//! Example: Compiler-in-the-Loop (CITL) Training
//!
//! This example demonstrates the CITL module for:
//! 1. Storing and retrieving fix patterns with hybrid RAG retrieval
//! 2. Correlating compiler decisions with compilation errors
//! 3. Using Tarantula fault localization to identify suspicious decisions
//!
//! Run with: cargo run --example citl --features citl

#[cfg(feature = "citl")]
use entrenar::citl::{
    CompilationOutcome, DecisionCITL, DecisionPatternStore, DecisionTrace, FixPattern,
    PatternStoreConfig, SourceSpan,
};

#[cfg(feature = "citl")]
fn main() {
    println!("=== Compiler-in-the-Loop (CITL) Training Examples ===\n");

    // Example 1: Pattern Store
    println!("1. FIX PATTERN STORE\n");
    pattern_store_example();

    // Example 2: Decision Tracing
    println!("\n2. DECISION TRACING & CORRELATION\n");
    decision_tracing_example();

    // Example 3: Tarantula Fault Localization
    println!("\n3. TARANTULA FAULT LOCALIZATION\n");
    tarantula_example();

    // Example 4: Complete Workflow
    println!("\n4. COMPLETE CITL WORKFLOW\n");
    complete_workflow_example();

    // Example 5: APR Persistence
    println!("\n5. APR PERSISTENCE\n");
    apr_persistence_example();

    println!("\n=== Examples Complete ===\n");
}

#[cfg(feature = "citl")]
fn pattern_store_example() {
    println!("DecisionPatternStore with hybrid BM25 + dense retrieval\n");

    // Create store with custom config
    let config = PatternStoreConfig { chunk_size: 256, embedding_dim: 384, rrf_k: 60.0 };
    let mut store = DecisionPatternStore::with_config(config).expect("operation should succeed");

    println!("Configuration:");
    println!("  Chunk size: {}", store.config().chunk_size);
    println!("  Embedding dim: {}", store.config().embedding_dim);
    println!("  RRF k: {}", store.config().rrf_k);

    // Index some fix patterns
    let patterns = vec![
        FixPattern::new("E0308", "- let x: i32 = \"hello\";\n+ let x: &str = \"hello\";")
            .with_decision("type_inference")
            .with_decision("type_mismatch_detected"),
        FixPattern::new("E0308", "- let x: String = 42;\n+ let x: i32 = 42;")
            .with_decision("type_inference")
            .with_decision("integer_literal"),
        FixPattern::new("E0382", "- let y = x;\n- let z = x;\n+ let y = x.clone();\n+ let z = x;")
            .with_decision("borrow_check")
            .with_decision("move_tracking"),
    ];

    for mut pattern in patterns {
        // Simulate some success history
        pattern.record_success();
        pattern.record_success();
        pattern.record_failure();

        println!(
            "\nIndexing pattern for {}: {} decisions, success rate {:.0}%",
            pattern.error_code,
            pattern.decision_sequence.len(),
            pattern.success_rate() * 100.0
        );
        store.index_fix(pattern).expect("operation should succeed");
    }

    println!("\nTotal patterns indexed: {}", store.len());

    // Query for suggestions
    println!("\n--- Querying for E0308 fixes ---");
    let context = vec!["type_inference".to_string()];
    let suggestions = store.suggest_fix("E0308", &context, 5).expect("operation should succeed");

    println!("Found {} suggestions:", suggestions.len());
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. Score: {:.3}, Weighted: {:.3}",
            i + 1,
            suggestion.score,
            suggestion.weighted_score()
        );
        println!(
            "     Error: {}, Success rate: {:.0}%",
            suggestion.pattern.error_code,
            suggestion.pattern.success_rate() * 100.0
        );
        println!("     Decisions: {:?}", suggestion.pattern.decision_sequence);
    }

    // Export to JSON
    println!("\n--- JSON Export ---");
    let json = store.export_json().expect("operation should succeed");
    println!("Exported {} bytes of JSON", json.len());

    // Create new store and import
    let mut new_store = DecisionPatternStore::new().expect("operation should succeed");
    let count = new_store.import_json(&json).expect("operation should succeed");
    println!("Imported {count} patterns into new store");
}

#[cfg(feature = "citl")]
fn decision_tracing_example() {
    println!("Tracking compiler decisions through compilation\n");

    // Create decision traces
    let traces = vec![
        DecisionTrace::new("parse_001", "parsing", "Parsed function signature")
            .with_span(SourceSpan::line("main.rs", 5))
            .with_timestamp(1_000_000),
        DecisionTrace::new("type_001", "type_inference", "Inferred return type i32")
            .with_span(SourceSpan::line("main.rs", 5))
            .with_timestamp(2_000_000)
            .with_dependency("parse_001"),
        DecisionTrace::new("type_002", "type_inference", "Checked argument types")
            .with_span(SourceSpan::new("main.rs", 6, 1, 8, 20))
            .with_timestamp(3_000_000)
            .with_dependency("type_001"),
        DecisionTrace::new("borrow_001", "borrow_check", "Verified borrow rules")
            .with_span(SourceSpan::line("main.rs", 7))
            .with_timestamp(4_000_000)
            .with_dependency("type_002"),
    ];

    println!("Decision chain:");
    for trace in &traces {
        println!("  {} [{}]: {}", trace.id, trace.decision_type, trace.description);
        if let Some(ref span) = trace.span {
            println!("    at {span}");
        }
        if !trace.depends_on.is_empty() {
            println!("    depends on: {:?}", trace.depends_on);
        }
    }

    // Check span overlap
    println!("\n--- Span Analysis ---");
    let error_span = SourceSpan::line("main.rs", 7);
    println!("Error at: {error_span}");

    for trace in &traces {
        if let Some(ref span) = trace.span {
            if span.overlaps(&error_span) {
                println!("  Overlapping decision: {} - {}", trace.id, trace.decision_type);
            }
        }
    }
}

#[cfg(feature = "citl")]
fn tarantula_example() {
    println!("Statistical fault localization using Tarantula scoring\n");

    let mut trainer = DecisionCITL::new().expect("operation should succeed");

    // Ingest failed sessions with type_inference
    println!("Ingesting 8 failed sessions with type_inference...");
    for i in 0..8 {
        trainer
            .ingest_session(
                vec![DecisionTrace::new(format!("fail_{i}"), "type_inference", "Bad inference")],
                CompilationOutcome::failure(vec!["E0308".to_string()], vec![], vec![]),
                None,
            )
            .expect("operation should succeed");
    }

    // Ingest successful sessions with type_inference
    println!("Ingesting 2 successful sessions with type_inference...");
    for i in 0..2 {
        trainer
            .ingest_session(
                vec![DecisionTrace::new(
                    format!("success_ti_{i}"),
                    "type_inference",
                    "Good inference",
                )],
                CompilationOutcome::success(),
                None,
            )
            .expect("operation should succeed");
    }

    // Ingest successful sessions with borrow_check (control)
    println!("Ingesting 10 successful sessions with borrow_check...");
    for i in 0..10 {
        trainer
            .ingest_session(
                vec![DecisionTrace::new(
                    format!("success_bc_{i}"),
                    "borrow_check",
                    "Successful check",
                )],
                CompilationOutcome::success(),
                None,
            )
            .expect("operation should succeed");
    }

    println!(
        "\nTotal sessions: {} ({} success, {} failure)",
        trainer.session_count(),
        trainer.success_count(),
        trainer.failure_count()
    );

    // Calculate suspiciousness
    println!("\n--- Tarantula Suspiciousness Scores ---");
    println!("Formula: suspiciousness = fail_freq / (fail_freq + success_freq)");
    println!();

    let top = trainer.top_suspicious_types(5);
    for (decision_type, score) in top {
        let interpretation = if score > 0.7 {
            "HIGHLY SUSPICIOUS"
        } else if score > 0.4 {
            "Moderately suspicious"
        } else {
            "Low suspicion"
        };
        println!("  {decision_type}: {score:.3} - {interpretation}");
    }

    println!("\nExpected: type_inference should have high suspiciousness");
    println!("         (appears in 8/10 failures, only 2/12 successes)");
    println!("         borrow_check should have low suspiciousness");
    println!("         (appears in 0 failures, 10/12 successes)");
}

#[cfg(feature = "citl")]
fn complete_workflow_example() {
    println!("End-to-end CITL workflow for CI/CD integration\n");

    let mut trainer = DecisionCITL::new().expect("operation should succeed");

    // Phase 1: Build history from past compilations
    println!("Phase 1: Building history from CI/CD logs...");

    // Simulated CI/CD history
    let ci_history = vec![
        // Build 1: Failed - type mismatch
        (
            vec![DecisionTrace::new("b1_t1", "type_inference", "Inferred i32")
                .with_span(SourceSpan::line("src/lib.rs", 42))],
            CompilationOutcome::failure(
                vec!["E0308".to_string()],
                vec![SourceSpan::line("src/lib.rs", 42)],
                vec!["type mismatch".to_string()],
            ),
            Some("- let x: i32 = s;\n+ let x: &str = s;".to_string()),
        ),
        // Build 2: Success
        (
            vec![DecisionTrace::new("b2_t1", "type_inference", "Correct type")
                .with_span(SourceSpan::line("src/lib.rs", 42))],
            CompilationOutcome::success(),
            None,
        ),
        // Build 3: Failed - similar issue
        (
            vec![DecisionTrace::new("b3_t1", "type_inference", "Inferred wrong")
                .with_span(SourceSpan::line("src/main.rs", 15))],
            CompilationOutcome::failure(
                vec!["E0308".to_string()],
                vec![SourceSpan::line("src/main.rs", 15)],
                vec![],
            ),
            None,
        ),
        // Build 4: Failed - borrow issue
        (
            vec![DecisionTrace::new("b4_t1", "borrow_check", "Move detected")
                .with_span(SourceSpan::line("src/utils.rs", 100))],
            CompilationOutcome::failure(
                vec!["E0382".to_string()],
                vec![SourceSpan::line("src/utils.rs", 100)],
                vec!["use after move".to_string()],
            ),
            Some("+ .clone()".to_string()),
        ),
        // Build 5: Success
        (
            vec![DecisionTrace::new("b5_t1", "borrow_check", "Valid borrows")
                .with_span(SourceSpan::line("src/utils.rs", 100))],
            CompilationOutcome::success(),
            None,
        ),
    ];

    for (traces, outcome, fix) in ci_history {
        trainer.ingest_session(traces, outcome, fix).expect("operation should succeed");
    }

    println!(
        "  Ingested {} builds ({} success, {} failure)",
        trainer.session_count(),
        trainer.success_count(),
        trainer.failure_count()
    );
    println!("  Patterns indexed: {}", trainer.pattern_store().len());

    // Phase 2: Analyze a new error
    println!("\nPhase 2: New compilation error detected...");

    let new_error_span = SourceSpan::line("src/new_module.rs", 25);
    let new_error_code = "E0308";

    println!("  Error: {new_error_code} at {new_error_span}");

    let correlation =
        trainer.correlate_error(new_error_code, &new_error_span).expect("operation should succeed");

    println!("\nPhase 3: Analysis results...");

    println!("\n  Suspicious decisions:");
    if correlation.suspicious_decisions.is_empty() {
        println!("    (none in overlapping span)");
    } else {
        for suspicious in &correlation.suspicious_decisions {
            println!(
                "    {} (score={:.2}): {}",
                suspicious.decision.decision_type, suspicious.suspiciousness, suspicious.reason
            );
        }
    }

    println!("\n  Fix suggestions:");
    if correlation.fix_suggestions.is_empty() {
        println!("    (no matching patterns found)");
    } else {
        for (i, suggestion) in correlation.fix_suggestions.iter().enumerate() {
            println!(
                "    {}. [score={:.2}] {}",
                i + 1,
                suggestion.weighted_score(),
                suggestion.pattern.fix_diff.lines().next().unwrap_or("")
            );
        }
    }

    // Phase 4: Decision statistics
    println!("\nPhase 4: Global statistics...");

    println!("\n  Top suspicious decision types:");
    for (dtype, score) in trainer.top_suspicious_types(3) {
        println!("    {dtype}: {score:.2}");
    }

    println!("\n  Decisions by file:");
    for (file, decisions) in trainer.decisions_by_file() {
        println!("    {}: {} decisions", file, decisions.len());
    }

    // Phase 5: Export for sharing
    println!("\nPhase 5: Exporting learned patterns...");
    let json = trainer.pattern_store().export_json().expect("operation should succeed");
    println!("  Exported {} patterns ({} bytes)", trainer.pattern_store().len(), json.len());
    println!("  Ready for: artifact storage, cross-team sharing, etc.");
}

#[cfg(feature = "citl")]
fn apr_persistence_example() {
    println!("Binary persistence with .apr format (zstd compressed)\n");

    // Create and populate a store
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    let patterns = vec![
        FixPattern::new("E0308", "- i32\n+ &str")
            .with_decision("type_inference")
            .with_decision("string_literal"),
        FixPattern::new("E0382", "- x\n+ x.clone()")
            .with_decision("borrow_check")
            .with_decision("move_tracking"),
        FixPattern::new("E0499", "- &mut x\n+ { let r = &mut x; r }")
            .with_decision("borrow_check")
            .with_decision("mutable_borrow"),
    ];

    for mut pattern in patterns {
        pattern.record_success();
        pattern.record_success();
        store.index_fix(pattern).expect("operation should succeed");
    }

    println!("Created store with {} patterns", store.len());

    // Save to APR format
    let temp_path = std::env::temp_dir().join("citl_example.apr");
    println!("\n--- Saving to APR ---");
    store.save_apr(&temp_path).expect("operation should succeed");

    let apr_size = std::fs::metadata(&temp_path).expect("operation should succeed").len();
    println!("  Path: {}", temp_path.display());
    println!("  Size: {apr_size} bytes");

    // Compare with JSON size
    let json = store.export_json().expect("operation should succeed");
    let json_size = json.len();
    let ratio = (apr_size as f64 / json_size as f64) * 100.0;
    println!("\n--- Size Comparison ---");
    println!("  JSON: {json_size} bytes");
    println!("  APR:  {apr_size} bytes ({ratio:.0}% of JSON)");

    // Load from APR
    println!("\n--- Loading from APR ---");
    let loaded = DecisionPatternStore::load_apr(&temp_path).expect("operation should succeed");
    println!("  Loaded {} patterns", loaded.len());

    // Verify RAG index is rebuilt
    let suggestions = loaded
        .suggest_fix("E0308", &["type_inference".into()], 5)
        .expect("operation should succeed");
    println!("  RAG index rebuilt: {} suggestions for E0308", suggestions.len());

    // Verify config preserved
    println!("\n--- Config Preserved ---");
    println!("  Chunk size: {}", loaded.config().chunk_size);
    println!("  Embedding dim: {}", loaded.config().embedding_dim);
    println!("  RRF k: {}", loaded.config().rrf_k);

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);
    println!("\n  Cleaned up temp file");
}

#[cfg(not(feature = "citl"))]
fn main() {
    eprintln!("This example requires the 'citl' feature.");
    eprintln!("Run with: cargo run --example citl --features citl");
}
