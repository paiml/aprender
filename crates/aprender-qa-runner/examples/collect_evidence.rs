//! Example: Collecting Test Evidence
//!
//! This example demonstrates how to collect evidence from test runs
//! using the `EvidenceCollector` from apr-qa-runner.
//!
//! Run with: `cargo run --example collect_evidence -p apr-qa-runner`

#![allow(clippy::missing_panics_doc)]

use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use apr_qa_runner::{Evidence, EvidenceCollector, Outcome};

fn main() {
    // Create test scenarios (org, name)
    let model = ModelId::new("microsoft", "phi-3-mini-4k-instruct");

    // Create an evidence collector
    let mut collector = EvidenceCollector::new();

    // Add corroborated evidence (test passed)
    collector.add(Evidence::corroborated(
        "G1-LOAD",
        QaScenario::new(
            model.clone(),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "What is 2+2?".to_string(),
            42,
        ),
        "Model loaded successfully in 1.2s",
        1200,
    ));

    // Add falsified evidence (test failed)
    collector.add(Evidence::falsified(
        "G4-VALID",
        QaScenario::new(
            model,
            Modality::Chat,
            Backend::Cpu,
            Format::Gguf,
            "Explain quantum computing simply.".to_string(),
            43,
        ),
        "Output contained repetitive garbage",
        "The answer is the the the the...",
        500,
    ));

    // Display collected evidence
    println!("Evidence Collector Summary");
    println!("==========================");
    println!();

    for evidence in collector.all() {
        let status = match evidence.outcome {
            Outcome::Corroborated => "PASS",
            Outcome::Falsified => "FAIL",
            Outcome::Skipped => "SKIP",
            Outcome::Timeout => "TIME",
            Outcome::Crashed => "CRASH",
        };

        println!("[{}] Gate: {}", status, evidence.gate_id);
        println!("  Scenario: {}", evidence.scenario.id);
        println!("  Reason:   {}", evidence.reason);
        println!("  Duration: {}ms", evidence.metrics.duration_ms);
        println!();
    }

    // Summary statistics
    let passed = collector
        .all()
        .iter()
        .filter(|e| matches!(e.outcome, Outcome::Corroborated))
        .count();
    let failed = collector
        .all()
        .iter()
        .filter(|e| matches!(e.outcome, Outcome::Falsified))
        .count();

    println!("Summary: {passed} passed, {failed} failed");
}
